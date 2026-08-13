"""Romberg integration aka adaptive trapezoid with Richardson extrapolation."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .adjoint import (
    AbstractAdjoint,
    DirectAdjoint,
    QuadratureOps,
    _ConvertedFunction,
    build_integrand,
    closure_convert,
)
from .utils import (
    QuadratureInfo,
    _pnorm,
    _real_dtype,
    bounded_while_loop,
    errorif,
    map_interval,
    resolve_dtypes,
    tanhsinh_transform,
    wrap_func,
)


@eqx.filter_jit
def romberg(
    fun: Callable[..., jax.Array],
    interval: ArrayLike,
    args: tuple = (),
    full_output: bool = False,
    epsabs: ArrayLike | None = None,
    epsrel: ArrayLike | None = None,
    divmax: int = 20,
    norm: float | int | Callable[[jax.Array], jax.Array] = jnp.inf,
    adjoint: AbstractAdjoint = DirectAdjoint(),
):
    """Romberg integration of a callable function or method.

    Returns the integral of `fun` (a function of one variable) over `interval`.

    Good for non-smooth or piecewise smooth integrands.

    Not recommended for infinite intervals, or functions with singularities.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration. Use np.inf to denote infinite intervals.
        Its dtype sets the working precision: the integrand is called with an ``x`` of
        this dtype, and the result follows it unless the integrand upcasts. A integer
        types or python floats falls back to the JAX default. Must be real; complex
        integrands are supported, complex limits are not.
    args : tuple
        additional arguments passed to fun
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float
        Absolute and relative tolerances. If I1 and I2 are two
        successive approximations to the integral, algorithm terminates
        when abs(I1-I2) < max(epsabs, epsrel*|I2|). Default is the square root of the
        machine precision of the working dtype, ie of `interval`, or of the integrand's
        own dtype if that is the coarser of the two.
    divmax : int, optional
        Maximum order of extrapolation. Default is 20.
        Total number of function evaluations will be at
        most 2**divmax + 1
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.

    Returns
    -------
    y  : float, Array
        Approximation to the integral
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (int) Flag indicating reason for termination. status of 0 means
          normal termination, any other value indicates a possible error. A human
          readable message can be obtained by ``print(quadax.STATUS[status])``
        * info : (dict or None) Other information returned by the algorithm.
          Only present if ``full_output`` is True. Contains the following:

          * table : (ndarray, size(dixmax+1, divmax+1, ...)) Estimate of the integral
            from each level of discretization and each step of extrapolation.

    Notes
    -----
    Due to limitations on dynamically sized arrays in JAX, this algorithm is fully
    sequential and does not vectorize integrand evaluations, so may not be the most
    efficient on GPU/TPU.

    """
    interval = jnp.atleast_1d(jnp.asarray(interval))
    if not jnp.issubdtype(interval.dtype, jnp.inexact):
        # integration limits must be inexact: they are differentiated with respect to,
        # and integer leaves would otherwise be treated as static metadata
        interval = interval.astype(jnp.result_type(float))
    errorif(
        len(interval) != 2,
        NotImplementedError,
        "Romberg integration with breakpoints not supported",
    )
    dtypes = resolve_dtypes(interval, fun, args)
    if epsabs is None:
        epsabs = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    if epsrel is None:
        epsrel = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    epsabs = jnp.asarray(epsabs, dtypes.etype)
    epsrel = jnp.asarray(epsrel, dtypes.etype)
    if callable(norm):
        _norm: Callable[[jax.Array], jax.Array] = norm
    else:
        _norm: Callable[[jax.Array], jax.Array] = partial(_pnorm, p=norm)

    return _romberg(
        fun,
        interval,
        args,
        full_output,
        epsabs,
        epsrel,
        divmax,
        _norm,
        adjoint,
        build_integrand,
        dtypes.xtype,
    )


def _romberg(
    fun,
    interval,
    args,
    full_output,
    epsabs,
    epsrel,
    divmax,
    _norm,
    adjoint,
    build,
    xtype,
):
    """Shared driver for ``romberg`` and ``rombergts``, differing only in ``build``."""
    # Closure conversion has to happen on the user's function, before any wrapping:
    # once a transformed integrand crosses a filter_jit boundary its leaves become
    # tracers that closure_convert cannot hoist.
    f_conv, consts = closure_convert(fun, args, xtype)
    # Romberg has no subdivision to reuse, so `rebuild`/`on_mesh` are left unset and
    # DirectAdjoint falls back to differentiating through the loop.
    ops = QuadratureOps(
        build=partial(build, f_conv=f_conv),
        solve=partial(_romberg_solve, divmax=divmax, _norm=_norm),
        # Romberg has no subdivision to reuse, but it does settle on a number of
        # Richardson levels. Freezing that makes the result a fixed linear functional of
        # the integrand, which is what DirectAdjoint needs to differentiate in either
        # direction. It has to go through a custom primitive rather than being
        # differentiated directly, because evaluating it still involves a fori_loop with
        # dynamic bounds that JAX cannot reverse differentiate.
        frozen=lambda state: state["n"],
        frozen_solve=partial(_romberg_levels, divmax=divmax),
    )
    y, state = adjoint.quadrature(ops, None, interval, args, consts, epsabs, epsrel, {})
    info = state["table"] if full_output else None
    out = QuadratureInfo(state["err_sum"], state["neval"], state["status"], info)
    return y, out


def _build_tanhsinh(interval, args, consts, *, f_conv):
    """Build the integrand for ``rombergts``: tanh-sinh, then map to the reference."""
    fun = _ConvertedFunction(f_conv, args, consts)
    fun_t, interval_t = tanhsinh_transform(fun, interval)
    fun_m, interval_m = map_interval(fun_t, interval_t)
    return wrap_func(fun_m, (), interval_m.dtype), interval_m


def _romberg_solve(rule, vfunc, interval, epsabs, epsrel, kwargs, *, divmax, _norm):
    """Run the Romberg/Richardson extrapolation loop."""
    del rule, kwargs
    a, b = interval
    f = jax.eval_shape(vfunc, (a + b) / 2)

    result = jnp.zeros((divmax + 1, divmax + 1, *f.shape), f.dtype)
    result = result.at[0, 0].set(vfunc(a) + vfunc(b))
    neval = 2
    # Explicitly typed rather than left a weak python float: this is a loop carry, and
    # has to match what `_norm` writes back into it. Real, because the error in a
    # complex valued integral is still real.
    err = jnp.array(jnp.inf, _real_dtype(f.dtype))
    state = (result, 1, neval, err)

    def ncond(state):
        result, n, neval, err = state
        return (n < divmax + 1) & (
            err > jnp.maximum(epsabs, epsrel * _norm(result[n, n]))
        )

    def nloop(state):
        # loop over outer number of subdivisions
        result, n, neval, err = state
        h = (b - a) / 2**n
        s = jnp.zeros(f.shape, f.dtype)

        def sloop(i, s):
            # loop to evaluate fun. Can't be vectorized due to different number
            # of evals per nloop step
            s += vfunc(a + h * (2 * i - 1))
            return s

        result = result.at[n, 0].set(
            0.5 * result[n - 1, 0] + h * jax.lax.fori_loop(1, (2**n) // 2 + 1, sloop, s)
        )
        neval += (2**n) // 2

        def mloop(m, result):
            # richardson extrapolation
            temp = 1 / (4.0**m - 1.0) * (result[n, m - 1] - result[n - 1, m - 1])
            result = result.at[n, m].set(result[n, m - 1] + temp)
            return result

        result = jax.lax.fori_loop(1, n + 1, mloop, result)
        err = _norm(result[n, n] - result[n - 1, n - 1])
        return result, n + 1, neval, err

    result, n, neval, err = bounded_while_loop(ncond, nloop, state, divmax + 1)

    y = result[n - 1, n - 1]
    status = 2 * (err > jnp.maximum(epsabs, epsrel * _norm(y)))
    state = {
        "table": result,
        "err_sum": err,
        "neval": neval,
        "status": status,
        "n": n,  # Richardson levels used; frozen by DirectAdjoint
    }
    return y, state


def _romberg_levels(rule, vfunc, interval, n, kwargs, *, divmax):
    """Evaluate the Richardson table at a fixed number of levels.

    With ``n`` fixed this is a fixed linear combination of the integrand at fixed nodes,
    so its forward and reverse derivatives are exact transposes of one another. Mirrors
    the loop in ``_romberg_solve`` exactly so the two agree.
    """
    del rule, kwargs
    a, b = interval[0], interval[-1]
    f = jax.eval_shape(vfunc, (a + b) / 2)
    result = jnp.zeros((divmax + 1, divmax + 1, *f.shape), f.dtype)
    result = result.at[0, 0].set(vfunc(a) + vfunc(b))

    def nloop(k, result):
        h = (b - a) / 2**k

        def sloop(i, s):
            return s + vfunc(a + h * (2 * i - 1))

        s = jax.lax.fori_loop(1, (2**k) // 2 + 1, sloop, jnp.zeros(f.shape, f.dtype))
        result = result.at[k, 0].set(0.5 * result[k - 1, 0] + h * s)

        def mloop(m, result):
            temp = 1 / (4.0**m - 1.0) * (result[k, m - 1] - result[k - 1, m - 1])
            return result.at[k, m].set(result[k, m - 1] + temp)

        return jax.lax.fori_loop(1, k + 1, mloop, result)

    result = jax.lax.fori_loop(1, n, nloop, result)
    return result[n - 1, n - 1]


@eqx.filter_jit
def rombergts(
    fun: Callable[..., jax.Array],
    interval: ArrayLike,
    args: tuple = (),
    full_output: bool = False,
    epsabs: ArrayLike | None = None,
    epsrel: ArrayLike | None = None,
    divmax: int = 20,
    norm: float | int | Callable[[jax.Array], jax.Array] = jnp.inf,
    adjoint: AbstractAdjoint = DirectAdjoint(),
):
    """Romberg integration with tanh-sinh (aka double exponential) transformation.

    Returns the integral of `fun` (a function of one variable) over `interval`.

    Performs well for functions with singularities at the endpoints or integration
    over infinite intervals. May be slightly less efficient than ``quadgk`` or
    ``quadcc`` for smooth integrands.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration. Use np.inf to denote infinite intervals.
        Its dtype sets the working precision: the integrand is called with an ``x`` of
        this dtype, and the result follows it unless the integrand upcasts. A integer
        types or python floats falls back to the JAX default. Must be real; complex
        integrands are supported, complex limits are not.
    args : tuple
        additional arguments passed to fun
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float
        Absolute and relative tolerances. If I1 and I2 are two
        successive approximations to the integral, algorithm terminates
        when abs(I1-I2) < max(epsabs, epsrel*|I2|). Default is the square root of the
        machine precision of the working dtype, ie of `interval`, or of the integrand's
        own dtype if that is the coarser of the two.
    divmax : int, optional
        Maximum order of extrapolation. Default is 20.
        Total number of function evaluations will be at
        most 2**divmax + 1
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.


    Returns
    -------
    y  : float, Array
        Approximation to the integral
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (int) Flag indicating reason for termination. status of 0 means
          normal termination, any other value indicates a possible error. A human
          readable message can be obtained by ``print(quadax.STATUS[status])``
        * info : (dict or None) Other information returned by the algorithm.
          Only present if ``full_output`` is True. Contains the following:

          * table : (ndarray, size(dixmax+1, divmax+1, ...)) Estimate of the integral
            from each level of discretization and each step of extrapolation.

    Notes
    -----
    Due to limitations on dynamically sized arrays in JAX, this algorithm is fully
    sequential and does not vectorize integrand evaluations, so may not be the most
    efficient on GPU/TPU.

    """
    interval = jnp.atleast_1d(jnp.asarray(interval))
    if not jnp.issubdtype(interval.dtype, jnp.inexact):
        # integration limits must be inexact: they are differentiated with respect to,
        # and integer leaves would otherwise be treated as static metadata
        interval = interval.astype(jnp.result_type(float))
    errorif(
        len(interval) != 2,
        NotImplementedError,
        "tanh-sinh transformation with breakpoints not supported",
    )
    dtypes = resolve_dtypes(interval, fun, args)
    if epsabs is None:
        epsabs = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    if epsrel is None:
        epsrel = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    epsabs = jnp.asarray(epsabs, dtypes.etype)
    epsrel = jnp.asarray(epsrel, dtypes.etype)
    if callable(norm):
        _norm: Callable[[jax.Array], jax.Array] = norm
    else:
        _norm: Callable[[jax.Array], jax.Array] = partial(_pnorm, p=norm)

    return _romberg(
        fun,
        interval,
        args,
        full_output,
        epsabs,
        epsrel,
        divmax,
        _norm,
        adjoint,
        _build_tanhsinh,
        dtypes.xtype,
    )
