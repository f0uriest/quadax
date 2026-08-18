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
    check_size,
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
    extrapolate: bool = True,
    adjoint: AbstractAdjoint = DirectAdjoint(),
    batch_size: int | None = None,
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
    extrapolate : bool, optional
        Whether to accelerate convergence by Richardson extrapolation, which is what
        makes this Romberg's method rather than plain repeated bisection. On by default.
        Turning it off leaves the same nodes and the same halving schedule, reading the
        un-extrapolated estimate instead, which is worth having when the integrand is
        not smooth enough for the extrapolation's error expansion to hold. There it
        can amplify rather than cancel, and the honest estimate is the better one.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel. Default
        is one at a time. Each refinement level doubles the number of new points, so
        raising this is usually worth a lot on GPU/TPU, at the cost of peak memory
        scaling with it. Levels with fewer new points than one batch are padded up to a
        full batch, so the early levels of a run cost ``batch_size`` evaluations each
        however few points they place; that padding is what keeps a single batch shape
        traced for every level rather than one per level. Clipped to the largest level
        ``divmax`` allows.

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
            from each level of discretization and each step of extrapolation. With
            ``extrapolate=False`` only the first column is filled.

    Notes
    -----
    The number of new points a refinement level places is only known at run time, so
    there is no single shape to vectorize over. Integrand evaluations are made in
    fixed size batches of ``batch_size``, defaulting to one at a time; raise it to get
    parallelism on GPU/TPU.

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
    check_size(batch_size)
    # No level ever places more than `2**(divmax - 1)` new points, so a larger batch
    # would only ever be padding.
    batch_size = min(batch_size or 1, 2 ** max(divmax - 1, 0))
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
        extrapolate,
        batch_size,
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
    extrapolate,
    batch_size,
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
        solve=partial(
            _romberg_solve,
            divmax=divmax,
            _norm=_norm,
            extrapolate=extrapolate,
            batch_size=batch_size,
        ),
        # Romberg has no subdivision to reuse, but it does settle on a number of
        # Richardson levels. Freezing that makes the result a fixed linear functional of
        # the integrand, which is what DirectAdjoint needs to differentiate in either
        # direction. It has to go through a custom primitive rather than being
        # differentiated directly, because evaluating it still involves a fori_loop with
        # dynamic bounds that JAX cannot reverse differentiate.
        frozen=lambda state: state["n"],
        frozen_solve=partial(
            _romberg_levels,
            divmax=divmax,
            extrapolate=extrapolate,
            batch_size=batch_size,
        ),
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


def _level_sum(vfunc, a, h, npts, batch_size, shape, dtype):
    """Sum the integrand over the new nodes of one refinement level.

    Level ``k`` adds the ``npts = 2**(k - 1)`` points sitting at odd multiples of ``h``
    above ``a``, interleaving the nodes the previous levels already placed. ``npts`` is
    only known at run time, so the points are evaluated in fixed size batches and the
    last batch is padded up to a full one. Padding rather than shaping each level to its
    own point count is what keeps one batch traced for all of them; the cost is that a
    level with fewer points than a batch still pays for a whole one.

    The padded lanes repeat the first point of their batch, which is always one this
    level genuinely evaluates, not an arbitrary placeholder. An integrand that is
    singular somewhere in the domain would return a non-finite value at a made up point,
    and masking that out of the sum afterwards still leaves a NaN in its derivative.

    ``vfunc`` takes a whole batch of abscissae; the callers wrap it to guarantee that.
    """
    nbatch = (npts + batch_size - 1) // batch_size
    offs = jnp.arange(1, batch_size + 1)

    def bodyfun(j, s):
        i = j * batch_size + offs
        used = i <= npts
        x = a + h * (2 * i - 1)
        x = jnp.where(used, x, x[0])
        f: jax.Array = vfunc(x)
        mask = used.reshape((-1,) + (1,) * (f.ndim - 1))
        return s + jnp.sum(jnp.where(mask, f, 0), axis=0)

    return jax.lax.fori_loop(0, nbatch, bodyfun, jnp.zeros(shape, dtype)), nbatch


def _romberg_solve(
    rule,
    vfunc,
    interval,
    epsabs,
    epsrel,
    kwargs,
    *,
    divmax,
    _norm,
    extrapolate=True,
    batch_size=1,
):
    """Run the refinement loop, with Richardson extrapolation if it is switched on.

    Without it this is plain adaptive bisection of the trapezoidal rule (or of the
    tanh-sinh rule, for ``rombergts``): the same nodes and the same halving schedule,
    reading the un-extrapolated column of the table instead of its diagonal.
    """
    del rule, kwargs
    a, b = interval
    # Vectorize whatever we were handed The primal integrand arrives vectorized already,
    # but the adjoints solve against one they build per point (the tangent or the
    # cotangent of the mapped integrand) which takes a scalar only. Wrapping here rather
    # than mapping at each use keeps one contract for the loop below: ``vfunc`` accepts
    # a batch of abscissae.
    vfunc = wrap_func(vfunc, (), interval.dtype)
    f = jax.eval_shape(vfunc, (a + b) / 2)

    # Which entry of row `k` is the estimate. Richardson's is the diagonal, having
    # applied `k` rounds of extrapolation to the trapezoidal values in column 0; without
    # it the estimate is that column, and the rest of the table is never written.
    best = (lambda res, k: res[k, k]) if extrapolate else (lambda res, k: res[k, 0])

    result = jnp.zeros((divmax + 1, divmax + 1, *f.shape), f.dtype)
    # The trapezoid rule at one interval.
    result = result.at[0, 0].set((b - a) / 2 * (vfunc(a) + vfunc(b)))
    neval = 2
    # Explicitly typed rather than left a weak python float: this is a loop carry, and
    # has to match what `_norm` writes back into it. Real, because the error in a
    # complex valued integral is still real.
    err = jnp.array(jnp.inf, _real_dtype(f.dtype))
    state = (result, 1, neval, err)

    def ncond(state):
        result, n, neval, err = state
        return (n < divmax + 1) & (
            err > jnp.maximum(epsabs, epsrel * _norm(best(result, n)))
        )

    def nloop(state):
        # loop over outer number of subdivisions
        result, n, neval, err = state
        h = (b - a) / 2**n
        s, nbatch = _level_sum(vfunc, a, h, (2**n) // 2, batch_size, f.shape, f.dtype)
        result = result.at[n, 0].set(0.5 * result[n - 1, 0] + h * s)
        # The padded lanes of the last batch are evaluations of the integrand like any
        # other, so they are counted here even though they do not reach the sum.
        neval += nbatch * batch_size

        def mloop(m, result):
            # richardson extrapolation
            temp = 1 / (4.0**m - 1.0) * (result[n, m - 1] - result[n - 1, m - 1])
            result = result.at[n, m].set(result[n, m - 1] + temp)
            return result

        if extrapolate:
            result = jax.lax.fori_loop(1, n + 1, mloop, result)
        err = _norm(best(result, n) - best(result, n - 1))
        return result, n + 1, neval, err

    result, n, neval, err = bounded_while_loop(ncond, nloop, state, divmax + 1)

    y = best(result, n - 1)
    status = 2 * (err > jnp.maximum(epsabs, epsrel * _norm(y)))
    state = {
        "table": result,
        "err_sum": err,
        "neval": neval,
        "status": status,
        "n": n,  # Richardson levels used; frozen by DirectAdjoint
    }
    return y, state


def _romberg_levels(
    rule, vfunc, interval, n, kwargs, *, divmax, extrapolate=True, batch_size=1
):
    """Evaluate the table at a fixed number of levels.

    With ``n`` fixed this is a fixed linear combination of the integrand at fixed nodes,
    so its forward and reverse derivatives are exact transposes of one another. Mirrors
    the loop in ``_romberg_solve`` exactly so the two agree, including which entry of
    the table is read.
    """
    del rule, kwargs
    a, b = interval[0], interval[-1]
    vfunc = wrap_func(vfunc, (), interval.dtype)  # see ``_romberg_solve``
    f = jax.eval_shape(vfunc, (a + b) / 2)
    result = jnp.zeros((divmax + 1, divmax + 1, *f.shape), f.dtype)
    # The trapezoid rule at one interval.
    result = result.at[0, 0].set((b - a) / 2 * (vfunc(a) + vfunc(b)))

    def nloop(k, result):
        h = (b - a) / 2**k
        s, _ = _level_sum(vfunc, a, h, (2**k) // 2, batch_size, f.shape, f.dtype)
        result = result.at[k, 0].set(0.5 * result[k - 1, 0] + h * s)

        def mloop(m, result):
            temp = 1 / (4.0**m - 1.0) * (result[k, m - 1] - result[k - 1, m - 1])
            return result.at[k, m].set(result[k, m - 1] + temp)

        if not extrapolate:
            return result
        return jax.lax.fori_loop(1, k + 1, mloop, result)

    result = jax.lax.fori_loop(1, n, nloop, result)
    return result[n - 1, n - 1] if extrapolate else result[n - 1, 0]


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
    extrapolate: bool = True,
    adjoint: AbstractAdjoint = DirectAdjoint(),
    batch_size: int | None = None,
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
    extrapolate : bool, optional
        Whether to accelerate convergence by Richardson extrapolation, which is what
        makes this Romberg's method rather than plain repeated bisection. On by default.
        Turning it off leaves the same nodes and the same halving schedule, reading the
        un-extrapolated estimate instead, which is worth having when the integrand is
        not smooth enough for the extrapolation's error expansion to hold. There it
        can amplify rather than cancel, and the honest estimate is the better one.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel. Default
        is one at a time. Each refinement level doubles the number of new points, so
        raising this is usually worth a lot on GPU/TPU, at the cost of peak memory
        scaling with it. Levels with fewer new points than one batch are padded up to a
        full batch, so the early levels of a run cost ``batch_size`` evaluations each
        however few points they place; that padding is what keeps a single batch shape
        traced for every level rather than one per level. Clipped to the largest level
        ``divmax`` allows.


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
            from each level of discretization and each step of extrapolation. With
            ``extrapolate=False`` only the first column is filled.

    Notes
    -----
    The number of new points a refinement level places is only known at run time, so
    there is no single shape to vectorize over. Integrand evaluations are made in
    fixed size batches of ``batch_size``, defaulting to one at a time; raise it to get
    parallelism on GPU/TPU.

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
    check_size(batch_size)
    # No level ever places more than `2**(divmax - 1)` new points, so a larger batch
    # would only ever be padding.
    batch_size = min(batch_size or 1, 2 ** max(divmax - 1, 0))
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
        extrapolate,
        batch_size,
    )
