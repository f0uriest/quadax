"""Romberg integration aka adaptive trapezoid with Richardson extrapolation."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .utils import (
    QuadratureInfo,
    bounded_while_loop,
    errorif,
    map_interval,
    setdefault,
    tanhsinh_transform,
    wrap_func,
)


@eqx.filter_jit
def romberg(
    fun,
    interval,
    args=(),
    full_output=False,
    epsabs=None,
    epsrel=None,
    divmax=20,
    norm=jnp.inf,
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
    args : tuple
        additional arguments passed to fun
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float
        Absolute and relative tolerances. If I1 and I2 are two
        successive approximations to the integral, algorithm terminates
        when abs(I1-I2) < max(epsabs, epsrel*|I2|). Default is square root of
        machine precision.
    divmax : int, optional
        Maximum order of extrapolation. Default is 20.
        Total number of function evaluations will be at
        most 2**divmax + 1
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.

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

    Also, it is currently only forward mode differentiable.

    """
    errorif(
        len(interval) != 2,
        NotImplementedError,
        "Romberg integration with breakpoints not supported",
    )
    epsabs = setdefault(epsabs, jnp.sqrt(jnp.finfo(jnp.array(1.0)).eps))
    epsrel = setdefault(epsrel, jnp.sqrt(jnp.finfo(jnp.array(1.0)).eps))
    _norm = norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
    # map a, b -> [-1, 1]
    fun, interval = map_interval(fun, interval)
    vfunc = wrap_func(fun, args)
    a, b = interval
    f = jax.eval_shape(vfunc, (a + b) / 2)

    result = jnp.zeros((divmax + 1, divmax + 1, *f.shape), f.dtype)
    result = result.at[0, 0].set(vfunc(a) + vfunc(b))
    neval = 2
    err = jnp.inf
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
    info = result if full_output else None
    out = QuadratureInfo(err, neval, status, info)
    return y, out


@eqx.filter_jit
def rombergts(
    fun,
    interval,
    args=(),
    full_output=False,
    epsabs=None,
    epsrel=None,
    divmax=20,
    norm=jnp.inf,
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
    args : tuple
        additional arguments passed to fun
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float
        Absolute and relative tolerances. If I1 and I2 are two
        successive approximations to the integral, algorithm terminates
        when abs(I1-I2) < max(epsabs, epsrel*|I2|). Default is square root of
        machine precision.
    divmax : int, optional
        Maximum order of extrapolation. Default is 20.
        Total number of function evaluations will be at
        most 2**divmax + 1
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.


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

    Also, it is currently only forward mode differentiable.

    """
    fun, interval = tanhsinh_transform(fun, interval)
    return romberg(fun, interval, args, full_output, epsabs, epsrel, divmax, norm)
