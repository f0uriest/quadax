"""Romberg integration aka adaptive trapezoid with Richardson extrapolation."""

from collections import namedtuple

import jax
import jax.numpy as jnp

from .utils import map_interval


def romberg(fun, a, b, args=(), epsabs=1e-8, epsrel=1e-8, divmax=20):
    """Romberg integration of a callable function or method.

    Returns the integral of `fun` (a function of one variable)
    over the interval (`a`, `b`).

    Parameters
    ----------
    fun : callable
        Function to be integrated.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple
        additional arguments passed to fun
    epsabs, epsrel : float
        Absolute and relative tolerances. If I1 and I2 are two
        successive approximations to the integral, algorithm terminates
        when abs(I1-I2) < max(epsabs, epsrel*|I2|)
    divmax : int, optional
        Maximum order of extrapolation. Default is 10.
        Total number of function evaluations will be at
        most 2**divmax + 1

    Returns
    -------
    y  : float
        Approximation to the integral
    info : namedtuple
        Extra information:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * table : (ndarray, size(dixmax+1, divmax+1)) Estimate of the integral form
          each level of discretization and each extrapolation step.

    """
    vfunc = jax.jit(jnp.vectorize(lambda x: fun(x, *args)))
    # map a, b -> [-1, 1]
    vfunc = map_interval(vfunc, a, b)
    eps = jnp.finfo(jnp.array(1.0)).eps
    # avoid evaluating at endpoints to avoid possible singularities
    a, b = -1 + eps, 1 - eps

    result = jnp.zeros((divmax + 1, divmax + 1))
    result = result.at[0, 0].set(vfunc(a) + vfunc(b))
    neval = 2
    err = jnp.inf
    state = (result, 1, neval, err)

    def ncond(state):
        result, n, neval, err = state
        return (n < divmax + 1) & (err > jnp.maximum(epsabs, epsrel * result[n, n]))

    def nloop(state):
        result, n, neval, err = state
        h = (b - a) / 2**n
        s = 0.0

        def sloop(i, s):
            s += vfunc(a + h * (2 * i - 1))
            return s

        result = result.at[n, 0].set(
            0.5 * result[n - 1, 0]
            + h * jax.lax.fori_loop(1, (2**n) // 2 + 1, sloop, s)
        )
        neval += (2**n) // 2

        def mloop(m, result):
            temp = 1 / (4.0**m - 1.0) * (result[n, m - 1] - result[n - 1, m - 1])
            result = result.at[n, m].set(result[n, m - 1] + temp)
            return result

        result = jax.lax.fori_loop(1, n + 1, mloop, result)
        err = abs(result[n, n] - result[n, n - 1])
        return result, n + 1, neval, err

    result, n, neval, err = jax.lax.while_loop(ncond, nloop, state)

    romberg_info = namedtuple("romberg_info", "err neval table")
    info = romberg_info(err, neval, result)
    return result[n - 1, n - 1], info
