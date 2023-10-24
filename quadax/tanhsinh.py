"""Integration using tanh-sinh method."""

from collections import namedtuple

import jax
import jax.numpy as jnp

from .utils import map_interval


def tanhsinh_transform(fun):
    """Transform a function by mapping with tanh-sinh."""
    xk = lambda t: jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
    wk = lambda t: jnp.pi / 2 * jnp.cosh(t) / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2
    func = lambda t: fun(xk(t)) * wk(t)
    return jax.jit(func)


def get_tmax(xmax):
    """Inverse of tanh-sinh transform."""
    tanhinv = jax.jit(lambda x: 1 / 2 * jnp.log((1 + x) / (1 - x)))
    sinhinv = jax.jit(lambda x: jnp.log(x + jnp.sqrt(x**2 + 1)))
    return sinhinv(2 / jnp.pi * tanhinv(xmax))


def quadts(fun, a, b, args=(), epsabs=1e-8, epsrel=1e-8, divmax=20):
    """Global adaptive quadrature using tanh-sinh (aka double exponential) method.

    Integrate fun from a to b using a h-adaptive scheme with error estimate.

    Differentiation wrt args is done via Liebniz rule.

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
        * table : (ndarray, size(dixmax+1)) Estimate of the integral form each level
          of discretization.

    """
    func = jnp.vectorize(lambda x: fun(x, *args))
    # map a, b -> [-1, 1]
    func = map_interval(func, a, b)
    # map [-1, 1] to [-inf, inf], but with mass concentrated near 0
    func = tanhsinh_transform(func)

    # we generally only need to integrate ~[-3, 3] or ~[-4, 4]
    # we don't want to include the endpoint that maps to x==1 to avoid
    # possible singularities, so we find the largest t s.t. x(t) < 1
    # and use that as our interval
    # inverse of tanh-sinh transformation for x = 1-eps
    tmax = get_tmax(jnp.array(1.0) - 10 * jnp.finfo(jnp.array(1.0)).eps)

    a, b = -tmax, tmax
    result = jnp.zeros(divmax + 1)
    result = result.at[0].set((b - a) / 2 * (func(a) + func(b)))
    neval = 2
    err = jnp.inf
    state = (result, 1, neval, err)

    def ncond(state):
        result, n, neval, err = state
        return (n < divmax + 1) & (err > jnp.maximum(epsabs, epsrel * result[n]))

    def nloop(state):
        result, n, neval, err = state
        h = (b - a) / 2**n
        s = 0.0

        def sloop(i, s):
            s += func(a + h * (2 * i - 1))
            return s

        result = result.at[n].set(
            0.5 * result[n - 1] + h * jax.lax.fori_loop(1, (2**n) // 2 + 1, sloop, s)
        )
        neval += (2**n) // 2

        err = abs(result[n] - result[n - 1])
        return result, n + 1, neval, err

    result, n, neval, err = jax.lax.while_loop(ncond, nloop, state)

    quadts_info = namedtuple("quadts_info", "err neval table")
    info = quadts_info(err, neval, result)
    return result[n - 1], info
