"""Integration using tanh-sinh method."""

from functools import partial

import jax
import jax.numpy as jnp

from .utils import QuadratureInfo, map_interval, wrap_func


def tanhsinh_transform(fun):
    """Transform a function by mapping with tanh-sinh."""
    xk = lambda t: jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
    wk = lambda t: jnp.pi / 2 * jnp.cosh(t) / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2
    func = lambda t, *args: fun(xk(t), *args) * wk(t)
    return jax.jit(func)


def get_tmax(xmax):
    """Inverse of tanh-sinh transform."""
    tanhinv = jax.jit(lambda x: 1 / 2 * jnp.log((1 + x) / (1 - x)))
    sinhinv = jax.jit(lambda x: jnp.log(x + jnp.sqrt(x**2 + 1)))
    return sinhinv(2 / jnp.pi * tanhinv(xmax))


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def quadts(
    fun, a, b, args=(), full_output=False, epsabs=1.4e-8, epsrel=1.4e-8, divmax=20
):
    """Global adaptive quadrature using tanh-sinh (aka double exponential) method.

    Integrate fun from a to b using a p-adaptive scheme with error estimate.

    Performs well for functions with singularities at the endpoints or integration
    over infinite intervals. May be slightly less efficient than ``quadgk`` or
    ``quadcc`` for smooth integrands.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float. Should be JAX transformable.
    a, b : float
        Lower and upper limits of integration. Use np.inf to denote infinite intervals.
    args : tuple
        additional arguments passed to fun
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
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
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (int) Flag indicating reason for termination. status of 0 means
          normal termination, any other value indicates a possible error. A human
          readable message can be obtained by ``print(quadax.STATUS[status])``
        * info : (dict or None) Other information returned by the algorithm.
          Only present if ``full_output`` is True. Contains the following:

            - table : (ndarray, size(dixmax+1)) Estimate of the integral from each level
              of discretization.

    """
    # map a, b -> [-1, 1]
    fun = map_interval(fun, a, b)
    # map [-1, 1] to [-inf, inf], but with mass concentrated near 0
    fun = tanhsinh_transform(fun)
    func = wrap_func(fun, args)
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

    y = result[n - 1]
    status = 2 * (err > jnp.maximum(epsabs, epsrel * y))
    info = result if full_output else None
    out = QuadratureInfo(err, neval, status, info)
    return y, out


@quadts.defjvp
def _quadts_jvp(fun, primals, tangents):
    a, b, args = primals[:3]
    adot, bdot, argsdot = tangents[:3]
    f1, info1 = quadts(fun, *primals)

    def df(x, *args):
        return jax.jvp(fun, (x, *args), (jnp.zeros_like(x), *argsdot))[1]

    f2, info2 = quadts(df, *primals)
    return (f1, info1), (fun(b, *args) * bdot - fun(a, *args) * adot + f2, info2)
