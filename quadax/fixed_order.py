"""Fixed order quadrature."""

import functools

import jax
import jax.numpy as jnp

from .quad_weights import cc_weights, gk_weights, ts_weights
from .utils import wrap_func


def _dot(w, f):
    return jnp.sum(w * f.T, axis=-1).T


@functools.partial(jax.jit, static_argnums=(0, 4, 5))
def fixed_quadgk(fun, a, b, args=(), norm=jnp.inf, n=21):
    """Integrate a function from a to b using a fixed order Gauss-Konrod rule.

    Integration is performed using an order n Konrod rule with error estimated
    using an embedded n//2 order Gauss rule.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    a, b : float
        Lower and upper limits of integration. Must be finite.
    args : tuple, optional
        Extra arguments passed to fun.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    n : {15, 21, 31, 41, 51, 61}
        Order of integration scheme.

    Returns
    -------
    y : float, Array
        Estimate of the integral of fun from a to b
    err : float
        Estimate of the absolute error in y from nested Gauss rule.
    y_abs : float, Array
        Estimate of the integral of abs(fun) from a to b
    y_mmn : float, Array
        Estimate of the integral of abs(fun - <fun>) from a to b, where <fun>
        is the mean value of fun over the interval.

    """
    _norm = norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
    vfun = wrap_func(fun, args)

    def truefun():
        f = jax.eval_shape(vfun, jnp.array(0.0))
        z = jnp.zeros(f.shape, f.dtype)
        return z, 0.0, z, z

    def falsefun():
        try:
            xk, wk, wg = (
                gk_weights[n]["xk"],
                gk_weights[n]["wk"],
                gk_weights[n]["wg"],
            )
        except KeyError as e:
            raise NotImplementedError(
                f"order {n} not implemented, should be one of {gk_weights.keys()}"
            ) from e

        halflength = (b - a) / 2
        center = (b + a) / 2
        f = vfun(center + halflength * xk)
        result_konrod = _dot(wk, f) * halflength
        result_gauss = _dot(wg, f) * halflength

        integral_abs = _dot(wk, jnp.abs(f))  # ~integral of abs(fun)
        integral_mmn = _dot(
            wk, jnp.abs(f - result_konrod / (b - a))
        )  # ~ integral of abs(fun - mean(fun))

        result = result_konrod

        uflow = jnp.finfo(f.dtype).tiny
        eps = jnp.finfo(f.dtype).eps
        abserr = jnp.abs(result_konrod - result_gauss)
        abserr = jnp.where(
            (integral_mmn != 0.0) & (abserr != 0.0),
            integral_mmn * jnp.minimum(1.0, (200.0 * abserr / integral_mmn) ** 1.5),
            abserr,
        )
        abserr = jnp.where(
            (integral_abs > uflow / (50.0 * eps)),
            jnp.maximum((eps * 50.0) * integral_abs, abserr),
            abserr,
        )
        return result, _norm(abserr), integral_abs, integral_mmn

    return jax.lax.cond(a == b, truefun, falsefun)


@functools.partial(jax.jit, static_argnums=(0, 4, 5))
def fixed_quadcc(fun, a, b, args=(), norm=jnp.inf, n=32):
    """Integrate a function from a to b using a fixed order Clenshaw-Curtis rule.

    Integration is performed using an order n rule with error estimated
    using an embedded n//2 order rule.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    a, b : float
        Lower and upper limits of integration. Must be finite.
    args : tuple, optional
        Extra arguments passed to fun.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    n : {8, 16, 32, 64, 128, 256}
        Order of integration scheme.

    Returns
    -------
    y : float, Array
        Estimate of the integral of fun from a to b
    err : float
        Estimate of the absolute error in y from nested rule.
    y_abs : float, Array
        Estimate of the integral of abs(fun) from a to b
    y_mmn : float, Array
        Estimate of the integral of abs(fun - <fun>) from a to b, where <fun>
        is the mean value of fun over the interval.

    """
    _norm = norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
    vfun = wrap_func(fun, args)

    def truefun():
        f = jax.eval_shape(vfun, jnp.array(0.0))
        z = jnp.zeros(f.shape, f.dtype)
        return z, 0.0, z, z

    def falsefun():
        try:
            xc, wc, we = (
                cc_weights[n]["xc"],
                cc_weights[n]["wc"],
                cc_weights[n]["we"],
            )
        except KeyError as e:
            raise NotImplementedError(
                f"order {n} not implemented, should be one of {cc_weights.keys()}"
            ) from e

        halflength = (b - a) / 2
        center = (b + a) / 2
        fp = vfun(center + halflength * xc)
        fm = vfun(center - halflength * xc)
        result_2 = _dot(wc, (fp + fm)) * halflength
        result_1 = _dot(we, (fp + fm)) * halflength

        integral_abs = _dot(wc, (jnp.abs(fp) + jnp.abs(fm)))  # ~integral of abs(fun)
        integral_mmn = _dot(
            wc, jnp.abs(fp + fm - result_2 / (b - a))
        )  # ~ integral of abs(fun - mean(fun))

        result = result_2

        uflow = jnp.finfo(fp.dtype).tiny
        eps = jnp.finfo(fp.dtype).eps
        abserr = jnp.abs(result_2 - result_1)
        abserr = jnp.where(
            (integral_mmn != 0.0) & (abserr != 0.0),
            integral_mmn * jnp.minimum(1.0, (200.0 * abserr / integral_mmn) ** 1.5),
            abserr,
        )
        abserr = jnp.where(
            (integral_abs > uflow / (50.0 * eps)),
            jnp.maximum((eps * 50.0) * integral_abs, abserr),
            abserr,
        )
        return result, _norm(abserr), integral_abs, integral_mmn

    return jax.lax.cond(a == b, truefun, falsefun)


@functools.partial(jax.jit, static_argnums=(0, 4, 5))
def fixed_quadts(fun, a, b, args=(), norm=jnp.inf, n=61):
    """Integrate a function from a to b using a fixed order tanh-sinh rule.

    Integration is performed using an order n rule with error estimated
    using an embedded n//2 order rule.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    a, b : float
        Lower and upper limits of integration. Must be finite.
    args : tuple, optional
        Extra arguments passed to fun.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    n : {41, 61, 81, 101}
        Order of integration scheme.

    Returns
    -------
    y : float, Array
        Estimate of the integral of fun from a to b
    err : float
        Estimate of the absolute error in y from nested rule.
    y_abs : float, Array
        Estimate of the integral of abs(fun) from a to b
    y_mmn : float, Array
        Estimate of the integral of abs(fun - <fun>) from a to b, where <fun>
        is the mean value of fun over the interval.

    """
    _norm = norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
    vfun = wrap_func(fun, args)

    def truefun():
        f = jax.eval_shape(vfun, jnp.array(0.0))
        z = jnp.zeros(f.shape, f.dtype)
        return z, 0.0, z, z

    def falsefun():
        try:
            xt, wt, we = (
                ts_weights[n]["xt"],
                ts_weights[n]["wt"],
                ts_weights[n]["we"],
            )
        except KeyError as e:
            raise NotImplementedError(
                f"order {n} not implemented, should be one of {ts_weights.keys()}"
            ) from e

        halflength = (b - a) / 2
        center = (b + a) / 2
        f = vfun(center + halflength * xt) * halflength

        result_2 = _dot(wt, f)
        result_1 = _dot(we, f[::2])

        integral_abs = _dot(wt, jnp.abs(f))  # ~integral of abs(fun)
        integral_mmn = _dot(
            wt, jnp.abs(f - result_2 / (b - a))
        )  # ~ integral of abs(fun - mean(fun))

        result = result_2

        uflow = jnp.finfo(f.dtype).tiny
        eps = jnp.finfo(f.dtype).eps
        abserr = jnp.abs(result_2 - result_1)
        abserr = jnp.where(
            (integral_mmn != 0.0) & (abserr != 0.0),
            integral_mmn * jnp.minimum(1.0, (200.0 * abserr / integral_mmn) ** 1.5),
            abserr,
        )
        abserr = jnp.where(
            (integral_abs > uflow / (50.0 * eps)),
            jnp.maximum((eps * 50.0) * integral_abs, abserr),
            abserr,
        )
        return result, _norm(abserr), integral_abs, integral_mmn

    return jax.lax.cond(a == b, truefun, falsefun)
