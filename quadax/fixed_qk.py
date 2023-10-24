"""Fixed order quadrature."""

import functools

import jax
import jax.numpy as jnp

from .quad_weights import quad_weights


@functools.partial(jax.jit, static_argnums=(0, 4))
def fixed_quadgk(fun, a, b, args, n=21):
    """Integrate a function from a to b using a fixed order Gauss-Konrod rule.

    Integration is performed using and order n Konrod rule with error estimated
    using an embedded n//2 order Gauss rule.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        fun(x, *args) -> float. Should be JAX transformable.
    a, b : float
        Lower and upper limits of integration.
    args : tuple, optional
        Extra arguments passed to fun, and possibly a, b.
    n : {15, 21, 31, 41, 51, 61}
        Order of integration scheme.

    Returns
    -------
    y : float
        Estimate of the integral of fun from a to b
    err : float
        Estimate of the absolute error in y from nested Gauss rule.
    y_abs : float
        Estimate of the integral of abs(fun) from a to b
    y_mmn : float
        Estimate of the integral of abs(fun - <fun>) from a to b, where <fun>
        is the mean value of fun over the interval.

    """
    vfun = jnp.vectorize(lambda x: fun(x, *args))

    def truefun():
        return 0.0, 0.0, 0.0, 0.0

    def falsefun():
        try:
            xk, wk, wg = (
                quad_weights[n]["konrod_nodes"],
                quad_weights[n]["konrod_weights"],
                quad_weights[n]["gauss_weights"],
            )
        except KeyError as e:
            raise NotImplementedError(
                f"order {n} not implemented, should be one of {quad_weights.keys()}"
            ) from e

        halflength = (b - a) / 2
        center = (b + a) / 2
        f = vfun(center + halflength * xk)
        result_konrod = jnp.sum(wk * f) * halflength
        result_gauss = jnp.sum(wg * f) * halflength

        integral_abs = jnp.sum(wk * jnp.abs(result_konrod))  # ~integral of abs(fun)
        integral_mmn = jnp.sum(
            wk * jnp.abs(f - result_konrod / (b - a))
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
        return result, abserr, integral_abs, integral_mmn

    return jax.lax.cond(a == b, truefun, falsefun)
