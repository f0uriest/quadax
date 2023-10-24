"""Utility functions for parsing inputs, mapping coordinates etc."""

import jax
import jax.numpy as jnp


def _x_map_linear(x, a, b):
    c = (b - a) / 2
    d = (b + a) / 2
    x = d + c * x
    w = c
    return x.squeeze(), w.squeeze()


def _x_map_ninfinf(x, a, b):
    x2 = x * x
    px1 = 1 - x2
    spx1 = 1 / px1**0.5
    x = x * spx1
    w = spx1 / px1
    return x.squeeze(), w.squeeze()


def _x_map_ainf(x, a, b):
    u = 2 / (x + 1)
    x = a - 1 + u
    w = 0.5 * u**2
    return x.squeeze(), w.squeeze()


def _x_map_ninfb(x, a, b):
    u = 2 / (x + 1)
    x = b + 1 - u
    w = 0.5 * u**2
    return x.squeeze(), w.squeeze()


def map_interval(fun, a, b):
    """Map a function over an arbitrary interval [a, b] to the interval [-1, 1]."""
    sgn = (-1) ** (a > b)
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)

    # bitmask to select mapping case
    # 0 : both sides finite
    # 1 : a = -inf, b finite
    # 2 : a finite, b = inf
    # 3 : both infinite
    bitmask = jnp.isinf(a) + 2 * jnp.isinf(b)

    @jax.jit
    def fun_mapped(x):
        x, w = jax.lax.switch(
            bitmask, [_x_map_linear, _x_map_ninfb, _x_map_ainf, _x_map_ninfinf], x, a, b
        )
        return sgn * w * fun(x)

    return fun_mapped


messages = {
    0: "Algorithm terminated normally, desired tolerances assumed reached",
    1: (
        """
        Maximum number of subdivisions allowed has been achieved. One can allow more
        subdivisions by increasing the value of max_ninter. However,if this yields no
        improvement it is advised to analyze the integrand in order to determine the
        integration difficulties. If the position of a local difficulty can be
        determined (e.g. singularity, discontinuity within the interval) one will
        probably gain from splitting up the interval at this point and calling the
        integrator on the sub-ranges. If possible, an appropriate special-purpose
        integrator should be used, which is designed for handling the type of
        difficulty involved.
        """
    ),
    2: (
        """
        The occurrence of roundoff error is detected, which prevents the requested
        tolerance from being achieved. The error may be under-estimated.
        """
    ),
    3: (
        """
        Extremely bad integrand behavior occurs at some points of the integration
        interval.
        """
    ),
    4: (
        """
        The algorithm does not converge. Roundoff error is detected in the
        extrapolation table. It is assumed that the requested tolerance cannot be
        achieved, and that the returned result is the best which can be obtained.
        """
    ),
    5: "The integral is probably divergent, or slowly convergent.",
}
