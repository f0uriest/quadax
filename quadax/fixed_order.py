"""Fixed order quadrature."""

import abc
import functools
import warnings
from collections.abc import Callable
from typing import Any, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from .quad_weights import gk_weights
from .utils import wrap_func


def _dot(w, f):
    return jnp.sum(w * f.T, axis=-1).T


class AbstractQuadratureRule(eqx.Module):
    """Abstract base class for 1D quadrature rules.

    Subclasses should implement the ``integrate`` method for integrating a function
    over a fixed interval using the given rule.

    Subclasses may also override the ``norm`` method for measuring error for vector
    valued integrands. Default is the infinity (max) norm.
    """

    @abc.abstractmethod
    def integrate(
        self,
        fun: Callable,
        a: float,
        b: float,
        args: tuple[Any],
    ) -> tuple[float, float, float, float]:
        """Integrate ``fun(x, *args)`` from a to b.

        Parameters
        ----------
        fun : callable
            Function to integrate, should have a signature of the form
            ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
        a, b : float
            Lower and upper limits of integration. Must be finite.
        args : tuple, optional
            Extra arguments passed to fun.

        Returns
        -------
        y : float, Array
            Estimate of the integral of fun from a to b
        err : float
            Estimate of the absolute error in y.
        y_abs : float, Array
            Estimate of the integral of abs(fun) from a to b
        y_mmn : float, Array
            Estimate of the integral of abs(fun - <fun>) from a to b, where <fun>
            is the mean value of fun over the interval.

        """

    def norm(self, x: jax.Array) -> float:
        """Norm to use for measuring error for vector valued integrands."""
        return jnp.linalg.norm(jnp.asarray(x).flatten(), ord=jnp.inf)


class NestedRule(AbstractQuadratureRule):
    """Base class for nested quadrature rules.

    Nested rules consist of a set of nodes (xh) and weights (wh) for a high order rule,
    along with an additional set of weights (wl) for a lower order rule that shares
    nodes with the high order rule.
    """

    _xh: jax.Array
    _wh: jax.Array
    _wl: jax.Array
    _norm: Callable

    @eqx.filter_jit
    def integrate(
        self,
        fun: Callable,
        a: float,
        b: float,
        args: tuple[Any],
    ) -> tuple[float, float, float, float]:
        """Integrate a function from a to b using a nested rule.

        Parameters
        ----------
        fun : callable
            Function to integrate, should have a signature of the form
            ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
        a, b : float
            Lower and upper limits of integration. Must be finite.
        args : tuple, optional
            Extra arguments passed to fun.

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
        vfun = wrap_func(fun, args)

        def truefun():
            f = jax.eval_shape(vfun, jnp.array(0.0))
            z = jnp.zeros(f.shape, f.dtype)
            return z, self.norm(z), jnp.abs(z), jnp.abs(z)

        def falsefun():

            halflength = (b - a) / 2
            center = (b + a) / 2
            f = vfun(center + halflength * self._xh)
            result_kronrod = _dot(self._wh, f) * halflength
            result_gauss = _dot(self._wl, f) * halflength

            integral_abs = _dot(self._wh, jnp.abs(f))  # ~integral of abs(fun)
            integral_mmn = _dot(
                self._wh, jnp.abs(f - result_kronrod / (b - a))
            )  # ~ integral of abs(fun - mean(fun))

            result = result_kronrod

            uflow = jnp.finfo(f.dtype).tiny
            eps = jnp.finfo(f.dtype).eps
            abserr = jnp.abs(result_kronrod - result_gauss)
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
            return result, self.norm(abserr), integral_abs, integral_mmn

        return jax.lax.cond(a == b, truefun, falsefun)

    def norm(self, x):
        """Norm to use for measuring error for vector valued integrands."""
        return self._norm(x)


class GaussKronrodRule(NestedRule):
    """Integrate a function from a to b using a fixed order Gauss-Kronrod rule.

    Integration is performed using an order n Kronrod rule with error estimated
    using an embedded n//2 order Gauss rule.

    Parameters
    ----------
    order : {15, 21, 31, 41, 51, 61}
        Order of integration scheme.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    """

    def __init__(self, order: int = 21, norm: Union[Callable, int] = jnp.inf):
        self._norm = (
            norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
        )
        try:
            self._xh, self._wh, self._wl = (
                jnp.array(gk_weights[order]["xk"]),
                jnp.array(gk_weights[order]["wk"]),
                jnp.array(gk_weights[order]["wg"]),
            )
        except KeyError as e:
            raise NotImplementedError(
                f"order {order} not implemented, should be one of {gk_weights.keys()}"
            ) from e


class ClenshawCurtisRule(NestedRule):
    """Integrate a function from a to b using a fixed order Clenshaw-Curtis rule.

    Integration is performed using an order n rule with error estimated
    using an embedded n//2 order rule.

    Parameters
    ----------
    n : int
        Order of integration scheme. Must be even.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    """

    def __init__(self, order: int = 32, norm: Union[Callable, int] = jnp.inf):
        self._norm = (
            norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
        )

        def _cc_get_weights(N):
            d = 2 / (1 - (jnp.arange(0, N + 1, 2)) ** 2)
            d = d.at[0].multiply(1 / 2)
            d = d.at[-1].multiply(1 / 2)
            k = jnp.arange(N // 2 + 1)
            n = jnp.arange(N // 2 + 1)
            D = 2 / N * jnp.cos(k[:, None] * n[None, :] * jnp.pi / (N // 2))
            D = jnp.where((n == 0) | (n == N // 2), D * 1 / 2, D)
            w = D.T @ d  # can be done faster with fft
            t = jnp.arange(0, 1 + N // 2) * jnp.pi / N
            x = jnp.cos(t)
            w = w.at[-1].multiply(2)
            return x, w

        order = 2 * (order // 2)  # make sure its even
        xh, wh = _cc_get_weights(order)
        wl = _cc_get_weights(order // 2)[1]
        wl = jnp.zeros_like(wh).at[::2].set(wl)

        self._xh = jnp.concatenate([xh, -xh[:-1][::-1]])
        self._wh = jnp.concatenate([wh, wh[:-1][::-1]])
        self._wl = jnp.concatenate([wl, wl[:-1][::-1]])


class TanhSinhRule(NestedRule):
    """Integrate a function from a to b using a fixed order Tanh-Sinh trapezoidal rule.

    Integration is performed using an order n rule with error estimated
    using an embedded n//2 order rule.

    Parameters
    ----------
    order : int
        Order of integration scheme. Must be odd.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    """

    def __init__(self, order: int = 61, norm: Union[Callable, int] = jnp.inf):
        self._norm = (
            norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
        )
        _xts = lambda t: jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
        _wts = (
            lambda t: jnp.pi / 2 * jnp.cosh(t) / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2
        )

        def _get_tmax(xmax):
            # Inverse of tanh-sinh transform.
            tanhinv = lambda x: 1 / 2 * jnp.log((1 + x) / (1 - x))
            sinhinv = lambda x: jnp.log(x + jnp.sqrt(x**2 + 1))
            return sinhinv(2 / jnp.pi * tanhinv(xmax))

        tmax = _get_tmax(jnp.array(1.0) - 10 * jnp.finfo(jnp.array(1.0).dtype).eps)
        a, b = -tmax, tmax

        order = 2 * (order // 2) + 1  # make sure its odd

        th = jnp.linspace(a, b, order)
        tl = jnp.linspace(a, b, order // 2 + 1)

        xh = _xts(th)
        wh = _wts(th) * jnp.diff(th)[0]
        wl = _wts(tl) * jnp.diff(tl)[0]
        wl = jnp.zeros_like(wh).at[::2].set(wl)
        wh *= 2 / wh.sum()
        wl *= 2 / wl.sum()

        self._xh = xh
        self._wh = wh
        self._wl = wl


@functools.partial(jax.jit, static_argnums=(0, 4, 5))
def fixed_quadgk(fun, a, b, args=(), norm=jnp.inf, n=21):
    """Integrate a function from a to b using a fixed order Gauss-Kronrod rule.

    Integration is performed using an order n Kronrod rule with error estimated
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
    warnings.warn(
        "fixed_quadgk is deprecated and will be removed in a future release. "
        "Please use ``quadax.GaussKronrodRule(n, norm).integrate(fun, a, b, args)``",
        FutureWarning,
    )
    return GaussKronrodRule(n, norm).integrate(fun, a, b, args)


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
    warnings.warn(
        "fixed_quadcc is deprecated and will be removed in a future release. "
        "Please use ``quadax.ClenshawCurtisRule(n, norm).integrate(fun, a, b, args)``",
        FutureWarning,
    )
    return ClenshawCurtisRule(n, norm).integrate(fun, a, b, args)


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
    warnings.warn(
        "fixed_quadts is deprecated and will be removed in a future release. "
        "Please use ``quadax.TanhSinhRule(n, norm).integrate(fun, a, b, args)``",
        FutureWarning,
    )
    return TanhSinhRule(n, norm).integrate(fun, a, b, args)
