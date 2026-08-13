"""Fixed order quadrature."""

import abc
from collections.abc import Callable
from typing import Any

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
        fun: Callable[..., jax.Array],
        a: float,
        b: float,
        args: tuple[Any, ...],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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

    def _apply(
        self,
        fun: Callable[..., jax.Array],
        a: float,
        b: float,
        args: tuple[Any, ...],
    ) -> jax.Array:
        """Integrate ``fun(x, *args)`` from a to b, without an error estimate.

        Internal: used by the adjoints where many sub-intervals are evaluated at once
        and only the values are wanted. Users writing a custom rule do not need to
        touch this; the default below is correct, and subclasses only override it when
        they can compute the value more cheaply than by discarding the error estimate
        from :meth:`integrate`.

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
            Estimate of the integral of fun from a to b.

        """
        return self.integrate(fun, a, b, args)[0]

    def norm(self, x: jax.Array) -> jax.Array:
        """Norm to use for measuring error for vector valued integrands."""
        return jnp.linalg.norm(jnp.asarray(x).flatten(), ord=jnp.inf)


class NestedRule(AbstractQuadratureRule):
    """Base class for nested quadrature rules.

    Nested rules consist of a set of nodes (xh) and weights (wh) for a high order rule,
    along with an additional set of weights (wl) for a lower order rule that shares
    nodes with the high order rule.

    Notes
    -----
    The error estimate is derived from the difference between the two rules, so it is
    only meaningful while the nodes resolve the integrand. For oscillatory integrands,
    below roughly three points per oscillation both rules alias and agree spuriously,
    and the estimate can fall well below the true error; above roughly eight it is
    reliably conservative. This is a property of nested rules in general rather than of
    any particular order, since raising the order only raises the frequency at which it
    sets in. Strongly oscillatory integrands are better served by a specialized method.

    References
    ----------
    .. [1] R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber, D. K. Kahaner.
           "QUADPACK: A Subroutine Package for Automatic Integration". Springer Series
           in Computational Mathematics, vol. 1. Springer-Verlag, Berlin, 1983.
           doi:10.1007/978-3-642-61786-7
    """

    _xh: jax.Array
    _wh: jax.Array
    _wl: jax.Array
    _norm: float | int | Callable

    @eqx.filter_jit
    def integrate(
        self,
        fun: Callable[..., jax.Array],
        a: float,
        b: float,
        args: tuple[Any, ...],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
            f: jax.Array = vfun(center + halflength * self._xh)
            result_kronrod = _dot(self._wh, f) * halflength
            result_gauss = _dot(self._wl, f) * halflength

            # Both of these are sums over the reference interval [-1, 1] and so, like
            # the two results above, need the Jacobian of the map onto [a, b] to be an
            # estimate of an integral over [a, b]. QUADPACK scales all four by
            # ``dhlgth``; the error estimate below compares ``abserr`` against
            # ``integral_mmn``, so the two have to be on the same scale for the
            # ``200 ... **1.5`` interpolation to mean what it was tuned to mean.
            dhalflength = jnp.abs(halflength)
            integral_abs = (
                _dot(self._wh, jnp.abs(f)) * dhalflength
            )  # ~integral of abs(fun)
            integral_mmn = (
                _dot(self._wh, jnp.abs(f - result_kronrod / (b - a))) * dhalflength
            )  # ~ integral of abs(fun - mean(fun))

            result = result_kronrod

            uflow = jnp.finfo(f.dtype).tiny
            eps = jnp.finfo(f.dtype).eps

            # The difference between the two rules is dominated by the error of the
            # *low* order one, so it says little about the error of ``result``, which
            # comes from the high order rule and is typically far smaller. It is only a
            # starting point, and is rescaled below rather than reported directly.
            abserr = jnp.abs(result_kronrod - result_gauss)

            # Measure that discrepancy against how much the integrand varies over the
            # interval. This rescaling, and the 200 and 1.5 in it, are QUADPACK's, fit
            # empirically there rather than derived; see [1] and the ``dqk*`` routines,
            # which use the same two constants at every order from 15 through 61.
            # With ``r = abserr / integral_mmn`` the estimate becomes
            # ``integral_mmn * min(1, (200*r)**1.5)``, which has three regimes:
            #   - ``r >= 1/200``: saturate at ``integral_mmn``. The rules disagree at
            #     the scale of the variation of the integrand, so nothing has been
            #     resolved and the whole variation is the only honest bound.
            #   - middle: inflate the raw difference, by up to ~200x. This is most of
            #     the useful range, so the rescaling is usually pessimistic rather than
            #     optimistic, contrary to how the formula first reads.
            #   - ``r < 200**-1.5``: deflate, the regime where the two rules agree to
            #     near machine precision and the difference genuinely overstates the
            #     error of the high order rule.
            # The exponent is the ratio of the two rules' convergence rates: a rule
            # exact to degree ``d`` has local error ``~h**(d+2)``, giving
            # ``(d_high+2)/(d_low+2)``. 1.5 is the large-order limit of that ratio for
            # Gauss-Kronrod, and a lower bound across every rule implemented here, so it
            # is the conservative choice, a larger exponent would shrink the estimate.
            # The guard covers a constant integrand, where ``integral_mmn`` is zero and
            # the ratio would be 0/0.
            abserr = jnp.where(
                (integral_mmn != 0.0) & (abserr != 0.0),
                integral_mmn * jnp.minimum(1.0, (200.0 * abserr / integral_mmn) ** 1.5),
                abserr,
            )

            # No error estimate can be meaningful below the noise of the evaluation
            # itself. This floor is not a count of summed terms (XLA's pairwise
            # reduction holds summation error near ``eps`` whatever the rule size) but
            # covers the conditioning of the integrand: nodes carry ``~eps*|x|``, which
            # the integrand amplifies by ``|f'|``, so the achievable accuracy degrades
            # as the integrand varies faster. 50 is a compromise across that, generous
            # for smooth integrands and mildly optimistic for strongly oscillatory ones.
            # The ``uflow`` guard keeps the product from underflowing to zero.
            abserr = jnp.where(
                (integral_abs > uflow / (50.0 * eps)),
                jnp.maximum((eps * 50.0) * integral_abs, abserr),
                abserr,
            )
            return result, self.norm(abserr), integral_abs, integral_mmn

        return jax.lax.cond(a == b, truefun, falsefun)

    @eqx.filter_jit
    def _apply(
        self,
        fun: Callable[..., jax.Array],
        a: float,
        b: float,
        args: tuple[Any, ...],
    ) -> jax.Array:
        """Integrate a function from a to b, without an error estimate.

        Only the high order rule is summed, skipping the low order rule and the two
        auxiliary sums that ``integrate`` needs for its error estimate.
        """
        vfun = wrap_func(fun, args)
        halflength = (b - a) / 2
        center = (b + a) / 2
        f: jax.Array = vfun(center + halflength * self._xh)
        return _dot(self._wh, f) * halflength

    def norm(self, x: jax.Array) -> jax.Array:
        """Norm to use for measuring error for vector valued integrands."""
        if callable(self._norm):
            return self._norm(x)
        return jnp.linalg.norm(x.flatten(), ord=self._norm)


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

    def __init__(self, order: int = 21, norm: Callable | float | int = jnp.inf):
        self._norm = norm

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

    Notes
    -----
    On integrands with an endpoint singularity the error estimate can under-state the
    true error, increasingly so at higher order, because the endpoint-clustered nodes
    make the two rules agree while neither has converged. An adaptive integration may
    then report success while missing the requested tolerance.
    """

    def __init__(self, order: int = 32, norm: Callable | float | int = jnp.inf):
        self._norm = norm

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

    Notes
    -----
    Below about order 15 the embedded rule is too coarse for the error estimate to be
    trusted on any integrand with structure, including peaked and endpoint-singular ones
    that the other rules handle at much lower order; halving the points of a
    doubly-exponential rule costs far more accuracy than halving a polynomial one.
    """

    def __init__(self, order: int = 61, norm: Callable | float | int = jnp.inf):
        self._norm = norm

        _xts = lambda t: jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
        _wts = lambda t: (
            jnp.pi / 2 * jnp.cosh(t) / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2
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
