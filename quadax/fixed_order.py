"""Fixed order quadrature."""

import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from .quad_weights import (
    get_cc_table,
    get_fejer2_table,
    get_tanhsinh_table,
    gk_weights,
)
from .utils import _real_dtype, check_size, tanhsinh_tmax, wrap_func


def _dot(w, f):
    return jnp.sum(w * f.T, axis=-1).T


def _endpoint_mass(d0, d1, f0, f1):
    """Mass of an integrand between an endpoint and the nearest node to it.

    Models ``|f|`` as ``C * d**-p`` in the distance ``d`` from the endpoint, fits ``p``
    to the two nodes nearest it, and integrates that model over the gap those nodes
    leave, giving ``f0 * d0 / (1 - p)``.

    An algebraic endpoint singularity is exactly this model, so the fit is the truth
    there whatever ``p`` is, while the terms of the sum itself have not necessarily
    started to fall by the time the nodes run out. A singularity mild enough to be
    integrable always has an integrable model, and one that is not reports an unbounded
    gap, which is the honest answer for an integrand with no finite integral.

    On a singularity that is not a power of the distance the fitted exponent picks up
    the local log-log slope, which for the logarithmic case comes out on the
    conservative side.
    """
    ok = (f0 > 0) & (f1 > 0) & (d0 > 0) & (d1 > d0)
    # The arguments are substituted rather than the result masked afterwards: outside
    # ``ok`` they are zero or infinite, and the logarithm of either is a nan or carries
    # an infinite derivative, both of which ``where`` propagates back through the good
    # branch. The stand-ins are any pair giving an exponent safely below one.
    #
    # Taken as a difference of logarithms rather than the logarithm of a ratio, which
    # is the same quantity but does not overflow when the second node happens to fall
    # near a zero of the integrand and the ratio of the two is enormous.
    lf0, lf1 = jnp.log(jnp.where(ok, f0, 2.0)), jnp.log(jnp.where(ok, f1, 1.0))
    ld0, ld1 = jnp.log(jnp.where(ok, d0, 1.0)), jnp.log(jnp.where(ok, d1, 8.0))
    exponent = (lf0 - lf1) / (ld1 - ld0)
    integrable = ok & (exponent < 1)
    safe = jnp.where(integrable, exponent, 0.0)
    return jnp.where(integrable, f0 * d0 / (1 - safe), jnp.where(ok, jnp.inf, 0.0))


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
    _batch_size: int | None

    @property
    def nodes_per_call(self) -> int:
        """How many evaluations of the integrand one application of the rule costs.

        Simply the number of nodes: ``batch_size`` changes how they are grouped, never
        how many there are. Note this is the node count and not ``order``, which are not
        always the same: an order ``n`` Clenshaw-Curtis rule has ``n + 1`` nodes.
        """
        return len(self._xh)

    def _nodes_weights(self, xtype) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Nodes and weights of the rule for use at abscissa dtype ``xtype``.

        The tables are stored at the highest precision available and cast at the point
        of use, so a float64 user loses nothing and a float32 user gets a table rounded
        once from float64 rather than one computed in float32. Only the nodes are cast
        here; the weights are cast to the accumulation dtype by the caller, which cannot
        know it until the integrand has been evaluated.

        Subclasses whose *table itself* depends on the precision rather than merely
        being rounded to it should override this. See ``TanhSinhRule``.
        """
        return self._xh.astype(xtype), self._wh, self._wl

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
        # The dtype of the limits is the statement of what precision was asked for: the
        # abscissae, and so the `x` the user's integrand sees, follow it.
        xtype = jnp.result_type(a, b)
        vfun = wrap_func(fun, args, xtype, self._batch_size)
        xh, wh_table, wl_table = self._nodes_weights(xtype)

        def falsefun():

            halflength = (b - a) / 2
            center = (b + a) / 2
            x = center + halflength * xh
            f: jax.Array = vfun(x)
            # An integrand that upcasts internally is respected, so the accumulation
            # follows both the limits and the integrand. The weights are cast to the
            # *real* counterpart, which lets a complex integrand promote on its own.
            etype = _real_dtype(jnp.result_type(xtype, f.dtype))
            wh = wh_table.astype(etype)
            wl = wl_table.astype(etype)
            result_kronrod = _dot(wh, f) * halflength
            result_gauss = _dot(wl, f) * halflength

            # Both of these are sums over the reference interval [-1, 1] and so, like
            # the two results above, need the Jacobian of the map onto [a, b] to be an
            # estimate of an integral over [a, b]. QUADPACK scales all four by
            # ``dhlgth``; the error estimate below compares ``abserr`` against
            # ``integral_mmn``, so the two have to be on the same scale for the
            # ``200 ... **1.5`` interpolation to mean what it was tuned to mean.
            dhalflength = jnp.abs(halflength)
            integral_abs = _dot(wh, jnp.abs(f)) * dhalflength  # ~integral of abs(fun)
            integral_mmn = (
                _dot(wh, jnp.abs(f - result_kronrod / (b - a))) * dhalflength
            )  # ~ integral of abs(fun - mean(fun))

            result = result_kronrod

            # Compile time constants, taken as python floats rather than as arrays of
            # the working dtype: `uflow / (50 * eps)` evaluated *in* half precision is a
            # needless underflow risk, and as a weakly typed python float the threshold
            # promotes to whatever it is compared against anyway.
            uflow = float(jnp.finfo(etype).tiny)
            eps = float(jnp.finfo(etype).eps)

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

            # double where trick to avoid nans when ratio would be zero or inf
            # The scaling is only defined when both quantities are nonzero. The ratio
            # must also be substituted, not just masked afterwards: ``x ** 1.5`` has an
            # infinite second derivative at ``x == 0``, so differentiating the
            # unselected branch twice yields ``inf * 0 == nan``, which ``where``
            # then propagates.
            # ``abserr`` is exactly zero whenever the two rules agree to the last bit,
            # which a smooth enough integrand does reach.
            scalable = (integral_mmn != 0.0) & (abserr != 0.0)
            mmn_safe = jnp.where(scalable, integral_mmn, 1.0)
            ratio = jnp.where(scalable, abserr / mmn_safe, 1.0)

            # The saturation is applied inside the power rather than outside it:
            # forming ``200 * ratio`` first overflows in half precision for any
            # ``ratio > 328``, and the result is then discarded by the outer ``min``
            # regardless. The inner clamp is the identity whenever the outer one is, so
            # this is bit for bit ``min(1, (200*ratio)**1.5)`` wherever that expression
            # does not overflow.
            abserr = jnp.where(
                scalable,
                integral_mmn * jnp.minimum(200.0 * jnp.minimum(ratio, 1.0), 1.0) ** 1.5,
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
            # The nested pair only sees what the nodes span. A rule whose outermost
            # node stops short of the endpoint has a second, independent shortfall
            # that no comparison of the two rules can reveal, since both truncate at
            # the same place, so it is summed rather than max of the two.
            abserr = abserr + self._truncation(x, f, a, b)

            return result, self.norm(abserr), integral_abs, integral_mmn

        def truefun():
            # Zeros shaped and typed exactly like what the other branch produces.
            out = jax.eval_shape(falsefun)
            return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), out)

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
        xtype = jnp.result_type(a, b)
        vfun = wrap_func(fun, args, xtype, self._batch_size)
        xh, wh_table, _ = self._nodes_weights(xtype)
        halflength = (b - a) / 2
        center = (b + a) / 2
        f: jax.Array = vfun(center + halflength * xh)
        etype = _real_dtype(jnp.result_type(xtype, f.dtype))
        return _dot(wh_table.astype(etype), f) * halflength

    def _truncation(self, x: jax.Array, f: jax.Array, a: float, b: float) -> jax.Array:
        """Bound on the mass lying outside the span the rule's nodes cover.

        Internal: the error estimate adds this to the one it gets from the nested
        pair. Zero here, which is right for an interpolatory rule, whose weights
        integrate the whole of ``[a, b]`` however its nodes are placed within it.
        Subclasses that truncate an infinite sum instead override it; see
        ``TanhSinhRule``.

        Parameters
        ----------
        x : jax.Array
            Abscissae of the high order rule, ordered from ``a`` to ``b``.
        f : jax.Array
            Integrand values at those abscissae, leading axis over the nodes.
        a, b : float
            Lower and upper limits of integration.

        Returns
        -------
        trunc : jax.Array
            Bound on the omitted mass, shaped like one value of the integrand.

        """
        del x, a, b
        return jnp.zeros(f.shape[1:], _real_dtype(f.dtype))

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
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel. Default
        is all of the rule's nodes at once, which is fastest but makes peak memory scale
        with the order. Values above the number of nodes are clipped to it. A value that
        does not divide the number of nodes leaves a remainder, which is evaluated
        together in one smaller batch, so the integrand is traced twice but never
        evaluated at more points than the rule has nodes.
    """

    def __init__(
        self,
        order: int = 21,
        norm: Callable | float | int = jnp.inf,
        batch_size: int | None = None,
    ):
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
        check_size(batch_size)
        self._batch_size = (
            None if batch_size is None else min(batch_size, len(self._xh))
        )


class ClenshawCurtisRule(NestedRule):
    """Integrate a function from a to b using a fixed order Clenshaw-Curtis rule.

    Integration is performed using an order n rule with error estimated
    using an embedded n//2 order rule.

    Parameters
    ----------
    n : int
        Order of integration scheme. Must be a multiple of 4 with ``closed=True``, or
        any even order of at least 4 with ``closed=False``.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel. Default
        is all of the rule's nodes at once, which is fastest but makes peak memory scale
        with the order. Values above the number of nodes are clipped to it. A value that
        does not divide the number of nodes leaves a remainder, which is evaluated
        together in one smaller batch, so the integrand is traced twice but never
        evaluated at more points than the rule has nodes.
    closed : bool, optional
        Whether the interval endpoints are among the nodes. The default closed rule uses
        ``order + 1`` points and is exact to degree ``order``. The open (Fejer-2) rule
        uses the same node family with the endpoints dropped, giving ``order - 1``
        points exact to degree ``order - 1``, and never evaluates the integrand at ``a``
        or ``b``. Both nest 2:1 against an embedded ``order // 2`` rule.

    Notes
    -----
    On integrands with an endpoint singularity the error estimate can under-state the
    true error, increasingly so at higher order, because the endpoint-clustered nodes
    make the two rules agree while neither has converged. An adaptive integration may
    then report success while missing the requested tolerance. This applies to both
    variants; the clustering is a property of the node family, not of the endpoints.

    Which is cheaper depends on the integrand. The closed rule wins on smooth, peaked
    and oscillatory ones, by up to about a factor of two in evaluations; the open rule
    wins on endpoint singularities and by a wide margin on infinite intervals whose
    integrand decays algebraically.
    """

    def __init__(
        self,
        order: int = 32,
        norm: Callable | float | int = jnp.inf,
        batch_size: int | None = None,
        closed: bool = True,
    ):
        self._norm = norm
        xh, wh, wl = (get_cc_table if closed else get_fejer2_table)(order)
        self._xh, self._wh, self._wl = jnp.asarray(xh), jnp.asarray(wh), jnp.asarray(wl)
        check_size(batch_size)
        self._batch_size = (
            None if batch_size is None else min(batch_size, len(self._xh))
        )


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
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel. Default
        is all of the rule's nodes at once, which is fastest but makes peak memory scale
        with the order. Values above the number of nodes are clipped to it. A value that
        does not divide the number of nodes leaves a remainder, which is evaluated
        together in one smaller batch, so the integrand is traced twice but never
        evaluated at more points than the rule has nodes.

    Notes
    -----
    Below about order 15 the embedded rule is too coarse for the error estimate to be
    trusted on any integrand with structure, including peaked and endpoint-singular ones
    that the other rules handle at much lower order; halving the points of a
    doubly-exponential rule costs far more accuracy than halving a polynomial one.

    The nodes stop short of the endpoints, leaving a sliver of the interval unsampled at
    each end, and the reported error includes a bound on the mass out there. Comparing
    the two rules cannot supply it, both of them stopping at the same place. The term is
    at the level of roundoff on a bounded integrand and can be the whole of the error on
    one singular at an endpoint.
    """

    _order: int

    def __init__(
        self,
        order: int = 61,
        norm: Callable | float | int = jnp.inf,
        batch_size: int | None = None,
    ):
        self._norm = norm
        self._order = order
        # The stored table is the one for the default dtype; `_nodes_weights` rebuilds
        # it whenever the quadrature actually runs at a different precision.
        xh, wh, wl = get_tanhsinh_table(
            self._order, tanhsinh_tmax(jnp.result_type(float), self._order)
        )
        self._xh, self._wh, self._wl = jnp.asarray(xh), jnp.asarray(wh), jnp.asarray(wl)
        check_size(batch_size)
        self._batch_size = (
            None if batch_size is None else min(batch_size, len(self._xh))
        )

    def _truncation(self, x: jax.Array, f: jax.Array, a: float, b: float) -> jax.Array:
        """Mass beyond the outermost node, charged at both ends.

        The nodes stop at the last one still distinct from the endpoint, so the rule
        never sees the sliver past it. That shortfall belongs to the map and not to
        the order: raising the order refines the mesh in ``t`` without extending its
        range, and the nested pair truncates in both rules at the same place, so the
        comparison between them says nothing about it. On a bounded integrand it is
        at the level of roundoff, and on one singular at an endpoint it can be the
        whole error.

        The pair fitted is taken from the outermost node the rule really used, which
        is not always the end of the table. On a sub-interval whose endpoints are
        large relative to its own width the outermost nodes round onto the endpoint
        itself, an integrand singular there returns a non-finite value, and the
        wrapper masks it away. Reading that as "no tail" is exactly backwards, it
        being the case where the tail is largest. Such a node covers no sliver of its
        own, so the fit steps over it to the first one still distinct from the
        endpoint.

        Only that case is stepped over, and not every value that came out zero: an
        integrand that simply vanishes over a stretch of the interval would otherwise
        send the search into the bulk, where the samples say nothing about the
        endpoint.

        The node the fit is taken against is the nearest one at a *different*
        abscissa, which on a narrow sub-interval is not the adjacent one. The
        clustering is doubly exponential while the representable points near an
        endpoint are only linearly spaced, so a sub-interval whose width approaches
        the endpoint's own ulp has many nodes sharing each point. A pair sitting at
        one distance carries no slope, and taking it would drop the whole tail rather
        than measure it.
        """
        mag = jnp.abs(f)
        gone = ~jnp.any(mag > 0, axis=tuple(range(1, mag.ndim)))
        n = x.shape[0]
        first = jnp.argmin(gone & (x == a))
        second = jnp.minimum(jnp.sum(x <= x[first]), n - 1)
        last = n - 1 - jnp.argmin((gone & (x == b))[::-1])
        penultimate = jnp.maximum(jnp.sum(x < x[last]) - 1, 0)
        return _endpoint_mass(
            jnp.abs(x[first] - a), jnp.abs(x[second] - a), mag[first], mag[second]
        ) + _endpoint_mass(
            jnp.abs(b - x[last]),
            jnp.abs(b - x[penultimate]),
            mag[last],
            mag[penultimate],
        )

    def _nodes_weights(self, xtype) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Rebuild the table at ``xtype`` rather than casting the stored one.

        The tanh-sinh nodes are cut off at the last one still distinct from the
        endpoint, so the extent of the table -- not just its rounding -- depends on the
        precision it will be used at. Casting a float64 table down to bfloat16 would
        collapse its outer nodes onto the endpoint and silently drop the effective
        order; rebuilding spreads the same ``order`` nodes over the range that dtype can
        actually resolve.
        """
        xh, wh, wl = get_tanhsinh_table(self._order, tanhsinh_tmax(xtype, self._order))
        return jnp.asarray(xh, xtype), jnp.asarray(wh), jnp.asarray(wl)
