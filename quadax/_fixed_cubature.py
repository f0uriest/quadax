"""Fixed order cubature over a box."""

import abc
from collections.abc import Callable, Sequence
from functools import reduce
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ._fixed_order import _ERR_INFLATION, NestedRule, _dot
from ._quad_weights import get_genz_malik_table
from ._utils import _ROUNDOFF_FLOOR, _real_dtype, check_size, errorif, wrap_func

# Coefficient on the extrapolated null rule value in the Genz-Malik error estimate,
# per degree.
#
# These are empirical fudge factors designed to keep the error estimate honest,
# fitted over eight integrand families, dimensions 2 to 8 and two tolerances, as the
# smallest value at which no run on a smooth integrand reports an error smaller than the
# one it made.
#
# The coefficient grows with the degree because the value is extrapolated one step down
# the decay of the levels before it is scaled, and a rule of higher degree resolves an
# integrand faster, so that step shrinks the estimate further.
_RATIO_SAFETY = {7: 100.0, 9: 1000.0, 11: 10000.0, 13: 10000.0}


def _dot_rows(w, f):
    """Contract the node axis of ``f`` against each row of a stack of weights.

    ``w`` holds one weight vector per row, and the result has that row axis in front of
    the integrand's own. Used for quantities a rule reports one of per dimension.
    """
    return jnp.einsum("kn,n...->k...", w, f)


def _outer(vs: Sequence[jax.Array]) -> jax.Array:
    """Flattened outer product of a weight vector per axis.

    Ordered so that the last axis varies fastest, matching a ``meshgrid`` built with
    ``indexing="ij"`` and flattened, which is how the nodes are laid out.
    """
    return reduce(lambda a, b: (a[:, None] * b[None, :]).ravel(), vs)


class AbstractCubatureRule(eqx.Module):
    """Abstract base class for cubature rules over an n-dimensional box.

    Subclasses should implement ``ndim`` and the ``integrate`` method for integrating a
    function over a fixed box using the given rule.

    Subclasses may also override the ``norm`` method for measuring error for vector
    valued integrands. Default is the infinity (max) norm.
    """

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """Dimension of the domain this rule integrates over."""

    @abc.abstractmethod
    def integrate(
        self,
        fun: Callable[..., jax.Array],
        a: jax.Array,
        b: jax.Array,
        args: tuple[Any, ...],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Integrate ``fun(x, *args)`` over the box with corners a and b.

        Parameters
        ----------
        fun : callable
            Function to integrate, should have a signature of the form
            ``fun(x, *args)`` -> float, Array, where ``x`` has shape ``(ndim,)``.
            Should be JAX transformable.
        a, b : Array, shape(ndim,)
            Opposite corners of the box to integrate over. Must be finite.
        args : tuple, optional
            Extra arguments passed to fun.

        Returns
        -------
        y : float, Array
            Estimate of the integral of fun over the box.
        err : float
            Estimate of the absolute error in y.
        y_abs : float, Array
            Estimate of the integral of abs(fun) over the box.
        y_mmn : float, Array
            Estimate of the integral of abs(fun - <fun>) over the box, where <fun>
            is the mean value of fun over the box.
        split : Array, shape(ndim,)
            How much each axis contributes to the difficulty of the integrand, for
            choosing which axis to cut when subdividing. Only the ordering of its
            entries is meaningful, and it is largest for the axis worth cutting.

        """

    def _apply(
        self,
        fun: Callable[..., jax.Array],
        a: jax.Array,
        b: jax.Array,
        args: tuple[Any, ...],
    ) -> jax.Array:
        """Integrate ``fun(x, *args)`` over the box, without an error estimate.

        Internal: used where many boxes are evaluated at once and only the values are
        wanted. Users writing a custom rule do not need to touch this; the default below
        is correct, and subclasses only override it when they can compute the value more
        cheaply than by discarding the error estimate from :meth:`integrate`.

        Parameters
        ----------
        fun : callable
            Function to integrate, should have a signature of the form
            ``fun(x, *args)`` -> float, Array, where ``x`` has shape ``(ndim,)``.
            Should be JAX transformable.
        a, b : Array, shape(ndim,)
            Opposite corners of the box to integrate over. Must be finite.
        args : tuple, optional
            Extra arguments passed to fun.

        Returns
        -------
        y : float, Array
            Estimate of the integral of fun over the box.

        """
        return self.integrate(fun, a, b, args)[0]

    def norm(self, x: jax.Array) -> jax.Array:
        """Norm to use for measuring error for vector valued integrands."""
        return jnp.linalg.norm(jnp.asarray(x).flatten(), ord=jnp.inf)

    def _with_norm(
        self, norm: float | int | Callable[[jax.Array], jax.Array]
    ) -> "AbstractCubatureRule":
        """A copy of this rule that measures vector valued error with ``norm``.

        Internal: used by the adjoints, which may run an error controlled solve whose
        vector is not the integrand's output, and so needs a norm of its own. Users
        writing a custom rule only need this if they want that to work; the default
        below is correct for any rule built from a norm, and a rule that measures error
        some other way should override it or let it raise.
        """
        errorif(
            not hasattr(self, "_norm"),
            NotImplementedError,
            f"{type(self).__name__} was not built from a norm, so it cannot be rebuilt "
            "with a different one.",
        )
        return eqx.tree_at(lambda rule: rule._norm, self, norm)

    def _drop_axis(self, axis: int) -> "AbstractCubatureRule":
        """A rule of the same family over every axis but ``axis``.

        Internal: used by :class:`~quadax.LeibnizAdjoint`, whose boundary term is an
        integral over a face of the box, which is one dimension lower than the box.
        Users writing a custom rule only need this if they want that adjoint to work
        with it; the alternative is to hand the adjoint a face rule directly, which is
        what its ``options_face`` is for.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot build a rule over one fewer axis, which is "
            "what integrating over a face of the box needs. Pass a rule of dimension "
            "ndim-1 as options_face={'rule': ...} on the adjoint instead."
        )


class NestedCubatureRule(AbstractCubatureRule):
    """Base class for nested cubature rules.

    Nested rules consist of a set of nodes (xh) and weights (wh) for a high degree rule,
    along with an additional set of weights (wl) for a lower degree rule that shares
    nodes with the high degree rule, exactly as in one dimension. Alongside them is a
    weight vector per axis (wsplit) combining the same nodes into the per axis
    ``split`` that :meth:`integrate` reports.

    Notes
    -----
    Writing the per axis measure as weights is what lets one implementation serve rules
    that arrive at it very differently. A tensor product rule takes the difference
    between its two rules along one axis while integrating the others normally, which
    is that axis' share of the error. A Genz-Malik rule takes a fourth difference along
    the axis, which measures the smoothness there rather than the error. Both are linear
    in the integrand, so both are a contraction against a row of weights, and only the
    table distinguishes them.

    The error estimate is derived from the difference between the two rules, so it
    inherits the caveat that carries in one dimension: it is only meaningful while the
    nodes resolve the integrand, and an integrand oscillating faster than they can
    follow makes both rules alias and agree spuriously.
    """

    _xh: jax.Array
    _wh: jax.Array
    _wl: jax.Array
    _wsplit: jax.Array
    _norm: float | int | Callable
    _batch_size: int | None
    _error_exponent: float = eqx.field(static=True)

    @property
    def ndim(self) -> int:
        """Dimension of the domain this rule integrates over."""
        return self._xh.shape[1]

    @property
    def nodes_per_call(self) -> int:
        """How many evaluations of the integrand one application of the rule costs.

        Simply the number of nodes: ``batch_size`` changes how they are grouped, never
        how many there are.
        """
        return self._xh.shape[0]

    def _nodes_weights(
        self, xtype
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Nodes and weights of the rule for use at abscissa dtype ``xtype``.

        The tables are stored at the highest precision available and cast at the point
        of use, so a float64 user loses nothing and a float32 user gets a table rounded
        once from float64 rather than one computed in float32. Only the nodes are cast
        here; the weights are cast to the accumulation dtype by the caller, which cannot
        know it until the integrand has been evaluated.

        Subclasses whose *table itself* depends on the precision rather than merely
        being rounded to it should override this. See ``TensorProductRule``.
        """
        return self._xh.astype(xtype), self._wh, self._wl, self._wsplit

    @eqx.filter_jit
    def integrate(
        self,
        fun: Callable[..., jax.Array],
        a: jax.Array,
        b: jax.Array,
        args: tuple[Any, ...],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Integrate a function over a box using a nested cubature rule.

        Parameters
        ----------
        fun : callable
            Function to integrate, should have a signature of the form
            ``fun(x, *args)`` -> float, Array, where ``x`` has shape ``(ndim,)``.
            Should be JAX transformable.
        a, b : Array, shape(ndim,)
            Opposite corners of the box to integrate over. Must be finite.
        args : tuple, optional
            Extra arguments passed to fun.

        Returns
        -------
        y : float, Array
            Estimate of the integral of fun over the box.
        err : float
            Estimate of the absolute error in y from the embedded lower degree rule.
        y_abs : float, Array
            Estimate of the integral of abs(fun) over the box.
        y_mmn : float, Array
            Estimate of the integral of abs(fun - <fun>) over the box, where <fun>
            is the mean value of fun over the box.
        split : Array, shape(ndim,)
            How much each axis contributes to the difficulty of the integrand.

        """
        a, b = jnp.asarray(a), jnp.asarray(b)
        # The dtype of the corners is the statement of what precision was asked for: the
        # abscissae, and so the `x` the user's integrand sees, follow it.
        xtype = jnp.result_type(a, b)
        vfun = wrap_func(fun, args, xtype, self._batch_size, ndim=self.ndim)
        xh, wh_table, wl_table, ws_table = self._nodes_weights(xtype)

        def falsefun():
            halfwidth = (b - a) / 2
            center = (b + a) / 2
            x = center + halfwidth * xh
            f: jax.Array = vfun(x)
            # An integrand that upcasts internally is respected, so the accumulation
            # follows both the corners and the integrand. The weights are cast to the
            # *real* counterpart, which lets a complex integrand promote on its own.
            etype = _real_dtype(jnp.result_type(xtype, f.dtype))
            wh = wh_table.astype(etype)
            wl = wl_table.astype(etype)
            ws = ws_table.astype(etype)

            # Jacobian of the map from the reference cube onto the box. Signed, so that
            # a box given with its corners the other way round integrates to the
            # negative of the same value, as reversed limits do in one dimension.
            volume = jnp.prod(halfwidth)
            result = _dot(wh, f) * volume

            # Both of these are sums over the reference cube and so, like the two
            # results above, need the Jacobian to be an estimate of an integral over
            # the box. The error estimate below compares ``abserr`` against
            # ``integral_mmn``, so the two have to be on the same scale for the
            # rescaling to mean what it was tuned to mean.
            dvolume = jnp.abs(volume)
            # Contracted against the *magnitudes* of the weights. A Genz-Malik rule has
            # substantially negative weights, a larger share of them the more dimensions
            # there are, so summing ``|f|`` against the signed weights is not a
            # quadrature of ``|f|`` at all and comes out negative for a strictly
            # positive integrand. Both quantities below are meant to be positive
            # measures of size and variation, and a negative one silently disables the
            # roundoff floor and the rescaling that use them. Taking magnitudes gives
            # the rule's absolute quadrature, which is what the roundoff floor wants
            # anyway, being the quantity that bounds the summation error. Every 1D rule
            # quadax implements has positive weights, and so does every tensor product
            # of them, so there the magnitudes are the weights themselves.
            wabs = jnp.abs(wh)
            integral_abs = _dot(wabs, jnp.abs(f)) * dvolume  # ~integral of abs(fun)
            integral_mmn = (
                _dot(wabs, jnp.abs(f - result / jnp.prod(b - a))) * dvolume
            )  # ~ integral of abs(fun - mean(fun))
            # One number per axis, each collapsed over the integrand's own axes the same
            # way the error is, so a vector valued integrand is charged its worst
            # component here too.
            split = jax.vmap(self.norm)(_dot_rows(ws, f)) * dvolume

            abserr = self._error_estimate(
                f, wl, result, integral_abs, integral_mmn, volume, etype
            )

            # Compile time constants, taken as python floats rather than as arrays
            # of the working dtype: `uflow / (50 * eps)` evaluated *in* half precision
            # is a needless underflow risk, and as a weakly typed python float the
            # threshold promotes to whatever it is compared against anyway.
            uflow = float(jnp.finfo(etype).tiny)
            eps = float(jnp.finfo(etype).eps)

            # No error estimate can be meaningful below the noise of the evaluation
            # itself. This floor is not a count of summed terms (XLA's pairwise
            # reduction holds summation error near ``eps`` whatever the rule size) but
            # covers the conditioning of the integrand: nodes carry ``~eps*|x|``, which
            # the integrand amplifies by its own variation, so the achievable accuracy
            # degrades as the integrand varies faster. The ``uflow`` guard keeps the
            # product from underflowing to zero.
            abserr = jnp.where(
                (integral_abs > uflow / (_ROUNDOFF_FLOOR * eps)),
                jnp.maximum((eps * _ROUNDOFF_FLOOR) * integral_abs, abserr),
                abserr,
            )

            return result, self.norm(abserr), integral_abs, integral_mmn, split

        def truefun():
            # Zeros shaped and typed exactly like what the other branch produces.
            out = jax.eval_shape(falsefun)
            return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), out)

        # A box with no extent in any one axis has no volume, and the rule's nodes all
        # collapse onto the face, so the integral is zero however the others are placed.
        return jax.lax.cond(jnp.any(a == b), truefun, falsefun)

    @eqx.filter_jit
    def _error_estimate(
        self,
        f: jax.Array,
        wl: jax.Array,
        result: jax.Array,
        integral_abs: jax.Array,
        integral_mmn: jax.Array,
        volume: jax.Array,
        etype,
    ) -> jax.Array:
        """Estimate the absolute error of ``result`` from the rule's node values.

        Returned before the roundoff floor and before the norm over a vector valued
        integrand's own axes, both of which :meth:`integrate` applies to the result.

        Parameters
        ----------
        f : Array, shape(npts, ...)
            The integrand at the rule's nodes.
        wl : Array, shape(npts,)
            Weights of the embedded low degree rule, at the accumulation dtype.
        result : Array
            The high degree rule's estimate of the integral.
        integral_abs, integral_mmn : Array
            Estimates of the integral of ``abs(fun)`` and of ``abs(fun - <fun>)``.
        volume : Array
            Jacobian of the map from the reference cube onto the box.
        etype : dtype
            The accumulation dtype.

        """
        # The difference between the two rules is dominated by the error of the
        # *low* degree one, so it says little about the error of ``result``, which
        # comes from the high degree rule and is typically far smaller. It is only a
        # starting point, and is rescaled below rather than reported directly.
        #
        # This difference is a null rule: it annihilates every polynomial the low
        # degree rule integrates exactly. For a Genz-Malik table it is the only one
        # of its degree, the rule carrying exactly enough orbits to be interpolatory
        # and one more to lift the degree, which leaves a one dimensional null
        # space. Comparing several lower degree null rules instead, to judge whether
        # the rule is in its asymptotic regime, is what :class:`GenzMalikRule` does
        # by overriding this; a rule that carries only the one null rule has nothing
        # to compare it against and falls back on the rescaling below.
        result_lo = _dot(wl, f) * volume
        abserr = jnp.abs(result - result_lo)

        # Measure that discrepancy against how much the integrand varies over the
        # box, the same rescaling QUADPACK applies in one dimension. With
        # ``r = abserr / integral_mmn`` the estimate becomes
        # ``integral_mmn * min(1, (200*r)**exponent)``, saturating at the whole
        # variation once the two rules disagree at that scale, inflating the raw
        # difference over most of the useful range, and deflating it only where the
        # two rules agree to near machine precision.
        #
        # The exponent is the ratio of the two rules' convergence rates. A degree
        # ``d`` rule on a region of diameter ``h`` in ``ndim`` dimensions has local
        # error ``~h**(d+1)`` over a volume ``~h**ndim``, so the rates go as
        # ``d + 1 + ndim`` and the ratio is that of the two degrees. In one
        # dimension this is the familiar ``(d_high+2)/(d_low+2)``; in more it tends
        # towards one, since the volume factor the two rules share comes to dominate
        # the difference between their degrees. See ``_error_exponent``.
        #
        # The 200 is QUADPACK's, fit empirically in one dimension, and is kept here
        # for want of a better measured value rather than because it transfers.

        # double where trick to avoid nans when ratio would be zero or inf
        # The scaling is only defined when both quantities are nonzero. The ratio
        # must also be substituted, not just masked afterwards: ``x ** p`` for
        # non-integer ``p`` has an infinite second derivative at ``x == 0``, so
        # differentiating the unselected branch twice yields ``inf * 0 == nan``,
        # which ``where`` then propagates.
        scalable = (integral_mmn != 0.0) & (abserr != 0.0)
        mmn_safe = jnp.where(scalable, integral_mmn, 1.0)
        ratio = jnp.where(scalable, abserr / mmn_safe, 1.0)

        # The saturation is applied inside the power rather than outside it:
        # forming ``200 * ratio`` first overflows in half precision for any
        # ``ratio > 328``, and the result is then discarded by the outer ``min``
        # regardless. The inner clamp is the identity whenever the outer one is.
        abserr = jnp.where(
            scalable,
            integral_mmn
            * jnp.minimum(_ERR_INFLATION * jnp.minimum(ratio, 1.0), 1.0)
            ** self._error_exponent,
            abserr,
        )

        return abserr

    def _apply(
        self,
        fun: Callable[..., jax.Array],
        a: jax.Array,
        b: jax.Array,
        args: tuple[Any, ...],
    ) -> jax.Array:
        """Integrate a function over a box, without an error estimate.

        Only the high degree rule is summed, skipping the low degree rule and the three
        auxiliary sums that ``integrate`` needs for its error estimate and split.
        """
        a, b = jnp.asarray(a), jnp.asarray(b)
        xtype = jnp.result_type(a, b)
        vfun = wrap_func(fun, args, xtype, self._batch_size, ndim=self.ndim)
        xh, wh_table, _, _ = self._nodes_weights(xtype)
        halfwidth = (b - a) / 2
        center = (b + a) / 2
        f: jax.Array = vfun(center + halfwidth * xh)
        etype = _real_dtype(jnp.result_type(xtype, f.dtype))
        return _dot(wh_table.astype(etype), f) * jnp.prod(halfwidth)

    def norm(self, x: jax.Array) -> jax.Array:
        """Norm to use for measuring error for vector valued integrands."""
        if callable(self._norm):
            return self._norm(x)
        return jnp.linalg.norm(jnp.asarray(x).flatten(), ord=self._norm)


class GenzMalikRule(NestedCubatureRule):
    """Integrate a function over a box using a fixed degree Genz-Malik rule.

    Integration is performed using a fully symmetric rule of the given polynomial
    degree, with error estimated from an embedded rule two degrees lower sharing its
    nodes [1]_ and from null rules on those nodes and one extra orbit [2]_.

    Parameters
    ----------
    ndim : int
        Dimension of the box to integrate over. One dimension is allowed, where the
        rule reduces to a ``degree`` point rule on the interval, but a 1D rule such as
        :class:`~quadax.GaussKronrodRule` reaches a far higher degree for the same
        nodes and is the better choice there.
    degree : int
        Polynomial degree of the rule, one of 7, 9, 11 or 13. Higher degrees are worth
        their extra nodes only on integrands smooth enough to be resolved by them.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    batch_size : int, optional
        Number of points to evaluate the integrand at simultaneously. The default
        evaluates all the rule's nodes at once.

    Notes
    -----
    The number of integrand evaluations per application is ``2**ndim`` from the corner
    orbit, plus a polynomial in ``ndim`` of degree ``(degree - 3) // 2`` from the rest,
    plus ``2*ndim`` for the orbit the error estimate adds:

    ======  =========  =========  =========  =========
    ndim    degree 7   degree 9   degree 11  degree 13
    ======  =========  =========  =========  =========
    1               9         11         13         15
    2              21         33         49         69
    3              39         77        143        245
    4              65        153        345        705
    5             103        273        723       1725
    6             161        453       1365       3729
    8             417       1105       3921      13345
    10           1265       2605       9405      37409
    ======  =========  =========  =========  =========

    The extra orbit is what lets the error be estimated over a space of null rules
    rather than from a single one, which is worth its ``2*ndim`` nodes several times
    over; the same counts appear in [2]_. It carries no integration weight, so the
    value the rule returns is the classical one.

    Raising the degree therefore costs more the higher the dimension, and no degree
    escapes the exponential corner term, which overtakes the polynomial part around
    ``ndim`` of 7 at degree 7 and progressively later at higher degrees. Both growths
    are in the rule itself, before an adaptive routine multiplies them by a number of
    regions that grows with ``ndim`` in its own right.

    The rule has negative weights, increasingly so as ``ndim`` grows, so it is not
    guaranteed to return a positive value for a positive integrand and can lose
    precision to cancellation in high dimensions. Degrees 9 and 11 carry rather less
    negative weight than degree 7 does, so their extra nodes buy conditioning as well
    as degree; degree 13 carries considerably more, and in low precision can give back
    what its degree gains.

    References
    ----------
    .. [1] A. C. Genz, A. A. Malik. "An imbedded family of fully symmetric numerical
           integration rules". SIAM Journal on Numerical Analysis, vol. 20, no. 3,
           1983, pp. 580-588.
    .. [2] J. Berntsen, T. O. Espelid, A. Genz. "An adaptive algorithm for the
           approximate calculation of multiple integrals". ACM Transactions on
           Mathematical Software, vol. 17, no. 4, 1991, pp. 437-451.

    """

    _nulls: jax.Array
    _null_sizes: tuple = eqx.field(static=True)
    _ratio_scale: float = eqx.field(static=True)
    _rho: jax.Array
    _l1: jax.Array
    _l1_top: float = eqx.field(static=True)
    _degree: int = eqx.field(static=True)

    def __init__(
        self,
        ndim: int,
        degree: int = 9,
        norm: Callable | float | int = jnp.inf,
        batch_size: int | None = None,
    ):
        self._norm = norm
        # The table carries one orbit the rule itself does not use, weighted zero in
        # both embedded rules, so that the error estimate has a space of null rules to
        # work over rather than a single one.
        table = get_genz_malik_table(ndim, degree)
        self._degree = degree
        self._xh = jnp.asarray(table.x)
        self._wh = jnp.asarray(table.wh)
        self._wl = jnp.asarray(table.wl)
        self._wsplit = jnp.asarray(table.wsplit)
        self._error_exponent = (degree + 1 + ndim) / (degree - 1 + ndim)
        self._nulls = jnp.asarray(table.nulls)
        self._null_sizes = table.null_sizes
        self._rho = jnp.asarray(table.rho)
        self._l1 = jnp.asarray(table.l1)
        self._l1_top = table.l1_top
        self._ratio_scale = _RATIO_SAFETY[degree] * 2.0**ndim
        check_size(batch_size)
        self._batch_size = (
            None if batch_size is None else min(batch_size, self._xh.shape[0])
        )

    def _drop_axis(self, axis: int) -> "GenzMalikRule":
        """The same family and degree over one fewer axis.

        The dimension enters the construction only as the length the generators are
        padded to and the number of parts a partition may have, so every dimension of
        the family is built from the same constants and there is nothing to project.
        ``axis`` goes unused because the rule is fully symmetric. The table is cached,
        so rebuilding costs nothing after the first time.
        """
        return GenzMalikRule(self.ndim - 1, self._degree, self._norm, self._batch_size)

    def _error_estimate(
        self,
        f: jax.Array,
        wl: jax.Array,
        result: jax.Array,
        integral_abs: jax.Array,
        integral_mmn: jax.Array,
        volume: jax.Array,
        etype,
    ) -> jax.Array:
        """Estimate the error by a ratio test over null rules of several degrees.

        The rule carries null rules of three descending degrees. Their values decay
        across degrees where the rule is resolving the integrand and do not where it is
        not, so the rate of that decay says whether the rule is in its asymptotic
        regime. Where it is, the rule's own error is one further step of the same decay,
        which is what this extrapolates; where it is not, there is no sequence to
        extrapolate and the estimate of :class:`NestedCubatureRule` stands.

        The value carried forward is the largest any null rule of the top level
        produces, each measured against the 1-norm of its own weights, rather than the
        value of one particular rule. A single null rule can be near blind to a given
        integrand and report far less error than the rule made, which is what forces a
        large safety coefficient; taking the largest over the level needs every
        direction to fail at once.

        Two independent tests have to agree before the extrapolation is used: the null
        rules must decay, and the two embedded rules must differ by little enough
        against the integrand's variation that the base rescaling is not saturated. The
        result is capped by the base estimate in any case, so this only ever talks an
        estimate down, never up.

        See :meth:`NestedCubatureRule._error_estimate` for the parameters.

        """
        base = super()._error_estimate(
            f, wl, result, integral_abs, integral_mmn, volume, etype
        )

        dvolume = jnp.abs(volume)
        values = _dot_rows(self._nulls.astype(etype), f) * dvolume

        # A level holds every null rule of one degree, and there is more than one of
        # them at every level but the top. Reduce each level to a single number by the
        # 2-norm over its rules, which is what the ratio test below compares. The rows
        # of a level are orthonormal, so this depends on the node values alone and not
        # on the basis chosen to span the level.
        levels = []
        start = 0
        for size in self._null_sizes:
            block = values[start : start + size]
            levels.append(jnp.sqrt(jnp.sum(jnp.abs(block) ** 2, axis=0)))
            start += size
        e1, e2, e3 = levels[0], levels[1], levels[2]

        # Equation (8) from [2]_. The levels descend in degree, so ``e1`` is built from
        # the null rules that annihilate the most and is the smallest of the three
        # exactly when the rule is resolving the integrand. Asking each level to be
        # smaller than the one below it is asking whether that decay is present.
        #
        # [2]_ demands the decay be by a factor of ``c1``, ``c2``, both 4 to 10 by its
        # Table IV; this asks only that the levels descend at all. The looser test is
        # paired with the second one below, and the two were fitted together rather
        # than each being justified on its own.
        #
        # The divisions are guarded on both sides so that an unselected branch cannot
        # produce a nan for the gradient to propagate.
        r12 = jnp.where(e2 > 0, e1 / jnp.where(e2 > 0, e2, 1.0), jnp.inf)
        r23 = jnp.where(e3 > 0, e2 / jnp.where(e3 > 0, e3, 1.0), jnp.inf)
        ratio = jnp.maximum(r12, r23)

        # The saturation condition of the base rescaling, as a second opinion on whether
        # the region is resolved, written without the division that forms it there. It
        # has no counterpart in [2]_, which gates on the decay alone. The two tests are
        # independent: one reads the null rules, the other the embedded pair.
        raw = jnp.abs(result - _dot(wl, f) * volume)
        unsaturated = _ERR_INFLATION * raw < integral_mmn
        asymptotic = (ratio < 1.0) & unsaturated

        # Equation (7) from [2]_, applied to the top level. Rather than report what one
        # null rule of that level says, report the largest value any rule in its span
        # says, each measured against the 1-norm of its own weights. A single null rule
        # can be near blind to a given integrand and report far less error than the rule
        # made; for the maximum over the span to be small, every direction has to be
        # blind at once which is far less likely.
        #
        # The maximum is a ratio of a function linear in the combination to a piecewise
        # linear convex one, so it is monotone between the turns of the latter and is
        # attained at one of them or in the limit of an infinite coefficient, which is
        # the second argument here. Those turns are fixed by the weights, so everything
        # but the combining is done when the table is built.
        #
        # Only the top level is treated this way. [2]_ maximises over three overlapping
        # pairs of levels, but only the top level is two dimensional by construction
        # here, and the lower two are read for the ratio test and nothing else.
        #
        # [2]_ instead normalises every null rule to a 1-norm of ``2**ndim`` up front,
        # its equation (6). The tables here are orthonormalized in the 2-norm, so the
        # 1-norm enters only in this division and the ``2**ndim`` is folded into
        # ``_ratio_scale``. Dividing by the 1-norm carries the scale of the node set
        # along with the estimate, which is what lets the coefficient below be one
        # number per degree rather than one per dimension as well.
        top, second = values[0], values[1]
        shape = (-1,) + (1,) * jnp.ndim(top)
        rho = self._rho.astype(etype).reshape(shape)
        largest = jnp.maximum(
            jnp.max(
                jnp.abs(rho * top[None] + second[None])
                / self._l1.astype(etype).reshape(shape),
                axis=0,
            ),
            jnp.abs(top) / self._l1_top,
        )

        # Equations (9) and (10), the two branches. [2]_ reports the top level as it
        # stands where the decay test passes and a flat multiple of the largest level
        # where it fails. Here the passing branch takes one more step down the measured
        # decay, multiplying by ``ratio``, before the per degree coefficient; and the
        # failing branch is the base rescaling, which also caps the passing one, so this
        # only ever talks an estimate down. See ``_RATIO_SAFETY`` for why extrapolating
        # is worth the larger coefficient it needs.
        #
        # That coefficient outgrows the whole range of the narrowest dtype quadax
        # supports, so the scaling and the cap are taken in at least single precision
        # and only the capped result, which is no larger than ``base``, is cast back.
        ratio_safe = jnp.where(asymptotic, ratio, 0.0)
        wide = jnp.promote_types(etype, jnp.float32)
        scaled = jnp.asarray(self._ratio_scale, wide) * (largest * ratio_safe).astype(
            wide
        )
        capped = jnp.minimum(scaled, base.astype(wide)).astype(etype)
        return jnp.where(asymptotic, capped, base)


class TensorProductRule(NestedCubatureRule):
    """Integrate a function over a box using a tensor product of 1D nested rules.

    The nodes are every combination of one axis' nodes with the others, and the weights
    the corresponding products, so an axis is integrated exactly as the one dimensional
    rule given for it would integrate it. Error is estimated from the embedded low order
    rule of every axis at once.

    Parameters
    ----------
    rules : NestedRule or sequence of NestedRule
        The one dimensional rule to apply along each axis. A single rule is used for
        every axis, and then ``ndim`` says how many there are. A sequence gives each
        axis its own rule, and ``ndim`` is taken from its length.
    ndim : int, optional
        Dimension of the box to integrate over. Required when ``rules`` is a single
        rule, and must not be given when it is a sequence.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    batch_size : int, optional
        Number of points to evaluate the integrand at simultaneously. The default
        evaluates all the rule's nodes at once.

    Notes
    -----
    The cost is the product of the axes' node counts, so it grows exponentially in
    ``ndim``: a 15 point Gauss-Kronrod rule costs 225 evaluations in two dimensions,
    3375 in three, and 50625 in four. A Genz-Malik rule of the highest degree costs 69,
    245 and 705, and of the default degree 33, 77 and 153. Tensor products earn their
    cost where the integrand is smooth enough for a high order rule to pay off along
    each axis, or where the axes differ enough to be worth giving different rules, and
    are a poor choice in high dimensions.

    Examples
    --------
    The same rule on every axis, or one per axis::

        TensorProductRule(GaussKronrodRule(21), ndim=3)
        TensorProductRule([GaussKronrodRule(21), ClenshawCurtisRule(32)])

    """

    _rules: tuple[NestedRule, ...]

    def __init__(
        self,
        rules: NestedRule | Sequence[NestedRule],
        ndim: int | None = None,
        norm: Callable | float | int = jnp.inf,
        batch_size: int | None = None,
    ):
        if isinstance(rules, NestedRule):
            errorif(
                ndim is None,
                ValueError,
                "ndim is required when rules is a single rule, since there is nothing "
                "else to say how many axes to apply it to. Either pass ndim, or pass "
                "one rule per axis as a sequence.",
            )
            errorif(
                not isinstance(ndim, int) or isinstance(ndim, bool) or ndim < 1,
                ValueError,
                f"ndim must be a positive integer, got ndim={ndim}.",
            )
            rules = (rules,) * ndim  # type: ignore[operator]
        else:
            rules = tuple(rules)
            errorif(
                ndim is not None,
                ValueError,
                f"ndim must not be given when rules is a sequence, since the sequence "
                f"already fixes it at {len(rules)}, but got ndim={ndim}.",
            )
            errorif(
                len(rules) < 1,
                ValueError,
                "A tensor product rule needs at least one axis, but the sequence of "
                "rules is empty.",
            )
        for i, rule in enumerate(rules):
            errorif(
                not isinstance(rule, NestedRule),
                TypeError,
                f"Every axis rule should be an instance of quadax.NestedRule, which "
                f"is what carries the embedded low order weights the error estimate "
                f"needs, but the rule for axis {i} is a {type(rule).__name__}.",
            )
        self._rules = rules
        self._norm = norm
        # The convergence rates of the two rules along one axis are what the tensor
        # product inherits, the other axes contributing the same factor to both, so this
        # is the one dimensional exponent. 1.5 is its large order Gauss-Kronrod limit
        # and a lower bound across the 1D rules quadax implements, hence the
        # conservative choice: a larger exponent would shrink the estimate. Taken as a
        # bound rather than computed because the axis rules do not carry their degree,
        # and a tanh-sinh axis has no polynomial degree to carry.
        self._error_exponent = 1.5
        self._xh, self._wh, self._wl, self._wsplit = self._build(jnp.result_type(float))
        check_size(batch_size)
        self._batch_size = (
            None if batch_size is None else min(batch_size, self._xh.shape[0])
        )

    def _drop_axis(self, axis: int) -> "TensorProductRule":
        """The product of the axis rules with the one for ``axis`` left out.

        The sequence of axis rules is the whole of what makes this rule ``ndim``
        dimensional, so which axis is dropped matters whenever they are not all the
        same rule.
        """
        errorif(
            len(self._rules) < 2,
            NotImplementedError,
            "A one dimensional tensor product rule has no axis left to integrate over "
            "once one is dropped.",
        )
        return TensorProductRule(
            self._rules[:axis] + self._rules[axis + 1 :],
            norm=self._norm,
            batch_size=self._batch_size,
        )

    def _build(self, xtype) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Assemble the product grid from the axes' own tables at dtype ``xtype``."""
        tables = [rule._nodes_weights(xtype) for rule in self._rules]
        xs = [t[0] for t in tables]
        whs = [t[1] for t in tables]
        wls = [t[2] for t in tables]
        ndim = len(self._rules)
        xh = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), axis=-1).reshape(-1, ndim)
        # Row k differences the two rules along axis k alone while the others are
        # integrated by the high order rule, which is that axis' share of the error.
        wsplit = jnp.stack(
            [
                _outer([whs[j] if j != k else whs[k] - wls[k] for j in range(ndim)])
                for k in range(ndim)
            ]
        )
        return xh, _outer(whs), _outer(wls), wsplit

    def _nodes_weights(
        self, xtype
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Nodes and weights of the product rule for use at abscissa dtype ``xtype``.

        Rebuilt rather than cast, because an axis rule may have a table that depends on
        the precision rather than merely being rounded to it, and asking it for its own
        table at ``xtype`` is what lets it say so. See ``TanhSinhRule``.
        """
        return self._build(xtype)
