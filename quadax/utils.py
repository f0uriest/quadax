"""Utility functions for parsing inputs, mapping coordinates etc."""

import functools
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.internal import unvmap_any
from jax.typing import ArrayLike


def errorif(cond: bool | jax.Array, err: type[Exception] = ValueError, msg: str = ""):
    """Raise an error if condition is met.

    Similar to assert but allows wider range of Error types, rather than
    just AssertionError.
    """
    if cond:
        raise err(msg)


class DTypes(NamedTuple):
    """The working dtypes of a quadrature.

    quadax takes the dtype of ``interval`` as the statement of what precision the user
    wants, and derives everything else from it and from the integrand. An integrand that
    deliberately upcasts is respected: it is still *called* with an abscissa at the
    requested precision, but its own output dtype is carried through to the result.

    Parameters
    ----------
    xtype : dtype
        Abscissae. The sub-interval endpoints, the node tables, and the ``x`` the user's
        integrand is called with. Taken from ``interval``.
    ytype : dtype
        Integrand values, the per-interval contributions, and the returned integral. May
        be complex.
    etype : dtype
        Real. Weight tables, error estimates, tolerances. The real counterpart of
        ``ytype``, so that a complex integrand contracted with real weights promotes to
        complex on its own.
    toltype : dtype
        Real. Sets the default ``epsabs``/``epsrel`` of ``sqrt(eps)``. The coarser of
        ``xtype`` and ``etype``: a float32 abscissa limits the achievable accuracy
        however precisely the integrand itself is evaluated.

    """

    xtype: Any
    ytype: Any
    etype: Any
    toltype: Any


def _real_dtype(dtype) -> Any:
    """Real counterpart of ``dtype`` (float32 for complex64, and itself for real)."""
    return jnp.finfo(dtype).dtype


def tree_where(cond, new, old):
    """``jnp.where(cond, new, old)`` leafwise over two pytrees of the same structure."""
    return jax.tree_util.tree_map(lambda n, o: jnp.where(cond, n, o), new, old)


def _coarser_dtype(dtype1, dtype2) -> Any:
    """Whichever of the two has the larger machine epsilon."""
    if float(jnp.finfo(dtype1).eps) >= float(jnp.finfo(dtype2).eps):
        return dtype1
    return dtype2


def resolve_dtypes(
    interval: jax.Array, fun: Callable[..., jax.Array], args: tuple[Any, ...] = ()
) -> DTypes:
    """Work out the dtypes of a quadrature from its limits and its integrand.

    The single point at which quadax decides what precision it is working in. See
    :class:`DTypes` for what each one governs.
    """
    xtype = jnp.asarray(interval).dtype
    # `jnp.zeros((), xtype)` rather than `jnp.array(0.0)`: the latter is *weakly* typed,
    # which both hides the requested precision and lets different expressions involving
    # it settle on different dtypes. See the note on `MAPFUNS`.
    f = jax.eval_shape(fun, jnp.zeros((), xtype), *args)
    ytype = jnp.result_type(xtype, f.dtype)
    etype = _real_dtype(ytype)
    return DTypes(xtype, ytype, etype, _coarser_dtype(xtype, etype))


def _map_identity(t: jax.Array, a: jax.Array, b: jax.Array):
    """For finite intervals no mapping is needed.

    Mapping twice introduces extra roundoff error.
    """
    del a, b
    return t.squeeze(), jnp.ones_like(t).squeeze()


def _map_identity_inv(x: jax.Array, a: jax.Array, b: jax.Array):
    """Leave a point in [a, b] where it is."""
    del a, b
    return x.squeeze()


def _map_ninfinf(t: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point t in [-1, 1] to x in [-inf, inf]."""
    x = jnp.tan(t * jnp.pi / 2)
    w = jnp.pi / 2 / jnp.cos(jnp.pi * t / 2) ** 2
    return x.squeeze(), w.squeeze()


def _map_ninfinf_inv(x: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point x in [-inf, inf] to t in [-1, 1]."""
    t = jnp.arctan(x) / (jnp.pi / 2)
    return t.squeeze()


def _map_ainf(t: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point t in [-1, 1] to x in [a, inf]."""
    # The distance from the finite endpoint is (1+t)/(1-t), and writing it that way
    # keeps every bit `t` has: `1+t` is exact for `t` near -1. The algebraically equal
    # `a - 1 + 2/(1-t)` instead forms it as the difference of two numbers near 1 and
    # keeps only `d/eps` of it, so the nodes closest to the endpoint, ie the ones that
    # decide the answer when the integrand is singular there, come out wrong by a factor
    # of two at the last of them.
    x = a + (1 + t) / (1 - t)
    w = 2 / (1 - t) ** 2
    return x.squeeze(), w.squeeze()


def _map_ainf_inv(x: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point x in [a, inf] to t in [-1, 1]."""
    t = (a - x + 1) / (a - x - 1)
    return t.squeeze()


def _map_ninfb(t: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point t in [-1, 1] to x in [-inf, b]."""
    # Distance from the finite endpoint as (1-t)/(1+t) rather than 1 - 2/(t+1); see
    # ``_map_ainf``, which this mirrors.
    x = b - (1 - t) / (1 + t)
    w = 2 / (t + 1) ** 2
    return x.squeeze(), w.squeeze()


def _map_ninfb_inv(x: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point x in [-inf, b] to t in [-1, 1]."""
    t = (x - b + 1) / (b - x + 1)
    return t.squeeze()


# A finite interval stays where it is; the three infinite cases have to be brought into
# [-1, 1] because there is no other way to subdivide them.
MAPFUNS = [_map_identity, _map_ninfb, _map_ainf, _map_ninfinf]
MAPFUNS_INV = [_map_identity_inv, _map_ninfb_inv, _map_ainf_inv, _map_ninfinf_inv]

# These are the branches of a `lax.switch`, so all four have to return the same dtypes,
# and they only do so if `t`, `a` and `b` agree. Note in particular that they must not
# be given a *weakly* typed `t`: the four differ in which of `a`/`b` they use, and a
# weak `t` lets each branch settle on whichever of the two is present. `_map_ainf`'s
# `a + (1 + t) / (1 - t)` would follow a strong float32 `a`, while `_map_ninfinf`'s
# `jnp.tan(t * jnp.pi / 2)` would stay at the weak default, and the switch would not
# build.
# This is why the integrand is probed with `jnp.zeros((), xtype)`, not `jnp.array(0.0)`.


def map_interval(fun: Callable[..., jax.Array], interval: ArrayLike):
    """Map a function over an arbitrary interval [a, b] to one that can be subdivided.

    Transform a function such that integral(fun) on interval is the same as
    integral(fun_t) on interval_t

    Parameters
    ----------
    fun : callable
        Integrand to transform.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals.

    Returns
    -------
    fun_t : callable
        Transformed integrand.
    interval_t : float
        New lower and upper limits of integration with possible breakpoints.
    """
    interval = jnp.asarray(interval)
    errorif(
        not jnp.issubdtype(interval.dtype, jnp.floating),
        TypeError,
        "integration limits must be real floating point, got dtype "
        f"{interval.dtype}. Complex limits are not supported: the subdivision has to "
        "order the breakpoints, which complex numbers do not admit.",
    )
    a, b = interval[0], interval[-1]
    # An `xtype` scalar rather than the integer `(-1) ** (a > b)`, so that it cannot
    # participate in promotion downstream.
    sgn = jnp.where(a > b, -1, 1).astype(interval.dtype)
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    # catch breakpoints that are outside the domain, replace with endpoints
    # this creates intervals of 0 length which will be ignored later
    interval = jnp.where(interval < a, a, interval)
    interval = jnp.where(interval > b, b, interval)
    interval = jnp.sort(interval)

    # bit mask to select mapping case
    # 0 : both sides finite
    # 1 : a = -inf, b finite
    # 2 : a finite, b = inf
    # 3 : both infinite
    bitmask = jnp.isinf(a) + 2 * jnp.isinf(b)

    fun_mapped = _MappedFunction(fun, bitmask, sgn, a, b)
    # map original breakpoints to new domain
    # An infinite limit gets mapped to +/-1, which the inverse maps reach only as a
    # limit: the arithmetic there is inf/inf and evaluates to nan, so needs a double
    # where type trick to avoid nan in reverse mode.
    finite = jnp.where(jnp.isinf(a), jnp.where(jnp.isinf(b), 0.0, b), a)
    interval_finite = jnp.where(jnp.isinf(interval), finite, interval)
    interval_t: jax.Array = jax.lax.switch(bitmask, MAPFUNS_INV, interval_finite, a, b)
    interval_t = jnp.where(interval == jnp.inf, 1, interval_t)
    interval_t = jnp.where(interval == -jnp.inf, -1, interval_t)
    return fun_mapped, interval_t


class _MappedFunction(eqx.Module):
    """Function mapped to an interval a fixed rule can be applied over."""

    fun: Callable[..., jax.Array]
    bitmask: jax.Array
    sgn: jax.Array
    a: jax.Array
    b: jax.Array

    @eqx.filter_jit
    def __call__(self, t: jax.Array, *args):
        x, w = jax.lax.switch(self.bitmask, MAPFUNS, t, self.a, self.b)
        return self.sgn * w * self.fun(x, *args)


def tanhsinh_tmax(dtype, order: int | None = None) -> float:
    """Largest ``t`` whose tanh-sinh node is still distinct from the endpoint.

    The tanh-sinh nodes ``x = tanh(pi/2 sinh(t))`` cluster double-exponentially at the
    endpoints, which is what lets the rule handle an endpoint singularity. The cutoff is
    the last node that survives being written down: ``x = 1 - eps`` is two ulps below
    the endpoint, and one step further ``1 - d`` rounds to the endpoint itself, at which
    point an integrand singular there is evaluated at the singularity and returns a
    non-finite value rather than merely a useless one. That bound is set by the
    precision the nodes will be used at, which is why this takes a dtype rather than
    being a constant.

    This is a representability bound, not an accuracy one, but the two coincide: the
    truncation error of the trapezoidal rule in ``t`` falls monotonically as the range
    grows, so the best reachable cutoff is the largest one, and measured optima across
    dtypes and rule orders sit on this bound rather than inside it. A margin costs
    nothing at float64, where truncation is far below eps either way, and a great deal
    at half precision, where it is not.

    The bound assumes the reference interval. Composing with the map onto ``[a, b]``
    needs ``d > eps*|a + b| / |b - a|``, so a sub-interval far from the origin relative
    to its own width loses its outermost nodes to the same rounding regardless.

    Given ``order``, the range is additionally cut back until all nodes are unique
    in the given dtype. Reaching the endpoint is worth nothing if the last two nodes
    reach it together: the rule would spend two evaluations on one point and quietly
    lose an order. That constraint is the one place the range depends on how many nodes
    are being spread over it, and it only binds where the mantissa is short enough that
    the clustering is already marginal.

    Warns when the precision is coarse enough that the clustering has essentially been
    lost. Computed in float64 on the host; the result is a compile time constant.
    """
    eps = float(jnp.finfo(dtype).eps)
    closest = eps
    if closest > 1e-4:
        warnings.warn(
            f"tanh-sinh quadrature in {jnp.dtype(dtype).name} can place a node no "
            f"closer than {closest:.1e} of the half width from an endpoint (float64 "
            "reaches 2.2e-16), so the double exponential clustering that makes the "
            "method good at endpoint singularities is largely gone. Results are still "
            "valid but no better than a plain trapezoidal rule near the endpoints; use "
            "float32 or better, or use quadgk/quadcc instead.",
            UserWarning,
            stacklevel=2,
        )
    tanhinv = lambda x: 0.5 * np.log((1 + x) / (1 - x))
    sinhinv = lambda x: np.log(x + np.sqrt(x**2 + 1))
    tmax = float(sinhinv(2 / np.pi * tanhinv(1.0 - closest)))
    if order is None or order < 2:
        return tmax

    def nodes_resolve(t):
        """Whether an ``order`` point rule's nodes are all distinct, inside (-1, 1)."""
        nodes = np.tanh(np.pi / 2 * np.sinh(np.linspace(-t, t, order)))
        # `jnp.dtype` resolves bfloat16 to the ml_dtypes scalar numpy understands, so
        # the round trip stays on the host: this runs while a trace may be open, and a
        # jnp cast would be staged into it rather than evaluated.
        cast = nodes.astype(jnp.dtype(dtype)).astype(np.float64)
        return bool(len(np.unique(cast)) == order and np.max(np.abs(cast)) < 1.0)

    # Shrink until they do. float32 and above pass on the first test at every order, so
    # this costs nothing where the clustering is healthy. It binds only on the short
    # mantissas, and more as the order rises and the nodes crowd: bfloat16 gives up 2%
    # of the range at order 61 and 18% at order 121.
    while tmax > 0.5 and not nodes_resolve(tmax):
        tmax *= 0.98
    return tmax


def tanhsinh_complement(t: jax.Array) -> jax.Array:
    """Distance from the nearer end of [-1, 1] to the tanh-sinh node at ``t``.

    That is ``1 - |tanh(pi/2 sinh t)|``, formed as a reciprocal rather than as a
    subtraction so that the distance keeps its own exponent instead of being rounded
    against the one it is measured from. Distances below an eps are the whole point of a
    doubly exponential rule, and subtracting loses every one of them.
    """
    z = jnp.pi / 2 * jnp.sinh(jnp.abs(t))
    # 1 - tanh(z) = 1/(exp(z) cosh(z)). Splitting the denominator across two factors
    # rather than writing it as (exp(2z) + 1)/2 keeps each of them in range until the
    # product itself leaves it, and that happens by overflowing to +inf, so the distance
    # flushes to zero rather than to a nan.
    return 1 / (jnp.exp(z) * jnp.cosh(z))


def _saturated(w: jax.Array, inside: jax.Array):
    """Drop the weight of a node whose offset from the endpoint has rounded away.

    The range runs out to where the *offset* stops being representable, which is far
    past where the position does. Beyond that the abscissa sits on the endpoint however
    much further the offset shrinks, so its weight is one computed for a place it is not
    and cannot stand in for the mass out there. Dropping it says so: the estimate then
    reads the outermost node that is where it claims to be, instead of a node that has
    stopped moving while its weight went on decaying. The mass it gives up is bounded by
    an eps of the endpoint whatever the integrand does, since that is how far in the
    abscissa can still resolve.
    """
    return jnp.where(inside, w, 0)


def _ts_finite(t: jax.Array, c: jax.Array, a: jax.Array, b: jax.Array):
    """Tanh-sinh node in a finite [a, b], placed as an offset from the near endpoint."""
    alpha = (b - a) / 2
    x = jnp.where(t > 0, b - alpha * c, a + alpha * c)
    w = alpha * jnp.pi / 2 * jnp.cosh(t) * c * (2 - c)
    return x.squeeze(), _saturated(w, (x > a) & (x < b)).squeeze()


def _ts_ainf(t: jax.Array, c: jax.Array, a: jax.Array, b: jax.Array):
    """Tanh-sinh node in [a, inf], through ``x = a + (1 + r)/(1 - r)``."""
    del b
    # The composed Jacobian is ``1 - r**2`` from the substitution times ``2/(1 - r)**2``
    # from the map, which together are twice the offset. Evaluating the two factors
    # separately would overflow at nodes whose offset is still comfortably finite, since
    # the offset is the ratio of the two rather than either one of them.
    #
    # Which of the two distances goes on top is selected rather than derived, because
    # recovering one from the other costs exactly what carrying them was meant to save:
    # ``2 - c`` rounds to 2 for every ``c`` below an eps, and subtracting it back off
    # returns zero instead of the distance.
    offset = jnp.where(t > 0, 2 - c, c) / jnp.where(t > 0, c, 2 - c)
    x = a + offset
    w = _saturated(jnp.pi * jnp.cosh(t) * offset, x > a)
    return x.squeeze(), w.squeeze()


def _ts_ninfb(t: jax.Array, c: jax.Array, a: jax.Array, b: jax.Array):
    """Tanh-sinh node in [-inf, b], through ``x = b - (1 - r)/(1 + r)``."""
    del a
    offset = jnp.where(t > 0, c, 2 - c) / jnp.where(t > 0, 2 - c, c)
    x = b - offset
    w = _saturated(jnp.pi * jnp.cosh(t) * offset, x < b)
    return x.squeeze(), w.squeeze()


def _ts_ninfinf(t: jax.Array, c: jax.Array, a: jax.Array, b: jax.Array):
    """Tanh-sinh node in [-inf, inf], through ``x = tan(pi r / 2)``."""
    del a, b
    # Both sines are taken of the small angle rather than of one near pi/2: that puts
    # the node at t = 0 exactly on the origin, and lets the outermost ones reach the
    # largest representable abscissa instead of stopping where a tangent near its pole
    # loses its argument. The ``1 - r**2`` of the substitution is split across the two
    # so that neither the squared sine nor the secant is ever formed alone; either one
    # leaves the range while their combination is still well inside it.
    half = jnp.pi / 2 * c
    x = jnp.sign(t) * jnp.sin(jnp.pi / 2 * (1 - c)) / jnp.sin(half)
    w = (
        jnp.pi
        / 2
        * jnp.cosh(t)
        * (c / jnp.sin(half))
        * (jnp.pi / 2 * (2 - c) / jnp.sin(half))
    )
    return x.squeeze(), w.squeeze()


# The four cases of ``MAPFUNS``, each composed with the tanh-sinh substitution and
# rewritten in terms of the node's distance from the end of [-1, 1] it clusters against.
# Composing them rather than applying one after the other is what keeps the outermost
# nodes: the substitution reaches far closer to an endpoint than a node written down as
# a position can record, and the two Jacobians have factors that cancel and would
# otherwise leave the range on their own. Being branches of a ``switch``, these carry
# the same dtype requirement as ``MAPFUNS``.
TS_MAPFUNS = [_ts_finite, _ts_ninfb, _ts_ainf, _ts_ninfinf]


def tanhsinh_tmax_complement(dtype) -> float:
    """Largest ``t`` at which a tanh-sinh node and its weight are both representable.

    The counterpart of :func:`tanhsinh_tmax` for nodes carried as a distance from the
    endpoint rather than as a position. A distance is bounded below by the smallest
    normal instead of by one eps, most of the exponent range further out, so what
    actually sets the cutoff is the largest weight: on an unbounded interval the weight
    grows like the reciprocal of the distance, and so overflows before the distance
    underflows. The bound below is the weight one, with the representability of the
    distance as a floor under it.

    Both move only logarithmically in ``t``, the clustering being doubly exponential, so
    a margin costs almost nothing and the iteration settles in a step or two.

    Warns when the precision is coarse enough that the clustering only survives against
    an endpoint of zero. Computed in float64 on the host; the result is a compile time
    constant.
    """
    eps = float(jnp.finfo(dtype).eps)
    if eps > 1e-4:
        warnings.warn(
            f"tanh-sinh quadrature in {jnp.dtype(dtype).name} places its nodes as "
            "offsets from the endpoints, so the double exponential clustering that "
            "makes the method good at endpoint singularities survives only where the "
            f"endpoint is zero. Anywhere else the offset is lost below {eps:.1e} of "
            "the endpoint's own magnitude (float64 reaches 2.2e-16), leaving the rule "
            "no better than a plain trapezoidal one there. Results are still valid; "
            "use float32 or better, or use quadgk/quadcc instead.",
            UserWarning,
            stacklevel=2,
        )
    tiny = float(jnp.finfo(dtype).tiny)
    huge = float(jnp.finfo(dtype).max)
    # Inverting c = 1/(exp(z) cosh(z)) with z = pi/2 sinh(t), ie exp(2z) = 2/c - 1.
    reach = lambda c: float(np.arcsinh(np.log(2.0 / c - 1.0) / np.pi))
    # The largest weight any of the maps gives a node at distance ``c`` is
    # ``2 pi cosh(t) / c``, from the two semi-infinite ones. Solving for the ``c`` that
    # keeps it finite needs the ``t`` it is reached at, hence the iteration; taking the
    # running minimum is safe whether or not it has settled, since the iterates
    # alternate around the fixed point rather than approaching it from one side. The
    # margin leaves the outermost weight an order of magnitude short of overflowing,
    # rather than exactly on it, for a cost in ``t`` of well under a percent.
    limit = huge / 16
    tmax = reach(tiny)
    for _ in range(3):
        tmax = min(tmax, reach(max(tiny, 2 * np.pi * np.cosh(tmax) / limit)))
    return tmax


def tanhsinh_transform(fun, interval):
    """Transform a function by mapping with tanh-sinh.

    Transform a function such that integral(fun) on interval is the same as
    integral(fun_t) on interval_t

    The substitution is composed with the map onto ``interval`` rather than applied
    after it, so that every node is built as an offset from the endpoint it clusters
    against. Written down as a position instead, a node could get no closer to an
    endpoint than one eps of that endpoint's own magnitude, and on an integrand singular
    there that distance is the accuracy floor.

    Parameters
    ----------
    fun : callable
        Integrand to transform.
    interval : array-like
        Lower and upper limits of integration. Use np.inf to denote infinite intervals.

    Returns
    -------
    fun_t : callable
        Transformed integrand.
    interval_t : float
        New lower and upper limits.
    """
    errorif(
        len(interval) != 2,
        NotImplementedError,
        "tanh-sinh transformation with breakpoints not supported",
    )
    interval = jnp.asarray(interval)
    errorif(
        not jnp.issubdtype(interval.dtype, jnp.floating),
        TypeError,
        "integration limits must be real floating point, got dtype "
        f"{interval.dtype}. Complex limits are not supported: the substitution has to "
        "know which endpoint each node is approaching, which complex numbers do not "
        "admit.",
    )
    xtype = interval.dtype
    a, b = interval[0], interval[-1]
    # An `xtype` scalar rather than the integer `(-1) ** (a > b)`, so that it cannot
    # participate in promotion downstream.
    sgn = jnp.where(a > b, -1, 1).astype(xtype)
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    # bit mask to select mapping case, as in `map_interval`
    bitmask = jnp.isinf(a) + 2 * jnp.isinf(b)
    # The substitution lands in [-1, 1] whatever the original limits were, so how far
    # out the range runs is a question about the arithmetic and not about `interval`.
    tmax = tanhsinh_tmax_complement(xtype)
    interval_t = jnp.array([-tmax, tmax], dtype=xtype)
    return _TanhSinhTransformedFunction(fun, bitmask, sgn, a, b), interval_t


class _TanhSinhTransformedFunction(eqx.Module):
    """Function under the tanh-sinh substitution composed with the interval map."""

    fun: Callable[..., jax.Array]
    bitmask: jax.Array
    sgn: jax.Array
    a: jax.Array
    b: jax.Array

    @eqx.filter_jit
    def __call__(self, t, *args):
        c = tanhsinh_complement(t)
        x, w = jax.lax.switch(self.bitmask, TS_MAPFUNS, t, c, self.a, self.b)
        return self.sgn * w * self.fun(x, *args)


messages = {
    # NORMAL_EXIT
    0: "Algorithm terminated normally, desired tolerances assumed reached",
    # MAX_NINTER
    1: (
        "Maximum number of subdivisions allowed has been achieved. One can allow more "
        + "subdivisions by increasing the value of max_ninter. However,if this yields "
        + "no improvement it is advised to analyze the integrand in order to determine "
        + "the integration difficulties. If the position of a local difficulty can be "
        + "determined (e.g. singularity, discontinuity within the interval) one will "
        + "probably gain from splitting up the interval at this point and calling the "
        + "integrator on the sub-ranges. If possible, an appropriate special-purpose "
        + "integrator should be used, which is designed for handling the type of "
        + "difficulty involved."
    ),
    # ROUNDOFF
    2: (
        "The occurrence of roundoff error is detected, which prevents the requested "
        + "tolerance from being achieved. The error may be under-estimated."
    ),
    # BAD_INTEGRAND
    3: (
        "Extremely bad integrand behavior occurs at some points of the integration "
        + "interval."
    ),
    # NO_CONVERGE
    4: (
        "The algorithm does not converge. Roundoff error is detected in the "
        + "extrapolation table. It is assumed that the requested tolerance cannot be "
        + "achieved, and that the returned result is the best which can be obtained."
    ),
    # DIVERGENT
    5: "The integral is probably divergent, or slowly convergent.",
}


def _decode_status(status):
    if status == 0:
        msg = messages[0]
    else:
        status = f"{status:06b}"[::-1]
        msg = ""
        for s, m in zip(status, messages.values()):
            if int(s):
                msg += m + "\n\n"
    return msg


STATUS = {i: _decode_status(i) for i in range(int(2**6))}


def wrap_func(
    fun: Callable[..., jax.Array],
    args: tuple[Any, ...],
    xtype,
    batch_size: int | None = None,
    safe: bool = False,
):
    """Vectorize, jit, and mask out inf/nan.

    ``xtype`` is the dtype the integrand will be called at, and the integrand is probed
    at that dtype rather than at a weakly typed default. See the note on ``MAPFUNS``.

    ``batch_size`` bounds how many points the returned function evaluates at once. The
    default evaluates however many it is given.

    ``safe`` asks for a mask that can be differentiated in reverse, at the cost of a
    second evaluation of the integrand; see :class:`_WrappedFunction`. Only the
    evaluations that are actually differentiated need it, so it is off by default and
    the adjoints turn it on for the ones that are.
    """
    # Wrapping an already wrapped integrand again is not merely redundant, it defeats
    # the masking: the outer ``vectorize`` hands the inner wrapper one abscissa at a
    # time, leaving it unable to tell an abscissa the integrand blew up at from the rest
    # of the rule's, which is exactly what the AD-safe substitution needs. The local
    # rules re-wrap whatever integrand they are handed, so this is the common path and
    # not a corner case. The outer call's options win, since they are the ones the
    # caller asked for; ``safe`` is the exception, being a property of the integrand's
    # differentiability rather than a request about how to evaluate it.
    if isinstance(fun, _WrappedFunction) and not args:
        return _WrappedFunction(
            fun.fun, fun.args, fun.outsig, batch_size, safe or fun.safe
        )

    f = jax.eval_shape(fun, jnp.zeros((), xtype), *args)
    # need to make sure we get the correct shape for array valued integrands
    outsig = "(" + ",".join("n" + str(i) for i in range(len(f.shape))) + ")"

    return _WrappedFunction(fun, args, outsig, batch_size, safe)


def _bad_abscissae(bad, x):
    """Which abscissae the integrand cannot be linearized at.

    ``bad`` flags individual non-finite *values*, and carries the integrand's own axes
    after ``x``'s. Those are collapsed here because where to linearize is a choice per
    abscissa, not per component: every component of a vector valued integrand is
    evaluated at the same point and differentiated with respect to the same parameters,
    so one component blowing up is enough to poison the derivatives of the others
    through the parameters they share.
    """
    return jnp.any(jnp.reshape(bad, jnp.shape(x) + (-1,)), axis=-1)


class _WrappedFunction(eqx.Module):
    """Wraps a function in jit/vectorize and masks out inf/nans.

    Evaluates at most ``batch_size`` points at once, scanning over the batches. The
    number of points is fixed at trace time here, so the batches are cut to fit rather
    than the points being padded up to a whole number of batches: the leftovers are
    evaluated together in one smaller batch. That costs one extra tracing of the
    integrand and never an extra evaluation of it, which is the right way round when an
    evaluation is the expensive part, which is the case ``batch_size`` exists for.

    Callers whose point count is only known at run time cannot do this, and pad instead;
    see ``_level_sum`` in the Romberg solver.

    With ``safe`` set the mask is also correct in reverse mode, which costs a second
    evaluation of the integrand. See ``__call__``.
    """

    fun: Callable[..., jax.Array]
    args: tuple[Any, ...]
    outsig: str
    batch_size: int | None = None
    safe: bool = eqx.field(static=True, default=False)

    def _vectorize(self, x: jax.Array) -> jax.Array:
        return jnp.vectorize(
            self.fun,
            excluded=tuple(range(1, len(self.args) + 1)),
            signature="()->" + self.outsig,
        )(x, *self.args)

    def _evaluate(self, x: jax.Array) -> jax.Array:
        """The integrand at every point of ``x``, in batches, without any masking."""
        b = self.batch_size
        # A scalar abscissa has nothing to batch: Romberg calls the integrand one point
        # at a time, and every caller probes it with a scalar under eval_shape.
        if b is None or x.ndim == 0 or x.shape[0] <= b:
            return self._vectorize(x)
        n = x.shape[0]
        nfull = n // b
        full = jax.lax.map(self._vectorize, x[: nfull * b].reshape(nfull, b))
        parts = [full.reshape(-1, *full.shape[2:])]
        if n % b:
            parts.append(self._vectorize(x[nfull * b :]))
        return jnp.concatenate(parts)

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        if not self.safe:
            f: jax.Array = self._evaluate(x)
            return jnp.where(jnp.isfinite(f), f, 0.0)
        # Need to use a double-where type trick to avoid NaNs in reverse mode.
        # which means knowing where the integrand is finite before differentiating
        # anything, hence the probe, under `stop_gradient` so that this pass is never
        # itself linearized.
        probe: jax.Array = jax.lax.stop_gradient(self._evaluate(x))
        bad = ~jnp.isfinite(probe)
        bad_x = _bad_abscissae(bad, x)
        # Linearize at an abscissa the integrand was just seen to be finite at. Taking
        # one from the same set rather than some fixed point is what keeps this from
        # walking into the singularity: any fixed choice (the midpoint of the domain,
        # say) is itself the singularity for some integrand. Only finiteness matters,
        # since whatever it evaluates to there is masked back out.
        good = ~jnp.reshape(bad_x, (-1,))
        substitute = jnp.reshape(x, (-1,))[jnp.argmax(good)]
        # With no finite abscissa there is nothing to borrow, and every value is masked
        # to zero regardless, so the second evaluation is skipped rather than made at a
        # substitute that is itself singular. Skipping it is the point: the derivative
        # of an evaluation that stays in the graph would be the NaN this whole path
        # exists to avoid. `unvmap_any` keeps the predicate a scalar under `vmap`, so a
        # batch is evaluated as soon as one of its elements has a finite abscissa.
        # `_evaluate` goes through a lambda because `lax.cond` hashes its branches, and
        # a bound method of this module is not hashable once its fields carry tracers.
        f = jax.lax.cond(
            unvmap_any(jnp.any(good)),
            lambda x_: self._evaluate(x_),
            lambda _: jnp.zeros(probe.shape, probe.dtype),
            jnp.where(bad_x, substitute, x),
        )
        # Where the integrand was finite this is the ordinary evaluation, unchanged. At
        # a bad abscissa the value comes from the probe, so a vector valued integrand
        # keeps the components that were finite there and only the non-finite ones are
        # zeroed, exactly as masking the output alone would have done. Their derivative
        # is what is given up: the probe is a constant, so those components contribute
        # nothing to the tangent. That trades one abscissa's contribution to the
        # derivative for a derivative that exists at all.
        mask = jnp.reshape(bad_x, jnp.shape(bad_x) + (1,) * (jnp.ndim(f) - jnp.ndim(x)))
        return jnp.where(bad, 0.0, jnp.where(mask, probe, f))


def check_size(size: int | None, name: str = "batch_size") -> None:
    """Raise if ``size`` is neither ``None`` nor a positive integer.

    Shared by the options that set how many evaluations are grouped together:
    ``batch_size`` on the quadrature routines and ``chunk_size`` on the adjoints.
    """
    errorif(
        size is not None and (not isinstance(size, (int, np.integer)) or size < 1),
        ValueError,
        f"{name} must be None or a positive integer, got {size}",
    )


class QuadratureInfo(NamedTuple):
    """Information about quadrature.

    Parameters
    ----------
    err : float
        Estimate of the error in the quadrature result.
    neval : int
        Number of evaluations of the integrand.
    status : int
        Flag indicating reason for termination. status of 0 means normal termination,
        any other value indicates a possible error. A human readable message can be
        obtained by ``print(quadax.STATUS[status])``
    info : dict or None
        Other information returned by the algorithm. See specific algorithm for
        details. Only present if ``full_output`` is True.
    """

    err: float | jax.Array
    neval: int | jax.Array
    status: int | jax.Array
    info: Any


def bounded_while_loop(condfun, bodyfun, init_val, bound):
    """While loop for bounded number of iterations, implemented using cond and scan.

    Implemented with ``scan`` rather than ``lax.while_loop`` so that it can be reverse
    mode differentiated.

    Each iteration is gated twice. The outer gate is
    ``unvmap_any(condfun(state))``, a scalar, so it stays a real branch under ``vmap``
    and the loop stops doing work once *every* batch element has converged. Without it
    the raw predicate is per-element, the branch degrades to a ``select``, and the body
    runs for all ``bound`` iterations however few elements still need it. The inner gate
    is the raw predicate, which unbatched is a second cheap branch and batched is the
    per-element select that leaves already-converged elements untouched. Results are
    unchanged either way.
    """
    # could do some fancy stuff with checkpointing here like in equinox but the loops
    # in quadax usually only do ~100 iterations max so probably not worth it.

    def scanfun(state, *args):
        keep = condfun(state)

        def stepfun(state):
            # Inner branch on the raw predicate. Unbatched this is a second real branch
            # and costs almost nothing; batched it becomes a select, which is the
            # per-element masking that keeps already-converged elements untouched.
            return jax.lax.cond(keep, bodyfun, lambda x: x, state)

        return jax.lax.cond(unvmap_any(keep), stepfun, lambda x: x, state), None

    return jax.lax.scan(scanfun, init_val, None, bound)[0]


def _pnorm(x: jax.Array, p: int | float | jax.Array) -> jax.Array:
    return jnp.linalg.norm(x.flatten(), ord=p)


def wrap_jit(*args, **kwargs):
    """Wrap a function with jit with optional extra args.

    This is a helper to ensure docstrings and type hints are correctly propagated
    to the wrapped function, bc vscode seems to have issues with regular jitted funcs.
    """

    def wrapper(fun):
        foo = jax.jit(fun, *args, **kwargs)
        foo = functools.wraps(fun)(foo)
        return foo

    return wrapper
