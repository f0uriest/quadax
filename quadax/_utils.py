"""Utility functions for parsing inputs, mapping coordinates etc."""

import functools
import warnings
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.internal import unvmap_any
from jax.typing import ArrayLike

# Floor under any absolute error estimate, as a multiple of eps times the integral of
# |f| over the same domain. No estimate is meaningful below the noise of evaluating the
# integrand and summing it, so every rule and every driver in the package clamps its
# estimate here rather than reporting an accuracy the arithmetic cannot support.
#
# The multiplier is not a count of summed terms: XLA reduces pairwise, which holds the
# summation error near eps whatever the rule size. What it covers is the conditioning of
# the integrand. Abscissae carry ~eps*|x|, which the integrand amplifies by |f'|, so the
# achievable accuracy degrades as the integrand varies faster, and no fixed multiple of
# eps can be right for every integrand. 50 is QUADPACK's compromise across that:
# generous for smooth integrands, mildly optimistic for strongly oscillatory ones.
#
# Products with this are guarded against underflow where the integral of |f| can be
# denormal, since a floor that has underflowed to zero is a no-op precisely where the
# integrand is smallest.
_ROUNDOFF_FLOOR = 50.0


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
    interval: jax.Array | Sequence[jax.Array],
    fun: Callable[..., jax.Array],
    args: tuple[Any, ...] = (),
    ndim: int | None = None,
) -> DTypes:
    """Work out the dtypes of a quadrature from its limits and its integrand.

    The single point at which quadax decides what precision it is working in. See
    :class:`DTypes` for what each one governs.

    ``interval`` is either one array of limits or, for a box, one array per axis, in
    which case the working precision is the widest of them. ``ndim`` says what one
    abscissa looks like: ``None`` is a scalar, the 1D case, and an int is the length of
    the vector a cubature rule hands the integrand.
    """
    if isinstance(interval, (list, tuple)):
        xtype = jnp.result_type(*[jnp.asarray(axis) for axis in interval])
    else:
        xtype = jnp.asarray(interval).dtype
    # `jnp.zeros((), xtype)` rather than `jnp.array(0.0)`: the latter is *weakly* typed,
    # which both hides the requested precision and lets different expressions involving
    # it settle on different dtypes. See the note on `TS_MAPFUNS`.
    xprobe = jnp.zeros(() if ndim is None else (ndim,), xtype)
    f = jax.eval_shape(fun, xprobe, *args)
    ytype = jnp.result_type(xtype, f.dtype)
    etype = _real_dtype(ytype)
    return DTypes(xtype, ytype, etype, _coarser_dtype(xtype, etype))


## Coordinate mapping
# In all cases a finite (sub)interval stays where it is. An infinite interval
# without breakpoints is mapped to a reference interval of [-1, 1]. An infinite
# interval with finite breakpoints is done piecewise: The finite parts are left
# alone, while the infinite parts are mapped to a finite length and joined at
# ``junctions`` (the largest and smallest finite points in the interval). The size
# the infinite part is mapped to is determined by the largest finite sub-interval,
# or 1 when there is none, given by ``width``.


def _map_scale(interval: jax.Array) -> jax.Array:
    """The characteristic scale for the domain.

    The scale is taken to be the size of the largest finite sub-interval from the user
    supplied interval with possible breakpoints. If the domain is infinite without any
    breakpoints, we fall back to a scale of 1.
    """
    # interval is already sorted so these are non-negative
    gaps = jnp.diff(interval)
    # An infinite gap is the unbounded end and says nothing about scale; a nan one comes
    # from two infinite limits landing on top of each other after the clamp.
    gaps = jnp.where(jnp.isfinite(gaps), gaps, 0)
    widest = jnp.max(gaps, initial=jnp.zeros((), interval.dtype))
    return jnp.where(widest > 0, widest, 1).astype(interval.dtype)


def _map_width(scale: jax.Array, anchored: jax.Array) -> jax.Array:
    """How much reference coordinate one unbounded end is given.

    An unbounded sub-interval is mapped to a finite stretch of the reference
    coordinate, and this is how big that stretch is. It is set by asking that the
    halfway point of a tail's reference span sit one ``scale`` out along the real axis
    from the nearest finite breakpoint.
    """
    return jnp.where(anchored, scale, 3 * scale / 2)


def _map_tail(
    offset: jax.Array, rest: jax.Array, width: jax.Array, anchored: jax.Array
):
    """One unbounded end, as the physical distance from the point it hangs off.

    Takes the reference distance from the junction (``offset``) and the distance from
    the far end of the tail's span (``rest``), which sum to ``width`` in exact math.
    Both are passed rather than one derived from the other because each is exact only
    at its own end: near the far end the distance from the junction is a number of order
    ``width`` and cannot record how far the node still is from the boundary, which is
    precisely the distance that decides where the outermost node lands and how much of
    the tail the rule ever sees.

    The reference distance runs over ``[0, width]`` and the physical one over
    ``[0, inf)``. Two properties are what let this one function cover every case:

    - It divides by an exact zero at ``width``, so an unbounded limit comes back as a
      true infinity that the inf/nan mask and the rules' own boundary tests recognize.
    - Its derivative at the junction is that of the identity, so a tail joins the
      part the map leaves alone without a kink.

    ``anchored`` selects between two profiles, and says whether the junction this
    tail hangs off is a point of the interval or one fabricated for want of any finite
    point at all. Anchored, it is a Mobius function of the reference coordinate, whose
    only singularity is the pole at ``width`` carrying the unbounded end: an integrand
    decaying like a power of the abscissa then composes with it to give an exact power
    of the distance from the reference boundary times a factor analytic across the whole
    span, which is the form the rules and the extrapolation resolve best. Its second
    derivative at the junction is not that of the identity, so the join carries a break
    in curvature -- which costs nothing, an anchored junction being a point of the mesh
    that no sub-interval straddles.

    Unanchored, the junction is interior to a sub-interval and that break would sit
    where two tails meet, at the centre of the reference domain, where a symmetric rule
    mis-estimates it symmetrically and a nested pair agrees on a value neither has
    resolved. Squaring the denominator matches the second derivative as well, making the
    two tails one analytic map of the whole line.
    """
    grow = width / rest
    # `1 + offset/width` folded into the denominator turns the Mobius profile into the
    # odd rational one, `offset / (1 - s**2)`, without forming either separately.
    spread = jnp.where(anchored, 1, (width + offset) / width)
    rise = jnp.where(anchored, 1, 1 + (offset / width) ** 2)
    grow = grow / spread
    return offset * grow, rise * grow * grow


def _map_tail_inv(offset: jax.Array, width: jax.Array, anchored: jax.Array):
    """The reference distance from a junction, given the physical one."""
    # Over the ratio `width / offset` rather than as `width * offset` over their sum, so
    # that an infinite offset gives the width exactly instead of `inf / inf`.
    q = width / offset
    # The Mobius root of `offset = d / (1 - d/width)`, and the odd rational one of
    # `offset = d / (1 - (d/width)**2)`, the latter being the root of `q s**2 + s - q`
    # in [0, 1]. Both denominators are bounded away from zero for every offset the map
    # can produce, so this needs no branch and no masking.
    return width * jnp.where(anchored, 1 / (1 + q), 2 / (q + jnp.sqrt(q * q + 4)))


def _map(
    t: jax.Array,
    scale: jax.Array,
    shift: jax.Array,
    anchored: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
):
    """Map the reference coordinate to x, transforming only the unbounded ends.

    Between ``lo`` and ``hi``, the outermost finite points of the interval, this is the
    identity, and each unbounded end is a tail hanging off the finite point next to it.
    Only the part of the interval that actually needs bringing into a finite domain is
    transformed, so a sub-interval far from the origin, or one carrying a singularity,
    keeps its own coordinate instead of being compressed against a reference boundary
    along with everything else.
    """
    width = _map_width(scale, anchored)
    # The junctions as the reference coordinate sees them. A node's distance from one is
    # formed against these rather than by translating the node back first: the whole
    # point of placing the domain is that its ends are small, and an intermediate at the
    # junction's own magnitude would round the distance to an eps of that instead.
    jlo, jhi = lo + shift, hi + shift
    t_lo, t_hi = jlo - width, jhi + width
    # below, above are false for a finite interval, true only when a point t lies in
    # the part of the reference domain that maps to the semi-infinite x domain.
    below, above = t < jlo, t > jhi
    # A branch that is not selected is handed the junction itself, so that the pole a
    # tail carries at its far end is never reached by one about to be discarded: an
    # infinity in a discarded branch is still a nan in reverse mode.
    left, wleft = _map_tail(
        jnp.where(below, jlo - t, 0), jnp.where(below, t - t_lo, width), width, anchored
    )
    right, wright = _map_tail(
        jnp.where(above, t - jhi, 0), jnp.where(above, t_hi - t, width), width, anchored
    )
    x = jnp.where(below, lo - left, jnp.where(above, hi + right, t - shift))
    w = jnp.where(below, wleft, jnp.where(above, wright, jnp.ones_like(t)))
    return x.squeeze(), w.squeeze()


def _map_inv(
    x: jax.Array,
    scale: jax.Array,
    shift: jax.Array,
    anchored: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
):
    """Map x back to the reference coordinate, inverting :func:`_map`."""
    width = _map_width(scale, anchored)
    jlo, jhi = lo + shift, hi + shift
    below, above = x < lo, x > hi
    t = jnp.where(
        below,
        jlo - _map_tail_inv(lo - x, width, anchored),
        jnp.where(above, jhi + _map_tail_inv(x - hi, width, anchored), x + shift),
    )
    return t.squeeze()


def _map_junctions(
    interval: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The outermost finite points of an interval, which its tails hang off.

    An end that is already finite is its own junction and grows no tail. Limits that are
    both infinite leave nothing to anchor to and fall back to the origin, which is
    reported alongside them: a junction taken from the interval is a point of the mesh
    and a fabricated one is not, and that is what decides the profile of the tails
    hanging off it.
    """
    finite = jnp.isfinite(interval)
    inf = jnp.array(jnp.inf, interval.dtype)
    lo = jnp.min(jnp.where(finite, interval, inf))
    hi = jnp.max(jnp.where(finite, interval, -inf))
    anchored = jnp.any(finite)
    zero = jnp.zeros((), interval.dtype)
    return jnp.where(anchored, lo, zero), jnp.where(anchored, hi, zero), anchored


def _map_bounds(interval: jax.Array):
    """Everything :func:`_map` needs, and where the reference domain ends.

    Returns
    -------
    scale : jax.Array
        Characteristic size of the domain (largest finite sub-interval)
    shift : jax.Array
        How much the domain is shifted relative to the reference domain
    anchored : jax.Array
        Whether the original domain had any finite points to anchor the reference domain
    lo, hi : jax.Array
        Smallest and largest finite points of original domain.
    t_lo, t_hi : jax.Array
        Endpoints of mapped domain
    """
    scale = _map_scale(interval)
    lo, hi, anchored = _map_junctions(interval)
    width = _map_width(scale, anchored)
    # A tail is grown only where the limit on that side is unbounded; a finite one is
    # its own image, up to where the domain is placed.
    unbounded_lo = interval[0] == -jnp.inf
    unbounded_hi = interval[-1] == jnp.inf
    end_lo = jnp.where(unbounded_lo, lo - width, lo)
    end_hi = jnp.where(unbounded_hi, hi + width, hi)
    # shift is zero except in the case of a semi-infinite domain with no interior
    # breakpoints, in which case we shift the finite end to the origin. This avoids
    # some roundoff error in abscissae near the endpoint. We could shift all domains
    # to the origin, but we want to keep finite breakpoints in place since we need to
    # compare direct equality in places, and (x+y)-y may not equal x in floating point
    shift = jnp.where(lo == hi, -(end_lo + end_hi) / 2, jnp.zeros_like(lo))
    # Formed exactly as :func:`_map` forms them, so that the boundary it returns an
    # infinity at is the same float the mesh is built out of.
    jlo, jhi = lo + shift, hi + shift
    t_lo = jnp.where(unbounded_lo, jlo - width, jlo)
    t_hi = jnp.where(unbounded_hi, jhi + width, jhi)
    return scale, shift, anchored, lo, hi, t_lo, t_hi


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

    scale, shift, anchored, lo, hi, t_lo, t_hi = _map_bounds(interval)

    fun_mapped = _MappedFunction(
        fun, sgn, a, b, scale, shift, anchored, lo, hi, t_lo, t_hi
    )
    # Map the original breakpoints to the new domain. A finite one lies between the
    # junctions by construction and keeps its own coordinate up to where the domain has
    # been placed, and an infinite limit goes to the end of the reference domain, which
    # the map reaches only as a limit.
    interval_t = jnp.where(
        interval == jnp.inf,
        t_hi,
        jnp.where(interval == -jnp.inf, t_lo, interval + shift),
    )
    return fun_mapped, interval_t


class _MappedFunction(eqx.Module):
    """Function mapped to an interval a fixed rule can be applied over."""

    fun: Callable[..., jax.Array]
    sgn: jax.Array
    a: jax.Array
    b: jax.Array
    scale: jax.Array
    shift: jax.Array
    anchored: jax.Array
    lo: jax.Array
    hi: jax.Array
    t_lo: jax.Array
    t_hi: jax.Array

    @eqx.filter_jit
    def __call__(self, t: jax.Array, *args):
        x, w = _map(t, self.scale, self.shift, self.anchored, self.lo, self.hi)
        return self.sgn * w * self.fun(x, *args)


def _as_box_intervals(interval) -> tuple[jax.Array | Callable, ...]:
    """Normalize either accepted form of ``interval`` to one entry per axis.

    An array is the corners and nothing else, a non-array sequence is one interval per
    axis and may carry breakpoints. Dispatching on the type rather than the shape is
    what keeps the two unambiguous: an ``(ndim, 2)`` array and the sequence obtained by
    iterating over it mean the same thing, so the convenience form cannot silently
    denote a different box than the general one.

    An entry of the sequence form may instead be a callable giving that axis' limits as
    a function of the coordinates before it, and comes back as it was given: what it
    denotes is known only once it is called, so it is checked where it is probed, in
    :func:`_box_axis_knots`.
    """
    if isinstance(interval, (jax.Array, np.ndarray)):
        errorif(
            interval.ndim != 2 or interval.shape[1] != 2,
            ValueError,
            f"an array of limits must have shape (ndim, 2), giving the two limits of "
            f"each axis, but got shape {interval.shape}. Note this is the transpose of "
            "stacking the two corners: use jnp.stack([a, b], axis=-1). To pass "
            "breakpoints, give a list or tuple of one array per axis instead.",
        )
        intervals = tuple(interval[k] for k in range(interval.shape[0]))
    else:
        errorif(
            not isinstance(interval, (list, tuple)),
            TypeError,
            f"limits must be an array of shape (ndim, 2) or a list or tuple of one "
            f"array per axis, got {type(interval).__name__}.",
        )
        intervals = tuple(
            axis if callable(axis) else jnp.asarray(axis) for axis in interval
        )
    errorif(
        len(intervals) == 0,
        ValueError,
        "limits must cover at least one axis, but got an empty sequence.",
    )
    for k, axis in enumerate(intervals):
        if callable(axis):
            continue
        errorif(
            axis.ndim != 1 or axis.shape[0] < 2,
            ValueError,
            f"the limits of axis {k} must be a one dimensional array holding at least "
            f"its two endpoints, with any breakpoints between them, but got shape "
            f"{axis.shape}.",
        )
        errorif(
            jnp.issubdtype(axis.dtype, jnp.complexfloating),
            TypeError,
            f"integration limits must be real, but axis {k} has dtype {axis.dtype}. "
            "The subdivision has to order the breakpoints, which complex numbers do "
            "not admit.",
        )
    # Limits are differentiated with respect to, so an integer or boolean axis has to
    # become inexact here rather than being left for AD to treat as static metadata.
    return tuple(
        axis
        if callable(axis) or jnp.issubdtype(axis.dtype, jnp.inexact)
        else axis.astype(jnp.result_type(float))
        for axis in intervals
    )


def _box_axis_knots(intervals: Sequence, args: tuple = ()) -> tuple[tuple, Any]:
    """How many knots each axis carries, and the dtype they are carried at.

    A fixed axis states both outright. A variable one has to be probed for them, its
    limits being whatever its callable returns, which :func:`jax.eval_shape` reads off
    without evaluating anything. The probe needs a dtype to form the preceding
    coordinates at, so the fixed axes settle one first and the variable ones may only
    widen it afterwards.
    """
    fixed = [axis for axis in intervals if not callable(axis)]
    xtype = jnp.result_type(*fixed) if fixed else jnp.result_type(float)
    lengths, dtypes = [], []
    for k, axis in enumerate(intervals):
        if not callable(axis):
            lengths.append(axis.shape[0])
            continue
        # Probed through the same `asarray` the map applies, so that a callable
        # returning a list of limits is read here as it is read there.
        out = jax.eval_shape(
            lambda *a: jnp.asarray(axis(*a)), jnp.zeros((k,), xtype), *args
        )
        errorif(
            len(out.shape) != 1 or out.shape[0] < 2,
            ValueError,
            f"the limits of axis {k} must be a one dimensional array holding at least "
            f"its two endpoints, with any breakpoints between them, but the callable "
            f"given for it returns shape {out.shape}.",
        )
        errorif(
            jnp.issubdtype(out.dtype, jnp.complexfloating),
            TypeError,
            f"integration limits must be real, but the callable given for axis {k} "
            f"returns dtype {out.dtype}. The subdivision has to order the breakpoints, "
            "which complex numbers do not admit.",
        )
        lengths.append(out.shape[0])
        dtypes.append(out.dtype)
    return tuple(lengths), jnp.result_type(xtype, *dtypes) if dtypes else xtype


def _box_reference_intervals(
    intervals: Sequence, args: tuple = ()
) -> tuple[jax.Array, ...]:
    """Constant limits for every axis, standing in for the ones a callable gives.

    A variable axis is integrated over a fixed reference interval and mapped onto its
    real limits inside the integrand, where the coordinates those limits depend on are
    known. So what the mesh, the tolerances and the adjoints see for such an axis is
    this: ``[0, 1]``, with one interior knot for each breakpoint its callable returns. A
    fixed axis is its own reference.
    """
    lengths, xtype = _box_axis_knots(intervals, args)
    return tuple(
        jnp.linspace(0, 1, m, dtype=xtype) if callable(axis) else axis.astype(xtype)
        for axis, m in zip(intervals, lengths)
    )


def box_corners(interval: Sequence[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Opposite corners of a box given as one interval per axis.

    Parameters
    ----------
    interval : sequence of Array
        One interval per axis, as returned by :func:`~quadax._utils.map_box`.

    Returns
    -------
    a, b : Array, shape(ndim,)
        Opposite corners of the box, the first and last entry of each axis.
    """
    return (
        jnp.stack([axis[0] for axis in interval]),
        jnp.stack([axis[-1] for axis in interval]),
    )


def map_box(fun: Callable[..., jax.Array], interval, args: tuple = ()):
    """Map a function over an arbitrary box to one that can be subdivided.

    Transform a function such that the integral of ``fun`` over the region given by
    ``interval`` is the same as the integral of ``fun_t`` over ``interval_t``, which is
    a box, finite along every axis, whatever ``interval`` was.

    Parameters
    ----------
    fun : callable
        Integrand to transform, with signature ``fun(x, *args)`` where ``x`` has shape
        ``(ndim,)``.
    interval : Array, shape(ndim, 2), or sequence of array-like or callable
        Limits of integration. An array gives the two limits of each axis and no
        breakpoints. A list or tuple gives one entry per axis, each holding that axis'
        limits with any breakpoints between them, so the axes may carry different
        numbers of them. Use np.inf to denote infinite extent along an axis.

        An entry of the list or tuple form may instead be a callable
        ``lim(x_prev, *args)`` returning those limits, where ``x_prev`` has shape
        ``(k,)`` and holds the coordinates of the axes before it. The region is then no
        longer a box, and axis ``k`` may depend only on axes ``0`` to ``k-1``.
    args : tuple
        The extra arguments ``fun`` and any limit callables will be called with.

    Returns
    -------
    fun_t : callable
        Transformed integrand, taking ``x`` of shape ``(ndim,)``.
    interval_t : tuple of Array
        One interval per axis, finite along every axis, each as long as the
        corresponding entry of ``interval``. Use :func:`~quadax._utils.box_corners` to
        recover the two corners.

    Notes
    -----
    Breakpoints do not change the transformation, which acts on a point at a time; they
    are carried through it so that whatever subdivides the box afterwards can start
    from them.

    An axis whose limits are a callable is integrated over ``[0, 1]`` instead, and the
    map onto the limits it returns is folded into the integrand along with its Jacobian.
    A breakpoint the callable moves therefore sits at a fixed coordinate of the mapped
    box, which is what lets a curved feature be marked and cut along like an axis
    aligned one. The transform is exact, but the integrand it produces is only as smooth
    as the limit functions are: a curved boundary moves its curvature into the axes
    before it, so a region bounded by a square root costs far more subdivision than one
    bounded by a straight line.

    The map is separable, so its Jacobian is the product of the one dimensional ones and
    reaches the ``ndim`` th power of the magnitude a single axis would give. On a
    low precision dtype that product can overflow at the nodes furthest out, where the
    integrand is correspondingly small; the non-finite value is masked away rather than
    contributing, so the node is lost. Mapping an infinite domain in half precision is
    limited by this rather than by the rule applied afterwards.
    """
    intervals = _as_box_intervals(interval)
    lims = tuple(axis if callable(axis) else None for axis in intervals)
    # One dtype across the axes, so that the per axis bounds can be stacked: each axis
    # derives its own junctions and unit, and a box mixing a bounded axis with an
    # unbounded one would otherwise settle on a different dtype for each. A variable
    # axis stands in as the reference interval it is integrated over.
    intervals = list(_box_reference_intervals(intervals, args))
    xtype = jnp.result_type(*intervals)

    a, b = box_corners(intervals)
    # Reversing an axis flips the sign of the integral, so every axis is put in
    # ascending order and the signs are collected into one factor.
    sgn = jnp.prod(jnp.where(a > b, -1, 1).astype(xtype))
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    # Breakpoints outside their axis are pulled to its endpoints, which leaves sub
    # boxes of zero width for whatever subdivides the box to ignore.
    intervals = [jnp.sort(jnp.clip(v, a[k], b[k])) for k, v in enumerate(intervals)]

    # One set of junctions and one unit per axis, each from that axis's own
    # breakpoints, since the axes are mapped independently and need not share a scale.
    # The loop is over a static number of axes, so it unrolls at trace time.
    bounds = [_map_bounds(v) for v in intervals]
    scale, shift, anchored, lo, hi, t_lo, t_hi = (
        jnp.stack([bd[k] for bd in bounds]) for k in range(7)
    )

    fun_mapped = _MappedBoxFunction(
        fun, sgn, a, b, scale, shift, anchored, lo, hi, lims, tuple(intervals)
    )

    interval_t = tuple(
        jnp.where(
            v == jnp.inf, t_hi[k], jnp.where(v == -jnp.inf, t_lo[k], v + shift[k])
        )
        for k, v in enumerate(intervals)
    )
    return fun_mapped, interval_t


def _map_variable_axis(
    u: jax.Array, knots: jax.Array, lim: Callable, x_prev: jax.Array, args: tuple
):
    """One axis whose limits are a function of the coordinates before it.

    ``u`` is the axis' coordinate in the reference interval it is integrated over,
    ``knots`` the points of that interval, and ``lim`` the callable giving the axis'
    real limits at the coordinates ``x_prev``. The two are joined by the piecewise
    affine map that sends knot to knot, so a breakpoint the callable moves keeps a fixed
    reference coordinate however far it moves, and the mesh can be cut along it like an
    axis aligned one. Beyond that the axis is mapped exactly as a fixed one is, which is
    what lets the limits a callable returns be unbounded.

    Returns the abscissa, the derivative of the whole composition, and the sign, which
    unlike a fixed axis' is a property of the point: only the callable knows which way
    round its limits come out, and it need not come out the same way everywhere.
    """
    iv = jnp.asarray(lim(x_prev, *args))
    sgn = jnp.where(iv[-1] < iv[0], -1, 1).astype(iv.dtype)
    a, b = jnp.minimum(iv[0], iv[-1]), jnp.maximum(iv[0], iv[-1])
    # Breakpoints outside their axis are pulled to its endpoints, as they are for a
    # fixed axis, which leaves pieces of zero width contributing nothing.
    iv = jnp.sort(jnp.clip(iv, a, b))
    scale, shift, anchored, lo, hi, t_lo, t_hi = _map_bounds(iv)
    # Where each knot has to land, formed exactly as `map_box` forms the limits it
    # returns, so that an unbounded limit lands on the boundary the tail map reaches
    # only as a limit rather than on some other float near it.
    t_knots = jnp.where(
        iv == jnp.inf, t_hi, jnp.where(iv == -jnp.inf, t_lo, iv + shift)
    )
    # The last knot belongs to the piece below it, so that the top of the interval maps
    # to the top rather than off the end of the table.
    j = jnp.clip(jnp.searchsorted(knots, u, side="right") - 1, 0, knots.shape[0] - 2)
    slope = (t_knots[j + 1] - t_knots[j]) / (knots[j + 1] - knots[j])
    x, w = _map(t_knots[j] + (u - knots[j]) * slope, scale, shift, anchored, lo, hi)
    return x, slope * w, sgn


class _MappedBoxFunction(eqx.Module):
    """Function mapped to a box a fixed cubature rule can be applied over."""

    fun: Callable[..., jax.Array]
    sgn: jax.Array
    a: jax.Array
    b: jax.Array
    scale: jax.Array
    shift: jax.Array
    anchored: jax.Array
    lo: jax.Array
    hi: jax.Array
    lims: tuple
    knots: tuple

    @eqx.filter_jit
    def __call__(self, t: jax.Array, *args):
        ndim = self.a.shape[0]
        xw = [
            _map(
                t[k],
                self.scale[k],
                self.shift[k],
                self.anchored[k],
                self.lo[k],
                self.hi[k],
            )
            for k in range(ndim)
        ]
        x = [p[0] for p in xw]
        # The map acts on each axis alone, so its Jacobian is diagonal and the volume
        # element is the product of the per axis derivatives.
        w = jnp.prod(jnp.stack([p[1] for p in xw]))
        sgn = self.sgn
        # An axis whose limits are a callable is still in its reference coordinate; it
        # is mapped onto its real limits here, after the axes those limits depend on
        # have their own coordinates. Only such axes are ordered against each other, and
        # a box of constant limits has none, so this loop is empty and costs nothing.
        for k, lim in enumerate(self.lims):
            if lim is None:
                continue
            x_prev = jnp.stack(x[:k]) if k else jnp.zeros((0,), self.a.dtype)
            x[k], w_k, sgn_k = _map_variable_axis(
                x[k], self.knots[k], lim, x_prev, args
            )
            w, sgn = w * w_k, sgn * sgn_k
        return sgn * w * self.fun(jnp.stack(x), *args)


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


def mapping_of(fun):
    """The :class:`_MappedFunction` inside ``fun``, or None if it is not mapped.

    A rule is handed an integrand that ``map_interval`` has already composed with its
    change of variable, so the abscissae it sees are the reference ones and not the
    caller's. Almost every rule is right to ignore that. One that is not -- because it
    carries something of its own expressed in the caller's coordinate, such as the
    location of a weight's singularity -- needs to know which map was applied, and this
    is how it asks.
    """
    seen = set()
    while fun is not None and id(fun) not in seen:
        if isinstance(fun, _MappedFunction):
            return fun
        seen.add(id(fun))
        fun = getattr(fun, "fun", None)
    return None


def apply_mapping(mapping, t: jax.Array):
    """Send reference abscissae back to the caller's coordinate.

    Returns the caller's ``x`` and the map's Jacobian ``dx/dt`` there. ``mapping`` of
    ``None`` means no map was applied, which is the case when a rule is called directly
    rather than through a routine, and the two coordinates are the same thing.

    Used by the rules that carry something declared in the caller's coordinate -- a
    weight's singular point, an oscillator's phase -- and so cannot simply take the
    abscissa they are handed at face value. See :func:`mapping_of`.
    """
    if mapping is None:
        return t, jnp.ones_like(t)
    return _map(
        t, mapping.scale, mapping.shift, mapping.anchored, mapping.lo, mapping.hi
    )


def invert_mapping(mapping, x: jax.Array):
    """Send points of the caller's coordinate to the reference one.

    The counterpart of :func:`apply_mapping`, and the same function ``map_interval``
    puts the breakpoints through, so a point declared in the caller's coordinate and a
    breakpoint spliced in beside it land on the same float.
    """
    if mapping is None:
        return x
    return _map_inv(
        x, mapping.scale, mapping.shift, mapping.anchored, mapping.lo, mapping.hi
    )


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


# The four cases a limit can fall into, each composed with the tanh-sinh substitution
# and rewritten in terms of the node's distance from the end of [-1, 1] it clusters
# against. Composing them rather than applying one after the other is what keeps the
# outermost nodes: the substitution reaches far closer to an endpoint than a node
# written down as a position can record, and the two Jacobians have factors that cancel
# and would otherwise leave the range on their own.
#
# These are the branches of a `lax.switch`, so all four have to return the same dtypes,
# and they only do so if `t`, `a` and `b` agree. Note in particular that they must not
# be given a *weakly* typed `t`: the four differ in which of `a`/`b` they use, and a
# weak `t` lets each branch settle on whichever of the two is present. `_ts_ainf`'s
# `a + offset` would follow a strong float32 `a`, while `_ts_ninfinf` touches neither
# limit and would stay at the weak default, and the switch would not build.
# This is why the integrand is probed with `jnp.zeros((), xtype)`, not `jnp.array(0.0)`.
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


def wrap_func(
    fun: Callable[..., jax.Array],
    args: tuple[Any, ...],
    xtype,
    batch_size: int | None = None,
    safe: bool = False,
    ndim: int | None = None,
):
    """Vectorize, jit, and mask out inf/nan.

    ``xtype`` is the dtype the integrand will be called at, and the integrand is probed
    at that dtype rather than at a weakly typed default. See the note on
    ``TS_MAPFUNS``.

    ``ndim`` says what one abscissa looks like. ``None`` is a scalar, the 1D case. An
    int is the length of the vector a cubature rule hands the integrand, which becomes
    a core dimension of the vectorization rather than one of the axes looped over.

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
            fun.fun, fun.args, fun.outsig, batch_size, safe or fun.safe, fun.ndim
        )

    xprobe = jnp.zeros(() if ndim is None else (ndim,), xtype)
    f = jax.eval_shape(fun, xprobe, *args)
    # need to make sure we get the correct shape for array valued integrands
    outsig = "(" + ",".join("n" + str(i) for i in range(len(f.shape))) + ")"

    return _WrappedFunction(fun, args, outsig, batch_size, safe, ndim)


def _bad_abscissae(bad, x, ncore=0):
    """Which abscissae the integrand cannot be linearized at.

    ``bad`` flags individual non-finite *values*, and carries the integrand's own axes
    after the ones ``x`` is looped over. Those are collapsed here because where to
    linearize is a choice per abscissa, not per component: every component of a vector
    valued integrand is evaluated at the same point and differentiated with respect to
    the same parameters, so one component blowing up is enough to poison the derivatives
    of the others through the parameters they share.

    ``ncore`` is the rank of a single abscissa, so the trailing ``ncore`` axes of ``x``
    are part of one point rather than axes to loop over. The result is shaped like the
    axes that are looped over.
    """
    loopshape = jnp.shape(x)[: jnp.ndim(x) - ncore]
    return jnp.any(jnp.reshape(bad, loopshape + (-1,)), axis=-1)


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
    ndim: int | None = eqx.field(static=True, default=None)

    @property
    def _ncore(self) -> int:
        """Rank of a single abscissa: 0 for a scalar, 1 for a point in ``ndim`` axes."""
        return 0 if self.ndim is None else 1

    def _vectorize(self, x: jax.Array) -> jax.Array:
        return jnp.vectorize(
            self.fun,
            excluded=tuple(range(1, len(self.args) + 1)),
            signature=("()" if self.ndim is None else "(d)") + "->" + self.outsig,
        )(x, *self.args)

    def _evaluate(self, x: jax.Array) -> jax.Array:
        """The integrand at every point of ``x``, in batches, without any masking."""
        b = self.batch_size
        # A single abscissa has nothing to batch: Romberg calls the integrand one point
        # at a time, and every caller probes it with one point under eval_shape. That is
        # ``x.ndim == 0`` for a scalar abscissa and ``x.ndim == 1`` for a vector one,
        # which is what makes the test the core rank rather than zero.
        if b is None or x.ndim == self._ncore or x.shape[0] <= b:
            return self._vectorize(x)
        n = x.shape[0]
        nfull = n // b
        # ``x.shape[1:]`` is empty for a scalar abscissa and ``(ndim,)`` for a vector
        # one, so the split is along the looped axis either way.
        full = jax.lax.map(
            self._vectorize, x[: nfull * b].reshape(nfull, b, *x.shape[1:])
        )
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
        bad_x = _bad_abscissae(bad, x, self._ncore)
        # Linearize at an abscissa the integrand was just seen to be finite at. Taking
        # one from the same set rather than some fixed point is what keeps this from
        # walking into the singularity: any fixed choice (the midpoint of the domain,
        # say) is itself the singularity for some integrand. Only finiteness matters,
        # since whatever it evaluates to there is masked back out.
        good = ~jnp.reshape(bad_x, (-1,))
        # Flattened to one row per abscissa, so a row *is* a point: a scalar for a 1D
        # rule, a length ``ndim`` vector for a cubature one.
        substitute = jnp.reshape(x, (-1, *x.shape[x.ndim - self._ncore :]))[
            jnp.argmax(good)
        ]
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
            jnp.where(
                jnp.reshape(bad_x, bad_x.shape + (1,) * self._ncore), substitute, x
            ),
        )
        # Where the integrand was finite this is the ordinary evaluation, unchanged. At
        # a bad abscissa the value comes from the probe, so a vector valued integrand
        # keeps the components that were finite there and only the non-finite ones are
        # zeroed, exactly as masking the output alone would have done. Their derivative
        # is what is given up: the probe is a constant, so those components contribute
        # nothing to the tangent. That trades one abscissa's contribution to the
        # derivative for a derivative that exists at all.
        mask = jnp.reshape(
            bad_x, jnp.shape(bad_x) + (1,) * (jnp.ndim(f) - jnp.ndim(bad_x))
        )
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
        Code for why the routine terminated, one of ``quadax.STATUS``.
        ``STATUS.normal`` (0) means the requested tolerances were reached; every other
        code names a difficulty, whose message is ``print(quadax.STATUS[status])``.
        Where a run meets several conditions the most severe is reported.
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
