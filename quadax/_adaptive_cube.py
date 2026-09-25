"""Globally adaptive cubature over a box."""

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from . import _acceleration
from ._adaptive import (
    _MIN_WIDTH,
    _NO_PROGRESS,
    _ROUNDOFF_ACCEL_LIMIT,
    _STAGNANT_RTOL,
    _accelerate,
    _accept_extrapolation,
    _at_roundoff_floor,
)
from ._adjoint import (
    AbstractAdjoint,
    DirectAdjoint,
    LeibnizAdjoint,
    QuadratureOps,
    _box_boundary_term,
    _frozen_mesh,
    _frozen_replay,
    _quad_on_mesh,
    _rebuild_box_mesh,
    _replay_solve,
    build_box_integrand,
    closure_convert,
)
from ._fixed_cubature import AbstractCubatureRule, GenzMalikRule
from ._status import STATUS, error_if_flagged, escalate
from ._utils import (
    _ROUNDOFF_FLOOR,
    QuadratureInfo,
    _as_box_intervals,
    _box_reference_intervals,
    _real_dtype,
    bounded_while_loop,
    box_corners,
    errorif,
    resolve_dtypes,
)


@eqx.filter_jit
def cubegm(
    fun: Callable[..., jax.Array],
    interval: ArrayLike | Sequence[ArrayLike | Callable],
    args: tuple = (),
    full_output: bool = False,
    epsabs: Any = None,
    epsrel: Any = None,
    max_nregion: int = 1000,
    degree: int = 9,
    norm: float | int | Callable[[jax.Array], jax.Array] = jnp.inf,
    adjoint: AbstractAdjoint = DirectAdjoint(),
    extrapolate: bool = True,
    batch_size: int | None = None,
    throw: bool = False,
):
    """Global adaptive cubature over a box using a Genz-Malik rule.

    Integrate `fun` over the box `interval` using an h-adaptive scheme with error
    estimate. Breakpoints can be specified per axis in `interval` where integration
    difficulty may occur.

    The general purpose cubature to reach for first. Each subdivision cuts one region
    along one axis, chosen by the rule itself, so a difficulty that lies along an axis
    aligned plane is refined without the axes it spans being refined with it.

    Where the integrand has a jump or a singularity on a known axis aligned plane,
    passing that coordinate as a breakpoint on its axis is worth more than any change of
    method, since the subdivision no longer has to find it.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array, where ``x`` has shape ``(ndim,)``. Should be
        JAX transformable.
    interval : array-like or sequence of array-like or callable
        Limits of integration. Either an array of shape ``(ndim, 2)``, row ``k`` giving
        the two limits of axis ``k``, or a list or tuple of one entry per axis, each
        holding that axis' two limits with any breakpoints between them. The second
        form is what carries breakpoints, and the axes may carry different numbers of
        them. ``ndim`` is taken from the length of either. Use np.inf to denote
        unbounded axes. The dtype sets the working precision: the integrand is called
        with an ``x`` of this dtype, and the result follows it unless the integrand
        upcasts. Integer types or python floats fall back to the JAX default. Must be
        real; complex integrands are supported, complex limits are not.

        An entry of the list or tuple form may instead be a callable
        ``lim(x_prev, *args)`` returning those same limits, where ``x_prev`` has shape
        ``(k,)`` and holds the coordinates of the axes before it, so that the region
        need not be a box. Axis ``k`` may depend only on axes ``0`` to ``k-1``, which
        makes the order of the axes significant. Any breakpoints the callable returns
        may move with the coordinates too, which is how a feature lying along a curve
        is marked.
    args : tuple, optional
        Extra arguments passed to fun and any callable interval limit.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is the square root of the
        machine precision of the working dtype. Algorithm tries to obtain an accuracy of
        ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` = integral of `fun`
        over `interval`, and ``result`` is the numerical approximation.
    max_nregion : int, optional
        An upper bound on the number of regions used in the adaptive algorithm.
    degree : int
        Degree of polynomial the local rule integrates exactly, one of 7, 9, 11, 13.
        Higher degrees cost sharply more per region as ``ndim`` grows; see
        :class:`~quadax.GenzMalikRule` for the counts and for what the choice buys.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the cubature. :class:`~quadax.DirectAdjoint`,
        the default, differentiates the subdivision the solve settled on and is usually
        the cheaper choice. :class:`~quadax.LeibnizAdjoint` gives the derivative its own
        error control, at the cost of integrating over the faces of the box, which is an
        adaptive solve of one dimension fewer per axis whose limits move.
    extrapolate : bool, optional
        Whether to accelerate convergence by applying Wynn's epsilon algorithm to the
        sequence of running totals, on by default. Not needed for smooth integrands on
        finite domains, but can help significantly if there are algebraic singularities
        or infinite intervals. Can be turned off for integrands known to be smooth
        which can reduce overhead and improve wall time.
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel. Default
        is all of the local rule's nodes at once, which is fastest but makes peak memory
        scale with the node count. Lower it to reduce memory on an expensive integrand.
    throw : bool, optional
        Whether to raise an error if the routine does not converge. If True, a run
        that terminates for any reason other than reaching the requested tolerance
        raises with the message its ``status`` carries. If False, the default, that
        status is reported on the returned ``info`` and left to the caller to act on.

    Returns
    -------
    y : float, Array
        The integral of fun over the box.
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (quadax.STATUS) Why the routine terminated. ``STATUS.normal`` means
          the requested tolerances were reached; every other member names a difficulty
          and prints as the message explaining it.
        * info : (dict or None) Other information returned by the algorithm.
          Only present if `full_output` is True. Contains the following:

          * 'nregion' : (int) The number, K, of regions produced in the subdivision.
          * 'a_arr', 'b_arr' : (ndarray) shape (max_nregion, ndim), the first K rows of
            which are the opposite corners of the (remapped) regions.
          * 'r_arr' : (ndarray) the integral approximations on those regions.
          * 'e_arr' : (ndarray) the absolute error estimates on those regions.

    Notes
    -----
    An unbounded axis is mapped to a finite one by the same substitution the one
    dimensional routines use, applied to each axis on its own. Because that map is
    separable its Jacobian is a product over axes, which for an integrand decaying
    algebraically in ``|x|`` disagrees with the decay along different directions into a
    corner of the mapped box. The value is unaffected, the map being a bijection onto
    the original domain, but the mesh converges on it at first order there rather than
    at the rule's degree, and such a problem costs far more regions than its smoothness
    suggests. An integrand decaying exponentially, or one over an axis unbounded on only
    one side, is not affected. No breakpoint helps, the difficulty lying at a point the
    map sends to infinity.

    An axis whose limits are given as a callable is integrated over a fixed interval
    instead, and the map onto the limits it returns is folded into the integrand along
    with its Jacobian. The region is then a box again, and everything above applies to
    it unchanged; a limit the callable returns may be unbounded like any other. What
    such a region costs is smoothness: a curved boundary moves its curvature into the
    axes before it, so a disc costs far more subdivision than a triangle of the same
    area does, and no breakpoint helps there either, the difficulty being a square root
    rather than a kink. The limit functions are evaluated once per node per axis they
    govern.

    """
    ndim = len(_as_box_intervals(interval))
    rule = GenzMalikRule(ndim, degree, norm, batch_size)
    y, info = adaptive_cubature(
        rule,
        fun,
        interval,
        args,
        full_output,
        epsabs,
        epsrel,
        max_nregion,
        adjoint=adjoint,
        extrapolate=extrapolate,
        throw=throw,
    )
    info = QuadratureInfo(
        info.err, info.neval * rule.nodes_per_call, info.status, info.info
    )
    return y, info


@eqx.filter_jit
def adaptive_cubature(
    rule: AbstractCubatureRule,
    fun: Callable[..., jax.Array],
    interval: ArrayLike | Sequence[ArrayLike | Callable],
    args: tuple = (),
    full_output: bool = False,
    epsabs: Any = None,
    epsrel: Any = None,
    max_nregion: int = 1000,
    adjoint: AbstractAdjoint = DirectAdjoint(),
    extrapolate: bool = True,
    throw: bool = False,
    **kwargs,
):
    """Global adaptive cubature with user specified local rule.

    This is a lower level routine allowing for custom local cubature rules. For most
    applications :func:`~quadax.cubegm` is preferable; this is the way to reach a
    :class:`~quadax.TensorProductRule`, which is worth its cost mainly in two or three
    dimensions or where one axis is far harder than the others.

    Parameters
    ----------
    rule : AbstractCubatureRule
        Local cubature rule to use. Its ``ndim`` must match `interval`.
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array, where ``x`` has shape ``(ndim,)``. Should be
        JAX transformable.
    interval : array-like or sequence of array-like or callable
        Limits of integration. Either an array of shape ``(ndim, 2)``, row ``k`` giving
        the two limits of axis ``k``, or a list or tuple of one entry per axis, each
        holding that axis' two limits with any breakpoints between them. The second
        form is what carries breakpoints, and the axes may carry different numbers of
        them. ``ndim`` is taken from the length of either. Use np.inf to denote
        unbounded axes. The dtype sets the working precision: the integrand is called
        with an ``x`` of this dtype, and the result follows it unless the integrand
        upcasts. Integer types or python floats fall back to the JAX default. Must be
        real; complex integrands are supported, complex limits are not.

        An entry of the list or tuple form may instead be a callable
        ``lim(x_prev, *args)`` returning those same limits, where ``x_prev`` has shape
        ``(k,)`` and holds the coordinates of the axes before it, so that the region
        need not be a box. Axis ``k`` may depend only on axes ``0`` to ``k-1``, which
        makes the order of the axes significant. Any breakpoints the callable returns
        may move with the coordinates too, which is how a feature lying along a curve
        is marked.
    args : tuple, optional
        Extra arguments passed to fun and any callable interval limit.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is the square root of the
        machine precision of the working dtype. Algorithm tries to obtain an accuracy of
        ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` = integral of `fun`
        over `interval`, and ``result`` is the numerical approximation.
    max_nregion : int, optional
        An upper bound on the number of regions used in the adaptive algorithm.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the cubature. :class:`~quadax.DirectAdjoint`,
        the default, differentiates the subdivision the solve settled on and is usually
        the cheaper choice. :class:`~quadax.LeibnizAdjoint` gives the derivative its own
        error control, at the cost of integrating over the faces of the box, which is an
        adaptive solve of one dimension fewer per axis whose limits move.
    extrapolate : bool, optional
        Whether to accelerate convergence by applying Wynn's epsilon algorithm to the
        sequence of running totals, on by default. Not needed for smooth integrands on
        finite domains, but can help significantly if there are algebraic singularities
        or infinite intervals. Can be turned off for integrands known to be smooth
        which can reduce overhead and improve wall time.
    throw : bool, optional
        Whether to raise an error if the routine does not converge. If True, a run
        that terminates for any reason other than reaching the requested tolerance
        raises with the message its ``status`` carries. If False, the default, that
        status is reported on the returned ``info`` and left to the caller to act on.
    kwargs : dict, optional
        Additional keyword arguments passed to ``rule.integrate``.

    Returns
    -------
    y : float, Array
        The integral of fun over the box.
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of rule evaluations.
        * status : (quadax.STATUS) Why the routine terminated. ``STATUS.normal`` means
          the requested tolerances were reached; every other member names a difficulty
          and prints as the message explaining it.
        * info : (dict or None) Other information returned by the algorithm.
          Only present if `full_output` is True. Contains the following:

          * 'nregion' : (int) The number, K, of regions produced in the subdivision.
          * 'a_arr', 'b_arr' : (ndarray) shape (max_nregion, ndim), the first K rows of
            which are the opposite corners of the (remapped) regions.
          * 'r_arr' : (ndarray) the integral approximations on those regions.
          * 'e_arr' : (ndarray) the absolute error estimates on those regions.

    """
    errorif(
        not isinstance(rule, AbstractCubatureRule),
        TypeError,
        "rule should be an instance of quadax.AbstractCubatureRule, "
        f"got {type(rule)}. One dimensional rules go to quadax.adaptive_quadrature.",
    )
    # A boundary term over a box is an integral over a face, which needs a rule one
    # dimension down. Both shipped families build one; a rule that does not can still be
    # used by handing the adjoint a face rule directly. Checked here rather than where
    # the face is built so that the message names the rule the caller passed.
    errorif(
        isinstance(adjoint, LeibnizAdjoint)
        and rule.ndim > 1
        and "rule" not in {**adjoint.options, **adjoint.options_face}
        and type(rule)._drop_axis is AbstractCubatureRule._drop_axis,
        NotImplementedError,
        f"{type(rule).__name__} cannot build a rule over one fewer axis, which is what "
        "LeibnizAdjoint needs to integrate over a face of the box. Pass a rule of "
        f"dimension {rule.ndim - 1} as options_face={{'rule': ...}} on the adjoint, or "
        "use DirectAdjoint.",
    )
    axes = _as_box_intervals(interval)
    errorif(
        len(axes) != rule.ndim,
        ValueError,
        f"rule integrates over {rule.ndim} dimensions but interval gives limits for "
        f"{len(axes)}.",
    )
    # An axis whose limits are a callable is integrated over a fixed reference interval,
    # so that is what the mesh, the tolerances and the adjoints work in; the map onto
    # the limits it returns is folded into the integrand by `map_box`.
    intervals = _box_reference_intervals(axes, args)
    n_init = math.prod(len(axis) - 1 for axis in intervals)
    errorif(
        max_nregion < n_init,
        ValueError,
        f"max_nregion={max_nregion} is not enough for the {n_init} regions the "
        "breakpoints already divide the box into",
    )
    dtypes = resolve_dtypes(intervals, fun, args, ndim=rule.ndim)
    if epsabs is None:
        epsabs = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    if epsrel is None:
        epsrel = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    epsabs = jnp.asarray(epsabs, dtypes.etype)
    epsrel = jnp.asarray(epsrel, dtypes.etype)

    # The limit callables close over traced values exactly as the integrand does, and a
    # raw callable crossing into an adjoint would be static, so each is converted here
    # alongside it and its consts join the ones the adjoint differentiates.
    f_conv, consts = closure_convert(fun, args, dtypes.xtype, ndim=rule.ndim)
    lims: list = []
    all_consts = [consts]
    for k, axis in enumerate(axes):
        if not callable(axis):
            lims.append(None)
            continue
        lim_conv, lim_consts = closure_convert(axis, args, dtypes.xtype, ndim=k)
        lims.append(lim_conv)
        all_consts.append(lim_consts)
    consts = tuple(all_consts)

    # The options an adjoint may run its own solve with.
    opts = {
        "rule": rule,
        "epsabs": epsabs,
        "epsrel": epsrel,
        "max_nregion": max_nregion,
        "extrapolate": extrapolate,
    }
    ops = QuadratureOps(
        build=partial(
            build_box_integrand, f_conv=f_conv, ndim=rule.ndim, lims=tuple(lims)
        ),
        solve=_cubature_solve,
        rebuild=_rebuild_box_mesh,
        on_mesh=_quad_on_mesh,
        # An accelerated solve may return an extrapolated value rather than the sum over
        # the subdivision, so the fixed-discretization evaluation the adjoints reuse has
        # to replay the extrapolation too, not just the mesh.
        frozen=_frozen_replay if extrapolate else _frozen_mesh,
        frozen_solve=(
            partial(_replay_solve, rebuild=_rebuild_box_mesh) if extrapolate else None
        ),
        boundary=_box_boundary_term,
        mesh_is_primal=not extrapolate,
    )
    y, state = adjoint.quadrature(ops, intervals, args, consts, kwargs, opts)

    err = state["err_sum"]
    neval = state["neval"]
    status = state["status"]
    info = state if full_output else None
    out = QuadratureInfo(err, neval, status, info)
    if throw:
        y = error_if_flagged(y, status)
    return y, out


def _init_cubature_state(
    interval: Sequence[jax.Array], shape, xtype, ytype, etype, max_nregion, extrapolate
):
    """State of the subdivision loop before the initial regions are evaluated.

    The mesh holds the regions the breakpoints on each axis cut the box into, and every
    running quantity is at its identity. ``shape`` and the three dtypes are those of the
    integrand's value, the abscissae and the error estimates respectively.
    """
    ndim = len(interval)
    # Breakpoints cut each axis into cells, and the initial mesh is their tensor grid.
    # Every length here is an array shape and so is known at trace time so we can use
    # numpy and no padding is needed.
    ncells = [len(axis) - 1 for axis in interval]
    idx = np.stack(
        np.meshgrid(*[np.arange(n) for n in ncells], indexing="ij"), axis=-1
    ).reshape(-1, ndim)
    n_init = idx.shape[0]

    state = {}
    state["neval"] = 0  # number of applications of the local cubature rule
    state["nregion"] = n_init  # current number of regions
    state["r_arr"] = jnp.zeros((max_nregion, *shape), ytype)  # result from each region
    state["e_arr"] = jnp.zeros(max_nregion, etype)  # error est. from each region
    state["a_arr"] = jnp.zeros(
        (max_nregion, ndim), xtype
    )  # lower corner of each region
    state["b_arr"] = jnp.zeros(
        (max_nregion, ndim), xtype
    )  # upper corner of each region
    state["f_arr"] = jnp.zeros(
        (max_nregion, *shape), etype
    )  # local est. of integral of abs(fun) from each region
    # Which axis of each region is worth cutting, as the local rule reported it. Kept
    # rather than recomputed because the body needs it for the region it selects, not
    # for the one it has just evaluated.
    state["d_arr"] = jnp.zeros((max_nregion, ndim), etype)
    state["a_arr"] = (
        state["a_arr"]
        .at[:n_init]
        .set(jnp.stack([interval[k][idx[:, k]] for k in range(ndim)], axis=-1))
    )
    state["b_arr"] = (
        state["b_arr"]
        .at[:n_init]
        .set(jnp.stack([interval[k][idx[:, k] + 1] for k in range(ndim)], axis=-1))
    )
    state["status"] = STATUS.normal  # why the run stopped
    # Explicitly typed rather than left as weak python floats: these are `scan` carries,
    # so their dtype has to match what the loop body writes back into them.
    state["err_bnd"] = jnp.zeros((), etype)  # error bound we're trying to reach
    state["area"] = jnp.zeros(shape, ytype)  # current best estimate for I
    state["err_sum"] = jnp.zeros((), etype)  # current estimate for error in I
    # Where each region sits relative to the *initial* mesh, one column per axis: which
    # cell of that axis' breakpoints it was carved out of, and the fractions of the way
    # along that cell its two corners lie at. These are what stay fixed when a limit or
    # a breakpoint moves, so recording them lets the mesh be rebuilt as a smooth
    # function of `interval` by gather and rescale.
    state["owner"] = jnp.zeros((max_nregion, ndim), int).at[:n_init].set(idx)
    state["frac_a"] = jnp.zeros((max_nregion, ndim), xtype)
    state["frac_b"] = jnp.zeros((max_nregion, ndim), xtype).at[:n_init].set(1.0)

    if not extrapolate:
        return state

    # How deep in the subdivision each region sits, ie how many cuts separate it from
    # the cell of the initial grid containing it. This is the measure of whether the
    # mesh has localized: a region that has been cut `level_max` times is treated as
    # resolved, and the loop leaves it alone and extrapolates past it instead. Depth
    # rather than size, because with breakpoints the cells of the initial grid can
    # differ in size and a single threshold across the whole box would declare the small
    # ones resolved before they had been touched. Each cut halves a region's volume
    # whichever axis it falls on, so the count is the same quantity in every dimension.
    state["level"] = jnp.zeros(max_nregion, int)  # depth of each region
    state["level_max"] = 1  # depth at which a region counts as resolved
    # Which region to cut next, and its rank in the error ordering (when extrapolation
    # is used we don't always cut the largest error). Both are chosen at the *end* of an
    # iteration, since the choice is part of the extrapolation control flow, so they are
    # carried rather than recomputed from `e_arr` at the top of the body.
    state["bisect_next"] = jnp.zeros((), int)  # slot
    state["bisect_next_err_rank"] = jnp.zeros((), int)  # its 0-based rank by error

    state["accel_table"] = _acceleration.init_table(shape, ytype)  # the epsilon table
    state["accel_result"] = jnp.zeros(shape, ytype)  # best extrapolation so far
    # Its estimated error. Infinite until an extrapolation is accepted, which is how
    # "none was ever taken" is recognized at the end.
    state["accel_err"] = jnp.array(jnp.inf, etype)
    # How tightly that extrapolation settled, which is what candidates are ranked by.
    state["accel_sharp"] = jnp.array(jnp.inf, etype)
    state["n_stalled"] = jnp.zeros((), int)  # extrapolations with no improvement
    state["accelerating"] = jnp.zeros((), bool)  # mesh localized, table being fed
    state["no_accel"] = jnp.zeros((), bool)  # acceleration abandoned for good
    # Readings that told the table nothing, and the flag they raise. What is counted is
    # a term fed to the table that moved neither the running total nor its error; see
    # the body for why this is not QUADPACK's per-bisection count. `area_fed` and
    # `err_fed` are the two as they stood at the last reading, and `n_append_seen` is
    # how many readings had been taken when the body last looked, which is how it
    # recognizes that one has happened.
    state["roundoff_accel"] = jnp.zeros((), int)
    state["roundoff_in_table"] = jnp.zeros((), bool)  # the sequence has stagnated
    state["area_fed"] = jnp.zeros(shape, ytype)
    state["err_fed"] = jnp.zeros((), etype)
    state["n_append_seen"] = jnp.zeros((), int)
    state["correc"] = jnp.zeros((), etype)  # what to widen `accel_err` by if it has
    state["err_accel_target"] = jnp.zeros((), etype)  # tolerance to accept one at
    # total error that we think can still be reduced by subdivision.
    state["err_unlocalized"] = jnp.zeros((), etype)
    state["accel_done"] = jnp.zeros((), bool)  # the extrapolation block's exits
    # Regions whose local rule saturated on the first pass.
    state["ndin"] = jnp.zeros(max_nregion, bool)
    # Bookkeeping for `_replay_solve`, which is how an accelerated solve is
    # differentiated. None of it is read by the integrator itself. The parent arrays and
    # the birth times are indexed by the slot a cut creates, which is unique to that
    # step; `birth` is indexed by slot and holds the birth time of whatever occupies it
    # now. Time is counted in regions rather than steps, so it starts at the initial
    # count.
    state["birth"] = jnp.full(max_nregion, n_init, int)
    state["p_owner"] = jnp.zeros((max_nregion, ndim), int)
    state["p_frac_a"] = jnp.zeros((max_nregion, ndim), xtype)
    state["p_frac_b"] = jnp.zeros((max_nregion, ndim), xtype)
    state["p_birth"] = jnp.zeros(max_nregion, int)
    state["append_mask"] = jnp.zeros(max_nregion, bool)
    state["n_append"] = jnp.zeros((), int)
    state["accel_ncall"] = jnp.zeros((), int)
    return state


def _cubature_solve(
    vfunc,
    interval,
    kwargs,
    *,
    rule,
    epsabs,
    epsrel,
    max_nregion,
    extrapolate=False,
    norm=None,
):
    """Run the globally adaptive subdivision loop over a box.

    With ``extrapolate=False`` each iteration takes the region with the largest error
    estimate, asks the local rule which of its axes is worth cutting, and bisects it
    along that axis, until the errors sum to less than the tolerance.

    With ``extrapolate=True`` the same subdivision runs, but the sequence of running
    totals it produces is also fed to Wynn's epsilon algorithm, and the limit that
    infers may be returned in place of the sum over the mesh. That changes which region
    to cut: the sequence is only extrapolate-able if its terms keep coming from the same
    process, so once the mesh has localized onto the difficulty the loop stops refining
    there and works on the rest of the box instead, feeding the table a term each time
    it does. See ``_acceleration`` for the table itself, ``_accelerate_full`` for the
    control flow that decides all this, and ``_accept_extrapolation`` for the choice
    between the two answers at the end.

    ``norm`` replaces the one the rule was built with, for a caller that measures a
    vector other than the integrand's output. ``None``, the primal's case, keeps the
    rule's own.
    """
    if norm is not None:
        rule = rule._with_norm(norm)
    intfun = partial(rule.integrate, **kwargs) if kwargs else rule.integrate
    _norm = rule.norm
    lo, hi = box_corners(interval)
    f = jax.eval_shape(vfunc, (lo + hi) / 2)
    shape = f.shape
    # Derived here rather than threaded in, so that this stays correct when an adjoint
    # calls it with a tangent integrand whose dtype is not the primal's. `vfunc` has
    # already been through `map_box`, whose Jacobian is at `xtype`, so its output dtype
    # is the accumulation dtype by construction.
    xtype = jnp.result_type(*interval)
    ytype = f.dtype
    etype = _real_dtype(ytype)  # errors and the integral of |f| are real
    # Roundoff in the arithmetic that forms the sums, versus roundoff in the mesh: the
    # first bounds how small an error estimate can honestly be, the second how narrow a
    # region can get before its corners stop being distinguishable.
    epmach = float(jnp.finfo(etype).eps)
    epmach_x = float(jnp.finfo(xtype).eps)
    # "Too narrow" is per axis, and only meaningful against the span being subdivided.
    halfspan = jnp.abs(hi - lo) / 2

    state = _init_cubature_state(
        interval, shape, xtype, ytype, etype, max_nregion, extrapolate
    )

    def init_body(i, state):
        a = state["a_arr"][i]
        b = state["b_arr"][i]
        result, abserr, intabs, intmmn, split = intfun(vfunc, a, b, ())

        if extrapolate:
            # A cell of the initial grid whose error estimate reached the saturation
            # value (the whole variation of the integrand over it) told the rule nothing
            # about it at all. Those are promoted to the head of the error ordering
            # below, so that the pieces the caller flagged as difficult by putting a
            # breakpoint at them are the first ones cut. The test is ``>=`` and not
            # equality because a rule may add to the saturated value. An integrand with
            # no variation to saturate against is excluded rather than counted as
            # unresolved, the estimate then being a roundoff floor sitting above a
            # variation of zero.
            variation = _norm(intmmn)
            state["ndin"] = (
                state["ndin"].at[i].set((abserr >= variation) & (variation != 0))
            )

        state["neval"] += 1
        state["r_arr"] = state["r_arr"].at[i].set(result)
        state["e_arr"] = state["e_arr"].at[i].set(abserr)
        state["f_arr"] = state["f_arr"].at[i].set(intabs)
        state["d_arr"] = state["d_arr"].at[i].set(split)
        state["area"] = jnp.sum(state["r_arr"], axis=0)
        state["err_sum"] = jnp.sum(state["e_arr"])
        return state

    state = jax.lax.fori_loop(0, state["nregion"], init_body, state)

    state["err_bnd"] = jnp.maximum(epsabs, epsrel * _norm(state["area"]))
    # check for roundoff error - error too big but relative error is small
    state["status"] = escalate(
        state["status"], STATUS.roundoff, _at_roundoff_floor(state, epmach, _norm)
    )

    # check for max regions exceeded
    state["status"] = escalate(
        state["status"], STATUS.max_nregion, state["nregion"] >= max_nregion
    )

    if extrapolate:
        # Give the saturated cells the whole error estimate, which puts them at the head
        # of the ordering however small their own contribution was. This comes after the
        # roundoff check on purpose: that check asks whether the honest sum of the local
        # error estimates has already bottomed out at the arithmetic's floor, and
        # inflating one of them first would hide that.
        state["e_arr"] = jnp.where(
            state["ndin"], jnp.sum(state["e_arr"]), state["e_arr"]
        )
        state["err_sum"] = jnp.sum(state["e_arr"])
        # Total integral of |f| over the whole box, as the initial grid sees it. Only
        # used for the divergence test at the end, and deliberately the *initial* value:
        # it is the scale the answer is compared against, not a running quantity.
        abs_total = _norm(jnp.sum(state["f_arr"], axis=0))
        # False says the integral came out far smaller than the integral of |f|, ie the
        # answer is the residue of heavy cancellation, which makes the ratio test at the
        # end meaningless on a value near zero. True says the integrand did not change
        # sign, to within roundoff, so the two are the same size and the ratio means
        # something. The slack is the same roundoff level the error estimates use.
        state["sign_known"] = (
            _norm(state["area"]) >= (1 - _ROUNDOFF_FLOOR * epmach) * abs_total
        )
        state["abs_total"] = abs_total
        state["bisect_next"] = jnp.argmax(state["e_arr"])
        state["err_accel_target"] = state["err_bnd"]
        state["err_unlocalized"] = state["err_sum"]
        # The total over the initial grid is the first term of the sequence. It seeds
        # the table directly, with no extrapolation performed on it, so it must not
        # count towards `n_calls`, and it is the value the next reading is judged to
        # have moved from.
        state["accel_table"] = _acceleration.append(state["accel_table"], state["area"])
        state["area_fed"] = state["area"]
        state["err_fed"] = state["err_sum"]

    def condfun(state):
        keep_going = (
            (state["status"] == STATUS.normal)
            & (0 <= state["err_sum"])
            & (state["err_bnd"] <= state["err_sum"])
        )
        if extrapolate:
            # The extrapolation block has its own two exits: an extrapolated value that
            # meets the tolerance, and a table that has stopped improving.
            keep_going &= ~state["accel_done"]
        return keep_going

    def bodyfun(state):
        if extrapolate:
            # Whether the last reading told the table anything, which is what says the
            # sequence being fed has stagnated. QUADPACK counts instead the *bisections*
            # that moved neither the value nor its error, and that counter does not
            # transfer to a box: a symmetric integrand has up to ``2**ndim * ndim!``
            # regions carrying identical error estimates, the ordering walks them one
            # after another, and one fact about the integrand is recorded once per
            # equivalent region, saturating any such counter while the total is still
            # coming down. Counting readings instead is one event per term fed to the
            # table however many regions are equivalent, so multiplicity cannot reach
            # it. `n_append` is bumped whenever a term is fed, so comparing it against
            # what was seen last iteration says a reading has happened since, and
            # `area` is then the term that was fed.
            #
            # Both halves of QUADPACK's test are kept, at this granularity: the total
            # did not move, *and* its error estimate did not come down. The first alone
            # fires on any problem the mesh is already resolving, where successive
            # totals agree to well inside the threshold while the run is making
            # perfectly good progress.
            stagnant = max(_STAGNANT_RTOL, _ROUNDOFF_FLOOR * epmach)
            fed = state["n_append"] > state["n_append_seen"]
            moved = _norm(state["area"] - state["area_fed"]) > stagnant * _norm(
                state["area"]
            )
            progressed = state["err_sum"] < _NO_PROGRESS * state["err_fed"]
            state["roundoff_accel"] += fed & ~moved & ~progressed
            state["roundoff_in_table"] |= (
                state["roundoff_accel"] >= _ROUNDOFF_ACCEL_LIMIT
            )
            state["area_fed"] = jnp.where(fed, state["area"], state["area_fed"])
            state["err_fed"] = jnp.where(fed, state["err_sum"], state["err_fed"])
            state["n_append_seen"] = state["n_append"]

        # Cut the region the extrapolation control flow selected, which without it is
        # always the one with the largest error estimate, along whichever of its axes
        # the local rule reported as worth cutting. `d_arr` is only ordinally
        # meaningful and local rules may build it differently, so `argmax` is all
        # that may be read from it.
        i = state["bisect_next"] if extrapolate else jnp.argmax(state["e_arr"])
        k = jnp.argmax(state["d_arr"][i])
        # The cut turns one region into two, so the extra one goes in the first free
        # slot, which is the current region count, before incrementing it.
        n = state["nregion"]
        state["nregion"] += 1
        a_i = state["a_arr"][i]
        b_i = state["b_arr"][i]
        mid = 0.5 * (a_i[k] + b_i[k])
        a1, b1 = a_i, b_i.at[k].set(mid)
        a2, b2 = a_i.at[k].set(mid), b_i

        r_i = state["r_arr"][i]
        # The parent's error estimate, read before it is overwritten below. The
        # extrapolation's accounting of how much error is still worth subdividing
        # compares the two halves against the parent, so it has to be captured here.
        err_i = state["e_arr"][i]
        area1, error1, intabs1, _, split1 = intfun(vfunc, a1, b1, ())
        state["neval"] += 1
        area2, error2, intabs2, _, split2 = intfun(vfunc, a2, b2, ())
        state["neval"] += 1

        # A local error estimate reads one region's function values, so a feature that
        # falls between the abscissae is invisible to it and the region is retired with
        # an estimate far below the error it actually made. Bisecting it puts abscissae
        # somewhere new, and the disagreement between the parent's approximation and the
        # sum of its two children measures directly what the parent missed. That
        # disagreement is handed down to the children in proportion to what they
        # themselves report, so it lands on the half that looks worse and follows the
        # feature down the tree instead of being forgotten one level below where it was
        # found. Sharing it equally instead spends the same margin on the clean half of
        # every cut, which measures as several times the cost for no gain in
        # reliability. Each region carries only the disagreement of its own bisection:
        # the term is replaced, not accumulated, when the region is itself cut.
        disagreement = _norm(r_i - (area1 + area2)).astype(error1.dtype)
        total = error1 + error2
        # Split evenly when neither child claims any error, which is the only case where
        # proportion says nothing about where the missed feature is.
        safe = jnp.where(total > 0, total, 1)
        share = jnp.where(total > 0, error1 / safe, 0.5)
        error1 = error1 + share * disagreement
        error2 = error2 + (1 - share) * disagreement
        # What the two halves contribute to the running total, which is what leaves the
        # unlocalized error when they are localized. Taken after the handout so that it
        # matches the estimates actually recorded, and so `err_sum`.
        erro12 = error1 + error2

        # Which half keeps slot `i` and which takes the new slot `n`: the larger error
        # goes into `i`, the slot the ordering was already pointing at, so that the two
        # halves are ranked correctly relative to each other without consulting the rest
        # of the mesh. Every array the two halves write is placed the same way, corners
        # included, because a corner is a row here rather than a pair of scalars.
        swap = error2 > error1

        def place(arr, x1, x2):
            """Write the two halves into slots `i` and `n`, ordered by `swap`."""
            return (
                arr.at[i]
                .set(jnp.where(swap, x2, x1))
                .at[n]
                .set(jnp.where(swap, x1, x2))
            )

        # Both halves stay inside whichever cell of the initial mesh this region came
        # from, splitting its span along the cut axis at the midpoint of the fractions.
        # The other axes are untouched, exactly as the corners are.
        frac_a_i = state["frac_a"][i]
        frac_b_i = state["frac_b"][i]
        frac_mid = 0.5 * (frac_a_i[k] + frac_b_i[k])
        frac_a1, frac_b1 = frac_a_i, frac_b_i.at[k].set(frac_mid)
        frac_a2, frac_b2 = frac_a_i.at[k].set(frac_mid), frac_b_i
        owner_i = state["owner"][i]

        state["e_arr"] = place(state["e_arr"], error1, error2)
        state["r_arr"] = place(state["r_arr"], area1, area2)
        state["f_arr"] = place(state["f_arr"], intabs1, intabs2)
        state["d_arr"] = place(state["d_arr"], split1, split2)
        state["a_arr"] = place(state["a_arr"], a1, a2)
        state["b_arr"] = place(state["b_arr"], b1, b2)
        state["owner"] = place(state["owner"], owner_i, owner_i)
        state["frac_a"] = place(state["frac_a"], frac_a1, frac_a2)
        state["frac_b"] = place(state["frac_b"], frac_b1, frac_b2)

        if extrapolate:
            # Both halves sit one level deeper than the region they came from.
            levcur = state["level"][i] + 1
            state["level"] = state["level"].at[i].set(levcur).at[n].set(levcur)
            # Record the region this step consumed, and when the three regions involved
            # entered the running total. Together with the final subdivision this is
            # every region that ever existed, which is what `_replay_solve` needs to
            # rebuild the sequence the table was fed. None of it depends on which half
            # ends up in which slot: the two halves are born together and are only ever
            # summed.
            state["p_owner"] = state["p_owner"].at[n].set(owner_i)
            state["p_frac_a"] = state["p_frac_a"].at[n].set(frac_a_i)
            state["p_frac_b"] = state["p_frac_b"].at[n].set(frac_b_i)
            state["p_birth"] = state["p_birth"].at[n].set(state["birth"][i])
            state["birth"] = state["birth"].at[i].set(n + 1).at[n].set(n + 1)

        # Both running totals are summed afresh from the per-region contributions rather
        # than carried forward as `total += new - old`. Accumulating discards ~eps times
        # the largest term ever subtracted on every iteration, and that drift
        # random-walks while the total it is tracking shrinks. For `err_sum` the two
        # move in opposite directions and the drift can outgrow the total outright, at
        # which point the loop exits through the `0 <= err_sum` guard in `condfun` with
        # `status` still normal, reporting a nonsense error estimate and a clean bill of
        # health. Summing afresh keeps the error at O(eps * total).
        state["err_sum"] = jnp.sum(state["e_arr"])
        state["area"] = jnp.sum(state["r_arr"], axis=0)
        state["err_bnd"] = jnp.maximum(epsabs, epsrel * _norm(state["area"]))

        # Whether the tolerance was reached on this iteration. Reaching it takes
        # precedence over every flag below, so an iteration that both reaches the
        # tolerance and, say, consumes the last slot still exits cleanly: the answer is
        # good, and what it cost getting there is not a failure.
        converged = state["err_sum"] <= state["err_bnd"]

        # Roundoff is reported when the error has bottomed out at the floor the
        # arithmetic imposes, and on that test alone. QUADPACK backs the same test up
        # with counters over how often a bisection leaves the local contribution
        # unmoved and its error estimate undiminished, which counts something else
        # entirely over a box: a symmetric integrand has up to ``2**ndim * ndim!``
        # regions carrying identical error estimates, the ordering walks them one after
        # another, and so a single fact about the integrand is recorded once per
        # equivalent region and saturates such a counter while the total is still coming
        # down. In one dimension at most two regions can be equivalent, which is why the
        # counters work there. A run that stops buying accuracy without reaching the
        # arithmetic floor therefore exhausts its region budget and reports that, which
        # is both true and the more actionable of the two messages.
        state["status"] = escalate(
            state["status"],
            STATUS.roundoff,
            ~converged & _at_roundoff_floor(state, epmach, _norm),
        )

        # test for max number of regions
        state["status"] = escalate(
            state["status"],
            STATUS.max_nregion,
            ~converged & (state["nregion"] >= max_nregion),
        )

        # Test for bad behavior of the integrand, ie the mesh getting too fine to
        # resolve. Only the cut axis just changed, and a region may be legitimately thin
        # along an axis nothing is cutting any more, so this asks about that axis alone.
        # It is about the *mesh*, not the values, so it scales with the precision the
        # abscissae are carried at rather than the precision of the sums.
        state["status"] = escalate(
            state["status"],
            STATUS.bad_integrand,
            ~converged & ((b1[k] - a1[k]) <= (_MIN_WIDTH * epmach_x * halfspan[k])),
        )

        if extrapolate:
            state = _accelerate(
                state,
                i,
                erro12,
                err_i,
                converged,
                _norm,
                epsabs,
                epsrel,
                epmach,
                state["nregion"],
                max_nregion,
            )
        return state

    state = bounded_while_loop(condfun, bodyfun, state, max_nregion + 1)

    y = jnp.sum(state["r_arr"], axis=0)
    if extrapolate:
        state, y = _accept_extrapolation(state, y, _norm)
    return y, state
