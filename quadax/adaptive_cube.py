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

from ._status import STATUS, error_if_flagged, escalate
from .adaptive import _MIN_WIDTH, _at_roundoff_floor
from .adjoint import (
    AbstractAdjoint,
    DirectAdjoint,
    QuadratureOps,
    build_box_integrand,
    closure_convert,
)
from .fixed_cubature import AbstractCubatureRule, GenzMalikRule
from .utils import (
    QuadratureInfo,
    _as_box_intervals,
    _real_dtype,
    bounded_while_loop,
    box_corners,
    errorif,
    resolve_dtypes,
)


@eqx.filter_jit
def cubgm(
    fun: Callable[..., jax.Array],
    interval: ArrayLike | Sequence[ArrayLike],
    args: tuple = (),
    full_output: bool = False,
    epsabs: Any = None,
    epsrel: Any = None,
    max_nregion: int = 1000,
    degree: int = 9,
    norm: float | int | Callable[[jax.Array], jax.Array] = jnp.inf,
    adjoint: AbstractAdjoint = DirectAdjoint(),
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
    interval : array-like or sequence of array-like
        Limits of integration. Either an array of shape ``(ndim, 2)``, row ``k`` giving
        the two limits of axis ``k``, or a list or tuple of one array per axis, each
        holding that axis' two limits with any breakpoints between them. The second
        form is what carries breakpoints, and the axes may carry different numbers of
        them. ``ndim`` is taken from the length of either. Use np.inf to denote
        unbounded axes. The dtype sets the working precision: the integrand is called
        with an ``x`` of this dtype, and the result follows it unless the integrand
        upcasts. Integer types or python floats fall back to the JAX default. Must be
        real; complex integrands are supported, complex limits are not.
    args : tuple, optional
        Extra arguments passed to fun.
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
        How to compute derivatives of the cubature. Only :class:`~quadax.DirectAdjoint`
        is supported.
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
    interval: ArrayLike | Sequence[ArrayLike],
    args: tuple = (),
    full_output: bool = False,
    epsabs: Any = None,
    epsrel: Any = None,
    max_nregion: int = 1000,
    adjoint: AbstractAdjoint = DirectAdjoint(),
    throw: bool = False,
    **kwargs,
):
    """Global adaptive cubature with user specified local rule.

    This is a lower level routine allowing for custom local cubature rules. For most
    applications :func:`~quadax.cubgm` is preferable; this is the way to reach a
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
    interval : array-like or sequence of array-like
        Limits of integration. Either an array of shape ``(ndim, 2)``, row ``k`` giving
        the two limits of axis ``k``, or a list or tuple of one array per axis, each
        holding that axis' two limits with any breakpoints between them. The second
        form is what carries breakpoints, and the axes may carry different numbers of
        them. ``ndim`` is taken from the length of either. Use np.inf to denote
        unbounded axes. The dtype sets the working precision: the integrand is called
        with an ``x`` of this dtype, and the result follows it unless the integrand
        upcasts. Integer types or python floats fall back to the JAX default. Must be
        real; complex integrands are supported, complex limits are not.
    args : tuple, optional
        Extra arguments passed to fun.
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
        How to compute derivatives of the cubature. Only :class:`~quadax.DirectAdjoint`
        is supported.
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
    errorif(
        not isinstance(adjoint, DirectAdjoint),
        NotImplementedError,
        f"{type(adjoint).__name__} is not supported for cubature, only DirectAdjoint. "
        "Giving the derivative its own error control requires the integrand's jump "
        "across a moving breakpoint integrated over the face it lies in, which is an "
        "(ndim-1) dimensional cubature that quadax does not have.",
    )
    intervals = _as_box_intervals(interval)
    errorif(
        len(intervals) != rule.ndim,
        ValueError,
        f"rule integrates over {rule.ndim} dimensions but interval gives limits for "
        f"{len(intervals)}.",
    )
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

    f_conv, consts = closure_convert(fun, args, dtypes.xtype, ndim=rule.ndim)

    # The options an adjoint may run its own solve with.
    opts = {
        "rule": rule,
        "epsabs": epsabs,
        "epsrel": epsrel,
        "max_nregion": max_nregion,
    }
    # Only `build` and `solve`: rebuilding the mesh as a smooth function of the limits
    # needs per-axis owner and fraction bookkeeping that nothing records, so the
    # derivative goes straight through the loop instead.
    ops = QuadratureOps(
        build=partial(build_box_integrand, f_conv=f_conv, ndim=rule.ndim),
        solve=_cubature_solve,
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
    interval: Sequence[jax.Array], shape, xtype, ytype, etype, max_nregion
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
    norm=None,
):
    """Run the globally adaptive subdivision loop over a box.

    Each iteration takes the region with the largest error estimate, asks the local rule
    which of its axes is worth cutting, and bisects it along that axis, until the errors
    sum to less than the tolerance.

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

    state = _init_cubature_state(interval, shape, xtype, ytype, etype, max_nregion)

    def init_body(i, state):
        a = state["a_arr"][i]
        b = state["b_arr"][i]
        result, abserr, intabs, _, split = intfun(vfunc, a, b, ())

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

    def condfun(state):
        return (
            (state["status"] == STATUS.normal)
            & (0 <= state["err_sum"])
            & (state["err_bnd"] <= state["err_sum"])
        )

    def bodyfun(state):
        # Cut the region with the largest error estimate, along whichever of its axes
        # the local rule reported as worth cutting. `d_arr` is only ordinally
        # meaningful and local rules may build it differently, so `argmax` is all
        # that may be read from it.
        i = jnp.argmax(state["e_arr"])
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

        state["e_arr"] = place(state["e_arr"], error1, error2)
        state["r_arr"] = place(state["r_arr"], area1, area2)
        state["f_arr"] = place(state["f_arr"], intabs1, intabs2)
        state["d_arr"] = place(state["d_arr"], split1, split2)
        state["a_arr"] = place(state["a_arr"], a1, a2)
        state["b_arr"] = place(state["b_arr"], b1, b2)

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
        return state

    state = bounded_while_loop(condfun, bodyfun, state, max_nregion + 1)

    return jnp.sum(state["r_arr"], axis=0), state
