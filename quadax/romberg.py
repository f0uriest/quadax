"""Romberg integration aka adaptive trapezoid with Richardson extrapolation."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .adjoint import (
    AbstractAdjoint,
    DirectAdjoint,
    QuadratureOps,
    _ConvertedFunction,
    build_integrand,
    closure_convert,
)
from .utils import (
    QuadratureInfo,
    _pnorm,
    _real_dtype,
    bounded_while_loop,
    check_size,
    errorif,
    map_interval,
    resolve_dtypes,
    tanhsinh_transform,
    wrap_func,
)


@eqx.filter_jit
def romberg(
    fun: Callable[..., jax.Array],
    interval: ArrayLike,
    args: tuple = (),
    full_output: bool = False,
    epsabs: ArrayLike | None = None,
    epsrel: ArrayLike | None = None,
    divmax: int = 20,
    norm: float | int | Callable[[jax.Array], jax.Array] = jnp.inf,
    extrapolate: bool = True,
    adjoint: AbstractAdjoint = DirectAdjoint(),
    batch_size: int | None = None,
    divmin: int = 4,
):
    """Romberg integration of a callable function or method.

    Returns the integral of `fun` (a function of one variable) over `interval`.

    Good for non-smooth or piecewise smooth integrands.

    Not recommended for infinite intervals, or functions with singularities.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration. Use np.inf to denote infinite intervals.
        Its dtype sets the working precision: the integrand is called with an ``x`` of
        this dtype, and the result follows it unless the integrand upcasts. A integer
        types or python floats falls back to the JAX default. Must be real; complex
        integrands are supported, complex limits are not.
    args : tuple
        additional arguments passed to fun
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float
        Absolute and relative tolerances. The algorithm terminates once its estimate of
        the error in the current approximation ``I`` falls below ``max(epsabs,
        epsrel*|I|)``, which takes at least two refinement levels whatever the tolerance
        and ``divmin``, since the first level's estimate has no earlier one to be judged
        against. Default is the square root of the machine precision of the working
        dtype, ie of `interval`, or of the integrand's own dtype if that is the coarser
        of the two.
    divmax : int, optional
        Maximum order of extrapolation. Default is 20. Total number of function
        evaluations will be at most 2**divmax + 1
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    extrapolate : bool, optional
        Whether to accelerate convergence by Richardson extrapolation, which is what
        makes this Romberg's method rather than plain repeated bisection. On by default.
        Turning it off leaves the same nodes and the same halving schedule, reading the
        un-extrapolated estimate instead, which is worth having when the integrand is
        not smooth enough for the extrapolation's error expansion to hold. There it
        can amplify rather than cancel, and the honest estimate is the better one.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel.
        Defaults to ``2**divmin``, which is one batch for the whole starting grid and
        exactly one for the refinement level after it. Each refinement level doubles the
        number of new points, so raising this together with ``divmin`` is usually worth
        a lot on GPU/TPU, at the cost of peak memory scaling with it. Levels with fewer
        new points than one batch are padded up to a full batch, so a level costs
        ``batch_size`` evaluations however few points it places; that padding is what
        keeps a single batch shape traced for every level rather than one per level.
        Clipped to the largest number of points any one level places.
    divmin : int, optional
        Number of halvings the run starts from, default 4: it begins on a grid of
        ``2**divmin`` intervals rather than working up to one. Mirrors ``divmax``, and
        must not exceed it.

        That starting grid contains every coarser grid of the halving sequence, so the
        coarser rows of the table are filled from the same evaluations and no
        extrapolation is lost - the table is the one a run from ``divmin=0`` would have
        built by the time it reached the same mesh. What is bought is that the early
        work happens in one batch instead of a handful of levels evaluating a few points
        each, which is where the default schedule wastes time on GPU/TPU. What is paid
        is a floor of ``2**divmin + 1`` evaluations even on an integrand that needed
        fewer.

    Returns
    -------
    y  : float, Array
        Approximation to the integral
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation. Built from how far
          the estimate moved over the last few refinement levels rather than over the
          last one alone, plus the tail that movement's own contraction rate implies,
          and floored at the precision the integrand can be summed to.
        * neval : (int) Total number of function evaluations.
        * status : (int) Flag indicating reason for termination. status of 0 means
          normal termination, any other value indicates a possible error. A human
          readable message can be obtained by ``print(quadax.STATUS[status])``
        * info : (dict or None) Other information returned by the algorithm.
          Only present if ``full_output`` is True. Contains the following:

          * table : (ndarray, size(dixmax+1, divmax+1, ...)) Estimate of the integral
            from each level of discretization and each step of extrapolation. With
            ``extrapolate=False`` only the first column is filled.

    Notes
    -----
    The number of new points a refinement level places is only known at run time, so
    there is no single shape to vectorize over. Integrand evaluations are made in
    fixed size batches of ``batch_size``, defaulting to one at a time; raise it to get
    parallelism on GPU/TPU.

    """
    interval = jnp.atleast_1d(jnp.asarray(interval))
    if not jnp.issubdtype(interval.dtype, jnp.inexact):
        # integration limits must be inexact: they are differentiated with respect to,
        # and integer leaves would otherwise be treated as static metadata
        interval = interval.astype(jnp.result_type(float))
    errorif(
        len(interval) != 2,
        NotImplementedError,
        "Romberg integration with breakpoints not supported",
    )
    dtypes = resolve_dtypes(interval, fun, args)
    if epsabs is None:
        epsabs = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    if epsrel is None:
        epsrel = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    epsabs = jnp.asarray(epsabs, dtypes.etype)
    epsrel = jnp.asarray(epsrel, dtypes.etype)
    check_size(batch_size)
    errorif(
        not isinstance(divmin, (int, np.integer)) or divmin < 0,
        ValueError,
        f"divmin must be a non-negative integer, got {divmin}",
    )
    errorif(
        not isinstance(divmax, (int, np.integer)) or divmax < 0,
        ValueError,
        f"divmax must be a non-negative integer, got {divmax}",
    )
    errorif(
        divmin > divmax,
        ValueError,
        f"divmin must not exceed divmax, got divmin={divmin}, divmax={divmax}",
    )
    # The starting sweep places `2**divmin - 1` interior points and no later level
    # places more than `2**(divmax - 1)`, so a larger batch would only ever be padding.
    batch_size = min(batch_size or 2**divmin, max(2**divmin, 2 ** max(divmax - 1, 0)))
    if callable(norm):
        _norm: Callable[[jax.Array], jax.Array] = norm
    else:
        _norm: Callable[[jax.Array], jax.Array] = partial(_pnorm, p=norm)

    return _romberg(
        fun,
        interval,
        args,
        full_output,
        epsabs,
        epsrel,
        divmax,
        _norm,
        adjoint,
        build_integrand,
        dtypes.xtype,
        extrapolate,
        batch_size,
        divmin,
    )


def _romberg(
    fun,
    interval,
    args,
    full_output,
    epsabs,
    epsrel,
    divmax,
    _norm,
    adjoint,
    build,
    xtype,
    extrapolate,
    batch_size,
    divmin,
    truncation=None,
):
    """Shared driver for ``romberg`` and ``rombergts``.

    They differ in ``build``, and in whether that map truncates: ``truncation``
    estimates the mass left outside the range the solve integrates over, and is unset
    for the maps that leave none.
    """
    # Closure conversion has to happen on the user's function, before any wrapping:
    # once a transformed integrand crosses a filter_jit boundary its leaves become
    # tracers that closure_convert cannot hoist.
    f_conv, consts = closure_convert(fun, args, xtype)
    # Romberg has no subdivision to reuse, so `rebuild`/`on_mesh` are left unset and
    # DirectAdjoint falls back to differentiating through the loop.
    ops = QuadratureOps(
        build=partial(build, f_conv=f_conv),
        solve=partial(
            _romberg_solve,
            divmax=divmax,
            _norm=_norm,
            extrapolate=extrapolate,
            batch_size=batch_size,
            divmin=divmin,
            truncation=truncation,
        ),
        # Romberg has no subdivision to reuse, but it does settle on a number of
        # Richardson levels. Freezing that makes the result a fixed linear functional of
        # the integrand, which is what DirectAdjoint needs to differentiate in either
        # direction. It has to go through a custom primitive rather than being
        # differentiated directly, because evaluating it still involves a fori_loop with
        # dynamic bounds that JAX cannot reverse differentiate.
        frozen=lambda state: state["n"],
        frozen_solve=partial(
            _romberg_levels,
            divmax=divmax,
            extrapolate=extrapolate,
            batch_size=batch_size,
            divmin=divmin,
        ),
    )
    y, state = adjoint.quadrature(ops, None, interval, args, consts, epsabs, epsrel, {})
    info = state["table"] if full_output else None
    out = QuadratureInfo(state["err_sum"], state["neval"], state["status"], info)
    return y, out


def _build_tanhsinh(interval, args, consts, *, f_conv, safe=False):
    """Build the integrand for ``rombergts``: tanh-sinh, then map to the reference.

    See ``build_integrand`` for what ``safe`` buys and what it costs.
    """
    fun = _ConvertedFunction(f_conv, args, consts)
    fun_t, interval_t = tanhsinh_transform(fun, interval)
    fun_m, interval_m = map_interval(fun_t, interval_t)
    return wrap_func(fun_m, (), interval_m.dtype, safe=safe), interval_m


def _outermost(i, f, npts):
    """Magnitude of the first and last of the points a sweep places, as a stacked pair.

    The two sit nearest the ends of the range, which is what the truncation estimate
    reads. Picked out of whichever batch happens to hold them rather than evaluated
    again; the lanes that hold neither contribute zero to the sum over batches.
    """
    keep = lambda which: jnp.sum(
        jnp.where((i == which).reshape((-1,) + (1,) * (f.ndim - 1)), jnp.abs(f), 0),
        axis=0,
    )
    return jnp.stack([keep(1), keep(npts)])


def _level_sum(vfunc, a, h, npts, batch_size, shape, dtype, *, step=2):
    """Sum the integrand over ``npts`` nodes spaced ``step`` multiples of ``h`` apart.

    A refinement level adds the ``npts = m * 2**(k - 1)`` points sitting at odd
    multiples of ``h`` above ``a``, interleaving the nodes the previous levels already
    placed: ``step=2``, the default. The first level instead sums the interior nodes of
    the starting grid, consecutive multiples of its own step: ``step=1``. ``npts`` is
    only known at run time, so the points are evaluated in fixed size batches and the
    last batch is padded up to a full one. Padding rather than shaping each level to its
    own point count is what keeps one batch traced for all of them; the cost is that a
    level with fewer points than a batch still pays for a whole one.

    The padded lanes repeat the first point of their batch, which is always one this
    level genuinely evaluates, not an arbitrary placeholder. An integrand that is
    singular somewhere in the domain would return a non-finite value at a made up point,
    and masking that out of the sum afterwards still leaves a NaN in its derivative.

    ``vfunc`` takes a whole batch of abscissae; the callers wrap it to guarantee that.

    Returns the sum, the sum of ``abs`` of the same values, the magnitude of the
    outermost new node either side, and the number of batches.
    The middle two are what the error estimate's roundoff floor and its truncation term
    are built from; neither needs extra evaluations, so both are always accumulated and
    left to be eliminated as dead code on the paths that do not read them.
    """
    nbatch = (npts + batch_size - 1) // batch_size
    offs = jnp.arange(1, batch_size + 1)

    def bodyfun(j, carry):
        s, sabs, ends = carry
        i = j * batch_size + offs
        used = i <= npts
        x = a + h * (1 + step * (i - 1))
        x = jnp.where(used, x, x[0])
        f: jax.Array = vfunc(x)
        mask = used.reshape((-1,) + (1,) * (f.ndim - 1))
        f = jnp.where(mask, f, 0)
        return (
            s + jnp.sum(f, axis=0),
            sabs + jnp.sum(jnp.abs(f), axis=0),
            ends + _outermost(i, f, npts),
        )

    rzero = jnp.zeros(shape, _real_dtype(dtype))
    init = (jnp.zeros(shape, dtype), rzero, jnp.stack([rzero, rzero]))
    s, sabs, ends = jax.lax.fori_loop(0, nbatch, bodyfun, init)
    return s, sabs, ends, nbatch


def _initial_rows(vfunc, a, b, divmin, batch_size, shape, dtype):
    """Trapezoidal rule on every grid from 1 to ``2**divmin`` intervals, in one sweep.

    The ``2**divmin + 1`` point grid contains every coarser grid of the halving
    sequence: the points a level adds are those whose index has exactly as many factors
    of two as that level is coarse. So all ``divmin + 1`` rows of column 0 come out of a
    single pass over the finest grid, and the coarser rows cost no integrand evaluations
    at all. That is what separates ``divmin`` from starting the refinement on a finer
    mesh, which reaches the same finest row but leaves the table no coarser rows to
    extrapolate from.

    The rows are then built by the same halving recursion the refinement loop uses,
    rather than by summing each grid outright, so that the table does not depend on
    where a run started beyond the order its sums are accumulated in.

    Returns the rule indexed by row, the same rule applied to ``abs`` of the integrand
    on the finest of these grids, the magnitude of the integrand at the two endpoints
    and at the outermost interior node either side, and the number of evaluations spent.
    Only the finest row of the second is needed, since the refinement loop carries it
    forward by the same halving recursion. The last two are handed back rather than
    evaluated again because the truncation estimate reads exactly those nodes; see
    ``_tanhsinh_truncation``.
    """
    fa, fb = vfunc(a), vfunc(b)
    edges = jnp.stack([jnp.abs(fa), jnp.abs(fb)])
    fab = (jnp.abs(fa) + jnp.abs(fb)) / 2
    if divmin == 0:
        return (
            jnp.stack([(b - a) * (fa + fb) / 2]),
            (b - a) * fab,
            edges,
            jnp.zeros_like(edges),
            2,
        )

    npts = 2**divmin - 1  # interior points of the finest of these grids
    h = (b - a) / 2**divmin
    nbatch = (npts + batch_size - 1) // batch_size
    offs = jnp.arange(1, batch_size + 1)

    def bodyfun(k, carry):
        s, sabs, ends = carry
        i = k * batch_size + offs
        used = i <= npts
        x = a + h * i
        # padded lanes repeat a point this sweep genuinely evaluates; see `_level_sum`
        x = jnp.where(used, x, x[0])
        fx: jax.Array = vfunc(x)
        mask = used.reshape((-1,) + (1,) * (fx.ndim - 1))
        fx = jnp.where(mask, fx, 0)
        # Split a level at a time rather than against a (levels, batch) membership
        # matrix, which would multiply the peak memory of one batch by `divmin`.
        added = []
        for j in range(1, divmin + 1):
            stride = 2 ** (divmin - j)
            # the points level `j` adds: on its grid, but not on the one before it
            new = ((i % stride) == 0) & ((i % (2 * stride)) != 0)
            new = new.reshape((-1,) + (1,) * (fx.ndim - 1))
            added.append(jnp.sum(jnp.where(new, fx, 0), axis=0))
        # `abs` needs no such split: only the finest grid's value is ever read.
        return (
            s + jnp.stack(added),
            sabs + jnp.sum(jnp.abs(fx), axis=0),
            ends + _outermost(i, fx, npts),
        )

    rzero = jnp.zeros(shape, _real_dtype(dtype))
    init = (jnp.zeros((divmin, *shape), dtype), rzero, jnp.stack([rzero, rzero]))
    added, sabs, outer = jax.lax.fori_loop(0, nbatch, bodyfun, init)

    col0 = [(b - a) * (fa + fb) / 2]
    for j in range(1, divmin + 1):
        col0.append(0.5 * col0[j - 1] + ((b - a) / 2**j) * added[j - 1])
    resabs = ((b - a) / 2**divmin) * (sabs + fab)
    return jnp.stack(col0), resabs, edges, outer, 2 + nbatch * batch_size


def _extrapolate_row(result, n, extrapolate):
    """Fill row ``n``'s extrapolation columns from row ``n - 1``."""
    if not extrapolate:
        return result

    def mloop(col, result):
        # richardson extrapolation
        temp = 1 / (4.0**col - 1.0) * (result[n, col - 1] - result[n - 1, col - 1])
        return result.at[n, col].set(result[n, col - 1] + temp)

    return jax.lax.fori_loop(1, n + 1, mloop, result)


# Two levels have to be complete before convergence may be declared, because the first
# movement has nothing to be judged against. Without it the level 0 and level 1
# estimates agreeing by accident - because a narrow feature falls between the three
# points those levels use, or because the integrand is non-finite at an endpoint and
# both levels inherit the same substituted value - is read as convergence, and the
# routine returns an answer it never sampled with a status of 0. A ``divmin`` of 2 or
# more satisfies this on its own.
_MIN_LEVELS = 2
# Largest contraction ratio the tail term will extrapolate from, so a sequence that has
# stopped contracting is charged a bounded penalty - here 9x the last movement - rather
# than the infinite one the geometric series would give at a ratio of 1.
_TAIL_CAP = 0.9
# How far below the previous contraction ratio this one has to fall for the movement to
# count as an anomaly rather than a trend. Loose enough that a sequence whose ratio
# improves from level to level, which is what Richardson does on a smooth integrand, is
# still read as a trend.
_ANOMALY = 0.1


def _romberg_err(
    d: jax.Array,
    dprev: jax.Array,
    dprev2: jax.Array,
    resabs: jax.Array,
    _norm: Callable,
    eps: float,
    uflow: float,
):
    """Estimate the error in the level whose estimate moved by ``d``.

    ``d``, ``dprev`` and ``dprev2`` are the last three level-to-level movements of the
    reported estimate, newest first, normed to scalars; the two older ones are zero
    before there is any history. ``resabs`` is the same rule applied to ``abs`` of the
    integrand.

    The movement between successive levels measures how far the estimate last *went*,
    which is only a proxy for how far it still has *to go*. Two things make it a poor
    one, and each gets a correction:

    - Two levels can land close together on the way past a value neither has resolved,
      which reads as convergence but is a coincidence. What distinguishes the two is
      whether the movement follows the trend: a sequence in the regime its error
      expansion describes contracts by a roughly steady ratio, or by an improving one,
      whereas a coincidence shows up as a single ratio far below the one before it.
      Where the ratio collapses like that, the largest of the last three movements is
      reported instead, so that one lucky near-repeat cannot end the run.
    - Even a well behaved movement understates the total remaining, which is the whole
      geometric tail rather than its first term. Adding ``d * r / (1 - r)`` charges that
      tail: negligible for a rapidly contracting sequence, and growing without bound as
      the contraction stalls, which is where the movement alone is least informative.
      For the trapezoidal column, whose ratio settles at 1/4, the two together come to
      ``4/3`` of the movement, which is that column's error exactly.

    Finally the result is floored at ``50 * eps * resabs``, since no estimate is
    meaningful below the noise of evaluating and summing the integrand.
    """
    # Substituted rather than masked, in both ratios: the denominators are exactly zero
    # both before there is any history and whenever two levels agree to the last bit,
    # and dividing by them would put a NaN into the estimate by way of `inf * 0`.
    # A missing `dprev` is read as no contraction, and a missing `dprev2` as no trend to
    # depart from, which are the conservative readings of each.
    ratio = jnp.where(dprev > 0, d / jnp.where(dprev > 0, dprev, 1), _TAIL_CAP)
    prev_ratio = jnp.where(
        dprev2 > 0, dprev / jnp.where(dprev2 > 0, dprev2, 1), jnp.inf
    )
    anomaly = ratio < _ANOMALY * prev_ratio
    err = jnp.where(anomaly, jnp.maximum(d, jnp.maximum(dprev, dprev2)), d)

    ratio = jnp.minimum(ratio, _TAIL_CAP)
    err = err + d * ratio / (1 - ratio)

    # The floor covers the conditioning of the integrand rather than the summation:
    # abscissae carry `~eps*|x|`, which the integrand amplifies by `|f'|`. 50 is
    # QUADPACK's constant for the same quantity. The guard keeps the product from
    # underflowing to zero, which would make the floor a no-op precisely where the
    # integrand is smallest.
    absnorm = _norm(resabs)
    return jnp.where(
        absnorm > uflow / (50.0 * eps),
        jnp.maximum((50.0 * eps) * absnorm, err),
        err,
    )


def _tanhsinh_truncation(edges, outer, rtype, _norm):
    """Estimate the mass the tanh-sinh map leaves outside the range it is truncated to.

    ``rombergts`` integrates over ``[-tmax, tmax]``, and the substitution is a change of
    variable, so what it omits in ``t`` is exactly the sliver in ``x`` beyond the
    outermost node. That is a shortfall of the map rather than of the mesh: refining
    converges onto it, so the movement between levels says nothing about it at all.
    Left out of the estimate it is the one error a run can have while reporting success,
    which is why it is charged whatever the table is doing.

    Its size is read off the outermost term of the sum, which is the standard tanh-sinh
    indicator (``d4`` of [1]_ section 5). Past the cutoff the weight falls double
    exponentially, so the last term bounds what is left of the sum wherever the
    integrand is not itself growing faster than that.

    ``edges`` is the term at the cutoff and ``outer`` the term at the outermost node any
    level has placed and got a value from. The cutoff is preferred, and the fallback is
    what makes the estimate work where it is unusable: on an interval whose endpoints
    are large relative to its width the outermost nodes round onto the endpoint itself,
    and an integrand singular there returns a non-finite value that the wrapper masks
    away. Reading that as "no tail" is exactly backwards, it being the case where the
    tail is largest. Both ends are omitted, so both are charged.

    References
    ----------
    .. [1] Bailey, Jeyabalan and Li, "A comparison of three high-precision quadrature
       schemes", Experimental Mathematics 14.3 (2005).

    """
    term = jnp.where(edges > 0, edges, outer).astype(rtype)
    return _norm(term[0] + term[1])


def _romberg_solve(
    rule,
    vfunc,
    interval,
    epsabs,
    epsrel,
    kwargs,
    *,
    divmax,
    _norm,
    extrapolate=True,
    batch_size=1,
    divmin=4,
    truncation=None,
):
    """Run the refinement loop, with Richardson extrapolation if it is switched on.

    Without it this is plain adaptive bisection of the trapezoidal rule (or of the
    tanh-sinh rule, for ``rombergts``): the same nodes and the same halving schedule,
    reading column 0 of the table rather than choosing among its extrapolations.
    """
    del rule, kwargs
    a, b = interval
    # Vectorize whatever we were handed The primal integrand arrives vectorized already,
    # but the adjoints solve against one they build per point (the tangent or the
    # cotangent of the mapped integrand) which takes a scalar only. Wrapping here rather
    # than mapping at each use keeps one contract for the loop below: ``vfunc`` accepts
    # a batch of abscissae.
    vfunc = wrap_func(vfunc, (), interval.dtype)
    f = jax.eval_shape(vfunc, (a + b) / 2)
    rtype = _real_dtype(f.dtype)
    # Compile time constants, as python floats rather than arrays of the working dtype:
    # forming `uflow / (50 * eps)` in half precision is a needless underflow risk.
    eps = float(jnp.finfo(rtype).eps)
    uflow = float(jnp.finfo(rtype).tiny)

    # Which entry of row `k` is the estimate. Richardson's is the diagonal, having
    # applied `k` rounds of extrapolation to the trapezoidal values in column 0; without
    # it the estimate is that column, and the rest of the table is never written.
    best = (lambda res, k: res[k, k]) if extrapolate else (lambda res, k: res[k, 0])

    result = jnp.zeros((divmax + 1, divmax + 1, *f.shape), f.dtype)
    col0, resabs, edges, outer, neval = _initial_rows(
        vfunc, a, b, divmin, batch_size, f.shape, f.dtype
    )
    result = result.at[: divmin + 1, 0].set(col0)

    def trunc_of(outer):
        """What the map leaves outside the range the solve integrates over.

        Zero for the variants whose map truncates nothing, which is every one but
        tanh-sinh.
        """
        if truncation is None:
            return jnp.array(0.0, rtype)
        return truncation(edges, outer, rtype, _norm)

    def keep_outer(outer, ends):
        """Take the newer pair of outermost nodes, for whichever end got a value.

        Each level places its outermost nodes closer to the ends than the last did, so a
        value from this level supersedes the one carried. A zero does not: it is a node
        that has rounded onto the endpoint and been masked away, and the last node that
        resolved remains the best that end has to say.
        """
        return jnp.where(ends > 0, ends, outer)

    def total_err(d, dprev, dprev2, resabs, outer):
        """The error in the current estimate.

        What the mesh has left to give, plus what the map never had. The two are
        independent shortfalls, so they add.
        """
        return _romberg_err(d, dprev, dprev2, resabs, _norm, eps, uflow) + trunc_of(
            outer
        )

    def advance(result, n, yprev):
        """Extrapolate row ``n`` and measure how far its estimate moved."""
        result = _extrapolate_row(result, n, extrapolate)
        y = best(result, n)
        return result, y, _norm(y - yprev)

    def unconverged(y, err, nlevels):
        """Whether the run has to keep going.

        Either the estimate is still above tolerance, or there are not yet enough
        levels for it to rest on; see ``_MIN_LEVELS``.
        """
        return (err > jnp.maximum(epsabs, epsrel * _norm(y))) | (nlevels < _MIN_LEVELS)

    # Rows 1 through `divmin` came out of the sweep above, so they are processed
    # unconditionally: their evaluations are already spent, and running them is what
    # gives the run a history before the first refinement rather than after it. Only
    # the movements are kept, since the error estimate reads the last three of them and
    # anything it would have said about an intermediate row is never looked at.
    def initloop(k, carry):
        result, yprev, d, dprev, _ = carry
        result, y, dnew = advance(result, k, yprev)
        return result, y, dnew, d, dprev

    # Explicitly typed rather than left weak python floats: these are loop carries, and
    # have to match what `_norm` writes back into them. Real, because the error in a
    # complex valued integral is still real. The movements start at zero, which
    # `_romberg_err` reads as "no history yet".
    zero = jnp.array(0.0, rtype)
    result, y, d, dprev, dprev2 = jax.lax.fori_loop(
        1, divmin + 1, initloop, (result, result[0, 0], zero, zero, zero)
    )
    err = total_err(d, dprev, dprev2, resabs, outer)
    # A run given no rows to compare has no estimate at all, whatever the table says:
    # with no movements the formula would return its roundoff floor, which reads as
    # convergence on the strength of a single trapezoidal value.
    if divmin < 1:
        err = jnp.array(jnp.inf, rtype)

    state = (result, divmin + 1, neval, err, y, resabs, d, dprev, outer)

    def ncond(state):
        result, n, neval, err, y, resabs, dprev, dprev2, outer = state
        # `n` is the row about to be computed, so `n - 1` rows are complete and `y` is
        # the value read off the last of them.
        tol = jnp.maximum(epsabs, epsrel * _norm(y))
        trunc = trunc_of(outer)
        # Refining is pointless once the mesh has converged onto a floor the map itself
        # holds above the tolerance: every further level doubles the evaluations while
        # the reported error stays where it is. Since the two shares are added,
        # `err <= 2 * trunc` is the mesh's own share having fallen to that floor. The
        # run still ends unconverged, which is the honest outcome; this only declines to
        # spend the rest of `divmax` reaching it.
        # A zero tolerance is not a threshold that can be missed, it is a request to
        # refine as far as `divmax` allows, so it is never read as unreachable.
        hopeless = (
            (tol > 0) & (trunc > tol) & (err <= 2 * trunc) & (n - 1 >= _MIN_LEVELS)
        )
        return (n < divmax + 1) & unconverged(y, err, n - 1) & ~hopeless

    def nloop(state):
        # loop over outer number of subdivisions
        result, n, neval, err, yprev, resabs, dprev, dprev2, outer = state
        h = (b - a) / 2**n
        s, sabs, ends, nbatch = _level_sum(
            vfunc, a, h, 2 ** (n - 1), batch_size, f.shape, f.dtype
        )
        result = result.at[n, 0].set(0.5 * result[n - 1, 0] + h * s)
        resabs = 0.5 * resabs + h * sabs
        outer = keep_outer(outer, ends)
        # The padded lanes of the last batch are evaluations of the integrand like any
        # other, so they are counted here even though they do not reach the sum.
        neval += nbatch * batch_size
        result, y, d = advance(result, n, yprev)
        err = total_err(d, dprev, dprev2, resabs, outer)
        return result, n + 1, neval, err, y, resabs, d, dprev, outer

    result, n, neval, err, y, resabs, dprev, dprev2, outer = bounded_while_loop(
        ncond, nloop, state, max(divmax - divmin, 0) + 1
    )

    status = 2 * unconverged(y, err, n - 1)
    state = {
        "table": result,
        "err_sum": err,
        "neval": neval,
        "status": status,
        "n": n,  # Richardson levels used; frozen by DirectAdjoint
    }
    return y, state


def _romberg_levels(
    rule,
    vfunc,
    interval,
    n,
    kwargs,
    *,
    divmax,
    extrapolate=True,
    batch_size=1,
    divmin=4,
):
    """Evaluate the table at a fixed number of levels.

    With ``n`` fixed this is a fixed linear combination of the integrand at fixed nodes,
    so its forward and reverse derivatives are exact transposes of one another. Mirrors
    the schedule in ``_romberg_solve`` exactly so the two agree, including which entry
    of the table is read.
    """
    del rule, kwargs
    a, b = interval[0], interval[-1]
    vfunc = wrap_func(vfunc, (), interval.dtype)  # see ``_romberg_solve``
    f = jax.eval_shape(vfunc, (a + b) / 2)
    result = jnp.zeros((divmax + 1, divmax + 1, *f.shape), f.dtype)
    col0, *_ = _initial_rows(vfunc, a, b, divmin, batch_size, f.shape, f.dtype)
    result = result.at[: divmin + 1, 0].set(col0)
    result = jax.lax.fori_loop(
        1, divmin + 1, lambda k, res: _extrapolate_row(res, k, extrapolate), result
    )

    def nloop(k, result):
        h = (b - a) / 2**k
        s, *_ = _level_sum(vfunc, a, h, 2 ** (k - 1), batch_size, f.shape, f.dtype)
        result = result.at[k, 0].set(0.5 * result[k - 1, 0] + h * s)
        return _extrapolate_row(result, k, extrapolate)

    result = jax.lax.fori_loop(divmin + 1, n, nloop, result)
    return result[n - 1, n - 1] if extrapolate else result[n - 1, 0]


@eqx.filter_jit
def rombergts(
    fun: Callable[..., jax.Array],
    interval: ArrayLike,
    args: tuple = (),
    full_output: bool = False,
    epsabs: ArrayLike | None = None,
    epsrel: ArrayLike | None = None,
    divmax: int = 20,
    norm: float | int | Callable[[jax.Array], jax.Array] = jnp.inf,
    extrapolate: bool = True,
    adjoint: AbstractAdjoint = DirectAdjoint(),
    batch_size: int | None = None,
    divmin: int = 4,
):
    """Romberg integration with tanh-sinh (aka double exponential) transformation.

    Returns the integral of `fun` (a function of one variable) over `interval`.

    Performs well for functions with singularities at the endpoints or integration
    over infinite intervals. May be slightly less efficient than ``quadgk`` or
    ``quadcc`` for smooth integrands.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration. Use np.inf to denote infinite intervals.
        Its dtype sets the working precision: the integrand is called with an ``x`` of
        this dtype, and the result follows it unless the integrand upcasts. A integer
        types or python floats falls back to the JAX default. Must be real; complex
        integrands are supported, complex limits are not.
    args : tuple
        additional arguments passed to fun
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float
        Absolute and relative tolerances. The algorithm terminates once its estimate of
        the error in the current approximation ``I`` falls below ``max(epsabs,
        epsrel*|I|)``, which takes at least two refinement levels whatever the tolerance
        and ``divmin``, since the first level's estimate has no earlier one to be judged
        against. Default is the square root of the machine precision of the working
        dtype, ie of `interval`, or of the integrand's own dtype if that is the coarser
        of the two.
    divmax : int, optional
        Maximum order of extrapolation. Default is 20. Total number of function
        evaluations will be at most 2**divmax + 1
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    extrapolate : bool, optional
        Whether to accelerate convergence by Richardson extrapolation, which is what
        makes this Romberg's method rather than plain repeated bisection. On by default.
        Turning it off leaves the same nodes and the same halving schedule, reading the
        un-extrapolated estimate instead, which is worth having when the integrand is
        not smooth enough for the extrapolation's error expansion to hold. There it
        can amplify rather than cancel, and the honest estimate is the better one.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel.
        Defaults to ``2**divmin``, which is one batch for the whole starting grid and
        exactly one for the refinement level after it. Each refinement level doubles the
        number of new points, so raising this together with ``divmin`` is usually worth
        a lot on GPU/TPU, at the cost of peak memory scaling with it. Levels with fewer
        new points than one batch are padded up to a full batch, so a level costs
        ``batch_size`` evaluations however few points it places; that padding is what
        keeps a single batch shape traced for every level rather than one per level.
        Clipped to the largest number of points any one level places.
    divmin : int, optional
        Number of halvings the run starts from, default 4: it begins on a grid of
        ``2**divmin`` intervals rather than working up to one. Mirrors ``divmax``, and
        must not exceed it.

        That starting grid contains every coarser grid of the halving sequence, so the
        coarser rows of the table are filled from the same evaluations and no
        extrapolation is lost - the table is the one a run from ``divmin=0`` would have
        built by the time it reached the same mesh. What is bought is that the early
        work happens in one batch instead of a handful of levels evaluating a few points
        each, which is where the default schedule wastes time on GPU/TPU. What is paid
        is a floor of ``2**divmin + 1`` evaluations even on an integrand that needed
        fewer.

    Returns
    -------
    y  : float, Array
        Approximation to the integral
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation. Built from how far
          the estimate moved over the last few refinement levels rather than over the
          last one alone, plus the tail that movement's own contraction rate implies,
          and floored at the precision the integrand can be summed to.
        * neval : (int) Total number of function evaluations.
        * status : (int) Flag indicating reason for termination. status of 0 means
          normal termination, any other value indicates a possible error. A human
          readable message can be obtained by ``print(quadax.STATUS[status])``
        * info : (dict or None) Other information returned by the algorithm.
          Only present if ``full_output`` is True. Contains the following:

          * table : (ndarray, size(dixmax+1, divmax+1, ...)) Estimate of the integral
            from each level of discretization and each step of extrapolation. With
            ``extrapolate=False`` only the first column is filled.

    Notes
    -----
    The number of new points a refinement level places is only known at run time, so
    there is no single shape to vectorize over. Integrand evaluations are made in
    fixed size batches of ``batch_size``, defaulting to one at a time; raise it to get
    parallelism on GPU/TPU.

    """
    interval = jnp.atleast_1d(jnp.asarray(interval))
    if not jnp.issubdtype(interval.dtype, jnp.inexact):
        # integration limits must be inexact: they are differentiated with respect to,
        # and integer leaves would otherwise be treated as static metadata
        interval = interval.astype(jnp.result_type(float))
    errorif(
        len(interval) != 2,
        NotImplementedError,
        "tanh-sinh transformation with breakpoints not supported",
    )
    dtypes = resolve_dtypes(interval, fun, args)
    if epsabs is None:
        epsabs = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    if epsrel is None:
        epsrel = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    epsabs = jnp.asarray(epsabs, dtypes.etype)
    epsrel = jnp.asarray(epsrel, dtypes.etype)
    check_size(batch_size)
    errorif(
        not isinstance(divmin, (int, np.integer)) or divmin < 0,
        ValueError,
        f"divmin must be a non-negative integer, got {divmin}",
    )
    errorif(
        divmin > divmax,
        ValueError,
        f"divmin must not exceed divmax, got divmin={divmin}, divmax={divmax}",
    )
    # The starting sweep places `2**divmin - 1` interior points and no later level
    # places more than `2**(divmax - 1)`, so a larger batch would only ever be padding.
    batch_size = min(batch_size or 2**divmin, max(2**divmin, 2 ** max(divmax - 1, 0)))
    if callable(norm):
        _norm: Callable[[jax.Array], jax.Array] = norm
    else:
        _norm: Callable[[jax.Array], jax.Array] = partial(_pnorm, p=norm)

    return _romberg(
        fun,
        interval,
        args,
        full_output,
        epsabs,
        epsrel,
        divmax,
        _norm,
        adjoint,
        _build_tanhsinh,
        dtypes.xtype,
        extrapolate,
        batch_size,
        divmin,
        truncation=_tanhsinh_truncation,
    )
