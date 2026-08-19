"""Adjoint methods controlling how derivatives of quadrature are computed."""

import abc
from collections.abc import Callable, Sequence
from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.internal import unvmap_any
from jax._src import core as jcore
from jax.extend.core import Primitive
from jax.flatten_util import ravel_pytree
from jax.interpreters import ad, batching, mlir

from . import _acceleration
from .fixed_order import AbstractQuadratureRule
from .utils import (
    _real_dtype,
    check_size,
    map_interval,
    tree_where,
    wrap_func,
)


class _ConvertedFunction(eqx.Module):
    """Closure-converted integrand, with args and hoisted constants bound."""

    f_conv: Callable
    args: tuple
    consts: tuple

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.f_conv(x, self.args, *self.consts)


def closure_convert(fun, args, xtype):
    """Hoist values closed over by ``fun`` so that they are visible to AD.

    Custom derivative rules only see their explicit arguments. Anything ``fun`` closes
    over would otherwise silently get a zero gradient, so pull it out into ``consts``
    and pass it in explicitly.

    ``xtype`` is the dtype the abscissa will be carried at. It matters here rather
    than only downstream because ``closure_convert`` traces ``fun`` to a jaxpr at the
    dtype it is given, and that jaxpr is what every later evaluation of the integrand
    goes through.
    """
    f_conv, consts = jax.closure_convert(
        lambda x, args_: fun(x, *args_), jnp.zeros((), xtype), args
    )
    return f_conv, tuple(consts)


def build_integrand(interval, args, consts, *, f_conv, safe=False):
    """Map the integrand to the reference domain and wrap it for vectorization.

    ``safe`` asks for an integrand whose inf/nan mask survives being transposed. It
    costs a second evaluation, so it is requested only for the evaluations that are
    actually differentiated, not for the primal solve which, for every adjoint that
    carries a custom rule, is not differentiated at all.
    """
    fun = _ConvertedFunction(f_conv, args, consts)
    fun_mapped, interval_t = map_interval(fun, interval)
    return wrap_func(fun_mapped, (), interval_t.dtype, safe=safe), interval_t


class QuadratureOps(NamedTuple):
    """Primitive operations that an adjoint composes to build a quadrature.

    This is an internal plumbing object, constructed by ``adaptive_quadrature`` and
    ``romberg``. All fields are static (they close over no traced values), so they may
    be passed to a custom derivative rule as a non-differentiable keyword argument.

    Parameters
    ----------
    build : callable
        ``build(interval, args, consts, safe=False) -> (vfunc, interval_t)``. Maps the
        integrand to the reference domain and wraps it for vectorized evaluation.
        ``safe`` asks for an inf/nan mask that survives reverse mode, which the
        adjoints request for the evaluations they differentiate.
    solve : callable
        ``solve(rule, vfunc, interval_t, epsabs, epsrel, kwargs) -> (y, state)``. Runs
        the full (adaptive) quadrature.
    rebuild : callable or None
        ``rebuild(interval_t, state) -> (a_arr, b_arr)``. Rebuilds the subdivision from
        ``interval_t`` using the owner/fraction bookkeeping recorded in ``state``, so
        that the mesh carries correct derivatives with respect to the integration
        limits. ``None`` for methods that have no subdivision (e.g. Romberg).
    on_mesh : callable or None
        ``on_mesh(rule, vfunc, a_arr, b_arr, kwargs) -> y``. Applies the local rule on a
        given subdivision. ``None`` for methods that have no subdivision.
    frozen : callable or None
        ``frozen(state) -> discretization``. Extracts whatever the primal solve settled
        on, for methods whose fixed-discretization evaluation JAX cannot differentiate
        in reverse (Romberg, whose level loop has dynamic bounds). Set together with
        ``frozen_solve``; when both are set and there is no subdivision to rebuild,
        ``DirectAdjoint`` routes through a custom primitive instead of differentiating
        the evaluation directly.
    frozen_solve : callable or None
        ``frozen_solve(rule, vfunc, interval_t, discretization, kwargs) -> y``.
        Evaluates the quadrature on a fixed discretization.
    mesh_is_primal : bool
        Whether the value the solve returns is the sum over the subdivision. False when
        convergence acceleration may return an extrapolated value instead; the
        adjoints then take their derivative from ``frozen_solve``, which reproduces the
        extrapolation as well as the mesh, rather than from ``on_mesh``.
    """

    build: Callable
    solve: Callable
    rebuild: Callable | None = None
    on_mesh: Callable | None = None
    frozen: Callable | None = None
    frozen_solve: Callable | None = None
    mesh_is_primal: bool = True


def _with_options(ops, checkpoint, chunk_size):
    """Thread the adjoint's own options down to the fixed-subdivision quadrature.

    Only the operations that evaluate a fixed subdivision take these; ``solve`` is the
    primal adaptive loop and is unaffected by either.
    """
    opts = {"checkpoint": checkpoint, "chunk_size": chunk_size}
    upd = {}
    if ops.on_mesh is not None:
        upd["on_mesh"] = partial(ops.on_mesh, **opts)
    if ops.rebuild is not None and ops.frozen_solve is not None:
        upd["frozen_solve"] = partial(ops.frozen_solve, **opts)
    return ops._replace(**upd) if upd else ops


# ---------------------------------------------------------------------------------
# Fixed-discretization evaluation for the adaptive methods.
#
# These are the ``QuadratureOps`` fields that only the adjoints call: rebuilding the
# subdivision the primal solve settled on as a smooth function of the integration
# limits, and evaluating the quadrature on it. The primal solve never uses them, it
# builds the subdivision as it goes.


def _rebuild_mesh(interval, frozen):
    """Rebuild the subdivision from `interval`, as a function of the integration limits.

    Bisection never crosses a breakpoint, so every sub-interval stays inside whichever
    of the original sub-intervals it was carved out of, at a fixed dyadic fraction of
    the way along it. The primal loop records that owner and those fractions, which are
    exactly the parts that do not vary smoothly. Rebuilding the mesh is then a gather
    and a rescale, no loop, and no dependence on how many bisections were performed,
    while still letting the mesh move when a limit or a breakpoint moves.
    """
    owner, frac_a, frac_b = frozen
    lo = interval[owner]
    hi = interval[owner + 1]
    width = hi - lo
    return lo + frac_a * width, lo + frac_b * width


# Default for the adjoints' ``chunk_size``: how many sub-intervals of a fixed
# subdivision are evaluated at once. Evaluating them all together is fastest but makes
# peak memory scale with ``max_ninter``, which is a safety bound users tend to set
# generously; evaluating one at a time streams but serializes. Measured on a scalar
# integrand with an order 21 rule, 8 is where the curve turns over: it matches
# one-at-a-time peak memory at large ``max_ninter`` while being noticeably faster in
# reverse mode, and larger blocks buy little more speed for a lot more memory. That
# measurement is for one shape of problem, which is why this is a default and not a
# constant.
_CHUNK = 8


def _block_mesh(rule, vfunc, a_arr, b_arr, chunk_size):
    """Group a fixed subdivision into blocks of sub-intervals ready for evaluation.

    Returns the blocked endpoints and mask to scan over, a function evaluating one
    block, the shape and dtype of one sub-interval's contribution, and how many slots
    the subdivision has.
    """
    # Sub-intervals are independent, so they are evaluated in blocks: ``vmap`` within a
    # block, ``scan`` across blocks. A plain ``scan`` over every sub-interval would make
    # a gradient cost ``max_ninter`` rather than the number of sub-intervals actually
    # used, because reverse mode stacks residuals for every iteration whether or not it
    # did any work. A plain ``vmap`` fixes that but materializes the whole subdivision
    # at once.

    # Slots past the end of the subdivision are empty (``a == b``). A block with no used
    # slot in it is skipped entirely with a ``cond``, which is what keeps the cost of a
    # derivative tracking the sub-intervals the solve actually used rather than
    # ``max_ninter``; the solve fills slots from the front, so the used blocks are the
    # leading ones. The predicate is reduced with ``unvmap_any`` so that it stays a
    # scalar under ``vmap`` and the skip survives batching, at the cost of a block being
    # evaluated for every batch element as soon as one of them needs it.

    # Within a block the empty slots are still handed a real sub-interval and masked out
    # afterwards rather than skipped: a per-slot ``cond`` sits inside the ``vmap``,
    # where a batched ``cond`` becomes a ``select`` carrying a ``stop_gradient`` that
    # cannot be transposed. Substituting a real sub-interval also stops an integrand
    # that is singular somewhere in the mapped domain from poisoning the unused slots
    # with a NaN that the mask would then propagate. So the granularity of the skip is
    # ``chunk_size``.
    used = a_arr != b_arr
    a_safe = jnp.where(used, a_arr, a_arr[0])
    b_safe = jnp.where(used, b_arr, b_arr[0])

    nslot = a_arr.shape[0]
    chunk = min(chunk_size, nslot)
    pad = -nslot % chunk
    reshape = lambda x, fill: jnp.pad(x, (0, pad), constant_values=fill).reshape(
        -1, chunk
    )

    apply1 = lambda a, b: rule._apply(vfunc, a, b, ())
    sds = jax.eval_shape(apply1, a_arr[0], b_arr[0])
    # The mask multiplies the *values*, so it takes their (real) dtype rather than the
    # mesh's. With the mesh at float64 and the values at float32 the latter would
    # otherwise be promoted straight back to float64 here.
    blocks = (
        reshape(a_safe, a_arr[0]),
        reshape(b_safe, b_arr[0]),
        reshape(used.astype(_real_dtype(sds.dtype)), 0.0),
    )

    def evaluate(block):
        """One block's masked contributions, zeros if no slot in it is used."""
        a, b, m = block

        def run(_):
            y = jax.vmap(apply1)(a, b)
            return y * m.reshape((-1,) + (1,) * (y.ndim - 1))

        # `unvmap_any` keeps the predicate a scalar under `vmap`, so the block is
        # skipped whenever *no* batch element uses it instead of degrading to a select
        # that evaluates every block. No inner per-element gate is needed: `m` already
        # zeroes the slots an individual element does not use.
        return jax.lax.cond(
            unvmap_any(jnp.any(m != 0)),
            run,
            lambda _: jnp.zeros((chunk, *sds.shape), sds.dtype),
            None,
        )

    return blocks, evaluate, sds, nslot


def _checkpointed(bodyfun, checkpoint):
    """Recompute a scan body during the backward pass rather than storing it.

    Without this reverse mode keeps the integrand's value at every node of every
    sub-interval, which dominates its memory; recomputing them trades a second pass over
    the integrand for that storage.
    """
    return jax.checkpoint(bodyfun) if checkpoint else bodyfun


def _quad_on_mesh(
    rule, vfunc, a_arr, b_arr, kwargs, *, checkpoint=False, chunk_size=_CHUNK
):
    """Apply the local rule on a fixed subdivision and sum the contributions."""
    del kwargs
    blocks, evaluate, sds, _ = _block_mesh(rule, vfunc, a_arr, b_arr, chunk_size)

    def bodyfun(total, block):
        return total + jnp.sum(evaluate(block), axis=0), None

    total, _ = jax.lax.scan(
        _checkpointed(bodyfun, checkpoint), jnp.zeros(sds.shape, sds.dtype), blocks
    )
    return total


def _values_on_mesh(
    rule, vfunc, a_arr, b_arr, kwargs, *, checkpoint=False, chunk_size=_CHUNK
):
    """Apply the local rule on a fixed subdivision, keeping the contributions separate.

    As ``_quad_on_mesh``, except that the per-sub-interval values are returned rather
    than summed, because the replay has to recombine them in more than one way.
    """
    del kwargs
    blocks, evaluate, sds, nslot = _block_mesh(rule, vfunc, a_arr, b_arr, chunk_size)

    def bodyfun(carry, block):
        return carry, evaluate(block)

    _, values = jax.lax.scan(_checkpointed(bodyfun, checkpoint), None, blocks)
    return values.reshape(-1, *sds.shape)[:nslot]


def _frozen_mesh(state):
    """The parts of the subdivision that do not vary smoothly with the limits."""
    return (state["owner"], state["frac_a"], state["frac_b"])


def _mesh_solve(
    rule, vfunc, interval, frozen, kwargs, *, checkpoint=False, chunk_size=_CHUNK
):
    """Quadrature on the subdivision implied by `frozen`, as a function of interval."""
    a_arr, b_arr = _rebuild_mesh(interval, frozen)
    return _quad_on_mesh(
        rule, vfunc, a_arr, b_arr, kwargs, checkpoint=checkpoint, chunk_size=chunk_size
    )


class _ReplayRecord(NamedTuple):
    """What an accelerated solve settled on, enough to reproduce it differentiably.

    Every field is integer or boolean apart from the fractions, so nothing here carries
    a derivative: these are exactly the decisions the primal made, frozen. Each is
    carried under its own name in the integrator state.

    ``mesh`` is the final subdivision, as for a plain solve. ``parents`` describes the
    sub-intervals that no longer exist -- each was bisected, so each is the *parent* of
    one step -- and the birth times record when every sub-interval entered and left the
    running total, which is what lets the whole sequence of running totals be rebuilt.
    Both the parent arrays and the birth times are indexed by the slot the bisection
    created, which is unique to that step, so the step needs no separate counter.
    """

    owner: jax.Array
    frac_a: jax.Array
    frac_b: jax.Array
    birth: jax.Array
    p_owner: jax.Array
    p_frac_a: jax.Array
    p_frac_b: jax.Array
    p_birth: jax.Array
    append_mask: jax.Array
    accel_ncall: jax.Array
    used_accel: jax.Array

    @property
    def mesh(self):
        """Frozen description of the final subdivision, for ``_rebuild_mesh``."""
        return (self.owner, self.frac_a, self.frac_b)

    @property
    def parents(self):
        """Frozen description of the bisected sub-intervals, for ``_rebuild_mesh``."""
        return (self.p_owner, self.p_frac_a, self.p_frac_b)


def _frozen_replay(state):
    """The parts of an accelerated solve that do not vary smoothly with the limits."""
    return _ReplayRecord(**{name: state[name] for name in _ReplayRecord._fields})


def _replay_solve(
    rule, vfunc, interval, frozen, kwargs, *, checkpoint=False, chunk_size=_CHUNK
):
    """Re-run an accelerated quadrature on the decisions the primal settled on.

    An accelerated solve may return an extrapolated value rather than the sum over the
    subdivision, so differentiating it means differentiating the extrapolation as well
    as the mesh. Everything the acceleration decided -- which sub-interval to bisect,
    when to feed the table, which extrapolation to keep -- was settled on error
    estimates and is integer or boolean, so freezing it leaves a fixed, ordinary
    function of the limits and the integrand: rebuild the subdivision, rebuild the
    sequence of running totals, and run the epsilon algorithm over it again.

    Rebuilding the running totals is the part that is not simply a mesh sum. The total
    at the point where ``t`` sub-intervals exist is the sum over those alive then, and a
    coarse sub-interval's value is not the sum of the values of the two halves it was
    cut into, so it is not a prefix sum of the final subdivision. Recording
    when each sub-interval entered the total and when it left turns it into one instead:
    add each value at its birth, subtract it again at its death, and the running totals
    are the cumulative sum. Every sub-interval that ever existed is either in the final
    subdivision or was bisected, so evaluating the final subdivision and the parents
    covers all of them, and costs the same number of rule evaluations as the primal.
    """
    mesh = _rebuild_mesh(interval, frozen.mesh)
    parents = _rebuild_mesh(interval, frozen.parents)
    values = _values_on_mesh(
        rule,
        vfunc,
        jnp.concatenate([mesh[0], parents[0]]),
        jnp.concatenate([mesh[1], parents[1]]),
        kwargs,
        checkpoint=checkpoint,
        chunk_size=chunk_size,
    )
    nslot = mesh[0].shape[0]
    v_mesh, v_parent = values[:nslot], values[nslot:]
    shape, ytype = values.shape[1:], values.dtype

    # Births and deaths, as a signed contribution at each point on the timeline. A
    # sub-interval of the final subdivision never dies. A parent dies at the step that
    # bisected it, which is the step that created slot `n`, so at `n + 1`. Unused slots
    # carry a zero value and cannot disturb either sum.
    n_init = interval.shape[0] - 1
    timeline = jnp.zeros((nslot + 2, *shape), ytype)
    timeline = timeline.at[frozen.birth].add(v_mesh)
    timeline = timeline.at[frozen.p_birth].add(v_parent)
    timeline = timeline.at[jnp.arange(nslot) + 1].add(-v_parent)
    running = jnp.cumsum(timeline, axis=0)

    # The subsequence that was actually fed to the table, gathered into fixed positions.
    # The initial total seeds it, as in the primal, and the appends follow in order;
    # everything else is parked in a slot that is never read.
    unused = nslot + 1
    position = jnp.cumsum(frozen.append_mask)
    sequence = jnp.zeros((nslot + 2, *shape), ytype).at[0].set(running[n_init])
    sequence = sequence.at[jnp.where(frozen.append_mask, position, unused)].set(
        running[jnp.arange(nslot) + 1]
    )

    table = _acceleration.append(_acceleration.init_table(shape, ytype), sequence[0])

    def call(j, table):
        fed = _acceleration.step(table, sequence[j], rule.norm)
        # Stop at the extrapolation the primal kept, which is the last one that improved
        # on the one before it. Later calls happened, but their results were discarded.
        return tree_where(j <= frozen.accel_ncall, fed, table)

    table = jax.lax.fori_loop(1, nslot + 1, call, table)
    return jnp.where(frozen.used_accel, table.result, running[nslot])


def _zero_tangent(tree):
    """Zero tangent for a pytree, materialized rather than symbolic.

    Nothing in the integrator state is differentiable, so every leaf gets an explicit
    zero: an ordinary zeros array for the inexact leaves, and a ``float0`` array (the
    tangent type of a non-inexact primal, and zero-sized) for the integer and boolean
    bookkeeping. Returning ``None`` for the latter would be more natural but this causes
    problems with jax 0.7.0 and 0.7.1.
    """

    def z(x):
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.inexact):
            return jnp.zeros_like(x)
        return np.zeros(x.shape, dtype=jax.dtypes.float0)

    return jax.tree.map(z, tree)


def _fill_tangents(dyn, tangents):
    """Replace ``None`` tangents (undifferentiated arguments) with explicit zeros.

    ``eqx.filter_custom_jvp`` hands us ``None`` in place of the tangent of anything that
    is not being differentiated (possibly deep inside a pytree) but ``jax.jvp``
    requires a tangent for every primal, so fill those in with zeros.
    """
    is_none = lambda x: x is None
    dyn_leaves, treedef = jax.tree.flatten(dyn, is_leaf=is_none)
    tan_leaves = jax.tree.flatten(tangents, is_leaf=is_none)[0]
    filled = [
        None if d is None else (jnp.zeros_like(d) if t is None else t)
        for d, t in zip(dyn_leaves, tan_leaves)
    ]
    return jax.tree.unflatten(treedef, filled)


class AbstractAdjoint(eqx.Module):
    """Abstract base class for adjoint methods.

    An adjoint determines *how* derivatives of a quadrature are computed, without
    changing what the quadrature itself returns. Subclasses implement ``quadrature``,
    which runs the primal solve and attaches whatever custom differentiation rule it
    wants.

    See the Adjoints section of the API documentation for the adjoints quadax ships and
    how to choose between them.
    """

    @abc.abstractmethod
    def quadrature(
        self,
        ops: QuadratureOps,
        rule: AbstractQuadratureRule | None,
        interval: jax.Array,
        args: tuple,
        consts: tuple,
        epsabs: jax.Array,
        epsrel: jax.Array,
        kwargs: dict,
    ) -> tuple[jax.Array, dict]:
        """Evaluate the quadrature and define how it is differentiated.

        Parameters
        ----------
        ops : QuadratureOps
            Primitive operations for this quadrature method.
        rule : AbstractQuadratureRule
            Local quadrature rule. Ignored by methods that do not use one.
        interval : jax.Array
            Limits of integration with possible breakpoints, in the original
            (unmapped) coordinates.
        args : tuple
            Extra arguments to the integrand.
        consts : tuple
            Values closed over by the integrand, hoisted out by
            ``jax.closure_convert`` so that they are visible to AD.
        epsabs, epsrel : jax.Array
            Absolute and relative error tolerances.
        kwargs : dict
            Additional keyword arguments passed to ``rule``.

        Returns
        -------
        y : jax.Array
            The value of the integral.
        state : dict
            Full state of the integrator.
        """


class _UnrolledDirectAdjoint(AbstractAdjoint):
    """Differentiate by unrolling the quadrature loop, with no custom rule.

    This is the original quadax behavior. It is kept, private and unexported, as the
    reference implementation that :class:`DirectAdjoint` is tested against. It is
    correct but expensive: reverse mode must store residuals for every iteration of the
    loop, including iterations that did no work.
    """

    def quadrature(self, ops, rule, interval, args, consts, epsabs, epsrel, kwargs):
        """Evaluate the quadrature, differentiating straight through the loop."""
        vfunc, interval_t = ops.build(interval, args, consts, safe=True)
        return ops.solve(rule, vfunc, interval_t, epsabs, epsrel, kwargs)


class DirectAdjoint(AbstractAdjoint):
    """Differentiate the quadrature exactly, reusing the converged subdivision.

    Commonly called "discretize then optimize": the quadrature is discretized first, and
    the derivative is then taken of that discretization.

    This is the default, and is the cheaper option for a cheap integrand in either mode.
    When one evaluation of the integrand is expensive, :class:`LeibnizAdjoint` can be an
    order of magnitude faster or more; see the Adjoints section of the API documentation
    for the trade-offs.

    It works by running the primal solve recording the final adaptive mesh, and then
    using the same mesh (with corrections when differentiating the interval itself) to
    integrate the derivative of the integrand.

    The derivative therefore inherits the subdivision chosen for the integral, and no
    error control of its own is paid for. That is the reason to reach for a Leibniz
    adjoint when the derivative needs resolving that the integral did not pay for, and
    the reason a derivative costs roughly what the converged subdivision costs however
    generous ``max_ninter`` was.

    Methods with no subdivision to reuse (Romberg) are handled the same way in spirit:
    the number of Richardson levels the solve settled on is frozen instead of a mesh.
    There the fixed discretization still contains a ``fori_loop`` with dynamic bounds
    that JAX cannot reverse differentiate, so the derivative is routed through a custom
    primitive that supplies the two directions explicitly. Both modes work either way.

    Parameters
    ----------
    checkpoint : bool
        Whether to recompute the quadrature on each block of sub-intervals during the
        backward pass rather than storing it, off by default. Without it reverse mode
        keeps the integrand's value at every node of every sub-interval, which dominates
        its memory and grows with ``max_ninter`` however few sub-intervals are really
        used. Turning checkpoint on can reduce memory by 3x to 30x depending on how
        large ``max_ninter`` is at the expense of additional integrand evaluations. No
        effect in forward mode.
    chunk_size : int
        How many sub-intervals of the frozen subdivision to evaluate at once: ``vmap``
        within a chunk, ``scan`` across chunks. Trades peak memory against speed, and is
        the reverse-mode counterpart of the quadrature routines' ``batch_size``, which
        bounds the work *within* one sub-interval. The two multiply, so a gradient
        evaluates the integrand at up to ``chunk_size`` times ``batch_size`` points at a
        time. Lower it when a derivative runs out of memory and ``checkpoint`` was not
        enough; raise it when the subdivision is small and the scan is pure overhead.


    Notes
    -----
    When differentiating a moving jump or singularity, mark the jump or singularity
    with a breakpoint, and build that breakpoint from the same parameter that positions
    the feature. Marking a feature that is genuinely there is never worse than leaving
    it unmarked, and for derivatives it is frequently the difference between a correct
    answer and a silently wrong one. The one way marking can hurt is marking something
    that is not there, which is the third case below.

    Differentiating a jump gives a delta, which no quadrature of the integrand's tangent
    can represent, so it is recovered from the motion of the breakpoint instead, and
    that works only when the breakpoint moves with the discontinuity::

        step = lambda t, z: jnp.where(t > z[0], 1.0, 0.0)   # jumps at t = z[0]

        # correct: one parameter `s` positions both the jump and the breakpoint
        f = lambda s: quadgk(step, jnp.array([-1.0, s, 1.0]), (jnp.array([s]),))[0]
        jax.grad(f)(0.3)            # -1.0

    Two versions that look equivalent are not. Both return the same primal *value*,
    0.7, and both give zero where the derivative is ``-1``::

        # WRONG: the jump is unmarked, so nothing tracks it
        f = lambda s: quadgk(step, jnp.array([-1.0, 1.0]), (jnp.array([s]),))[0]

        # WRONG: marked, but with a constant, which carries no derivative. Tempting,
        # because it does help the primal value: it cuts the subdivision from 13
        # sub-intervals to 2.
        f = lambda s: quadgk(step, jnp.array([-1.0, 0.3, 1.0]), (jnp.array([s]),))[0]

    Splitting the feature across two parameters is what the requirement rules out. The
    derivative with respect to a breakpoint on its own is not a well posed question:
    the integral is the same whatever the mesh is cut at, so the breakpoint only means
    anything in combination with the integrand it is marking. Ask for it anyway, by
    differentiating with respect to ``[breakpoint, jump location]`` at ``[0.3, 0.3]``,
    and the answer is ``[-1, 0]``. The total over the two is right, and how it is
    divided between them is an artifact of having written one feature as two
    parameters, not a property of the quadrature.

    A moving *singularity* needs the same treatment, for the same reason: unmarked, it
    slides across a subdivision that does not move with it.

    Only a feature that is steep but *continuous* needs none of this: a sharp ``tanh``
    differentiates correctly unmarked under either adjoint. Marking one anyway is
    harmless, the breakpoint then only helps the subdivision, and contributes nothing
    to the derivative since the integrand has the same limit from both sides of it.
    """

    checkpoint: bool = False
    chunk_size: int = _CHUNK

    def __post_init__(self):
        check_size(self.chunk_size, "chunk_size")

    def quadrature(self, ops, rule, interval, args, consts, epsabs, epsrel, kwargs):
        """Evaluate the quadrature and differentiate it on the converged subdivision."""
        ops = _with_options(ops, self.checkpoint, self.chunk_size)
        if ops.rebuild is None:
            if ops.frozen_solve is not None:
                # No subdivision, but a discretization we can freeze (Romberg). Route
                # through the primitive so that both modes work.
                return _leibniz(
                    rule,
                    interval,
                    args,
                    consts,
                    epsabs,
                    epsrel,
                    kwargs,
                    ops=ops,
                    freeze=True,
                )
            vfunc, interval_t = ops.build(interval, args, consts, safe=True)
            return ops.solve(rule, vfunc, interval_t, epsabs, epsrel, kwargs)
        return _direct(rule, interval, args, consts, epsabs, epsrel, kwargs, ops=ops)


@eqx.filter_custom_jvp
def _direct(rule, interval, args, consts, epsabs, epsrel, kwargs, *, ops):
    vfunc, interval_t = ops.build(interval, args, consts)
    return ops.solve(rule, vfunc, interval_t, epsabs, epsrel, kwargs)


@_direct.def_jvp
def _direct_jvp(primals, tangents, *, ops):
    y, state = _direct(*primals, ops=ops)

    # Everything the mesh depends on that is *not* smooth (which interval was bisected
    # at each step, and how the two halves were ordered) is integer valued and was
    # already decided by the primal solve. Freeze it, then rebuild the mesh as a smooth
    # function of the limits so that moving a breakpoint moves the mesh with it.
    dyn, static = eqx.partition(primals, eqx.is_inexact_array)
    dyn_t = _fill_tangents(dyn, tangents)

    # When the limits are not being differentiated the converged subdivision can be used
    # as-is. filter_custom_jvp gives a `None` tangent for anything not being
    # differentiated, so this is known at trace time.
    interval_perturbed = any(t is not None for t in jax.tree.leaves(tangents[1]))

    def fixed_mesh(dyn_):
        rule_, interval_, args_, consts_, _, _, kwargs_ = eqx.combine(dyn_, static)
        vfunc, interval_t = ops.build(interval_, args_, consts_, safe=True)
        if not ops.mesh_is_primal:
            # Convergence acceleration may have returned an extrapolated value instead
            # of the mesh sum, so the mesh alone is not what was differentiated. The
            # frozen evaluation replays the extrapolation as well, and rebuilds the
            # subdivision from the limits whether or not they are being perturbed --
            # unlike the mesh sum it needs the sub-intervals that no longer exist, which
            # were never stored as endpoints.
            return ops.frozen_solve(
                rule_, vfunc, interval_t, ops.frozen(state), kwargs_
            )
        if interval_perturbed:
            a_arr, b_arr = ops.rebuild(interval_t, ops.frozen(state))
        else:
            a_arr, b_arr = state["a_arr"], state["b_arr"]
        return ops.on_mesh(rule_, vfunc, a_arr, b_arr, kwargs_)

    y_dot = jax.jvp(fixed_mesh, (dyn,), (dyn_t,))[1]
    return (y, state), (y_dot, _zero_tangent(state))


# ---------------------------------------------------------------------------------
# Leibniz adjoint.
#
# JAX allows a function to carry a custom JVP rule or a custom VJP rule, but not both,
# The way around that is to put the *tangent* map in a primitive of its own. The
# integral itself stays an ordinary custom_jvp (it is not linear in the parameters), but
# its JVP rule emits this primitive, which is linear in the tangent and carries an
# explicit transpose rule. JAX then gets forward mode from the JVP and reverse mode by
# transposing it, and each direction runs the solve it actually needs: a scalar
# integrand contracted with the tangent going forwards, the vector-valued adjoint
# integrand coming back.
_leibniz_p = Primitive("quadax_leibniz_tangent")


def _leibniz_unpack(flat, n, treedef, static, frozen_treedef):
    """Split the operands into (tangent, differentiable primal, primal, frozen)."""
    # ``treedef`` is the authority on which leaves are differentiable. It was recorded
    # when the primitive was bound, and the tangent, the residuals and the cotangents
    # all use it, so the three stay in step by construction.

    # Recovering that structure from the operand *values* instead -- filtering the
    # reassembled primals for inexact arrays -- looks equivalent but is not. An operand
    # that was a concrete python float at bind time (a tolerance passed as
    # ``epsabs=1e-8`` rather than left to default) becomes a jaxpr literal, and
    # ``backward_pass`` hands literals back as raw python floats. Filtering drops them,
    # and the transpose rule then returns fewer cotangents than the primitive has linear
    # operands, which JAX reports as "foreach() argument 2 is shorter than argument 1".
    # Hence ``jnp.asarray``: the residuals are all inexact by construction, so promoting
    # a literal back to an array just undoes the unwrapping.

    tangent = jax.tree.unflatten(treedef, list(flat[:n]))
    dyn = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in flat[n : 2 * n]])
    frozen = (
        None
        if frozen_treedef is None
        else jax.tree.unflatten(frozen_treedef, list(flat[2 * n :]))
    )
    return tangent, dyn, eqx.combine(dyn, static), frozen


def _run_solve(ops, rule, integrand, interval_t, epsabs, epsrel, kwargs, frozen):
    """Adaptive solve, or evaluation on a frozen discretization if there is one."""
    if frozen is None:
        return ops.solve(rule, integrand, interval_t, epsabs, epsrel, kwargs)[0]
    return ops.frozen_solve(rule, integrand, interval_t, frozen, kwargs)


# The ratio between successive probe offsets used to read a breakpoint's one-sided
# limits, and how closely two estimates of one have to agree to be believed.
#
# The ratio only has to be wide enough that an unbounded integrand changes visibly
# across it: inverse-square-root decay over a factor of 16 quarters the value, which no
# tolerance below would accept. The tolerance is an agreement test between two estimates
# that are equal in exact arithmetic whenever the limit exists, so it only has to clear
# rounding noise, and the estimates cancel the integrand's slope rather than tolerating
# it, so it does not have to make room for steepness. That last point is what keeps the
# constants from implying a largest believable slope: tolerating the slope instead would
# reject a genuine jump sitting on one of about 1e9 in double precision, and on one of
# 100 in single.
#
# What is left is a precision floor rather than a modelling choice. Once the integrand's
# values are large enough that rounding them swamps the jump, the estimates stop
# agreeing and the jump is dropped rather than guessed at, which needs a slope beyond
# about 1e12 in double precision for a jump of order one, and about 1e4 in single.
_JUMP_RATIO = 16
_JUMP_RTOL = 1e-3


def _side_limit(probes: Sequence[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """A one-sided limit from three probes, and whether it exists.

    ``probes`` are the integrand at three separations from a breakpoint, each ``r``
    times the last. Close to the breakpoint the integrand is its limit plus a term
    linear in the separation, from its slope, so the Richardson combination of two
    probes cancels the slope and leaves the limit. Two such combinations are formed and
    they agree whenever that model holds, however steep the slope. Where the integrand
    is unbounded instead they disagree, which is the test for the limit existing at all.
    """
    v1, v2, v3 = probes
    r = _JUMP_RATIO
    first = (r * v1 - v2) / (r - 1)
    second = (r * v2 - v3) / (r - 1)
    scale = jnp.maximum(jnp.abs(first), jnp.abs(second))
    converged = (
        jnp.isfinite(first)
        & jnp.isfinite(second)
        & (jnp.abs(first - second) <= _JUMP_RTOL * scale)
    )
    return first, converged


def _breakpoint_jumps(vfunc, interval_t):
    """The integrand's jump across each interior breakpoint, or zero where it has none.

    Moving a breakpoint moves the boundary between two adjoining sub-integrals, which
    contributes ``f(c-) - f(c+)`` per unit of motion. That is zero unless the integrand
    jumps at ``c``, and a jump the integrand defines by a comparison on the abscissa
    resolves within one ulp of it, so the one-sided values are read a few ulp either
    side.

    Reading them once is not enough. An integrand *unbounded* at ``c`` makes both values
    enormous, and their difference is then the asymmetry between the two sides amplified
    by the singularity rather than a jump: ``2/sqrt|t - c|`` on one side against
    ``1/sqrt|t - c|`` on the other gives 1e8 where the answer is zero. A steep but
    perfectly ordinary slope contaminates the reading too, in proportion to how far out
    it was taken.

    So the jump is estimated twice, by ``_side_limit``, in a way that cancels the
    slope exactly and leaves a singular contribution behind, and it is believed only
    when the two estimates agree. Everything continuous comes out at zero, a genuine
    jump survives however steep the integrand is around it, and a jump superposed on a
    symmetric singularity survives as well, that singularity contributing equally to
    both sides and cancelling on its own.

    The precision of the arithmetic is the floor: once the integrand's values are large
    enough that rounding them swamps the jump, the estimates stop agreeing and the jump
    is dropped rather than guessed at. In double precision that needs a slope above
    about 1e12 with a jump of order one.

    Returns ``None`` when there are no interior breakpoints, which is a static property
    of the interval.
    """
    c = interval_t[1:-1]
    if c.shape[0] == 0:
        return None
    # The probe width is one ulp of the breakpoint, computed arithmetically rather than
    # with `nextafter`. Two reasons: `nextafter` has no differentiation rule in JAX, so
    # it breaks second derivatives with respect to the limits outright, the primal still
    # having to be traced through it; and it steps by the true spacing, which is
    # asymmetric at a binade boundary, where the ulp below is half the ulp above. The
    # span floors it so that a breakpoint at zero still gets an offset on the scale of
    # the domain.
    #
    # Frozen: the width of a probe that exists to be infinitesimal is a property of the
    # arithmetic, not of the problem, and carries no meaningful derivative. The probe
    # points are still built from ``c``, so they move with the breakpoint and the jump
    # keeps its dependence on where the breakpoint sits.
    eps = float(jnp.finfo(c.dtype).eps)
    span = jnp.abs(interval_t[-1] - interval_t[0])
    h = jax.lax.stop_gradient(eps * jnp.maximum(jnp.abs(c), span))
    offsets = (1, _JUMP_RATIO, _JUMP_RATIO**2)
    below = [vfunc(c - m * h) for m in offsets]
    above = [vfunc(c + m * h) for m in offsets]
    left, left_ok = _side_limit(below)
    right, right_ok = _side_limit(above)
    # A side with no limit contributes nothing, which is what a singularity pinned to an
    # outer limit does as well: the solve returns the finite part and the boundary term
    # that was regularized away must not be added back. The other side still counts, so
    # the two are taken separately rather than as one difference.
    separate = jnp.where(left_ok, left, 0.0) - jnp.where(right_ok, right, 0.0)
    # Only when *neither* side has a limit is the difference the sole hope: a jump
    # sitting on a symmetric singularity has unbounded one-sided values whose singular
    # parts are equal, so they cancel in the difference and leave the jump behind.
    joint, joint_ok = _side_limit([lo - hi for lo, hi in zip(below, above)])
    return jnp.where(left_ok | right_ok, separate, jnp.where(joint_ok, joint, 0.0))


def _endpoint_term(vfunc, interval_t, *, ops, static):
    """The boundary half of the Leibniz rule, as a function of the primals.

    Differentiating ``int_a^b f`` gives an integral of ``df``, plus the boundary term
    ``f(b) db - f(a) da``. The solve supplies the first half by integrating the tangent
    between fixed limits, so whatever dependence on the limits survives ``ops.build``
    (that is, whatever ends up in ``interval_t`` rather than folded into the integrand)
    is missing from it and has to be added back.

    An interior breakpoint is the same statement one level down: it is the upper limit
    of the sub-integral below it and the lower limit of the one above, so moving it
    contributes ``(f(c-) - f(c+)) dc``, which is zero unless the integrand jumps there.
    Summed over the whole interval the sub-integrals telescope and only the two outer
    limits and the jumps survive. See :func:`_breakpoint_jumps` for the one-sided
    values.

    Whether the outer terms survive depends on the mapping. ``tanhsinh_transform`` and
    the mappings for an infinite interval both hand back a fixed domain, so they are
    identically zero and cost only two evaluations of the integrand. A finite interval
    left alone by ``map_interval`` is the case that needs them: the mapping is the
    identity, so the limits are exactly where the whole derivative lives.

    A singularity pinned to a limit is why the outer values are read *at* the limit
    rather than one ulp inside it, unlike the breakpoints. The value comes back
    non-finite, the inf/nan mask sends it to zero, and zero is right: the divergent
    boundary term has already been regularized away by the solve, which returns the
    finite part, so adding a large stand-in for the infinity would double-count it.

    The integrand is held at its primal value here, so only the limits are
    differentiated and the term comes out as the ``f(b) db - f(a) da`` above. Whatever
    is left of the chain rule (how ``interval_t`` depends on the original limits,
    including the reordering ``map_interval`` does for reversed ones) is left to AD.
    """
    lo, hi = vfunc(interval_t[0]), vfunc(interval_t[-1])
    jumps = _breakpoint_jumps(vfunc, interval_t)

    def term(dyn_):
        _, interval, args, consts, _, _ = eqx.combine(dyn_, static)
        _, limits = ops.build(interval, args, consts)
        out = hi * limits[-1] - lo * limits[0]
        if jumps is not None:
            out = out + jnp.tensordot(limits[1:-1], jumps, axes=(0, 0))
        return out

    return term


def _integrand_at(dyn_, *, t, ops, static):
    """The mapped integrand evaluated at ``t``, as a function of the primals.

    ``ops.build`` folds the limits, the arguments and the closed-over constants into the
    integrand, so all of them reach the value at a point through here. Both directions
    differentiate this: forward mode pushes a tangent through it, reverse mode pulls a
    cotangent back.
    """
    _, interval, args, consts, _, _ = eqx.combine(dyn_, static)
    vf, _ = ops.build(interval, args, consts, safe=True)
    return vf(t)


def _tangent_integrand(dyn, dyn_t, *, ops, static):
    """The integrand's tangent along ``dyn_t``, as a function of the abscissa."""

    def dvfunc(t):
        at_t = partial(_integrand_at, t=t, ops=ops, static=static)
        return jax.jvp(at_t, (dyn,), (dyn_t,))[1]

    return dvfunc


def _without_interval(tree):
    """Zero out the limits' component of a tangent or cotangent, keeping the rest.

    The limits sit at position 1 of the primal tuple. Zeroing rather than dropping keeps
    the pytree structure the solve and ``ravel_pytree`` expect.
    """
    return (tree[0], jax.tree.map(jnp.zeros_like, tree[1]), *tree[2:])


def _leibniz_impl(
    *flat,
    ops,
    n,
    treedef,
    static,
    kwargs_items,
    frozen_treedef,
    freeze,
    interval_from_solve,
    out_sds,
):
    """Forward direction: integrate the tangent of the mapped integrand."""
    dyn_t, dyn, primals, frozen = _leibniz_unpack(
        flat, n, treedef, static, frozen_treedef
    )
    rule, interval, args, consts, epsabs, epsrel = primals
    kwargs = dict(kwargs_items)
    vfunc, interval_t = ops.build(interval, args, consts)

    del out_sds
    y_dot = _run_solve(
        ops,
        rule,
        _tangent_integrand(dyn, dyn_t, ops=ops, static=static),
        interval_t,
        epsabs,
        epsrel,
        kwargs,
        frozen if freeze else None,
    )
    if interval_from_solve:
        # Integrating the tangent between fixed limits misses the boundary term whenever
        # the limits themselves carry a derivative, which is exactly when the solve is
        # the thing that has to produce it.
        term = _endpoint_term(vfunc, interval_t, ops=ops, static=static)
        y_dot = y_dot + jax.jvp(term, (dyn,), (dyn_t,))[1]
    return y_dot


def _leibniz_transpose(
    ct,
    *flat,
    ops,
    n,
    treedef,
    static,
    kwargs_items,
    frozen_treedef,
    freeze,
    interval_from_solve,
    out_sds,
):
    """Reverse direction: integrate the cotangent of the mapped integrand."""
    del out_sds
    _, dyn, primals, frozen = _leibniz_unpack(flat, n, treedef, static, frozen_treedef)
    rule, interval, args, consts, epsabs, epsrel = primals
    kwargs = dict(kwargs_items)
    vfunc, interval_t = ops.build(interval, args, consts)
    _, unravel = ravel_pytree(dyn)

    def adjoint_integrand(t):
        at_t = partial(_integrand_at, t=t, ops=ops, static=static)
        _, vjp = jax.vjp(at_t, dyn)
        ct_dyn = vjp(ct)[0]
        if not interval_from_solve:
            # Drop the limits' components before they reach the solve. Nobody asked for
            # them, so integrating them buys nothing, and it costs, because they are the
            # components that misbehave: differentiating an integral whose integrand is
            # unbounded at a limit gives an unbounded adjoint integrand, and the error
            # control is driven by `rule.norm` over the whole raveled vector, so one
            # divergent component sets the mesh for every component. Convergence
            # acceleration couples them harder still, since the epsilon table's
            # structural decisions are made on the norm as well. Forward mode zeroes the
            # tangent's limits the same way, see `_leibniz_impl`; without this the two
            # modes are not solving the same problem.
            ct_dyn = _without_interval(ct_dyn)
        return ravel_pytree(ct_dyn)[0]

    flat_ct = _run_solve(
        ops,
        rule,
        adjoint_integrand,
        interval_t,
        epsabs,
        epsrel,
        kwargs,
        frozen if freeze else None,
    )
    ct_tree = unravel(flat_ct)
    if interval_from_solve:
        # The adjoint integrand carries the limits' cotangent only through the
        # integrand, so the boundary term is added here, as in forward mode.
        term = _endpoint_term(vfunc, interval_t, ops=ops, static=static)
        ct_tree = jax.tree.map(jnp.add, ct_tree, jax.vjp(term, dyn)[1](ct)[0])
    ct_leaves = jax.tree.flatten(ct_tree)[0]
    # cotangents for the linear operands, then None for every residual operand
    n_res = len(flat) - n
    return tuple(ct_leaves) + (None,) * n_res


def _leibniz_batch(args, dims, **params):
    return jax.vmap(partial(_leibniz_impl, **params), in_axes=tuple(dims))(*args), 0


_leibniz_p.def_impl(_leibniz_impl)
_leibniz_p.def_abstract_eval(
    lambda *a, out_sds, **kw: jcore.ShapedArray(out_sds.shape, out_sds.dtype)
)
mlir.register_lowering(
    _leibniz_p, mlir.lower_fun(_leibniz_impl, multiple_results=False)
)
ad.primitive_transposes[_leibniz_p] = _leibniz_transpose
batching.primitive_batchers[_leibniz_p] = _leibniz_batch


class LeibnizAdjoint(AbstractAdjoint):
    r"""Differentiate by the Leibniz rule, either mode, with its own error control.

    The derivative is evaluated with its own adaptive solve, so it gets its own error
    control rather than inheriting the subdivision chosen for the integral, see the
    Adjoints section of the API documentation for when that is worth paying for.

    Because each mode picks its own subdivision, forward and reverse results agree to
    quadrature accuracy rather than exactly.

    Notes
    -----
    When differentiating a moving jump or singularity, mark the jump or singularity
    with a breakpoint, and build that breakpoint from the same parameter that positions
    the feature. Marking a feature that is genuinely there is never worse than leaving
    it unmarked, and for derivatives it is frequently the difference between a correct
    answer and a silently wrong one. The one way marking can hurt is marking something
    that is not there, which is the third case below.

    Differentiating a jump gives a delta, which no quadrature of the integrand's tangent
    can represent, so it is recovered from the motion of the breakpoint instead, and
    that works only when the breakpoint moves with the discontinuity::

        step = lambda t, z: jnp.where(t > z[0], 1.0, 0.0)   # jumps at t = z[0]

        # correct: one parameter `s` positions both the jump and the breakpoint
        f = lambda s: quadgk(step, jnp.array([-1.0, s, 1.0]), (jnp.array([s]),))[0]
        jax.grad(f)(0.3)            # -1.0

    Two versions that look equivalent are not. Both return the same primal *value*,
    0.7, and both give zero where the derivative is ``-1``::

        # WRONG: the jump is unmarked, so nothing tracks it
        f = lambda s: quadgk(step, jnp.array([-1.0, 1.0]), (jnp.array([s]),))[0]

        # WRONG: marked, but with a constant, which carries no derivative. Tempting,
        # because it does help the primal value: it cuts the subdivision from 13
        # sub-intervals to 2.
        f = lambda s: quadgk(step, jnp.array([-1.0, 0.3, 1.0]), (jnp.array([s]),))[0]

    Splitting the feature across two parameters is what the requirement rules out. The
    derivative with respect to a breakpoint on its own is not a well posed question:
    the integral is the same whatever the mesh is cut at, so the breakpoint only means
    anything in combination with the integrand it is marking. Ask for it anyway, by
    differentiating with respect to ``[breakpoint, jump location]`` at ``[0.3, 0.3]``,
    and the answer is ``[-1, 0]``. The total over the two is right, and how it is
    divided between them is an artifact of having written one feature as two
    parameters, not a property of the quadrature.

    A moving *singularity* needs the same treatment, for the same reason: unmarked, it
    slides across a subdivision that does not move with it.

    Only a feature that is steep but *continuous* needs none of this: a sharp ``tanh``
    differentiates correctly unmarked under either adjoint. Marking one anyway is
    harmless, the breakpoint then only helps the subdivision, and contributes nothing
    to the derivative since the integrand has the same limit from both sides of it.
    """

    def quadrature(self, ops, rule, interval, args, consts, epsabs, epsrel, kwargs):
        """Evaluate the quadrature, differentiating it by the Leibniz rule."""
        return _leibniz(
            rule, interval, args, consts, epsabs, epsrel, kwargs, ops=ops, freeze=False
        )


@eqx.filter_custom_jvp
def _leibniz(
    rule, interval, args, consts, epsabs, epsrel, kwargs, *, ops, freeze=False
):
    del freeze
    vfunc, interval_t = ops.build(interval, args, consts)
    return ops.solve(rule, vfunc, interval_t, epsabs, epsrel, kwargs)


@_leibniz.def_jvp
def _leibniz_jvp(primals, tangents, *, ops, freeze=False):
    y, state = _leibniz(*primals, ops=ops, freeze=freeze)
    # `kwargs` is a dict, and primitive parameters have to be hashable, so it travels
    # separately from the pytree that is flattened into the primitive's operands.
    kwargs = primals[-1]
    dyn, static = eqx.partition(primals[:-1], eqx.is_inexact_array)
    dyn_t = _fill_tangents(dyn, tangents[:-1])

    lin_leaves, treedef = jax.tree.flatten(dyn_t)
    res_leaves = jax.tree.flatten(dyn)[0]
    # Whether the solve is the thing that has to produce the limits' cotangent, which it
    # is whenever they are being differentiated at all. `filter_custom_jvp` tells us at
    # trace time by handing us a `None` tangent for anything that is not.
    interval_from_solve = any(t is not None for t in jax.tree.leaves(tangents[1]))
    if freeze:
        frozen_leaves, frozen_treedef = jax.tree.flatten(
            jax.lax.stop_gradient(ops.frozen(state))
        )
    else:
        frozen_leaves, frozen_treedef = [], None
    y_dot = _leibniz_p.bind(
        *lin_leaves,
        *res_leaves,
        *frozen_leaves,
        ops=ops,
        n=len(lin_leaves),
        treedef=treedef,
        static=static,
        kwargs_items=tuple(sorted(kwargs.items())),
        frozen_treedef=frozen_treedef,
        freeze=freeze,
        interval_from_solve=interval_from_solve,
        out_sds=jax.ShapeDtypeStruct(jnp.shape(y), jnp.result_type(y)),
    )
    return (y, state), (y_dot, _zero_tangent(state))
