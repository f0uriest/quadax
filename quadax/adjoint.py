"""Adjoint methods controlling how derivatives of quadrature are computed."""

import abc
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jcore
from jax.extend.core import Primitive
from jax.flatten_util import ravel_pytree
from jax.interpreters import ad, batching, mlir

from .fixed_order import AbstractQuadratureRule
from .utils import map_interval, wrap_func


class _ConvertedFunction(eqx.Module):
    """Closure-converted integrand, with args and hoisted constants bound."""

    f_conv: Callable
    args: tuple
    consts: tuple

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.f_conv(x, self.args, *self.consts)


def closure_convert(fun, args):
    """Hoist values closed over by ``fun`` so that they are visible to AD.

    Custom derivative rules only see their explicit arguments. Anything ``fun`` closes
    over would otherwise silently get a zero gradient, so pull it out into ``consts``
    and pass it in explicitly.
    """
    f_conv, consts = jax.closure_convert(
        lambda x, args_: fun(x, *args_), jnp.array(0.0), args
    )
    return f_conv, tuple(consts)


def build_integrand(interval, args, consts, *, f_conv):
    """Map the integrand to the reference domain and wrap it for vectorization."""
    fun = _ConvertedFunction(f_conv, args, consts)
    fun_mapped, interval_t = map_interval(fun, interval)
    return wrap_func(fun_mapped, ()), interval_t


class QuadratureOps(NamedTuple):
    """Primitive operations that an adjoint composes to build a quadrature.

    This is an internal plumbing object, constructed by ``adaptive_quadrature`` and
    ``romberg``. All fields are static (they close over no traced values), so they may
    be passed to a custom derivative rule as a non-differentiable keyword argument.

    Parameters
    ----------
    build : callable
        ``build(interval, args, consts) -> (vfunc, interval_t)``. Maps the integrand to
        the reference domain and wraps it for vectorized evaluation.
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
        ``frozen_solve``; when both are set ``DirectAdjoint`` routes through a custom
        primitive instead of differentiating the evaluation directly.
    frozen_solve : callable or None
        ``frozen_solve(rule, vfunc, interval_t, discretization, kwargs) -> y``.
        Evaluates the quadrature on a fixed discretization.
    """

    build: Callable
    solve: Callable
    rebuild: Callable | None = None
    on_mesh: Callable | None = None
    frozen: Callable | None = None
    frozen_solve: Callable | None = None


def _with_checkpoint(ops, checkpoint):
    """Thread the checkpointing choice down to the fixed-subdivision quadrature."""
    upd = {}
    if ops.on_mesh is not None:
        upd["on_mesh"] = partial(ops.on_mesh, checkpoint=checkpoint)
    if ops.rebuild is not None and ops.frozen_solve is not None:
        upd["frozen_solve"] = partial(ops.frozen_solve, checkpoint=checkpoint)
    return ops._replace(**upd) if upd else ops


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

    This is the original quadax behaviour. It is kept, private and unexported, as the
    reference implementation that :class:`DirectAdjoint` is tested against. It is
    correct but expensive: reverse mode must store residuals for every iteration of the
    loop, including iterations that did no work.
    """

    def quadrature(self, ops, rule, interval, args, consts, epsabs, epsrel, kwargs):
        """Evaluate the quadrature, differentiating straight through the loop."""
        vfunc, interval_t = ops.build(interval, args, consts)
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
        backward pass rather than storing it. Without it reverse mode keeps the
        integrand's value at every node of every sub-interval, which dominates its
        memory and grows with ``max_ninter`` however few sub-intervals are really used;
        recomputing cuts that by around 3x at the default budget and by 30x or more
        when ``max_ninter`` is generous, at no measured cost in speed, so it is on by
        default. Turning it off has not been found to pay for itself even on integrands
        costing megaflops per evaluation, so treat it mainly as a diagnostic knob. No
        effect in forward mode.

    """

    checkpoint: bool = True

    def quadrature(self, ops, rule, interval, args, consts, epsabs, epsrel, kwargs):
        """Evaluate the quadrature and differentiate it on the converged subdivision."""
        ops = _with_checkpoint(ops, self.checkpoint)
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
            vfunc, interval_t = ops.build(interval, args, consts)
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
        vfunc, interval_t = ops.build(interval_, args_, consts_)
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


def _leibniz_impl(
    *flat,
    ops,
    n,
    treedef,
    static,
    kwargs_items,
    frozen_treedef,
    freeze,
    split,
    out_sds,
):
    """Forward direction: integrate the tangent of the mapped integrand."""
    dyn_t, dyn, primals, frozen = _leibniz_unpack(
        flat, n, treedef, static, frozen_treedef
    )
    rule, interval, args, consts, epsabs, epsrel = primals
    kwargs = dict(kwargs_items)
    _, interval_t = ops.build(interval, args, consts)

    def dvfunc(t):
        def at_t(dyn_):
            _, interval_, args_, consts_, _, _ = eqx.combine(dyn_, static)
            vf, _ = ops.build(interval_, args_, consts_)
            return vf(t)

        return jax.jvp(at_t, (dyn,), (dyn_t,))[1]

    del out_sds
    # The split below rebuilds the subdivision, which only exists (and is only reverse
    # differentiable) for the adaptive routines. Romberg has neither a subdivision nor
    # breakpoints, so there is nothing to split and its level loop cannot be transposed.
    if freeze or not split:
        return _run_solve(
            ops,
            rule,
            dvfunc,
            interval_t,
            epsabs,
            epsrel,
            kwargs,
            frozen if freeze else None,
        )

    # Split the tangent. Derivatives with respect to the *limits* have to go through the
    # subdivision, because a breakpoint sitting on a discontinuity contributes a jump
    # term that no amount of integrating a derivative can see: in mapped coordinates the
    # jump moves relative to a fixed mesh, and quadrature of df/dx misses the delta.
    # Rebuilding the mesh from the limits tracks the breakpoint and recovers it, exactly
    # as DirectAdjoint does. Everything else keeps the error-controlled solve.
    dyn_t_rest = (dyn_t[0], jax.tree.map(jnp.zeros_like, dyn_t[1]), *dyn_t[2:])

    def dvfunc_rest(t):
        def at_t(dyn_):
            _, interval_, args_, consts_, _, _ = eqx.combine(dyn_, static)
            vf, _ = ops.build(interval_, args_, consts_)
            return vf(t)

        return jax.jvp(at_t, (dyn,), (dyn_t_rest,))[1]

    y_dot = ops.solve(rule, dvfunc_rest, interval_t, epsabs, epsrel, kwargs)[0]
    return (
        y_dot
        + jax.jvp(
            partial(
                _mesh_quad,
                ops=ops,
                rule=rule,
                args=args,
                consts=consts,
                frozen=frozen,
                kwargs=kwargs,
            ),
            (interval,),
            (dyn_t[1],),
        )[1]
    )


def _mesh_quad(interval, *, ops, rule, args, consts, frozen, kwargs):
    """Quadrature on the rebuilt subdivision, as a function of the limits alone."""
    vfunc, interval_t = ops.build(interval, args, consts)
    return ops.frozen_solve(rule, vfunc, interval_t, frozen, kwargs)


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
    split,
    out_sds,
):
    """Reverse direction: integrate the cotangent of the mapped integrand."""
    del out_sds
    _, dyn, primals, frozen = _leibniz_unpack(flat, n, treedef, static, frozen_treedef)
    rule, interval, args, consts, epsabs, epsrel = primals
    kwargs = dict(kwargs_items)
    _, interval_t = ops.build(interval, args, consts)
    _, unravel = ravel_pytree(dyn)

    def adjoint_integrand(t):
        def at_t(dyn_):
            _, interval_, args_, consts_, _, _ = eqx.combine(dyn_, static)
            vf, _ = ops.build(interval_, args_, consts_)
            return vf(t)

        _, vjp = jax.vjp(at_t, dyn)
        return ravel_pytree(vjp(ct)[0])[0]

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
    if split:
        # the limits' cotangent comes from the rebuilt subdivision instead, so that a
        # breakpoint on a discontinuity picks up its jump term (see _leibniz_impl)
        ct_iv = jax.vjp(
            partial(
                _mesh_quad,
                ops=ops,
                rule=rule,
                args=args,
                consts=consts,
                frozen=frozen,
                kwargs=kwargs,
            ),
            interval,
        )[1](ct)[0]
        ct_tree = (ct_tree[0], ct_iv, *ct_tree[2:])
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

    Derivatives with respect to the integration limits are taken on the subdivision the
    primal solve settled on, rather than from the error-controlled solve. In mapped
    coordinates a moving breakpoint slides a discontinuity across a fixed mesh, and
    integrating ``df/dx`` cannot represent the resulting delta; rebuilding the mesh from
    the limits tracks the breakpoint and recovers the jump term. Only the derivative
    with respect to ``args`` gets its own error control.

    Parameters
    ----------
    checkpoint : bool
        Whether to recompute the quadrature on each block of sub-intervals during the
        backward pass rather than storing it. Without it reverse mode keeps the
        integrand's value at every node of every sub-interval, which dominates its
        memory and grows with ``max_ninter`` however few sub-intervals are really used;
        recomputing cuts that by around 3x at the default budget and by 30x or more
        when ``max_ninter`` is generous, at no measured cost in speed, so it is on by
        default. Turning it off has not been found to pay for itself even on integrands
        costing megaflops per evaluation, so treat it mainly as a diagnostic knob. No
        effect in forward mode.
    """

    checkpoint: bool = True

    def quadrature(self, ops, rule, interval, args, consts, epsabs, epsrel, kwargs):
        """Evaluate the quadrature, differentiating it by the Leibniz rule."""
        ops = _with_checkpoint(ops, self.checkpoint)
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
    # The limits' contribution has to go through the subdivision so that a breakpoint on
    # a discontinuity picks up its jump term, but that costs an extra fixed-mesh pass
    # and carries the mesh around. Only do it when the limits are actually being
    # differentiated, which filter_custom_jvp tells us at trace time by handing us a
    # `None` tangent for anything that is not. Romberg has no subdivision to split.
    split = (
        not freeze
        and ops.rebuild is not None
        and any(t is not None for t in jax.tree.leaves(tangents[1]))
    )
    if freeze or split:
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
        split=split,
        out_sds=jax.ShapeDtypeStruct(jnp.shape(y), jnp.result_type(y)),
    )
    return (y, state), (y_dot, _zero_tangent(state))
