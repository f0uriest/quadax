"""Functions for globally h-adaptive quadrature."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import unvmap_any
from jax.typing import ArrayLike

from .adjoint import (
    AbstractAdjoint,
    DirectAdjoint,
    QuadratureOps,
    build_integrand,
    closure_convert,
)
from .fixed_order import (
    AbstractQuadratureRule,
    ClenshawCurtisRule,
    GaussKronrodRule,
    TanhSinhRule,
)
from .utils import (
    QuadratureInfo,
    _real_dtype,
    bounded_while_loop,
    errorif,
    resolve_dtypes,
)

NORMAL_EXIT = 0
MAX_NINTER = 1
ROUNDOFF = 2
BAD_INTEGRAND = 3
NO_CONVERGE = 4
DIVERGENT = 5


@eqx.filter_jit
def quadgk(
    fun: Callable[..., jax.Array],
    interval: ArrayLike,
    args: tuple = (),
    full_output: bool = False,
    epsabs: ArrayLike | None = None,
    epsrel: ArrayLike | None = None,
    max_ninter: int = 50,
    order: int = 21,
    norm: float | int | Callable[[jax.Array], jax.Array] = jnp.inf,
    adjoint: AbstractAdjoint = DirectAdjoint(),
):
    """Global adaptive quadrature using Gauss-Kronrod rule.

    Integrate fun from `interval[0]` to `interval[-1]` using a h-adaptive scheme with
    error estimate. Breakpoints can be specified in `interval` where integration
    difficulty may occur.

    Basically the same as ``scipy.integrate.quad`` but without extrapolation. A good
    general purpose integrator for most reasonably well behaved functions over finite
    or infinite intervals.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals. Its dtype sets the working precision: the integrand
        is called with an ``x`` of this dtype, and the result follows it unless the
        integrand upcasts. A integer types or python floats falls back to the JAX
        default. Must be real; complex integrands are supported, complex limits are not.
    args : tuple, optional
        Extra arguments passed to fun.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is the square root of the
        machine precision of the working dtype, ie of `interval`, or of the integrand's
        own dtype if that is the coarser of the two. Algorithm tries to obtain an
        accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` =
        integral of `fun` over `interval`, and ``result`` is the numerical
        approximation.
    max_ninter : int, optional
        An upper bound on the number of sub-intervals used in the adaptive
        algorithm.
    order : {15, 21, 31, 41, 51, 61}
        Order of local integration rule.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.

    Returns
    -------
    y : float, Array
        The integral of fun from `a` to `b`.
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (int) Flag indicating reason for termination. status of 0 means
          normal termination, any other value indicates a possible error. A human
          readable message can be obtained by ``print(quadax.STATUS[status])``
        * info : (dict or None) Other information returned by the algorithm.
          Only present if ``full_output`` is True. Contains the following:

          * 'ninter' : (int) The number, K, of sub-intervals produced in the
            subdivision process.
          * 'a_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the left end points of the (remapped) sub-intervals
            in the partition of the integration range.
          * 'b_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the right end points of the (remapped) sub-intervals.
          * 'r_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the integral approximations on the sub-intervals.
          * 'e_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the moduli of the absolute error estimates on the
            sub-intervals.

    Notes
    -----
    Adaptive algorithms are inherently somewhat sequential, so perfect parallelism
    is generally not achievable. The local quadrature rule vmaps integrand evaluation at
    ``order`` points, so using higher order methods will generally be more efficient on
    GPU/TPU.

    """
    rule = GaussKronrodRule(order, norm)
    y, info = adaptive_quadrature(
        rule,
        fun,
        interval,
        args,
        full_output,
        epsabs,
        epsrel,
        max_ninter,
        adjoint=adjoint,
    )
    info = QuadratureInfo(info.err, info.neval * order, info.status, info.info)
    return y, info


@eqx.filter_jit
def quadcc(
    fun: Callable[..., jax.Array],
    interval: ArrayLike,
    args: tuple = (),
    full_output: bool = False,
    epsabs: ArrayLike | None = None,
    epsrel: ArrayLike | None = None,
    max_ninter: int = 50,
    order: int = 32,
    norm: float | int | Callable[[jax.Array], jax.Array] = jnp.inf,
    adjoint: AbstractAdjoint = DirectAdjoint(),
):
    """Global adaptive quadrature using Clenshaw-Curtis rule.

    Integrate fun from `interval[0]` to `interval[-1]` using a h-adaptive scheme with
    error estimate. Breakpoints can be specified in `interval` where integration
    difficulty may occur.

    A good general purpose integrator for most reasonably well behaved functions over
    finite or infinite intervals.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals. Its dtype sets the working precision: the integrand
        is called with an ``x`` of this dtype, and the result follows it unless the
        integrand upcasts. A integer types or python floats falls back to the JAX
        default. Must be real; complex integrands are supported, complex limits are not.
    args : tuple, optional
        Extra arguments passed to fun.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is the square root of the
        machine precision of the working dtype, ie of `interval`, or of the integrand's
        own dtype if that is the coarser of the two. Algorithm tries to obtain an
        accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` =
        integral of `fun` over `interval`, and ``result`` is the numerical
        approximation.
    max_ninter : int, optional
        An upper bound on the number of sub-intervals used in the adaptive
        algorithm.
    order : {8, 16, 32, 64, 128, 256}
        Order of local integration rule.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.

    Returns
    -------
    y : float, Array
        The integral of fun from `a` to `b`.
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (int) Flag indicating reason for termination. status of 0 means
          normal termination, any other value indicates a possible error. A human
          readable message can be obtained by ``print(quadax.STATUS[status])``
        * info : (dict or None) Other information returned by the algorithm.
          Only present if ``full_output`` is True. Contains the following:

          * 'ninter' : (int) The number, K, of sub-intervals produced in the
            subdivision process.
          * 'a_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the left end points of the (remapped) sub-intervals
            in the partition of the integration range.
          * 'b_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the right end points of the (remapped) sub-intervals.
          * 'r_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the integral approximations on the sub-intervals.
          * 'e_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the moduli of the absolute error estimates on the
            sub-intervals.

    Notes
    -----
    Adaptive algorithms are inherently somewhat sequential, so perfect parallelism
    is generally not achievable. The local quadrature rule vmaps integrand evaluation at
    ``order`` points, so using higher order methods will generally be more efficient on
    GPU/TPU.

    """
    rule = ClenshawCurtisRule(order, norm)
    y, info = adaptive_quadrature(
        rule,
        fun,
        interval,
        args,
        full_output,
        epsabs,
        epsrel,
        max_ninter,
        adjoint=adjoint,
    )
    info = QuadratureInfo(info.err, info.neval * order, info.status, info.info)
    return y, info


@eqx.filter_jit
def quadts(
    fun: Callable[..., jax.Array],
    interval: ArrayLike,
    args: tuple = (),
    full_output: bool = False,
    epsabs: ArrayLike | None = None,
    epsrel: ArrayLike | None = None,
    max_ninter: int = 50,
    order: int = 61,
    norm: float | int | Callable[[jax.Array], jax.Array] = jnp.inf,
    adjoint: AbstractAdjoint = DirectAdjoint(),
):
    """Global adaptive quadrature using trapezoidal tanh-sinh rule.

    Integrate fun from `interval[0]` to `interval[-1]` using a h-adaptive scheme with
    error estimate. Breakpoints can be specified in `interval` where integration
    difficulty may occur.

    Especially good for integrands with singular behavior at an endpoint.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals. Its dtype sets the working precision: the integrand
        is called with an ``x`` of this dtype, and the result follows it unless the
        integrand upcasts. A integer types or python floats falls back to the JAX
        default. Must be real; complex integrands are supported, complex limits are not.
    args : tuple, optional
        Extra arguments passed to fun.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is the square root of the
        machine precision of the working dtype, ie of `interval`, or of the integrand's
        own dtype if that is the coarser of the two. Algorithm tries to obtain an
        accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` =
        integral of `fun` over `interval`, and ``result`` is the numerical
        approximation.
    max_ninter : int, optional
        An upper bound on the number of sub-intervals used in the adaptive
        algorithm.
    order : {41, 61, 81, 101}
        Order of local integration rule.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.

    Returns
    -------
    y : float, Array
        The integral of fun from `a` to `b`.
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (int) Flag indicating reason for termination. status of 0 means
          normal termination, any other value indicates a possible error. A human
          readable message can be obtained by ``print(quadax.STATUS[status])``
        * info : (dict or None) Other information returned by the algorithm.
          Only present if ``full_output`` is True. Contains the following:

          * 'ninter' : (int) The number, K, of sub-intervals produced in the
            subdivision process.
          * 'a_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the left end points of the (remapped) sub-intervals
            in the partition of the integration range.
          * 'b_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the right end points of the (remapped) sub-intervals.
          * 'r_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the integral approximations on the sub-intervals.
          * 'e_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the moduli of the absolute error estimates on the
            sub-intervals.

    Notes
    -----
    Adaptive algorithms are inherently somewhat sequential, so perfect parallelism
    is generally not achievable. The local quadrature rule vmaps integrand evaluation at
    ``order`` points, so using higher order methods will generally be more efficient on
    GPU/TPU.

    """
    rule = TanhSinhRule(order, norm)
    y, info = adaptive_quadrature(
        rule,
        fun,
        interval,
        args,
        full_output,
        epsabs,
        epsrel,
        max_ninter,
        adjoint=adjoint,
    )
    info = QuadratureInfo(info.err, info.neval * order, info.status, info.info)
    return y, info


@eqx.filter_jit
def adaptive_quadrature(
    rule: AbstractQuadratureRule,
    fun: Callable[..., jax.Array],
    interval: ArrayLike,
    args: tuple = (),
    full_output: bool = False,
    epsabs: ArrayLike | None = None,
    epsrel: ArrayLike | None = None,
    max_ninter: int = 50,
    adjoint: AbstractAdjoint = DirectAdjoint(),
    **kwargs,
):
    """Global adaptive quadrature.

    This is a lower level routine allowing for custom local quadrature rules. For most
    applications the higher order methods ``quadgk``, ``quadcc``, ``quadts`` are
    preferable.

    Parameters
    ----------
    rule : AbstractQuadratureRule
        Local quadrature rule to use.
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals. Its dtype sets the working precision: the integrand
        is called with an ``x`` of this dtype, and the result follows it unless the
        integrand upcasts. A integer types or python floats falls back to the JAX
        default. Must be real; complex integrands are supported, complex limits are not.
    args : tuple, optional
        Extra arguments passed to fun.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is the square root of the
        machine precision of the working dtype, ie of `interval`, or of the integrand's
        own dtype if that is the coarser of the two. Algorithm tries to obtain an
        accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` =
        integral of `fun` over `interval`, and ``result`` is the numerical
        approximation.
    max_ninter : int, optional
        An upper bound on the number of sub-intervals used in the adaptive
        algorithm.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. ``LeibnizAdjoint`` gives the derivative
        its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter``
        is generous; see the Adjoints section of the API documentation for when that
        is worth paying for.
    kwargs : dict
        Additional keyword arguments passed to ``rule``.

    Returns
    -------
    y : float, Array
        The integral of fun from `a` to `b`.
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of rule evaluations.
        * status : (int) Flag indicating reason for termination. status of 0 means
          normal termination, any other value indicates a possible error. A human
          readable message can be obtained by ``print(quadax.STATUS[status])``
        * info : (dict or None) Other information returned by the algorithm.
          Only present if ``full_output`` is True. Contains the following:

          * 'ninter' : (int) The number, K, of sub-intervals produced in the
            subdivision process.
          * 'a_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the left end points of the (remapped) sub-intervals
            in the partition of the integration range.
          * 'b_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the right end points of the (remapped) sub-intervals.
          * 'r_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the integral approximations on the sub-intervals.
          * 'e_arr' : (ndarray) rank-1 array of length max_ninter, the first K
            elements of which are the moduli of the absolute error estimates on the
            sub-intervals.

    """
    errorif(
        not isinstance(rule, AbstractQuadratureRule),
        TypeError,
        "rule should be an instance of quadax.AbstractQuadratureRule, "
        f"got {type(rule)}",
    )
    interval = jnp.atleast_1d(jnp.asarray(interval))
    if not jnp.issubdtype(interval.dtype, jnp.inexact):
        # integration limits must be inexact: they are differentiated with respect to,
        # and integer leaves would otherwise be treated as static metadata
        interval = interval.astype(jnp.result_type(float))
    errorif(
        max_ninter < len(interval) - 1,
        ValueError,
        f"max_ninter={max_ninter} is not enough for {len(interval) - 1} breakpoints",
    )
    dtypes = resolve_dtypes(interval, fun, args)
    if epsabs is None:
        epsabs = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    if epsrel is None:
        epsrel = jnp.sqrt(jnp.finfo(dtypes.toltype).eps)
    epsabs = jnp.asarray(epsabs, dtypes.etype)
    epsrel = jnp.asarray(epsrel, dtypes.etype)

    f_conv, consts = closure_convert(fun, args, dtypes.xtype)

    ops = QuadratureOps(
        build=partial(build_integrand, f_conv=f_conv),
        solve=partial(_adaptive_solve, max_ninter=max_ninter),
        rebuild=_rebuild_mesh,
        on_mesh=_quad_on_mesh,
        frozen=_frozen_mesh,
        frozen_solve=_mesh_solve,
    )
    y, state = adjoint.quadrature(
        ops, rule, interval, args, consts, epsabs, epsrel, kwargs
    )

    err = state["err_sum"]
    neval = state["neval"]
    status = state["status"]
    info = state if full_output else None
    out = QuadratureInfo(err, neval, status, info)
    return y, out


# How many sub-intervals of a fixed subdivision are evaluated at once. Evaluating all of
# them together is fastest but makes peak memory scale with ``max_ninter``, which is a
# safety bound users tend to set generously; evaluating one at a time streams but
# serializes. Measured on a scalar integrand with an order 21 rule, 8 is where the curve
# turns over: it matches one-at-a-time peak memory at large ``max_ninter`` while being
# noticeably faster in reverse mode, and larger blocks buy little more speed for a lot
# more memory.
_CHUNK = 8


def _quad_on_mesh(rule, vfunc, a_arr, b_arr, kwargs, *, checkpoint=True):
    """Apply the local rule on a fixed subdivision and sum the contributions."""
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
    # ``_CHUNK``.

    del kwargs
    used = a_arr != b_arr
    a_safe = jnp.where(used, a_arr, a_arr[0])
    b_safe = jnp.where(used, b_arr, b_arr[0])

    nslot = a_arr.shape[0]
    chunk = min(_CHUNK, nslot)
    pad = -nslot % chunk
    reshape = lambda x, fill: jnp.pad(x, (0, pad), constant_values=fill).reshape(
        -1, chunk
    )
    a_c, b_c = reshape(a_safe, a_arr[0]), reshape(b_safe, b_arr[0])

    apply1 = lambda a, b: rule._apply(vfunc, a, b, ())
    sds = jax.eval_shape(apply1, a_arr[0], b_arr[0])
    # The mask multiplies the *values*, so it takes their (real) dtype rather than the
    # mesh's. With the mesh at float64 and the values at float32 the latter would
    # otherwise be promoted straight back to float64 here.
    used_c = reshape(used.astype(_real_dtype(sds.dtype)), 0.0)

    def bodyfun(total, block):
        a, b, m = block

        def evaluate(_):
            y = jax.vmap(apply1)(a, b)
            y = y * m.reshape((-1,) + (1,) * (y.ndim - 1))
            return jnp.sum(y, axis=0)

        # `unvmap_any` keeps the predicate a scalar under `vmap`, so the block is
        # skipped whenever *no* batch element uses it instead of degrading to a select
        # that evaluates every block. No inner per-element gate is needed: `m` already
        # zeroes the slots an individual element does not use.
        contrib = jax.lax.cond(
            unvmap_any(jnp.any(m != 0)),
            evaluate,
            lambda _: jnp.zeros(sds.shape, sds.dtype),
            None,
        )
        return total + contrib, None

    if checkpoint:
        # Recompute each block during the backward pass instead of keeping the
        # integrand's value at every node of every sub-interval. Those values dominate
        # reverse mode otherwise, and recomputing them is nearly free here.
        bodyfun = jax.checkpoint(bodyfun)

    total, _ = jax.lax.scan(
        bodyfun, jnp.zeros(sds.shape, sds.dtype), (a_c, b_c, used_c)
    )
    return total


def _frozen_mesh(state):
    """The parts of the subdivision that do not vary smoothly with the limits."""
    return (state["owner"], state["frac_a"], state["frac_b"])


def _mesh_solve(rule, vfunc, interval, frozen, kwargs, *, checkpoint=True):
    """Quadrature on the subdivision implied by `frozen`, as a function of interval."""
    a_arr, b_arr = _rebuild_mesh(interval, frozen)
    return _quad_on_mesh(rule, vfunc, a_arr, b_arr, kwargs, checkpoint=checkpoint)


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


def _at_roundoff_floor(state, epmach, norm):
    """Report whether the error has bottomed out while still above the tolerance.

    The local rule floors each sub-interval's error estimate at ``50*eps*int|f|`` over
    that sub-interval, so the total can never fall below that floor summed over the
    partition -- and that sum stays near ``50*eps*int|f|`` over the whole domain however
    finely the mesh is refined. Once the total has reached the floor and is still above
    ``err_bnd``, no amount of further subdivision will reach the requested tolerance,
    because the tolerance is below what the arithmetic can resolve.

    QUADPACK makes this test only in its initial phase, before the subdivision loop, so
    a request below the achievable precision is left to exhaust the subdivision budget
    and report that instead. quadax makes it every iteration, which reports the actual
    difficulty and stops early.
    """
    intabs = norm(jnp.sum(state["f_arr"], axis=0))
    return (state["err_sum"] <= 100.0 * epmach * intabs) & (
        state["err_sum"] > state["err_bnd"]
    )


def _adaptive_solve(rule, vfunc, interval, epsabs, epsrel, kwargs, *, max_ninter):
    """Run the globally adaptive subdivision loop."""
    intfun = partial(rule.integrate, **kwargs) if kwargs else rule.integrate
    _norm = rule.norm
    f = jax.eval_shape(vfunc, (interval[0] + interval[-1]) / 2)
    shape = f.shape
    # Derived here rather than threaded in, so that this stays correct when the adjoints
    # call it with a tangent integrand whose dtype is not the primal's. `vfunc` has
    # already been through `map_interval`, whose Jacobian is at `xtype`, so its output
    # dtype is the accumulation dtype by construction.
    xtype = interval.dtype
    ytype = f.dtype
    etype = _real_dtype(ytype)  # errors and the integral of |f| are real
    # Roundoff in the arithmetic that forms the sums, versus roundoff in the mesh: the
    # first bounds how small an error estimate can honestly be, the second how narrow a
    # sub-interval can get before its endpoints stop being distinguishable.
    epmach = float(jnp.finfo(etype).eps)
    epmach_x = float(jnp.finfo(xtype).eps)

    state = {}
    state["neval"] = 0  # number of evaluations of local quadrature rule
    state["ninter"] = len(interval) - 1  # current number of intervals
    state["r_arr"] = jnp.zeros(
        (max_ninter, *shape), ytype
    )  # local results from each interval
    state["e_arr"] = jnp.zeros(max_ninter, etype)  # local error est. from each interval
    state["a_arr"] = jnp.zeros(max_ninter, xtype)  # start of each interval
    state["b_arr"] = jnp.zeros(max_ninter, xtype)  # end of each interval
    state["s_arr"] = jnp.zeros(
        (max_ninter, *shape), ytype
    )  # global est. of I from n intervals
    state["f_arr"] = jnp.zeros(
        (max_ninter, *shape), etype
    )  # local est. of integral of abs(fun) from each interval
    state["a_arr"] = state["a_arr"].at[: state["ninter"]].set(interval[:-1])
    state["b_arr"] = state["b_arr"].at[: state["ninter"]].set(interval[1:])
    state["roundoff1"] = 0  # for keeping track of roundoff errors
    state["roundoff2"] = 0  # for keeping track of roundoff errors
    state["status"] = 0  # error flag
    # Explicitly typed rather than left as weak python floats: these are `scan` carries,
    # so their dtype has to match what the loop body writes back into them.
    state["err_bnd"] = jnp.zeros((), etype)  # error bound we're trying to reach
    state["area"] = jnp.zeros(shape, ytype)  # current best estimate for I
    state["err_sum"] = jnp.zeros((), etype)  # current estimate for error in I
    # Where each sub-interval sits relative to the *original* sub-intervals: which one
    # it was carved out of, and the fractions of the way along it that its ends lie at.
    # These are what stay fixed when a limit or breakpoint moves, so recording them lets
    # the mesh be rebuilt as a smooth function of `interval` by gather and rescale.
    state["owner"] = (
        jnp.zeros(max_ninter, int)
        .at[: state["ninter"]]
        .set(jnp.arange(state["ninter"]))
    )
    state["frac_a"] = jnp.zeros(max_ninter, xtype)
    state["frac_b"] = jnp.zeros(max_ninter, xtype).at[: state["ninter"]].set(1.0)

    def init_body(i, state):
        a = state["a_arr"][i]
        b = state["b_arr"][i]
        result, abserr, intabs, intmmn = intfun(vfunc, a, b, ())

        state["neval"] += 1
        state["r_arr"] = state["r_arr"].at[i].set(result)
        state["e_arr"] = state["e_arr"].at[i].set(abserr)
        state["f_arr"] = state["f_arr"].at[i].set(intabs)
        state["area"] = jnp.sum(state["r_arr"], axis=0)
        state["err_sum"] = jnp.sum(state["e_arr"])
        state["s_arr"] = state["s_arr"].at[i].set(state["area"])
        return state

    state = jax.lax.fori_loop(0, state["ninter"], init_body, state)
    state["err_bnd"] = jnp.maximum(epsabs, epsrel * _norm(state["area"]))
    # check for roundoff error - error too big but relative error is small
    state["status"] += 2**ROUNDOFF * _at_roundoff_floor(state, epmach, _norm)

    # check for max intervals exceeded
    state["status"] += 2**MAX_NINTER * (state["ninter"] >= max_ninter)

    def condfun(state):
        return (
            (state["status"] == 0)
            & (0 <= state["err_sum"])
            & (state["err_bnd"] <= state["err_sum"])
        )

    def bodyfun(state):
        # bisect the sub-interval with the largest error estimate.
        i = jnp.argmax(state["e_arr"])
        # The bisection turns one sub-interval into two, so the extra half goes in the
        # first free slot, which is the *current* interval count, before incrementing
        # it. Taking the count after the increment instead would skip a slot and, on the
        # final iteration, index off the end of the arrays, silently dropping that half
        # of the interval from the result.
        n = state["ninter"]
        state["ninter"] += 1
        a1 = state["a_arr"][i]
        b1 = 0.5 * (state["a_arr"][i] + state["b_arr"][i])
        a2 = b1
        b2 = state["b_arr"][i]

        area1, error1, intabs1, intmmn1 = intfun(vfunc, a1, b1, ())
        state["neval"] += 1
        area2, error2, intabs2, intmmn2 = intfun(vfunc, a2, b2, ())
        state["neval"] += 1

        # improve previous approximations to integral and error and test for accuracy.
        area12 = area1 + area2
        erro12 = error1 + error2
        # The parent's contribution and error estimate, read before either is
        # overwritten below. These are QUADPACK's `rlist(maxerr)`/`errmax`, and the
        # stagnation test further down compares against the *parent*, so they have to be
        # captured here rather than read back out of the arrays.
        area_i = state["r_arr"][i]
        err_i = state["e_arr"][i]

        # Which half keeps slot `i` and which takes the new slot `n`: as in QUADPACK the
        # larger error goes first. Only the placement depends on this, not either total,
        # so both arrays are written here and the branch at the end of the body is left
        # with the endpoints and the fractions.
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

        # Both running totals are summed afresh from the per-interval contributions
        # rather than carried forward as `total += new - old`. Accumulating discards
        # ~eps times the largest term ever subtracted on every iteration, and that drift
        # random-walks while the total it is tracking shrinks. For `err_sum` the two
        # move in opposite directions and the drift can outgrow the total outright: on
        # an integrand with a tall narrow peak it goes negative, and the loop then exits
        # through the `0 <= err_sum` guard in `condfun` with `status` still 0, reporting
        # a nonsense error estimate and a clean bill of health. Summing afresh keeps the
        # error at O(eps * total)
        state["err_sum"] = jnp.sum(state["e_arr"])
        state["area"] = jnp.sum(state["r_arr"], axis=0)
        state["s_arr"] = state["s_arr"].at[n].set(state["area"])
        state["err_bnd"] = jnp.maximum(epsabs, epsrel * _norm(state["area"]))

        # Did the local rule resolve both halves at all? The error estimate saturates at
        # exactly the integral of |f - <f>| when the rule learned nothing about that
        # half, in which case a stagnant area is evidence of an unresolved integrand
        # rather than of roundoff, and QUADPACK skips both counters. Without this a hard
        # but tractable integrand accumulates stagnation counts while it is still making
        # legitimate progress. The equality is exact on purpose: it asks whether the
        # `min(1, ...)` clamped, not whether two quantities are merely close. Both sides
        # go through `_norm` because that is the reduction the error estimate itself
        # already went through for vector valued integrands.
        resolved = (error1 != _norm(intmmn1)) & (error2 != _norm(intmmn2))

        # test for roundoff error
        # is the area estimate not changing and error not getting smaller?
        # QUADPACK's threshold is a flat 1e-5, which is only meaningful while it sits
        # above the noise floor of the difference it is applied to, ~eps*|area12|. It
        # does at float32 and float64 (`50*eps` is 6e-6 and 1.1e-14, so the maximum is
        # 1e-5 either way and this is QUADPACK's test unchanged) but not in half
        # precision, where a flat 1e-5 is below the noise and this counter could never
        # fire. `50` is the same as in the local rule's roundoff floor.
        stagnant = max(1e-5, 50 * epmach)
        state["roundoff1"] += (
            resolved
            & (_norm(area_i - area12) <= stagnant * _norm(area12))
            & (erro12 >= 0.99 * err_i)
        )
        # are errors getting larger as we go to smaller intervals?
        state["roundoff2"] += resolved & (state["ninter"] > 10) & (erro12 > err_i)

        # Whether the tolerance was reached on this iteration. QUADPACK jumps past
        # every `ier` assignment once `errsum <= errbnd`, so an iteration that both
        # reaches the tolerance and, say, consumes the last subdivision slot still exits
        # cleanly. The counters above are still updated, matching the original.
        converged = state["err_sum"] <= state["err_bnd"]

        # Roundoff is reported either because the error has bottomed out at the floor
        # the arithmetic imposes, or because the two counters say subdivision has
        # stopped buying anything.
        state["status"] += (
            2**ROUNDOFF
            * ~converged
            * (
                _at_roundoff_floor(state, epmach, _norm)
                | (state["roundoff1"] >= 10)
                | (state["roundoff2"] >= 20)
            )
        )

        # test for max number of intervals
        state["status"] += 2**MAX_NINTER * ~converged * (state["ninter"] >= max_ninter)

        # test for bad behavior of the integrand (ie, intervals are getting too small)
        # This one is about the *mesh*, not the values, so it scales with the precision
        # the abscissae are carried at rather than the precision of the sums.
        state["status"] += (
            2**BAD_INTEGRAND
            * ~converged
            * (jnp.maximum(jnp.abs(b1 - a1), jnp.abs(b2 - a2)) <= (100.0 * epmach_x))
        )

        # update the arrays of interval starts/ends etc

        # both halves stay inside whichever original sub-interval this one came from,
        # splitting its span at the midpoint of the fractions
        owner_i = state["owner"][i]
        frac_a1 = state["frac_a"][i]
        frac_b2 = state["frac_b"][i]
        frac_mid = 0.5 * (frac_a1 + frac_b2)
        state["owner"] = state["owner"].at[n].set(owner_i)

        # `e_arr` and `r_arr` were placed above; what is left is to give the two halves
        # the endpoints and fractions matching the slots they landed in.
        def error1big(state):
            state["a_arr"] = state["a_arr"].at[n].set(a2)
            state["b_arr"] = state["b_arr"].at[i].set(b1)
            state["b_arr"] = state["b_arr"].at[n].set(b2)
            state["frac_b"] = state["frac_b"].at[i].set(frac_mid)
            state["frac_a"] = state["frac_a"].at[n].set(frac_mid)
            state["frac_b"] = state["frac_b"].at[n].set(frac_b2)
            return state

        def error2big(state):
            state["a_arr"] = state["a_arr"].at[i].set(a2)
            state["a_arr"] = state["a_arr"].at[n].set(a1)
            state["b_arr"] = state["b_arr"].at[n].set(b1)
            state["frac_a"] = state["frac_a"].at[i].set(frac_mid)
            state["frac_a"] = state["frac_a"].at[n].set(frac_a1)
            state["frac_b"] = state["frac_b"].at[n].set(frac_mid)
            return state

        state = jax.lax.cond(swap, error2big, error1big, state)
        return state

    state = bounded_while_loop(condfun, bodyfun, state, max_ninter + 1)

    y = jnp.sum(state["r_arr"], axis=0)
    return y, state
