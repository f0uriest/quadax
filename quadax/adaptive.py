"""Functions for globally h-adaptive quadrature."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
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
from .utils import QuadratureInfo, _get_eps, bounded_while_loop, errorif

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
        denote infinite intervals.
    args : tuple, optional
        Extra arguments passed to fun.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is square root of
        machine precision. Algorithm tries to obtain an accuracy of
        ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` = integral of
        `fun` over `interval`, and ``result`` is the numerical approximation.
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
        which is gives the exact derivative of the discretized problem, supports both
        forward and reverse mode, and is usually the fastest option.
        ``LeibnizAdjoint`` is slower but gives the derivative its own error control
        (ie, can better approximate the true continuous derivative), and in reverse
        mode uses much less memory; see ``AbstractAdjoint`` for when that is worth
        paying for.

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
        denote infinite intervals.
    args : tuple, optional
        Extra arguments passed to fun.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is square root of
        machine precision. Algorithm tries to obtain an accuracy of
        ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` = integral of
        `fun` over `interval`, and ``result`` is the numerical approximation.
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
        which is gives the exact derivative of the discretized problem, supports both
        forward and reverse mode, and is usually the fastest option.
        ``LeibnizAdjoint`` is slower but gives the derivative its own error control
        (ie, can better approximate the true continuous derivative), and in reverse
        mode uses much less memory; see ``AbstractAdjoint`` for when that is worth
        paying for.

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
        denote infinite intervals.
    args : tuple, optional
        Extra arguments passed to fun.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is square root of
        machine precision. Algorithm tries to obtain an accuracy of
        ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` = integral of
        `fun` over `interval`, and ``result`` is the numerical approximation.
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
        which is gives the exact derivative of the discretized problem, supports both
        forward and reverse mode, and is usually the fastest option.
        ``LeibnizAdjoint`` is slower but gives the derivative its own error control
        (ie, can better approximate the true continuous derivative), and in reverse
        mode uses much less memory; see ``AbstractAdjoint`` for when that is worth
        paying for.

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
        denote infinite intervals.
    args : tuple, optional
        Extra arguments passed to fun.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is square root of
        machine precision. Algorithm tries to obtain an accuracy of
        ``abs(i-result) <= max(epsabs, epsrel*abs(i))`` where ``i`` = integral of
        `fun` over `interval`, and ``result`` is the numerical approximation.
    max_ninter : int, optional
        An upper bound on the number of sub-intervals used in the adaptive
        algorithm.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which is gives the exact derivative of the discretized problem, supports both
        forward and reverse mode, and is usually the fastest option.
        ``LeibnizAdjoint`` is slower but gives the derivative its own error control
        (ie, can better approximate the true continuous derivative), and in reverse
        mode uses much less memory; see ``AbstractAdjoint`` for when that is worth
        paying for.
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
    if epsabs is None:
        epsabs = jnp.sqrt(_get_eps(jnp.array(1.0)))
    if epsrel is None:
        epsrel = jnp.sqrt(_get_eps(jnp.array(1.0)))
    epsabs = jnp.asarray(epsabs)
    epsrel = jnp.asarray(epsrel)

    f_conv, consts = closure_convert(fun, args)

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
    """Apply the local rule on a fixed subdivision and sum the contributions.

    Sub-intervals are independent, so they are evaluated in blocks: ``vmap`` within a
    block, ``scan`` across blocks. A plain ``scan`` over every sub-interval would make a
    gradient cost ``max_ninter`` rather than the number of sub-intervals actually used,
    because reverse mode stacks residuals for every iteration whether or not it did any
    work. A plain ``vmap`` fixes that but materializes the whole subdivision at once.

    Slots past the end of the subdivision are empty (``a == b``). They are handed a real
    sub-interval and masked out afterwards rather than skipped with a ``cond``: under
    ``vmap`` a batched ``cond`` becomes a ``select`` carrying a ``stop_gradient`` that
    cannot be transposed. Substituting a real sub-interval also stops an integrand that
    is singular somewhere in the mapped domain from poisoning the unused slots with a
    NaN that the mask would then propagate.
    """
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
    used_c = reshape(used.astype(a_arr.dtype), 0.0)

    apply1 = lambda a, b: rule._apply(vfunc, a, b, ())
    sds = jax.eval_shape(apply1, a_arr[0], b_arr[0])

    def bodyfun(total, block):
        a, b, m = block
        y = jax.vmap(apply1)(a, b)
        y = y * m.reshape((-1,) + (1,) * (y.ndim - 1))
        return total + jnp.sum(y, axis=0), None

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


def _adaptive_solve(rule, vfunc, interval, epsabs, epsrel, kwargs, *, max_ninter):
    """Run the globally adaptive subdivision loop."""
    intfun = partial(rule.integrate, **kwargs) if kwargs else rule.integrate
    _norm = rule.norm
    f = jax.eval_shape(vfunc, (interval[0] + interval[-1]) / 2)
    epmach = _get_eps(f)
    shape = f.shape

    state = {}
    state["neval"] = 0  # number of evaluations of local quadrature rule
    state["ninter"] = len(interval) - 1  # current number of intervals
    state["r_arr"] = jnp.zeros(
        (max_ninter, *shape), f.dtype
    )  # local results from each interval
    state["e_arr"] = jnp.zeros(max_ninter)  # local error est. from each interval
    state["a_arr"] = jnp.zeros(max_ninter)  # start of each interval
    state["b_arr"] = jnp.zeros(max_ninter)  # end of each interval
    state["s_arr"] = jnp.zeros(
        (max_ninter, *shape), f.dtype
    )  # global est. of I from n intervals
    state["a_arr"] = state["a_arr"].at[: state["ninter"]].set(interval[:-1])
    state["b_arr"] = state["b_arr"].at[: state["ninter"]].set(interval[1:])
    state["roundoff1"] = 0  # for keeping track of roundoff errors
    state["roundoff2"] = 0  # for keeping track of roundoff errors
    state["status"] = 0  # error flag
    state["err_bnd"] = 0.0  # error bound we're trying to reach
    state["area"] = jnp.zeros(shape, f.dtype)  # current best estimate for I
    state["err_sum"] = 0.0  # current estimate for error in I
    # Where each sub-interval sits relative to the *original* sub-intervals: which one
    # it was carved out of, and the fractions of the way along it that its ends lie at.
    # These are what stay fixed when a limit or breakpoint moves, so recording them lets
    # the mesh be rebuilt as a smooth function of `interval` by gather and rescale.
    state["owner"] = (
        jnp.zeros(max_ninter, int)
        .at[: state["ninter"]]
        .set(jnp.arange(state["ninter"]))
    )
    state["frac_a"] = jnp.zeros(max_ninter)
    state["frac_b"] = jnp.zeros(max_ninter).at[: state["ninter"]].set(1.0)

    def init_body(i, state_):
        state, intabs_ = state_
        a = state["a_arr"][i]
        b = state["b_arr"][i]
        result, abserr, intabs, intmmn = intfun(vfunc, a, b, ())

        intabs_ += intabs
        state["neval"] += 1
        state["area"] += result
        state["err_sum"] += abserr
        state["r_arr"] = state["r_arr"].at[i].set(result)
        state["e_arr"] = state["e_arr"].at[i].set(abserr)
        state["s_arr"] = state["s_arr"].at[i].set(state["area"])
        return state, intabs_

    state, intabs_ = jax.lax.fori_loop(
        0, state["ninter"], init_body, (state, jnp.zeros(shape))
    )
    state["err_bnd"] = jnp.maximum(epsabs, epsrel * _norm(state["area"]))
    # check for roundoff error - error too big but relative error is small
    state["status"] += 2**ROUNDOFF * (
        (state["err_sum"] <= (100.0 * epmach * _norm(intabs_)))
        & (state["err_sum"] > state["err_bnd"])
    )

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
        state["err_sum"] += erro12 - state["e_arr"][i]
        state["area"] += area12 - state["r_arr"][i]
        state["r_arr"] = state["r_arr"].at[i].set(area1)
        state["r_arr"] = state["r_arr"].at[n].set(area2)
        state["s_arr"] = state["s_arr"].at[n].set(state["area"])
        state["err_bnd"] = jnp.maximum(epsabs, epsrel * _norm(state["area"]))

        # test for roundoff error
        # is the area estimate not changing and error not getting smaller?
        state["roundoff1"] += (
            _norm(state["r_arr"][i] - area12) <= 0.1e-4 * _norm(area12)
        ) & (erro12 >= 0.99 * jnp.max(state["e_arr"]))
        # are errors getting larger as we go to smaller intervals?
        state["roundoff2"] += (state["ninter"] > 10) & (
            erro12 > jnp.max(state["e_arr"])
        )
        state["status"] += 2**ROUNDOFF * (
            (state["roundoff1"] >= 10) | (state["roundoff2"] >= 20)
        )

        # test for max number of intervals
        state["status"] += 2**MAX_NINTER * (state["ninter"] >= max_ninter)

        # test for bad behavior of the integrand (ie, intervals are getting too small)
        state["status"] += 2**BAD_INTEGRAND * (
            jnp.maximum(jnp.abs(b1 - a1), jnp.abs(b2 - a2)) <= (100.0 * epmach)
        )

        # update the arrays of interval starts/ends etc

        # both halves stay inside whichever original sub-interval this one came from,
        # splitting its span at the midpoint of the fractions
        owner_i = state["owner"][i]
        frac_a1 = state["frac_a"][i]
        frac_b2 = state["frac_b"][i]
        frac_mid = 0.5 * (frac_a1 + frac_b2)
        state["owner"] = state["owner"].at[n].set(owner_i)

        def error1big(state):
            state["a_arr"] = state["a_arr"].at[n].set(a2)
            state["b_arr"] = state["b_arr"].at[i].set(b1)
            state["b_arr"] = state["b_arr"].at[n].set(b2)
            state["e_arr"] = state["e_arr"].at[i].set(error1)
            state["e_arr"] = state["e_arr"].at[n].set(error2)
            state["frac_b"] = state["frac_b"].at[i].set(frac_mid)
            state["frac_a"] = state["frac_a"].at[n].set(frac_mid)
            state["frac_b"] = state["frac_b"].at[n].set(frac_b2)
            return state

        def error2big(state):
            state["a_arr"] = state["a_arr"].at[i].set(a2)
            state["a_arr"] = state["a_arr"].at[n].set(a1)
            state["b_arr"] = state["b_arr"].at[n].set(b1)
            state["r_arr"] = state["r_arr"].at[i].set(area2)
            state["r_arr"] = state["r_arr"].at[n].set(area1)
            state["e_arr"] = state["e_arr"].at[i].set(error2)
            state["e_arr"] = state["e_arr"].at[n].set(error1)
            state["frac_a"] = state["frac_a"].at[i].set(frac_mid)
            state["frac_a"] = state["frac_a"].at[n].set(frac_a1)
            state["frac_b"] = state["frac_b"].at[n].set(frac_mid)
            return state

        state = jax.lax.cond(error2 > error1, error2big, error1big, state)
        return state

    state = bounded_while_loop(condfun, bodyfun, state, max_ninter + 1)

    y = jnp.sum(state["r_arr"], axis=0)
    return y, state
