"""Functions for globally h-adaptive quadrature.

References
----------
.. [1] R. Piessens, E. de Doncker-Kapenga, C. W. Uberhuber, D. K. Kahaner. "QUADPACK: A
       Subroutine Package for Automatic Integration". Springer Series in Computational
       Mathematics, vol. 1. Springer-Verlag, Berlin, 1983. doi:10.1007/978-3-642-61786-7
       The subdivision strategy, the empirical constants in the error tests, and the
       control flow around the convergence acceleration all come from here.
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import unvmap_any
from jax.typing import ArrayLike

from . import _acceleration
from ._status import STATUS, error_if_flagged, escalate, withdraw
from .adjoint import (
    AbstractAdjoint,
    DirectAdjoint,
    QuadratureOps,
    _frozen_mesh,
    _frozen_replay,
    _mesh_solve,
    _quad_on_mesh,
    _rebuild_mesh,
    _replay_solve,
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
    _ROUNDOFF_FLOOR,
    QuadratureInfo,
    _real_dtype,
    bounded_while_loop,
    errorif,
    resolve_dtypes,
    tree_where,
)

# The constants below are QUADPACK's [1], empirical there rather than derived. They
# govern when the loop gives up on a line of attack, never what it reports as the
# answer, so each one trades evaluations against the risk of stopping early on a hard
# integrand.

# Extrapolations in a row that fail to improve on the best the table has produced,
# after which the table is judged to have stopped making progress. Paired with
# `_STALL_SHARP`: a table that is not improving but still reports an error comparable
# to the mesh's has not given up on anything.
_STALL_LIMIT = 5

# How far under the mesh's own error estimate the table's has to sit for the stall test
# to fire. The comparison is against the uncorrected estimate, which measures the spread
# of the last few extrapolants, so this asks whether the table has settled rather than
# how far the answer might be off.
_STALL_SHARP = 1e-3

# How close two successive area estimates have to be for a bisection to count as having
# bought nothing, as a relative tolerance on the area. Floored at the roundoff level in
# use, since a flat threshold only means anything while it sits above the noise of the
# difference it is applied to.
_STAGNANT_RTOL = 1e-5

# ...and how little the error estimate has to shrink over that same bisection. The two
# together say the sub-interval was split and neither the value nor its error moved.
_NO_PROGRESS = 0.99

# Bisections that bought nothing, tolerated before the run reports roundoff. Counted
# over the whole run.
_ROUNDOFF1_LIMIT = 10

# The same events counted over the extrapolating phase alone, where reaching the limit
# forces the extrapolation to proceed without waiting for the mesh to localize any
# further rather than ending the run.
_ROUNDOFF_ACCEL_LIMIT = 5

# Bisections that made the error estimate *worse*, tolerated before the run reports
# roundoff. A larger budget than `_ROUNDOFF1_LIMIT` because a single such bisection says
# much less on its own, and only counted at all once the mesh has grown past
# `_ROUNDOFF2_MIN_NINTER` sub-intervals, below which it is ordinary coarse mesh noise.
_ROUNDOFF2_LIMIT = 20
_ROUNDOFF2_MIN_NINTER = 10

# How far the extrapolated value may sit from the running mesh total before the sequence
# is judged never to have been converging. Applied both ways round, so an extrapolation
# a hundredfold larger and one a hundredfold smaller are equally suspect.
_DIVERGENCE_RATIO = 100.0

# Fraction of the integral of |f| that the result has to reach for the divergence test
# above to mean anything. Below it the integral is the residue of heavy cancellation and
# the ratio is a comparison of two near-zero numbers.
_CANCELLATION_FRAC = 0.01

# Smallest sub-interval width, as a multiple of eps times the half span of the domain,
# that the abscissae can still resolve. Unlike every test above this one is about the
# mesh rather than the values, so it scales with the precision the abscissae are carried
# at rather than the precision of the sums.
_MIN_WIDTH = 100.0


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
    extrapolate: bool = True,
    batch_size: int | None = None,
    throw: bool = False,
):
    """Global adaptive quadrature using Gauss-Kronrod rule.

    Integrate fun from `interval[0]` to `interval[-1]` using a h-adaptive scheme with
    error estimate. Breakpoints can be specified in `interval` where integration
    difficulty may occur.

    Basically the same algorithm as ``scipy.integrate.quad``, including the convergence
    acceleration. The general purpose integrator to reach for first, over finite or
    infinite intervals. It is generally the most robust and often also the most
    efficient, on smooth and non-smooth integrands alike.

    Where an integrand has a jump or a singularity at a known interior point, passing
    that point as a breakpoint in `interval` is worth more than any change of method,
    since the subdivision no longer has to find it.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals. Its dtype sets the working precision: the integrand
        is called with an ``x`` of this dtype, and the result follows it unless the
        integrand upcasts. Integer types or python floats fall back to the JAX
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
    order : int
        Order of local integration rule, one of 15, 21, 31, 41, 51, 61.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    extrapolate : bool, optional
        Whether to accelerate convergence by applying Wynn's epsilon algorithm to the
        sequence of running totals, on by default. Not needed for smooth integrands on
        finite domains, but can help significantly if there are algebraic singularities
        or infinite intervals. The additional cost is small and constant, so it is only
        worth switching off for a very cheap integrand where performance is critical.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. :class:`~quadax.LeibnizAdjoint` gives the
        derivative its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter`` is
        generous; see :ref:`adjoints` for when that is worth paying for.
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel. Default
        is all of the local rule's nodes at once, which is fastest but makes peak memory
        scale with the order. Lower it to reduce memory on an expensive integrand.
        Values larger than the number of nodes are clipped to it.
    throw : bool, optional
        Whether to raise an error if the routine does not converge. If True, a run
        that terminates for any reason other than reaching the requested tolerance
        raises with the message its ``status`` carries. If False, the default, that
        status is reported on the returned ``info`` and left to the caller to act on.

    Returns
    -------
    y : float, Array
        The integral of fun from `a` to `b`.
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (int) Code for why the routine terminated, one of ``quadax.STATUS``.
          ``STATUS.normal`` (0) means the requested tolerances were reached; every other
          code names a difficulty, whose message is ``print(quadax.STATUS[status])``.
          Where a run meets more than one condition the most severe is reported.
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
    is generally not achievable. The local quadrature rule evaluates the integrand at
    all of its nodes at once, so using higher order methods will generally be more
    efficient on GPU/TPU. ``batch_size`` splits that evaluation up where the memory it
    needs is the binding constraint instead.

    """
    rule = GaussKronrodRule(order, norm, batch_size)
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
        extrapolate=extrapolate,
        throw=throw,
    )
    info = QuadratureInfo(
        info.err, info.neval * rule.nodes_per_call, info.status, info.info
    )
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
    extrapolate: bool = True,
    batch_size: int | None = None,
    closed: bool = True,
    throw: bool = False,
):
    """Global adaptive quadrature using Clenshaw-Curtis rule.

    Integrate fun from `interval[0]` to `interval[-1]` using a h-adaptive scheme with
    error estimate. Breakpoints can be specified in `interval` where integration
    difficulty may occur.

    A good general purpose integrator for most reasonably well behaved functions over
    finite or infinite intervals, and a reasonable alternative to
    :func:`~quadax.quadgk`. It's main advantage is in allowing arbitrary high orders,
    which can be useful for smooth but highly oscillatory integrands in the absence of a
    specialized solver, in which case choosing order to have ~7-8 points per period
    is often the most efficient.

    As with :func:`~quadax.quadgk`, an interior jump or singularity is best passed as a
    breakpoint in `interval` rather than left for the subdivision to find.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals. Its dtype sets the working precision: the integrand
        is called with an ``x`` of this dtype, and the result follows it unless the
        integrand upcasts. Integer types or python floats fall back to the JAX
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
    order : int
        Order of local integration rule. Must be a multiple of 4, or with
        ``closed=False`` any even order of at least 4; see
        :class:`~quadax.ClenshawCurtisRule`.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    extrapolate : bool, optional
        Whether to accelerate convergence by applying Wynn's epsilon algorithm to the
        sequence of running totals, on by default. Not needed for smooth integrands on
        finite domains, but can help significantly if there are algebraic singularities
        or infinite intervals. The additional cost is small and constant, so it is only
        worth switching off for a very cheap integrand where performance is critical.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. :class:`~quadax.LeibnizAdjoint` gives the
        derivative its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter`` is
        generous; see :ref:`adjoints` for when that is worth paying for.
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel. Default
        is all of the local rule's nodes at once, which is fastest but makes peak memory
        scale with the order. Lower it to reduce memory on an expensive integrand.
        Values larger than the number of nodes are clipped to it.
    closed : bool, optional
        Whether the interval endpoints are among the nodes of the local rule. The
        default closed rule is cheaper on smooth, peaked and oscillatory integrands. The
        open (Fejer-2) rule never evaluates the integrand at an interval endpoint, which
        is what to use for integrands that are singular or undefined there; it is
        markedly cheaper on infinite intervals whose integrand decays algebraically,
        and on endpoint singularities. See :class:`~quadax.ClenshawCurtisRule`.
    throw : bool, optional
        Whether to raise an error if the routine does not converge. If True, a run
        that terminates for any reason other than reaching the requested tolerance
        raises with the message its ``status`` carries. If False, the default, that
        status is reported on the returned ``info`` and left to the caller to act on.

    Returns
    -------
    y : float, Array
        The integral of fun from `a` to `b`.
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (int) Code for why the routine terminated, one of ``quadax.STATUS``.
          ``STATUS.normal`` (0) means the requested tolerances were reached; every other
          code names a difficulty, whose message is ``print(quadax.STATUS[status])``.
          Where a run meets more than one condition the most severe is reported.
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
    is generally not achievable. The local quadrature rule evaluates the integrand at
    all of its nodes at once, so using higher order methods will generally be more
    efficient on GPU/TPU. ``batch_size`` splits that evaluation up where the memory it
    needs is the binding constraint instead.

    """
    rule = ClenshawCurtisRule(order, norm, batch_size, closed)
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
        extrapolate=extrapolate,
        throw=throw,
    )
    info = QuadratureInfo(
        info.err, info.neval * rule.nodes_per_call, info.status, info.info
    )
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
    extrapolate: bool = False,
    batch_size: int | None = None,
    throw: bool = False,
):
    """Global adaptive quadrature using trapezoidal tanh-sinh rule.

    Integrate fun from `interval[0]` to `interval[-1]` using a h-adaptive scheme with
    error estimate. Breakpoints can be specified in `interval` where integration
    difficulty may occur.

    This can often be the most efficient method for smooth integrands or weak endpoint
    singularities (up to around ``x**-0.5``, including those induced by an algebraically
    decaying integrand on an infinite interval). Beyond that the truncation floor
    in the map (limited by working precision) dominates and no amount of refinement can
    do better. In those cases the extrapolating method :func:`~quadax.quadgk` is
    the more reliable choice.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals. Its dtype sets the working precision: the integrand
        is called with an ``x`` of this dtype, and the result follows it unless the
        integrand upcasts. Integer types or python floats fall back to the JAX
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
    order : int
        Order of local integration rule. Must be odd.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    extrapolate : bool, optional
        Whether to accelerate convergence by applying Wynn's epsilon algorithm to the
        sequence of running totals, off by default. Unlike the other adaptive routines
        this rarely helps here: the tanh-sinh rule converges doubly exponentially, so
        the running totals have no geometric tail for the epsilon algorithm to sum.
        Where a tanh-sinh integration is inaccurate the limit is generally the
        resolution of the abscissas near the endpoints, which acceleration cannot
        recover.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. :class:`~quadax.LeibnizAdjoint` gives the
        derivative its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter`` is
        generous; see :ref:`adjoints` for when that is worth paying for.
    batch_size : int, optional
        Maximum number of points at which to evaluate the integrand in parallel. Default
        is all of the local rule's nodes at once, which is fastest but makes peak memory
        scale with the order. Lower it to reduce memory on an expensive integrand.
        Values larger than the number of nodes are clipped to it.
    throw : bool, optional
        Whether to raise an error if the routine does not converge. If True, a run
        that terminates for any reason other than reaching the requested tolerance
        raises with the message its ``status`` carries. If False, the default, that
        status is reported on the returned ``info`` and left to the caller to act on.

    Returns
    -------
    y : float, Array
        The integral of fun from `a` to `b`.
    info : QuadratureInfo
        Named tuple with the following fields:

        * err : (float) Estimate of the error in the approximation.
        * neval : (int) Total number of function evaluations.
        * status : (int) Code for why the routine terminated, one of ``quadax.STATUS``.
          ``STATUS.normal`` (0) means the requested tolerances were reached; every other
          code names a difficulty, whose message is ``print(quadax.STATUS[status])``.
          Where a run meets more than one condition the most severe is reported.
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
    is generally not achievable. The local quadrature rule evaluates the integrand at
    all of its nodes at once, so using higher order methods will generally be more
    efficient on GPU/TPU. ``batch_size`` splits that evaluation up where the memory it
    needs is the binding constraint instead.

    """
    rule = TanhSinhRule(order, norm, batch_size)
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
        extrapolate=extrapolate,
        throw=throw,
    )
    info = QuadratureInfo(
        info.err, info.neval * rule.nodes_per_call, info.status, info.info
    )
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
    extrapolate: bool = True,
    throw: bool = False,
    **kwargs,
):
    """Global adaptive quadrature with user specified local rule.

    This is a lower level routine allowing for custom local quadrature rules. For most
    applications the higher order methods :func:`~quadax.quadgk`,
    :func:`~quadax.quadcc`, :func:`~quadax.quadts` are preferable.

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
        integrand upcasts. Integer types or python floats fall back to the JAX
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
    extrapolate : bool, optional
        Whether to accelerate convergence by applying Wynn's epsilon algorithm to the
        sequence of running totals, on by default. Not needed for smooth integrands on
        finite domains, but can help significantly if there are algebraic singularities
        or infinite intervals. The additional cost is small and constant, so it is only
        worth switching off for a very cheap integrand where performance is critical.
    adjoint : AbstractAdjoint, optional
        How to compute derivatives of the quadrature. Default is ``DirectAdjoint()``,
        which gives the exact derivative of the discretized problem, and is the
        cheaper option for a cheap integrand. :class:`~quadax.LeibnizAdjoint` gives the
        derivative its own error control (ie, can better approximate the true continuous
        derivative), and is faster when the integrand is expensive or ``max_ninter`` is
        generous; see :ref:`adjoints` for when that is worth paying for.
    throw : bool, optional
        Whether to raise an error if the routine does not converge. If True, a run
        that terminates for any reason other than reaching the requested tolerance
        raises with the message its ``status`` carries. If False, the default, that
        status is reported on the returned ``info`` and left to the caller to act on.
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
        * status : (int) Code for why the routine terminated, one of ``quadax.STATUS``.
          ``STATUS.normal`` (0) means the requested tolerances were reached; every other
          code names a difficulty, whose message is ``print(quadax.STATUS[status])``.
          Where a run meets more than one condition the most severe is reported.
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

    # The options an adjoint may run its own solve with.
    opts = {
        "rule": rule,
        "epsabs": epsabs,
        "epsrel": epsrel,
        "max_ninter": max_ninter,
        "extrapolate": extrapolate,
    }
    ops = QuadratureOps(
        build=partial(build_integrand, f_conv=f_conv),
        solve=_adaptive_solve,
        rebuild=_rebuild_mesh,
        on_mesh=_quad_on_mesh,
        # An accelerated solve may return an extrapolated value rather than the sum over
        # the subdivision, so the fixed-discretization evaluation the adjoints reuse has
        # to replay the extrapolation too, not just the mesh.
        frozen=_frozen_replay if extrapolate else _frozen_mesh,
        frozen_solve=_replay_solve if extrapolate else _mesh_solve,
        mesh_is_primal=not extrapolate,
    )
    y, state = adjoint.quadrature(ops, interval, args, consts, kwargs, opts)

    err = state["err_sum"]
    neval = state["neval"]
    status = state["status"]
    info = state if full_output else None
    out = QuadratureInfo(err, neval, status, info)
    if throw:
        y = error_if_flagged(y, status)
    return y, out


def _at_roundoff_floor(state, epmach, norm):
    """Report whether the error has bottomed out while still above the tolerance.

    The local rule floors each sub-interval's error estimate at the roundoff level over
    that sub-interval, so the total can never fall below that floor summed over the
    partition, and that sum stays at the roundoff level over the whole domain however
    finely the mesh is refined. Once the total has reached the floor and is still above
    ``err_bnd``, no amount of further subdivision will reach the requested tolerance,
    because the tolerance is below what the arithmetic can resolve.

    QUADPACK makes this test only once, before the subdivision loop begins, so a request
    below the achievable precision is left to exhaust the subdivision budget and report
    that instead. quadax makes it every iteration, which stops as soon as the floor is
    reached and reports the actual difficulty.
    """
    intabs = norm(jnp.sum(state["f_arr"], axis=0))
    # Twice the floor, so that a total sitting just above it still reads as having
    # bottomed out rather than as one more iteration's worth of progress away.
    return (state["err_sum"] <= 2 * _ROUNDOFF_FLOOR * epmach * intabs) & (
        state["err_sum"] > state["err_bnd"]
    )


def _accelerate(
    state, i, erro12, err_i, converged, norm, epsabs, epsrel, epmach, max_ninter
):
    """One pass of the extrapolation control flow, skipped where it is a no-op.

    Most iterations of a run that extrapolates are ordinary bisection: the acceleration
    has not started, the pointer into the error ordering is at the head, and the worst
    sub-interval can still be subdivided within the current depth budget. On those the
    whole block below reduces to two updates: which sub-interval to bisect next, and
    the running total of the error still sitting in sub-intervals that are not yet
    localized. The ordering, the epsilon table and the acceptance tests are all
    unchanged.
    """
    # `bisect_next_err_rank == 0` says the pointer never walked down the ordering, so
    # the sub-interval with the largest error is the one at its head and `argmax` finds
    # it without the sort. Ties go the same way: `argsort` is stable, so its first entry
    # and `argmax` both take the lowest index among equal errors.
    bisect_next = jnp.argmax(state["e_arr"])
    # `can_bisect` is the depth test the full pass makes on the worst sub-interval:
    # bisecting it again would keep both halves within the current depth budget, so the
    # mesh still has room to refine there and no extrapolation is called for yet.
    can_bisect = (state["level"][bisect_next] + 1) <= state["level_max"]
    # `ordinary` is the fast path itself, the three conditions under which the full pass
    # would change nothing: the acceleration has not started, the pointer into the
    # error ordering is still at its head so `bisect_next` is the sub-interval the
    # sorted ranking would have picked, and the depth test says to bisect it.
    ordinary = (
        ~state["accelerating"] & (state["bisect_next_err_rank"] == 0) & can_bisect
    )
    # `proceed` says this iteration still has something to record. A run that reached
    # the tolerance or raised a flag is over and reports the state it finished with, so
    # every update below is gated on it.
    proceed = ~converged & (state["status"] == STATUS.normal)
    # Depth of the two halves the bisection just created. Both were recorded before this
    # is called, so slot `i` carries it.
    levcur = state["level"][i]

    def skip(state):
        active = proceed & ~state["no_accel"]
        # remove error from parent that was just bisected
        err_unlocalized = state["err_unlocalized"] - err_i
        # add the child error in, but if the halves are at max depth, we treat the
        # error as localized and don't try further bisection. remaining error is
        # left to extrapolation.
        err_unlocalized += jnp.where(levcur + 1 <= state["level_max"], erro12, 0)
        state["bisect_next"] = jnp.where(proceed, bisect_next, state["bisect_next"])
        state["err_unlocalized"] = jnp.where(
            active, err_unlocalized, state["err_unlocalized"]
        )
        return state

    def run(state):
        return _accelerate_full(
            state,
            i,
            levcur,
            erro12,
            err_i,
            converged,
            norm,
            epsabs,
            epsrel,
            epmach,
            max_ninter,
        )

    return jax.lax.cond(unvmap_any(~ordinary), run, skip, state)


def _accelerate_full(
    state, i, levcur, erro12, err_i, converged, norm, epsabs, epsrel, epmach, max_ninter
):
    """One pass of the extrapolation control flow.

    Runs at the end of a bisection and referees between the two things that want the
    next one: the subdivision, and the table that infers the limit of the sequence of
    running totals the subdivision produces.

    Bisecting wherever the error estimate is largest would home in on any singular area,
    and left to itself would go on halving there and never touch the rest of the domain.
    That refinement near the singularity is what the table wants, but we don't want it
    contaminated by error from elsewhere in the domain. In order for extrapolation to
    work, we must have a sequence of estimates where the error is dominated by the
    singular region.

    What the table constrains is not where to bisect but when to take a reading. Each
    term it is fed has to be a clean sample of one process: the integral with the
    difficulty resolved to a given depth and the rest of the domain already tidy. A term
    taken while the rest is still coarse carries two errors at once, the geometrically
    decaying tail and leftover smooth error following no such pattern, and a sequence of
    those has no trend in it to extrapolate.

    So the override runs the opposite way round to what the competition suggests. It is
    the extrapolation that causes sub-intervals well down the error ranking to be
    bisected, ones the subdivision would never choose for itself, while the difficult
    region is held frozen. It is not frozen for long (it is deepened once per round) but
    on a schedule rather than whenever it happens to carry the largest error, and that
    is what makes each term comparable to the one before it.

    A round therefore runs in four stages, numbered here and in the body below. One pass
    carries out one stage; the loop comes back here after every bisection, and the
    stages advance across those visits.

    1. *Has the mesh localized?* Let the subdivision home in. While the worst
       sub-interval is still within the depth budget this is the ordinary adaptive loop
       and nothing else happens. Once it reaches the budget the difficult region is
       resolved as tightly as this round allows, and is frozen.

    2. *Is anything else worth bisecting first?* Clean up elsewhere, which is what earns
       the coming reading the right to be taken. A sub-interval further down the ranking
       that still has depth left is bisected in preference to feeding the table, for as
       long as enough error remains in such sub-intervals to be worth collecting.

       This is not a sweep that levels the domain. The test is on their *total* error
       and each pass takes the largest of them, so the cleanup stops partway down the
       ranking and sub-intervals whose error is already negligible are never reached.
       What it drains towards is the caller's own tolerance, so that the part of the
       domain still being subdivided is inside the whole error budget and the
       extrapolation is left accounting for the frozen part alone. Cleanup also ends
       early when nothing is left with depth to spare, which forces the reading however
       much error remains.

    3. *Is the extrapolation good enough to stop on?* Take the reading: the running
       total goes to the table, and what comes back is kept if it improves on the best
       so far, and stops the run if it also meets the tolerance. There are two ways to
       give up here instead: a sequence that has stopped improving, and a table with
       nothing left in it.

    4. Otherwise raise the depth budget by one, unfreeze the difficult region, and begin
       the next round against a mesh allowed to localize one level further.

    Which sub-interval to bisect is therefore settled at the *end* of an iteration
    rather than the start, which is why it is carried in the state rather than
    recomputed from the error estimates. Whether the extrapolated value is returned at
    all is not settled here; see ``_accept_extrapolation``.

    Note that under vmap this is run if *any* vmapped element needs acceleration,
    so still needs to be a no-op for those that don't. The cond in _accelerate only
    skips if all elements don't need it.
    """
    # --- Setup: the ranking, the gating flags, and the unlocalized error ------------
    # The acceleration needs the sub-intervals ranked by error estimate, not just the
    # worst one: once it starts extrapolating it walks down the ranking looking for a
    # sub-interval that is still worth bisecting.
    order = jnp.argsort(-state["e_arr"])
    # The pointer must never sit below the sub-interval just bisected, or the walk would
    # start past an error larger than any it can then find. Bisection does not always
    # reduce an error estimate (two halves of an unresolved sub-interval can between
    # them report more error than their parent did) so slot `i` may have moved *up*
    # the ranking, and the pointer is clamped to follow it up when it does.
    bisect_next_err_rank = jnp.minimum(
        state["bisect_next_err_rank"], jnp.argmax(order == i)
    )
    state["bisect_next_err_rank"] = bisect_next_err_rank
    bisect_next = order[bisect_next_err_rank]

    # Everything below is skipped on an iteration that reached the tolerance or raised a
    # flag: in both cases the run is over and the mesh result is the one that will be
    # reported. It is skipped for good once `no_accel` is set, which abandons the
    # acceleration and lets the run finish as an ordinary subdivision.
    proceed = ~converged & (state["status"] == STATUS.normal)
    active = proceed & ~state["no_accel"]

    # The error still sitting in sub-intervals that are not yet localized, ie those the
    # subdivision has not yet driven down to the current depth. The parent's share
    # leaves it, and the children's returns only if they are still large enough to be
    # worth subdividing.
    err_unlocalized = state["err_unlocalized"] - err_i
    err_unlocalized += jnp.where(levcur + 1 <= state["level_max"], erro12, 0)
    err_unlocalized = jnp.where(active, err_unlocalized, state["err_unlocalized"])

    # --- 1. Has the mesh localized? -------------------------------------------------
    # While the worst sub-interval can still be subdivided within the current depth
    # budget there is more to be had from refining the mesh, and this stays the ordinary
    # adaptive loop.
    can_bisect = (state["level"][bisect_next] + 1) <= state["level_max"]
    keep_bisecting = active & ~state["accelerating"] & can_bisect
    # whether to start accelerating now
    begin = active & ~state["accelerating"] & ~can_bisect
    # whether we are now accelerating, ie have started extrapolating
    accelerating = state["accelerating"] | begin
    bisect_next_err_rank = jnp.where(begin, 1, bisect_next_err_rank)

    # --- 2. Is something else worth bisecting first? --------------------------------
    # Before extrapolating, look further down the ranking for a sub-interval that still
    # has room to bisect within the current depth budget. Bisecting one of those brings
    # the unlocalized error down without re-refining the region the table is already
    # extrapolating past; refining that region instead would move the running total by
    # the very tail the extrapolation is inferring, so the sequence would stop being the
    # smoothly converging one the epsilon algorithm assumes. The search starts at the
    # pointer and runs no further than the subdivisions still available, since lower
    # ranks can never be reached before the budget runs out.
    last = state["ninter"]
    jupbnd = jnp.where(last > 2 + max_ninter // 2, max_ninter + 3 - last, last)
    ranks = jnp.arange(max_ninter)
    can_bisect_ranked = (state["level"][order] + 1) <= state["level_max"]
    candidate = can_bisect_ranked & (ranks >= bisect_next_err_rank) & (ranks < jupbnd)
    # A table already known to be running on a stagnant sequence skips the search: more
    # subdivision has been shown not to help it.
    #
    # The threshold is floored at the roundoff level. It decides when the error left in
    # the unlocalized sub-intervals has become small enough that extrapolating past them
    # is worthwhile, which is a control decision rather than a convergence test, and a
    # caller asking for a tolerance below what the arithmetic can deliver (`epsabs=0`
    # as shorthand for "do your best") would otherwise leave it permanently false.
    # Every pass would then find something else to bisect and the table would hardly be
    # fed, so the answer would fall back to what the mesh alone can do. QUADPACK
    # never meets this because it refuses such a tolerance at input validation; quadax
    # clamps rather than refuses, so that "do your best" means what it says. Only the
    # *search* is floored, the acceptance test below still uses the tolerance as
    # asked, so this can never report success on an accuracy that was not reached.
    accel_target = jnp.maximum(
        state["err_accel_target"], _ROUNDOFF_FLOOR * epmach * norm(state["area"])
    )
    search = ~state["roundoff_in_table"] & (err_unlocalized > accel_target)
    found = active & search & ~keep_bisecting & jnp.any(candidate)
    found_rank = jnp.argmax(candidate)
    bisect_next_err_rank = jnp.where(found, found_rank, bisect_next_err_rank)
    bisect_next = jnp.where(found, order[found_rank], bisect_next)

    # --- 3. Is the extrapolation good enough to stop on? ----------------------------
    take_step = active & ~keep_bisecting & ~found

    # Feed the running total to the table.
    fed = _acceleration.step(state["accel_table"], state["area"], norm)
    accel_table = tree_where(take_step, fed, state["accel_table"])

    # A pass that only recorded its value did not extrapolate, if so don't judge it.
    ran = take_step & _acceleration.ready(state["accel_table"])
    n_stalled = jnp.where(ran, state["n_stalled"] + 1, state["n_stalled"])
    # The table has been asked `_STALL_LIMIT` times running for something better than
    # it already has and has not produced it, while claiming an error far under what the
    # mesh reports. Nothing further is going to come of it.
    stalled = (
        ran
        & (n_stalled > _STALL_LIMIT)
        & (state["accel_sharp"] < _STALL_SHARP * state["err_sum"])
    )

    # QUADPACK ranks candidate extrapolations by the same error estimate it reports.
    # That is only safe while the two are the same number. Here the reported one carries
    # a tail term that is largest exactly where the table is struggling, so ranking on
    # it would let that term decide which value is kept and make the *answer* worse
    # rather than only its error bar. Which one settled tightest and how far it might
    # still be from the limit are different questions, and only the first should choose.
    improved = ran & (fed.abserr_sharp < state["accel_sharp"])
    n_stalled = jnp.where(improved, 0, n_stalled)
    accel_result = jnp.where(improved, fed.result, state["accel_result"])
    accel_sharp = jnp.where(improved, fed.abserr_sharp, state["accel_sharp"])
    accel_err = jnp.where(improved, fed.abserr, state["accel_err"])
    # Not in QUADPACK, which reports the estimate the kept extrapolation came with and
    # never revisits it. The value kept is the one that settled tightest, which selects
    # the most optimistic reading of a quantity carrying no bound, and every later
    # extrapolation is evidence about that choice: if the sequence has since moved
    # further from the kept value than its estimate allows, one of the two is wrong by
    # at least the difference and the estimate has to cover it. Without this a single
    # optimistic reading is reported for the rest of the run however far the table
    # wanders afterwards, which is most of why the reported error was chaotic on
    # problems the acceleration cannot fit.
    disagreement = jnp.asarray(norm(fed.result - accel_result))
    accel_err = jnp.asarray(
        jnp.where(ran, jnp.maximum(accel_err, disagreement), accel_err)
    )
    # If the error from the non-singular region was flat  but large then the error
    # estimate from the table is too optimistic, because it assumed that error would
    # continue to decay geometrically. The error from the non-singular region is still
    # there, so add it back to the table's estimate. This is just the amount to add,
    # the decision to add it is made later.
    correc = jnp.where(improved, err_unlocalized, state["correc"])
    # on the next round, we want the unlocalized error to be smaller in order to get
    # another clean reading from the table.
    err_accel_target = jnp.asarray(
        jnp.where(
            improved,
            jnp.maximum(epsabs, epsrel * norm(fed.result)),
            state["err_accel_target"],
        )
    )
    # Can we exit with the current estimate? QUADPACK tests this only on a pass that
    # improved on the estimate it already had, which works there because ranking and
    # reporting are the same number, so the pass that lowers the estimate is exactly the
    # pass that could bring it under the tolerance. Here they are separate, and tying
    # the exit to both events coinciding strands a run that has already arrived: a table
    # that has converged stops improving, so it would carry on to a stall and report
    # failure over an answer that met the tolerance several passes earlier. The question
    # is about the value being kept, not about the pass that produced it.
    accepted = ran & (accel_err < err_accel_target)
    done = accepted | stalled

    # The epsilon table truncates itself when its differences stop being safe to divide
    # by, and can be left holding a single entry. There is nothing to extrapolate from
    # then, and refilling it means feeding it the same running totals that emptied it,
    # so the acceleration is abandoned and the run finishes as a plain subdivision.
    no_accel = state["no_accel"] | (ran & ~accepted & (fed.n == 1))

    # --- 4. Raise the depth budget and begin the next round -------------------------
    # go back to the largest error, allow the subdivision one more level of
    # depth, and let the mesh localize further before the next extrapolation.
    reset = take_step & ~done
    bisect_next = jnp.where(reset, order[0], bisect_next)
    bisect_next_err_rank = jnp.where(reset, 0, bisect_next_err_rank)
    accelerating &= ~reset
    level_max = jnp.where(reset, state["level_max"] + 1, state["level_max"])
    err_unlocalized = jnp.where(reset, state["err_sum"], err_unlocalized)

    # --- Writeback ------------------------------------------------------------------
    # Which passes fed the table, and which extrapolation was the one kept. Read only by
    # `_replay_solve`; the slot this bisection created labels the step.
    n_append = state["n_append"] + take_step
    updates = {
        "append_mask": state["append_mask"].at[state["ninter"] - 1].set(take_step),
        "n_append": n_append,
        "accel_ncall": jnp.where(improved, n_append, state["accel_ncall"]),
        "bisect_next": bisect_next,
        "bisect_next_err_rank": bisect_next_err_rank,
        "accelerating": accelerating,
        "level_max": level_max,
        "err_unlocalized": err_unlocalized,
        "err_accel_target": err_accel_target,
        "correc": correc,
        "accel_table": accel_table,
        "accel_result": accel_result,
        "accel_err": accel_err,
        "accel_sharp": accel_sharp,
        "n_stalled": n_stalled,
        "accel_done": done,
        "no_accel": no_accel,
    }
    for key, value in updates.items():
        state[key] = tree_where(proceed, value, state[key])
    state["status"] = escalate(state["status"], STATUS.no_converge, stalled)
    return state


def _accept_extrapolation(state, mesh_y, norm):
    """Choose between the extrapolated value and the mesh sum, once the loop has ended.

    The extrapolated value is not automatically preferred. It carries no rigorous bound,
    only the spread of the last three extrapolants, so it is kept only where its
    *relative* error estimate beats the one the mesh reports honestly -- and where the
    subdivision raised no flag at all, the comparison is not even made, because then the
    mesh result is the one with the trustworthy bound.
    """
    accel_y = state["accel_result"]
    mesh_err = state["err_sum"]
    # An error still at its starting value means no extrapolation was ever accepted.
    # Neither is one wanted where the subdivision reached the tolerance on its own: the
    # mesh result is then the one carrying a real error bound, and there is nothing to
    # be gained by replacing it with a heuristic. Without this a table fed early, while
    # the mesh was still coarse, can displace a converged answer with a much worse one
    # and still report success.
    have_accel = jnp.isfinite(state["accel_err"]) & (mesh_err > state["err_bnd"])
    # Where the table was running on a sequence that had stopped improving, its own
    # estimate understates the error by whatever was still outstanding in the
    # sub-intervals it had passed over, since the extrapolation assumed that outstanding
    # amount would be recovered by the trend. Add it back.
    accel_err = jnp.where(
        state["roundoff_in_table"],
        state["accel_err"] + state["correc"],
        state["accel_err"],
    )
    # Whether anything was flagged at all, by the subdivision or by the table.
    flagged = (state["status"] != STATUS.normal) | state["roundoff_in_table"]

    scale_accel = norm(accel_y)
    scale_mesh = norm(mesh_y)
    tiny = float(jnp.finfo(mesh_err.dtype).tiny)
    degenerate = (scale_accel == 0) | (scale_mesh == 0)
    accel_rel_err = accel_err / jnp.maximum(scale_accel, tiny)
    mesh_rel_err = mesh_err / jnp.maximum(scale_mesh, tiny)
    worse = accel_rel_err > mesh_rel_err
    use_mesh = ~have_accel | (
        flagged & jnp.where(degenerate, accel_err > mesh_err, worse)
    )
    # A mesh sum of exactly zero leaves nothing to compare a ratio against, so the
    # extrapolated value is returned without being tested for divergence.
    untestable = degenerate & ~(accel_err > mesh_err) & (scale_mesh == 0)

    # Did the extrapolated value run away from the running total? An extrapolation that
    # differs from the mesh by `_DIVERGENCE_RATIO` either way, or comes out with the
    # opposite sign, or a mesh whose own error estimate exceeds the size of what it
    # estimates, all say the same thing: the sequence was never converging, and what the
    # recursion inferred a limit from was noise. QUADPACK states the magnitude and sign
    # parts as one signed ratio; the sign half is written out separately here so that it
    # carries over to vector and complex integrands, where a signed ratio has no
    # meaning. For a real scalar the two forms agree exactly.
    ratio = scale_accel / jnp.maximum(scale_mesh, tiny)
    opposed = jnp.real(jnp.sum(accel_y * jnp.conj(mesh_y))) < 0
    diverging = (
        (ratio < 1 / _DIVERGENCE_RATIO)
        | (ratio > _DIVERGENCE_RATIO)
        | opposed
        | (mesh_err > scale_mesh)
    )
    # Where the integral is the residue of heavy cancellation the ratio is a comparison
    # of two near-zero numbers and means nothing, so the test is made only when the
    # result is an appreciable fraction of the integral of |f|.
    testable = state["sign_known"] | (
        jnp.maximum(scale_accel, scale_mesh) > _CANCELLATION_FRAC * state["abs_total"]
    )
    divergent = have_accel & ~use_mesh & ~untestable & testable & diverging

    # Roundoff detected inside the table, where the subdivision itself reported nothing.
    state["status"] = escalate(
        state["status"],
        STATUS.roundoff,
        have_accel & flagged & (state["status"] == STATUS.normal) & ~use_mesh,
    )
    state["status"] = escalate(state["status"], STATUS.divergent, divergent)
    state["used_accel"] = have_accel & ~use_mesh
    state["err_sum"] = jnp.where(use_mesh, mesh_err, accel_err)
    # QUADPACK never clears a flag once raised, which costs it nothing because its
    # stall test and its exit test read the same estimate: a run that stalls there has
    # by construction not met the tolerance. Splitting the two makes a stall possible on
    # a run that did meet it, so the flag has to be withdrawn or the routine reports
    # failure over an answer inside tolerance. A stall says the table stopped making
    # progress, which is a statement about how the answer was reached rather than about
    # the answer; if the error attached to what is returned meets the bound asked for
    # then the request was met, whichever of the two supplied it. Only this flag is
    # cleared, since the others describe the returned value and stand on their own.
    reached = state["err_sum"] <= state["err_bnd"]
    state["status"] = withdraw(state["status"], STATUS.no_converge, reached)
    return state, jnp.where(use_mesh, mesh_y, accel_y)


def _init_state(interval, shape, xtype, ytype, etype, max_ninter, extrapolate):
    """State of the subdivision loop before the initial sub-intervals are evaluated.

    The mesh holds the sub-intervals ``interval`` was given with, and every running
    quantity is at its identity. ``shape`` and the three dtypes are those of the
    integrand's value, the abscissae and the error estimates respectively.

    Extrapolation state is present only when ``extrapolate`` is set, so that a run
    without it traces to a loop carrying none of it.
    """
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
    state["status"] = STATUS.normal  # why the run stopped
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

    if not extrapolate:
        return state

    # How deep in the subdivision each sub-interval sits, ie how many bisections
    # separate it from the original sub-interval containing it. This is the measure
    # of whether the mesh has localized: a sub-interval that has been bisected
    # `level_max` times is treated as resolved, and the loop leaves it alone and
    # extrapolates past it instead. Depth rather than width, because with
    # breakpoints the original sub-intervals can differ in width and a single width
    # threshold across the whole domain would declare the narrow ones resolved before
    # they had been touched.
    state["level"] = jnp.zeros(max_ninter, int)  # depth of each sub-interval
    state["level_max"] = 1  # depth at which a sub-interval counts as resolved
    # Which sub-interval to bisect next, and its rank in the error ordering (when
    # extrapolation is used we don't always bisect the largest error). Both are chosen
    # at the *end* of an iteration, since the choice is part of the extrapolation
    # control flow, so they are carried rather than recomputed from `e_arr` at the top
    # of the body.
    state["bisect_next"] = jnp.zeros((), int)  # slot
    state["bisect_next_err_rank"] = jnp.zeros(
        (), int
    )  # its 0-based rank by error estimate

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
    # The stagnation count accumulated *while extrapolating* is kept apart from the
    # one accumulated before, because five of them mean something quite specific:
    # the table is being fed a sequence that has stopped improving, and its error
    # estimate has to be widened by `correc` at the end.
    state["roundoff_accel"] = jnp.zeros((), int)
    state["roundoff_in_table"] = jnp.zeros((), bool)  # the sequence has stagnated
    state["correc"] = jnp.zeros((), etype)  # what to widen `accel_err` by if it has
    state["err_accel_target"] = jnp.zeros((), etype)  # tolerance to accept one at
    # total error that we think can still be reduced by subdivision.
    state["err_unlocalized"] = jnp.zeros((), etype)
    state["accel_done"] = jnp.zeros((), bool)  # the extrapolation block's exits
    # Sub-intervals whose local rule saturated on the first pass.
    state["ndin"] = jnp.zeros(max_ninter, bool)
    # Bookkeeping for `_replay_solve`, which is how an accelerated solve is
    # differentiated. None of it is read by the integrator itself. The parent arrays
    # and the birth times are indexed by the slot a bisection creates, which is
    # unique to that step; `birth` is indexed by slot and holds the birth time of
    # whatever occupies it now. Time is counted in sub-intervals rather than steps,
    # so it starts at the initial count.
    state["birth"] = jnp.full(max_ninter, state["ninter"], int)
    state["p_owner"] = jnp.zeros(max_ninter, int)
    state["p_frac_a"] = jnp.zeros(max_ninter, xtype)
    state["p_frac_b"] = jnp.zeros(max_ninter, xtype)
    state["p_birth"] = jnp.zeros(max_ninter, int)
    state["append_mask"] = jnp.zeros(max_ninter, bool)
    state["n_append"] = jnp.zeros((), int)
    state["accel_ncall"] = jnp.zeros((), int)
    return state


def _adaptive_solve(
    vfunc,
    interval,
    kwargs,
    *,
    rule,
    epsabs,
    epsrel,
    max_ninter,
    extrapolate=False,
    norm=None,
):
    """Run the globally adaptive subdivision loop.

    With ``extrapolate=False`` this bisects whichever sub-interval currently has the
    largest error estimate, until the errors sum to less than the tolerance.

    With ``extrapolate=True`` the same subdivision runs, but the sequence of running
    totals it produces is also fed to Wynn's epsilon algorithm, and the limit that
    infers may be returned in place of the sum over the mesh. That changes which
    sub-interval to bisect: the sequence is only extrapolable if its terms keep coming
    from the same process, so once the mesh has localized onto the difficulty the loop
    stops refining there and works on the rest of the domain instead, feeding the table
    a term each time it does. See ``_acceleration`` for the table itself,
    ``_accelerate_full`` for the control flow that decides all this, and
    ``_accept_extrapolation`` for the choice between the two answers at the end.

    ``norm`` replaces the one the rule was built with, for a caller that measures a
    vector other than the integrand's output. ``None``, the primal's case, keeps the
    rule's own.
    """
    if norm is not None:
        rule = rule._with_norm(norm)
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
    # "Too narrow" is only meaningful against the span being subdivided.
    halfspan = jnp.abs(interval[-1] - interval[0]) / 2

    state = _init_state(interval, shape, xtype, ytype, etype, max_ninter, extrapolate)

    def init_body(i, state):
        a = state["a_arr"][i]
        b = state["b_arr"][i]
        result, abserr, intabs, intmmn = intfun(vfunc, a, b, ())

        if extrapolate:
            # An original sub-interval whose error estimate reached the saturation
            # value (the whole variation of the integrand over it) told the rule
            # nothing about it at all. Those are promoted to the head of the error
            # ordering below, so that the pieces the caller flagged as difficult by
            # putting a breakpoint at them are the first ones bisected. The test is
            # ``>=`` and not equality because a rule may add to the saturated value,
            # as a tanh-sinh one does with the mass beyond its outermost node. An
            # integrand with no variation to saturate against is excluded rather than
            # counted as unresolved, the estimate then being a roundoff floor sitting
            # above a variation of zero.
            variation = _norm(intmmn)
            state["ndin"] = (
                state["ndin"].at[i].set((abserr >= variation) & (variation != 0))
            )

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
    state["status"] = escalate(
        state["status"], STATUS.roundoff, _at_roundoff_floor(state, epmach, _norm)
    )

    # check for max intervals exceeded
    state["status"] = escalate(
        state["status"], STATUS.max_ninter, state["ninter"] >= max_ninter
    )

    if extrapolate:
        # Give the saturated sub-intervals the whole error estimate, which puts them at
        # the head of the ordering however small their own contribution was. This comes
        # after the roundoff check on purpose: that check asks whether the honest sum of
        # the local error estimates has already bottomed out at the arithmetic's floor,
        # and inflating one of them first would hide that.
        state["e_arr"] = jnp.where(
            state["ndin"], jnp.sum(state["e_arr"]), state["e_arr"]
        )
        state["err_sum"] = jnp.sum(state["e_arr"])
        # Total integral of |f| over the whole domain, as the initial mesh sees it. Only
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
        # The total over the initial mesh is the first term of the sequence. It seeds
        # the table directly, with no extrapolation performed on it, so it must not
        # count towards `n_calls`.
        state["accel_table"] = _acceleration.append(state["accel_table"], state["area"])

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
        # bisect the sub-interval with the bisect_next_err_rank-th largest error
        # estimate. Without extrapolation that is always the largest, and the ordering
        # is not needed.
        if extrapolate:
            i = state["bisect_next"]
        else:
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
        # overwritten below. The stagnation test further down compares the two halves
        # against the *parent*, so these have to be captured here rather than read back
        # out of the arrays.
        area_i = state["r_arr"][i]
        err_i = state["e_arr"][i]

        # Which half keeps slot `i` and which takes the new slot `n`: the larger error
        # goes into `i`, the slot the ordering was already pointing at, so that the two
        # halves are ranked correctly relative to each other without consulting the rest
        # of the mesh. Only the placement depends on this, not either total, so both
        # arrays are written here and the branch at the end of the body is left with the
        # endpoints and the fractions.
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
        # rather than of roundoff, and neither counter below should fire. Without this a
        # hard but tractable integrand accumulates stagnation counts while it is still
        # making legitimate progress. The equality is exact on purpose: it asks whether
        # `min(1, ...)` clamped, not whether two quantities are merely close. Both sides
        # go through `_norm` because that is the reduction the error estimate itself
        # already went through for vector valued integrands.
        resolved = (error1 != _norm(intmmn1)) & (error2 != _norm(intmmn2))

        # test for roundoff error
        # is the area estimate not changing and error not getting smaller?
        # `_STAGNANT_RTOL` is only meaningful while it sits above the noise floor of
        # the difference it is applied to, ~eps*|area12|. It does at float32 and float64
        # (the roundoff level is 6e-6 and 1.1e-14 there, so the maximum is
        # `_STAGNANT_RTOL` either way and this is QUADPACK's test unchanged) but not in
        # half precision, where it would sit below the noise and this counter could
        # never fire.
        stagnant = max(_STAGNANT_RTOL, _ROUNDOFF_FLOOR * epmach)
        stagnated = (
            resolved
            & (_norm(area_i - area12) <= stagnant * _norm(area12))
            & (erro12 >= _NO_PROGRESS * err_i)
        )
        state["roundoff1"] += stagnated
        # are errors getting larger as we go to smaller intervals?
        state["roundoff2"] += (
            resolved & (state["ninter"] > _ROUNDOFF2_MIN_NINTER) & (erro12 > err_i)
        )

        if extrapolate:
            # The same stagnation events, counted again over the extrapolating phase
            # alone. Reaching the limit says the table is being fed a sequence that has
            # stopped improving, which both forces the extrapolation to go ahead without
            # waiting for the mesh to localize any further, and widens the error it
            # reports.
            state["roundoff_accel"] += stagnated & state["accelerating"]
            state["roundoff_in_table"] |= (
                state["roundoff_accel"] >= _ROUNDOFF_ACCEL_LIMIT
            )
            # Both halves sit one level deeper than the sub-interval they came from.
            levcur = state["level"][i] + 1
            state["level"] = state["level"].at[i].set(levcur).at[n].set(levcur)
            # Record the sub-interval this step consumed, and when the three
            # sub-intervals involved entered the running total. Together with the final
            # subdivision this is every sub-interval that ever existed, which is what
            # `_replay_solve` needs to rebuild the sequence the table was fed. None of
            # it depends on which half ends up in which slot: the two halves are born
            # together and are only ever summed.
            state["p_owner"] = state["p_owner"].at[n].set(state["owner"][i])
            state["p_frac_a"] = state["p_frac_a"].at[n].set(state["frac_a"][i])
            state["p_frac_b"] = state["p_frac_b"].at[n].set(state["frac_b"][i])
            state["p_birth"] = state["p_birth"].at[n].set(state["birth"][i])
            state["birth"] = state["birth"].at[i].set(n + 1).at[n].set(n + 1)

        # Whether the tolerance was reached on this iteration. Reaching it takes
        # precedence over every flag below, so an iteration that both reaches the
        # tolerance and, say, consumes the last subdivision slot still exits cleanly:
        # the answer is good, and what it cost getting there is not a failure. The
        # counters above are still updated either way.
        converged = state["err_sum"] <= state["err_bnd"]

        # Roundoff is reported either because the error has bottomed out at the floor
        # the arithmetic imposes, or because the two counters say subdivision has
        # stopped buying anything.
        state["status"] = escalate(
            state["status"],
            STATUS.roundoff,
            ~converged
            & (
                _at_roundoff_floor(state, epmach, _norm)
                | (state["roundoff1"] >= _ROUNDOFF1_LIMIT)
                | (state["roundoff2"] >= _ROUNDOFF2_LIMIT)
            ),
        )

        # test for max number of intervals
        state["status"] = escalate(
            state["status"],
            STATUS.max_ninter,
            ~converged & (state["ninter"] >= max_ninter),
        )

        # test for bad behavior of the integrand (ie, intervals are getting too small)
        state["status"] = escalate(
            state["status"],
            STATUS.bad_integrand,
            ~converged
            & (
                jnp.maximum(jnp.abs(b1 - a1), jnp.abs(b2 - a2))
                <= (_MIN_WIDTH * epmach_x * halfspan)
            ),
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
                max_ninter,
            )
        return state

    state = bounded_while_loop(condfun, bodyfun, state, max_ninter + 1)

    y = jnp.sum(state["r_arr"], axis=0)
    if extrapolate:
        state, y = _accept_extrapolation(state, y, _norm)
    return y, state
