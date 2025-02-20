"""Functions for globally h-adaptive quadrature."""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp

from .fixed_order import (
    AbstractQuadratureRule,
    ClenshawCurtisRule,
    GaussKronrodRule,
    TanhSinhRule,
)
from .utils import (
    QuadratureInfo,
    bounded_while_loop,
    errorif,
    map_interval,
    setdefault,
    wrap_func,
)

NORMAL_EXIT = 0
MAX_NINTER = 1
ROUNDOFF = 2
BAD_INTEGRAND = 3
NO_CONVERGE = 4
DIVERGENT = 5


@eqx.filter_jit
def quadgk(
    fun,
    interval,
    args=(),
    full_output=False,
    epsabs=None,
    epsrel=None,
    max_ninter=50,
    order=21,
    norm=jnp.inf,
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
    )
    info = QuadratureInfo(info.err, info.neval * order, info.status, info.info)
    return y, info


@eqx.filter_jit
def quadcc(
    fun,
    interval,
    args=(),
    full_output=False,
    epsabs=None,
    epsrel=None,
    max_ninter=50,
    order=32,
    norm=jnp.inf,
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
    )
    info = QuadratureInfo(info.err, info.neval * order, info.status, info.info)
    return y, info


@eqx.filter_jit
def quadts(
    fun,
    interval,
    args=(),
    full_output=False,
    epsabs=None,
    epsrel=None,
    max_ninter=50,
    order=61,
    norm=jnp.inf,
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
    )
    info = QuadratureInfo(info.err, info.neval * order, info.status, info.info)
    return y, info


@eqx.filter_jit
def adaptive_quadrature(
    rule,
    fun,
    interval,
    args=(),
    full_output=False,
    epsabs=None,
    epsrel=None,
    max_ninter=50,
    norm=jnp.inf,
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
    if not isinstance(rule, AbstractQuadratureRule):
        warnings.warn(
            "Passing a callable for ``rule`` is deprecated and in the future will "
            "raise an error. Users should instead subclass "
            "``quadax.AbstractQuadratureRule``",
            FutureWarning,
        )
        intfun = rule
        norm = kwargs.pop("norm", jnp.inf)
        _norm = (
            norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
        )
    else:
        intfun = rule.integrate
        _norm = rule.norm
    errorif(
        max_ninter < len(interval) - 1,
        ValueError,
        f"max_ninter={max_ninter} is not enough for {len(interval)-1} breakpoints",
    )
    epsabs = setdefault(epsabs, jnp.sqrt(jnp.finfo(jnp.array(1.0)).eps))
    epsrel = setdefault(epsrel, jnp.sqrt(jnp.finfo(jnp.array(1.0)).eps))
    fun, interval = map_interval(fun, interval)
    vfunc = wrap_func(fun, args)
    f = jax.eval_shape(vfunc, (interval[0] + interval[-1]) / 2)
    epmach = jnp.finfo(f.dtype).eps
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

    def init_body(i, state_):
        state, intabs_ = state_
        a = state["a_arr"][i]
        b = state["b_arr"][i]
        result, abserr, intabs, intmmn = intfun(vfunc, a, b, (), **kwargs)

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
        state["ninter"] += 1

        # bisect the sub-interval with the largest error estimate.
        i = jnp.argmax(state["e_arr"])
        n = state["ninter"]
        a1 = state["a_arr"][i]
        b1 = 0.5 * (state["a_arr"][i] + state["b_arr"][i])
        a2 = b1
        b2 = state["b_arr"][i]

        area1, error1, intabs1, intmmn1 = intfun(vfunc, a1, b1, (), **kwargs)
        state["neval"] += 1
        area2, error2, intabs2, intmmn2 = intfun(vfunc, a2, b2, (), **kwargs)
        state["neval"] += 1

        # ! improve previous approximations to integral
        # ! and error and test for accuracy.

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
        state["roundoff2"] += (n > 10) & (erro12 > jnp.max(state["e_arr"]))
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

        def error1big(state):
            state["a_arr"] = state["a_arr"].at[n].set(a2)
            state["b_arr"] = state["b_arr"].at[i].set(b1)
            state["b_arr"] = state["b_arr"].at[n].set(b2)
            state["e_arr"] = state["e_arr"].at[i].set(error1)
            state["e_arr"] = state["e_arr"].at[n].set(error2)
            return state

        def error2big(state):
            state["a_arr"] = state["a_arr"].at[i].set(a2)
            state["a_arr"] = state["a_arr"].at[n].set(a1)
            state["b_arr"] = state["b_arr"].at[n].set(b1)
            state["r_arr"] = state["r_arr"].at[i].set(area2)
            state["r_arr"] = state["r_arr"].at[n].set(area1)
            state["e_arr"] = state["e_arr"].at[i].set(error2)
            state["e_arr"] = state["e_arr"].at[n].set(error1)
            return state

        state = jax.lax.cond(error2 > error1, error2big, error1big, state)
        return state

    state = bounded_while_loop(condfun, bodyfun, state, max_ninter + 1)

    y = jnp.sum(state["r_arr"], axis=0)
    err = state["err_sum"]
    neval = state["neval"]
    status = state["status"]
    info = state if full_output else None
    out = QuadratureInfo(err, neval, status, info)
    return y, out
