"""Functions for globally h-adaptive quadrature."""

import jax
import jax.numpy as jnp

from .fixed_qk import fixed_quadgk
from .utils import map_interval

NORMAL_EXIT = 0
MAX_NINTER = 1
ROUNDOFF = 2
BAD_INTEGRAND = 3
NO_CONVERGE = 4
DIVERGENT = 5


def quadgk(
    fun,
    a,
    b,
    args=(),
    full_output=False,
    epsabs=1.4e-8,
    epsrel=1.4e-8,
    max_ninter=50,
    order=21,
):
    """Global adaptive quadrature using Gauss-Konrod rule.

    Integrate fun from a to b using a h-adaptive scheme with error estimate.

    Differentiation wrt args is done via Liebniz rule.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        fun(x, *args) -> float. Should be JAX transformable.
    a, b : float
        Lower and upper limits of integration.
    args : tuple, optional
        Extra arguments passed to fun, and possibly a, b.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is 1.4e-8. Algorithm tries to
        obtain an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
        where ``i`` = integral of `fun` from `a` to `b`, and ``result`` is the
        numerical approximation.
    max_ninter : int, optional
        An upper bound on the number of sub-intervals used in the adaptive
        algorithm.
    order : {15, 21, 31, 41, 51, 61}
        Order of local integration rule.

    Returns
    -------
    y : float
        The integral of fun from `a` to `b`.
    err : float
        An estimate of the absolute error in the result.
    state : dict
        Final state of the algorithm. Only returned if full_output=True
        The entries are:

        'neval'
            The number of function evaluations.
        'ninter'
            The number, K, of sub-intervals produced in the subdivision process.
        'a_arr'
            A rank-1 array of length max_ninter, the first K elements of which are
            the left end points of the (remapped) sub-intervals in the partition of
            the integration range.
        'b_arr'
            A rank-1 array of length max_ninter, the first K elements of which are
            the right end points of the (remapped) sub-intervals.
        'r_arr'
            A rank-1 array of length max_ninter, the first K elements of which are
            the integral approximations on the sub-intervals.
        'e_arr'
            A rank-1 array of length max_ninter, the first K elements of which are
            the moduli of the absolute error estimates on the sub-intervals.

            TODO: other entries
    """
    out = adaptive_quadrature(
        fun, a, b, args, full_output, epsabs, epsrel, max_ninter, fixed_quadgk, n=order
    )
    if full_output:
        out[2]["neval"] *= order
    return out


def adaptive_quadrature(
    fun,
    a,
    b,
    args=(),
    return_state=False,
    epsabs=1.4e-8,
    epsrel=1.4e-8,
    max_ninter=50,
    rule=None,
    **kwargs
):
    """Global adaptive quadrature.

    Integrate fun from a to b using an adaptive scheme with error estimate.

    Differentiation wrt args is done via Liebniz rule.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        fun(x, *args) -> float. Should be JAX transformable.
    a, b : float
        Lower and upper limits of integration.
    args : tuple, optional
        Extra arguments passed to fun, and possibly a, b.
    full_output : bool, optional
        If True, return the full state of the integrator. See below for more
        information.
    epsabs, epsrel : float, optional
        Absolute and relative error tolerance. Default is 1.4e-8. Algorithm tries to
        obtain an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
        where ``i`` = integral of `fun` from `a` to `b`, and ``result`` is the
        numerical approximation.
    max_ninter : int, optional
        An upper bound on the number of sub-intervals used in the adaptive
        algorithm.
    rule : callable
        Local quadrature rule to use. It should have a signature of the form
        rule(fun, a, b, **kwargs) -> out, where out is array-like with 4 elements:

            1. Estimate of the integral of fun from a to b
            2. Estimate of the absolute error in the integral (ie, from a
              nested scheme).
            3. Estimate of the integral of abs(fun) from a to b
            4. Estimate of the integral of abs(fun - <fun>) from a to b, where <fun>
              is the mean value of fun over the interval.

    Returns
    -------
    y : float
        The integral of fun from `a` to `b`.
    err : float
        An estimate of the absolute error in the result.
    state : dict
        Final state of the algorithm. Only returned if full_output=True
        The entries are:

        'neval'
            The number of calls to ``rule``.
        'ninter'
            The number, K, of sub-intervals produced in the subdivision process.
        'a_arr'
            A rank-1 array of length max_ninter, the first K elements of which are
            the left end points of the (remapped) sub-intervals in the partition of
            the integration range.
        'b_arr'
            A rank-1 array of length max_ninter, the first K elements of which are
            the right end points of the (remapped) sub-intervals.
        'r_arr'
            A rank-1 array of length max_ninter, the first K elements of which are
            the integral approximations on the sub-intervals.
        'e_arr'
            A rank-1 array of length max_ninter, the first K elements of which are
            the moduli of the absolute error estimates on the sub-intervals.

            TODO: other entries
    """
    vfunc = jax.jit(jnp.vectorize(lambda x: fun(x, *args)))
    vfunc = map_interval(vfunc, a, b)

    f = vfunc(jnp.array([0.5 * (a + b)]))  # call it once to get dtype info
    uflow = jnp.finfo(f.dtype).tiny
    epmach = jnp.finfo(f.dtype).eps
    a, b = -1, 1

    state = {}
    state["neval"] = 0  # number of evaluations of local quadrature rule
    state["ninter"] = 0
    state["r_arr"] = jnp.zeros(max_ninter)  # local results from each interval
    state["e_arr"] = jnp.zeros(max_ninter)  # local error est. from each interval
    state["a_arr"] = jnp.zeros(max_ninter)  # start of each interval
    state["b_arr"] = jnp.zeros(max_ninter)  # end of each interval
    state["s_arr"] = jnp.zeros(max_ninter)  # global est. of I from n intervals
    state["a_arr"] = state["a_arr"].at[0].set(a)
    state["b_arr"] = state["b_arr"].at[0].set(b)
    state["roundoff1"] = 0  # for keeping track of roundoff errors
    state["roundoff2"] = 0  # for keeping track of roundoff errors
    state["status"] = 0  # error flag
    state["err_bnd"] = 0.0  # error bound we're trying to reach
    state["area"] = 0.0  # current best estimate for I
    state["err_sum"] = jnp.inf  # current estimate for error in I

    result, abserr, intabs, intmmn = rule(vfunc, a, b, (), **kwargs)

    state["neval"] += 1
    state["area"] = result
    state["err_sum"] = abserr
    state["err_bnd"] = jnp.maximum(epsabs, epsrel * jnp.abs(result))
    state["r_arr"] = state["r_arr"].at[0].set(result)
    state["e_arr"] = state["e_arr"].at[0].set(abserr)
    state["s_arr"] = state["s_arr"].at[0].set(result)

    # check for roundoff error - error too big but relative error is small
    state["status"] += (
        2**ROUNDOFF
        * ROUNDOFF
        * ((abserr <= (100.0 * epmach * intabs)) & (abserr > state["err_bnd"]))
    )
    # check for max intervals exceeded
    state["status"] += 2**MAX_NINTER * MAX_NINTER * (max_ninter == 0)

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

        area1, error1, intabs1, intmmn1 = rule(vfunc, a1, b1, (), **kwargs)
        state["neval"] += 1
        area2, error2, intabs2, intmmn2 = rule(vfunc, a2, b2, (), **kwargs)
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
        state["err_bnd"] = jnp.maximum(epsabs, epsrel * jnp.abs(state["area"]))

        # test for roundoff error
        # is the area estimate not changing and error not getting smaller?
        state["roundoff1"] += (
            jnp.abs(state["r_arr"][i] - area12) <= 0.1e-4 * jnp.abs(area12)
        ) & (erro12 >= 0.99 * jnp.max(state["e_arr"]))
        # are errors getting larger as we go to smaller intervals?
        state["roundoff2"] += (n > 10) & (erro12 > jnp.max(state["e_arr"]))
        state["status"] += (
            2**ROUNDOFF
            * ROUNDOFF
            * ((state["roundoff1"] >= 10) | (state["roundoff2"] >= 20))
        )

        # test for max number of intervals
        state["status"] += 2**MAX_NINTER * MAX_NINTER * (n == max_ninter)

        # test for bad behavior of the integrand (ie, intervals are getting too small)
        state["status"] += (
            2**BAD_INTEGRAND
            * BAD_INTEGRAND
            * (
                jnp.maximum(jnp.abs(a1), jnp.abs(b2))
                <= (1.0 + 100.0 * epmach) * (jnp.abs(a2) + 1000.0 * uflow)
            )
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

    while condfun(state):
        state = bodyfun(state)

    result = jnp.sum(state["r_arr"])
    abserr = state["err_sum"]
    out = (result, abserr)
    out += (state,) if return_state else ()
    return out
