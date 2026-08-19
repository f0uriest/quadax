"""Tests for adaptive quadrature routines."""

import os
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy
from jax import config

import quadax
from quadax import (
    GaussKronrodRule,
    adaptive_quadrature,
    quadcc,
    quadgk,
    quadts,
    romberg,
    rombergts,
)

from .problems import (
    ALL,
    CONVERGENT_TOLS,
    PROBLEMS,
    RESOLVED_BY_ACCELERATION,
    SLOP,
    SMOOTH,
    TOLS,
    ULP_ATOL,
    ULP_RTOL,
    assert_contract,
    complex_dtypes,
    exp_neg,
    problem_id,
    real_dtypes,
    real_of,
    xfail_if_known,
)

config.update("jax_enable_x64", True)


# Extrapolation is what makes the hard problems solvable, so it is switched on for the
# whole suite. The unaccelerated path is swept over the smooth problems only, where
# the subdivision converges on its own and there is a right answer to hold it to; on
# the rest it can only fail, which the acceleration tests below pin down directly.
CASES = [(i, True) for i in ALL] + [(i, False) for i in SMOOTH]


# A list rather than a callable: pytest passes an id function each argument of the pair
# separately, so it cannot name the two together.
CASE_IDS = [
    f"{problem_id(i)}-{'extrap' if extrapolate else 'plain'}"
    for i, extrapolate in CASES
]


# The tightest tolerance is where the subdivision runs deepest and most of the sweep's
# runtime goes, so it carries the `slow` marker and `-m "not slow"` leaves a suite that
# still covers every problem through every routine.
TOL_PARAMS = [
    pytest.param(t, marks=[pytest.mark.slow] if t == min(TOLS) else [], id=f"{t:g}")
    for t in TOLS
]


@pytest.mark.parametrize("method", [quadgk, quadcc, quadts], ids=["gk", "cc", "ts"])
@pytest.mark.parametrize("tol", TOL_PARAMS)
@pytest.mark.parametrize("i, extrapolate", CASES, ids=CASE_IDS)
class TestAdaptive:
    """Every example problem, through every adaptive routine, at every tolerance.

    Two things are asserted, and only two. Where the requested tolerance is reachable
    the routine is expected to reach it and say so, and whatever it returns has to come
    with an error estimate that does not understate the true error. Nothing here
    encodes which failure code an unconverged run produces or how far off it lands: a
    routine that cannot solve a problem only has to report that honestly.
    """

    def test_value_and_error(self, request, method, tol, i, extrapolate):
        """The answer is good to the tolerance asked for, and the error is honest."""
        prob = PROBLEMS[i]
        if extrapolate:
            xfail_if_known(request, method, prob, tol)
        y, info = method(
            prob["fun"],
            prob["interval"],
            epsabs=jnp.asarray(tol),
            epsrel=jnp.asarray(tol),
            extrapolate=extrapolate,
        )
        if extrapolate and tol in CONVERGENT_TOLS:
            assert int(info.status) == 0, (
                f"{prob['name']} at tol={tol:g}: {quadax.STATUS[int(info.status)]}"
            )
        assert_contract(y, info, prob, tol)


# The tabulated orders of each rule. Clenshaw-Curtis starts at 16 rather than 8 for
# the mesh comparison: order 8 is coarse enough to hit the roundoff floor on the
# infinite range problems, which makes it useless for a comparison against the orders
# that do converge.
GK_ORDERS = [15, 21, 31, 41, 51, 61]
CC_ORDERS = [16, 32, 64, 128, 256]
TS_ORDERS = [41, 61, 81, 101]
ORDERS = [(quadgk, GK_ORDERS), (quadcc, CC_ORDERS), (quadts, TS_ORDERS)]
ORDER_IDS = ["gk", "cc", "ts"]


@pytest.mark.parametrize("method, orders", ORDERS, ids=ORDER_IDS)
class TestRuleOrders:
    """The subdivision loop drives every tabulated order, not just the default one.

    The accuracy of each order in isolation is pinned in ``tests/test_fixed_order.py``;
    what is under test here is the loop wrapped around it.
    """

    # one smooth problem and one the local rule cannot resolve on its own
    PROBS = [0, 17]
    TOL = 1e-8

    @pytest.mark.parametrize("i", PROBS)
    def test_contract_holds_at_every_order(self, request, method, orders, i):
        """Changing the order changes the cost, never the promise."""
        prob = PROBLEMS[i]
        # the default order is swept in `TestAdaptive`, so a routine listed as failing
        # this problem there is not expected to hold the contract at any other order
        xfail_if_known(request, method, prob, self.TOL)
        for order in orders:
            y, info = method(
                prob["fun"],
                prob["interval"],
                epsabs=self.TOL,
                epsrel=self.TOL,
                order=order,
                extrapolate=True,
            )
            assert int(info.status) == 0, (
                f"{prob['name']} at order {order}: {quadax.STATUS[int(info.status)]}"
            )
            assert_contract(y, info, prob, self.TOL)

    @pytest.mark.parametrize("i", SMOOTH)
    def test_higher_order_needs_fewer_intervals(self, method, orders, i):
        """A higher order rule resolves a smooth integrand on a coarser mesh.

        This is the whole reason the order is exposed. It is a statement about the mesh
        and not about the cost: raising the order raises the price of each sub-interval,
        so the total evaluation count often goes up as the interval count comes down.

        Not strict, because most of these problems fit in a single panel at every order
        and there is nothing left to improve; the claim has teeth on the infinite range
        problems, where the coarsest order needs an order of magnitude more intervals.
        """
        prob = PROBLEMS[i]
        ninter = []
        for order in orders:
            _, info = method(
                prob["fun"],
                prob["interval"],
                epsabs=1e-10,
                epsrel=1e-10,
                order=order,
                full_output=True,
            )
            assert int(info.status) == 0, (
                f"{prob['name']} at order {order}: {quadax.STATUS[int(info.status)]}"
            )
            ninter.append(int(info.info["ninter"]))
        assert all(hi <= lo for lo, hi in zip(ninter, ninter[1:])), (
            f"{prob['name']}: orders {orders} gave ninter {ninter}"
        )


class TestExtrapolation:
    """Tests for the convergence acceleration in the adaptive solvers."""

    @pytest.mark.parametrize("i", RESOLVED_BY_ACCELERATION, ids=problem_id)
    @pytest.mark.parametrize("quad", [quadgk, quadcc])
    def test_singularities_are_resolved(self, quad, i):
        """Acceleration should reach near machine precision where the mesh cannot."""
        prob = PROBLEMS[i]
        kwargs = dict(epsabs=1e-12, epsrel=1e-12, max_ninter=200, full_output=True)
        y_off, info_off = quad(
            prob["fun"], prob["interval"], extrapolate=False, **kwargs
        )
        y_on, info_on = quad(prob["fun"], prob["interval"], extrapolate=True, **kwargs)
        exact = np.asarray(prob["val"])
        err_off = np.max(np.abs(np.asarray(y_off) - exact)) / np.max(np.abs(exact))
        err_on = np.max(np.abs(np.asarray(y_on) - exact)) / np.max(np.abs(exact))
        # Two orders of magnitude is well inside the margin: measured gains run from
        # 1e3 on the mildest of these to 1e11 on the strongest.
        assert err_on < err_off / 100, f"{prob['name']}: {err_off:.2e} -> {err_on:.2e}"
        assert err_on < 1e-10
        # and it should get there on a far coarser subdivision
        assert info_on.info["ninter"] < info_off.info["ninter"]

    @pytest.mark.parametrize("alpha, finite_part", [(1.5, -2.0), (2.0, -1.0)])
    def test_divergent_integral_is_flagged(self, alpha, finite_part):
        """A divergent integrand must not come back looking converged.

        The table returns the analytic continuation of the convergent case, the
        epsilon algorithm sums a divergent series the way Pade approximants do, and is
        indifferent to whether the limit it infers exists. scipy returns the same value.
        What keeps that from being silently wrong is the flag, not the value.
        """
        y, info = quadgk(
            lambda t: t**-alpha,
            jnp.array([0.0, 1.0]),
            epsabs=1e-10,
            epsrel=1e-10,
            max_ninter=200,
            extrapolate=True,
        )
        np.testing.assert_allclose(float(y), finite_part, rtol=1e-6)
        assert int(info.status) & 2**quadax.adaptive.DIVERGENT
        # the docstrings tell users to look the code up in STATUS, so it has to be there
        assert quadax.STATUS[int(info.status)].strip()

    def test_no_asymptotic_structure_falls_back(self):
        """``sin(1/x)`` has no trend to extrapolate, so the table must not win.

        The reference is built by splitting at the oscillation's turning points rather
        than taken from ``scipy.quad``, which cannot reach this tolerance on the whole
        interval in one go and says so with a warning.
        """
        a = 1e-3
        # 1/t runs from 1 to 1000, so put a breakpoint at every multiple of pi in that
        # range and the integrand is smooth on each piece.
        turns = 1 / (np.pi * np.arange(1, int(1 / (np.pi * a)) + 1))[::-1]
        edges = np.concatenate([[a], turns[turns > a], [1.0]])
        ref = sum(
            scipy.integrate.quad(
                lambda t: np.sin(1 / t), lo, hi, epsabs=1e-13, epsrel=1e-13
            )[0]
            for lo, hi in zip(edges[:-1], edges[1:])
        )
        y, info = quadgk(
            lambda t: jnp.sin(1 / t),
            jnp.array([a, 1.0]),
            epsabs=1e-12,
            epsrel=1e-12,
            max_ninter=100,
            extrapolate=True,
        )
        # Either it is right, or it says it is not; what it must not do is both be
        # wrong and report success.
        if int(info.status) == 0:
            np.testing.assert_allclose(float(y), ref, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("i", ALL, ids=problem_id)
    def test_tighter_tolerance_never_hurts(self, i):
        """Asking for more accuracy must not deliver less.

        ``epsabs = epsrel = 0`` is a common shorthand for "do the best you can inside
        the budget", but can be dangerous without the right guards. The threshold
        deciding when to extrapolate rather than subdivide further is compared
        against the requested tolerance, so a tolerance of zero left it permanently
        preferring to subdivide, the table was hardly ever fed, and the answer fell back
        to what the mesh alone manages, about eight digits on the singular ones,
        where a reachable tolerance gets fifteen. scipy refuses a tolerance this small
        at input validation instead; quadax accepts it, so it has to behave.
        """
        prob = PROBLEMS[i]
        exact = np.asarray(prob["val"])
        # Floored at one, so a problem whose exact value is zero measures absolute
        # error rather than dividing by it. The comparison below is between runs of
        # the same problem, so the choice of scale only has to be consistent.
        scale = max(np.max(np.abs(exact)), 1.0)

        def err(tol):
            y, _ = quadgk(
                prob["fun"],
                prob["interval"],
                epsabs=jnp.asarray(tol),
                epsrel=jnp.asarray(tol),
                order=21,
                max_ninter=200,
                extrapolate=True,
            )
            return float(np.max(np.abs(np.asarray(y) - exact)) / scale)

        errs = {tol: err(tol) for tol in (1e-12, 1e-13, 1e-14, 1e-15, 0.0)}
        best = min(errs.values())
        # Not exact monotonicity -- the subdivision genuinely differs between runs
        # -- but no cliff: the unreachable tolerances must stay in the same league as
        # the best any tolerance reached, rather than falling back several orders.
        for tol in (1e-14, 1e-15, 0.0):
            assert errs[tol] <= max(100 * best, 1e-13), (
                f"{prob['name']}: tol={tol:g} gives {errs[tol]:.2e}, "
                f"best over the sweep is {best:.2e} ({errs})"
            )

    @pytest.mark.parametrize("i", SMOOTH, ids=problem_id)
    def test_smooth_problems_never_extrapolate(self, i):
        """Where the subdivision converges, the table must stay out of the way.

        The mesh sum is the one carrying an honest error bound, so a table fed early
        on a coarse mesh must not be able to replace it: once the subdivision has
        reached the tolerance on its own the extrapolated value is not considered at
        all, and the answer is bit for bit the one the flag-off run produces.

        Checked all the way down to a tolerance of zero, because that is the setting
        that most changes the balance between subdividing and extrapolating, and a
        smooth integrand is where an accelerated value would be least justified.
        """
        prob = PROBLEMS[i]
        for tol in (1e-8, 1e-12, 0.0):
            y, info = quadgk(
                prob["fun"],
                prob["interval"],
                epsabs=jnp.asarray(tol),
                epsrel=jnp.asarray(tol),
                order=21,
                max_ninter=200,
                full_output=True,
                extrapolate=True,
            )
            assert not bool(info.info["used_accel"]), (
                f"{prob['name']} at tol={tol:g} returned an extrapolated value"
            )
            # and the answer is the subdivision's own, unchanged by the flag
            y_off, _ = quadgk(
                prob["fun"],
                prob["interval"],
                epsabs=jnp.asarray(tol),
                epsrel=jnp.asarray(tol),
                order=21,
                max_ninter=200,
                full_output=True,
                extrapolate=False,
            )
            np.testing.assert_allclose(
                np.asarray(y),
                np.asarray(y_off),
                rtol=ULP_RTOL,
                atol=ULP_ATOL,
                err_msg=f"{prob['name']}, tol={tol:g}",
            )

    def test_a_converged_component_does_not_spoil_the_others(self):
        """A vector integrand accelerates as well as its hardest component alone.

        The table's arithmetic is per component while its structural decisions go
        through the norm. A component the local rule integrates exactly has differences
        of exactly zero, which the norm -- driven by the singular component -- reads as
        safe to divide by. See ``tests/test_acceleration.py`` for the table's own test;
        this is the path that reaches it, and it is the shape every raveled adjoint
        integrand has.
        """
        scalar = lambda t: jnp.asarray(t**-0.5)  # noqa: E731
        reference, ref_info = quadgk(
            scalar, jnp.array([0.0, 1.0]), epsabs=0.0, epsrel=0.0, extrapolate=True
        )
        np.testing.assert_allclose(float(reference), 2.0, atol=1e-13)
        for other in (lambda t: 0.0 * t, lambda t: t, lambda t: t**2):
            paired = lambda t: jnp.array([t**-0.5, other(t)])  # noqa: E731, B023
            y, info = quadgk(
                paired, jnp.array([0.0, 1.0]), epsabs=0.0, epsrel=0.0, extrapolate=True
            )
            np.testing.assert_allclose(float(np.asarray(y)[0]), 2.0, atol=1e-13)
            assert int(info.neval) <= 2 * int(ref_info.neval)


def test_escaped_tracers():
    """Test that no tracers escape, related to gh issue 18."""

    @jax.jit
    def integral_quadgk(interval):
        return quadgk(jnp.square, interval)

    with jax.checking_leaks():
        jax.block_until_ready(integral_quadgk([0.0, 1.0]))

    @jax.jit
    def integral_quadcc(interval):
        return quadcc(jnp.square, interval)

    with jax.checking_leaks():
        jax.block_until_ready(integral_quadcc([0.0, 1.0]))

    @jax.jit
    def integral_quadts(interval):
        return quadts(jnp.square, interval)

    with jax.checking_leaks():
        jax.block_until_ready(integral_quadts([0.0, 1.0]))

    @jax.jit
    def integral_romberg(interval):
        return romberg(jnp.square, interval)

    with jax.checking_leaks():
        jax.block_until_ready(integral_romberg([0.0, 1.0]))

    @jax.jit
    def integral_rombergts(interval):
        return rombergts(jnp.square, interval)

    with jax.checking_leaks():
        jax.block_until_ready(integral_rombergts([0.0, 1.0]))


@pytest.mark.parametrize("quad", [quadgk, quadcc, quadts])
@pytest.mark.parametrize("max_ninter", [8, 12, 20, 33])
def test_subdivision_tiles_the_domain(quad, max_ninter):
    """The sub-intervals must tile the whole domain, even when max_ninter is hit.

    Regression test: the new half of a bisection used to be written at the interval
    count *after* incrementing it, which skipped a slot and, on the last iteration,
    scattered out of bounds. That silently dropped one sub-interval from the mesh, so
    part of the domain was never integrated and ``ninter`` over-reported by one.
    """
    # needs far more subdivisions than allowed, so max_ninter is always reached
    fun = lambda t: 1.0 / jnp.sqrt(jnp.abs(t - 0.3) + 1e-9)
    interval = [0.0, 1.0]
    y, info = quad(fun, interval, (), True, max_ninter=max_ninter)

    a_arr, b_arr = info.info["a_arr"], info.info["b_arr"]
    ninter = int(info.info["ninter"])
    span = interval[-1] - interval[0]
    np.testing.assert_allclose(float(jnp.sum(b_arr - a_arr)), span, rtol=0, atol=1e-14)
    # and every counted interval must actually be present
    assert int(jnp.sum(jnp.asarray(info.info["r_arr"]) != 0)) == ninter
    assert ninter <= max_ninter


@pytest.mark.parametrize("quad", [quadgk, quadcc, quadts])
def test_truncated_result_is_still_a_partition(quad):
    """Sub-intervals must be disjoint and contiguous when max_ninter is reached."""
    fun = lambda t: 1.0 / jnp.sqrt(jnp.abs(t - 0.3) + 1e-9)
    interval = [0.0, 1.0]
    _, info = quad(fun, interval, (), True, max_ninter=16)
    n = int(info.info["ninter"])
    a = np.asarray(info.info["a_arr"])[:n]
    b = np.asarray(info.info["b_arr"])[:n]
    order = np.argsort(a)
    a, b = a[order], b[order]
    np.testing.assert_allclose(a[0], interval[0], atol=1e-14)
    np.testing.assert_allclose(b[-1], interval[-1], atol=1e-14)
    np.testing.assert_allclose(a[1:], b[:-1], atol=1e-14)


@pytest.mark.parametrize("max_ninter", [22, 23, 24, 25, 26])
def test_converged_iteration_exits_clean(max_ninter):
    """Meeting the tolerance as the budget runs out is not a failure.

    Reaching the tolerance takes precedence over every status flag, so an iteration that
    reaches it exits cleanly even if it also consumed the last subdivision slot: the
    answer met the request, and what it cost getting there is not a failure. Setting the
    flags unconditionally and leaving termination to the loop predicate on the next pass
    instead makes an iteration that does both report a spurious failure.
    """
    tol = 1e-10
    y, info = quadgk(
        lambda t: jnp.log(t),
        jnp.array([0.0, 1.0]),
        epsabs=tol,
        epsrel=tol,
        max_ninter=max_ninter,
    )
    err = float(info.err)
    if 0 <= err <= max(tol, tol * abs(float(y))):
        assert int(info.status) == 0, f"reported failure at err={err:.3e} <= {tol:.0e}"


# A tall narrow peak: 1e8 at its top, integral 3.1e4, so the error estimates the loop
# starts from are ~15 orders of magnitude above the total it ends at.
_PEAK = lambda t: 1 / ((t - 0.5) ** 2 + 1e-8)
_PEAK_VAL = 31411.926535951257  # mpmath, split at the peak


def test_no_spurious_roundoff_on_unresolved_integrand():
    """A peaked but tractable integrand must not be written off as roundoff-limited.

    Two things have to hold for the roundoff counters to mean what they claim. The
    stagnation test has to compare the two halves against the *parent's* value, captured
    before either overwrites it, rather than against a slot already holding one of the
    halves. And both counters have to be suppressed when the local rule did not resolve
    a half at all -- recognizable by the error estimate coming back at its saturation
    value -- since a stagnant area is then evidence of an unresolved integrand rather
    than of roundoff. Without either, the loop gives up on this integrand early,
    reporting ROUNDOFF with an error five orders of magnitude worse than achievable.
    """
    y, info = quadgk(_PEAK, jnp.array([0.0, 1.0]), epsabs=1e-12, epsrel=1e-12)
    assert int(info.status) == 0, quadax.STATUS[int(info.status)]
    np.testing.assert_allclose(float(y), _PEAK_VAL, rtol=1e-13, atol=0)


def test_tolerance_below_roundoff_floor_reports_roundoff():
    """Asking for more precision than the arithmetic allows is a ROUNDOFF verdict.

    The local rule floors each sub-interval's error estimate at ``50*eps*int|f|``, so
    the total cannot fall below that floor summed over the partition however fine the
    mesh gets. Here that floor is ~1.1e-14 relative, so a request of 1e-14 is out of
    reach. Testing for that only before the subdivision loop, as QUADPACK does, leaves
    such a request to burn through the whole subdivision budget and report MAX_NINTER --
    true, but not the reason; quadax tests it every iteration instead.
    """
    _, info = quadgk(_PEAK, jnp.array([0.0, 1.0]), epsabs=1e-14, epsrel=1e-14)
    assert int(info.status) & 2**2, quadax.STATUS[int(info.status)]
    # and the error it stopped at really is the accumulated floor
    np.testing.assert_allclose(
        float(info.err), 50 * np.finfo(np.float64).eps * _PEAK_VAL, rtol=1e-3
    )


def _scaled_gaussian(t, s):
    """``exp(-(t/s)**2)``, with the scale taken as an argument to avoid recompile."""
    return jnp.exp(-((t / s) ** 2))


def _inv_sqrt(t):
    return t**-0.5


class TestIntervalScaling:
    """Where the interval sits on the axis must not change the answer.

    End to end rather than on ``map_interval`` alone: what is under test is that the
    abscissae reaching the integrand are correctly rounded at every scale, and only a
    full solve subdivides deeply enough for a cancellation in the map to show up.
    """

    @pytest.mark.parametrize("scale", [1e-8, 1e-3, 1.0, 1e3, 1e8])
    @pytest.mark.parametrize("method", [quadgk, quadcc, quadts])
    def test_the_result_does_not_depend_on_the_scale_of_the_interval(
        self, method, scale
    ):
        """The width floor is relative, so rescaling the problem rescales the answer."""
        s = float(scale)
        y, info = method(
            _scaled_gaussian,
            jnp.array([0.0, 3 * s]),
            (jnp.asarray(s),),
            epsabs=jnp.asarray(0.0),
            epsrel=jnp.asarray(1e-10),
        )
        assert int(info.status) == 0, quadax.STATUS[int(info.status)]
        np.testing.assert_allclose(float(y) / s, 0.8862073482595214, rtol=1e-10)

    @pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
    def test_a_singularity_at_the_origin_is_scale_invariant(self, scale):
        """``x**-1/2`` on ``[0, s]``: the case a single affine map makes exact.

        ``0 + halflength*(1 + x_node)`` has nothing to cancel, so every abscissa is the
        correctly rounded distance from the singularity however deep the subdivision
        goes, and the answer no longer depends on where the interval sits.
        """
        s = float(scale)
        y, _ = quadgk(
            _inv_sqrt,
            jnp.array([0.0, s]),
            epsabs=jnp.asarray(0.0),
            epsrel=jnp.asarray(1e-12),
        )
        np.testing.assert_allclose(float(y), 2 * np.sqrt(s), rtol=1e-8)


class TestErrors:
    """Invalid arguments must raise, rather than silently doing something else."""

    def test_rule_must_be_a_quadrature_rule(self):
        """Passing a bare callable for ``rule`` was deprecated and now raises.

        Custom rules must subclass ``AbstractQuadratureRule`` so that the adjoints have
        a real object to hand back to AD.
        """
        with pytest.raises(TypeError, match="should be an instance of"):
            adaptive_quadrature(
                lambda fun, a, b, args: 0.0,  # pyright: ignore[reportArgumentType]
                lambda t: t,
                jnp.array([0.0, 1.0]),
            )

    def test_max_ninter_must_cover_the_breakpoints(self):
        """``max_ninter`` below the number of breakpoints cannot be satisfied."""
        with pytest.raises(ValueError, match="is not enough for"):
            quadgk(lambda t: t, jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), max_ninter=2)

    def test_unsupported_rule_order(self):
        """Only the tabulated Gauss-Kronrod orders are available."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            GaussKronrodRule(order=7)


# The dtype of `interval` is the statement of what precision the user wants. The tests
# below pin the four dtypes described by `quadax.utils.DTypes`: the abscissa the
# integrand is called with, the integrand values and returned integral, the error
# estimate, and the default tolerances.

adaptive_methods = [quadgk, quadcc, quadts]
all_methods = adaptive_methods + [romberg, rombergts]


@pytest.mark.usefixtures("quiet_tanhsinh")
class TestWorkingDType:
    """The dtype of ``interval`` selects the precision, and is respected end to end."""

    @pytest.mark.parametrize("method", all_methods)
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_round_trip(self, method, dtype):
        """Y and err come back at the requested precision, and the answer is right."""
        interval = jnp.array([0.0, 1.0], dtype=dtype)
        y, info = method(exp_neg, interval)

        assert y.dtype == dtype
        assert jnp.asarray(info.err).dtype == dtype
        # exp(-x) on [0, 1]; only asking for sqrt(eps)-ish accuracy
        tol = SLOP * np.sqrt(float(jnp.finfo(dtype).eps))
        np.testing.assert_allclose(float(y), 1 - np.exp(-1), atol=tol)

    @pytest.mark.parametrize("method", all_methods)
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_integrand_is_called_at_the_requested_dtype(self, method, dtype):
        """The integrand sees ``x`` at the interval's dtype, not the default one.

        A node table stored at float64 would hand the user's function a float64 ``x``
        whatever precision it asked for. The assert runs at trace time.
        """
        seen = []

        def fun(x):
            seen.append(x.dtype)
            return jnp.exp(-x)

        method(fun, jnp.array([0.0, 1.0], dtype=dtype))
        assert seen, "integrand was never traced"
        assert set(seen) == {jnp.dtype(dtype)}, f"integrand saw {set(seen)}"

    @pytest.mark.parametrize("method", all_methods)
    def test_integrand_may_upcast(self, method):
        """An integrand that deliberately upcasts is respected.

        The abscissa stays at the precision that was asked for, but the accumulation and
        the result follow the integrand.
        """
        seen = []

        def fun(x):
            seen.append(x.dtype)
            return jnp.exp(-x.astype(jnp.float64))

        y, info = method(fun, jnp.array([0.0, 1.0], dtype=jnp.float32))
        assert set(seen) == {jnp.dtype(jnp.float32)}
        assert y.dtype == jnp.float64
        assert jnp.asarray(info.err).dtype == jnp.float64

    @pytest.mark.parametrize("method", all_methods)
    @pytest.mark.parametrize("dtype", complex_dtypes)
    def test_complex_integrand(self, method, dtype):
        """Complex values, real limits: the error estimate stays real."""
        rtype = real_of[dtype]
        interval = jnp.array([0.0, 1.0], dtype=rtype)
        fun = lambda x: (jnp.exp(-x) + 1j * jnp.sin(x)).astype(dtype)
        y, info = method(fun, interval)

        assert y.dtype == dtype
        assert jnp.asarray(info.err).dtype == rtype
        tol = SLOP * np.sqrt(float(jnp.finfo(rtype).eps))
        np.testing.assert_allclose(complex(y).real, 1 - np.exp(-1), atol=tol)
        np.testing.assert_allclose(complex(y).imag, 1 - np.cos(1), atol=tol)

    @pytest.mark.parametrize("method", all_methods)
    def test_complex_limits_rejected(self, method):
        """Complex limits have no ordering, so the subdivision cannot be defined."""
        with pytest.raises(TypeError, match="real floating point"):
            method(lambda x: x, jnp.array([0.0 + 0j, 1.0 + 0j]))

    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_infinite_limits(self, dtype):
        """All four branches of the interval map agree on a dtype.

        The integrand is probed to determine the output dtype; if that probe is done
        with a weakly typed scalar the four branches can settle on different dtypes and
        the ``switch`` between them fails to build.
        """
        tol = SLOP * np.sqrt(float(jnp.finfo(dtype).eps))
        for interval, expected in [
            ([0.0, jnp.inf], np.sqrt(np.pi) / 2),
            ([-jnp.inf, 0.0], np.sqrt(np.pi) / 2),
            ([-jnp.inf, jnp.inf], np.sqrt(np.pi)),
        ]:
            y, _ = quadgk(lambda x: jnp.exp(-(x**2)), jnp.array(interval, dtype=dtype))
            assert y.dtype == dtype
            np.testing.assert_allclose(float(y), expected, atol=10 * tol, rtol=10 * tol)

    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_breakpoints_and_vector_valued(self, dtype):
        """Breakpoints keep the mesh at the abscissa dtype for a vector integrand."""
        interval = jnp.array([0.0, 0.5, 1.0], dtype=dtype)
        y, info = quadgk(lambda x: jnp.array([jnp.exp(-x), x**2]), interval)
        assert y.dtype == dtype
        assert jnp.asarray(info.err).dtype == dtype
        tol = SLOP * np.sqrt(float(jnp.finfo(dtype).eps))
        np.testing.assert_allclose(np.asarray(y), [1 - np.exp(-1), 1 / 3], atol=tol)


@pytest.mark.usefixtures("quiet_tanhsinh")
class TestErrorEstimateDTypes:
    """The reported error must never under-estimate, at any precision."""

    @pytest.mark.parametrize("method", adaptive_methods)
    @pytest.mark.parametrize("dtype", real_dtypes)
    @pytest.mark.parametrize(
        "fun, exact",
        [
            (lambda x: jnp.exp(-(x**2)), 0.7468241328124271),
            (lambda x: x**4 - 2 * x + 1, 1 / 5 - 1 + 1),
        ],
    )
    def test_error_is_an_upper_bound(self, method, dtype, fun, exact):
        """Reported error bounds the true error.

        At float16/bfloat16 this is the only thing worth asserting: the roundoff floor
        forces the QUADPACK estimator into its saturated regime, where it reports the
        total variation of the integrand rather than a sharp estimate. Conservative, but
        valid, which is what this pins.
        """
        y, info = method(fun, jnp.array([0.0, 1.0], dtype=dtype))
        true_err = abs(float(y) - exact)
        reported = float(jnp.asarray(info.err))
        assert reported >= true_err, (
            f"{jnp.dtype(dtype).name}: reported {reported:.3e} < true {true_err:.3e}"
        )


@pytest.mark.usefixtures("quiet_tanhsinh")
class TestToleranceDTypes:
    """Default tolerances follow the working dtype."""

    @pytest.mark.parametrize("method", all_methods)
    @pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
    def test_default_tolerance_tracks_dtype(self, method, dtype):
        """With no tolerance given, accuracy lands near sqrt(eps) of the dtype."""
        y, _ = method(exp_neg, jnp.array([0.0, 1.0], dtype=dtype))
        err = abs(float(y) - (1 - np.exp(-1)))
        assert err <= SLOP * np.sqrt(float(jnp.finfo(dtype).eps))

    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_explicit_tolerances_do_not_promote(self, dtype):
        """A python float tolerance must not drag the working dtype up with it."""
        y, info = quadgk(
            exp_neg,
            jnp.array([0.0, 1.0], dtype=dtype),
            epsabs=1e-3,
            epsrel=1e-3,
        )
        assert y.dtype == dtype
        assert jnp.asarray(info.err).dtype == dtype


def fresh_exp_neg():
    """A new ``exp(-x)`` object each call, so the solver has to trace it again.

    The integrand is a static argument, so tests normally share one to avoid compiling
    the same problem repeatedly. The warning tests below must do the opposite: it is
    raised while the node table is built, which happens once per trace, so a call that
    lands on a warm compilation cache is silent whatever the dtype. Sharing an integrand
    there would leave them asserting on collection order rather than on the warning.
    """
    return lambda x: jnp.exp(-x)


class TestTanhSinhPrecision:
    """Half precision costs the tanh-sinh rules their double exponential clustering."""

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    @pytest.mark.parametrize("method", [quadts, rombergts])
    def test_warns_in_half_precision(self, dtype, method):
        """The user is told when the rule cannot deliver what it usually does."""
        with pytest.warns(UserWarning, match="tanh-sinh quadrature in"):
            method(fresh_exp_neg(), jnp.array([0.0, 1.0], dtype=dtype))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("method", [quadts, rombergts])
    def test_silent_at_float32_and_above(self, dtype, method):
        """No warning where the clustering is fine."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            method(fresh_exp_neg(), jnp.array([0.0, 1.0], dtype=dtype))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_quadgk_never_warns(self, dtype):
        """The warning belongs to the tanh-sinh rules, not to quadrature generally."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quadgk(fresh_exp_neg(), jnp.array([0.0, 1.0], dtype=dtype))


def test_x64_disabled():
    """Everything works, and stays float32, with x64 off.

    A subprocess because ``jax_enable_x64`` is process global and the rest of the suite
    asserts to float64 precision.
    """
    script = """
import jax.numpy as jnp
import numpy as np
from quadax import quadgk, quadcc, quadts, romberg, rombergts

for method in [quadgk, quadcc, quadts, romberg, rombergts]:
    seen = []
    def fun(x):
        seen.append(x.dtype)
        return jnp.exp(-x)
    y, info = method(fun, jnp.array([0.0, 1.0]))
    assert y.dtype == jnp.float32, (method.__name__, y.dtype)
    assert info.err.dtype == jnp.float32, (method.__name__, info.err.dtype)
    assert set(seen) == {jnp.dtype(jnp.float32)}, (method.__name__, set(seen))
    np.testing.assert_allclose(float(y), 1 - np.exp(-1), atol=1e-4)

# the default tolerance is sqrt(eps32) here, as it always has been
y_d, i_d = quadgk(jnp.sin, jnp.array([0.0, 1.0]))
tol = float(np.sqrt(np.finfo(np.float32).eps))
y_e, i_e = quadgk(jnp.sin, jnp.array([0.0, 1.0]), epsabs=tol, epsrel=tol)
np.testing.assert_array_equal(np.asarray(y_d), np.asarray(y_e))
print("ok")
"""
    env = {**os.environ, "JAX_ENABLE_X64": "0"}
    out = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# Straddling the node count of the default rule for each routine: one point at a time, a
# size that leaves a partial last batch, and one above the node count so the clip fires.
BATCH_SIZES = [1, 4, 7, 10**4]


# A size that leaves a partial last batch, so the two-trace path is the one taken
# wherever a single size has to stand for the list above.
PARTIAL_BATCH = 7
BATCH_TOL = 1e-10


def _run_batched(method, batch_size, i=0, extrapolate=True):
    prob = PROBLEMS[i]
    return method(
        prob["fun"],
        prob["interval"],
        epsabs=BATCH_TOL,
        epsrel=BATCH_TOL,
        full_output=True,
        extrapolate=extrapolate,
        batch_size=batch_size,
    )


def _assert_same_run(got, want):
    """Value, error estimate, status and subdivision all agree to roundoff."""
    (y_b, info_b), (y, info) = got, want
    np.testing.assert_allclose(
        np.asarray(y_b), np.asarray(y), rtol=ULP_RTOL, atol=ULP_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(info_b.err), np.asarray(info.err), rtol=ULP_RTOL, atol=ULP_ATOL
    )
    assert int(info_b.status) == int(info.status)
    assert int(info_b.info["ninter"]) == int(info.info["ninter"])


@pytest.mark.parametrize("method", [quadgk], ids=["gk"])
class TestBatchSize:
    """Splitting the local rule's node evaluation into batches.

    The rule's node count is fixed when it is built, so the batches are cut to fit and
    the subdivision loop sees the same values it would have seen unbatched, in the same
    order, with no term reordered. That is a stronger claim than Romberg's, where the
    level sizes are dynamic and batching does reassociate a sum, and it is checked as
    such: the tolerance below is ULP-scale rather than the one the quadrature was asked
    for.

    It is deliberately not asserted bitwise. Evaluating the integrand under a different
    batch shape lets XLA fuse the arithmetic differently, which can move a result by
    around eps with no reordering involved, so a bitwise assertion would be testing the
    compiler rather than this option. ``batch_size`` is a statement about memory and
    scheduling, and the tests below are what pins it to that.
    """

    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=str)
    def test_run_is_unchanged(self, method, batch_size):
        """Value, error estimate, status and subdivision all come out identical."""
        _assert_same_run(_run_batched(method, batch_size), _run_batched(method, None))

    @pytest.mark.usefixtures("quiet_tanhsinh")
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_dtypes(self, method, dtype):
        """Batching must not disturb the working precision.

        Slicing and rejoining the abscissae carries their dtype through, so the batched
        run is asked for its answer in the same precision the unbatched one is.
        """
        y, info = method(
            exp_neg,
            jnp.array([0.0, 1.0], dtype),
            epsabs=jnp.array(1e-3, dtype),
            epsrel=jnp.array(1e-3, dtype),
            batch_size=PARTIAL_BATCH,
        )
        assert y.dtype == dtype
        tol = SLOP * np.sqrt(float(jnp.finfo(dtype).eps))
        np.testing.assert_allclose(float(y), 1 - np.exp(-1), atol=tol)

    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=str)
    def test_neval_does_not_move(self, method, batch_size):
        """Batching costs no extra evaluations, so the reported count is unchanged.

        The node count is fixed when the rule is built, so a batch size that does not
        divide it leaves a smaller final batch rather than a padded one. This is the
        test that would fail if the remainder were ever padded instead, and it is the
        one that would catch ``neval`` being scaled by the order where the rule's node
        count is something else.
        """
        run = lambda bs: _run_batched(method, bs)[1]  # noqa: E731
        assert int(run(batch_size).neval) == int(run(None).neval)


@pytest.mark.parametrize("i", [0, 17], ids=str)
@pytest.mark.parametrize("extrapolate", [False, True], ids=["plain", "extrap"])
def test_batch_size_is_independent_of_extrapolation(i, extrapolate):
    """The acceleration reads the sequence of results, not how each was evaluated.

    So it has nothing to interact with, on an easy problem or on one whose tail the
    extrapolation is what resolves. Run through one routine, since the subdivision loop
    and the acceleration around it are shared by all three.
    """
    _assert_same_run(
        _run_batched(quadgk, PARTIAL_BATCH, i, extrapolate),
        _run_batched(quadgk, None, i, extrapolate),
    )


@pytest.mark.parametrize("method", [quadgk, quadcc, quadts], ids=["gk", "cc", "ts"])
@pytest.mark.parametrize("batch_size", [0, -1, 2.5], ids=["zero", "negative", "float"])
def test_bad_batch_size_rejected(method, batch_size):
    """A batch size that is not a positive integer is a mistake, not a default."""
    with pytest.raises(ValueError, match="batch_size"):
        method(exp_neg, jnp.array([0.0, 1.0]), batch_size=batch_size)
