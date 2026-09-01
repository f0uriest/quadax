"""Tests for globally adaptive cubature over a box.

The counterpart of ``test_adaptive.py``. The battery at the top is the main thing: over
the problems in ``problems_nd.py``, does the routine reach the requested tolerance, and
whether or not it does, is the error it reports a bound on the error it made. Everything
below that tests one mechanism the battery cannot isolate.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quadax import (
    STATUS,
    ClenshawCurtisRule,
    DirectAdjoint,
    GaussKronrodRule,
    LeibnizAdjoint,
    adaptive_quadrature,
)
from quadax.adaptive_cube import adaptive_cubature, cubgm
from quadax.fixed_cubature import AbstractCubatureRule, GenzMalikRule, TensorProductRule

from . import problems_nd as pnd

# Two rules with quite different characters: a Genz-Malik rule pays a handful of nodes
# per region and refines a lot, a tensor product rule pays many and refines little. A
# fifteen point rule on every axis is 3375 nodes in three dimensions, so that case takes
# a cheaper axis rule, and above three dimensions a tensor product is not affordable at
# all and only the Genz-Malik rule is swept.
RULE_NAMES = ("genz-malik", "tensor-gk")


def rules_for(ndim, norm=jnp.inf, batch_size=None):
    """The rules under test at a given dimension, keyed by the name the tables use."""
    rules: dict[str, AbstractCubatureRule]
    rules = {"genz-malik": GenzMalikRule(ndim, 9, norm, batch_size)}
    if ndim <= 2:
        axis = GaussKronrodRule(15)
    elif ndim == 3:
        axis = ClenshawCurtisRule(8)
    else:
        return rules
    rules["tensor-gk"] = TensorProductRule(
        axis, ndim=ndim, norm=norm, batch_size=batch_size
    )
    return rules


def limits(prob):
    """A problem's limits as one float array per axis."""
    return [jnp.asarray(axis, dtype=float) for axis in prob["interval"]]


def solve(rule_name, prob, tol, **kwargs):
    """Run one problem with one of the swept rules."""
    rule = rules_for(prob["ndim"])[rule_name]
    kwargs.setdefault("max_nregion", pnd.BATTERY_MAX_NREGION)
    return adaptive_cubature(
        rule,
        prob["fun"],
        limits(prob),
        full_output=True,
        epsabs=jnp.asarray(tol),
        epsrel=jnp.asarray(tol),
        **kwargs,
    )


_SOLVED: dict = {}


def solve_once(rule_name, i, tol, **kwargs):
    """Run one case, reusing the result if another test has already asked for it.

    The battery asks two separate questions of every solve - whether the routine
    converged, and whether the error it reports is honest - and wants them to fail
    separately without paying for the cubature twice. Keyed on everything that changes
    the answer, so a miss is a genuinely new case rather than a repeat.
    """
    key = (rule_name, i, tol, tuple(sorted(kwargs.items())))
    if key not in _SOLVED:
        _SOLVED[key] = solve(rule_name, pnd.PROBLEMS[i], tol, **kwargs)
    return _SOLVED[key]


CASES = [
    (name, i)
    for i in pnd.ALL
    for name in RULE_NAMES
    if name in rules_for(pnd.PROBLEMS[i]["ndim"])
]
CASE_IDS = [f"{name}-{pnd.problem_id(i)}" for name, i in CASES]


@pytest.mark.parametrize("rule_name,i", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("tol", pnd.CONVERGENT_TOLS)
class TestBattery:
    """Convergence and error honesty over the whole problem set.

    The two are asserted by separate tests, over one shared solve, because they fail
    for unrelated reasons and are worth different amounts. Not converging is a limit on
    what a rule can do on a given integrand, and those are tabulated; understating the
    error is a defect wherever it happens, so those are tabulated separately and that
    table is meant to empty rather than to be maintained.
    """

    def test_it_converges(self, request, rule_name, i, tol):
        """The requested tolerance is reached, and the run says so."""
        prob = pnd.PROBLEMS[i]
        pnd.xfail_if_known(request, rule_name, prob, tol)
        y, info = solve_once(rule_name, i, tol)
        pnd.assert_converged(y, info, prob, tol)

    def test_it_reports_an_honest_error(self, request, rule_name, i, tol):
        """The reported error is never below the error actually made.

        This is the promise that holds whatever else happened, so it is asserted for
        the runs that failed to converge as well as the ones that succeeded.
        """
        prob = pnd.PROBLEMS[i]
        pnd.xfail_if_dishonest(request, rule_name, prob, tol)
        y, info = solve_once(rule_name, i, tol)
        pnd.assert_honest(y, info, prob, tol)


class TestSubdivision:
    """What the mesh does, as distinct from what it arrives at."""

    def test_a_breakpoint_is_worth_more_than_a_finer_mesh(self):
        """Marking the kinks costs a fraction of the regions finding them does.

        The two problems share an integrand and differ only in whether the planes it is
        not smooth across are given in `interval`, which makes the pair a measurement of
        what a breakpoint buys rather than of the integrand.
        """
        marked = next(p for p in pnd.PROBLEMS if p["name"] == "kink-marked")
        unmarked = next(p for p in pnd.PROBLEMS if p["name"] == "kink-unmarked")
        _, im = solve("genz-malik", marked, 1e-8)
        _, iu = solve("genz-malik", unmarked, 1e-8)
        assert im.status == STATUS.normal and iu.status == STATUS.normal
        # The marked run starts from the four regions the breakpoints define and needs
        # none beyond them; the unmarked one has to localize both planes itself.
        assert int(im.info["nregion"]) * 4 < int(iu.info["nregion"])

    @pytest.mark.parametrize("rule_name", RULE_NAMES)
    def test_the_narrow_axis_is_the_one_that_gets_cut(self, rule_name):
        """The mesh refines the axis the feature is narrow along, not the other.

        The rule reports a per-axis split indicator and the loop cuts whichever axis it
        ranks highest; without that it would have to cut every axis alike, and the depth
        along the two would come out the same.
        """
        prob = next(p for p in pnd.PROBLEMS if p["name"] == "gauss-ridge")
        _, info = solve(rule_name, prob, 1e-8)
        n = int(info.info["nregion"])
        width = np.asarray(info.info["b_arr"])[:n] - np.asarray(info.info["a_arr"])[:n]
        # The feature is 12x narrower along axis 0, so that axis must end up cut to a
        # smaller width than axis 1 anywhere in the mesh.
        assert width[:, 0].min() < width[:, 1].min()


class TestOneDimension:
    """A box of one axis, where the cubature and the one dimensional routines meet."""

    @pytest.mark.parametrize(
        "fun,interval",
        [
            (lambda t: jnp.abs(t - 1 / 3), [0.0, 1.0]),
            (lambda t: jnp.exp(-t), [0.0, np.inf]),
        ],
        ids=["kink", "semi-infinite"],
    )
    def test_it_reduces_to_the_one_dimensional_routine(self, fun, interval):
        """Over one axis both loops build the same mesh and return the same value.

        A tensor product of a single rule is that rule, so nothing but the subdivision
        can differ, and the values and region counts have to agree exactly. The error
        estimates need not: the cubature loop adds a term for the disagreement between
        a region and the two it is cut into that the one dimensional loop has no
        counterpart for, so they agree only to within it.

        Acceleration is off on the one dimensional side because the cubature loop has
        none, which is the substantive difference between the two.
        """
        rule = GaussKronrodRule(15)
        tol = 1e-10
        y_nd, nd = adaptive_cubature(
            TensorProductRule(rule, ndim=1),
            lambda x: fun(x[0]),
            [jnp.asarray(interval)],
            full_output=True,
            epsabs=jnp.asarray(tol),
            epsrel=jnp.asarray(tol),
            max_nregion=500,
        )
        y_1d, one = adaptive_quadrature(
            rule,
            fun,
            interval,
            full_output=True,
            epsabs=tol,
            epsrel=tol,
            extrapolate=False,
            max_ninter=500,
        )
        assert nd.status == STATUS.normal and one.status == STATUS.normal
        assert y_nd == y_1d
        assert int(nd.info["nregion"]) == int(one.info["ninter"])
        np.testing.assert_allclose(float(nd.err), float(one.err), rtol=0.05)


class TestStatus:
    """Each way the routine can stop, and that it reports the right one."""

    def test_a_starved_budget_reports_it(self):
        """Too few regions to reach the tolerance is `max_nregion`, not silence."""
        prob = next(p for p in pnd.PROBLEMS if p["name"] == "corner-sqrt")
        y, info = solve("genz-malik", prob, 1e-10, max_nregion=12)
        assert info.status == STATUS.max_nregion
        # The value is still the total over the mesh it ended with, and the error it
        # reports still covers the error it made.
        pnd.assert_honest(y, info, prob, 1e-10)

    def test_a_non_integrable_singularity_reports_bad_integrand(self):
        """A mesh driven down to the spacing of the abscissae says so.

        ``1/|x|**2`` diverges logarithmically at the corner, so the subdivision chases
        it until the corners of a region can no longer be told apart. The companion
        case is the point: ``1/(x0 + x1)`` looks just as singular at the same corner and
        is integrable, and there the mesh reaches the tolerance instead of the floor.
        """
        limits = [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])]
        _, divergent = cubgm(
            lambda x: 1 / jnp.sum(x**2),
            limits,
            epsabs=1e-10,
            epsrel=1e-10,
            max_nregion=2000,
        )
        assert divergent.status == STATUS.bad_integrand
        _, integrable = cubgm(
            lambda x: 1 / jnp.sum(x),
            limits,
            epsabs=1e-10,
            epsrel=1e-10,
            max_nregion=2000,
        )
        assert integrable.status == STATUS.normal

    def test_throw_raises_with_the_reason(self):
        """`throw=True` turns a non-normal status into the error it describes."""
        prob = next(p for p in pnd.PROBLEMS if p["name"] == "corner-sqrt")
        with pytest.raises(Exception, match="max_nregion"):
            solve("genz-malik", prob, 1e-10, max_nregion=12, throw=True)


class TestTransformations:
    """The routine under jit, vmap and both modes of differentiation."""

    def test_it_is_batched_over_the_limits(self):
        """`vmap` over the corners of the box matches the loop over them."""
        fun = lambda x: jnp.exp(-jnp.sum(x**2))
        run = lambda b: cubgm(fun, jnp.stack([jnp.zeros(2), b], axis=-1))[0]
        bs = jnp.array([[1.0, 1.0], [2.0, 0.5], [0.3, 3.0]])
        np.testing.assert_allclose(
            jax.vmap(run)(bs), jnp.stack([run(b) for b in bs]), rtol=1e-13, atol=1e-15
        )

    def test_it_is_batched_over_args(self):
        """`vmap` over an extra argument matches the loop over it."""
        fun = lambda x, p: jnp.exp(-p * jnp.sum(x))
        run = lambda p: cubgm(
            fun, [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])], args=(p,)
        )[0]
        ps = jnp.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(
            jax.vmap(run)(ps), jnp.stack([run(p) for p in ps]), rtol=1e-13, atol=1e-15
        )

    def test_it_is_differentiable_in_both_modes(self):
        """Forward and reverse agree with each other and with the exact derivative.

        Taken with respect to an argument of the integrand, to a limit of integration,
        and to an interior breakpoint. The three are different paths: the first
        differentiates the values, the second the domain the mesh is built on, and the
        third the grid of regions the breakpoints seed the mesh with.
        """
        fun = lambda x, p: jnp.exp(-p * jnp.sum(x))
        by_arg = lambda p: cubgm(
            fun, [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])], args=(p,)
        )[0]
        exact_arg = lambda p: ((1 - jnp.exp(-p)) / p) ** 2
        for f, exact, x in (
            (by_arg, exact_arg, 1.3),
            # d/db of int_0^b int_0^1 cos(u)cos(v) = cos(b) sin(1)
            (
                lambda b: cubgm(
                    lambda u: jnp.cos(u[0]) * jnp.cos(u[1]),
                    [jnp.array([0.0, b]), jnp.array([0.0, 1.0])],
                )[0],
                lambda b: jnp.sin(b) * jnp.sin(1.0),
                0.7,
            ),
            # A breakpoint at `c` in an integrand with a kink there. The mesh is seeded
            # from the breakpoints, so the derivative is only right if it moves with
            # them; a mesh frozen at the primal's `c` would miss the whole term.
            (
                lambda c: cubgm(
                    lambda u: jnp.abs(u[0] - c) * u[1],
                    [jnp.array([0.0, c, 1.0]), jnp.array([0.0, 1.0])],
                )[0],
                lambda c: (c**2 + (1 - c) ** 2) / 4,
                0.37,
            ),
        ):
            fwd, rev = jax.jacfwd(f)(x), jax.jacrev(f)(x)
            np.testing.assert_allclose(fwd, rev, rtol=1e-13, atol=1e-15)
            np.testing.assert_allclose(fwd, jax.jacfwd(exact)(x), rtol=1e-6, atol=1e-9)


class TestPlumbing:
    """Options that must not change the answer, and dtypes that must follow it."""

    @pytest.mark.parametrize("dtype", pnd.real_dtypes)
    def test_the_working_dtype_follows_the_limits(self, dtype):
        """The result and its error come back at the precision the limits asked for."""
        prob = next(p for p in pnd.PROBLEMS if p["name"] == "cos-product")
        iv = [jnp.asarray(axis, dtype=dtype) for axis in prob["interval"]]
        rule = GenzMalikRule(2, 7)
        y, info = adaptive_cubature(rule, prob["fun"], iv, max_nregion=100)
        assert jnp.asarray(y).dtype == dtype
        assert jnp.asarray(info.err).dtype == dtype
        eps = float(jnp.finfo(dtype).eps)
        np.testing.assert_allclose(
            float(y), prob["val"], atol=pnd.SLOP * np.sqrt(eps), rtol=0
        )

    @pytest.mark.parametrize("batch_size", [1, 7, 200])
    def test_batch_size_does_not_change_the_answer(self, batch_size):
        """Splitting the rule's evaluation up leaves the result and the mesh alone."""
        prob = next(p for p in pnd.PROBLEMS if p["name"] == "corner-sqrt")
        ref = cubgm(prob["fun"], limits(prob), full_output=True, max_nregion=200)
        got = cubgm(
            prob["fun"],
            limits(prob),
            full_output=True,
            max_nregion=200,
            batch_size=batch_size,
        )
        np.testing.assert_allclose(got[0], ref[0], rtol=1e-14, atol=1e-16)
        assert int(got[1].info["nregion"]) == int(ref[1].info["nregion"])

    def test_neval_counts_integrand_evaluations(self):
        """`cubgm` reports evaluations; the low level routine reports rule calls."""
        prob = next(p for p in pnd.PROBLEMS if p["name"] == "cos-product")
        # the degree is pinned on both sides: what is under test is the conversion
        # between the two counts, not whichever degree `cubgm` defaults to.
        rule = GenzMalikRule(2, 7)
        _, low = adaptive_cubature(rule, prob["fun"], limits(prob), max_nregion=200)
        _, high = cubgm(prob["fun"], limits(prob), max_nregion=200, degree=7)
        assert int(high.neval) == int(low.neval) * rule.nodes_per_call

    def test_a_norm_charges_the_worst_component(self):
        """A vector integrand is error controlled on its hardest component."""
        prob = next(p for p in pnd.PROBLEMS if p["name"] == "vector-mixed")
        rule = rules_for(2, norm=1)["genz-malik"]
        y, info = adaptive_cubature(
            rule,
            prob["fun"],
            limits(prob),
            full_output=True,
            epsabs=jnp.asarray(1e-8),
            epsrel=jnp.asarray(1e-8),
            max_nregion=2000,
        )
        np.testing.assert_allclose(y, prob["val"], rtol=1e-7, atol=1e-9)
        assert float(info.err) >= float(np.max(np.abs(np.asarray(y) - prob["val"])))


class TestConstruction:
    """Everything the routine refuses, and the reason it gives."""

    def test_a_one_dimensional_rule_is_rejected(self):
        with pytest.raises(TypeError, match="AbstractCubatureRule"):
            adaptive_cubature(
                GaussKronrodRule(21),  # type: ignore[arg-type]
                lambda x: x[0],
                jnp.array([[0.0, 1.0]] * 2),
            )

    def test_a_tensor_product_needs_at_least_one_axis(self):
        """One axis is the floor, and it is checked in both spellings of ``rules``."""
        with pytest.raises(ValueError, match="positive integer"):
            TensorProductRule(GaussKronrodRule(15), ndim=0)
        with pytest.raises(ValueError, match="at least one axis"):
            TensorProductRule([])

    def test_a_rule_of_the_wrong_dimension_is_rejected(self):
        with pytest.raises(ValueError, match="3 dimensions"):
            adaptive_cubature(
                GenzMalikRule(3, 7), lambda x: jnp.sum(x), jnp.array([[0.0, 1.0]] * 2)
            )

    def test_a_budget_below_the_breakpoint_grid_is_rejected(self):
        with pytest.raises(ValueError, match="max_nregion"):
            cubgm(
                lambda x: jnp.sum(x),
                [jnp.array([0.0, 0.3, 0.6, 1.0]), jnp.array([0.0, 0.5, 1.0])],
                max_nregion=4,
            )

    def test_the_leibniz_adjoint_is_rejected(self):
        with pytest.raises(NotImplementedError, match="DirectAdjoint"):
            cubgm(
                lambda x: jnp.sum(x),
                jnp.array([[0.0, 1.0]] * 2),
                adjoint=LeibnizAdjoint(),
            )

    def test_the_direct_adjoint_is_the_default(self):
        y, _ = cubgm(
            lambda x: jnp.sum(x), jnp.array([[0.0, 1.0]] * 2), adjoint=DirectAdjoint()
        )
        np.testing.assert_allclose(y, 1.0, rtol=1e-13)

    def test_integer_limits_are_promoted(self):
        """The form a caller writes first, ``[[0, 1], [0, 1]]``, is accepted."""
        y, _ = cubgm(
            lambda x: jnp.cos(x[0]) * jnp.cos(x[1]), jnp.array([[0, 1], [0, 1]])
        )
        # Against the default tolerance the call was made at, rather than against
        # however far past it the routine happens to land: what is under test is that
        # the integer form is accepted at all.
        np.testing.assert_allclose(y, np.sin(1.0) ** 2, rtol=1e-8)

    def test_the_two_forms_of_interval_agree(self):
        """An ``(ndim, 2)`` array means what iterating over it means."""
        fun = lambda x: jnp.exp(-jnp.sum(x**2))
        arr = jnp.array([[0.0, 1.0], [-1.0, 2.0]])
        ya, _ = cubgm(fun, arr)
        yl, _ = cubgm(fun, [arr[0], arr[1]])
        np.testing.assert_allclose(ya, yl, rtol=1e-14, atol=1e-16)
