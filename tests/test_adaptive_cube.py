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
import scipy.special
from packaging.version import Version

from quadax import (
    STATUS,
    ClenshawCurtisRule,
    DirectAdjoint,
    GaussKronrodRule,
    LeibnizAdjoint,
    adaptive_quadrature,
)
from quadax._adaptive_cube import adaptive_cubature, cubegm
from quadax._fixed_cubature import (
    AbstractCubatureRule,
    GenzMalikRule,
    TensorProductRule,
)

from . import problems_nd as pnd

# XLA before jax 0.7.0 compiles the boundary term over a box to something that returns
# NaN when it is differentiated twice with a reverse pass on the outside. The program
# itself is right: on those versions the same call with `jit` disabled returns the
# correct value, and every nesting whose outer pass is forward is correct with it on.
# Only the box meets this, because there the boundary term is an integral over a face
# and runs a nested solve, where in one dimension a face is a point and there is no
# solve to compile.
_XLA_MISCOMPILES_THE_FACE_TERM = Version(jax.__version__) < Version("0.7.0")


def _grad_of_grad(f):
    """Reverse over reverse: the nesting whose outer pass transposes the inner one."""
    return jax.grad(jax.grad(f))


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


class TestExtrapolation:
    """The epsilon table on the sequence of running totals the subdivision produces.

    Both problems below are ones the subdivision can only ever bisect towards, so that
    the totals it produces have a limit worth inferring: a kink the mesh has to find for
    itself, and an algebraic singularity at a corner of the box. What the limit is worth
    differs between them, and each test asserts the gain its own problem buys.
    """

    # Tight enough that the mesh alone cannot reach it on either problem without the
    # table, which is what leaves the acceleration something to do.
    TOL = 1e-12

    def _run(self, name):
        """One problem, solved with the acceleration off and then on."""
        prob = next(p for p in pnd.PROBLEMS if p["name"] == name)
        off = solve("genz-malik", prob, self.TOL, extrapolate=False)
        on = solve("genz-malik", prob, self.TOL, extrapolate=True)
        return prob, off, on

    def _rel_err(self, y, prob):
        """Relative error of a value against the problem's exact one."""
        exact = np.asarray(prob["val"])
        return np.max(np.abs(np.asarray(y) - exact)) / np.max(np.abs(exact))

    # Whether the table's limit is the value kept turns on its own error estimate
    # rather than on its accuracy. A repeated extrapolation earns no place among the
    # three the estimate is the spread of, so once the table settles the estimate stops
    # at the distance to the last two values it moved through, which here is orders
    # above where it has in fact arrived. How close together those last two land, and
    # so whether the estimate clears the tolerance asked for, comes out differently on
    # different builds of the same program; how close the table got does not.
    @pytest.mark.xfail(
        strict=False,
        reason="whether the estimate clears the tolerance is platform dependent",
    )
    def test_a_limit_the_mesh_can_only_approach_is_named_outright(self):
        """On a kink the mesh has to localize, the gain is accuracy.

        The totals converge to a limit the table names to within a few ulp, two orders
        past where the subdivision alone stops.
        """
        prob, (y_off, _), (y_on, on) = self._run("kink-unmarked")
        assert bool(on.info["used_accel"]), (
            "the extrapolated value was not the one kept"
        )
        err_mesh = self._rel_err(y_off, prob)
        err_accel = self._rel_err(y_on, prob)
        # Measured gains run from 180x to 370x.
        assert err_accel < err_mesh / 100, f"{err_mesh:.2e} -> {err_accel:.2e}"
        pnd.assert_honest(y_on, on, prob, self.TOL)

    def test_the_same_accuracy_is_reached_on_a_coarser_mesh(self):
        """On a corner singularity, the gain is cost rather than accuracy.

        The extrapolated value is no more accurate than the one the subdivision arrives
        at by itself, and it is reached from a third fewer regions.
        """
        prob, (y_off, off), (y_on, on) = self._run("corner-sqrt")
        assert bool(on.info["used_accel"]), (
            "the extrapolated value was not the one kept"
        )
        err_off = self._rel_err(y_off, prob)
        err_on = self._rel_err(y_on, prob)
        # Both runs land within a hundred ulp of the exact value, so what separates them
        # is the order hundreds of regions happened to be summed in rather than anything
        # the acceleration did. The floor is what stops the comparison ranking two
        # answers on their roundoff. Above the floor the comparison still binds.
        floor = 100 * np.finfo(float).eps
        assert err_on <= max(err_off, floor), f"{err_off:.2e} -> {err_on:.2e}"
        # Measured gains run from 1.7x to 2.7x.
        assert int(on.info["nregion"]) < 0.75 * int(off.info["nregion"])
        pnd.assert_honest(y_on, on, prob, self.TOL)

    def test_a_divergent_integral_is_flagged(self):
        """A divergent integrand must not come back looking converged.

        The epsilon algorithm sums a divergent series the way Pade approximants do and
        is indifferent to whether the limit it infers exists, so it returns the analytic
        continuation of the convergent case. What keeps that from being silently wrong
        is the flag, not the value.
        """
        # int_0^1 int_0^1 (u + v)**-s = (2**(2-s) - 2) / ((1-s)(2-s)), which
        # continues to -3/4 at s = 3, where the integral itself diverges at the
        # origin.
        y, info = cubegm(
            lambda u: (u[0] + u[1]) ** -3.0,
            [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])],
            epsabs=1e-10,
            epsrel=1e-10,
            max_nregion=500,
            extrapolate=True,
        )
        np.testing.assert_allclose(float(y), -0.75, rtol=1e-6)
        assert int(info.status) == STATUS.divergent

    @pytest.mark.parametrize("i", pnd.SMOOTH, ids=pnd.problem_id)
    def test_a_smooth_problem_still_gets_what_it_asked_for(self, request, i):
        """Turning it on where there is nothing to accelerate must not cost accuracy.

        Unlike the one dimensional routines, a run over a box may return an extrapolated
        value on a smooth integrand: the initial mesh is a single cell, so the depth
        budget is reached after one cut and the table is fed while the subdivision is
        still coarse. That is allowed, the run stopping as soon as what it holds meets
        the tolerance whichever of the two supplied it, but what comes back still has to
        be inside the tolerance it claims.
        """
        prob = pnd.PROBLEMS[i]
        tol = 1e-8
        pnd.xfail_if_known(request, "genz-malik", prob, tol)
        y, info = solve("genz-malik", prob, tol, extrapolate=True)
        pnd.assert_converged(y, info, prob, tol)
        pnd.assert_honest(y, info, prob, tol)

    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    @pytest.mark.parametrize(
        "adjoint", [DirectAdjoint(), LeibnizAdjoint()], ids=["direct", "leibniz"]
    )
    def test_an_accelerated_solve_is_differentiable(self, adjoint, transform):
        """The value differentiated has to be the one returned.

        An accelerated solve returns the limit the table inferred rather than the sum
        over the mesh, so ``DirectAdjoint`` cannot take its derivative on the mesh
        alone: it replays the whole sequence of running totals on a subdivision rebuilt
        from the limits, and extrapolates that again. ``LeibnizAdjoint`` runs its own
        solve and needs none of it, which is what makes the two a check on each other.
        The run has to have kept an extrapolation for any of this to be under test.
        """
        fun = lambda u: 1 / jnp.sqrt(u[0] + u[1])  # noqa: E731
        run = lambda z, **kw: cubegm(  # noqa: E731
            fun,
            [jnp.stack([jnp.zeros_like(z), z]), jnp.array([0.0, 1.0])],
            epsabs=1e-10,
            epsrel=1e-10,
            max_nregion=300,
            extrapolate=True,
            **kw,
        )
        b = jnp.asarray(0.7)
        assert bool(run(b, full_output=True)[1].info["used_accel"]), "none was kept"
        f = lambda z: run(z, adjoint=adjoint)[0]  # noqa: E731
        # d/db int_0^b int_0^1 (u + v)**-0.5 = int_0^1 (b + v)**-0.5 dv
        want = 2 * (np.sqrt(1.7) - np.sqrt(0.7))
        np.testing.assert_allclose(float(transform(f)(b)), want, rtol=1e-8)


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
            max_ninter=500,
        )
        assert nd.status == STATUS.normal and one.status == STATUS.normal
        assert y_nd == y_1d
        assert int(nd.info["nregion"]) == int(one.info["ninter"])
        np.testing.assert_allclose(float(nd.err), float(one.err), rtol=0.05)

    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    def test_the_leibniz_boundary_term_reduces_too(self, transform):
        """Over one axis a face of the box is a point, and the term is a difference.

        The boundary term is an integral over a face, which has one dimension fewer
        than the box, so over a single axis it is an integral over nothing: one
        evaluation of the integrand at the single point the face consists of. Nothing
        special marks that case in the code, it is the general one at ``ndim - 1 == 0``,
        and this is what says so. A jump at a moving breakpoint is the case that would
        come back as zero if the term were dropped rather than degenerating.
        """
        step = lambda t, z: jnp.where(t > z[0], 1.0, 0.0)  # noqa: E731
        limits = lambda s: jnp.stack(  # noqa: E731
            [-jnp.ones_like(s), s, jnp.ones_like(s)]
        )
        nd = lambda s: adaptive_cubature(  # noqa: E731
            TensorProductRule(GaussKronrodRule(21), ndim=1),
            lambda x, z: step(x[0], z),
            [limits(s)],
            (jnp.atleast_1d(s),),
            adjoint=LeibnizAdjoint(),
        )[0]
        one = lambda s: adaptive_quadrature(  # noqa: E731
            GaussKronrodRule(21),
            step,
            limits(s),
            (jnp.atleast_1d(s),),
            extrapolate=False,
            adjoint=LeibnizAdjoint(),
        )[0]
        s = jnp.asarray(0.3)
        np.testing.assert_allclose(float(nd(s)), float(one(s)), rtol=1e-13)
        np.testing.assert_allclose(
            float(transform(nd)(s)), float(transform(one)(s)), rtol=1e-13, atol=1e-15
        )
        np.testing.assert_allclose(float(transform(nd)(s)), -1.0, atol=1e-9)


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

        The floor is something the subdivision runs into, so the acceleration has to be
        off for it to be what the run stops on. With it on the loop stops refining the
        singular region once it is deep enough and spends the budget elsewhere, so at
        this budget it exhausts that first and reports it instead. Both are honest, the
        reported error being the size of the answer either way, but only one names the
        difficulty; what the accelerating run has to promise is that it does not call
        this converged.
        """
        limits = [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])]
        _, divergent = cubegm(
            lambda x: 1 / jnp.sum(x**2),
            limits,
            epsabs=1e-10,
            epsrel=1e-10,
            max_nregion=2000,
            extrapolate=False,
        )
        assert divergent.status == STATUS.bad_integrand
        _, accelerated = cubegm(
            lambda x: 1 / jnp.sum(x**2),
            limits,
            epsabs=1e-10,
            epsrel=1e-10,
            max_nregion=2000,
        )
        assert accelerated.status != STATUS.normal
        _, integrable = cubegm(
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
        run = lambda b: cubegm(fun, jnp.stack([jnp.zeros(2), b], axis=-1))[0]
        bs = jnp.array([[1.0, 1.0], [2.0, 0.5], [0.3, 3.0]])
        np.testing.assert_allclose(
            jax.vmap(run)(bs), jnp.stack([run(b) for b in bs]), rtol=1e-13, atol=1e-15
        )

    def test_it_is_batched_over_args(self):
        """`vmap` over an extra argument matches the loop over it."""
        fun = lambda x, p: jnp.exp(-p * jnp.sum(x))
        run = lambda p: cubegm(
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
        by_arg = lambda p: cubegm(
            fun, [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])], args=(p,)
        )[0]
        exact_arg = lambda p: ((1 - jnp.exp(-p)) / p) ** 2
        for f, exact, x in (
            (by_arg, exact_arg, 1.3),
            # d/db of int_0^b int_0^1 cos(u)cos(v) = cos(b) sin(1)
            (
                lambda b: cubegm(
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
                lambda c: cubegm(
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

    @pytest.mark.parametrize(
        "adjoint", [LeibnizAdjoint(), DirectAdjoint()], ids=["leibniz", "direct"]
    )
    @pytest.mark.parametrize(
        "second",
        [
            lambda f: jax.jacfwd(jax.jacfwd(f)),
            jax.hessian,
            _grad_of_grad,
        ],
        ids=["jacfwd^2", "hessian", "grad^2"],
    )
    def test_second_derivatives(self, request, second, adjoint):
        """Both adjoints survive being differentiated twice, in every nesting.

        One parameter reaches the answer by every route at once: through an argument of
        the integrand, through a limit, and through a breakpoint that a discontinuity
        is pinned to. So the second derivative is taken of an interior solve, of an
        outer face and of a jump between two faces together, and the exact answer is
        wrong if any one of them is.
        """
        if (
            _XLA_MISCOMPILES_THE_FACE_TERM
            and second is _grad_of_grad
            and isinstance(adjoint, LeibnizAdjoint)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"jax {jax.__version__} compiles the boundary term to NaN",
                    strict=False,
                )
            )
        fun = lambda x, z: jnp.where(x[0] > z[0], 5.0, 1.0) * jnp.exp(-z[0] * x[1])
        f = lambda s: cubegm(  # noqa: E731
            fun,
            [
                jnp.stack([jnp.zeros_like(s), s, jnp.ones_like(s)]),
                jnp.stack([jnp.zeros_like(s), s]),
            ],
            (jnp.atleast_1d(s),),
            adjoint=adjoint,
        )[0]
        # int_0^1 int_0^s = (5 - 4s) (1 - exp(-s^2))/s, itself differentiated twice
        exact = lambda s: (5 - 4 * s) * (1 - jnp.exp(-s * s)) / s  # noqa: E731
        s = jnp.asarray(0.37)
        want = float(jax.jacfwd(jax.jacfwd(exact))(s))
        got = float(second(f)(s))
        assert np.isfinite(got)
        np.testing.assert_allclose(got, want, rtol=1e-8)

    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    def test_the_adjoints_options_do_not_move_the_derivative(self, transform):
        """Chunking and checkpointing trade memory against speed and nothing else.

        The derivative is taken on the mesh the solve settled on, which is evaluated in
        blocks of regions with the slots past the end of the mesh masked out, so a chunk
        size that does not divide the slot count is the one that has to pad. Taken with
        respect to a limit, which is the case that rebuilds the mesh from the limits
        rather than reusing the corners as they were recorded.
        """
        fun = lambda x, p: jnp.exp(-p * jnp.sum(x**2))
        run = lambda adjoint: transform(
            lambda z: cubegm(
                fun,
                [jnp.array([0.0, z]), jnp.array([0.0, 1.0])],
                args=(1.3,),
                max_nregion=50,
                adjoint=adjoint,
            )[0]
        )(0.8)
        want = run(DirectAdjoint())
        for adjoint in (
            DirectAdjoint(chunk_size=3),
            DirectAdjoint(checkpoint=True),
            DirectAdjoint(chunk_size=1, checkpoint=True),
        ):
            np.testing.assert_allclose(run(adjoint), want, rtol=1e-13, atol=1e-15)


class TestLeibnizAdjoint:
    """The boundary term over a box, which is an integral over a face of it."""

    # A jump on a hyperplane normal to the first axis, marked with a breakpoint tied to
    # the same parameter that positions it, which is the supported way to write one.
    # `int_0^1 int_0^1 = (5 - 4s) sin(1)`, so the whole derivative is the jump term.
    JUMP = staticmethod(lambda x, z: jnp.where(x[0] > z[0], 5.0, 1.0) * jnp.cos(x[1]))

    def _jump_problem(self, adjoint, **kwargs):
        return lambda s: cubegm(
            self.JUMP,
            [
                jnp.stack([jnp.zeros_like(s), s, jnp.ones_like(s)]),
                jnp.array([0.0, 1.0]),
            ],
            (jnp.atleast_1d(s),),
            adjoint=adjoint,
            **kwargs,
        )[0]

    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    @pytest.mark.parametrize(
        "adjoint", [LeibnizAdjoint(), DirectAdjoint()], ids=["leibniz", "direct"]
    )
    def test_a_moving_jump_gets_its_face_term(self, adjoint, transform):
        """A discontinuity moving with its breakpoint, differentiated two ways.

        Differentiating a jump gives a delta that no quadrature of the integrand's
        tangent can represent, so it has to come from the motion of the breakpoint.
        The two adjoints recover it by different routes and neither is a check on the
        other: ``DirectAdjoint`` differentiates the mesh, which moves with the
        breakpoint, so the two contributions cancel term by term at shared abscissae;
        ``LeibnizAdjoint`` integrates the integrand's jump over the face the breakpoint
        lies in.
        """
        f = self._jump_problem(adjoint)
        s = jnp.asarray(0.37)
        np.testing.assert_allclose(float(f(s)), (5 - 4 * 0.37) * np.sin(1.0), atol=1e-9)
        np.testing.assert_allclose(
            float(transform(f)(s)), -4 * np.sin(1.0), rtol=1e-6, atol=1e-9
        )

    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    def test_the_face_is_built_on_the_axis_that_moves(self, transform):
        """Which axis carries the moving feature has to survive the trip to the term.

        An axis that is not being differentiated is handed down as ``None``, which is a
        pytree in its own right and vanishes from a flattened interval unless it is
        asked for. Losing it renumbers the axes, and the boundary term is then built on
        the wrong face, so this puts the only moving axis in the middle where a
        renumbering cannot come out right by luck.
        """
        fun = lambda x, z: (  # noqa: E731
            jnp.where(x[1] > z[0], 5.0, 1.0) * jnp.exp(-jnp.sum(x[::2] ** 2))
        )
        f = lambda s: cubegm(  # noqa: E731
            fun,
            [
                jnp.array([0.0, 1.0]),
                jnp.stack([jnp.zeros_like(s), s, jnp.ones_like(s)]),
                jnp.array([0.0, 1.0]),
            ],
            (jnp.atleast_1d(s),),
            adjoint=LeibnizAdjoint(),
        )[0]
        # -4 (int_0^1 exp(-u^2) du)^2, the jump times the face it is integrated over
        want = -4 * (np.sqrt(np.pi) / 2 * scipy.special.erf(1.0)) ** 2
        np.testing.assert_allclose(
            float(transform(f)(jnp.asarray(0.37))), want, rtol=1e-8
        )

    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    def test_a_moving_limit_is_the_face_integral(self, transform):
        """In three dimensions a face is a genuine two dimensional cubature.

        ``d/db int_0^b int_0^1 int_0^1 cos(u)cos(v)cos(w) = cos(b) sin(1)^2``, the
        right hand side being exactly the integral over the face `u = b` that the
        boundary term has to produce.
        """
        f = lambda b: cubegm(  # noqa: E731
            lambda u: jnp.cos(u[0]) * jnp.cos(u[1]) * jnp.cos(u[2]),
            [jnp.array([0.0, b]), jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])],
            adjoint=LeibnizAdjoint(),
        )[0]
        b = jnp.asarray(0.7)
        np.testing.assert_allclose(
            float(transform(f)(b)), np.cos(0.7) * np.sin(1.0) ** 2, rtol=1e-8
        )

    def test_a_direction_that_moves_one_axis_only(self):
        """A face is skipped when the limits it would be multiplied by are not moving.

        ``interval`` is one array per axis, so an axis is live or not as a whole and
        whether it is actually moving is known only once there are values. Forward mode
        checks, and skips the faces of an axis whose limits come out stationary, which
        is exact rather than approximate: a face multiplied by zero contributes nothing.
        A directional derivative along one axis' limit is the case that has something to
        skip, and it has to agree with the column of the full Jacobian that it is.
        """
        fun = lambda x, p: jnp.exp(-p * jnp.sum(x**2))  # noqa: E731
        f = lambda z: cubegm(  # noqa: E731
            fun,
            [
                jnp.stack([jnp.zeros_like(z[0]), z[0]]),
                jnp.stack([jnp.zeros_like(z[1]), z[1]]),
            ],
            (1.3,),
            adjoint=LeibnizAdjoint(),
        )[0]
        z = jnp.array([0.8, 1.1])
        jac = jax.jacfwd(f)(z)
        for k in range(2):
            v = jnp.zeros(2).at[k].set(1.0)
            np.testing.assert_allclose(
                float(jax.jvp(f, (z,), (v,))[1]), float(jac[k]), rtol=1e-13, atol=1e-15
            )

    def test_the_face_options_are_scoped_to_the_faces(self):
        """``options_face`` reaches every face and nothing but the faces.

        It has no forward and reverse halves, unlike the options for the interior
        solve, because the face integrals are primal quantities and both directions
        integrate the identical ones. An unknown option is the cheapest way to see
        where a set of them lands: one in ``options_rev`` reaches only reverse mode,
        one in ``options_face`` reaches both.
        """
        s = jnp.asarray(0.37)
        run = lambda adjoint, transform: transform(  # noqa: E731
            self._jump_problem(adjoint)
        )(s)
        # the primal runs whatever the adjoint was given, having no face to integrate
        self._jump_problem(LeibnizAdjoint(options_face={"nope": 1}))(s)

        rev_only = LeibnizAdjoint(options_rev={"nope": 1})
        run(rev_only, jax.jacfwd)
        with pytest.raises(TypeError, match="nope"):
            run(rev_only, jax.jacrev)

        faces = LeibnizAdjoint(options_face={"nope": 1})
        for transform in (jax.jacfwd, jax.jacrev):
            with pytest.raises(TypeError, match="nope"):
                run(faces, transform)

    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    def test_a_face_rule_may_be_given_outright(self, transform):
        """A rule of the face's own dimension in ``options_face`` is used as it stands.

        The default is to derive one from the box rule per axis, which a rule that
        cannot be built one axis shorter has no way of doing. Handing one over is the
        way round that, and it has to give the same answer.
        """
        face = TensorProductRule(GaussKronrodRule(21), ndim=1)
        want = transform(self._jump_problem(LeibnizAdjoint()))(jnp.asarray(0.37))
        got = transform(
            self._jump_problem(LeibnizAdjoint(options_face={"rule": face}))
        )(jnp.asarray(0.37))
        np.testing.assert_allclose(float(got), float(want), rtol=1e-8)


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
        ref = cubegm(prob["fun"], limits(prob), full_output=True, max_nregion=200)
        got = cubegm(
            prob["fun"],
            limits(prob),
            full_output=True,
            max_nregion=200,
            batch_size=batch_size,
        )
        np.testing.assert_allclose(got[0], ref[0], rtol=1e-14, atol=1e-16)
        assert int(got[1].info["nregion"]) == int(ref[1].info["nregion"])

    def test_neval_counts_integrand_evaluations(self):
        """`cubegm` reports evaluations; the low level routine reports rule calls."""
        prob = next(p for p in pnd.PROBLEMS if p["name"] == "cos-product")
        # the degree is pinned on both sides: what is under test is the conversion
        # between the two counts, not whichever degree `cubegm` defaults to.
        rule = GenzMalikRule(2, 7)
        _, low = adaptive_cubature(rule, prob["fun"], limits(prob), max_nregion=200)
        _, high = cubegm(prob["fun"], limits(prob), max_nregion=200, degree=7)
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


class TestVariableLimits:
    """Regions that are not boxes, given as limits that depend on the outer axes."""

    # One region per thing none of the others reaches: the triangle for a boundary that
    # closes to zero width at an apex, the disc for a curved one whose square root costs
    # the subdivision far more than its area suggests, the simplex for limits that
    # depend on limits, and the last for a dependent limit that is unbounded.
    REGIONS = [
        (
            "triangle",
            lambda x: x[0] * x[1],
            [jnp.array([0.0, 1.0]), lambda xp: jnp.array([0.0, 1.0 - xp[0]])],
            1 / 24,
        ),
        (
            "disc",
            lambda x: jnp.ones_like(x[0]),
            [
                jnp.array([-1.0, 1.0]),
                lambda xp: jnp.sqrt(1 - xp[0] ** 2) * jnp.array([-1.0, 1.0]),
            ],
            np.pi,
        ),
        (
            "simplex",
            lambda x: x[0] * x[1] * x[2],
            [
                jnp.array([0.0, 1.0]),
                lambda xp: jnp.array([0.0, 1.0 - xp[0]]),
                lambda xp: jnp.array([0.0, 1.0 - xp[0] - xp[1]]),
            ],
            1 / 720,
        ),
        (
            "unbounded-above-a-line",
            lambda x: jnp.exp(-x[1]),
            [jnp.array([0.0, 1.0]), lambda xp: jnp.array([xp[0], jnp.inf])],
            1 - np.exp(-1.0),
        ),
    ]

    @pytest.mark.parametrize(
        "fun,interval,val", [r[1:] for r in REGIONS], ids=[r[0] for r in REGIONS]
    )
    def test_it_integrates_over_the_region(self, fun, interval, val):
        y, info = cubegm(
            fun,
            interval,
            full_output=True,
            epsabs=jnp.asarray(1e-10),
            epsrel=jnp.asarray(1e-10),
            max_nregion=2000,
        )
        assert info.status == STATUS.normal
        np.testing.assert_allclose(y, val, rtol=1e-9)
        assert float(info.err) >= abs(float(y) - val)

    def test_constant_limits_mean_the_same_either_way(self):
        """A callable that ignores its argument is the axis it returns.

        The reduction the whole transform rests on: over constant limits the piecewise
        affine map and the tail map are both the identity, so the two spellings of the
        same box have to agree to roundoff rather than merely to the tolerance.
        """
        fun = lambda x: jnp.exp(-jnp.sum(x**2))
        axis = jnp.array([2.0, 4.0, 7.0])
        ya, _ = cubegm(fun, [jnp.array([0.0, 1.0]), axis])
        yc, _ = cubegm(fun, [jnp.array([0.0, 1.0]), lambda xp: axis])
        np.testing.assert_allclose(ya, yc, rtol=1e-14)

    def test_a_moving_breakpoint_straightens_a_curved_kink(self):
        """A breakpoint the limits carry is worth what an axis aligned one is worth.

        The kink lies along a curve, which no entry of a constant ``interval`` can mark.
        Returned as a breakpoint of the dependent axis it becomes a plane of the mapped
        box, and the pair measures what that is worth: the two runs share an integrand
        and a region and differ only in whether the curve is given.
        """
        c = lambda x: 0.3 + 0.4 * x**2
        fun = lambda x: jnp.abs(x[1] - c(x[0]))
        marked = [jnp.array([0.0, 1.0]), lambda xp: jnp.array([0.0, c(xp[0]), 1.0])]
        plain = [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])]
        ym, im = cubegm(
            fun, marked, full_output=True, epsabs=1e-10, epsrel=1e-10, max_nregion=4000
        )
        _, iu = cubegm(
            fun, plain, full_output=True, epsabs=1e-10, epsrel=1e-10, max_nregion=4000
        )
        # int_0^1 int_0^1 |y - c(x)| dy dx = int_0^1 (c**2 - c + 1/2) dx
        np.testing.assert_allclose(ym, 0.09 + 0.08 + 0.032 - (0.3 + 0.4 / 3) + 0.5)
        assert im.status == STATUS.normal
        assert int(im.info["nregion"]) * 100 < int(iu.info["nregion"])

    @pytest.mark.parametrize(
        "adjoint", [DirectAdjoint(), LeibnizAdjoint()], ids=["direct", "leibniz"]
    )
    def test_it_is_differentiable_in_both_modes(self, adjoint):
        """A parameter in the limits is an ordinary parameter of the integrand.

        The three cases are the three ways one can reach the answer: through a finite
        dependent limit, through the finite end of an unbounded one, and through a
        breakpoint the dependent axis carries. The last is the one a constant
        ``interval`` cannot get right in more than one dimension, there being no term
        for a breakpoint plane that moves; straightened, the plane does not move and
        there is nothing extra to compute.
        """
        zero, one, inf = (jnp.zeros(()), jnp.ones(()), jnp.array(jnp.inf))

        def area(k):  # {0<=x<=1, 0<=y<=k(1-x)} has area k/2
            interval = [
                jnp.array([0.0, 1.0]),
                lambda xp: jnp.stack([zero, k * (1 - xp[0])]),
            ]
            return cubegm(lambda x: one, interval, adjoint=adjoint)[0]

        def tail(k):  # int_0^1 int_{kx}^inf exp(-y) dy dx = (1 - exp(-k)) / k
            interval = [jnp.array([0.0, 1.0]), lambda xp: jnp.stack([k * xp[0], inf])]
            return cubegm(
                lambda x: jnp.exp(-x[1]),
                interval,
                adjoint=adjoint,
                epsabs=1e-11,
                epsrel=1e-11,
                max_nregion=2000,
            )[0]

        def kink(k):  # int_0^1 int_0^1 |y - (0.3 + k x**2)| dy dx
            c = lambda x: 0.3 + k * x**2
            interval = [
                jnp.array([0.0, 1.0]),
                lambda xp: jnp.stack([zero, c(xp[0]), one]),
            ]
            return cubegm(
                lambda x: jnp.abs(x[1] - c(x[0])),
                interval,
                adjoint=adjoint,
                epsabs=1e-11,
                epsrel=1e-11,
                max_nregion=2000,
            )[0]

        cases = [
            (area, lambda k: k / 2, 0.8),
            (tail, lambda k: (1 - jnp.exp(-k)) / k, 1.0),
            (kink, lambda k: 0.09 + 0.2 * k + k**2 / 5 - 0.3 - k / 3 + 0.5, 0.4),
        ]
        for f, exact, k in cases:
            fwd, rev = float(jax.jacfwd(f)(k)), float(jax.jacrev(f)(k))
            np.testing.assert_allclose(fwd, rev, rtol=1e-13)
            np.testing.assert_allclose(fwd, float(jax.jacfwd(exact)(k)), rtol=1e-6)


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
            cubegm(
                lambda x: jnp.sum(x),
                [jnp.array([0.0, 0.3, 0.6, 1.0]), jnp.array([0.0, 0.5, 1.0])],
                max_nregion=4,
            )

    def test_a_rule_with_no_face_rule_is_rejected(self):
        """LeibnizAdjoint integrates over a face, which needs a rule one axis shorter.

        Both shipped families build one. A rule that does not can still be used, by
        handing the adjoint a face rule of its own, and the message says so.
        """

        class _NoFaceRule(AbstractCubatureRule):
            @property
            def ndim(self):
                return 2

            def integrate(self, fun, a, b, args):
                z = jnp.zeros(())
                return z, z, z, z, jnp.zeros(2)

        interval = jnp.array([[0.0, 1.0]] * 2)
        with pytest.raises(NotImplementedError, match="options_face"):
            adaptive_cubature(
                _NoFaceRule(), lambda x: jnp.sum(x), interval, adjoint=LeibnizAdjoint()
            )
        # the same rule is fine under the adjoint that never builds a face
        adaptive_cubature(
            _NoFaceRule(), lambda x: jnp.sum(x), interval, adjoint=DirectAdjoint()
        )

    def test_the_direct_adjoint_is_the_default(self):
        y, _ = cubegm(
            lambda x: jnp.sum(x), jnp.array([[0.0, 1.0]] * 2), adjoint=DirectAdjoint()
        )
        np.testing.assert_allclose(y, 1.0, rtol=1e-13)

    def test_integer_limits_are_promoted(self):
        """The form a caller writes first, ``[[0, 1], [0, 1]]``, is accepted."""
        y, _ = cubegm(
            lambda x: jnp.cos(x[0]) * jnp.cos(x[1]), jnp.array([[0, 1], [0, 1]])
        )
        # Against the default tolerance the call was made at, rather than against
        # however far past it the routine happens to land: what is under test is that
        # the integer form is accepted at all.
        np.testing.assert_allclose(y, np.sin(1.0) ** 2, rtol=1e-8)

    @pytest.mark.parametrize(
        "lim,err,match",
        [
            (lambda xp: jnp.zeros(()), ValueError, "one dimensional"),
            (lambda xp: jnp.zeros(1), ValueError, "one dimensional"),
            (lambda xp: jnp.zeros(2, dtype=complex), TypeError, "must be real"),
        ],
        ids=["scalar", "one-limit", "complex"],
    )
    def test_what_a_limit_callable_may_return_is_checked(self, lim, err, match):
        """A callable is probed for its limits, and told what they have to look like."""
        with pytest.raises(err, match=match):
            cubegm(lambda x: jnp.sum(x), [jnp.array([0.0, 1.0]), lim])

    def test_the_two_forms_of_interval_agree(self):
        """An ``(ndim, 2)`` array means what iterating over it means."""
        fun = lambda x: jnp.exp(-jnp.sum(x**2))
        arr = jnp.array([[0.0, 1.0], [-1.0, 2.0]])
        ya, _ = cubegm(fun, arr)
        yl, _ = cubegm(fun, [arr[0], arr[1]])
        np.testing.assert_allclose(ya, yl, rtol=1e-14, atol=1e-16)
