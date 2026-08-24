"""Tests for the fixed order quadrature rules in quadax/fixed_order.py.

Each rule has a defining property, and these tests pin it down:

- ``GaussKronrodRule`` is a Gauss-Kronrod rule: an ``order`` (odd) point rule embedding
  an ``order // 2`` point Gauss rule, exactly integrating all polynomials up to degree
  ``3 * (order // 2) + 1`` (22, 31, 46, 61, 76, 91 for the six QUADPACK orders, i.e.
  ``~3/2 the order``).
- ``ClenshawCurtisRule`` exactly integrates all polynomials up to degree ``order``.
- ``TanhSinhRule`` is a doubly-exponential trapezoidal rule: for analytic integrands
  each doubling of the order raises the error to a power near two once inside the
  asymptotic tail, which is far faster than any algebraic convergence rate.

Polynomial exactness is checked in the Chebyshev basis rather than with the monomial
``x**k``. ``{T_0, ..., T_D}`` spans the polynomials of degree at most ``D``, so checking
every ``T_k`` is equivalent to checking every such polynomial, but the bounded and
orthogonal Chebyshev polynomials are much more stable to evaluate to high order.
Monomials are catastrophically ill-conditioned near ``|x| = 1`` at high degree, causing
the quadrature sum and the analytic reference both round away the tiny contribution of
the high modes, and the rule can spuriously appear exact long past where it has any
business being). All checks run on the reference interval ``[-1, 1]`` and on a
shifted/scaled interval, so the affine map used internally to reach ``[a, b]`` is tested
too.

The rules are a public entry point in their own right rather than only the inside of the
adaptive loop, so the last section here checks that they honour the dtype of the limits
on their own, without a solver around them.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
from jax import config

from quadax import ClenshawCurtisRule, GaussKronrodRule, TanhSinhRule

from .problems import SLOP, ULP_ATOL, ULP_RTOL, exp_neg, real_dtypes

config.update("jax_enable_x64", True)

# A rule that is exact in pure arithmetic still evaluates the integrand and sums with
# roundoff, so "machine precision" means a relative error of a few * eps.
MACHINE_PRECISION = 5e-14

# The "not exact" checks assert the rule is wrong by at least this much, far beyond the
# few-eps wall of the regions where it is exact.
NOT_EXACT = 1e-3

# The rules are tested on the unit interval and on a shifted/scaled one, deliberately
# not centered on the origin so that the even/odd symmetry of the node tables (which
# would integrate odd integrands to zero for free) cannot hide anything.
INTERVALS = [(-1.0, 1.0), (1.7, 3.9), (-1.0, 2.0)]


def chebyshev(degree, x):
    """Chebyshev polynomial ``T_degree`` evaluated stably on the reference interval."""
    return jnp.cos(degree * jnp.arccos(x))


def mapped_chebyshev(x, degree, a, b):
    """Integrand ``T_degree`` of ``x`` with ``[a, b]`` mapped onto ``[-1, 1]``.

    ``degree`` and the interval are taken as arguments rather than closed over so that
    they reach the rule as traced, dynamic ``args``: only the integrand function is a
    static JIT key, so the whole degree sweep compiles once.
    """
    return chebyshev(degree, (2 * x - (a + b)) / (b - a))


def chebyshev_args(a, b, degree):
    """The ``args`` tuple ``(degree, a, b)`` for `mapped_chebyshev`.

    The args are passed as arrays rather than python numbers: eqx.filter_jit treats
    non-array leaf as static so python int/float ``degree`` (or ``a``/``b``) threading
    through ``args`` would recompile the integrand wrapper for each value instead of
    tracing it as data.
    """
    return (jnp.array(degree), jnp.array(a), jnp.array(b))


def chebyshev_integral(interval, degree):
    """Exact integral of ``T_degree`` over ``[a, b]`` in the mapped coordinate."""
    a, b = interval
    halflength = (b - a) / 2
    if degree % 2 == 0:
        return halflength * 2 / (1 - degree**2)
    return 0.0


def assert_exact(rule, interval, degree, tolerance):
    """Integrate T_degree over interval, close to the exact value within tol."""
    a, b = interval
    y, _, _, _ = rule.integrate(mapped_chebyshev, a, b, chebyshev_args(a, b, degree))
    np.testing.assert_allclose(
        float(y),
        chebyshev_integral(interval, degree),
        rtol=tolerance,
        atol=tolerance,
    )


def assert_not_exact(rule, interval, degree, threshold):
    """Integrate T_degree over interval, require error larger than threshold."""
    a, b = interval
    exact = chebyshev_integral(interval, degree)
    y, _, _, _ = rule.integrate(mapped_chebyshev, a, b, chebyshev_args(a, b, degree))
    assert abs(float(y) - exact) > threshold * abs(exact)


# The algebraic degree of exactness of each Gauss-Kronrod order: with ``n = (order - 1)
# / 2`` embedded Gauss points the Kronrod rule is exact to degree ``3n + 1``.
GAUSS_KRONROD_DEGREE = {15: 22, 21: 31, 31: 46, 41: 61, 51: 76, 61: 91}


@pytest.mark.parametrize("order", GAUSS_KRONROD_DEGREE)
class TestGaussKronrodPolynomialExactness:
    """A Gauss-Kronrod rule of order ``2n + 1`` is exact up to degree ``3n + 1``."""

    @pytest.mark.parametrize("interval", INTERVALS)
    def test_exact_up_to_degree(self, order, interval):
        """Every polynomial of degree ``3n + 1`` integrates to machine precision."""
        degree_of_exactness = GAUSS_KRONROD_DEGREE[order]
        rule = GaussKronrodRule(order=order)
        for k in range(degree_of_exactness + 1):
            assert_exact(rule, interval, k, MACHINE_PRECISION)

    @pytest.mark.parametrize("interval", INTERVALS)
    def test_not_exact_beyond_degree(self, order, interval):
        """The first even degree past ``3n + 1`` is not integrated exactly.

        A sanity check, if the rule returned the exact answer for every polynomial,
        ``test_exact_up_to_degree`` would prove nothing. (Even, not odd, degree: the
        node tables are symmetric, so every odd integrand integrates to zero whether
        or not it is resolved, and only the even part of a polynomial is under test.)
        """
        degree_of_exactness = GAUSS_KRONROD_DEGREE[order]
        k = degree_of_exactness + (2 if degree_of_exactness % 2 == 0 else 3)
        assert_not_exact(GaussKronrodRule(order=order), interval, k, NOT_EXACT)


@pytest.mark.parametrize("order", [16, 32, 64, 128, 256, 512])
class TestClenshawCurtisPolynomialExactness:
    """A Clenshaw-Curtis rule of order ``n`` is exact up to polynomial degree ``n``."""

    @pytest.mark.parametrize("interval", INTERVALS)
    def test_exact_up_to_degree(self, order, interval):
        """Every polynomial of degree ``order`` integrates to machine precision."""
        rule = ClenshawCurtisRule(order=order)
        for k in range(order + 1):
            assert_exact(rule, interval, k, MACHINE_PRECISION)

    @pytest.mark.parametrize("interval", INTERVALS)
    def test_not_exact_beyond_degree(self, order, interval):
        """The first even degree past ``order`` is not integrated exactly."""
        k = order + 2
        assert_not_exact(ClenshawCurtisRule(order=order), interval, k, NOT_EXACT)


def _exp_fun(x):
    return jnp.exp(x)


def _exp_exact(a, b):
    return np.exp(b) - np.exp(a)


def _gaussian_fun(x):
    return jnp.exp(-(x**2))


def _gaussian_exact(a, b):
    return np.sqrt(np.pi) / 2 * (scipy.special.erf(b) - scipy.special.erf(a))


def _rational_fun(x):
    return 1 / (1 + x**2)


def _rational_exact(a, b):
    return np.arctan(b) - np.arctan(a)


ANALYTIC_INTEGRANDS = [
    (_exp_fun, _exp_exact),
    (_gaussian_fun, _gaussian_exact),
    (_rational_fun, _rational_exact),
]

# The first doubling (9 -> 17) is pre-asymptotic and only monotonicity is required of
# it; from 17 on the error is required to take a power per doubling until it bottoms out
# in rounding.
TANHSINH_ORDERS = [9, 17, 33, 65, 129]
# Up to this error the convolved factors of ~eps make the observed error, and so its
# behavior under doubling, dominated by rounding rather than by the quadrature error.
ROUNDOFF_FLOOR = 2e-14

# Required power e0 -> e0**p per doubling. The tanh-sinh error behaves like
# exp(-c*n/log(n)), so a doubling raises the power by 2*log(n)/(log(n) + log(2)), which
# approaches but never reaches 2: near squaring while the error is coarse, and
# measurably under it once the error is small enough for the log(n) to matter. The
# threshold sits below that band so the test measures the rate rather than the exact
# factor, which is sensitive to the node range at the few percent level.
CONVERGENCE_POWER = 1.5


@pytest.mark.parametrize("fun, exact", ANALYTIC_INTEGRANDS)
@pytest.mark.parametrize("a, b", [(-1.0, 1.0), (0.3, 2.4), (-1.0, 2.0)])
class TestTanhSinhConvergence:
    """For analytic integrands the error decays faster than any algebraic rate."""

    def test_tanhsinh_exponential_convergence(self, fun, exact, a, b):
        """Doubling ``order`` raises the error to a power > 1 until rounding sets in."""
        target = exact(a, b)
        errors = [
            abs(float(TanhSinhRule(order=order).integrate(fun, a, b, ())[0]) - target)
            for order in TANHSINH_ORDERS
        ]

        # the error decays monotonically, the pre-asymptotic 9 -> 17 doubling included
        for e0, e1 in zip(errors, errors[1:]):
            if e0 > ROUNDOFF_FLOOR:
                assert e1 <= e0

        # inside the asymptotic tail (from 17 -> 33 on) each doubling raises the error
        # to a power well above one. Algebraic convergence gives a fixed factor per
        # point doubling instead, ie a power tending to one, so over the orders swept
        # here this rejects any algebraic rate up to about n**-13 and separates the
        # doubly exponential tanh-sinh behavior from the polynomial-rate rules above.
        for e0, e1 in zip(errors[1:], errors[2:]):
            if e1 > ROUNDOFF_FLOOR:
                assert e1 <= e0**CONVERGENCE_POWER

        # the sweep ends in rounding noise rather than a stalled algebraic tail
        assert errors[-1] < ROUNDOFF_FLOOR


# ---------------------------------------------------------------------------
# Error estimates
# ---------------------------------------------------------------------------
# Every rule returns an error estimate alongside its value, and the whole of the
# adaptive layer above rests on that number: it decides which sub-interval to split,
# when to stop, and what the caller is finally told the answer is worth. The contract
# it has to keep is one-sided. An estimate that is too large only costs work, while one
# that is too small is a wrong answer reported as a right one, because a caller has
# nothing else to go on. So these tests ask for a bound and not for a ratio, and a rule
# that cannot deliver one on some integrand is listed in `RULE_KNOWN_DISHONEST`.


def _sine_fun(x):
    return jnp.sin(5 * x)


def _sine_exact(a, b):
    return (np.cos(5 * a) - np.cos(5 * b)) / 5


# ``(fun, exact, interval)``, roughly in order of how hard the integrand makes the
# estimate's job: analytic, then peaked, then non-smooth at an interior point, then
# singular at an endpoint, which is where an estimate is under the most pressure and
# where the three rules stop agreeing about how well they are doing.
ERROR_CASES = {
    "exp": (_exp_fun, _exp_exact, (-1.0, 1.0)),
    "exp-shifted": (_exp_fun, _exp_exact, (1.7, 3.9)),
    "gaussian": (_gaussian_fun, _gaussian_exact, (-1.0, 2.0)),
    "rational": (_rational_fun, _rational_exact, (-1.0, 1.0)),
    "cubic": (
        lambda x: x**3 - 2 * x + 1,
        lambda a, b: (b**4 - a**4) / 4 - (b**2 - a**2) + (b - a),
        (-1.0, 2.0),
    ),
    "sine": (_sine_fun, _sine_exact, (0.0, 1.0)),
    "runge": (
        lambda x: 1 / (1 + 25 * x**2),
        lambda a, b: (np.arctan(5 * b) - np.arctan(5 * a)) / 5,
        (-1.0, 1.0),
    ),
    "kink": (
        lambda x: jnp.abs(x - 0.3),
        lambda a, b: (
            ((b - 0.3) ** 2 * np.sign(b - 0.3) + (0.3 - a) ** 2 * np.sign(0.3 - a)) / 2
        ),
        (0.0, 1.0),
    ),
    "sqrt": (jnp.sqrt, lambda a, b: 2 / 3 * (b**1.5 - a**1.5), (0.0, 1.0)),
    "recip-sqrt": (
        lambda x: x**-0.5,
        lambda a, b: 2 * (b**0.5 - a**0.5),
        (0.0, 1.0),
    ),
    "log": (
        jnp.log,
        lambda a, b: scipy.special.xlogy(b, b) - b - (scipy.special.xlogy(a, a) - a),
        (0.0, 1.0),
    ),
    "pow-0.9": (lambda x: x**-0.9, lambda a, b: 10 * (b**0.1 - a**0.1), (0.0, 1.0)),
    "complex-osc": (
        lambda x: jnp.exp(5j * x),
        lambda a, b: (np.exp(5j * b) - np.exp(5j * a)) / 5j,
        (0.0, 1.0),
    ),
}

# Three widely separated orders per rule rather than every order it offers: the estimate
# is built the same way at each, and what changes across an order sweep is how much of
# the integrand the nodes resolve, which the extremes already bracket.
RULE_ORDERS = {
    "gk": (GaussKronrodRule, [15, 31, 61]),
    "cc": (ClenshawCurtisRule, [8, 32, 128]),
    "ts": (TanhSinhRule, [21, 61, 101]),
}
RULE_ORDER_PARAMS = [
    (cls, order) for cls, orders in RULE_ORDERS.values() for order in orders
]
RULE_ORDER_IDS = [
    f"{name}-{order}" for name, (_, orders) in RULE_ORDERS.items() for order in orders
]

# ``(rule, case) -> {orders}`` where the reported error comes out below the true one.
# Kept as a table rather than folded into a tolerance so that the exceptions stay
# countable, and so that one which starts passing shows up as an XPASS instead of
# disappearing into the slack. The single entry is the endpoint singularity caveat in
# `ClenshawCurtisRule`'s own documentation: its nodes cluster at the endpoints, so the
# two rules of the nested pair agree closely there while neither has resolved anything,
# and the difference between them understates what is left. It clears as the order
# rises, being 2.3x at order 8 and honest from order 32 on.
RULE_KNOWN_DISHONEST: dict[tuple[str, str], set[int]] = {
    ("cc", "pow-0.9"): {8},
}


def _rule_name(rule):
    """The short key a rule appears under in `RULE_ORDERS`."""
    return next(k for k, (cls, _) in RULE_ORDERS.items() if cls is rule)


def xfail_if_rule_dishonest(request, rule, order, case):
    """Mark the running test xfail if `RULE_KNOWN_DISHONEST` lists this case."""
    orders = RULE_KNOWN_DISHONEST.get((_rule_name(rule), case))
    if orders is not None and order in orders:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    f"{rule.__name__} order {order} understates its error on {case}"
                ),
                strict=False,
            )
        )


# Checked in single as well as double precision. The estimate is assembled from
# quantities of the working dtype and compared against thresholds derived from its eps,
# so a term that is right only at float64 would show up here rather than in the dtype
# plumbing tests below.
ERROR_DTYPES = ["float64", "float32"]


@pytest.mark.parametrize("rule, order", RULE_ORDER_PARAMS, ids=RULE_ORDER_IDS)
class TestErrorEstimates:
    """What ``integrate`` reports as ``err`` has to bound the error it actually made."""

    @pytest.mark.parametrize("case", ERROR_CASES, ids=str)
    @pytest.mark.parametrize("dtype", ERROR_DTYPES)
    def test_error_estimate_is_honest(self, request, rule, order, case, dtype):
        """The reported error is at least the true one, with no margin allowed."""
        xfail_if_rule_dishonest(request, rule, order, case)
        fun, exact, (a, b) = ERROR_CASES[case]
        y, err, _, _ = rule(order).integrate(
            fun, jnp.array(a, dtype), jnp.array(b, dtype), ()
        )
        true_err = abs(complex(y) - complex(exact(a, b)))
        assert true_err <= float(err), (
            f"{case} at {dtype}: reported {float(err):.3e} < true {true_err:.3e}"
        )

    def test_a_zero_integrand_reports_no_error(self, rule, order):
        """An integrand that vanishes everywhere is integrated exactly.

        Worth pinning because the estimate is a chain of ratios and logarithms of
        sampled quantities, every one of which is degenerate here, and any of them
        resolving to a nan rather than being substituted away would surface as an
        error estimate on an integral that is exactly right.
        """
        _, err, y_abs, y_mmn = rule(order).integrate(lambda x: 0.0 * x, 0.0, 1.0, ())
        for v in (err, y_abs, y_mmn):
            np.testing.assert_array_equal(np.asarray(v), 0.0)

    def test_a_vector_integrand_is_charged_its_worst_component(self, rule, order):
        """``err`` for a vector integrand is the norm over the component errors.

        The default norm is the max, so integrating three integrands together has to
        report exactly what the worst of them reports on its own. Any part of the
        estimate that reduced over the components too early, rather than being formed
        component by component and normed once at the end, would come out below this.
        """
        step = lambda x: jnp.where(x < 0.3, 0.0, 1.0)
        funs = [_exp_fun, jnp.sqrt, step]
        together = lambda x: jnp.stack([f(x) for f in funs], axis=-1)
        r = rule(order)
        combined = float(r.integrate(together, 0.0, 1.0, ())[1])
        separate = [float(r.integrate(f, 0.0, 1.0, ())[1]) for f in funs]
        np.testing.assert_allclose(
            combined, max(separate), rtol=ULP_RTOL, atol=ULP_ATOL
        )


# The tanh-sinh rule is the only one that leaves part of the interval unsampled, and the
# tests below are about that gap. The other two are interpolatory over the whole of
# ``[a, b]``: their weights are fixed by requiring that polynomials up to some degree be
# integrated exactly over the closed interval, so the strip between the outermost node
# and the endpoint is already accounted for by the outermost weight, however far in that
# node happens to sit. A tanh-sinh rule is instead the trapezoidal rule for an integral
# over the whole real line in the mapped variable, cut off at a finite ``t``, and the
# terms past the cutoff carry mass that nothing in the rule compensates for.


def _tanhsinh_errors(fun, a, b, orders=TANHSINH_ORDERS):
    """Reported error of the tanh-sinh rule at each of ``orders``."""
    return [float(TanhSinhRule(order=o).integrate(fun, a, b, ())[1]) for o in orders]


class TestTanhSinhTruncation:
    """The mass beyond the outermost node is charged to the error estimate."""

    def test_raising_the_order_does_not_clear_an_endpoint_singularity(self):
        """Refining the mesh in ``t`` cannot shrink the sliver the map leaves out.

        On an analytic integrand the estimate falls away to the roundoff floor as the
        order rises, since the only error left is the trapezoidal one and that is what
        more nodes buy. On an integrand singular at an endpoint the omitted sliver
        carries real mass, the cutoff sits where it sits whatever the order, and the
        estimate has to keep saying so. The two behaviors are separated by more than
        ten orders of magnitude, so this measures the distinction rather than a rate.
        """
        orders = [21, 101]
        analytic = _tanhsinh_errors(_exp_fun, 0.0, 1.0, orders)
        singular = _tanhsinh_errors(lambda x: x**-0.9, 0.0, 1.0, orders)

        assert analytic[-1] / analytic[0] < 1e-9
        assert singular[-1] / singular[0] > 1e-2
        assert analytic[-1] < 1e-13
        assert singular[-1] > 1.0

    def test_a_barely_integrable_endpoint_still_gets_a_finite_bound(self):
        """``x**-p`` leaves a finite mass unsampled for every ``p`` below one.

        That mass is ``d**(1 - p) / (1 - p)`` over a gap of width ``d``: large as ``p``
        approaches one, but bounded, and the estimate has to say how large rather than
        give up.
        """
        for p in (0.9, 0.99, 0.999):
            fun = lambda x, p=p: x**-p
            y, err, _, _ = TanhSinhRule(order=61).integrate(fun, 0.0, 1.0, ())
            true_err = abs(float(y) - 1 / (1 - p))
            assert np.isfinite(float(err)), f"p={p}: reported a non-finite error"
            assert true_err <= float(err), (
                f"p={p}: reported {float(err):.3e} < true {true_err:.3e}"
            )

    def test_a_non_integrable_endpoint_is_reported_as_unbounded(self):
        """``1/x`` on ``[0, 1]`` has no finite integral and so no finite error bound.

        The rule returns a number regardless, since it only ever sees finite samples,
        and an unbounded error is what says that number means nothing. This is the one
        case that reports a non-finite error, and it is the honest report rather than a
        gap in the estimate.
        """
        _, err, _, _ = TanhSinhRule(order=61).integrate(lambda x: 1 / x, 0.0, 1.0, ())
        assert not np.isfinite(float(err))

    def test_the_estimate_survives_a_node_rounding_onto_the_endpoint(self):
        """A sub-interval far from the origin relative to its width is the worst case.

        There the outermost nodes round onto the endpoint itself, an integrand singular
        there is evaluated at the singularity, and the non-finite value it returns is
        masked away. Read naively that says the outermost term is zero and so there is
        no tail, which is exactly backwards: it is the case where the tail is largest.
        """
        fun = lambda x: 1 / jnp.sqrt(1 - x**2)
        a, b = 0.5, 1.0
        exact = np.arcsin(b) - np.arcsin(a)

        for order in (21, 61, 101):
            rule = TanhSinhRule(order=order)
            # the premise of the test: the outermost node really has collapsed onto the
            # endpoint, so it is the masking path being exercised and not the ordinary
            # one. Without this the test would keep passing for the wrong reason if the
            # node placement ever changed.
            xh = rule._nodes_weights(jnp.float64)[0]
            assert float((b + a) / 2 + (b - a) / 2 * xh[-1]) == b

            y, err, _, _ = rule.integrate(fun, a, b, ())
            true_err = abs(float(y) - exact)
            assert true_err <= float(err), (
                f"order {order}: reported {float(err):.3e} < true {true_err:.3e}"
            )


# The dtype of the limits is the statement of what precision the caller wants, and the
# rules are a public entry point in their own right rather than only the inside of the
# adaptive loop, so they have to honour it on their own.

RULES = [GaussKronrodRule, ClenshawCurtisRule, TanhSinhRule]


@pytest.mark.usefixtures("quiet_tanhsinh")
class TestFixedOrderRuleDTypes:
    """The fixed order rules are a public entry point in their own right."""

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_integrate(self, rule, dtype):
        """All four outputs of ``integrate`` come back at the abscissa dtype."""
        a, b = jnp.array(0.0, dtype), jnp.array(1.0, dtype)
        y, err, y_abs, y_mmn = rule().integrate(exp_neg, a, b, ())
        assert y.dtype == dtype
        assert err.dtype == y_abs.dtype == y_mmn.dtype == dtype
        tol = SLOP * np.sqrt(float(jnp.finfo(dtype).eps))
        np.testing.assert_allclose(float(y), 1 - np.exp(-1), atol=tol)

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_degenerate_interval(self, rule, dtype):
        """``a == b`` takes the other branch of a ``cond``, which has to agree.

        Both branches are built for any integrand dtype, so the zero branch must be
        constructed at the same dtype the weights promote the real branch to.
        """
        a = jnp.array(0.5, dtype)
        out = rule().integrate(exp_neg, a, a, ())
        for v in out:
            assert v.dtype == dtype
            np.testing.assert_array_equal(np.asarray(v), 0.0)

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_apply(self, rule, dtype):
        """The low level ``_apply`` keeps the dtype too."""
        a, b = jnp.array(0.0, dtype), jnp.array(1.0, dtype)
        y = rule()._apply(exp_neg, a, b, ())
        assert y.dtype == dtype

    @pytest.mark.parametrize("rule", RULES)
    def test_weights_sum_to_two(self, rule):
        """Both the high and low order rules integrate 1 over [-1, 1] exactly."""
        r = rule()
        np.testing.assert_allclose(float(jnp.sum(r._wh)), 2.0, atol=1e-14)
        np.testing.assert_allclose(float(jnp.sum(r._wl)), 2.0, atol=1e-14)


@pytest.mark.usefixtures("quiet_tanhsinh")
@pytest.mark.parametrize("dtype", real_dtypes)
def test_tanhsinh_nodes_stay_inside_the_interval(dtype):
    """Rebuilt rather than cast, so no node collapses onto the endpoint.

    Casting a float64 table down to bfloat16 would round the outer nodes to exactly
    +/-1, silently dropping the effective order. Rebuilding at the target dtype
    spreads the same number of nodes over the range that dtype can resolve.
    """
    xh, _, _ = TanhSinhRule(order=61)._nodes_weights(dtype)
    assert xh.dtype == dtype
    assert len(np.unique(np.asarray(xh, dtype=np.float64))) == len(xh)
    assert np.all(np.abs(np.asarray(xh, dtype=np.float64)) < 1.0)


# The batch sizes worth covering, relative to a rule's node count ``n``: one point at a
# time, a divisor of nothing in particular so the last batch is partial, an exact
# divisor of nothing again but larger, and a value above ``n`` so the clip fires. Given
# as a callable of ``n`` because the three rules have different node counts, and the
# interesting values are the ones that straddle it.
BATCH_SIZES = [
    lambda n: 1,
    lambda n: 4,
    lambda n: 5,
    lambda n: n - 1,
    lambda n: n,
    lambda n: n + 7,
]
BATCH_IDS = ["1", "4", "5", "n-1", "n", "n+7"]


@pytest.mark.usefixtures("quiet_tanhsinh")
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=BATCH_IDS)
class TestBatchSize:
    """Splitting the node evaluation into batches must not change the answer.

    The node count is fixed when the rule is built, so the batches are cut to fit: the
    weighted sums see the same values an unbatched rule computes, in the same order,
    with no term reassociated, and the agreement is to the last few ulp rather than to
    the tolerance the quadrature was asked for. It is not bitwise, and
    must not be asserted as such, evaluating the integrand under a different batch
    shape lets XLA fuse the arithmetic differently, which moves results by around eps
    without any reordering having happened.
    """

    def test_integrate_is_unchanged(self, rule, batch_size):
        """All four outputs match the unbatched rule to within roundoff."""
        n = len(rule()._xh)
        want = rule().integrate(exp_neg, 0.0, 1.0, ())
        got = rule(batch_size=batch_size(n)).integrate(exp_neg, 0.0, 1.0, ())
        for w, g in zip(want, got):
            np.testing.assert_allclose(
                np.asarray(g), np.asarray(w), rtol=ULP_RTOL, atol=ULP_ATOL
            )

    def test_apply_is_unchanged(self, rule, batch_size):
        """And so does the value-only path the adjoints use."""
        n = len(rule()._xh)
        want = rule()._apply(exp_neg, 0.0, 1.0, ())
        got = rule(batch_size=batch_size(n))._apply(exp_neg, 0.0, 1.0, ())
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=ULP_RTOL, atol=ULP_ATOL
        )

    def test_vector_valued(self, rule, batch_size):
        """Batching slices the leading axis, so it must survive extra trailing ones."""
        fun = lambda x: jnp.array([jnp.sin(x), jnp.cos(x), x**2])  # noqa: E731
        n = len(rule()._xh)
        want = rule().integrate(fun, 0.0, 1.0, ())
        got = rule(batch_size=batch_size(n)).integrate(fun, 0.0, 1.0, ())
        for w, g in zip(want, got):
            np.testing.assert_allclose(
                np.asarray(g), np.asarray(w), rtol=ULP_RTOL, atol=ULP_ATOL
            )

    def test_nodes_per_call(self, rule, batch_size):
        """Batching regroups the nodes, it does not add any.

        The remainder is evaluated in one smaller batch rather than padded up, so no
        batch size costs an evaluation the unbatched rule would not have made.
        """
        n = len(rule()._xh)
        assert rule(batch_size=batch_size(n)).nodes_per_call == n


@pytest.mark.usefixtures("quiet_tanhsinh")
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("batch_size", [0, -1, 2.5], ids=["zero", "negative", "float"])
def test_bad_batch_size_rejected(rule, batch_size):
    """A batch size that is not a positive integer is a mistake, not a default."""
    with pytest.raises(ValueError, match="batch_size"):
        rule(batch_size=batch_size)
