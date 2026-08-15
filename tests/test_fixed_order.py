"""Tests for the fixed order quadrature rules in quadax/fixed_order.py.

Each rule has a defining property, and these tests pin it down:

- ``GaussKronrodRule`` is a Gauss-Kronrod rule: an ``order`` (odd) point rule embedding
  an ``order // 2`` point Gauss rule, exactly integrating all polynomials up to degree
  ``3 * (order // 2) + 1`` (22, 31, 46, 61, 76, 91 for the six QUADPACK orders, i.e.
  ``~3/2 the order``).
- ``ClenshawCurtisRule`` exactly integrates all polynomials up to degree ``order``.
- ``TanhSinhRule`` is a doubly-exponential trapezoidal rule: for analytic integrands
  each doubling of the order squares the error once inside the asymptotic tail, which is
  far faster than any algebraic convergence rate.

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
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
from jax import config

from quadax import ClenshawCurtisRule, GaussKronrodRule, TanhSinhRule

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
# it; from 17 on the error is required to square per doubling until it bottoms out in
# rounding.
TANHSINH_ORDERS = [9, 17, 33, 65, 129]
# Up to this error the convolved factors of ~eps make the observed error, and so its
# behavior under doubling, dominated by rounding rather than by the quadrature error.
ROUNDOFF_FLOOR = 2e-14


@pytest.mark.parametrize("fun, exact", ANALYTIC_INTEGRANDS)
@pytest.mark.parametrize("a, b", [(-1.0, 1.0), (0.3, 2.4), (-1.0, 2.0)])
class TestTanhSinhConvergence:
    """For analytic integrands the error decays faster than any algebraic rate."""

    def test_tanhsinh_exponential_convergence(self, fun, exact, a, b):
        """Doubling ``order`` squares the error until rounding sets in."""
        target = exact(a, b)
        errors = [
            abs(float(TanhSinhRule(order=order).integrate(fun, a, b, ())[0]) - target)
            for order in TANHSINH_ORDERS
        ]

        # the error decays monotonically, the pre-asymptotic 9 -> 17 doubling included
        for e0, e1 in zip(errors, errors[1:]):
            if e0 > ROUNDOFF_FLOOR:
                assert e1 <= e0

        # inside the asymptotic tail (from 17 -> 33 on) each doubling squares the error.
        # A rule with algebraic convergence of any order would improve by a fixed factor
        # per point doubling instead, so this separates the doubly-exponential tanh-sinh
        # behavior from polynomial-rate methods.
        for e0, e1 in zip(errors[1:], errors[2:]):
            if e1 > ROUNDOFF_FLOOR:
                assert e1 <= e0**2

        # the sweep ends in rounding noise rather than a stalled algebraic tail
        assert errors[-1] < ROUNDOFF_FLOOR
