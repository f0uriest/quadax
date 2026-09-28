"""Tests for the fixed order cubature rules in quadax/cubature.py.

Each rule has a defining property, and these tests pin it down:

- ``GenzMalikRule`` is a fully symmetric rule of a given polynomial degree: it exactly
  integrates every monomial of *total* degree at most ``degree``, and its embedded rule
  every monomial of total degree at most ``degree - 2``. Both are exact on any monomial
  with an odd exponent whatever their degree, the node set being symmetric, so only the
  even ones say anything.
- ``TensorProductRule`` integrates each axis exactly as the 1D rule given for that axis
  would, so it is exact on a product of one dimensional polynomials whenever every
  factor is within its own axis' degree, and the axes are independent of one another.

Exactness is checked on products of Chebyshev polynomials rather than on monomials, for
the reason the 1D tests give: ``{T_0, ..., T_D}`` spans the polynomials of degree at
most ``D``, so checking every ``T_k`` is equivalent, but Chebyshev polynomials stay
bounded and well conditioned where monomials do not.

The weights are solved for from the moment equations at build time rather than being
transcribed from a table, so the tests below check the assembled rule against
independently known quantities: the exact integral of a polynomial, the volume of the
box, and the degree at which exactness is expected to stop.
"""

import itertools
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quadax import ClenshawCurtisRule, GaussKronrodRule, TanhSinhRule, quadgk
from quadax._fixed_cubature import (
    AbstractCubatureRule,
    GenzMalikRule,
    TensorProductRule,
)
from quadax._quad_weights import (
    _FS_DELTA2,
    _fs_generators,
    _fs_orbit_squares,
    _fs_partitions,
    _orbit_weights,
    get_genz_malik_table,
)
from quadax._utils import box_corners, map_box

from .problems import ULP_ATOL, ULP_RTOL, real_dtypes

# Dimensions worth sweeping: two is the smallest a cubature rule is defined for and the
# only one where the corner orbit is not the dominant cost, three is the common case,
# and five is far enough out for the rule's negative weights to have grown large.
NDIMS = [2, 3, 5]
DEGREES = [7, 9, 11, 13]
# Tolerance for the exactness checks, by rule degree. The weights themselves are
# exact; what the tolerance covers is summing the rule in float64, over more nodes and
# with larger weights of both signs as the degree and dimension grow. A wrong rule
# misses by order one, as the tests just past the claimed degree show.
EXACTNESS_TOL = {7: 1e-12, 9: 1e-12, 11: 1e-11, 13: 1e-11}

# Boxes are deliberately not centred on the origin, so that the even/odd symmetry of the
# node tables (which would integrate odd integrands to zero for free) cannot hide
# anything, and not cubes, so that a Jacobian formed per axis is distinguishable from
# one formed as a single power.
BOXES = {
    "unit": (np.array([0.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])),
    "offset": (
        np.array([-1.0, 0.5, -2.0, 0.25, -0.5]),
        np.array([2.0, 1.5, 1.0, 1.75, 0.5]),
    ),
}


def box(name, ndim, dtype=jnp.float64):
    """The first ``ndim`` axes of one of ``BOXES``, as arrays of ``dtype``."""
    a, b = BOXES[name]
    return jnp.asarray(a[:ndim], dtype), jnp.asarray(b[:ndim], dtype)


def cheb_product(x, degrees, a, b):
    """Product of Chebyshev polynomials, one per axis, with ``[a, b]`` mapped to a cube.

    ``degrees`` and the box are taken as arguments rather than closed over so that they
    reach the rule as traced, dynamic ``args``: only the integrand function is a static
    JIT key, so a whole degree sweep compiles once.
    """
    t = (2 * x - (a + b)) / (b - a)
    return jnp.prod(jnp.cos(degrees * jnp.arccos(t)))


def cheb_product_args(a, b, degrees):
    """The ``args`` tuple ``(degrees, a, b)`` for `cheb_product`, as arrays."""
    return (jnp.asarray(degrees), jnp.asarray(a), jnp.asarray(b))


def cheb_product_integral(a, b, degrees):
    """Exact integral of `cheb_product` over the box, which factorizes over the axes."""
    out = 1.0
    for k, ai, bi in zip(degrees, np.asarray(a), np.asarray(b)):
        if k % 2:
            return 0.0
        out *= (bi - ai) / 2 * 2 / (1 - k**2)
    return out


def integrate_cheb(rule, a, b, degrees):
    """Value the rule gives for the Chebyshev product of the given per axis degrees."""
    return float(
        rule.integrate(cheb_product, a, b, cheb_product_args(a, b, degrees))[0]
    )


def exact_orbits(ndim, degree, bend=Fraction(1)):
    """The rule's orbits as the squared coordinates the moment equations act on.

    ``bend`` scales one generator away from where it belongs, so that a test can check
    the residual notices.
    """
    m = (degree - 1) // 2
    delta2 = _FS_DELTA2[degree]
    squares = [Fraction(0)] + _fs_generators(m - 1, delta2)
    squares[1] *= bend
    orbits = [
        _fs_orbit_squares([squares[j] for j in p], ndim)
        for p in _fs_partitions(m - 1, ndim)
    ]
    orbits.append(_fs_orbit_squares([delta2] * ndim, ndim))
    return orbits


def mixed_poly(total, ndim):
    """Exponents of a monomial of the given total degree, spread over two axes.

    A fully symmetric rule can integrate a pure power of one coordinate further than
    its nominal degree, the axis orbits carrying moments the mixed terms never see, so
    a poly that exactness has stopped has to put weight on more than one axis.
    """
    degrees = [0] * ndim
    degrees[0], degrees[1] = 4, total - 4
    return degrees


def compositions(total, ndim):
    """Every way of splitting ``total`` across ``ndim`` axes as exponents."""
    return [
        c for c in itertools.product(range(total + 1), repeat=ndim) if sum(c) == total
    ]


class TestGenzMalikExactness:
    """Genz-Malik is exact to its total degree and no further."""

    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("boxname", list(BOXES))
    @pytest.mark.parametrize("degree", DEGREES)
    def test_exact_up_to_its_degree(self, ndim, boxname, degree):
        """Every monomial of total degree at most ``degree`` is integrated exactly."""
        a, b = box(boxname, ndim)
        rule = GenzMalikRule(ndim, degree)
        for total in range(degree + 1):
            for degrees in compositions(total, ndim):
                got = integrate_cheb(rule, a, b, degrees)
                np.testing.assert_allclose(
                    got,
                    cheb_product_integral(a, b, degrees),
                    rtol=EXACTNESS_TOL[degree],
                    atol=EXACTNESS_TOL[degree],
                    err_msg=f"degrees {degrees}",
                )

    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("degree", DEGREES)
    def test_not_exact_one_degree_higher(self, ndim, degree):
        """Exactness stops one degree past the rule's own.

        The polynomial spreads the excess degree over two axes rather than putting it
        all in one: the axis orbits happen to carry enough moments for a pure power
        along a single axis, and it is the mixed term that the rule cannot reach.
        """
        a, b = box("unit", ndim)
        degrees = mixed_poly(degree + 1, ndim)
        exact = cheb_product_integral(a, b, degrees)
        got = integrate_cheb(GenzMalikRule(ndim, degree), a, b, degrees)
        assert abs(got - exact) > 0.01 * abs(exact)

    @pytest.mark.parametrize("degree", DEGREES)
    def test_one_axis_is_exact_to_its_degree(self, degree):
        """Over a single axis the rule and its embedded rule reach their degrees.

        Swept on its own rather than through ``NDIMS`` because a single axis is where
        the family degenerates: there is one orbit per generator and one moment
        equation per even power, so the system the weights come from is square and the
        residual the other dimensions are checked against is satisfied whatever the
        generators are. Exactness is then the only thing left that can be measured, and
        a pure power is the only polynomial there is to measure it with.
        """
        a, b = box("offset", 1)
        rule = GenzMalikRule(1, degree)
        table = get_genz_malik_table(1, degree)
        xh, wl = table.x, table.wl
        tol = EXACTNESS_TOL[degree]
        cheb = lambda k: np.cos(k * np.arccos(xh[:, 0]))
        for total in range(degree + 1):
            np.testing.assert_allclose(
                integrate_cheb(rule, a, b, [total]),
                cheb_product_integral(a, b, [total]),
                rtol=tol,
                atol=tol,
                err_msg=f"degree {total}",
            )
        for total in range(degree - 1):
            np.testing.assert_allclose(
                wl @ cheb(total),
                cheb_product_integral([-1.0], [1.0], [total]),
                rtol=tol,
                atol=tol,
                err_msg=f"embedded, degree {total}",
            )
        # and each stops where it is supposed to, one degree past its own
        exact = cheb_product_integral(a, b, [degree + 1])
        assert abs(integrate_cheb(rule, a, b, [degree + 1]) - exact) > 0.01 * abs(exact)
        exact = cheb_product_integral([-1.0], [1.0], [degree - 1])
        assert abs(wl @ cheb(degree - 1) - exact) > 0.01 * abs(exact)

    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("degree", DEGREES)
    def test_embedded_rule_is_two_degrees_lower(self, ndim, degree):
        """The embedded weights are exact two degrees lower, and fail one past that."""
        table = get_genz_malik_table(ndim, degree)
        xh, wl = table.x, table.wl
        for total in range(degree - 1):
            for degrees in compositions(total, ndim):
                v = np.prod(np.cos(np.asarray(degrees) * np.arccos(xh)), axis=1)
                np.testing.assert_allclose(
                    wl @ v,
                    cheb_product_integral(-np.ones(ndim), np.ones(ndim), degrees),
                    rtol=EXACTNESS_TOL[degree],
                    atol=EXACTNESS_TOL[degree],
                    err_msg=f"degrees {degrees}",
                )
        degrees = mixed_poly(degree - 1, ndim)
        v = np.prod(np.cos(np.asarray(degrees) * np.arccos(xh)), axis=1)
        exact = cheb_product_integral(-np.ones(ndim), np.ones(ndim), degrees)
        assert abs(wl @ v - exact) > 0.01 * abs(exact)


class TestTablesAreWellFormed:
    """Properties the assembled tables must have however the weights were solved for."""

    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("degree", DEGREES)
    def test_moment_residual_is_exactly_zero(self, ndim, degree):
        """The generators satisfy the moment equations they are supposed to.

        This is what stands behind the derived weights. The system is overdetermined
        and solved in exact arithmetic, so the leftover equations come out satisfied
        exactly for the right generators and not at all for any others.
        """
        orbits = exact_orbits(ndim, degree)
        assert _orbit_weights(orbits, ndim, degree)[1] == 0
        assert _orbit_weights(orbits[:-1], ndim, degree - 2)[1] == 0
        xh = get_genz_malik_table(ndim, degree).x
        # every node of the rule, plus the axis orbit only the null rules read
        assert sum(sum(o.values()) for o in orbits) + 2 * ndim == len(xh)
        if degree == 7:
            assert len(xh) == 2**ndim + 2 * ndim**2 + 4 * ndim + 1

    @pytest.mark.parametrize("degree", DEGREES)
    def test_every_node_lies_inside_the_cube(self, degree):
        """What the choice of the family's free parameter has to buy.

        A generator past one puts nodes outside the region being integrated, where the
        integrand need not be defined at all, so each degree takes the parameter that
        keeps its own generators in range. Degree 7 additionally comes out as the
        classical rule, whose generators are the simple rationals pinned here.
        """
        for ndim in NDIMS:
            xh = get_genz_malik_table(ndim, degree).x
            assert np.max(np.abs(xh)) < 1.0
        squares = _fs_generators((degree - 1) // 2 - 1, _FS_DELTA2[degree])
        if degree == 7:
            assert squares == [Fraction(9, 10), Fraction(9, 70)]
            assert _FS_DELTA2[degree] == Fraction(9, 19)

    @pytest.mark.parametrize("degree", DEGREES)
    def test_a_generator_off_its_value_is_caught(self, degree):
        """The residual is what makes the table self checking, so it has to bite."""
        orbits = exact_orbits(3, degree, bend=Fraction(1000001, 1000000))
        assert _orbit_weights(orbits, 3, degree)[1] != 0

    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("degree", DEGREES)
    def test_genz_malik_weights_sum_to_the_volume(self, ndim, degree):
        """Both rules integrate a constant, ie the volume of the reference cube."""
        table = get_genz_malik_table(ndim, degree)
        wh, wl, wsplit = table.wh, table.wl, table.wsplit
        np.testing.assert_allclose(wh.sum(), 2**ndim, rtol=1e-12)
        np.testing.assert_allclose(wl.sum(), 2**ndim, rtol=1e-12)
        # A fourth difference must be blind to a constant, or a flat integrand would
        # look like the hardest thing to integrate there is.
        np.testing.assert_allclose(wsplit.sum(axis=1), 0.0, atol=1e-14)

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_tensor_product_weights_sum_to_the_volume(self, ndim):
        """The product of 1D rules integrating a constant integrates one too."""
        rule = TensorProductRule(GaussKronrodRule(15), ndim=ndim)
        _, wh, wl, wsplit = rule._nodes_weights(jnp.float64)
        np.testing.assert_allclose(float(wh.sum()), 2**ndim, rtol=1e-12)
        np.testing.assert_allclose(float(wl.sum()), 2**ndim, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(wsplit.sum(axis=1)), 0.0, atol=1e-13)


class TestTensorProductExactness:
    """A tensor product is exact per axis, and the axes do not interfere."""

    @pytest.mark.parametrize("boxname", list(BOXES))
    def test_exact_up_to_each_axis_degree(self, boxname):
        """Gauss-Kronrod of order 15 is exact to degree 22 along every axis.

        Checked at the corner of the region of exactness, where every axis is at its
        own limit at once, which is the case a rule formed by a single flattened
        contraction could get wrong while getting one axis at a time right.
        """
        a, b = box(boxname, 2)
        rule = TensorProductRule(GaussKronrodRule(15), ndim=2)
        for degrees in [(0, 0), (22, 0), (0, 22), (22, 22), (10, 12), (21, 22)]:
            np.testing.assert_allclose(
                integrate_cheb(rule, a, b, degrees),
                cheb_product_integral(a, b, degrees),
                rtol=1e-11,
                atol=1e-11,
                err_msg=f"degrees {degrees}",
            )

    def test_not_exact_past_an_axis_degree(self):
        """One axis over its own limit is enough to lose exactness."""
        a, b = box("unit", 2)
        rule = TensorProductRule(GaussKronrodRule(15), ndim=2)
        degrees = (24, 0)
        exact = cheb_product_integral(a, b, degrees)
        assert abs(integrate_cheb(rule, a, b, degrees) - exact) > 1e-9 * abs(exact)

    def test_axes_may_carry_different_rules(self):
        """Each axis is exact to its own rule's degree, not the weakest one's.

        Gauss-Kronrod of order 15 reaches degree 22 and Clenshaw-Curtis of order 32
        reaches 32, so the product is exact at (22, 32) and at neither (24, 32) nor
        (22, 34).
        """
        a, b = box("offset", 2)
        rule = TensorProductRule([GaussKronrodRule(15), ClenshawCurtisRule(32)])
        assert rule.ndim == 2
        assert rule.nodes_per_call == 15 * 33
        np.testing.assert_allclose(
            integrate_cheb(rule, a, b, (22, 32)),
            cheb_product_integral(a, b, (22, 32)),
            rtol=1e-10,
            atol=1e-10,
        )
        for degrees in [(24, 32), (22, 34)]:
            exact = cheb_product_integral(a, b, degrees)
            got = integrate_cheb(rule, a, b, degrees)
            assert abs(got - exact) > 1e-9 * abs(exact), f"degrees {degrees}"


# Separable integrands, so that the exact value in ``ndim`` dimensions is the one
# dimensional one raised to the power ``ndim``. Roughly in order of how hard they make
# the error estimate's job: analytic, then peaked, then oscillatory, then singular in
# the derivative at a corner of the box.
FACTORS = {
    "exp": (lambda t: jnp.exp(t), np.e - 1.0),
    "cubic": (lambda t: t**3 - 2 * t + 1, 0.25 - 1.0 + 1.0),
    "peak": (lambda t: 1 / (1 + 100 * (t - 0.5) ** 2), 0.2 * np.arctan(5.0)),
    "osc": (lambda t: jnp.cos(8 * t), np.sin(8.0) / 8),
    "sqrt": (jnp.sqrt, 2 / 3),
}


def product_case(name, ndim):
    """``(fun, exact)`` for the separable integrand ``name`` over the unit box."""
    factor, value = FACTORS[name]
    return lambda x: jnp.prod(jax.vmap(factor)(x)), value**ndim


# Cases where the reported error is known to fall below the true one, as
# ``(rule, case, ndim)``. Kept as a table rather than a loosened tolerance so that each
# entry is a statement about one integrand, and so that the list shrinking is visible.
KNOWN_DISHONEST: set = set()


def xfail_if_dishonest(request, rule, case, ndim):
    """Mark the running test xfail if it is a known dishonest combination."""
    if (rule, case, ndim) in KNOWN_DISHONEST:
        request.node.add_marker(
            pytest.mark.xfail(
                reason=f"{rule} understates its error on {case} at ndim={ndim}",
                strict=True,
            )
        )


def rules_for(ndim, norm=jnp.inf, batch_size=None):
    """The rule families to sweep at this dimension, keyed by a short name.

    The tensor product's cost is the product over the axes, so the axis rule has to get
    cheaper as the dimension grows: Gauss-Kronrod of order 15 is 759375 evaluations in
    five dimensions, where the 9 node Clenshaw-Curtis rule is 59049.
    """
    axis = GaussKronrodRule(15) if ndim < 4 else ClenshawCurtisRule(8)
    return {
        "gm7": GenzMalikRule(ndim, 7, norm=norm, batch_size=batch_size),
        "gm9": GenzMalikRule(ndim, 9, norm=norm, batch_size=batch_size),
        "gm11": GenzMalikRule(ndim, 11, norm=norm, batch_size=batch_size),
        "gm13": GenzMalikRule(ndim, 13, norm=norm, batch_size=batch_size),
        "tp": TensorProductRule(axis, ndim=ndim, norm=norm, batch_size=batch_size),
    }


class TestErrorEstimates:
    """What the reported error is allowed to be."""

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("rulename", ["gm7", "gm9", "gm11", "gm13", "tp"])
    @pytest.mark.parametrize("case", list(FACTORS))
    @pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
    def test_error_estimate_is_honest(self, request, ndim, rulename, case, dtype):
        """The reported error is at least the true one, with no margin allowed."""
        xfail_if_dishonest(request, rulename, case, ndim)
        fun, exact = product_case(case, ndim)
        a, b = box("unit", ndim, dtype)
        y, err, _, _, _ = rules_for(ndim)[rulename].integrate(fun, a, b, ())
        true_err = abs(complex(y) - exact)
        assert true_err <= float(err), (
            f"{case} at {dtype.__name__}: reported {float(err):.3e} "
            f"< true {true_err:.3e}"
        )

    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    def test_a_zero_integrand_reports_no_error(self, rulename):
        """An integrand that vanishes everywhere is integrated exactly.

        Worth pinning because the estimate is a chain of ratios of sampled quantities,
        every one of which is degenerate here, and any of them resolving to a nan
        rather than being substituted away would surface as an error estimate on an
        integral that is exactly right.
        """
        a, b = box("unit", 2)
        rule = rules_for(2)[rulename]
        _, err, y_abs, y_mmn, split = rule.integrate(
            lambda x: 0.0 * jnp.sum(x), a, b, ()
        )
        for v in (err, y_abs, y_mmn, split):
            np.testing.assert_array_equal(np.asarray(v), 0.0)

    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    def test_a_vector_integrand_is_charged_its_worst_component(self, rulename):
        """``err`` for a vector integrand is the norm over the component errors.

        The default norm is the max, so integrating three integrands together has to
        report exactly what the worst of them reports on its own. Any part of the
        estimate that reduced over the components too early, rather than being formed
        component by component and normed once at the end, would come out below this.
        """
        a, b = box("unit", 2)
        rule = rules_for(2)[rulename]
        funs = [product_case(n, 2)[0] for n in ("exp", "sqrt", "osc")]
        together = lambda x: jnp.stack([f(x) for f in funs], axis=-1)
        combined = float(rule.integrate(together, a, b, ())[1])
        separate = [float(rule.integrate(f, a, b, ())[1]) for f in funs]
        # Looser than the ULP tolerances the values themselves are held to: the error
        # estimate is a power of a ratio of the sums rather than a sum, so the few ULP
        # that contracting a stacked array reassociates come out amplified here.
        np.testing.assert_allclose(combined, max(separate), rtol=1e-10, atol=ULP_ATOL)


class TestSplitIndicator:
    """``split`` says which axis is worth cutting."""

    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("rulename", ["gm7", "gm9", "gm11", "gm13", "tp"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_it_finds_the_difficult_axis(self, ndim, rulename, axis):
        """An integrand varying sharply along one axis is charged to that axis."""
        fun = lambda x: jnp.exp(-200.0 * (x[axis] - 0.5) ** 2) + 0.01 * jnp.sum(x)
        a, b = box("unit", ndim)
        *_, split = rules_for(ndim)[rulename].integrate(fun, a, b, ())
        assert split.shape == (ndim,)
        assert int(jnp.argmax(split)) == axis

    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    def test_a_separable_integrand_charges_its_axes_equally(self, rulename):
        """Symmetry in the integrand has to survive into the indicator."""
        a, b = box("unit", 3)
        fun, _ = product_case("peak", 3)
        *_, split = rules_for(3)[rulename].integrate(fun, a, b, ())
        np.testing.assert_allclose(
            np.asarray(split), float(split[0]), rtol=1e-10, atol=1e-14
        )


class TestCubatureRuleDTypes:
    """The corners state the precision, and everything downstream follows them."""

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_integrate(self, ndim, rulename, dtype):
        """Every output comes back at the dtype the corners asked for."""
        a, b = box("unit", ndim, dtype)
        fun, _ = product_case("exp", ndim)
        y, err, y_abs, y_mmn, split = rules_for(ndim)[rulename].integrate(fun, a, b, ())
        for v in (y, err, y_abs, y_mmn, split):
            assert jnp.asarray(v).dtype == dtype

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_degenerate_box(self, ndim, rulename, dtype):
        """A box with no extent in one axis integrates to zero, shaped and typed."""
        a, b = box("unit", ndim, dtype)
        b = b.at[0].set(a[0])
        fun, _ = product_case("exp", ndim)
        out = rules_for(ndim)[rulename].integrate(fun, a, b, ())
        for v in out:
            assert jnp.asarray(v).dtype == dtype
            np.testing.assert_array_equal(np.asarray(v), 0.0)
        assert out[-1].shape == (ndim,)

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_apply_matches_integrate(self, ndim, rulename, dtype):
        """``_apply`` is the value ``integrate`` returns, by a cheaper route."""
        a, b = box("offset", ndim, dtype)
        fun, _ = product_case("exp", ndim)
        rule = rules_for(ndim)[rulename]
        np.testing.assert_allclose(
            np.asarray(rule._apply(fun, a, b, ()), dtype=np.float64),
            np.asarray(rule.integrate(fun, a, b, ())[0], dtype=np.float64),
            rtol=float(jnp.finfo(a.dtype).eps) * 50,
            atol=float(jnp.finfo(a.dtype).eps) * 50,
        )

    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    @pytest.mark.parametrize("nflip", [1, 2])
    def test_reversing_an_axis_flips_the_sign(self, rulename, nflip):
        """Each axis given the other way round contributes a factor of -1.

        The Jacobian is a product over the axes, so reversing two of them in two
        dimensions leaves the value alone, and only an odd number of flips negates it.
        """
        a, b = box("offset", 2)
        fun, _ = product_case("exp", 2)
        rule = rules_for(2)[rulename]
        forward = float(rule.integrate(fun, a, b, ())[0])
        flipped_a = a.at[:nflip].set(b[:nflip])
        flipped_b = b.at[:nflip].set(a[:nflip])
        got = float(rule.integrate(fun, flipped_a, flipped_b, ())[0])
        np.testing.assert_allclose(
            got, (-1) ** nflip * forward, rtol=ULP_RTOL, atol=ULP_ATOL
        )

    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    def test_a_complex_integrand_promotes_on_its_own(self, rulename):
        """The value goes complex while the error and the split stay real.

        The weights are cast to the real counterpart of the accumulation dtype rather
        than to the accumulation dtype itself, which is what lets the integrand carry
        the promotion instead of the real valued corners forcing it back.
        """
        a, b = box("unit", 2)
        fun = lambda x: jnp.prod(jnp.exp(2j * x))
        exact = ((np.exp(2j) - 1) / 2j) ** 2
        y, err, y_abs, y_mmn, split = rules_for(2)[rulename].integrate(fun, a, b, ())
        assert y.dtype == jnp.complex128
        for v in (err, y_abs, y_mmn, split):
            assert jnp.asarray(v).dtype == jnp.float64
        assert abs(complex(y) - exact) <= float(err)

    def test_a_tanhsinh_axis_rebuilds_its_table_at_the_working_precision(self):
        """The product asks each axis rule for its table, rather than casting one.

        ``TanhSinhRule``'s abscissae depend on the precision rather than merely being
        rounded to it, so a product that baked its grid at construction time would hand
        a float32 solve the float64 rule's nodes.
        """
        rule = TensorProductRule(TanhSinhRule(41), ndim=2)
        x64 = rule._nodes_weights(jnp.float64)[0]
        x32 = rule._nodes_weights(jnp.float32)[0]
        assert x32.dtype == jnp.float32
        assert not np.allclose(np.asarray(x64), np.asarray(x32, dtype=np.float64))


class TestBatchSize:
    """Grouping the evaluations differently cannot change the answer."""

    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    @pytest.mark.parametrize("batch_size", [1, 7, 16, 17, 512])
    def test_integrate_is_unchanged(self, rulename, batch_size):
        a, b = box("offset", 3)
        fun, _ = product_case("exp", 3)
        ref = rules_for(3)[rulename].integrate(fun, a, b, ())
        got = rules_for(3, batch_size=batch_size)[rulename].integrate(fun, a, b, ())
        for r, g in zip(ref, got):
            np.testing.assert_allclose(
                np.asarray(g), np.asarray(r), rtol=ULP_RTOL, atol=ULP_ATOL
            )

    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    def test_nodes_per_call_ignores_batching(self, rulename):
        """Batching changes how the nodes are grouped, never how many there are."""
        assert (
            rules_for(3, batch_size=4)[rulename].nodes_per_call
            == rules_for(3)[rulename].nodes_per_call
        )

    def test_genz_malik_node_count(self):
        # The higher degrees have no such closed form, their orbit lists depending on
        # how the degree partitions across the axes, so they are pinned at the
        # dimensions swept here. Every count includes the 2*ndim nodes of the orbit
        # the error estimate adds on top of the rule's own.
        counts = {
            9: {2: 33, 3: 77, 5: 273},
            11: {2: 49, 3: 143, 5: 723},
            13: {2: 69, 3: 245, 5: 1725},
        }
        for ndim in NDIMS:
            expected = 2**ndim + 2 * ndim**2 + 4 * ndim + 1
            assert GenzMalikRule(ndim, 7).nodes_per_call == expected
            for degree, expected_n in counts.items():
                assert GenzMalikRule(ndim, degree).nodes_per_call == expected_n[ndim]


class TestVmapAndNorm:
    """The pieces an adaptive layer will need from these rules."""

    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    def test_apply_vmaps_over_boxes(self, rulename):
        """Many boxes at once, which is how a mesh of regions will be evaluated."""
        rule = rules_for(2)[rulename]
        # A polynomial well inside both rules' degree, so that subdividing reproduces
        # the whole exactly and the comparison is about the batching rather than about
        # how accurate the rule happens to be.
        fun = lambda x: x[0] ** 3 * x[1] ** 2 + 2 * x[0] - 1.0
        lo = jnp.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        hi = lo + 0.5
        ys = jax.vmap(lambda a, b: rule._apply(fun, a, b, ()))(lo, hi)
        assert ys.shape == (4,)
        # the four quarters of the unit box tile it, so they sum to the whole
        whole = rule._apply(fun, jnp.zeros(2), jnp.ones(2), ())
        np.testing.assert_allclose(
            float(ys.sum()), float(whole), rtol=1e-12, atol=1e-14
        )

    @pytest.mark.parametrize("rulename", ["gm7", "tp"])
    def test_with_norm_changes_how_error_is_measured(self, rulename):
        """A 1-norm charges the sum of the component errors, the max charges one."""
        a, b = box("unit", 2)
        funs = [product_case(n, 2)[0] for n in ("exp", "sqrt", "osc")]
        together = lambda x: jnp.stack([f(x) for f in funs], axis=-1)
        rule = rules_for(2)[rulename]
        one = rule._with_norm(1)
        assert float(one.integrate(together, a, b, ())[1]) > float(
            rule.integrate(together, a, b, ())[1]
        )

    def test_with_norm_needs_a_norm_to_replace(self):
        """A rule built without one says so rather than failing obscurely."""

        class NoNorm(AbstractCubatureRule):
            @property
            def ndim(self):
                return 2

            def integrate(self, fun, a, b, args):
                raise NotImplementedError

        with pytest.raises(NotImplementedError, match="was not built from a norm"):
            NoNorm()._with_norm(2)


class TestConstruction:
    """What the constructors accept, and what they say when they refuse."""

    def test_genz_malik_needs_at_least_one_dimension(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            GenzMalikRule(0)

    def test_genz_malik_rejects_an_unimplemented_degree(self):
        with pytest.raises(NotImplementedError, match="should be one of"):
            GenzMalikRule(3, degree=15)

    def test_tensor_product_needs_ndim_for_a_single_rule(self):
        with pytest.raises(ValueError, match="ndim is required"):
            TensorProductRule(GaussKronrodRule(15))

    def test_tensor_product_rejects_ndim_alongside_a_sequence(self):
        with pytest.raises(ValueError, match="must not be given"):
            TensorProductRule([GaussKronrodRule(15)] * 2, ndim=2)

    def test_tensor_product_needs_at_least_one_axis(self):
        with pytest.raises(ValueError, match="at least one axis"):
            TensorProductRule([])
        with pytest.raises(ValueError, match="positive integer"):
            TensorProductRule(GaussKronrodRule(15), ndim=0)

    def test_tensor_product_needs_nested_rules(self):
        """A rule without embedded low order weights cannot estimate error.

        The offending axis is named, since with one rule per axis the sequence can be
        long and the message is the only thing saying which entry is wrong.
        """
        with pytest.raises(TypeError, match="axis 1"):
            TensorProductRule([GaussKronrodRule(15), object()])  # type: ignore[list-item]

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "four"])
    def test_bad_batch_size_rejected(self, bad):
        with pytest.raises(ValueError, match="batch_size"):
            GenzMalikRule(2, batch_size=bad)


class TestInfiniteBox:
    """A box with unbounded axes, mapped by ``map_box`` and then integrated.

    The map is checked on its own in ``tests/test_utils.py``; what is under test here is
    that a rule applied afterwards recovers the integral, including when the axes are
    unbounded in different ways.
    """

    @pytest.mark.parametrize(
        "a, b, exact",
        [
            ([0.0, 0.0], [np.inf, np.inf], 1.0),
            ([-np.inf, -np.inf], [0.0, 0.0], 1.0),
            ([0.0, -np.inf], [np.inf, np.inf], np.sqrt(np.pi)),
            ([-np.inf, -np.inf], [np.inf, np.inf], np.pi),
        ],
        ids=["a_inf", "ninf_b", "mixed", "ninf_inf"],
    )
    def test_an_unbounded_box_is_integrated(self, a, b, exact):
        """Gaussian on the infinite axes and a decaying exponential on the rest."""

        def fun(x):
            lo, hi = jnp.asarray(a), jnp.asarray(b)
            semi = jnp.isinf(lo) ^ jnp.isinf(hi)
            return jnp.prod(jnp.where(semi, jnp.exp(-jnp.abs(x)), jnp.exp(-(x**2))))

        rule = TensorProductRule([GaussKronrodRule(61)] * 2)
        limits = jnp.stack([jnp.asarray(a), jnp.asarray(b)], axis=-1)
        fun_t, interval_t = map_box(fun, limits)
        y, err, _, _, _ = rule.integrate(fun_t, *box_corners(interval_t), ())
        np.testing.assert_allclose(float(y), exact, rtol=1e-8, atol=0)
        assert float(err) >= abs(float(y) - exact)

    def test_a_separable_integral_factorizes(self):
        """The two dimensional answer must be the one dimensional one squared.

        The Jacobian of the box map is a product over axes, so an error in it shows up
        as a mismatch with the same integrand done one axis at a time.
        """
        fun_1d = lambda x: jnp.exp(-x) / (1 + x**2)  # noqa: E731
        y_1d = quadgk(fun_1d, jnp.array([0.0, jnp.inf]))[0]
        fun = lambda x: jnp.prod(jnp.exp(-x) / (1 + x**2))  # noqa: E731
        fun_t, interval_t = map_box(fun, jnp.array([[0.0, np.inf], [0.0, np.inf]]))
        y = TensorProductRule([GaussKronrodRule(61)] * 2).integrate(
            fun_t, *box_corners(interval_t), ()
        )
        np.testing.assert_allclose(float(y[0]), float(y_1d) ** 2, rtol=1e-10, atol=0)

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_genz_malik_over_an_unbounded_box(self, ndim):
        """The fully symmetric rules go through the same map as the tensor product.

        The map concentrates a Gaussian near the centre of the reference box, which a
        single application of a fixed rule resolves only to of order ten percent. So
        what is asserted is that the answer is the right one to that accuracy and that
        the error estimate covers the distance to it, not a tight tolerance a fixed
        rule has no way to meet. The bound is a sanity check on the composition of a
        rule with a map and says nothing about either alone: how widely a map spreads
        this particular integrand across the reference box moves it by several times
        over, while subdividing removes the whole effect.
        """
        fun = lambda x: jnp.exp(-jnp.sum(x**2))  # noqa: E731
        limits = jnp.tile(jnp.array([-jnp.inf, jnp.inf]), (ndim, 1))
        fun_t, interval_t = map_box(fun, limits)
        y, err, *_ = GenzMalikRule(ndim, degree=13).integrate(
            fun_t, *box_corners(interval_t), ()
        )
        exact = np.pi ** (ndim / 2)
        np.testing.assert_allclose(float(y), exact, rtol=1.5e-1, atol=0)
        assert float(err) >= abs(float(y) - exact)


class TestFaceRules:
    """Building a rule of one fewer axis, which is what a face is integrated on."""

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("name", ["genz-malik", "tensor-gk"])
    def test_it_integrates_the_face_exactly(self, name, ndim):
        """The dropped rule is a real rule over the remaining axes, not a stand-in.

        Checked by integrating a Chebyshev product the family is exact for, over the
        box with one axis removed, so a rule that came back with the wrong node set or
        the wrong weights could not pass.
        """
        rules = {
            "genz-malik": GenzMalikRule(ndim, 7),
            "tensor-gk": TensorProductRule(GaussKronrodRule(15), ndim=ndim),
        }
        face = rules[name]._drop_axis(0)
        assert face.ndim == ndim - 1
        a, b = box("unit", ndim - 1)
        degrees = (2,) * (ndim - 1)
        np.testing.assert_allclose(
            integrate_cheb(face, a, b, degrees),
            cheb_product_integral(a, b, degrees),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_a_tensor_product_drops_the_axis_it_was_asked_for(self):
        """Which axis is dropped matters as soon as the axis rules are not all alike."""
        rules = [GaussKronrodRule(15), ClenshawCurtisRule(32), TanhSinhRule(41)]
        product = TensorProductRule(rules)
        for k in range(3):
            kept = [type(r).__name__ for r in product._drop_axis(k)._rules]
            assert kept == [type(r).__name__ for j, r in enumerate(rules) if j != k]

    def test_a_rule_that_cannot_build_one_says_so(self):
        """The base class refuses rather than guessing, and names the way round it."""

        class _Plain(AbstractCubatureRule):
            @property
            def ndim(self):
                return 2

            def integrate(self, fun, a, b, args):
                raise NotImplementedError

        with pytest.raises(NotImplementedError, match="options_face"):
            _Plain()._drop_axis(0)
        with pytest.raises(NotImplementedError, match="no axis left"):
            TensorProductRule(GaussKronrodRule(15), ndim=1)._drop_axis(0)
