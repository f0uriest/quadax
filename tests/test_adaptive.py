"""Tests for adaptive quadrature routines."""

import os
import subprocess
import sys
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy
from jax import config

import quadax
from quadax import (
    ClenshawCurtisRule,
    GaussKronrodRule,
    TanhSinhRule,
    adaptive_quadrature,
    quadcc,
    quadgk,
    quadts,
    romberg,
    rombergts,
)

config.update("jax_enable_x64", True)

example_problems = [
    # problem 0
    {"fun": lambda t: t * jnp.log(1 + t), "interval": [0, 1], "val": 1 / 4},
    # problem 1
    {
        "fun": lambda t: t**2 * jnp.arctan(t),
        "interval": [0, 1],
        "val": (jnp.pi - 2 + 2 * jnp.log(2)) / 12,
    },
    # problem 2
    {
        "fun": lambda t: jnp.exp(t) * jnp.cos(t),
        "interval": [0, jnp.pi / 2],
        "val": (jnp.exp(jnp.pi / 2) - 1) / 2,
    },
    # problem 3
    {
        "fun": lambda t: (
            jnp.arctan(jnp.sqrt(2 + t**2)) / ((1 + t**2) * jnp.sqrt(2 + t**2))
        ),
        "interval": [0, 1],
        "val": 5 * jnp.pi**2 / 96,
    },
    # problem 4
    {"fun": lambda t: jnp.sqrt(t) * jnp.log(t), "interval": [0, 1], "val": -4 / 9},
    # problem 5
    {"fun": lambda t: jnp.sqrt(1 - t**2), "interval": [0, 1], "val": jnp.pi / 4},
    # problem 6
    {
        "fun": lambda t: jnp.sqrt(t) / jnp.sqrt(1 - t**2),
        "interval": [0, 1],
        "val": 2
        * jnp.sqrt(jnp.pi)
        * scipy.special.gamma(3 / 4)
        / scipy.special.gamma(1 / 4),
    },
    # problem 7
    {"fun": lambda t: jnp.log(t) ** 2, "interval": [0, 1], "val": 2},
    # problem 8
    {
        "fun": lambda t: jnp.log(jnp.cos(t)),
        "interval": [0, jnp.pi / 2],
        "val": -jnp.pi * jnp.log(2) / 2,
    },
    # problem 9
    {
        "fun": lambda t: jnp.sqrt(jnp.tan(t)),
        "interval": [0, jnp.pi / 2],
        "val": jnp.pi * jnp.sqrt(2) / 2,
    },
    # problem 10
    {"fun": lambda t: 1 / (1 + t**2), "interval": [0, jnp.inf], "val": jnp.pi / 2},
    # problem 11
    {
        "fun": lambda t: jnp.exp(-t) / jnp.sqrt(t),
        "interval": [0, jnp.inf],
        "val": jnp.sqrt(jnp.pi),
    },
    # problem 12
    {
        "fun": lambda t: jnp.exp(-(t**2) / 2),
        "interval": [-jnp.inf, jnp.inf],
        "val": jnp.sqrt(2 * jnp.pi),
    },
    # problem 13
    {"fun": lambda t: jnp.exp(-t) * jnp.cos(t), "interval": [0, jnp.inf], "val": 1 / 2},
    # problem 14 - vector valued integrand made of up problems 0 and 1
    {
        "fun": lambda t: jnp.array([t * jnp.log(1 + t), t**2 * jnp.arctan(t)]),
        "interval": [0, 1],
        "val": jnp.array([1 / 4, (jnp.pi - 2 + 2 * jnp.log(2)) / 12]),
    },
    # problem 15 - intergral with breakpoints
    {
        "fun": lambda t: jnp.log((t - 1) ** 2),
        "interval": [0, 1, 2],
        "val": -4,
    },
    # problem 16 - complex function
    {
        "fun": lambda t: t * jnp.log(1 + t) * 1j,
        "interval": [0, 1],
        "val": 0.25j,
    },
    # Problems 17-25 are algebraic singularities, the family bisection alone cannot
    # resolve: for `t**-alpha` the mass below a panel of width h is exactly
    # `h**(1-alpha)` of the total, so an integrator that never samples closer than h to
    # the singular point carries that much relative error whatever its local rule does.
    # They span the axes that turn out to matter: how strong the singularity is, which
    # end it sits at, whether there is one or several, and whether the caller marked it.
    #
    # problem 17 - mild endpoint algebraic singularity
    {"fun": lambda t: t**-0.5, "interval": [0, 1], "val": 2.0},
    # problem 18 - strong endpoint algebraic singularity
    {"fun": lambda t: t**-0.9, "interval": [0, 1], "val": 10.0},
    # problem 19 - extreme endpoint singularity, near the divergence at alpha=1
    {"fun": lambda t: t**-0.99, "interval": [0, 1], "val": 100.0},
    # problem 20 - the same strength at the *right* endpoint, which the mapping onto the
    # reference interval treats differently from the left
    {"fun": lambda t: (1 - t) ** -0.9, "interval": [0, 1], "val": 10.0},
    # problem 21 - both endpoints singular, to different strengths. Two decaying modes
    # arriving at different rates, which is the case a single running total cannot
    # separate. Beta(3/4, 1/4) = gamma(3/4)gamma(1/4) = pi/sin(pi/4).
    {
        "fun": lambda t: t**-0.25 * (1 - t) ** -0.75,
        "interval": [0, 1],
        "val": jnp.pi * jnp.sqrt(2),
    },
    # problem 22 - logarithmic times algebraic
    {"fun": lambda t: jnp.log(t) / jnp.sqrt(t), "interval": [0, 1], "val": -4.0},
    # problem 23 - interior singularity, not marked
    {
        "fun": lambda t: jnp.abs(t - 0.3) ** -0.5,
        "interval": [0, 1],
        "val": 2 * (jnp.sqrt(0.3) + jnp.sqrt(0.7)),
    },
    # problem 24 - the same one, marked as a breakpoint so it lands on a panel end
    {
        "fun": lambda t: jnp.abs(t - 0.3) ** -0.5,
        "interval": [0, 0.3, 1],
        "val": 2 * (jnp.sqrt(0.3) + jnp.sqrt(0.7)),
    },
    # problem 25 - two interior singularities of different strengths
    {
        "fun": lambda t: jnp.abs(t - 0.3) ** -0.5 + jnp.abs(t - 0.7) ** -0.25,
        "interval": [0, 1],
        "val": 2 * (jnp.sqrt(0.3) + jnp.sqrt(0.7)) + (0.7**0.75 + 0.3**0.75) / 0.75,
    },
    # Problems 26-32 decay algebraically over an infinite range, which the mapping onto
    # the reference interval turns into an endpoint singularity, so they exercise the
    # same machinery as 17-25 but with the difficulty manufactured by the transform
    # rather than present in the integrand.
    #
    # For [a, inf) the map is `x = a - 1 + 2/(1-t)` with weight `2/(1-t)**2`, so an
    # integrand falling off as `x**-p` becomes `(1-t)**(p-2)`: the induced singularity
    # has strength `alpha = 2 - p`. The doubly infinite `tan` map gives the same law.
    # That makes these an exact counterpart of the finite cases: p = 1.1 induces the
    # same alpha = 0.9 as problem 18, and the pair measures how much of the difficulty
    # is the singularity itself and how much is the coordinate it is expressed in.
    #
    # problem 26 - semi-infinite, induced alpha = 0.5
    {"fun": lambda t: t**-1.5, "interval": [1, jnp.inf], "val": 2.0},
    # problem 27 - semi-infinite, induced alpha = 0.9
    {"fun": lambda t: t**-1.1, "interval": [1, jnp.inf], "val": 10.0},
    # problem 28 - semi-infinite, induced alpha = 0.99, the slowest decay that converges
    {"fun": lambda t: t**-1.01, "interval": [1, jnp.inf], "val": 100.0},
    # problem 29 - the same decay from a finite left end, so the map is exercised
    # without the integrand also being singular at the start of the range
    {"fun": lambda t: (1 + t) ** -1.5, "interval": [0, jnp.inf], "val": 2.0},
    # problem 30 - the mirror image, which uses the other one sided map
    {"fun": lambda t: (1 - t) ** -1.5, "interval": [-jnp.inf, 0], "val": 2.0},
    # problem 31 - logarithmic factor on top of the algebraic decay
    {"fun": lambda t: jnp.log(t) / t**2, "interval": [1, jnp.inf], "val": 1.0},
    # problem 32 - doubly infinite with algebraic decay, so the transform induces a
    # singularity at *both* ends at once
    {
        "fun": lambda t: (1 + t**2) ** -0.75,
        "interval": [-jnp.inf, jnp.inf],
        "val": jnp.sqrt(jnp.pi) * scipy.special.gamma(0.25) / scipy.special.gamma(0.75),
    },
]

# Where extrapolation is off and on are meant to produce the same quantity, they are
# entitled to differ in the last place or two and no further.
ULP_RTOL = 1e-13
ULP_ATOL = 1e-15


class _BothExtrapolationModes:
    """Run every case in the class twice, with extrapolation off and on.

    The two modes are held to the same contract: the same status, and the same accuracy.
    Where a case genuinely behaves differently with acceleration, it says so with an
    ``extrap_status`` or ``extrap_fudge`` override rather than by being excluded.
    """

    @pytest.fixture(params=[False, True], ids=["plain", "extrap"], autouse=True)
    def _extrapolation_mode(self, request):
        self.extrapolate = request.param


class TestQuadGK(_BothExtrapolationModes):
    """Tests for Gauss-Kronrod quadrature."""

    def _base(self, i, tol, fudge=1.0, **kwargs):
        prob = example_problems[i]
        status = kwargs.pop("status", 0)
        # Overrides for cases whose behaviour genuinely differs once the extrapolation
        # is switched on, applied only in that mode.
        extrap_status = kwargs.pop("extrap_status", None)
        extrap_fudge = kwargs.pop("extrap_fudge", None)
        if self.extrapolate:
            status = status if extrap_status is None else extrap_status
            fudge = fudge if extrap_fudge is None else extrap_fudge
        y, info = quadgk(
            prob["fun"],
            prob["interval"],
            epsabs=tol,
            epsrel=tol,
            extrapolate=self.extrapolate,
            **kwargs,
        )
        assert info.status == status
        if status == 0:
            assert info.err < max(tol, tol * np.max(np.abs(y)))
        np.testing.assert_allclose(
            y,
            prob["val"],
            rtol=fudge * tol,
            atol=fudge * tol,
            err_msg=f"problem {i}, tol={tol}",
        )

    def test_prob0(self):
        """Test for example problem #0."""
        self._base(0, 1e-4, order=21)
        self._base(0, 1e-8, order=21)
        self._base(0, 1e-12, order=21)

    def test_prob1(self):
        """Test for example problem #1."""
        self._base(1, 1e-4, order=31)
        self._base(1, 1e-8, order=31)
        self._base(1, 1e-12, order=31)

    def test_prob2(self):
        """Test for example problem #2."""
        self._base(2, 1e-4, order=41)
        self._base(2, 1e-8, order=41)
        self._base(2, 1e-12, order=41)

    def test_prob3(self):
        """Test for example problem #3."""
        self._base(3, 1e-4, order=51)
        self._base(3, 1e-8, order=51)
        self._base(3, 1e-12, order=51)

    def test_prob4(self):
        """Test for example problem #4."""
        self._base(4, 1e-4, order=61)
        self._base(4, 1e-8, order=61)
        self._base(4, 1e-12, order=61)

    def test_prob5(self):
        """Test for example problem #5."""
        self._base(5, 1e-4, order=21)
        self._base(5, 1e-8, order=21)
        self._base(5, 1e-12, order=21)

    def test_prob6(self):
        """Test for example problem #6."""
        self._base(6, 1e-4, order=15)
        # endpoint singularity: order 15 tops out around 1e-8 however much budget it is
        # given, so it exhausts the subdivision limit rather than reaching the tolerance
        self._base(6, 1e-8, 100, order=15, status=2, extrap_status=0)
        self._base(6, 1e-12, 1e5, order=15, max_ninter=100, status=8, extrap_status=0)

    def test_prob7(self):
        """Test for example problem #7."""
        self._base(7, 1e-4, order=61)
        self._base(7, 1e-8, order=61)
        self._base(7, 1e-12, order=61)

    def test_prob8(self):
        """Test for example problem #8."""
        self._base(8, 1e-4, order=51)
        self._base(8, 1e-8, order=51)
        self._base(8, 1e-12, order=51)

    def test_prob9(self):
        """Test for example problem #9."""
        self._base(9, 1e-4, order=15)
        # as for problem 6, order 15 cannot certify 1e-8 on this endpoint singularity
        self._base(9, 1e-8, 100, order=15, status=2, extrap_status=0)
        self._base(9, 1e-12, 1e4, order=15, max_ninter=100, status=8, extrap_status=0)

    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4, order=15)
        self._base(10, 1e-8, order=15)
        self._base(10, 1e-12, order=15)

    def test_prob11(self):
        """Test for example problem #11."""
        self._base(11, 1e-4, order=21)
        # bisection concentrates on the singularity at t=0 until the sub-intervals stop
        # being resolvable, at a true error just above 1e-8
        self._base(11, 1e-8, 100, order=21, status=8, extrap_status=0)
        self._base(11, 1e-12, 1e4, order=21, status=8, max_ninter=100, extrap_status=0)

    def test_prob12(self):
        """Test for example problem #12."""
        self._base(12, 1e-4, order=15)
        self._base(12, 1e-8, order=15)
        self._base(12, 1e-12, order=15)

    def test_prob13(self):
        """Test for example problem #13."""
        self._base(13, 1e-4, order=31)
        self._base(13, 1e-8, order=31)
        self._base(13, 1e-12, order=31)

    def test_prob14(self):
        """Test for example problem #14."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob15(self):
        """Test for example problem #15."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob16(self):
        """Test for example problem #16."""
        self._base(16, 1e-4)
        self._base(16, 1e-8)
        self._base(16, 1e-12)


class TestQuadCC(_BothExtrapolationModes):
    """Tests for Clenshaw-Curtis quadrature."""

    def _base(self, i, tol, fudge=1.0, **kwargs):
        prob = example_problems[i]
        status = kwargs.pop("status", 0)
        # Overrides for cases whose behaviour genuinely differs once the extrapolation
        # is switched on, applied only in that mode.
        extrap_status = kwargs.pop("extrap_status", None)
        extrap_fudge = kwargs.pop("extrap_fudge", None)
        if self.extrapolate:
            status = status if extrap_status is None else extrap_status
            fudge = fudge if extrap_fudge is None else extrap_fudge
        y, info = quadcc(
            prob["fun"],
            prob["interval"],
            epsabs=tol,
            epsrel=tol,
            extrapolate=self.extrapolate,
            **kwargs,
        )
        assert info.status == status
        if status == 0:
            assert info.err < max(tol, tol * np.max(np.abs(y)))
        np.testing.assert_allclose(
            y,
            prob["val"],
            rtol=fudge * tol,
            atol=fudge * tol,
            err_msg=f"problem {i}, tol={tol}",
        )

    def test_prob0(self):
        """Test for example problem #0."""
        self._base(0, 1e-4, order=32)
        self._base(0, 1e-8, order=32)
        self._base(0, 1e-12, order=32)

    def test_prob1(self):
        """Test for example problem #1."""
        self._base(1, 1e-4, order=64)
        self._base(1, 1e-8, order=64)
        self._base(1, 1e-12, order=64)

    def test_prob2(self):
        """Test for example problem #2."""
        self._base(2, 1e-4, order=128)
        self._base(2, 1e-8, order=128)
        self._base(2, 1e-12, order=128)

    def test_prob3(self):
        """Test for example problem #3."""
        self._base(3, 1e-4, order=256)
        self._base(3, 1e-8, order=256)
        self._base(3, 1e-12, order=256)

    def test_prob4(self):
        """Test for example problem #4."""
        self._base(4, 1e-4, order=8)
        self._base(4, 1e-8, order=8)
        self._base(4, 1e-12, order=8, max_ninter=100)

    def test_prob5(self):
        """Test for example problem #5."""
        self._base(5, 1e-4, order=16)
        self._base(5, 1e-8, order=16)
        self._base(5, 1e-12, order=16)

    def test_prob6(self):
        """Test for example problem #6."""
        self._base(6, 1e-4)
        # endpoint singularity, see TestQuadGK.test_prob6
        self._base(6, 1e-8, 100, status=2, extrap_status=0)
        self._base(6, 1e-12, 1e5, max_ninter=100, status=8, extrap_status=0)

    def test_prob7(self):
        """Test for example problem #7."""
        self._base(7, 1e-4)
        self._base(7, 1e-8, 10)
        self._base(7, 1e-12)

    def test_prob8(self):
        """Test for example problem #8."""
        self._base(8, 1e-4)
        self._base(8, 1e-8)
        self._base(8, 1e-12)

    def test_prob9(self):
        """Test for example problem #9."""
        # `sqrt(tan(t))` is unbounded at the right endpoint, so the sum over the mesh
        # runs away -- about 50 against a true value of 2.22 -- while its error estimate
        # outgrows it in turn. The extrapolation recovers the value (to ~5e-12 at the
        # tightest tolerance here), but the divergence test at the end compares the two,
        # and an error estimate larger than the mesh sum itself is one of the things it
        # rejects on, so the right answer arrives carrying a divergence flag. That is
        # the test working as intended on a mesh that really has diverged: scipy is
        # spared only because the Gauss-Kronrod mesh converges on this integrand where
        # the Clenshaw-Curtis one does not, which is a property of the local rule. The
        # value is still checked against the tolerance on every line below.
        divergent = 2**quadax.adaptive.DIVERGENT
        self._base(9, 1e-4, extrap_status=divergent)
        self._base(9, 1e-8, max_ninter=100, status=8, extrap_status=divergent)
        self._base(9, 1e-12, 1e4, max_ninter=100, status=8, extrap_status=0)

    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4)
        self._base(10, 1e-8)
        self._base(10, 1e-12, 10)

    def test_prob11(self):
        """Test for example problem #11."""
        self._base(11, 1e-4)
        # singularity at t=0, see TestQuadGK.test_prob11
        self._base(11, 1e-8, 100, status=8, extrap_status=0)
        self._base(11, 1e-12, 1e4, status=8, extrap_status=0)

    def test_prob12(self):
        """Test for example problem #12."""
        self._base(12, 1e-4)
        self._base(12, 1e-8)
        self._base(12, 1e-12)

    def test_prob13(self):
        """Test for example problem #13."""
        self._base(13, 1e-4)
        self._base(13, 1e-8)
        self._base(13, 1e-12)

    def test_prob14(self):
        """Test for example problem #14."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob15(self):
        """Test for example problem #15."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob16(self):
        """Test for example problem #16."""
        self._base(16, 1e-4)
        self._base(16, 1e-8)
        self._base(16, 1e-12)


class TestQuadTS(_BothExtrapolationModes):
    """Tests for adaptive tanh-sinh quadrature."""

    def _base(self, i, tol, fudge=1.0, **kwargs):
        prob = example_problems[i]
        status = kwargs.pop("status", 0)
        # Overrides for cases whose behaviour genuinely differs once the extrapolation
        # is switched on, applied only in that mode.
        extrap_status = kwargs.pop("extrap_status", None)
        extrap_fudge = kwargs.pop("extrap_fudge", None)
        if self.extrapolate:
            status = status if extrap_status is None else extrap_status
            fudge = fudge if extrap_fudge is None else extrap_fudge
        y, info = quadts(
            prob["fun"],
            prob["interval"],
            epsabs=tol,
            epsrel=tol,
            extrapolate=self.extrapolate,
            **kwargs,
        )
        assert info.status == status
        if status == 0:
            assert info.err < max(tol, tol * np.max(np.abs(y)))
        np.testing.assert_allclose(
            y,
            prob["val"],
            rtol=fudge * tol,
            atol=fudge * tol,
            err_msg=f"problem {i}, tol={tol}",
        )

    def test_prob0(self):
        """Test for example problem #0."""
        self._base(0, 1e-4)
        self._base(0, 1e-8)
        self._base(0, 1e-12)

    def test_prob1(self):
        """Test for example problem #1."""
        self._base(1, 1e-4)
        self._base(1, 1e-8)
        self._base(1, 1e-12)

    def test_prob2(self):
        """Test for example problem #2."""
        self._base(2, 1e-4, order=41)
        self._base(2, 1e-8, order=41)
        # The answer here is exact to machine precision, but at order 41 the error
        # *estimate* comes down slowly enough that it is still ~15% above the bound when
        # the subdivision limit is reached (a larger max_ninter does get under it). The
        # value is still checked against 1e-12 below.
        self._base(2, 1e-12, order=41, status=2)

    def test_prob3(self):
        """Test for example problem #3."""
        self._base(3, 1e-4, order=61)
        self._base(3, 1e-8, order=61)
        self._base(3, 1e-12, order=61)

    def test_prob4(self):
        """Test for example problem #4."""
        self._base(4, 1e-4, order=81)
        self._base(4, 1e-8, order=81)
        self._base(4, 1e-12, order=81)

    def test_prob5(self):
        """Test for example problem #5."""
        self._base(5, 1e-4, order=101)
        self._base(5, 1e-8, order=101)
        self._base(5, 1e-12, order=101)

    def test_prob6(self):
        """Test for example problem #6.

        The 1e-12 request is out of reach on an endpoint singularity, and which of
        ROUNDOFF / BAD_INTEGRAND the loop gives up with depends on exactly where the
        mesh stops, the value is what this really guards.
        """
        self._base(6, 1e-4)
        self._base(6, 1e-8)
        self._base(6, 1e-12, 1e4, status=4)

    def test_prob7(self):
        """Test for example problem #7."""
        self._base(7, 1e-4)
        self._base(7, 1e-8)
        self._base(7, 1e-12)

    def test_prob8(self):
        """Test for example problem #8."""
        self._base(8, 1e-4)
        self._base(8, 1e-8)
        self._base(8, 1e-12)

    def test_prob9(self):
        """Test for example problem #9.

        The 1e-12 request is out of reach on an endpoint singularity, and which of
        ROUNDOFF / BAD_INTEGRAND / NO_CONVERGE the loop gives up with depends on exactly
        where the mesh stops, the value is what this really guards. With acceleration it
        is the table that runs out first: the sequence it is fed stops improving while
        the tanh-sinh abscissae are still short of the tolerance.
        """
        self._base(9, 1e-4)
        self._base(9, 1e-8, 10)
        self._base(9, 1e-12, 1e4, status=8, extrap_status=16)

    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4)
        self._base(10, 1e-8)
        self._base(10, 1e-12)

    def test_prob11(self):
        """Test for example problem #11.

        The 1e-12 request is out of reach on an endpoint singularity, and which of
        ROUNDOFF / BAD_INTEGRAND the loop gives up with depends on exactly where the
        mesh stops, the value is what this really guards.
        """
        self._base(11, 1e-4)
        self._base(11, 1e-8)
        self._base(11, 1e-12, 1e4, status=4)

    def test_prob12(self):
        """Test for example problem #12."""
        self._base(12, 1e-4)
        self._base(12, 1e-8)
        self._base(12, 1e-12)

    def test_prob13(self):
        """Test for example problem #13."""
        self._base(13, 1e-4)
        self._base(13, 1e-8)
        self._base(13, 1e-12)

    def test_prob14(self):
        """Test for example problem #14."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob15(self):
        """Test for example problem #15."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob16(self):
        """Test for example problem #16."""
        self._base(16, 1e-4)
        self._base(16, 1e-8)
        self._base(16, 1e-12)


class TestRombergTS(_BothExtrapolationModes):
    """Tests for tanh-sinh quadrature with adaptive refinement.

    Run in both settings of ``extrapolate``, with no per-case overrides: the tanh-sinh
    rule converges doubly exponentially, so Richardson has no error expansion in powers
    of the step to work on and neither adds nor removes accuracy here. ``TestRomberg``
    cannot do the same, because there the trapezoidal rule really does depend on it.
    """

    def _base(self, i, tol, fudge=1.0, **kwargs):
        prob = example_problems[i]
        y, info = rombergts(
            prob["fun"],
            prob["interval"],
            epsabs=tol,
            epsrel=tol,
            extrapolate=self.extrapolate,
            **kwargs,
        )
        if info.status == 0:
            assert info.err < max(tol, tol * np.max(np.abs(y)))
        np.testing.assert_allclose(
            y,
            prob["val"],
            rtol=fudge * tol,
            atol=fudge * tol,
            err_msg=f"problem {i}, tol={tol}",
        )

    def test_prob0(self):
        """Test for example problem #0."""
        self._base(0, 1e-4)
        self._base(0, 1e-8)
        self._base(0, 1e-12)

    def test_prob1(self):
        """Test for example problem #1."""
        self._base(1, 1e-4)
        self._base(1, 1e-8)
        self._base(1, 1e-12)

    def test_prob2(self):
        """Test for example problem #2."""
        self._base(2, 1e-4)
        self._base(2, 1e-8)
        self._base(2, 1e-12)

    def test_prob3(self):
        """Test for example problem #3."""
        self._base(3, 1e-4)
        self._base(3, 1e-8)
        self._base(3, 1e-12)

    def test_prob4(self):
        """Test for example problem #4."""
        self._base(4, 1e-4)
        self._base(4, 1e-8)
        self._base(4, 1e-12)

    def test_prob5(self):
        """Test for example problem #5."""
        self._base(5, 1e-4)
        self._base(5, 1e-8)
        self._base(5, 1e-12)

    def test_prob6(self):
        """Test for example problem #6."""
        self._base(6, 1e-4)
        self._base(6, 1e-8, fudge=10)
        self._base(6, 1e-12, divmax=22, fudge=1e5)

    def test_prob7(self):
        """Test for example problem #7."""
        self._base(7, 1e-4)
        self._base(7, 1e-8)
        self._base(7, 1e-12)

    def test_prob8(self):
        """Test for example problem #8."""
        self._base(8, 1e-4)
        self._base(8, 1e-8)
        self._base(8, 1e-12)

    def test_prob9(self):
        """Test for example problem #9."""
        self._base(9, 1e-4)
        self._base(9, 1e-8, fudge=10)
        self._base(9, 1e-12, fudge=1e5)

    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4)
        self._base(10, 1e-8)
        self._base(10, 1e-12)

    def test_prob11(self):
        """Test for example problem #11."""
        self._base(11, 1e-4)
        self._base(11, 1e-8, fudge=10)
        self._base(11, 1e-12, fudge=1e5)

    def test_prob12(self):
        """Test for example problem #12."""
        self._base(12, 1e-4)
        self._base(12, 1e-8)
        self._base(12, 1e-12)

    def test_prob13(self):
        """Test for example problem #13."""
        self._base(13, 1e-4)
        self._base(13, 1e-8)
        self._base(13, 1e-12)

    def test_prob14(self):
        """Test for example problem #14."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob15(self):
        """Test for example problem #15."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob16(self):
        """Test for example problem #16."""
        self._base(16, 1e-4)
        self._base(16, 1e-8)
        self._base(16, 1e-12)


class TestRomberg:
    """Tests for Romberg's method (only for well behaved integrands)."""

    def _base(self, i, tol, fudge=1, **kwargs):
        prob = example_problems[i]
        y, info = romberg(
            prob["fun"], prob["interval"], epsabs=tol, epsrel=tol, **kwargs
        )
        if info.status == 0:
            assert info.err < max(tol, tol * np.max(np.abs(y)))
        np.testing.assert_allclose(
            y,
            prob["val"],
            rtol=fudge * tol,
            atol=fudge * tol,
            err_msg=f"problem {i}, tol={tol}",
        )

    def test_prob0(self):
        """Test for example problem #0."""
        self._base(0, 1e-4)
        self._base(0, 1e-8)
        self._base(0, 1e-12)

    def test_prob1(self):
        """Test for example problem #1."""
        self._base(1, 1e-4)
        self._base(1, 1e-8)
        self._base(1, 1e-12)

    def test_prob2(self):
        """Test for example problem #2."""
        self._base(2, 1e-4)
        self._base(2, 1e-8)
        self._base(2, 1e-12)

    def test_prob3(self):
        """Test for example problem #3."""
        self._base(3, 1e-4)
        self._base(3, 1e-8)
        self._base(3, 1e-12)

    def test_prob4(self):
        """Test for example problem #4."""
        self._base(4, 1e-4)
        self._base(4, 1e-8)
        self._base(4, 1e-12, divmax=27)

    def test_prob5(self):
        """Test for example problem #5."""
        self._base(5, 1e-4)
        self._base(5, 1e-8)
        self._base(5, 1e-12, divmax=25)

    def test_prob6(self):
        """Test for example problem #6."""
        self._base(6, 1e-4, fudge=10)

    def test_prob7(self):
        """Test for example problem #7."""
        self._base(7, 1e-4)

    def test_prob8(self):
        """Test for example problem #8."""
        self._base(8, 1e-4)

    @pytest.mark.xfail
    def test_prob9(self):
        """Test for example problem #9."""
        self._base(9, 1e-4)

    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4)

    def test_prob11(self):
        """Test for example problem #11."""
        self._base(11, 1e-4, fudge=10)

    def test_prob12(self):
        """Test for example problem #12."""
        self._base(12, 1e-4)
        self._base(12, 1e-8)
        self._base(12, 1e-12)

    def test_prob13(self):
        """Test for example problem #13."""
        self._base(13, 1e-4)
        self._base(13, 1e-8)
        self._base(13, 1e-12)

    def test_prob14(self):
        """Test for example problem #14."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob15(self):
        """Test for example problem #15."""
        self._base(14, 1e-4)
        self._base(14, 1e-8)
        self._base(14, 1e-12)

    def test_prob16(self):
        """Test for example problem #16."""
        self._base(16, 1e-4)
        self._base(16, 1e-8)
        self._base(16, 1e-12)


class TestExtrapolation:
    """Tests for the convergence acceleration in the adaptive solvers."""

    # The singular families from problems 17-25, plus the algebraically decaying ones on
    # infinite ranges whose transform induces a singularity of the same kind. Bisection
    # alone gets a handful of digits on any of these. Problem 31 is left out: its decay
    # is fast enough that the induced exponent is zero and the mapped integrand is
    # already smooth, so there is nothing for the table to do, it is covered by
    # ``test_converged_mesh_is_not_displaced`` instead.
    SINGULAR = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32]

    @pytest.mark.parametrize("i", SINGULAR)
    @pytest.mark.parametrize("quad", [quadgk, quadcc])
    def test_singularities_are_resolved(self, quad, i):
        """Acceleration should reach near machine precision where the mesh cannot."""
        prob = example_problems[i]
        kwargs = dict(epsabs=1e-12, epsrel=1e-12, max_ninter=200, full_output=True)
        y_off, info_off = quad(prob["fun"], prob["interval"], **kwargs)
        y_on, info_on = quad(prob["fun"], prob["interval"], extrapolate=True, **kwargs)
        exact = np.asarray(prob["val"])
        err_off = np.max(np.abs(np.asarray(y_off) - exact)) / np.max(np.abs(exact))
        err_on = np.max(np.abs(np.asarray(y_on) - exact)) / np.max(np.abs(exact))
        # Two orders of magnitude is well inside the margin: measured gains run from
        # 1e3 on the mildest of these to 1e11 on the strongest.
        assert err_on < err_off / 100, f"problem {i}: {err_off:.2e} -> {err_on:.2e}"
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

    def test_converged_mesh_is_not_displaced(self):
        """Where the subdivision converges, its result stands.

        The mesh sum is the one carrying an honest error bound, so a table fed early on
        a coarse mesh must not be able to replace it. Once the subdivision has reached
        the tolerance on its own the extrapolated value is not considered at all.
        """
        for i in (0, 1, 2, 12, 13):
            prob = example_problems[i]
            kwargs: dict[str, Any] = dict(epsabs=1e-10, epsrel=1e-10, full_output=True)
            y_off, _ = quadgk(prob["fun"], prob["interval"], **kwargs)
            y_on, info = quadgk(
                prob["fun"], prob["interval"], extrapolate=True, **kwargs
            )
            np.testing.assert_allclose(
                np.asarray(y_off),
                np.asarray(y_on),
                rtol=ULP_RTOL,
                atol=ULP_ATOL,
                err_msg=f"problem {i}",
            )

    @pytest.mark.parametrize("i", range(len(example_problems)))
    def test_acceleration_does_no_harm(self, i):
        """Switching acceleration on must not make any problem materially worse.

        Deliberately over the whole problem list rather than the singular subset: the
        risk the flag carries is not that it fails to help, it is that it quietly costs
        accuracy somewhere nobody was looking. The factor of ten is slack for a mesh
        that legitimately differs, since the acceleration changes which sub-interval is
        bisected next and the two runs are not obliged to agree exactly.
        """
        prob = example_problems[i]
        kwargs: dict[str, Any] = dict(epsabs=1e-10, epsrel=1e-10, max_ninter=200)
        exact = np.asarray(prob["val"])
        scale = max(np.max(np.abs(exact)), 1.0)

        def err(extrapolate):
            y, _ = quadgk(
                prob["fun"], prob["interval"], extrapolate=extrapolate, **kwargs
            )
            return np.max(np.abs(np.asarray(y) - exact)) / scale

        off, on = err(False), err(True)
        assert on <= max(10 * off, 1e-14), f"problem {i}: {off:.2e} -> {on:.2e}"

    # The four that scipy fails on, plus a spread of everything else: smooth cases
    # where the table should never fire at all, a vector and a complex integrand, a
    # breakpoint case, endpoint and interior singularities, and two infinite ranges.
    @pytest.mark.parametrize(
        "i", [0, 1, 3, 6, 9, 14, 15, 16, 17, 18, 19, 21, 23, 25, 27, 32]
    )
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
        prob = example_problems[i]
        exact = np.asarray(prob["val"])
        scale = np.max(np.abs(exact))

        def err(tol):
            y, _ = quadgk(
                prob["fun"],
                prob["interval"],
                epsabs=tol,
                epsrel=tol,
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
                f"problem {i}: tol={tol:g} gives {errs[tol]:.2e}, "
                f"best over the sweep is {best:.2e} ({errs})"
            )

    # Genuinely smooth cases only. Problem 5 looks like one and is not: the semicircle
    # `sqrt(1-t**2)` has an infinite derivative at the endpoint, which is exactly the
    # kind of endpoint behaviour the acceleration exists for, and it does fire there.
    @pytest.mark.parametrize("i", [0, 1, 2, 3, 13, 14, 16])
    def test_smooth_problems_never_extrapolate(self, i):
        """Where the subdivision converges, the table must stay out of the way.

        Checked all the way down to a tolerance of zero, because that is the setting
        that most changes the balance between subdividing and extrapolating, and a
        smooth integrand is where an accelerated value would be least justified.
        """
        prob = example_problems[i]
        for tol in (1e-8, 1e-12, 0.0):
            y, info = quadgk(
                prob["fun"],
                prob["interval"],
                epsabs=tol,
                epsrel=tol,
                order=21,
                max_ninter=200,
                full_output=True,
                extrapolate=True,
            )
            assert not bool(info.info["used_accel"]), (
                f"problem {i} at tol={tol:g} returned an extrapolated value"
            )
            # and the answer is the subdivision's own, unchanged by the flag
            y_off, _ = quadgk(
                prob["fun"],
                prob["interval"],
                epsabs=tol,
                epsrel=tol,
                order=21,
                max_ninter=200,
                full_output=True,
            )
            np.testing.assert_allclose(
                np.asarray(y),
                np.asarray(y_off),
                rtol=ULP_RTOL,
                atol=ULP_ATOL,
                err_msg=f"problem {i}, tol={tol:g}",
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


class TestRombergExtrapolationFlag:
    """Richardson extrapolation, on and off, over the whole example suite.

    Romberg's method *is* the trapezoidal rule plus Richardson, so this flag turns it
    into something else rather than merely tuning it: the same nodes and the same
    halving schedule, reading the un-extrapolated column. The two methods want opposite
    things from it, which is why it is a flag and not a fixed choice.

    ``divmax`` is well below the default here so the plain mode is affordable to test.
    Without extrapolation the trapezoidal rule needs O(h**2) refinement, so its
    evaluation count runs to millions on the harder problems, and the contrast the tests
    below check for is visible long before that.
    """

    DIVMAX = 14
    TOL = 1e-8
    # problems with a smooth integrand and finite limits, where Richardson's error
    # expansion in even powers of the step is valid and it should pay for itself
    SMOOTH = [0, 1, 2, 3, 13, 14, 16]

    def _run(self, method, i, extrapolate, tol=None, divmax=None):
        prob = example_problems[i]
        y, info = method(
            prob["fun"],
            jnp.asarray(prob["interval"], float),
            epsabs=tol or self.TOL,
            epsrel=tol or self.TOL,
            divmax=divmax or self.DIVMAX,
            full_output=True,
            extrapolate=extrapolate,
        )
        exact = np.asarray(prob["val"])
        scale = max(np.max(np.abs(exact)), 1e-300)
        err = float(np.max(np.abs(np.asarray(y) - exact)) / scale)
        return y, err, int(info.status), int(info.neval)

    def _suite(self):
        """The example problems Romberg can accept, ie the ones with no breakpoints."""
        for i, prob in enumerate(example_problems):
            if len(np.atleast_1d(np.asarray(prob["interval"], float))) == 2:
                yield i

    @pytest.mark.parametrize("method", [romberg, rombergts], ids=["romberg", "ts"])
    @pytest.mark.parametrize("extrapolate", [False, True], ids=["plain", "extrap"])
    def test_no_problem_returns_garbage(self, method, extrapolate):
        """Neither setting may produce a NaN or an infinity on any problem.

        Deliberately over the whole suite, including the problems neither setting can
        actually solve: what matters there is that a failure to converge stays a large
        finite error with a status to match, rather than becoming a NaN.
        """
        for i in self._suite():
            y, err, status, _ = self._run(method, i, extrapolate)
            assert np.all(np.isfinite(np.asarray(y))), f"problem {i}"
            assert np.isfinite(err), f"problem {i}"

    def test_richardson_is_what_makes_romberg_work(self):
        """Without it the trapezoidal rule cannot keep up on a smooth integrand.

        This is the justification for the flag defaulting to on. The gap is not
        marginal: Richardson reaches machine precision in tens of evaluations where
        bisection alone is still several digits short after tens of thousands.
        """
        for i in self.SMOOTH:
            _, err_on, _, neval_on = self._run(romberg, i, True)
            _, err_off, _, neval_off = self._run(romberg, i, False)
            assert err_on < err_off, f"problem {i}: {err_on:.2e} vs {err_off:.2e}"
            assert neval_on < neval_off, f"problem {i}"

    def test_tanh_sinh_gains_nothing_from_richardson(self):
        """On tanh-sinh it is at best neutral, and usually just costs evaluations.

        The rule already converges doubly exponentially, so there is no expansion in
        powers of the step for Richardson to cancel. Measured over the suite it helps
        nothing, is slightly worse on a few problems, and reaches the same accuracy in
        fewer evaluations on around half of them.
        """
        for i in self._suite():
            _, err_on, _, _ = self._run(rombergts, i, True)
            _, err_off, _, _ = self._run(rombergts, i, False)
            floor = 1e-14 * max(
                np.max(np.abs(np.asarray(example_problems[i]["val"]))), 1
            )
            assert err_off <= max(10 * err_on, floor), (
                f"problem {i}: turning extrapolation off made it worse, "
                f"{err_off:.2e} vs {err_on:.2e}"
            )

    @pytest.mark.parametrize("method", [romberg, rombergts], ids=["romberg", "ts"])
    def test_the_flag_does_not_change_the_table_shape(self, method):
        """``full_output`` keeps its contract either way, column 0 always filled.

        The two settings do not generally stop at the same level (they are comparing
        different estimates from one refinement to the next, so they meet the tolerance
        at different depths) but the trapezoidal column is the same computation in
        both, so wherever they both reached it holds the same numbers. Only what is
        built on top of it differs.
        """
        prob = example_problems[0]
        tables = {}
        for extrapolate in (False, True):
            _, info = method(
                prob["fun"],
                jnp.asarray(prob["interval"], float),
                epsabs=self.TOL,
                epsrel=self.TOL,
                divmax=self.DIVMAX,
                full_output=True,
                extrapolate=extrapolate,
            )
            tables[extrapolate] = np.asarray(info.info)
        assert tables[False].shape == tables[True].shape
        columns = [tables[e][:, 0] for e in (False, True)]
        # How far each column got: the length of its leading run of nonzeros. Counted
        # as a run rather than located as the first zero, because a column filled to
        # the end has no zero in it and a search would have to report its absence.
        depth = min(*(int((c != 0).cumprod().sum()) for c in columns), self.DIVMAX)
        assert depth > 1, "neither setting filled the trapezoidal column"
        np.testing.assert_allclose(
            columns[0][:depth], columns[1][:depth], rtol=ULP_RTOL, atol=ULP_ATOL
        )


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
rules = [GaussKronrodRule, ClenshawCurtisRule, TanhSinhRule]

real_dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
complex_dtypes = [jnp.complex128, jnp.complex64]
# `interval` is always real; complex is a property of the integrand's values.
real_of = {jnp.complex128: jnp.float64, jnp.complex64: jnp.float32}

# How much worse than sqrt(eps) a converged result is allowed to be. Generous, because
# the point of these tests is dtype plumbing, not accuracy.
_SLOP = 50


@pytest.fixture
def quiet_tanhsinh():
    """Let the half precision tanh-sinh warning through without failing the test.

    ``pyproject.toml`` turns warnings into errors, which is right for the rest of the
    suite. The warning itself is asserted on separately in ``TestTanhSinhPrecision``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tanh-sinh quadrature in.*")
        yield


@pytest.mark.usefixtures("quiet_tanhsinh")
class TestWorkingDType:
    """The dtype of ``interval`` selects the precision, and is respected end to end."""

    @pytest.mark.parametrize("method", all_methods)
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_round_trip(self, method, dtype):
        """Y and err come back at the requested precision, and the answer is right."""
        interval = jnp.array([0.0, 1.0], dtype=dtype)
        y, info = method(lambda x: jnp.exp(-x), interval)

        assert y.dtype == dtype
        assert jnp.asarray(info.err).dtype == dtype
        # exp(-x) on [0, 1]; only asking for sqrt(eps)-ish accuracy
        tol = _SLOP * np.sqrt(float(jnp.finfo(dtype).eps))
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
        tol = _SLOP * np.sqrt(float(jnp.finfo(rtype).eps))
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
        tol = _SLOP * np.sqrt(float(jnp.finfo(dtype).eps))
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
        tol = _SLOP * np.sqrt(float(jnp.finfo(dtype).eps))
        np.testing.assert_allclose(np.asarray(y), [1 - np.exp(-1), 1 / 3], atol=tol)


@pytest.mark.usefixtures("quiet_tanhsinh")
class TestFixedOrderRuleDTypes:
    """The fixed order rules are a public entry point in their own right."""

    @pytest.mark.parametrize("rule", rules)
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_integrate(self, rule, dtype):
        """All four outputs of ``integrate`` come back at the abscissa dtype."""
        a, b = jnp.array(0.0, dtype), jnp.array(1.0, dtype)
        y, err, y_abs, y_mmn = rule().integrate(lambda x: jnp.exp(-x), a, b, ())
        assert y.dtype == dtype
        assert err.dtype == y_abs.dtype == y_mmn.dtype == dtype
        tol = _SLOP * np.sqrt(float(jnp.finfo(dtype).eps))
        np.testing.assert_allclose(float(y), 1 - np.exp(-1), atol=tol)

    @pytest.mark.parametrize("rule", rules)
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_degenerate_interval(self, rule, dtype):
        """``a == b`` takes the other branch of a ``cond``, which has to agree.

        Both branches are built for any integrand dtype, so the zero branch must be
        constructed at the same dtype the weights promote the real branch to.
        """
        a = jnp.array(0.5, dtype)
        out = rule().integrate(lambda x: jnp.exp(-x), a, a, ())
        for v in out:
            assert v.dtype == dtype
            np.testing.assert_array_equal(np.asarray(v), 0.0)

    @pytest.mark.parametrize("rule", rules)
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_apply(self, rule, dtype):
        """The low level ``_apply`` keeps the dtype too."""
        a, b = jnp.array(0.0, dtype), jnp.array(1.0, dtype)
        y = rule()._apply(lambda x: jnp.exp(-x), a, b, ())
        assert y.dtype == dtype

    @pytest.mark.parametrize("rule", rules)
    def test_weights_sum_to_two(self, rule):
        """Both the high and low order rules integrate 1 over [-1, 1] exactly."""
        r = rule()
        np.testing.assert_allclose(float(jnp.sum(r._wh)), 2.0, atol=1e-14)
        np.testing.assert_allclose(float(jnp.sum(r._wl)), 2.0, atol=1e-14)


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
        y, _ = method(lambda x: jnp.exp(-x), jnp.array([0.0, 1.0], dtype=dtype))
        err = abs(float(y) - (1 - np.exp(-1)))
        assert err <= _SLOP * np.sqrt(float(jnp.finfo(dtype).eps))

    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_explicit_tolerances_do_not_promote(self, dtype):
        """A python float tolerance must not drag the working dtype up with it."""
        y, info = quadgk(
            lambda x: jnp.exp(-x),
            jnp.array([0.0, 1.0], dtype=dtype),
            epsabs=1e-3,
            epsrel=1e-3,
        )
        assert y.dtype == dtype
        assert jnp.asarray(info.err).dtype == dtype


class TestTanhSinhPrecision:
    """Half precision costs the tanh-sinh rules their double exponential clustering."""

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    @pytest.mark.parametrize("method", [quadts, rombergts])
    def test_warns_in_half_precision(self, dtype, method):
        """The user is told when the rule cannot deliver what it usually does."""
        with pytest.warns(UserWarning, match="tanh-sinh quadrature in"):
            method(lambda x: jnp.exp(-x), jnp.array([0.0, 1.0], dtype=dtype))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("method", [quadts, rombergts])
    def test_silent_at_float32_and_above(self, dtype, method):
        """No warning where the clustering is fine."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            method(lambda x: jnp.exp(-x), jnp.array([0.0, 1.0], dtype=dtype))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_quadgk_never_warns(self, dtype):
        """The warning belongs to the tanh-sinh rules, not to quadrature generally."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quadgk(lambda x: jnp.exp(-x), jnp.array([0.0, 1.0], dtype=dtype))

    @pytest.mark.usefixtures("quiet_tanhsinh")
    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_nodes_stay_inside_the_interval(self, dtype):
        """Rebuilt rather than cast, so no node collapses onto the endpoint.

        Casting a float64 table down to bfloat16 would round the outer nodes to exactly
        +/-1, silently dropping the effective order. Rebuilding at the target dtype
        spreads the same number of nodes over the range that dtype can resolve.
        """
        xh, _, _ = TanhSinhRule(order=61)._nodes_weights(dtype)
        assert xh.dtype == dtype
        assert len(np.unique(np.asarray(xh, dtype=np.float64))) == len(xh)
        assert np.all(np.abs(np.asarray(xh, dtype=np.float64)) < 1.0)


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
