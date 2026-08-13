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
]


class TestQuadGK:
    """Tests for Gauss-Kronrod quadrature."""

    def _base(self, i, tol, fudge=1.0, **kwargs):
        prob = example_problems[i]
        status = kwargs.pop("status", 0)
        y, info = quadgk(
            prob["fun"],
            prob["interval"],
            epsabs=tol,
            epsrel=tol,
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
        self._base(6, 1e-8, 100, order=15, status=2)
        self._base(6, 1e-12, 1e5, order=15, max_ninter=100, status=8)

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
        self._base(9, 1e-8, 100, order=15, status=2)
        self._base(9, 1e-12, 1e4, order=15, max_ninter=100, status=8)

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
        self._base(11, 1e-8, 100, order=21, status=8)
        self._base(11, 1e-12, 1e4, order=21, status=8, max_ninter=100)

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


class TestQuadCC:
    """Tests for Clenshaw-Curtis quadrature."""

    def _base(self, i, tol, fudge=1.0, **kwargs):
        prob = example_problems[i]
        status = kwargs.pop("status", 0)
        y, info = quadcc(
            prob["fun"],
            prob["interval"],
            epsabs=tol,
            epsrel=tol,
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
        self._base(6, 1e-8, 100, status=2)
        self._base(6, 1e-12, 1e5, max_ninter=100, status=8)

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
        self._base(9, 1e-4)
        self._base(9, 1e-8, max_ninter=100, status=8)
        self._base(9, 1e-12, 1e4, max_ninter=100, status=8)

    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4)
        self._base(10, 1e-8)
        self._base(10, 1e-12, 10)

    def test_prob11(self):
        """Test for example problem #11."""
        self._base(11, 1e-4)
        # singularity at t=0, see TestQuadGK.test_prob11
        self._base(11, 1e-8, 100, status=8)
        self._base(11, 1e-12, 1e4, status=8)

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


class TestQuadTS:
    """Tests for adaptive tanh-sinh quadrature."""

    def _base(self, i, tol, fudge=1.0, **kwargs):
        prob = example_problems[i]
        status = kwargs.pop("status", 0)
        y, info = quadts(
            prob["fun"],
            prob["interval"],
            epsabs=tol,
            epsrel=tol,
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
        """Test for example problem #6."""
        self._base(6, 1e-4)
        self._base(6, 1e-8)
        self._base(6, 1e-12, 1e4, status=8)

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
        self._base(9, 1e-8, 10)
        self._base(9, 1e-12, 1e4, status=4)

    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4)
        self._base(10, 1e-8)
        self._base(10, 1e-12)

    def test_prob11(self):
        """Test for example problem #11."""
        self._base(11, 1e-4)
        self._base(11, 1e-8)
        self._base(11, 1e-12, 1e4, status=2)

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


class TestRombergTS:
    """Tests for tanh-sinh quadrature with adaptive refinement."""

    def _base(self, i, tol, fudge=1.0, **kwargs):
        prob = example_problems[i]
        y, info = rombergts(
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
    y, info = quad(fun, [0.0, 1.0], (), True, max_ninter=max_ninter)

    a_arr, b_arr = info.info["a_arr"], info.info["b_arr"]
    ninter = int(info.info["ninter"])
    # the mapped reference domain is [-1, 1], so the widths must sum to exactly 2
    np.testing.assert_allclose(float(jnp.sum(b_arr - a_arr)), 2.0, rtol=0, atol=1e-14)
    # and every counted interval must actually be present
    assert int(jnp.sum(jnp.asarray(info.info["r_arr"]) != 0)) == ninter
    assert ninter <= max_ninter


@pytest.mark.parametrize("quad", [quadgk, quadcc, quadts])
def test_truncated_result_is_still_a_partition(quad):
    """Sub-intervals must be disjoint and contiguous when max_ninter is reached."""
    fun = lambda t: 1.0 / jnp.sqrt(jnp.abs(t - 0.3) + 1e-9)
    _, info = quad(fun, [0.0, 1.0], (), True, max_ninter=16)
    n = int(info.info["ninter"])
    a = np.asarray(info.info["a_arr"])[:n]
    b = np.asarray(info.info["b_arr"])[:n]
    order = np.argsort(a)
    a, b = a[order], b[order]
    np.testing.assert_allclose(a[0], -1.0, atol=1e-14)
    np.testing.assert_allclose(b[-1], 1.0, atol=1e-14)
    np.testing.assert_allclose(a[1:], b[:-1], atol=1e-14)


@pytest.mark.parametrize("max_ninter", [22, 23, 24, 25, 26])
def test_converged_iteration_exits_clean(max_ninter):
    """Meeting the tolerance as the budget runs out is not a failure.

    QUADPACK jumps past every ``ier`` assignment once ``errsum <= errbnd``, so an
    iteration that reaches the tolerance exits with ``ier = 0`` even if it also
    consumed the last subdivision slot. The flags used to be set unconditionally, with
    termination left to the loop predicate on the next pass, so an iteration that did
    both reported a spurious failure.
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

    Two regressions here. The stagnation test compared the bisected halves against
    ``r_arr[i]`` *after* it had been overwritten with the left half, so it was really
    asking whether the right half was negligible rather than whether subdivision had
    stopped moving the parent's value. And neither counter was gated on QUADPACK's
    ``defab == error`` check, which suppresses them when the local rule did not resolve
    a half at all, since a stagnant area is then evidence of an unresolved integrand
    rather than of roundoff. Together they made the loop give up early here, reporting
    ROUNDOFF with an error five orders of magnitude worse than achievable.
    """
    y, info = quadgk(_PEAK, jnp.array([0.0, 1.0]), epsabs=1e-12, epsrel=1e-12)
    assert int(info.status) == 0, quadax.STATUS[int(info.status)]
    np.testing.assert_allclose(float(y), _PEAK_VAL, rtol=1e-13, atol=0)


def test_tolerance_below_roundoff_floor_reports_roundoff():
    """Asking for more precision than the arithmetic allows is a ROUNDOFF verdict.

    The local rule floors each sub-interval's error estimate at ``50*eps*int|f|``, so
    the total cannot fall below that floor summed over the partition however fine the
    mesh gets. Here that floor is ~1.1e-14 relative, so a request of 1e-14 is out of
    reach.
    quadax tests for this only before the subdivision loop, as QUADPACK does, which left
    such a request to burn through the whole subdivision budget and report MAX_NINTER --
    true, but not the reason.
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
