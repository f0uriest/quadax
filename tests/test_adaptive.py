"""Tests for adaptive quadrature routines."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy
from jax.config import config as jax_config

from quadax import quadcc, quadgk, quadts, romberg

jax_config.update("jax_enable_x64", True)

example_problems = [
    # problem 0
    {"fun": lambda t: t * jnp.log(1 + t), "a": 0, "b": 1, "val": 1 / 4},
    # problem 1
    {
        "fun": lambda t: t**2 * jnp.arctan(t),
        "a": 0,
        "b": 1,
        "val": (jnp.pi - 2 + 2 * jnp.log(2)) / 12,
    },
    # problem 2
    {
        "fun": lambda t: jnp.exp(t) * jnp.cos(t),
        "a": 0,
        "b": jnp.pi / 2,
        "val": (jnp.exp(jnp.pi / 2) - 1) / 2,
    },
    # problem 3
    {
        "fun": lambda t: jnp.arctan(jnp.sqrt(2 + t**2))
        / ((1 + t**2) * jnp.sqrt(2 + t**2)),
        "a": 0,
        "b": 1,
        "val": 5 * jnp.pi**2 / 96,
    },
    # problem 4
    {"fun": lambda t: jnp.sqrt(t) * jnp.log(t), "a": 0, "b": 1, "val": -4 / 9},
    # problem 5
    {"fun": lambda t: jnp.sqrt(1 - t**2), "a": 0, "b": 1, "val": jnp.pi / 4},
    # problem 6
    {
        "fun": lambda t: jnp.sqrt(t) / jnp.sqrt(1 - t**2),
        "a": 0,
        "b": 1,
        "val": 2
        * jnp.sqrt(jnp.pi)
        * scipy.special.gamma(3 / 4)
        / scipy.special.gamma(1 / 4),
    },
    # problem 7
    {"fun": lambda t: jnp.log(t) ** 2, "a": 0, "b": 1, "val": 2},
    # problem 8
    {
        "fun": lambda t: jnp.log(jnp.cos(t)),
        "a": 0,
        "b": jnp.pi / 2,
        "val": -jnp.pi * jnp.log(2) / 2,
    },
    # problem 9
    {
        "fun": lambda t: jnp.sqrt(jnp.tan(t)),
        "a": 0,
        "b": jnp.pi / 2,
        "val": jnp.pi * jnp.sqrt(2) / 2,
    },
    # problem 10
    {"fun": lambda t: 1 / (1 + t**2), "a": 0, "b": jnp.inf, "val": jnp.pi / 2},
    # problem 11
    {
        "fun": lambda t: jnp.exp(-t) / jnp.sqrt(t),
        "a": 0,
        "b": jnp.inf,
        "val": jnp.sqrt(jnp.pi),
    },
    # problem 12
    {
        "fun": lambda t: jnp.exp(-(t**2) / 2),
        "a": -jnp.inf,
        "b": jnp.inf,
        "val": jnp.sqrt(2 * jnp.pi),
    },
    # problem 13
    {"fun": lambda t: jnp.exp(-t) * jnp.cos(t), "a": 0, "b": jnp.inf, "val": 1 / 2},
]


class TestQuadGK:
    """Tests for Gauss-Konrod quadrature."""

    def _base(self, i, tol, fudge=1, **kwargs):
        prob = example_problems[i]
        status = kwargs.pop("status", 0)
        y, info = quadgk(
            prob["fun"],
            prob["a"],
            prob["b"],
            epsabs=tol,
            epsrel=tol,
            full_output=True,
            **kwargs,
        )
        assert info.status == status
        if status == 0:
            assert info.err < max(tol, tol * y)
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
        self._base(6, 1e-8, 100, order=15)
        self._base(6, 1e-12, 1e5, order=15, max_ninter=100, status=8)

    def test_prob7(self):
        """Test for example problem #7."""
        self._base(7, 1e-4, order=61)
        self._base(7, 1e-8, order=61)
        self._base(7, 1e-12, order=61, status=4)

    def test_prob8(self):
        """Test for example problem #8."""
        self._base(8, 1e-4, order=51)
        self._base(8, 1e-8, order=51)
        self._base(8, 1e-12, order=51, status=4)

    def test_prob9(self):
        """Test for example problem #9."""
        self._base(9, 1e-4, order=15)
        self._base(9, 1e-8, 100, order=15)
        self._base(9, 1e-12, 1e4, order=15, max_ninter=100, status=8)

    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4, order=15)
        self._base(10, 1e-8, order=15)
        self._base(10, 1e-12, order=15)

    def test_prob11(self):
        """Test for example problem #11."""
        self._base(11, 1e-4, order=21)
        self._base(11, 1e-8, 100, order=21)
        self._base(11, 1e-12, 1e4, order=21, status=8)

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


class TestQuadCC:
    """Tests for Clenshaw-Curtis quadrature."""

    def _base(self, i, tol, fudge=1, **kwargs):
        prob = example_problems[i]
        status = kwargs.pop("status", 0)
        y, info = quadcc(
            prob["fun"],
            prob["a"],
            prob["b"],
            epsabs=tol,
            epsrel=tol,
            full_output=True,
            **kwargs,
        )
        assert info.status == status
        if status == 0:
            assert info.err < max(tol, tol * y)
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
        self._base(6, 1e-8, 100)
        self._base(6, 1e-12, 1e5, max_ninter=100, status=8)

    def test_prob7(self):
        """Test for example problem #7."""
        self._base(7, 1e-4)
        self._base(7, 1e-8, 10)
        self._base(7, 1e-12, status=8)

    def test_prob8(self):
        """Test for example problem #8."""
        self._base(8, 1e-4)
        self._base(8, 1e-8)
        self._base(8, 1e-12, status=8)

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
        self._base(11, 1e-8, 100)
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


class TestQuadTS:
    """Tests for tanh-sinh quadrature with adaptive refinement."""

    def _base(self, i, tol, fudge=1, **kwargs):
        prob = example_problems[i]
        y, info = quadts(
            prob["fun"], prob["a"], prob["b"], epsabs=tol, epsrel=tol, **kwargs
        )
        assert info.err < max(tol, tol * y)
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
        self._base(6, 1e-8, 10)
        self._base(6, 1e-12, 1e5)

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
        self._base(9, 1e-12, 1e5)

    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4)
        self._base(10, 1e-8)
        self._base(10, 1e-12)

    def test_prob11(self):
        """Test for example problem #11."""
        self._base(11, 1e-4)
        self._base(11, 1e-8, 10)
        self._base(11, 1e-12, 1e5, divmax=25)

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


class TestRomberg:
    """Tests for Romberg's method (only for well behaved integrands)."""

    def _base(self, i, tol, fudge=1):
        prob = example_problems[i]
        y, info = romberg(prob["fun"], prob["a"], prob["b"], epsabs=tol, epsrel=tol)
        assert info.err < max(tol, tol * y)
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
        self._base(4, 1e-4, 100)
        self._base(4, 1e-8, 1e5)
        self._base(4, 1e-12, 1e7)

    def test_prob5(self):
        """Test for example problem #5."""
        self._base(5, 1e-4, 10)
        self._base(5, 1e-8, 1e4)
        self._base(5, 1e-12, 1e6)

    @pytest.mark.xfail
    def test_prob6(self):
        """Test for example problem #6."""
        self._base(6, 1e-4)
        self._base(6, 1e-8)
        self._base(6, 1e-12)

    @pytest.mark.xfail
    def test_prob7(self):
        """Test for example problem #7."""
        self._base(7, 1e-4)
        self._base(7, 1e-8)
        self._base(7, 1e-12)

    @pytest.mark.xfail
    def test_prob8(self):
        """Test for example problem #8."""
        self._base(8, 1e-4)
        self._base(8, 1e-8)
        self._base(8, 1e-12)

    @pytest.mark.xfail
    def test_prob9(self):
        """Test for example problem #9."""
        self._base(9, 1e-4)
        self._base(9, 1e-8)
        self._base(9, 1e-12)

    @pytest.mark.xfail
    def test_prob10(self):
        """Test for example problem #10."""
        self._base(10, 1e-4)
        self._base(10, 1e-8)
        self._base(10, 1e-12)

    @pytest.mark.xfail
    def test_prob11(self):
        """Test for example problem #11."""
        self._base(11, 1e-4)
        self._base(11, 1e-8)
        self._base(11, 1e-12)

    @pytest.mark.xfail
    def test_prob12(self):
        """Test for example problem #12."""
        self._base(12, 1e-4)
        self._base(12, 1e-8)
        self._base(12, 1e-12)

    @pytest.mark.xfail
    def test_prob13(self):
        """Test for example problem #13."""
        self._base(13, 1e-4)
        self._base(13, 1e-8)
        self._base(13, 1e-12)
