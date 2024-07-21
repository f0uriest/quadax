"""Tests for sampled quadrature routines."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

from quadax import cumulative_simpson, cumulative_trapezoid, simpson, trapezoid

config.update("jax_enable_x64", True)

rng = np.random.default_rng(0)
A0 = 0.5 - rng.random(10)

example_problems = [
    # problem 0
    {
        "fun": lambda t: jnp.log(t),
        "a": 1,
        "b": 3,
        "val": (3 * (jnp.log(3) - 1)) - (1 * (jnp.log(1) - 1)),
        "int": lambda t: t * (jnp.log(t) - 1),
    },
    # problem 1
    {
        "fun": lambda t: jnp.polyval(A0, t),
        "a": -1,
        "b": 1,
        "val": jnp.polyval(jnp.polyint(A0), 1) - jnp.polyval(jnp.polyint(A0), -1),
        "int": lambda t: jnp.polyval(jnp.polyint(A0), t),
    },
    # problem 2
    {
        "fun": lambda t: 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(t**2)),
        "a": -2,
        "b": 3,
        "val": jax.scipy.special.erf(3) - jax.scipy.special.erf(-2),
        "int": jax.scipy.special.erf,
    },
]


class TestTrapezoid:
    """Tests for trapezoidal integration from sampled values."""

    def _base(self, i, n, tol):
        prob = example_problems[i]
        a, b = prob["a"], prob["b"]
        # evenly spaced points
        x1 = a + (b - a) * np.linspace(0, 1, n)
        # unevenly spaced points
        x2 = a + (b - a) * np.linspace(0, 1, n) ** 2
        f1 = prob["fun"](x1)
        f2 = prob["fun"](x2)
        y1 = trapezoid(f1, x=x1)
        y2 = trapezoid(f2, x=x2)
        y3 = trapezoid(f1, dx=np.diff(x1)[0])
        np.testing.assert_allclose(y1, y3)
        np.testing.assert_allclose(y1, prob["val"], atol=tol, rtol=tol)
        np.testing.assert_allclose(y2, prob["val"], atol=tol, rtol=tol)

    def test_prob0(self):
        """Test integrating log(x)."""
        self._base(0, 10, 3e-3 / 4**0)
        self._base(0, 20, 3e-3 / 4**1)
        self._base(0, 40, 3e-3 / 4**2)
        self._base(0, 80, 3e-3 / 4**3)

    def test_prob1(self):
        """Test integrating a high order polynomial."""
        self._base(1, 10, 3e-2 / 4**0)
        self._base(1, 20, 3e-2 / 4**1)
        self._base(1, 40, 3e-2 / 4**2)
        self._base(1, 80, 3e-2 / 4**3)

    def test_prob2(self):
        """Test integrating a gaussian."""
        self._base(2, 10, 2e-3 / 4**0)
        self._base(2, 20, 2e-3 / 4**1)
        self._base(2, 40, 2e-3 / 4**2)
        self._base(2, 80, 2e-3 / 4**3)


class TestSimpson:
    """Tests for integration from sampled values using Simpsons rule."""

    def _base(self, i, n, tol):
        prob = example_problems[i]
        a, b = prob["a"], prob["b"]
        # evenly spaced points
        x1 = a + (b - a) * np.linspace(0, 1, n)
        f1 = prob["fun"](x1)
        y1 = simpson(f1, x=x1)
        y3 = simpson(f1, dx=np.diff(x1)[0])
        np.testing.assert_allclose(y1, y3)
        np.testing.assert_allclose(y1, prob["val"], atol=tol, rtol=tol)

    def test_prob0(self):
        """Test integrating log(x)."""
        self._base(0, 10, 2e-4 / 8**0)
        self._base(0, 20, 2e-4 / 8**1)
        self._base(0, 40, 2e-4 / 8**2)
        self._base(0, 80, 2e-4 / 8**3)

    def test_prob1(self):
        """Test integrating a high order polynomial."""
        self._base(1, 10, 1e-2 / 8**0)
        self._base(1, 20, 1e-2 / 8**1)
        self._base(1, 40, 1e-2 / 8**2)
        self._base(1, 80, 1e-2 / 8**3)

    def test_prob2(self):
        """Test integrating a gaussian."""
        self._base(2, 10, 1e-3 / 8**0)
        self._base(2, 20, 1e-3 / 8**1)
        self._base(2, 40, 1e-3 / 8**2)
        self._base(2, 80, 1e-3 / 8**3)


class TestCumulativeTrapezoid:
    """Tests for cumulative integration using trapezoidal rule."""

    def _base(self, i, n, tol):
        prob = example_problems[i]
        a, b = prob["a"], prob["b"]
        # evenly spaced points
        x1 = a + (b - a) * np.linspace(0, 1, n)
        f1 = prob["fun"](x1)
        y1 = cumulative_trapezoid(f1, x=x1, initial=0) + prob["int"](a)
        y3 = cumulative_trapezoid(f1, dx=np.diff(x1)[0], initial=0) + prob["int"](a)
        np.testing.assert_allclose(y1, y3)
        np.testing.assert_allclose(y1, prob["int"](x1), atol=tol, rtol=tol)

    def test_prob0(self):
        """Test integrating log(x)."""
        self._base(0, 10, 1e-2 / 4**0)
        self._base(0, 20, 1e-2 / 4**1)
        self._base(0, 40, 1e-2 / 4**2)
        self._base(0, 80, 1e-2 / 4**3)

    def test_prob1(self):
        """Test integrating a high order polynomial."""
        self._base(1, 10, 2e-2 / 4**0)
        self._base(1, 20, 2e-2 / 4**1)
        self._base(1, 40, 2e-2 / 4**2)
        self._base(1, 80, 2e-2 / 4**3)

    def test_prob2(self):
        """Test integrating a gaussian."""
        self._base(2, 10, 3e-2 / 4**0)
        self._base(2, 20, 3e-2 / 4**1)
        self._base(2, 40, 3e-2 / 4**2)
        self._base(2, 80, 3e-2 / 4**3)


class TestCumulativeSimpson:
    """Tests for cumulative integration using simpsons rule."""

    def _base(self, i, n, tol):
        prob = example_problems[i]
        a, b = prob["a"], prob["b"]
        # evenly spaced points
        x1 = a + (b - a) * np.linspace(0, 1, n)
        f1 = prob["fun"](x1)
        y1 = cumulative_simpson(f1, x=x1, initial=0) + prob["int"](a)
        y3 = cumulative_simpson(f1, dx=np.diff(x1)[0], initial=0) + prob["int"](a)
        np.testing.assert_allclose(y1, y3)
        np.testing.assert_allclose(y1, prob["int"](x1), atol=tol, rtol=tol)

    def test_prob0(self):
        """Test integrating log(x)."""
        self._base(0, 10, 1e-2 / 8**0)
        self._base(0, 20, 1e-2 / 8**1)
        self._base(0, 40, 1e-2 / 8**2)
        self._base(0, 80, 1e-2 / 8**3)

    def test_prob1(self):
        """Test integrating a high order polynomial."""
        self._base(1, 10, 2e-2 / 8**0)
        self._base(1, 20, 2e-2 / 8**1)
        self._base(1, 40, 2e-2 / 8**2)
        self._base(1, 80, 2e-2 / 8**3)

    def test_prob2(self):
        """Test integrating a gaussian."""
        self._base(2, 10, 3e-2 / 8**0)
        self._base(2, 20, 3e-2 / 8**1)
        self._base(2, 40, 3e-2 / 8**2)
        self._base(2, 80, 3e-2 / 8**3)
