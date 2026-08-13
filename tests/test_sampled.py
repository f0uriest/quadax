"""Tests for sampled quadrature routines."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


# (number of points, refinement level). Simpson's rule takes a different path for an
# odd than an even number of points: an odd count is the classic composite rule, while
# an even count leaves a spare interval that needs a separate correction. Both parities
# are checked at every resolution, against the same tolerance schedule.
SIMPSON_NPTS = [(9, 0), (10, 0), (19, 1), (20, 1), (39, 2), (40, 2), (79, 3), (80, 3)]


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

    @pytest.mark.parametrize("n, level", SIMPSON_NPTS)
    def test_prob0(self, n, level):
        """Test integrating log(x)."""
        self._base(0, n, 2e-4 / 8**level)

    @pytest.mark.parametrize("n, level", SIMPSON_NPTS)
    def test_prob1(self, n, level):
        """Test integrating a high order polynomial."""
        self._base(1, n, 1e-2 / 8**level)

    @pytest.mark.parametrize("n, level", SIMPSON_NPTS)
    def test_prob2(self, n, level):
        """Test integrating a gaussian."""
        self._base(2, n, 1e-3 / 8**level)


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


class TestAxis:
    """Integration along a chosen axis of a multidimensional array.

    ``axis`` is a documented argument of every routine here, but the tests above always
    leave it at its default on 1-D input, so none of the ``moveaxis``/``swapaxes``
    handling is otherwise covered. Each case stacks several scaled copies of a 1-D
    problem and checks that integrating the stack reproduces integrating each copy on
    its own, which pins the values and the orientation of the result together.
    """

    scales = (1.0, 2.0, -0.5)

    def _stack(self, i=2, n=21):
        """Return sample points and a stack of scaled copies of the integrand."""
        prob = example_problems[i]
        a, b = prob["a"], prob["b"]
        x = a + (b - a) * np.linspace(0, 1, n)
        return x, jnp.stack([prob["fun"](x) * s for s in self.scales])

    @pytest.mark.parametrize("quad", [trapezoid, simpson])
    def test_definite_along_each_axis(self, quad):
        """Integrating a stack matches integrating each row on its own."""
        x, y = self._stack()
        want = np.array([np.asarray(quad(row, x=x)) for row in y])
        np.testing.assert_allclose(np.asarray(quad(y, x=x, axis=-1)), want)
        np.testing.assert_allclose(np.asarray(quad(y.T, x=x, axis=0)), want)

    @pytest.mark.parametrize("quad", [cumulative_trapezoid, cumulative_simpson])
    def test_cumulative_along_each_axis(self, quad):
        """The cumulative variants preserve the orientation of the input."""
        x, y = self._stack()
        want = np.stack([np.asarray(quad(row, x=x, initial=0)) for row in y])
        np.testing.assert_allclose(np.asarray(quad(y, x=x, axis=-1, initial=0)), want)
        np.testing.assert_allclose(
            np.asarray(quad(y.T, x=x, axis=0, initial=0)), want.T
        )

    @pytest.mark.parametrize("quad", [trapezoid, simpson])
    def test_multidimensional_x(self, quad):
        """``x`` may be given with the same shape as ``y`` rather than 1-D."""
        x, y = self._stack()
        x_full = jnp.broadcast_to(jnp.asarray(x), y.shape)
        np.testing.assert_allclose(
            np.asarray(quad(y, x=x_full, axis=-1)),
            np.asarray(quad(y, x=x, axis=-1)),
        )

    @pytest.mark.parametrize("quad", [cumulative_trapezoid, cumulative_simpson])
    def test_cumulative_multidimensional_x(self, quad):
        """Same for the cumulative variants, which take a separate code path."""
        x, y = self._stack()
        x_full = jnp.broadcast_to(jnp.asarray(x), y.shape)
        np.testing.assert_allclose(
            np.asarray(quad(y, x=x_full, axis=-1, initial=0)),
            np.asarray(quad(y, x=x, axis=-1, initial=0)),
        )


class TestDegenerateSizes:
    """Too few points to form a parabolic segment falls back to the trapezoid rule."""

    def test_simpson_two_points(self):
        """With only two points there is no parabola to fit, so use the trapezoid."""
        y, x = np.array([1.0, 3.0]), np.array([0.0, 2.0])
        np.testing.assert_allclose(
            np.asarray(simpson(y, x=x)), np.asarray(trapezoid(y, x=x))
        )

    def test_cumulative_simpson_two_points(self):
        """Likewise for the cumulative version."""
        y = np.array([1.0, 3.0])
        np.testing.assert_allclose(
            np.asarray(cumulative_simpson(y, dx=2.0, initial=0)),
            np.asarray(cumulative_trapezoid(y, dx=2.0, initial=0)),
        )


class TestErrors:
    """Invalid shapes must be reported, rather than silently broadcasting."""

    y = np.linspace(1.0, 2.0, 11)
    x = np.linspace(0.0, 1.0, 11)

    def test_cumulative_trapezoid_x_ndim(self):
        """`x` must be 1-D or match the shape of `y`."""
        with pytest.raises(ValueError, match="shape of x must be"):
            cumulative_trapezoid(np.ones((4, 5)), x=np.ones((2, 2, 2)))

    def test_cumulative_trapezoid_x_length(self):
        """1-D `x` must have the same length as `y` along `axis`."""
        with pytest.raises(ValueError, match="length of x along axis"):
            cumulative_trapezoid(self.y, x=np.linspace(0, 1, 5))

    def test_cumulative_trapezoid_initial_not_scalar(self):
        """`initial` must be a scalar."""
        with pytest.raises(ValueError, match="`initial` parameter should be a scalar"):
            cumulative_trapezoid(self.y, x=self.x, initial=np.array([0.0, 1.0]))

    def test_simpson_x_shape(self):
        """`x` must be 1-D or match the shape of `y`."""
        with pytest.raises(ValueError, match="shape of x must be"):
            simpson(np.ones((4, 5)), x=np.ones((2, 2, 2)))

    def test_simpson_x_length(self):
        """1-D `x` must have the same length as `y` along `axis`."""
        with pytest.raises(ValueError, match="length of x along axis"):
            simpson(self.y, x=np.linspace(0, 1, 5))

    def test_cumulative_simpson_axis(self):
        """`axis` must be valid for the shape of `y`."""
        with pytest.raises(ValueError, match="is not valid for"):
            cumulative_simpson(self.y, x=self.x, axis=5)

    def test_cumulative_simpson_x_shape(self):
        """`x` must match `y`, or be 1-D with the right length along `axis`."""
        with pytest.raises(ValueError, match="shape of `x` must be"):
            cumulative_simpson(self.y, x=np.linspace(0, 1, 5))

    def test_cumulative_simpson_dx_shape(self):
        """`dx` must be a scalar or have one point along `axis`."""
        with pytest.raises(ValueError, match="`dx` must either be a scalar"):
            cumulative_simpson(self.y, dx=np.ones(7))

    def test_cumulative_simpson_initial_shape(self):
        """`initial` must be a scalar or have one point along `axis`."""
        with pytest.raises(ValueError, match="`initial` must either be a scalar"):
            cumulative_simpson(self.y, x=self.x, initial=np.ones(7))
