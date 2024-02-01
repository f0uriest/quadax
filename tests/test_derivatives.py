"""Tests for custom jvp for adaptive quadrature routines."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

from quadax import quadcc, quadgk, quadts, romberg, rombergts

config.update("jax_enable_x64", True)


rng = np.random.default_rng(0)
A0 = 0.5 - rng.random(3)


example_problems = [
    # problem 0
    {
        "fun": lambda t, c: jnp.log(c * t).squeeze(),
        "interval": [1, 3],
        "args": (np.array([3.12]),),
    },
    # problem 1
    {
        "fun": lambda t, m_s: jnp.exp(-((t - m_s[0]) ** 2) / m_s[1] ** 2).squeeze(),
        "interval": [-2, 3],
        "args": (np.array([1.23, 0.67]),),
    },
]


def finite_difference(f, x, eps=1e-8):
    """Util for 2nd order centered finite differences."""
    x0 = np.atleast_1d(x).squeeze()
    f0 = f((x0,))
    m = f0.size
    n = x0.size
    J = np.zeros((m, n))
    h = np.maximum(1.0, np.abs(x0)) * eps
    h_vecs = np.diag(np.atleast_1d(h))
    for i in range(n):
        x1 = x0 - h_vecs[i]
        x2 = x0 + h_vecs[i]
        if x0.ndim:
            dx = x2[i] - x1[i]
        else:
            dx = x2 - x1
        f1 = f((x1,))
        f2 = f((x2,))
        df = f2 - f1
        dfdx = df / dx
        J[:, i] = dfdx.flatten()
    if m == 1:
        J = np.ravel(J)
    return J


class TestQuadGKJac:
    """Tests for derivatives of quadgk."""

    def _base(self, i, tol, **kwargs):

        prob = example_problems[i]

        def integrate(args):
            y, err = quadgk(prob["fun"], prob["interval"], args, **kwargs)
            return y

        jacfd = finite_difference(integrate, prob["args"])
        jacadf = jax.jacfwd(integrate)(prob["args"])[0]
        jacadr = jax.jacrev(integrate)(prob["args"])[0]
        np.testing.assert_allclose(jacfd, jacadf, atol=tol, rtol=tol)
        np.testing.assert_allclose(jacfd, jacadr, atol=tol, rtol=tol)
        np.testing.assert_allclose(jacadr, jacadf, atol=1e-14, rtol=1e-14)

    def test_prob0(self):
        """Test for derivative of integral of log."""
        self._base(0, 1e-4)

    def test_prob1(self):
        """Test for derivative of integral of gaussian."""
        self._base(1, 1e-4)


class TestQuadCCJac:
    """Tests for derivatives of quadcc."""

    def _base(self, i, tol, **kwargs):

        prob = example_problems[i]

        def integrate(args):
            y, err = quadcc(prob["fun"], prob["interval"], args, **kwargs)
            return y

        jacfd = finite_difference(integrate, prob["args"])
        jacadf = jax.jacfwd(integrate)(prob["args"])[0]
        jacadr = jax.jacrev(integrate)(prob["args"])[0]
        np.testing.assert_allclose(jacfd, jacadf, atol=tol, rtol=tol)
        np.testing.assert_allclose(jacfd, jacadr, atol=tol, rtol=tol)
        np.testing.assert_allclose(jacadr, jacadf, atol=1e-14, rtol=1e-14)

    def test_prob0(self):
        """Test for derivative of integral of log."""
        self._base(0, 1e-4)

    def test_prob1(self):
        """Test for derivative of integral of gaussian."""
        self._base(1, 1e-4)


class TestQuadTSJac:
    """Tests for derivatives of quadts."""

    def _base(self, i, tol, **kwargs):

        prob = example_problems[i]

        def integrate(args):
            y, err = quadts(prob["fun"], prob["interval"], args, **kwargs)
            return y

        jacfd = finite_difference(integrate, prob["args"])
        jacadf = jax.jacfwd(integrate)(prob["args"])[0]
        jacadr = jax.jacrev(integrate)(prob["args"])[0]
        np.testing.assert_allclose(jacfd, jacadf, atol=tol, rtol=tol)
        np.testing.assert_allclose(jacfd, jacadr, atol=tol, rtol=tol)
        np.testing.assert_allclose(jacadr, jacadf, atol=1e-14, rtol=1e-14)

    def test_prob0(self):
        """Test for derivative of integral of log."""
        self._base(0, 1e-4)

    def test_prob1(self):
        """Test for derivative of integral of gaussian."""
        self._base(1, 1e-4)


class TestRombergJac:
    """Tests for derivatives of romberg."""

    def _base(self, i, tol, **kwargs):

        prob = example_problems[i]

        def integrate(args):
            y, err = romberg(prob["fun"], prob["interval"], args, **kwargs)
            return y

        jacfd = finite_difference(integrate, prob["args"])
        jacadf = jax.jacfwd(integrate)(prob["args"])[0]
        np.testing.assert_allclose(jacfd, jacadf, atol=tol, rtol=tol)

    def test_prob0(self):
        """Test for derivative of integral of log."""
        self._base(0, 1e-4)

    def test_prob1(self):
        """Test for derivative of integral of gaussian."""
        self._base(1, 1e-4)


class TestRombergTSJac:
    """Tests for derivatives of rombergts."""

    def _base(self, i, tol, **kwargs):

        prob = example_problems[i]

        def integrate(args):
            y, err = rombergts(prob["fun"], prob["interval"], args, **kwargs)
            print("y=", y)
            print("e=", err)
            return y

        jacfd = finite_difference(integrate, prob["args"])
        jacadf = jax.jacfwd(integrate)(prob["args"])[0]
        np.testing.assert_allclose(jacfd, jacadf, atol=tol, rtol=tol)

    def test_prob0(self):
        """Test for derivative of integral of log."""
        self._base(0, 1e-4)

    def test_prob1(self):
        """Test for derivative of integral of gaussian."""
        self._base(1, 1e-4)
