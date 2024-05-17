"""Tests for adaptive weighted quadrature routines."""

import numpy as np
from jax import config

from quadax import adaptive_quadrature, fixed_quadcc
from quadax.weighted import AlgLogLeftSingularRule

config.update("jax_enable_x64", True)


class TestAlgLogLeftSingularWeightedQuad:
    """Tests for algebraic-logarithmic weighted quadrature."""

    example_problems = [
        # problem 0 - algebraic weight
        {
            "base fun": lambda t: 1,
            "weight args": {"singularpoint": 0, "algpower": -1 / 2, "logpower": False},
            "interval": [0, 1],
            "val": 2,
        },
        # problem 1 - logarithmic weight
        {
            "base fun": lambda t: 1,
            "weight args": {"singularpoint": 0, "algpower": 0, "logpower": True},
            "interval": [0, 1],
            "val": -1,
        },
        # problem 2 - alg-log weight
        {
            "base fun": lambda t: 1,
            "weight args": {"singularpoint": 0, "algpower": -1 / 2, "logpower": True},
            "interval": [0, 1],
            "val": -4,
        },
        # problem 3 - alg-log weight with nonzero singular point
        {
            "base fun": lambda t: 1,
            "weight args": {"singularpoint": 1, "algpower": -1 / 2, "logpower": True},
            "interval": [1, 2],
            "val": -4,
        },
        # problem 4 - algebraic weight with nontrivial base function
        {
            "base fun": lambda t: t**2,
            "weight args": {"singularpoint": 1, "algpower": -1 / 3, "logpower": False},
            "interval": [1, 2],
            "val": 123 / 40,
        },
        # problem 5 - algebraic weight with positive power (non-singular integrand)
        {
            "base fun": lambda t: 1,
            "weight args": {"singularpoint": 0, "algpower": 1, "logpower": False},
            "interval": [0, np.pi],
            "val": np.pi**2 / 2,
        },
        # problem 6 - error using incorrect power argument for the algebraic term
        {
            "base fun": lambda t: 1,
            "weight args": {"singularpoint": 0, "algpower": -1, "logpower": False},
            "interval": [0, 1],
            "val": np.inf,
        },
        # problem 7 - break points
        {
            "base fun": lambda t: (t - 2) ** 2,
            "weight args": {"singularpoint": 1, "algpower": -1 / 2, "logpower": True},
            "interval": [1, 2, 3],
            "val": 14 * np.sqrt(2) / 225 * (15 * np.log(2) - 46),
        },
    ]

    def _base(self, i, tol, fudge=1, **kwargs):
        prob = self.example_problems[i]
        status = kwargs.pop("status", 0)
        weightedrule = AlgLogLeftSingularRule(**prob["weight args"])
        y, info = adaptive_quadrature(
            fixed_quadcc,
            prob["base fun"],
            prob["interval"],
            epsabs=tol,
            epsrel=tol,
            weight_function=weightedrule,
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
        """Test for example problem #0."""
        self._base(5, 1e-4)
        self._base(5, 1e-8)
        self._base(5, 1e-12)

    def test_prob06(self):
        """Test for example problem #6."""
        self._base(6, 1e-4)
        self._base(6, 1e-8)
        self._base(6, 1e-12)

    def test_prob7(self):
        """Test for example problem #7."""
        self._base(7, 1e-4)
        self._base(7, 1e-8)
        self._base(7, 1e-12)
