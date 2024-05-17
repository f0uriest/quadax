"""Tests for modified Chebyshev moments."""

import numpy as np
import pytest
from jax import config

from quadax.quad_weights import _chebmom_alg_log_plus, _chebmom_alg_plus

config.update("jax_enable_x64", True)

algmomentdata = [(2, 1, [2, 2 / 3, -2 / 3]), (0, 2, [8 / 3])]


@pytest.mark.parametrize("K, power, expected", algmomentdata)
def test_chebmom_alg_plus(K, power, expected):
    """Test Chebyshev algebraic moments."""
    assert np.allclose(_chebmom_alg_plus(K, power), expected, rtol=1e-12, atol=1e-12)


alglogmomentdata = [(2, 1, [-1, 1 / 9, 5 / 9]), (0, -1 / 2, [-4 * np.sqrt(2)])]


@pytest.mark.parametrize("K, power, expected", alglogmomentdata)
def test_chebmom_alg_log_plus(K, power, expected):
    """Test Chebyshev algebraic-logarithmic moments."""
    assert np.allclose(
        _chebmom_alg_log_plus(K, power), expected, rtol=1e-12, atol=1e-12
    )
