"""Tests for quadax utility functions.

The interval mapping itself. What the map is worth once a solver is wrapped around it is
checked by ``TestIntervalScaling`` in ``tests/test_adaptive.py``, and that the limits of
an unbounded interval can be differentiated end to end by ``test_infinite_limits`` in
``tests/test_derivatives.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

from quadax.utils import _map_ainf, map_interval

config.update("jax_enable_x64", True)


class TestMapping:
    """How the integrand reaches the reference nodes decides how accurate they are."""

    def test_a_finite_interval_is_left_where_it_is(self):
        """No normalization to [-1, 1], so only one affine map reaches the nodes."""
        _, interval_t = map_interval(lambda x: x, jnp.array([2.0, 3.5, 5.0]))
        np.testing.assert_array_equal(np.asarray(interval_t), [2.0, 3.5, 5.0])
        # the one caller that needs the reference interval can still ask for it
        _, interval_r = map_interval(
            lambda x: x, jnp.array([2.0, 3.5, 5.0]), reference=True
        )
        np.testing.assert_allclose(np.asarray(interval_r), [-1.0, 0.0, 1.0])

    def test_an_infinite_interval_is_mapped(self):
        """There is no way to subdivide an unbounded interval in place."""
        for iv in ([0.0, jnp.inf], [-jnp.inf, 0.0], [-jnp.inf, jnp.inf]):
            _, interval_t = map_interval(lambda x: x, jnp.array(iv))
            np.testing.assert_allclose(np.asarray(interval_t), [-1.0, 1.0], atol=0)

    def test_the_infinite_map_keeps_the_distance_from_its_finite_end(self):
        """``_map_ainf`` must not form that distance out of a cancellation.

        ``a - 1 + 2/(1-t)`` is the difference of two numbers near 1, so a distance ``d``
        survives it with only ``d/eps`` of its value - the outermost node came out a
        factor of two wrong. The algebraically equal ``(1+t)/(1-t)`` comes back
        correctly rounded at every scale.
        """
        d = np.array([2.0**-k for k in (10, 20, 30, 40, 52)])
        t = np.float64(-1.0) + d
        x, _ = _map_ainf(jnp.asarray(t), jnp.asarray(0.0), jnp.asarray(jnp.inf))
        ref = (1 + np.longdouble(t)) / (1 - np.longdouble(t))
        np.testing.assert_allclose(
            np.asarray(x, dtype=np.float64),
            np.asarray(ref, dtype=np.float64),
            rtol=np.finfo(np.float64).eps,
            atol=0,
        )

    @pytest.mark.parametrize(
        "iv",
        [
            [0.5, jnp.inf],
            [-jnp.inf, 1.0],
            [-jnp.inf, jnp.inf],
            [-jnp.inf, 0.3, 2.0],
            [0.0, 1.0],
        ],
        ids=["a_inf", "ninf_b", "ninf_inf", "breakpoint", "finite"],
    )
    def test_the_map_is_differentiable_in_both_modes(self, iv):
        """An infinite limit must not leave a nan behind in reverse mode."""
        iv = jnp.array(iv)
        limits = lambda v: map_interval(lambda x: x, v)[1]  # noqa: E731
        rev = np.asarray(jax.jacrev(limits)(iv))
        assert np.isfinite(rev).all()
        np.testing.assert_array_equal(rev, np.asarray(jax.jacfwd(limits)(iv)))
