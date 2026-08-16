"""Tests for quadax utility functions."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

import quadax
from quadax import quadcc, quadgk, quadts
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

    @pytest.mark.parametrize("scale", [1e-8, 1e-3, 1.0, 1e3, 1e8])
    @pytest.mark.parametrize("method", [quadgk, quadcc, quadts])
    def test_the_result_does_not_depend_on_the_scale_of_the_interval(
        self, method, scale
    ):
        """The width floor is relative, so rescaling the problem rescales the answer."""
        s = float(scale)
        y, info = method(
            lambda t: jnp.exp(-((t / s) ** 2)),
            jnp.array([0.0, 3 * s]),
            epsabs=0.0,
            epsrel=1e-10,
        )
        assert int(info.status) == 0, quadax.STATUS[int(info.status)]
        np.testing.assert_allclose(float(y) / s, 0.8862073482595214, rtol=1e-10)

    @pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
    def test_a_singularity_at_the_origin_is_scale_invariant(self, scale):
        """``x**-1/2`` on ``[0, s]``: the case a single affine map makes exact.

        ``0 + halflength*(1 + x_node)`` has nothing to cancel, so every abscissa is the
        correctly rounded distance from the singularity however deep the subdivision
        goes, and the answer no longer depends on where the interval sits.
        """
        s = float(scale)
        y, _ = quadgk(lambda t: t**-0.5, jnp.array([0.0, s]), epsabs=0.0, epsrel=1e-12)
        np.testing.assert_allclose(float(y), 2 * np.sqrt(s), rtol=1e-8)
