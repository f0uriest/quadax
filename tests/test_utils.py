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

from quadax.utils import _map_ainf, map_interval, wrap_func

config.update("jax_enable_x64", True)


class TestMapping:
    """How the integrand reaches the reference nodes decides how accurate they are."""

    def test_a_finite_interval_is_left_where_it_is(self):
        """No normalization to [-1, 1], so only one affine map reaches the nodes."""
        _, interval_t = map_interval(lambda x: x, jnp.array([2.0, 3.5, 5.0]))
        np.testing.assert_array_equal(np.asarray(interval_t), [2.0, 3.5, 5.0])

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


class TestWrapFunc:
    """One abscissa is a scalar in 1D and a point in n dimensions.

    ``wrap_func`` decides which by ``ndim``, and everything that batches or masks has to
    agree with it about where the looped axis ends and the point begins. The 1D case is
    what every quadrature rule in the package goes through, so it is pinned here
    alongside the vector one rather than left to the rules' own tests.
    """

    def test_a_scalar_abscissa_is_unchanged_by_the_ndim_option(self):
        """``ndim=None`` is the 1D path, and stays exactly what it was."""
        x = jnp.linspace(0.0, 1.0, 17)
        f = lambda t: jnp.stack([jnp.exp(t), t**2])  # noqa: E731
        got = wrap_func(f, (), x.dtype)(x)
        assert got.shape == (17, 2)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(jax.vmap(f)(x)), rtol=0, atol=0
        )

    @pytest.mark.parametrize("ndim", [1, 3])
    @pytest.mark.parametrize("batch_size", [1, 5, 23, 100])
    def test_a_vector_abscissa_batches_along_the_looped_axis(self, ndim, batch_size):
        """Only the leading axis is split into batches; the point stays whole.

        Compared against the unbatched wrapper rather than against ``vmap``: batching
        regroups the points without touching how any one of them is evaluated, so it
        has to agree bit for bit, where ``vmap`` reduces a point's own axis in a
        different order and lands an ulp away.
        """
        x = jnp.asarray(np.random.default_rng(0).normal(size=(23, ndim)))
        f = lambda p: jnp.sum(p**2)  # noqa: E731
        ref = wrap_func(f, (), x.dtype, ndim=ndim)(x)
        got = wrap_func(f, (), x.dtype, batch_size=batch_size, ndim=ndim)(x)
        assert ref.shape == got.shape == (23,)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
        # and the values are right, to within that reordering
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(jax.vmap(f)(x)), rtol=1e-15, atol=0
        )

    def test_a_non_finite_value_is_masked_per_point(self):
        """A point is bad or not as a whole, whatever the integrand's own shape."""
        x = jnp.asarray([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
        f = lambda p: jnp.stack([jnp.log(p[0]), p[1]])  # noqa: E731
        got = np.asarray(wrap_func(f, (), x.dtype, ndim=2)(x))
        # the first component is masked where it is not finite, the second survives
        np.testing.assert_array_equal(got[:, 0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(got[:, 1], [1.0, 1.0, -1.0])

    def test_the_safe_mask_is_differentiable_at_a_vector_abscissa(self):
        """Reverse mode must not pick up the nan the mask exists to remove.

        The substitute abscissa is a whole point rather than a scalar here, so a
        version that borrowed one coordinate would broadcast the wrong shape or
        linearize at a point the integrand was never seen to be finite at.
        """
        x = jnp.asarray([[1.0, 2.0], [-1.0, 2.0], [3.0, 4.0]])

        def loss(c):
            f = lambda p: c * jnp.log(p[0]) * p[1]  # noqa: E731
            return jnp.sum(wrap_func(f, (), x.dtype, safe=True, ndim=2)(x))

        grad = float(jax.grad(loss)(2.0))
        assert np.isfinite(grad)
        # only the two points with a positive first coordinate contribute
        np.testing.assert_allclose(grad, np.log(1.0) * 2 + np.log(3.0) * 4, rtol=1e-14)
