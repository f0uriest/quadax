"""Tests for quadax utility functions.

The interval and box mappings themselves. What a map is worth once a solver is wrapped
around it is checked by ``TestIntervalScaling`` in ``tests/test_adaptive.py`` and
``TestInfiniteBox`` in ``tests/test_fixed_cubature.py``, and that the limits of an
unbounded interval can be differentiated end to end by ``test_infinite_limits`` in
``tests/test_derivatives.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

from quadax._utils import (
    _map_tail,
    apply_mapping,
    box_corners,
    map_box,
    map_interval,
    mapping_of,
    resolve_dtypes,
    wrap_func,
)

config.update("jax_enable_x64", True)


class TestMapping:
    """How the integrand reaches the reference nodes decides how accurate they are."""

    def test_a_finite_interval_is_left_where_it_is(self):
        """No normalization to [-1, 1], so only one affine map reaches the nodes."""
        _, interval_t = map_interval(lambda x: x, jnp.array([2.0, 3.5, 5.0]))
        np.testing.assert_array_equal(np.asarray(interval_t), [2.0, 3.5, 5.0])

    @pytest.mark.parametrize(
        "iv, want",
        [
            ([0.0, jnp.inf], [-0.5, 0.5]),
            ([-jnp.inf, 0.0], [-0.5, 0.5]),
            ([-jnp.inf, jnp.inf], [-1.5, 1.5]),
        ],
        ids=["a_inf", "ninf_b", "ninf_inf"],
    )
    def test_an_infinite_interval_is_mapped(self, iv, want):
        """There is no way to subdivide an unbounded interval in place.

        Each unbounded end is given a fixed stretch of reference coordinate to fit into,
        measured in the unit the map works in, which without breakpoints is one. A
        single tail is given that unit and a pair sharing a junction gets half as much
        again, the two profiles reaching one unit along the axis at different widths.

        None of these has a finite stretch to keep a coordinate for, so the domain is
        placed on the origin rather than on the caller's limits.
        """
        _, interval_t = map_interval(lambda x: x, jnp.array(iv))
        np.testing.assert_allclose(np.asarray(interval_t), want, atol=0)

    @pytest.mark.parametrize("anchored", [True, False], ids=["anchored", "unanchored"])
    def test_a_tail_keeps_both_of_its_distances(self, anchored):
        """Neither end of a tail may be formed out of a cancellation.

        A tail spans the junction it hangs off and the boundary carrying the infinity,
        and the two distances that locate a node between them sum to the width, so
        either could be derived from the other. Neither may be: each is the one that
        keeps its bits where the other has run out of them, and a node placed by
        subtracting a distance of order the width off the width keeps only ``d/eps`` of
        a small ``d``. The end that decides the answer is the far one, an integrand
        reached through a tail being singular there and nowhere else, and a node there
        would come out a factor of two wrong or land on the boundary itself.
        """
        w = np.float64(1.5 if not anchored else 1.0)
        anc = jnp.asarray(anchored)
        scales = np.array([2.0**-k for k in (10, 20, 30, 40, 52)])

        def reference(d, rest):
            s = np.longdouble(d) / np.longdouble(w)
            grow = np.longdouble(w) / np.longdouble(rest)
            return np.longdouble(d) * (grow if anchored else grow / (1 + s))

        for d, rest in ((w * scales, w * (1 - scales)), (w * (1 - scales), w * scales)):
            got, _ = _map_tail(jnp.asarray(d), jnp.asarray(rest), jnp.asarray(w), anc)
            np.testing.assert_allclose(
                np.asarray(got, dtype=np.float64),
                np.asarray(reference(d, rest), dtype=np.float64),
                rtol=4 * np.finfo(np.float64).eps,
                atol=0,
            )

    def test_only_the_unbounded_ends_are_mapped(self):
        """A breakpoint beside an infinite limit leaves the bounded part alone.

        The junctions are the outermost finite points and each tail is one unit of the
        map's own scale beyond, so the bounded sub-intervals arrive at the rule in the
        coordinate they were written in. The tail is glued on with unit derivative, so
        the mapped integrand has no break at the junction to be resolved - and the
        junction is a point of ``interval`` anyway, so no sub-interval straddles it.
        """
        fun, interval_t = map_interval(lambda x: x, jnp.array([0.0, 1.0, jnp.inf]))
        np.testing.assert_array_equal(np.asarray(interval_t), [0.0, 1.0, 2.0])
        # x is its own image below the junction and pulls away from it above
        mapping = mapping_of(fun)
        inside = jnp.array([0.25, 0.5, 1.0])
        np.testing.assert_array_equal(
            np.asarray(apply_mapping(mapping, inside)[0]), np.asarray(inside)
        )
        # halfway along the tail's reference span is exactly one unit past the junction,
        # which is what ties the node placement to the widest sub-interval
        assert float(apply_mapping(mapping, jnp.asarray(1.5))[0]) == 2.0
        assert not np.isfinite(float(apply_mapping(mapping, jnp.asarray(2.0))[0]))
        # the Jacobian either side of the junction, which the glue makes continuous
        eps = 1e-8
        below = float(apply_mapping(mapping, jnp.asarray(1.0 - eps))[1])
        above = float(apply_mapping(mapping, jnp.asarray(1.0 + eps))[1])
        np.testing.assert_allclose([below, above], 1.0, rtol=1e-7)

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


class TestMapBox:
    """The box map is the one dimensional one applied to each axis on its own.

    Two things have no one dimensional counterpart to inherit correctness from: that the
    axes really are independent, so one box may mix finite with infinite ones, and that
    the Jacobian is the product of theirs. Both are pinned here. What the map is worth
    once a rule is wrapped around it is checked by ``TestInfiniteBox`` in
    ``tests/test_fixed_cubature.py``.
    """

    def test_axes_are_mapped_independently(self):
        """All four cases in one box, and a finite axis is left where it is."""
        interval = jnp.array(
            [[2.0, 3.5], [0.0, jnp.inf], [-jnp.inf, 1.0], [-jnp.inf, jnp.inf]]
        )
        _, interval_t = map_box(lambda x: x, interval)
        a_t, b_t = box_corners(interval_t)
        # the finite axis where it was, and each unbounded axis given its own tails and
        # placed on the origin, having no finite stretch of its own to keep
        np.testing.assert_array_equal(np.asarray(a_t), [2.0, -0.5, -0.5, -1.5])
        np.testing.assert_array_equal(np.asarray(b_t), [3.5, 0.5, 0.5, 1.5])

    def test_an_array_means_what_iterating_it_means(self):
        """The convenience form must not be able to denote a different box.

        An ``(ndim, 2)`` array and the sequence of its rows are dispatched by different
        branches, so that they agree is a property of the code rather than of the shape.
        """
        interval = jnp.array([[0.0, jnp.inf], [-jnp.inf, 2.0], [1.0, 4.0]])
        fun = lambda x: jnp.prod(jnp.exp(-jnp.abs(x)))  # noqa: E731
        _, from_array = map_box(fun, interval)
        _, from_rows = map_box(fun, list(interval))
        for u, v in zip(from_array, from_rows, strict=True):
            np.testing.assert_array_equal(np.asarray(u), np.asarray(v))

    def test_axes_may_carry_different_numbers_of_breakpoints(self):
        """Ragged is the point of the sequence form: nothing is padded into existence.

        Breakpoints ride through the map so that whatever subdivides the box afterwards
        can start from them, and each lands where the axis' own map sends it.
        """
        interval = [
            jnp.array([0.0, 1.0, 3.0, jnp.inf]),
            jnp.array([-jnp.inf, jnp.inf]),
            jnp.array([2.0, 2.5, 3.5]),
        ]
        _, interval_t = map_box(lambda x: x, interval)
        assert [len(v) for v in interval_t] == [4, 2, 3]
        # each breakpoint is where the one dimensional map of that axis puts it
        for axis, mapped in zip(interval, interval_t, strict=True):
            _, ref = map_interval(lambda x: x, axis)
            np.testing.assert_allclose(
                np.asarray(mapped), np.asarray(ref), rtol=1e-15, atol=0
            )

    def test_a_breakpoint_outside_its_axis_is_pulled_to_the_endpoint(self):
        """Which leaves a sub box of zero width, as it does in one dimension."""
        _, interval_t = map_box(
            lambda x: x, [jnp.array([0.0, 5.0, 2.0]), jnp.array([0.0, 1.0])]
        )
        np.testing.assert_array_equal(np.asarray(interval_t[0]), [0.0, 2.0, 2.0])

    def test_the_jacobian_is_the_product_over_axes(self):
        """Checked against the same map applied one axis at a time.

        On a separable integrand the mapped value must factor exactly, which pins the
        nodes and the volume element together.
        """
        interval = jnp.array(
            [[0.0, jnp.inf], [-jnp.inf, 2.0], [-jnp.inf, jnp.inf], [1.0, 4.0]]
        )
        # Interior to every axis's own reference domain: the boundary itself is where
        # the map is meant to return an infinity, which is not what is under test here.
        t = jnp.array([0.3, -0.25, 0.7, 2.0])
        fun = lambda x: jnp.prod(jnp.exp(-jnp.abs(x)))  # noqa: E731
        fun_t, _ = map_box(fun, interval)
        ref = 1.0
        for k in range(len(interval)):
            fun_1d, _ = map_interval(lambda x: jnp.exp(-jnp.abs(x)), interval[k])
            ref = ref * fun_1d(t[k])
        np.testing.assert_allclose(float(fun_t(t)), float(ref), rtol=1e-14, atol=0)

    @pytest.mark.parametrize("reversed_axes", [(), (0,), (1,), (0, 1)])
    def test_reversing_an_axis_flips_one_sign(self, reversed_axes):
        """Every reversed axis flips the integral, collected into one factor."""
        interval = np.array([[0.0, np.inf], [-np.inf, 2.0]])
        flipped = interval.copy()
        for k in reversed_axes:
            flipped[k] = flipped[k][::-1]
        fun = lambda x: jnp.prod(jnp.exp(-jnp.abs(x)))  # noqa: E731
        t = jnp.array([0.2, -0.4])
        ref, _ = map_box(fun, jnp.asarray(interval))
        got, _ = map_box(fun, jnp.asarray(flipped))
        assert float(got(t)) == (-1.0) ** len(reversed_axes) * float(ref(t))

    def test_the_map_is_differentiable_in_both_modes(self):
        """An infinite limit must not leave a nan behind in reverse mode.

        The per axis branches disagree about which endpoint they use, so a branch that
        is not taken can still be the one that produces a nan.
        """
        interval = jnp.array(
            [[0.0, jnp.inf], [-jnp.inf, 2.0], [-jnp.inf, jnp.inf], [1.0, 4.0]]
        )
        # Interior to every axis's own reference domain: the boundary itself is where
        # the map is meant to return an infinity, which is not what is under test here.
        t = jnp.array([0.3, -0.25, 0.7, 2.0])

        def value(limits):
            fun_t, interval_t = map_box(lambda x: jnp.sum(x**2), limits)
            return fun_t(t) + sum(jnp.sum(v) for v in interval_t)

        rev = np.asarray(jax.jacrev(value)(interval))
        assert np.isfinite(rev).all()
        # The two modes accumulate the same products in opposite orders, so they agree
        # to rounding rather than bit for bit, and the tail's derivative is large enough
        # near a boundary to make that a few ulp. What is under test is that neither
        # picks up a nan from a region of the map it is not in.
        np.testing.assert_allclose(
            rev, np.asarray(jax.jacfwd(value)(interval)), rtol=1e-13, atol=0
        )

    def test_limits_that_are_not_a_box_are_rejected(self):
        """The two forms are told apart by type, so each can say what it wanted."""
        with pytest.raises(ValueError, match=r"shape \(ndim, 2\)"):
            map_box(lambda x: x, jnp.zeros((2, 3)))
        with pytest.raises(TypeError, match="list or tuple"):
            map_box(lambda x: x, "nope")
        with pytest.raises(ValueError, match="at least its two endpoints"):
            map_box(lambda x: x, [jnp.zeros(1)])
        with pytest.raises(TypeError, match="must be real"):
            map_box(lambda x: x, [jnp.zeros(2, dtype=complex)])


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

        Splitting the point's own axis instead would reduce over the wrong thing, and
        both comparisons below would be out by the size of the point rather than by
        rounding.
        """
        x = jnp.asarray(np.random.default_rng(0).normal(size=(23, ndim)))
        f = lambda p: jnp.sum(p**2)  # noqa: E731
        ref = wrap_func(f, (), x.dtype, ndim=ndim)(x)
        got = wrap_func(f, (), x.dtype, batch_size=batch_size, ndim=ndim)(x)
        assert ref.shape == got.shape == (23,)
        # Batching regroups the points without touching how any one of them is
        # evaluated, so the arithmetic per point is the same sum of ``ndim`` terms. It
        # is not the same *program* though: a batch is a differently shaped call, and
        # the compiler is free to associate the sum differently in it, which is worth
        # an ulp per term summed and no more.
        eps = float(np.finfo(x.dtype).eps)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(ref), rtol=ndim * eps, atol=0
        )
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


class TestAbscissaShape:
    """What one abscissa looks like, which the box map and the 1D map disagree on.

    ``resolve_dtypes`` and ``wrap_func`` both probe the integrand before anything else
    happens, so both have to be told whether it takes a scalar or a vector. The scalar
    path is what every one dimensional routine goes through and is pinned here against
    being disturbed by the vector one.
    """

    def test_the_scalar_probe_is_the_default(self):
        """Omitting ``ndim`` probes with a scalar, as a 1D quadrature needs."""
        seen = []
        fun = lambda x: seen.append(jnp.shape(x)) or x  # noqa: E731
        dtypes = resolve_dtypes(jnp.array([0.0, 1.0]), fun)
        assert seen == [()]
        assert dtypes.xtype == jnp.float64

    def test_a_vector_probe_is_asked_for_by_ndim(self):
        """``ndim`` makes the probe a point of that length."""
        seen = []
        fun = lambda x: seen.append(jnp.shape(x)) or jnp.sum(x)  # noqa: E731
        resolve_dtypes(jnp.array([[0.0, 1.0]] * 3), fun, ndim=3)
        assert seen == [(3,)]

    def test_ragged_per_axis_limits_set_the_working_precision(self):
        """One array per axis, of differing lengths, resolves to the widest dtype."""
        axes = [
            jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32),
            jnp.array([0.0, 1.0], dtype=jnp.float64),
        ]
        dtypes = resolve_dtypes(axes, lambda x: jnp.sum(x), ndim=2)
        assert dtypes.xtype == jnp.float64

    def test_integer_box_limits_are_promoted(self):
        """Limits are differentiated with respect to, so they cannot stay integers.

        ``[[0, 1], [0, 1]]`` is the form a caller writes first, and left integer AD
        would treat it as static metadata rather than as something to differentiate.
        """
        _, interval_t = map_box(lambda x: jnp.sum(x), [[0, 1], [0, 1]])
        for axis in interval_t:
            assert jnp.issubdtype(axis.dtype, jnp.floating)


def test_renamed_submodules():
    """Submodules made private still import by their old names, with a warning."""
    # The old names exist only at runtime and are hidden from static type checkers.
    import quadax.romberg  # pyright: ignore[reportMissingImports]

    with pytest.warns(DeprecationWarning, match="quadax.adaptive"):
        from quadax.adaptive import (  # pyright: ignore[reportMissingImports]
            _adaptive_solve,
            quadgk,
        )
    assert quadgk is quadax.quadgk
    assert callable(_adaptive_solve)
    # an old module name that is also a public function still resolves to the function
    with pytest.warns(DeprecationWarning, match="quadax.romberg"):
        from quadax.romberg import rombergts  # pyright: ignore[reportMissingImports]
    assert rombergts is quadax.rombergts
    assert quadax.romberg is quadax._romberg.romberg
    with pytest.warns(DeprecationWarning, match="quadax.utils"):
        utils = quadax.utils  # pyright: ignore[reportAttributeAccessIssue]
        assert utils.map_interval is map_interval
