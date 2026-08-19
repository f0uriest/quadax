"""Tests for differentiating quadrature, and for the adjoints that control how.

Covers both halves of the same subject: that the derivatives are *correct* (checked
against finite differences and, where available, analytic values), and that each
:class:`~quadax.AbstractAdjoint` implementation computes them equivalently while
surviving the usual JAX transforms.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

from quadax import (
    DirectAdjoint,
    GaussKronrodRule,
    LeibnizAdjoint,
    quadcc,
    quadgk,
    quadts,
    romberg,
    rombergts,
)
from quadax.adaptive import _adaptive_solve
from quadax.adjoint import (
    _frozen_replay,
    _replay_solve,
    _UnrolledDirectAdjoint,
    build_integrand,
    closure_convert,
)

from .problems import ULP_ATOL, ULP_RTOL

config.update("jax_enable_x64", True)


adaptive_methods = [quadgk, quadcc, quadts]
romberg_methods = [romberg, rombergts]
all_methods = adaptive_methods + romberg_methods

# The three adaptive routines differ only in the local rule applied on each
# sub-interval; the subdivision loop and the adjoints wrapped around it are the same
# code, so tests of that machinery run through one of them rather than all three. The
# checks on the derivative itself keep the full list, since there each rule's own
# accuracy is part of what is under test.
machinery_methods = [quadgk]

# problems exercising the paths that differ: plain, interior breakpoints, an infinite
# limit, and a vector valued integrand.
example_problems = [
    {
        "fun": lambda t, c: jnp.log(c * t).squeeze(),
        "interval": [1.0, 3.0],
        "args": (np.array([3.12]),),
    },
    {
        "fun": lambda t, c: jnp.log(c * t).squeeze(),
        "interval": [1.0, 1.7, 2.2, 3.0],
        "args": (np.array([3.12]),),
    },
    {
        "fun": lambda t, m_s: jnp.exp(-((t - m_s[0]) ** 2) / m_s[1] ** 2).squeeze(),
        "interval": [-2.0, 3.0],
        "args": (np.array([1.23, 0.67]),),
    },
    {
        "fun": lambda t, m_s: jnp.exp(-((t - m_s[0]) ** 2) / m_s[1] ** 2).squeeze(),
        "interval": [-jnp.inf, jnp.inf],
        "args": (np.array([1.23, 0.67]),),
    },
    {  # vector valued
        "fun": lambda t, c: jnp.array([jnp.sin(c[0] * t), jnp.cos(c[1] * t)]).squeeze(),
        "interval": [0.0, 2.0],
        "args": (np.array([1.1, 0.7]),),
    },
]

# the two problems with finite limits and no breakpoints, used where a finite difference
# reference is wanted and the quadrature must be well converged
SMOOTH_PROBLEMS = [0, 2]


def finite_difference(f, x, eps=1e-8):
    """Util for 2nd order centered finite differences."""
    x0 = np.atleast_1d(x).squeeze()
    f0 = f((x0,))
    J = np.zeros((np.atleast_1d(f0).size, x0.size))
    h = np.maximum(1.0, np.abs(x0)) * eps
    h_vecs = np.diag(np.atleast_1d(h))
    for i in range(x0.size):
        x1, x2 = x0 - h_vecs[i], x0 + h_vecs[i]
        dx = (x2[i] - x1[i]) if x0.ndim else (x2 - x1)
        J[:, i] = ((f((x2,)) - f((x1,))) / dx).flatten()
    return np.ravel(J) if J.shape[0] == 1 else J


@pytest.mark.parametrize("quad", all_methods)
@pytest.mark.parametrize("i", SMOOTH_PROBLEMS)
class TestDefaultAdjointAgainstFiniteDifference:
    """Every method's default adjoint must agree with finite differences.

    This is the end to end check on the derivative itself, independent of which adjoint
    is used internally, so it is deliberately parametrized over all five entry points
    rather than duplicated per method.
    """

    def _integrate(self, quad, i):
        prob = example_problems[i]
        return lambda args: quad(prob["fun"], prob["interval"], args)[0]

    def test_jacfwd(self, quad, i):
        """Forward mode matches finite differences."""
        f = self._integrate(quad, i)
        np.testing.assert_allclose(
            finite_difference(f, example_problems[i]["args"]),
            np.asarray(jax.jacfwd(f)(example_problems[i]["args"])[0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_jacrev(self, quad, i):
        """Reverse mode matches finite differences."""
        f = self._integrate(quad, i)
        np.testing.assert_allclose(
            finite_difference(f, example_problems[i]["args"]),
            np.asarray(jax.jacrev(f)(example_problems[i]["args"])[0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_modes_agree(self, quad, i):
        """Forward and reverse agree far more tightly than either matches FD.

        They are transposes of the same linear functional, so they should agree to
        rounding rather than merely to the accuracy of the quadrature.
        """
        f = self._integrate(quad, i)
        args = example_problems[i]["args"]
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(f)(args)[0]),
            np.asarray(jax.jacrev(f)(args)[0]),
            rtol=1e-14,
            atol=1e-14,
        )


class TestDirectAdjointEquivalence:
    """DirectAdjoint must reproduce differentiating through the loop.

    This is the safety gate for reimplementing the default adjoint: the old
    implementation is kept privately as ``_UnrolledDirectAdjoint`` purely so that the
    new one can be compared against it.
    """

    def _jacs(self, quad, prob, wrt_interval, transform):
        interval = jnp.asarray(prob["interval"])
        args = prob["args"]

        def make(adjoint):
            if wrt_interval:
                return lambda v: quad(prob["fun"], v, args, adjoint=adjoint)[0]
            return lambda a: quad(prob["fun"], interval, a, adjoint=adjoint)[0]

        x = interval if wrt_interval else args
        old = transform(make(_UnrolledDirectAdjoint()))(x)
        new = transform(make(DirectAdjoint()))(x)
        return np.asarray(jax.tree.leaves(old)[0]), np.asarray(jax.tree.leaves(new)[0])

    @pytest.mark.parametrize("quad", machinery_methods)
    @pytest.mark.parametrize("i", range(len(example_problems)))
    def test_jacfwd_wrt_args(self, quad, i):
        """Forward mode w.r.t. args matches unrolled loop."""
        old, new = self._jacs(quad, example_problems[i], False, jax.jacfwd)
        np.testing.assert_allclose(old, new, rtol=ULP_RTOL, atol=ULP_ATOL)

    @pytest.mark.parametrize("quad", machinery_methods)
    @pytest.mark.parametrize("i", range(len(example_problems)))
    def test_jacfwd_wrt_interval(self, quad, i):
        """Forward mode w.r.t. limits and breakpoints matches unrolled loop."""
        old, new = self._jacs(quad, example_problems[i], True, jax.jacfwd)
        np.testing.assert_allclose(old, new, rtol=ULP_RTOL, atol=ULP_ATOL)

    @pytest.mark.parametrize("quad", machinery_methods)
    @pytest.mark.parametrize("i", range(len(example_problems)))
    def test_jacrev_wrt_args(self, quad, i):
        """Reverse mode w.r.t. args matches unrolled loop."""
        old, new = self._jacs(quad, example_problems[i], False, jax.jacrev)
        np.testing.assert_allclose(old, new, rtol=ULP_RTOL, atol=ULP_ATOL)

    @pytest.mark.parametrize("quad", machinery_methods)
    @pytest.mark.parametrize("i", range(len(example_problems)))
    def test_jacrev_matches_own_jacfwd(self, quad, i):
        """Direct adjoint's forward and reverse modes agree with each other."""
        prob = example_problems[i]
        interval = jnp.asarray(prob["interval"])
        f = lambda a: quad(prob["fun"], interval, a, adjoint=DirectAdjoint())[0]
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(f)(prob["args"])[0]),
            np.asarray(jax.jacrev(f)(prob["args"])[0]),
            rtol=ULP_RTOL,
            atol=ULP_ATOL,
        )


class TestBreakpointBoundaryTerm:
    """A breakpoint tied to a moving jump picks up the jump term.

    ``int_0^3 f`` with ``f = 1`` below the jump and ``5`` above it is ``15 - 4s``, so
    the derivative is ``-4``. The integrand's *values* do not depend on ``s`` at all
    here, only the position of the jump does, so the whole of that ``-4`` is the jump
    term and the test fails outright if it is dropped. ``test_jump_and_args_together``
    is the companion where both halves are non-zero and have opposite signs.

    Tying the breakpoint to the same parameter as the jump is the supported version,
    see the Notes on the adjoints for what the alternatives give.
    """

    fun = staticmethod(lambda t, c: jnp.where(t < c[0], 1.0, 5.0).squeeze())

    @staticmethod
    def tied(quad, adjoint):
        """One ``s`` positions both the discontinuity and the breakpoint."""
        return lambda s: quad(
            TestBreakpointBoundaryTerm.fun,
            jnp.stack([jnp.zeros_like(s), s, 3.0 * jnp.ones_like(s)]),
            (jnp.atleast_1d(s),),
            adjoint=adjoint,
        )[0]

    @pytest.mark.parametrize("quad", adaptive_methods)
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize(
        "adjoint", [DirectAdjoint(), LeibnizAdjoint()], ids=["direct", "leibniz"]
    )
    def test_tied_breakpoint(self, quad, transform, adjoint):
        """The jump moves with the breakpoint, and the total comes out right."""
        f = self.tied(quad, adjoint)
        x = jnp.array(1.5)
        np.testing.assert_allclose(float(f(x)), 15.0 - 4.0 * x, atol=1e-12)
        np.testing.assert_allclose(float(transform(f)(x)), -4.0, atol=1e-9)

    @pytest.mark.parametrize("quad", adaptive_methods)
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize(
        "adjoint", [DirectAdjoint(), LeibnizAdjoint()], ids=["direct", "leibniz"]
    )
    def test_outer_limits(self, quad, transform, adjoint):
        """The outer limits give ``f(b) db - f(a) da``, with the breakpoint held fixed.

        Differentiated on their own, so this says nothing about the interior entry.
        """
        f = lambda ab: quad(  # noqa: E731
            self.fun,
            jnp.stack([ab[0], jnp.array(1.5), ab[1]]),
            (jnp.array([1.5]),),
            adjoint=adjoint,
        )[0]
        got = np.asarray(transform(f)(jnp.array([0.0, 3.0])))
        np.testing.assert_allclose(got, np.array([-1.0, 5.0]), atol=1e-12)

    @pytest.mark.parametrize("quad", adaptive_methods)
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize(
        "adjoint", [DirectAdjoint(), LeibnizAdjoint()], ids=["direct", "leibniz"]
    )
    def test_jump_and_args_together(self, quad, transform, adjoint, request):
        """Both halves non-zero, and of opposite sign, so neither can hide the other.

        ``f = s`` below the jump and ``2s`` above it, over ``[-1, 1]`` with the jump at
        ``s``, integrates to ``3s - s**2``. The integrand's own dependence on ``s``
        contributes ``+2.7`` through ``args`` and the jump contributes ``-0.3`` through
        the breakpoint; the answer is ``2.4``, so dropping or double-counting either one
        is visible.

        ``quadcc`` under :class:`DirectAdjoint` is off by 4e-4 at every tolerance.
        Clenshaw-Curtis places a node on the breakpoint itself, where the integrand's
        comparison resolves to one side; ``DirectAdjoint`` differentiates that
        discretization, so the node keeps its weight in the derivative. Gauss-Kronrod
        and tanh-sinh have no node there and are exact.
        """
        if quad is quadcc and isinstance(adjoint, DirectAdjoint):
            request.applymarker(
                pytest.mark.xfail(
                    strict=True, reason="closed rule samples the breakpoint itself"
                )
            )
        f = lambda s: quad(  # noqa: E731
            lambda t, z: jnp.where(t > z[0], 2 * z[0], z[0]),
            jnp.stack([-jnp.ones_like(s), s, jnp.ones_like(s)]),
            (jnp.atleast_1d(s),),
            adjoint=adjoint,
        )[0]
        x = jnp.array(0.3)
        np.testing.assert_allclose(float(f(x)), 3 * 0.3 - 0.3**2, atol=1e-12)
        np.testing.assert_allclose(float(transform(f)(x)), 2.4, atol=1e-9)

    @pytest.mark.parametrize(
        "adjoint", [DirectAdjoint(), LeibnizAdjoint()], ids=["direct", "leibniz"]
    )
    def test_second_derivative_wrt_interval(self, adjoint):
        """Differentiating the limits twice, with an interior breakpoint present.

        The boundary term is built from the breakpoint's own position, so anything it
        computes there has to survive being differentiated again. A smooth integrand
        makes the expected value easy: ``int_a^b exp`` has second derivatives
        ``-exp(a)`` and ``exp(b)`` on the diagonal, zero off it, and the breakpoint
        contributes nothing at all.
        """
        f = lambda v: quadgk(  # noqa: E731
            lambda t: jnp.exp(t), v, adjoint=adjoint, epsabs=1e-12, epsrel=1e-12
        )[0]
        v = jnp.array([0.0, 1.0, 2.0])
        got = np.asarray(jax.hessian(f)(v))
        want = np.zeros((3, 3))
        want[0, 0], want[2, 2] = -np.exp(0.0), np.exp(2.0)
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want, atol=1e-7)

    @pytest.mark.parametrize("quad", machinery_methods)
    def test_matches_unrolled(self, quad):
        """And agrees with differentiating through the loop."""
        np.testing.assert_allclose(
            float(
                jax.jacfwd(self.tied(quad, _UnrolledDirectAdjoint()))(jnp.array(1.5))
            ),
            float(jax.jacfwd(self.tied(quad, DirectAdjoint()))(jnp.array(1.5))),
            rtol=ULP_RTOL,
            atol=ULP_ATOL,
        )

    @pytest.mark.parametrize(
        "adjoint", [DirectAdjoint(), LeibnizAdjoint()], ids=["direct", "leibniz"]
    )
    def test_split_parameters_sum_to_the_tied_answer(self, adjoint):
        """One feature written as two parameters: the total is what is defined.

        Marking the jump with one parameter and positioning it with another asks for a
        Jacobian whose entries are individually meaningless -- the value of the integral
        does not depend on where the mesh is cut, so a breakpoint carries no derivative
        of its own, and the jump has to be attributed to something. Only the sum over
        the two is well posed, and it agrees with what the single tied parameter gives,
        which is the quantity that has a finite difference to compare against.
        """
        split = lambda p: quadgk(  # noqa: E731
            self.fun,
            jnp.stack([jnp.zeros_like(p[0]), p[0], 3.0 * jnp.ones_like(p[0])]),
            (jnp.atleast_1d(p[1]),),
            adjoint=adjoint,
        )[0]
        jac = np.asarray(jax.jacrev(split)(jnp.array([1.5, 1.5])))
        tied = float(jax.grad(self.tied(quadgk, adjoint))(jnp.array(1.5)))
        np.testing.assert_allclose(jac.sum(), tied, atol=1e-9)
        np.testing.assert_allclose(tied, -4.0, atol=1e-9)


class TestLeibnizAdjoints:
    """The Leibniz adjoints give the derivative its own error control."""

    @pytest.mark.parametrize("quad", machinery_methods)
    @pytest.mark.parametrize("i", SMOOTH_PROBLEMS)
    def test_forward_matches_finite_difference(self, quad, i):
        """Leibniz adjoint agrees with finite differences in forward mode."""
        prob = example_problems[i]
        interval = jnp.asarray(prob["interval"])
        f = lambda a: quad(prob["fun"], interval, a, adjoint=LeibnizAdjoint())[0]
        np.testing.assert_allclose(
            finite_difference(f, prob["args"]),
            np.asarray(jax.jacfwd(f)(prob["args"])[0]),
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("quad", machinery_methods)
    @pytest.mark.parametrize("i", SMOOTH_PROBLEMS)
    def test_reverse_matches_finite_difference(self, quad, i):
        """Leibniz adjoint agrees with finite differences in reverse mode."""
        prob = example_problems[i]
        interval = jnp.asarray(prob["interval"])
        f = lambda a: quad(prob["fun"], interval, a, adjoint=LeibnizAdjoint())[0]
        np.testing.assert_allclose(
            finite_difference(f, prob["args"]),
            np.asarray(jax.jacrev(f)(prob["args"])[0]),
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("quad", machinery_methods)
    def test_leibniz_agrees_with_direct(self, quad):
        """Leibniz agrees with DirectAdjoint on a well resolved problem, both modes."""
        prob = example_problems[0]
        interval = jnp.asarray(prob["interval"])
        mk = lambda adj: lambda a: quad(prob["fun"], interval, a, adjoint=adj)[0]
        ref = np.asarray(jax.jacfwd(mk(DirectAdjoint()))(prob["args"])[0])
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(mk(LeibnizAdjoint()))(prob["args"])[0]),
            ref,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(jax.jacrev(mk(LeibnizAdjoint()))(prob["args"])[0]),
            ref,
            rtol=1e-10,
        )


class TestRomberg:
    """Romberg supports both modes of differentiation."""

    @pytest.mark.parametrize("quad", romberg_methods)
    def test_direct_adjoint_supports_both_modes(self, quad):
        """Romberg's level loop cannot be reverse differentiated directly.

        DirectAdjoint gets around it by freezing the number of Richardson levels and
        differentiating that fixed discretization through a custom primitive. Because
        the two directions are transposes of the same linear functional, they agree to
        rounding rather than merely to quadrature accuracy. Not asserted bitwise: the
        integrand's own jvp and vjp can still differ in the last bit, which they do for
        rombergts, where the integrand also carries a tanh-sinh transform.
        """
        prob = example_problems[0]
        interval = jnp.asarray(prob["interval"])
        f = lambda a: quad(prob["fun"], interval, a)[0]
        fwd = np.asarray(jax.jacfwd(f)(prob["args"])[0])
        np.testing.assert_allclose(
            finite_difference(f, prob["args"]), fwd, atol=1e-4, rtol=1e-4
        )
        for rev in (jax.jacrev(f), jax.jit(jax.jacrev(f))):
            np.testing.assert_allclose(
                fwd, np.asarray(rev(prob["args"])[0]), rtol=ULP_RTOL, atol=ULP_ATOL
            )

    @pytest.mark.parametrize("quad", romberg_methods)
    def test_direct_adjoint_wrt_interval(self, quad):
        """The frozen-level path also handles derivatives of the limits."""
        prob = example_problems[0]
        args = prob["args"]
        f = lambda v: quad(prob["fun"], v, args)[0]
        iv = jnp.asarray(prob["interval"])
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(f)(iv)), np.asarray(jax.grad(f)(iv)), rtol=1e-10
        )

    @pytest.mark.parametrize("quad", romberg_methods)
    def test_leibniz_enables_reverse_mode(self, quad):
        """The Leibniz adjoint makes romberg reverse mode differentiable."""
        prob = example_problems[0]
        interval = jnp.asarray(prob["interval"])
        f = lambda a: quad(prob["fun"], interval, a, adjoint=LeibnizAdjoint())[0]
        np.testing.assert_allclose(
            finite_difference(f, prob["args"]),
            np.asarray(jax.jacrev(f)(prob["args"])[0]),
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("quad", romberg_methods)
    def test_adjoint_does_not_change_the_value(self, quad):
        """The adjoint controls derivatives only, never the returned value."""
        prob = example_problems[0]
        interval = jnp.asarray(prob["interval"])
        ref = quad(prob["fun"], interval, prob["args"])[0]
        for adj in [LeibnizAdjoint()]:
            y = quad(prob["fun"], interval, prob["args"], adjoint=adj)[0]
            np.testing.assert_allclose(
                np.asarray(ref), np.asarray(y), rtol=ULP_RTOL, atol=ULP_ATOL
            )


class TestTransforms:
    """Adjoints must survive the usual JAX transforms."""

    @pytest.mark.parametrize("adjoint", [DirectAdjoint(), LeibnizAdjoint()])
    def test_jit_primal(self, adjoint):
        """The value is unchanged under jit and by the choice of adjoint."""
        prob = example_problems[0]
        interval = jnp.asarray(prob["interval"])
        f = lambda a: quadgk(prob["fun"], interval, a, adjoint=adjoint)[0]
        np.testing.assert_allclose(
            np.asarray(f(prob["args"])),
            np.asarray(jax.jit(f)(prob["args"])),
            rtol=ULP_RTOL,
            atol=ULP_ATOL,
        )

    @pytest.mark.parametrize("adjoint", [DirectAdjoint(), LeibnizAdjoint()])
    def test_vmap_jacfwd(self, adjoint):
        """Forward mode works under vmap."""
        fun = lambda t, c: jnp.log(c * t).squeeze()
        interval = jnp.array([1.0, 3.0])
        f = lambda c: quadgk(fun, interval, (c,), adjoint=adjoint)[0]
        cs = jnp.array([[2.0], [3.12], [4.5]])
        got = jax.vmap(jax.jacfwd(f))(cs)
        want = jnp.stack([jax.jacfwd(f)(c) for c in cs])
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12)

    @pytest.mark.parametrize("adjoint", [DirectAdjoint(), LeibnizAdjoint()])
    def test_vmap_grad(self, adjoint):
        """Reverse mode works under vmap."""
        fun = lambda t, c: jnp.log(c * t).squeeze()
        interval = jnp.array([1.0, 3.0])
        f = lambda c: quadgk(fun, interval, (c,), adjoint=adjoint)[0].sum()
        cs = jnp.array([[2.0], [3.12], [4.5]])
        got = jax.vmap(jax.grad(f))(cs)
        want = jnp.stack([jax.grad(f)(c) for c in cs])
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12)

    def test_second_derivatives(self):
        """Direct adjoint recurses, so higher derivatives work."""
        fun = lambda t, c: jnp.exp(c[0] * t).squeeze()
        interval = jnp.array([0.0, 1.0])
        f = lambda c: quadgk(fun, interval, (c,), adjoint=DirectAdjoint())[0]
        # int_0^1 exp(c t) dt = (e^c - 1)/c;  check against the unrolled reference
        g = lambda c: quadgk(fun, interval, (c,), adjoint=_UnrolledDirectAdjoint())[0]
        c = jnp.array([0.7])
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(jax.jacfwd(f))(c)),
            np.asarray(jax.jacfwd(jax.jacfwd(g))(c)),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("adjoint", [DirectAdjoint(), LeibnizAdjoint()])
    def test_hessian(self, adjoint):
        """Reverse inside forward works, not just forward inside forward.

        ``jax.hessian`` is ``jacfwd(jacrev(...))``, which linearizes through the custom
        rule rather than nesting two JVPs, and so reaches machinery that
        ``test_second_derivatives`` does not.
        """
        fun = lambda t, c: jnp.sum(jnp.exp(-c * t))
        interval = jnp.array([0.0, 1.0])
        f = lambda c: quadgk(fun, interval, (c,), adjoint=adjoint)[0]
        # int_0^1 sum_i exp(-c_i t) dt = sum_i (1 - exp(-c_i))/c_i, so the Hessian is
        # diagonal; check against the unrolled reference, which carries no custom rule.
        g = lambda c: quadgk(fun, interval, (c,), adjoint=_UnrolledDirectAdjoint())[0]
        c = jnp.linspace(0.5, 2.0, 3)
        got = np.asarray(jax.hessian(f)(c))
        assert got.shape == (3, 3)
        np.testing.assert_allclose(got, np.asarray(jax.hessian(g)(c)), rtol=1e-6)

    def test_closed_over_values_get_gradients(self):
        """Values closed over by the integrand must not silently get zero gradients."""

        def integrate(c):
            # c is closed over, not passed through `args`
            return quadgk(lambda t: jnp.log(c * t), jnp.array([1.0, 3.0]))[0]

        c = jnp.array(3.12)
        expected = jax.jacfwd(
            lambda cc: quadgk(
                lambda t, x: jnp.log(x * t), jnp.array([1.0, 3.0]), (cc,)
            )[0]
        )(c)
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(integrate)(c)), np.asarray(expected), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(jax.grad(integrate)(c)), np.asarray(expected), rtol=1e-12
        )


class TestUnifiedLeibnizAdjoint:
    """LeibnizAdjoint supports both modes from one object, via a custom primitive."""

    fun = staticmethod(lambda t, c: jnp.sum(jnp.exp(-c * t)))
    interval = jnp.array([0.0, 1.0])
    args = jnp.linspace(0.5, 2.0, 3)

    def _ref(self, quad, wrt_interval=False):
        if wrt_interval:
            f = lambda v: quad(self.fun, v, (self.args,), adjoint=DirectAdjoint())[0]
            return np.asarray(jax.jacfwd(f)(self.interval))
        f = lambda c: quad(self.fun, self.interval, (c,), adjoint=DirectAdjoint())[0]
        return np.asarray(jax.jacfwd(f)(self.args))

    @pytest.mark.parametrize("quad", machinery_methods)
    @pytest.mark.parametrize(
        "transform", [jax.jacfwd, jax.jacrev, jax.grad, lambda f: jax.jit(jax.grad(f))]
    )
    def test_both_modes_agree_with_direct(self, quad, transform):
        """Every mode of differentiation gives the same derivative."""
        f = lambda c: quad(self.fun, self.interval, (c,), adjoint=LeibnizAdjoint())[0]
        np.testing.assert_allclose(
            np.asarray(transform(f)(self.args)), self._ref(quad), rtol=1e-8
        )

    @pytest.mark.parametrize("quad", machinery_methods)
    def test_jvp_matches_a_directional_derivative(self, quad):
        """Forward mode contracts with the tangent rather than forming the Jacobian."""
        f = lambda c: quad(self.fun, self.interval, (c,), adjoint=LeibnizAdjoint())[0]
        v = jnp.ones_like(self.args)
        np.testing.assert_allclose(
            float(jax.jvp(f, (self.args,), (v,))[1]),
            float(np.dot(self._ref(quad), np.asarray(v))),
            rtol=1e-8,
        )

    @pytest.mark.parametrize("quad", machinery_methods)
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev])
    def test_wrt_interval(self, quad, transform):
        """Derivatives with respect to the limits work in both modes."""
        f = lambda v: quad(self.fun, v, (self.args,), adjoint=LeibnizAdjoint())[0]
        np.testing.assert_allclose(
            np.asarray(transform(f)(self.interval)),
            self._ref(quad, wrt_interval=True),
            rtol=1e-8,
        )

    @pytest.mark.parametrize("quad", romberg_methods)
    def test_romberg_gets_both_modes(self, quad):
        """Romberg is forward-only under DirectAdjoint; this gives it reverse too."""
        f = lambda c: quad(self.fun, self.interval, (c,), adjoint=LeibnizAdjoint())[0]
        fwd = np.asarray(jax.jacfwd(f)(self.args))
        rev = np.asarray(jax.grad(f)(self.args))
        np.testing.assert_allclose(fwd, rev, rtol=1e-8)
        np.testing.assert_allclose(
            fwd,
            finite_difference(lambda a: f(a[0]), (self.args,)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_vmap_and_second_derivatives(self):
        """The primitive carries batching and composes for higher order."""
        f = lambda c: quadgk(self.fun, self.interval, (c,), adjoint=LeibnizAdjoint())[0]
        batch = jnp.stack([self.args, self.args * 1.1])
        np.testing.assert_allclose(
            np.asarray(jax.vmap(jax.grad(f))(batch)),
            np.stack([np.asarray(jax.grad(f)(b)) for b in batch]),
            rtol=1e-10,
        )
        assert np.asarray(jax.hessian(f)(self.args)).shape == (3, 3)

    @pytest.mark.parametrize("quad", machinery_methods)
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev, jax.grad])
    @pytest.mark.parametrize("wrt_interval", [False, True])
    def test_python_float_tolerances(self, quad, transform, wrt_interval):
        """A tolerance given as a python float must differentiate like any other.

        Every other test here leaves the tolerances at their default, which the solver
        turns into arrays. A concrete float instead reaches the primitive as a jaxpr
        literal, and reverse mode used to lose that operand when the transpose rule
        rebuilt its cotangent tree by filtering values for inexact arrays -- returning
        one cotangent fewer than the primitive had linear operands.
        """
        if wrt_interval and transform is jax.grad:
            pytest.skip("grad needs a scalar output; jacrev covers the same path")
        kwargs = {"epsabs": 1e-8, "epsrel": 1e-8}
        if wrt_interval:
            f = lambda v: quad(
                self.fun, v, (self.args,), adjoint=LeibnizAdjoint(), **kwargs
            )[0]  # noqa: E501
            target = self.interval
        else:
            f = lambda c: quad(
                self.fun, self.interval, (c,), adjoint=LeibnizAdjoint(), **kwargs
            )[0]  # noqa: E501
            target = self.args
        np.testing.assert_allclose(
            np.asarray(transform(f)(target)),
            self._ref(quad, wrt_interval=wrt_interval),
            rtol=1e-7,
        )

    @pytest.mark.parametrize("quad", romberg_methods)
    def test_python_float_tolerances_romberg_direct(self, quad):
        """Romberg reaches the same primitive under ``DirectAdjoint``.

        It has no subdivision to reuse, so ``DirectAdjoint`` freezes the level count and
        routes through ``_leibniz`` to get a transposable rule -- which means the
        literal tolerance bug reached ``DirectAdjoint`` too, by this path only.
        """
        f = lambda c: quad(  # noqa: E731
            self.fun,
            self.interval,
            (c,),
            adjoint=DirectAdjoint(),
            epsabs=1e-8,
            epsrel=1e-8,
        )[0]
        np.testing.assert_allclose(
            np.asarray(jax.grad(f)(self.args)),
            np.asarray(jax.jacfwd(f)(self.args)),
            rtol=1e-7,
        )


@pytest.mark.parametrize("quad", all_methods)
@pytest.mark.parametrize("adjoint", [DirectAdjoint(), LeibnizAdjoint()])
def test_integer_limits(quad, adjoint):
    """Integer integration limits must still be differentiable.

    They are not inexact arrays, so without being cast they end up classified as static
    metadata rather than as values, which a custom derivative rule cannot accept.
    """
    fun = lambda t, c: jnp.sum(jnp.exp(-c * t))
    c = jnp.linspace(0.5, 2.0, 3)
    f = lambda x: quad(fun, [0, 1], (x,), adjoint=adjoint)[0]
    g = lambda x: quad(fun, jnp.array([0.0, 1.0]), (x,), adjoint=adjoint)[0]
    np.testing.assert_allclose(
        np.asarray(jax.grad(f)(c)), np.asarray(jax.grad(g)(c)), rtol=1e-12
    )


@pytest.mark.parametrize("quad", romberg_methods)
def test_romberg_differentiates_the_extrapolation(quad):
    """The derivative must run through the Richardson table, not just the finest level.

    Freezing the number of levels fixes *which* linear functional is applied, not which
    parts of it are differentiated, so the result has to match differentiating the whole
    loop exactly. Differentiating only the finest trapezoid level would still look
    plausible while being orders of magnitude less accurate, so compare against the
    analytic derivative too.
    """
    c = jnp.array([1.7])
    fun = lambda t, cc: jnp.exp(-cc[0] * t)
    interval = jnp.array([0.0, 1.0])
    # d/dc int_0^1 exp(-c t) dt = [c e^-c - (1 - e^-c)] / c^2
    exact = float((c[0] * jnp.exp(-c[0]) - (1 - jnp.exp(-c[0]))) / c[0] ** 2)

    direct = lambda x: quad(fun, interval, (x,), adjoint=DirectAdjoint())[0]
    unrolled = lambda x: quad(fun, interval, (x,), adjoint=_UnrolledDirectAdjoint())[0]

    fwd = np.asarray(jax.jacfwd(direct)(c))
    rev = np.asarray(jax.grad(lambda x: direct(x).sum())(c))
    # Matches differentiating the entire loop, which includes the extrapolation. This is
    # the property that matters: it pins that the derivative runs through the whole
    # Richardson table rather than just the finest trapezoid level. Differentiating only
    # the finest level lands orders of magnitude away, far outside this tolerance.
    np.testing.assert_allclose(
        fwd, np.asarray(jax.jacfwd(unrolled)(c)), rtol=ULP_RTOL, atol=ULP_ATOL
    )
    # Forward and reverse are transposes of the same frozen linear functional, so they
    # agree to within rounding. For rombergts the integrand also carries a tanh-sinh
    # transform, whose own jvp and vjp can differ in the last bit.
    np.testing.assert_allclose(fwd, rev, rtol=ULP_RTOL, atol=ULP_ATOL)
    # and accurate to Romberg's standard, not the trapezoid rule's
    assert abs(float(fwd[0]) - exact) < 1e-9


class TestDerivativeDTypes:
    """Derivatives keep the working dtype set by ``interval``.

    The dtype plumbing itself is covered in ``test_adaptive.py``; here it only has to
    survive the adjoints.
    """

    # how much worse than sqrt(eps) a converged result is allowed to be, since these
    # check dtype plumbing rather than accuracy
    slop = 50

    @pytest.mark.parametrize("adjoint", [DirectAdjoint(), LeibnizAdjoint()])
    @pytest.mark.parametrize("method", adaptive_methods)
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_grad_dtype(self, adjoint, method, dtype):
        """Reverse mode returns a gradient at the interval's dtype, and correct."""

        def total(c):
            y, _ = method(
                lambda x, c: jnp.exp(-c * x),
                jnp.array([0.0, 1.0], dtype=dtype),
                args=(c,),
                adjoint=adjoint,
            )
            return y

        c = jnp.array(1.0, dtype)
        g = jax.grad(total)(c)
        assert g.dtype == dtype
        # d/dc int_0^1 exp(-c x) dx = (exp(-c)*(c+1) - 1)/c^2
        expected = (np.exp(-1) * 2 - 1) / 1.0
        np.testing.assert_allclose(
            float(g), expected, rtol=self.slop * np.sqrt(float(jnp.finfo(dtype).eps))
        )

    @pytest.mark.parametrize("method", adaptive_methods)
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_jvp_dtype(self, method, dtype):
        """Forward mode keeps both the primal and the tangent at that dtype."""
        f = lambda a: method(jnp.sin, jnp.array([0.0, 1.0], dtype=dtype) * a)[0]
        y, y_dot = jax.jvp(f, (jnp.array(1.0, dtype),), (jnp.array(1.0, dtype),))
        assert y.dtype == dtype
        assert y_dot.dtype == dtype


adjoints = [DirectAdjoint(), LeibnizAdjoint()]
adjoint_ids = ["direct", "leibniz"]


class TestRombergWithoutRichardson:
    """Turning Richardson off must not cost Romberg its derivatives.

    ``DirectAdjoint`` freezes the number of levels the solve settled on and
    differentiates that fixed discretization through a custom primitive, so the frozen
    evaluation has to read the same entry of the table as the solve did. Reading the
    diagonal when the solve read column zero would give the derivative of a quantity
    that was never returned, and nothing about the value itself would show it.
    """

    fun = staticmethod(lambda t, c: jnp.exp(-c * t**2))
    interval = jnp.array([0.0, 2.0])
    args = jnp.asarray(0.7)

    def _quad(self, method, extrapolate, adjoint):
        return lambda c: method(
            self.fun,
            self.interval,
            (c,),
            epsabs=1e-10,
            epsrel=1e-10,
            divmax=14,
            extrapolate=extrapolate,
            adjoint=adjoint,
        )[0]

    @pytest.mark.parametrize("method", romberg_methods, ids=["romberg", "ts"])
    @pytest.mark.parametrize("extrapolate", [False, True], ids=["plain", "extrap"])
    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    def test_modes_agree(self, method, extrapolate, adjoint):
        """Forward and reverse give the same derivative in either setting."""
        f = self._quad(method, extrapolate, adjoint)
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(f)(self.args)),
            np.asarray(jax.grad(f)(self.args)),
            rtol=1e-10,
        )

    @pytest.mark.parametrize("method", romberg_methods, ids=["romberg", "ts"])
    @pytest.mark.parametrize("extrapolate", [False, True], ids=["plain", "extrap"])
    def test_matches_finite_differences(self, method, extrapolate):
        """And it is the right derivative, not merely a self-consistent one.

        The check that catches a frozen evaluation reading the wrong column: both modes
        would still agree with each other, because they would agree about the *wrong*
        quantity.
        """
        f = self._quad(method, extrapolate, DirectAdjoint())
        got = float(jax.jacfwd(f)(self.args))
        h = 1e-6
        fd = (float(f(self.args + h)) - float(f(self.args - h))) / (2 * h)
        np.testing.assert_allclose(got, fd, rtol=1e-6)

    @pytest.mark.parametrize("method", romberg_methods, ids=["romberg", "ts"])
    @pytest.mark.parametrize("extrapolate", [False, True], ids=["plain", "extrap"])
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev])
    def test_wrt_interval(self, method, extrapolate, transform):
        """Derivatives with respect to the limits work in either setting.

        Both modes, because on a finite interval the whole of this derivative is the
        boundary term rather than anything the solve integrates.
        """
        f = lambda iv: method(  # noqa: E731
            self.fun,
            iv,
            (self.args,),
            epsabs=1e-10,
            epsrel=1e-10,
            divmax=14,
            extrapolate=extrapolate,
        )[0]
        got = np.asarray(transform(f)(self.interval))
        # d/db of int_a^b f = f(b), and d/da = -f(a)
        expected = np.array(
            [
                -float(self.fun(self.interval[0], self.args)),
                float(self.fun(self.interval[1], self.args)),
            ]
        )
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)


# Smooth enough that the subdivision converges on its own and the table is never
# consulted, so `extrapolate=True` must change nothing at all.
UNACCELERATED_PROBLEMS = [
    {
        "name": "gaussian",
        "fun": lambda x, c: jnp.exp(-c * x**2),
        "interval": [0.0, 3.0],
        "args": jnp.asarray(1.0),
    },
    {
        "name": "oscillatory",
        "fun": lambda x, c: jnp.sin(c * x),
        "interval": [0.0, 10.0],
        "args": jnp.asarray(3.0),
    },
]


# Integrands whose derivative is worth taking through an extrapolation: each has a
# singularity the subdivision alone cannot resolve, so the accelerated primal is a
# different value from the mesh sum and the adjoint has to reproduce which one was
# returned. `exact` is d/dc of the integral where it is available in closed form.
EXTRAPOLATED_PROBLEMS = [
    {
        "name": "endpoint x**-0.5",
        "fun": lambda x, c: c * x**-0.5,
        "interval": [0.0, 1.0],
        "args": jnp.asarray(2.0),
        "exact": 2.0,
    },
    {
        "name": "endpoint x**-0.9",
        "fun": lambda x, c: c * x**-0.9,
        "interval": [0.0, 1.0],
        "args": jnp.asarray(1.0),
        "exact": 10.0,
    },
    {
        "name": "semi-infinite exp(-x)/sqrt(x)",
        "fun": lambda x, c: c * jnp.exp(-x) / jnp.sqrt(x),
        "interval": [0.0, jnp.inf],
        "args": jnp.asarray(1.0),
        "exact": float(np.sqrt(np.pi)),
    },
    {
        "name": "interior, not at a breakpoint",
        "fun": lambda x, c: jnp.abs(x - 0.3) ** -0.5 * jnp.exp(-c * x),
        "interval": [0.0, 1.0],
        "args": jnp.asarray(1.0),
        "exact": None,
    },
    {
        "name": "interior, marked as a breakpoint",
        "fun": lambda x, c: jnp.abs(x - 0.3) ** -0.5 * jnp.exp(-c * x),
        "interval": [0.0, 0.3, 1.0],
        "args": jnp.asarray(1.0),
        "exact": None,
    },
    {
        "name": "vector, one component singular",
        "fun": lambda x, c: jnp.array([c * x**-0.5, jnp.exp(-c * x)]),
        "interval": [0.0, 1.0],
        "args": jnp.asarray(1.0),
        "exact": None,
    },
]


def _integrate(prob, adjoint, extrapolate, wrt_interval=False):
    """`quadgk` on one of the problems above, as a function of what is varied."""
    interval = jnp.asarray(prob["interval"], float)

    def f(z):
        args, iv = (prob["args"], z) if wrt_interval else (z, interval)
        return quadgk(
            prob["fun"],
            iv,
            args=(args,),
            epsabs=1e-12,
            epsrel=1e-12,
            order=21,
            max_ninter=100,
            adjoint=adjoint,
            extrapolate=extrapolate,
        )[0]

    return f, (interval if wrt_interval else prob["args"])


class TestExtrapolatedAdjoints:
    """Differentiating a quadrature whose value came from an extrapolation.

    With ``extrapolate=True`` the value returned may be the epsilon algorithm's estimate
    of the limit of the running totals rather than the sum over the final subdivision,
    and the adjoints have to differentiate whichever one was returned. Everything the
    acceleration decided was decided on error estimates and is integer or boolean, so
    the derivative is taken with those decisions frozen, the same discretize-then-
    optimize bargain :class:`DirectAdjoint` already makes for the mesh.

    Derivatives with respect to a limit that sits *on* a singularity are excluded
    throughout. They are genuinely infinite -- ``d/da`` of ``int_a^1 x**-0.5`` is
    ``-a**-0.5`` -- so there is no value for a test to check against, with or without
    acceleration. Limits away from the singularity are covered below.
    """

    @pytest.mark.parametrize("prob", EXTRAPOLATED_PROBLEMS, ids=lambda p: p["name"])
    def test_replay_reproduces_the_accelerated_value(self, prob):
        """The function being differentiated is the one whose value was returned.

        This is the structural property the rest of the class rests on. A derivative
        taken on a replay that lands somewhere else is answering a different question,
        and no comparison against finite differences would reveal it, because the two
        would be wrong together.
        """
        interval = jnp.asarray(prob["interval"], float)
        f_conv, consts = closure_convert(prob["fun"], (prob["args"],), interval.dtype)
        vfunc, interval_t = build_integrand(
            interval, (prob["args"],), consts, f_conv=f_conv
        )
        rule = GaussKronrodRule(21)
        y, state = _adaptive_solve(
            rule,
            vfunc,
            interval_t,
            jnp.asarray(1e-12),
            jnp.asarray(1e-12),
            {},
            max_ninter=100,
            extrapolate=True,
        )
        replayed = _replay_solve(rule, vfunc, interval_t, _frozen_replay(state), {})
        # Not bit-identical: the replay rebuilds the running totals by accumulating
        # births and deaths where the solve re-sums the whole subdivision each pass, and
        # the epsilon algorithm amplifies the difference. The mesh path has the same
        # property, for the same reason.
        np.testing.assert_allclose(
            np.asarray(replayed), np.asarray(y), rtol=1e-11, atol=1e-14
        )

    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    @pytest.mark.parametrize("prob", EXTRAPOLATED_PROBLEMS, ids=lambda p: p["name"])
    def test_modes_agree(self, prob, adjoint):
        """Forward and reverse give the same derivative through an extrapolation."""
        f, x = _integrate(prob, adjoint, True)
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(f)(x)),
            np.asarray(jax.jacrev(f)(x)),
            rtol=1e-9,
            atol=1e-12,
        )

    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    @pytest.mark.parametrize(
        "prob",
        [p for p in EXTRAPOLATED_PROBLEMS if p["exact"] is not None],
        ids=lambda p: p["name"],
    )
    def test_derivative_is_at_least_as_good_as_the_mesh(self, prob, adjoint):
        """Accelerating the integral must not cost accuracy in its derivative.

        On these integrands it gains: the subdivision stops at the width floor with the
        mesh still far from the limit, and the derivative inherits exactly that error.
        """
        exact = prob["exact"]
        errs = {}
        for extrapolate in (False, True):
            f, x = _integrate(prob, adjoint, extrapolate)
            got = np.max(np.abs(np.atleast_1d(np.asarray(jax.jacfwd(f)(x)))))
            errs[extrapolate] = abs(got - exact) / abs(exact)
        assert errs[True] <= errs[False], (
            f"{prob['name']}: extrapolated derivative is worse "
            f"({errs[True]:.2e} vs {errs[False]:.2e})"
        )
        assert errs[True] < 1e-10

    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    @pytest.mark.parametrize("prob", UNACCELERATED_PROBLEMS, ids=lambda p: p["name"])
    def test_smooth_problems_differentiate_identically(self, prob, adjoint):
        """Where the table is never consulted the flag must cost nothing.

        The subdivision converges and no extrapolated value is accepted, so both
        settings return the same sum, but only one of them carries the table through
        the loop, so they are separate programs whose shared arithmetic is free to be
        reassociated between them, and the agreement is to rounding rather than bit for
        bit. The derivative has a second reason to differ in the last place: the replay
        reconstructs the subdivision's total from births and deaths, which is a
        different summation order from the blocked accumulation the mesh path uses, and
        neither is more correct than the other.
        """
        off, on = (_integrate(prob, adjoint, e)[0] for e in (False, True))
        np.testing.assert_allclose(
            np.asarray(on(prob["args"])),
            np.asarray(off(prob["args"])),
            rtol=ULP_RTOL,
            atol=ULP_ATOL,
        )
        for transform in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                np.asarray(transform(on)(prob["args"])),
                np.asarray(transform(off)(prob["args"])),
                rtol=ULP_RTOL,
                atol=ULP_ATOL,
            )

    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    def test_wrt_a_limit_away_from_the_singularity(self, adjoint):
        """The limits still move the mesh correctly under an extrapolation.

        The singularity sits at an interior breakpoint, so both limits are regular and
        ``d/db`` is just the integrand there.
        """
        prob = {
            "fun": lambda x, c: jnp.abs(x - 0.3) ** -0.5 * jnp.exp(-c * x),
            "interval": [0.0, 0.3, 1.0],
            "args": jnp.asarray(1.0),
        }
        expected = float(np.abs(1.0 - 0.3) ** -0.5 * np.exp(-1.0))
        for transform in (jax.jacfwd, jax.jacrev):
            f, iv = _integrate(prob, adjoint, True, wrt_interval=True)
            got = np.asarray(transform(f)(iv))
            np.testing.assert_allclose(got[-1], expected, rtol=1e-10)

    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    def test_transforms(self, adjoint):
        """jit, vmap and second derivatives all survive the replay."""
        prob = EXTRAPOLATED_PROBLEMS[0]
        f, x = _integrate(prob, adjoint, True)
        ref = np.asarray(jax.jacfwd(f)(x))
        np.testing.assert_allclose(np.asarray(jax.jit(jax.grad(f))(x)), ref, rtol=1e-9)
        batch = jnp.stack([x, x * 1.5])
        np.testing.assert_allclose(
            np.asarray(jax.vmap(jax.grad(f))(batch)),
            np.stack([np.asarray(jax.grad(f)(b)) for b in batch]),
            rtol=1e-9,
        )

    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    @pytest.mark.parametrize("d1", [jax.jacfwd, jax.jacrev], ids=["jacfwd", "jacrev"])
    @pytest.mark.parametrize("d2", [jax.jacfwd, jax.jacrev], ids=["jacfwd", "jacrev"])
    def test_second_derivatives(self, adjoint, d1, d2):
        """d2/dc2 of a linear-in-c integral is zero, for all combos of fwd/rev."""
        prob = EXTRAPOLATED_PROBLEMS[0]
        f, x = _integrate(prob, adjoint, True)
        second = np.asarray(d1(d2(f))(x))
        assert np.all(np.isfinite(second))
        np.testing.assert_allclose(second, 0.0, atol=1e-9)


class TestBatchSize:
    """Batching integrand evaluations must not disturb the derivatives.

    It puts new control flow in the differentiated path of both families: a ``scan``
    node batches inside the adaptive rules, and a dynamically bounded loop over point
    batches inside the Romberg levels. Romberg also introduces padded lanes, since its
    level sizes are only known at run time; those are substituted rather than masked
    precisely so they cannot poison a gradient, which is what the NaN check below is
    for.

    The adaptive comparison is at ULP scale, since cutting the batches to fit leaves the
    arithmetic the same. Romberg's is looser, because batching does reassociate
    the sum over each level.
    """

    @pytest.mark.parametrize("quad", adaptive_methods)
    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    @pytest.mark.parametrize("i", range(len(example_problems)))
    def test_adaptive_matches_unbatched(self, quad, adjoint, transform, i):
        """Same derivative, in either mode, under either adjoint."""
        prob = example_problems[i]
        interval = jnp.asarray(prob["interval"])
        make = lambda bs: (
            lambda a: quad(  # noqa: E731
                prob["fun"], interval, a, adjoint=adjoint, batch_size=bs
            )[0]
        )
        want = transform(make(None))(prob["args"])
        got = transform(make(4))(prob["args"])
        want, got = jax.tree.leaves(want)[0], jax.tree.leaves(got)[0]
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=ULP_RTOL, atol=ULP_ATOL
        )

    @pytest.mark.parametrize("quad", romberg_methods, ids=["romberg", "ts"])
    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    def test_romberg_matches_unbatched(self, quad, adjoint, transform):
        """Reverse mode included, which for Romberg goes through the custom primitive.

        ``DirectAdjoint`` freezes the number of levels and replays them, and that replay
        has to batch its points exactly as the solve did. Batching one and not the other
        would differentiate a discretization that was never evaluated.
        """
        fun = lambda t, c: jnp.exp(-c * t**2)  # noqa: E731
        interval = jnp.array([0.0, 2.0])
        args = jnp.asarray(0.7)
        make = lambda bs: (
            lambda c: quad(  # noqa: E731
                fun, interval, (c,), divmax=10, adjoint=adjoint, batch_size=bs
            )[0]
        )
        want = float(transform(make(None))(args))
        got = float(transform(make(8))(args))
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("quad", all_methods)
    def test_wrt_interval(self, quad):
        """The limits are differentiated with respect to as well as the args."""
        fun = lambda t: jnp.exp(-(t**2))  # noqa: E731
        make = lambda bs: lambda iv: quad(fun, iv, batch_size=bs)[0]  # noqa: E731
        iv = jnp.array([0.0, 2.0])
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(make(4))(iv)),
            np.asarray(jax.jacfwd(make(None))(iv)),
            rtol=1e-10,
            atol=1e-12,
        )

    @pytest.mark.parametrize("quad", all_methods)
    def test_padding_introduces_no_new_nan(self, quad):
        """An integrand singular at a limit, where a careless fill would land.

        Romberg's padded lanes repeat a point their batch genuinely evaluates, not a
        made up one, so they can only ever ask the integrand for a value it is already
        being asked for. Filling them with a placeholder would mask the value but leave
        a NaN in its derivative, which masking afterwards does not remove. The adaptive
        routines pad nothing and are here to show the claim holds trivially for them.

        The claim is that batching adds no NaN, not that there is none: a closed rule
        (and Romberg's trapezoidal level zero) places a node on the singularity itself
        and has an unusable derivative there whatever the batch size. So the batched run
        is held to what the unbatched one already manages, component by component.
        """
        fun = lambda t, c: jnp.log(c * t)  # noqa: E731
        f = lambda bs: jax.grad(  # noqa: E731
            lambda c: quad(fun, jnp.array([0.0, 1.0]), (c,), batch_size=bs)[0]
        )(jnp.asarray(2.0))
        want = np.isfinite(np.asarray(f(None)))
        got = np.isfinite(np.asarray(f(3)))
        np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize("quad", all_methods)
    def test_vmap(self, quad):
        """Batching sits inside the loops that ``vmap`` already has to survive."""
        fun = lambda t, c: jnp.exp(-c * t**2)  # noqa: E731
        f = lambda c: quad(  # noqa: E731
            fun, jnp.array([0.0, 2.0]), (c,), batch_size=4
        )[0]
        cs = jnp.array([0.5, 0.7, 1.1])
        np.testing.assert_allclose(
            np.asarray(jax.vmap(f)(cs)),
            np.asarray(jnp.array([f(c) for c in cs])),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jax.vmap(jax.grad(f))(cs)),
            np.asarray(jnp.array([jax.grad(f)(c) for c in cs])),
            rtol=1e-10,
            atol=1e-12,
        )


class TestChunkSize:
    """How many sub-intervals of the frozen subdivision an adjoint evaluates at once.

    The reverse-mode counterpart of ``batch_size``: that one bounds the work within a
    sub-interval, this one bounds how many sub-intervals are in flight together. It is
    purely a memory-against-speed choice, so like ``batch_size`` on the adaptive rules
    it must not move the derivative - unused slots in a chunk are handed a real
    sub-interval and masked out, so the chunking changes only the order the same
    contributions are summed in.
    """

    @pytest.mark.parametrize("quad", adaptive_methods)
    @pytest.mark.parametrize("adjoint", [DirectAdjoint], ids=["direct"])
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
    @pytest.mark.parametrize("chunk_size", [1, 3, 8, 64], ids=str)
    @pytest.mark.parametrize("i", range(len(example_problems)))
    def test_derivative_is_unchanged(self, quad, adjoint, transform, chunk_size, i):
        """Every chunk size gives the derivative the default one gives."""
        prob = example_problems[i]
        interval = jnp.asarray(prob["interval"])
        make = lambda adj: (
            lambda a: quad(  # noqa: E731
                prob["fun"], interval, a, adjoint=adj
            )[0]
        )
        want = transform(make(adjoint()))(prob["args"])
        got = transform(make(adjoint(chunk_size=chunk_size)))(prob["args"])
        np.testing.assert_allclose(
            np.asarray(jax.tree.leaves(got)[0]),
            np.asarray(jax.tree.leaves(want)[0]),
            rtol=ULP_RTOL,
            atol=ULP_ATOL,
        )

    @pytest.mark.parametrize("adjoint", [DirectAdjoint], ids=["direct"])
    def test_composes_with_batch_size(self, adjoint):
        """The two knobs are independent, and multiply to the real parallel width."""
        prob = example_problems[0]
        interval = jnp.asarray(prob["interval"])
        f = lambda adj, bs: jax.jacrev(  # noqa: E731
            lambda a: quadgk(prob["fun"], interval, a, adjoint=adj, batch_size=bs)[0]
        )(prob["args"])
        want = f(adjoint(), None)
        got = f(adjoint(chunk_size=3), 4)
        np.testing.assert_allclose(
            np.asarray(jax.tree.leaves(got)[0]),
            np.asarray(jax.tree.leaves(want)[0]),
            rtol=ULP_RTOL,
            atol=ULP_ATOL,
        )

    @pytest.mark.parametrize("adjoint", [DirectAdjoint], ids=["direct"])
    def test_romberg_is_unaffected(self, adjoint):
        """Romberg has no subdivision to chunk, so the option is simply inert there."""
        f = lambda adj: float(  # noqa: E731
            jax.grad(
                lambda c: romberg(
                    lambda t, c_: jnp.exp(-c_ * t**2),
                    jnp.array([0.0, 2.0]),
                    (c,),
                    adjoint=adj,
                )[0]
            )(jnp.asarray(0.7))
        )
        np.testing.assert_allclose(
            f(adjoint(chunk_size=2)), f(adjoint()), rtol=ULP_RTOL, atol=ULP_ATOL
        )

    @pytest.mark.parametrize("adjoint", [DirectAdjoint], ids=["direct"])
    @pytest.mark.parametrize(
        "chunk_size", [0, -1, 2.5], ids=["zero", "negative", "float"]
    )
    def test_bad_chunk_size_rejected(self, adjoint, chunk_size):
        """Rejected when the adjoint is built, not on first use."""
        with pytest.raises(ValueError, match="chunk_size"):
            adjoint(chunk_size=chunk_size)


# Integrands quadax cannot evaluate everywhere: `t**(p-1)` with `p < 1` is infinite at
# t = 0, and `|t|**(-1/2)` is infinite at the interior breakpoint. Both are integrable,
# so the *value* is finite and known, and quadax reaches it by masking the non-finite
# evaluations to zero. That mask is what these tests are about: masking the output alone
# is not differentiable in reverse, because the masked abscissa comes back with a
# cotangent of exactly zero, and zero times the infinite local derivative there is a
# NaN. Which abscissae land on the singularity depends on the rule, Clenshaw-Curtis
# includes the endpoints outright, tanh-sinh clusters against them, Gauss-Kronrod only
# gets there once a sub-interval has been bisected down near it, so every rule runs
# the full set.


def _endpoint_singular(t, c):
    """Integrand of int_0^1 t**(p-1) dt = 1/p, singular at the lower limit."""
    return t ** (c[0] - 1.0)


def _midpoint_singular(t, c):
    """Integrand of int_-1^1 c/sqrt(|t|) dt = 4c, singular at the breakpoint."""
    return c[0] / jnp.sqrt(jnp.abs(t))


@pytest.mark.parametrize("quad", adaptive_methods)
@pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
class TestSingularIntegrandDerivatives:
    """Derivatives of integrands the local rule evaluates at a singular point."""

    def test_endpoint_singularity_reverse(self, quad, adjoint):
        """Reverse mode returns a derivative, not a NaN."""
        p = jnp.array([0.4])
        f = lambda c: quad(  # noqa: E731
            _endpoint_singular,
            [0.0, 1.0],
            (c,),
            epsabs=1e-10,
            epsrel=1e-10,
            adjoint=adjoint,
        )[0]
        y = f(p)
        g = jax.jacrev(f)(p)
        assert np.isfinite(np.asarray(g)).all()
        # int_0^1 t**(p-1) = 1/p, so the derivative is -1/p**2
        np.testing.assert_allclose(np.asarray(y), 1 / 0.4, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(g).squeeze(), -1 / 0.4**2, rtol=1e-5)

    def test_endpoint_singularity_modes_agree(self, quad, adjoint):
        """Forward mode was never affected, so it pins down what reverse should give."""
        p = jnp.array([0.4])
        f = lambda c: quad(  # noqa: E731
            _endpoint_singular,
            [0.0, 1.0],
            (c,),
            epsabs=1e-10,
            epsrel=1e-10,
            adjoint=adjoint,
        )[0]
        np.testing.assert_allclose(
            np.asarray(jax.jacrev(f)(p)), np.asarray(jax.jacfwd(f)(p)), rtol=1e-10
        )

    def test_singularity_at_an_interior_breakpoint(self, quad, adjoint):
        """A singularity in the *interior* is not a special case of an endpoint one.

        Guards the choice of where to linearize instead of the singular abscissa: any
        fixed substitute (the middle of the domain, say) is itself the singularity for
        an integrand like this one.
        """
        p = jnp.array([1.4])
        f = lambda c: quad(  # noqa: E731
            _midpoint_singular,
            [-1.0, 0.0, 1.0],
            (c,),
            epsabs=1e-10,
            epsrel=1e-10,
            adjoint=adjoint,
        )[0]
        assert np.isfinite(np.asarray(f(p))).all()
        g = np.asarray(jax.jacrev(f)(p)).squeeze()
        assert np.isfinite(g).all()
        np.testing.assert_allclose(g, 4.0, rtol=2e-5)

    def test_smooth_component_survives_a_singular_one(self, quad, adjoint):
        """One singular component must not cost the others their derivatives.

        Every component is evaluated at the same abscissae and differentiated with
        respect to the same parameters, so a component that blows up is in a position to
        poison the rest. The smooth component's derivative is known exactly, so it says
        whether anything was lost.
        """
        p = jnp.array([1.4, 1.3])
        f = lambda c: quad(  # noqa: E731
            lambda t, c_: jnp.array([c_[0] / jnp.sqrt(jnp.abs(t)), jnp.cos(c_[1] * t)]),
            [-1.0, 0.0, 1.0],
            (c,),
            epsabs=1e-10,
            epsrel=1e-10,
            adjoint=adjoint,
        )[0]
        jac = np.asarray(jax.jacrev(f)(p))
        assert np.isfinite(jac).all()
        # int cos(c t) over [-1, 1] is 2 sin(c)/c, so d/dc is 2(c cos c - sin c)/c**2
        c = 1.3
        want = 2 * (c * np.cos(c) - np.sin(c)) / c**2
        np.testing.assert_allclose(jac[1, 1], want, rtol=1e-8)
        # and the smooth component does not depend on the singular one's parameter
        np.testing.assert_allclose(jac[1, 0], 0.0, atol=1e-12)
        # int c/sqrt|t| over [-1, 1] is 4c
        np.testing.assert_allclose(jac[0, 0], 4.0, rtol=2e-5)

    def test_limit_with_a_singularity_at_the_other_end(self, quad, adjoint):
        """A limit carries a derivative even when the opposite limit is singular.

        The derivative with respect to a limit is a value of the integrand there, which
        is finite at this end; what has to survive is the singular end being evaluated
        at all while that value is being computed.
        """
        b = jnp.array(4.0)
        # int_0^b t**(-1/2) dt = 2 sqrt(b), so d/db is 1/sqrt(b)
        f = lambda b_: quad(  # noqa: E731
            lambda t: 1 / jnp.sqrt(t),
            jnp.array([0.0, b_]),
            epsabs=1e-10,
            epsrel=1e-10,
            adjoint=adjoint,
        )[0]
        np.testing.assert_allclose(np.asarray(f(b)), 4.0, rtol=1e-8)
        g = float(jax.grad(f)(b))
        assert np.isfinite(g)
        np.testing.assert_allclose(g, 0.5, rtol=1e-6)

    def test_second_derivatives(self, quad, adjoint):
        """The mask survives being differentiated twice, in both nestings.

        ``jacfwd(jacfwd)`` nests two JVPs, while ``hessian`` is ``jacfwd(jacrev)`` and
        transposes through the custom rules instead; they reach different machinery.
        """
        p = jnp.array([0.4])
        f = lambda c: quad(  # noqa: E731
            _endpoint_singular,
            [0.0, 1.0],
            (c,),
            epsabs=1e-10,
            epsrel=1e-10,
            adjoint=adjoint,
        )[0].squeeze()
        # int_0^1 t**(p-1) = 1/p, so the second derivative is 2/p**3
        want = 2 / 0.4**3
        for name, h in (
            ("jacfwd^2", jax.jacfwd(jax.jacfwd(f))),
            ("hessian", jax.hessian(f)),
        ):
            got = np.asarray(h(p)).squeeze()
            assert np.isfinite(got).all(), f"{name} gave {got}"
            np.testing.assert_allclose(got, want, rtol=1e-4, err_msg=name)


# Integrands with a feature pinned to a breakpoint that is itself being differentiated.
# `s` reaches the answer twice over: through the integrand, and through the position of
# the breakpoint. The second contribution is the boundary term of the Leibniz rule,
# which is a jump if the integrand has one there and zero otherwise. Ordered by how
# singular the integrand is at the moving node.
SINGULAR = 1e-12  # a tolerance the smooth cases reach comfortably
MOVING_NODE_PROBLEMS = {
    # continuous at the node, and so is its derivative
    "kink": (
        lambda t, s: jnp.abs(t - s),
        lambda s: 0.5 * ((1 - s) ** 2 + (1 + s) ** 2),
    ),
    # continuous at the node, derivative unbounded there
    "sqrt": (
        lambda t, s: jnp.sqrt(jnp.abs(t - s)),
        lambda s: (2 / 3) * ((1 - s) ** 1.5 + (1 + s) ** 1.5),
    ),
    # unbounded at the node, symmetrically
    "invsqrt": (
        lambda t, s: 1 / jnp.sqrt(jnp.abs(t - s)),
        lambda s: 2 * jnp.sqrt(1 - s) + 2 * jnp.sqrt(1 + s),
    ),
    # unbounded at the node with a different coefficient on each side, which is what
    # stops the one-sided values being read straight off: they are both enormous and
    # their difference is the asymmetry between them amplified, not a jump
    "invsqrt-asym": (
        lambda t, s: jnp.where(t > s, 2.0, 1.0) / jnp.sqrt(jnp.abs(t - s)),
        lambda s: 2 * jnp.sqrt(1 + s) + 4 * jnp.sqrt(1 - s),
    ),
    # a genuine jump at the node, which is what the boundary term exists for
    "step": (lambda t, s: jnp.where(t > s, 1.0, 0.0), lambda s: 1.0 - s),
    # the same singularity with the node on a power of two, the one place where the
    # floating point spacing above and below a point differs.
    "invsqrt-binade": (
        lambda t, s: 1 / jnp.sqrt(jnp.abs(t - s)),
        lambda s: 2 * jnp.sqrt(s + 1.0) + 2 * jnp.sqrt(5.0 - s),
    ),
    # a jump superposed on a singularity: the singular parts of the two readings are
    # equal and cancel, so the jump survives
    "step-on-sing": (
        lambda t, s: jnp.where(t > s, 1.0, 0.0) + 1 / jnp.sqrt(jnp.abs(t - s)),
        lambda s: 1.0 - s + 2 * jnp.sqrt(1 - s) + 2 * jnp.sqrt(1 + s),
    ),
    # singular on one side and finite on the other, so the one-sided values are not
    # equal and cannot cancel in a difference. The singular side has no limit and
    # contributes nothing; the finite side still contributes its value, and dropping
    # both would lose that. int_-1^s (s-t)**(-1/2) + int_s^1 1
    "half-singular": (
        lambda t, s: jnp.where(t > s, 1.0, 1 / jnp.sqrt(jnp.abs(t - s))),
        lambda s: 2 * jnp.sqrt(s + 1.0) + (1.0 - s),
    ),
    # the same the other way round, so a sign error in either side shows up
    "half-singular-flipped": (
        lambda t, s: jnp.where(t > s, 1 / jnp.sqrt(jnp.abs(t - s)), 2.0),
        lambda s: 2.0 * (s + 1.0) + 2 * jnp.sqrt(1.0 - s),
    ),
    # steep but continuous, and marked: the breakpoint is doing no work beyond helping
    # the subdivision, and the derivative must be unaffected by its presence
    "steep-tanh": (
        lambda t, s: 0.5 * (1 + jnp.tanh((t - s) / 0.01)),
        lambda s: (
            0.5 * ((1.0 - s) + 0.01 * jnp.log(jnp.cosh((1.0 - s) / 0.01)))
            + 0.5 * ((1.0 + s) - 0.01 * jnp.log(jnp.cosh((1.0 + s) / 0.01)))
        ),
    ),
}


def _moving_node(name, adjoint):
    """The integral as a function of the node position, and its exact derivative.

    ``invsqrt-binade`` puts the node on 1.0 deliberately, which is a power of two and so
    the one place the spacing above and below the node differ; the others sit at 0.3.
    """
    fun, exact = MOVING_NODE_PROBLEMS[name]
    binade = name == "invsqrt-binade"
    s = jnp.asarray(1.0 if binade else 0.3)
    hi = 5.0 if binade else 1.0
    f = lambda s_: quadgk(  # noqa: E731
        fun,
        jnp.array([-1.0, s_, hi]),
        (s_,),
        epsabs=SINGULAR,
        epsrel=SINGULAR,
        adjoint=adjoint,
    )[0]
    return f, s, float(exact(s)), float(jax.grad(exact)(s))


@pytest.mark.parametrize("name", list(MOVING_NODE_PROBLEMS))
class TestMovingSingularity:
    """Differentiating where a feature *is*, not just how large it is."""

    @pytest.mark.parametrize("adjoint", adjoints, ids=adjoint_ids)
    @pytest.mark.parametrize("transform", ["fwd", "rev"])
    def test_derivative(self, name, adjoint, transform):
        """Both adjoints, both modes, against the exact derivative.

        ``DirectAdjoint`` differentiates the discretization, so the two contributions
        are formed at the same abscissae and cancel term by term whatever the integrand
        does at the node. ``LeibnizAdjoint`` takes the second contribution as a boundary
        term instead, read from the integrand's one-sided values at the node.
        """
        f, x, want_y, want_g = _moving_node(name, adjoint)
        np.testing.assert_allclose(float(f(x)), want_y, rtol=SINGULAR, atol=SINGULAR)
        g = (
            float(jax.grad(f)(x))
            if transform == "rev"
            else float(jax.jvp(f, (x,), (jnp.ones_like(x),))[1])
        )
        assert np.isfinite(g)
        np.testing.assert_allclose(g, want_g, rtol=1e-5, atol=1e-9)
