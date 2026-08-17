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

# Tolerance for two routes to the same computation - a different adjoint, jit against
# eager, forward mode against reverse - which should agree to ~1 ULP. Tight enough that
# any real difference in what is being computed shows up, loose enough not to depend on
# the operation order XLA happens to choose.
ULP_RTOL = 1e-13
ULP_ATOL = 1e-15


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
    """Moving a breakpoint that sits on a jump must produce the boundary term.

    Simply freezing the converged subdivision loses this: in mapped coordinates a frozen
    mesh edge does not move with the breakpoint, so it cannot see the jump slide across
    it, and the derivative comes out as ``[-3, 0, 3]`` instead of ``[-1, -4, 5]``.
    ``DirectAdjoint`` replays the subdivision instead of freezing it, which recovers it.
    """

    fun = staticmethod(lambda t, c: jnp.where(t < c[0], 1.0, 5.0).squeeze())
    interval = jnp.array([0.0, 1.5, 3.0])
    args = (jnp.array([1.5]),)
    expected = np.array([-1.0, -4.0, 5.0])

    @pytest.mark.parametrize("quad", adaptive_methods)
    @pytest.mark.parametrize("transform", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize("adjoint", [DirectAdjoint(), LeibnizAdjoint()])
    def test_boundary_term(self, quad, transform, adjoint):
        """Derivative w.r.t. the breakpoint picks up the jump in the integrand."""
        f = lambda v: quad(self.fun, v, self.args, adjoint=adjoint)[0]
        np.testing.assert_allclose(
            np.asarray(transform(f)(self.interval)), self.expected, atol=1e-12
        )

    @pytest.mark.parametrize("quad", machinery_methods)
    def test_matches_unrolled(self, quad):
        """And agrees with differentiating through the loop."""
        mk = lambda adj: lambda v: quad(self.fun, v, self.args, adjoint=adj)[0]
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(mk(_UnrolledDirectAdjoint()))(self.interval)),
            np.asarray(jax.jacfwd(mk(DirectAdjoint()))(self.interval)),
            rtol=ULP_RTOL,
            atol=ULP_ATOL,
        )


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
