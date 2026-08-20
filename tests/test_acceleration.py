"""Tests for the epsilon algorithm used to accelerate the adaptive quadrature."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)

from quadax._acceleration import (  # noqa: E402
    ERR_FLOOR,
    IRREGULAR,
    MAX_TABLE_SIZE,
    extrapolate,
    init_table,
)

real_dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
complex_dtypes = [jnp.complex128, jnp.complex64]

_inf_norm = lambda x: jnp.max(jnp.abs(jnp.atleast_1d(x)))


def accelerate(seq, dtype=jnp.float64, shape=()):
    """Feed a sequence in one term at a time, as the adaptive loop does.

    Returns the state after every term, so that the best estimate reached along the way
    is visible: the last one is not generally the best, which is why the integrator
    tracks the running best rather than reporting the final extrapolation.
    """
    seq = jnp.asarray(np.asarray(seq), dtype)
    step = jax.jit(lambda st, v: extrapolate(st, v, _inf_norm))
    state = init_table(shape, dtype)
    states = []
    for value in seq:
        state = step(state, value)
        states.append(state)
    return states


def best_relerr(seq, exact, dtype=jnp.float64):
    """Best relative error reached at any point in the run."""
    return min(
        abs(complex(st.result) - exact) / abs(exact) for st in accelerate(seq, dtype)
    )


# ---------------------------------------------------------------------------------
# Oracle: a direct transcription of `dqelg.f`, with real control flow rather than
# masking. The JAX version is a masked rewrite of exactly this, so any disagreement
# between them is a bug in the rewrite.
# ---------------------------------------------------------------------------------
def dqelg_reference(seq):
    """Yield ``(result, abserr)`` per appended term, following the Fortran directly.

    Kept on QUADPACK's own 1-based indexing, with ``epstab[0]`` unused, so that every
    line can be read straight off ``dqelg.f``. The implementation under test converts to
    0-based indices and to a 0-based ``n``; keeping the oracle in the original
    convention is what makes it an independent check of that conversion.
    """
    epmach = float(np.finfo(float).eps)
    oflow = float(np.finfo(float).max)
    epstab = np.zeros(MAX_TABLE_SIZE + 3)  # 1-based, so one longer
    res3la = np.zeros(4)  # 1-based
    n, nres = 0, 0  # `n` is a count of entries, as in the Fortran

    for value in seq:
        n += 1
        epstab[n] = value
        nres += 1
        result, abserr = epstab[n], oflow
        if n >= 3:
            epstab[n + 2] = epstab[n]
            newelm = (n - 1) // 2
            epstab[n] = oflow
            num, k1 = n, n
            for i in range(1, newelm + 1):
                k2, k3 = k1 - 1, k1 - 2
                res = epstab[k1 + 2]
                e0, e1, e2 = epstab[k3], epstab[k2], res
                e1abs = abs(e1)
                delta2, err2 = e2 - e1, abs(e2 - e1)
                tol2 = max(abs(e2), e1abs) * epmach
                delta3, err3 = e1 - e0, abs(e1 - e0)
                tol3 = max(e1abs, abs(e0)) * epmach
                if err2 <= tol2 and err3 <= tol3:
                    result, abserr = e2, err2 + err3
                    break
                e3 = epstab[k1]
                epstab[k1] = e1
                delta1, err1 = e1 - e3, abs(e1 - e3)
                tol1 = max(e1abs, abs(e3)) * epmach
                # `e3` is the marker on the first pass, not a real entry; QUADPACK
                # leans on `oflow` arithmetic to drop its term, which overflows in
                # half precision, so both this and the implementation say it outright.
                if (i != 1 and err1 <= tol1) or err2 <= tol2 or err3 <= tol3:
                    n = i + i - 1
                    break
                ss = (0.0 if i == 1 else 1.0 / delta1) + 1.0 / delta2 - 1.0 / delta3
                if abs(ss * e1) <= IRREGULAR:
                    n = i + i - 1
                    break
                res = e1 + 1.0 / ss
                epstab[k1] = res
                k1 -= 2
                error = err2 + abs(res - e2) + err3
                if error < abserr:
                    abserr, result = error, res
            # shift the table
            if n == MAX_TABLE_SIZE:
                n = 2 * (MAX_TABLE_SIZE // 2) - 1
            ib = 2 if num % 2 == 0 else 1
            for _ in range(newelm + 1):
                epstab[ib] = epstab[ib + 2]
                ib += 2
            if num != n:
                indx = num - n + 1
                for i in range(1, n + 1):
                    epstab[i] = epstab[indx]
                    indx += 1
        if nres < 4:
            res3la[nres] = result
            abserr = oflow
        else:
            abserr = sum(abs(result - res3la[i]) for i in (1, 2, 3))
            res3la[1], res3la[2], res3la[3] = res3la[2], res3la[3], result
        yield result, max(abserr, ERR_FLOOR * epmach * abs(result))


class TestMatchesQuadpack:
    """The masked rewrite must reproduce the Fortran exactly, step for step."""

    @pytest.mark.parametrize(
        "name, seq",
        [
            ("geometric", 10.0 - 3.0 * 0.8 ** np.arange(16)),
            ("slow-geometric", 10.0 - 3.0 * 0.9931 ** np.arange(30)),
            (
                "two-modes",
                10.0 - 3.0 * 0.7 ** np.arange(24) - 1.5 * 0.9 ** np.arange(24),
            ),
            ("alternating", np.cumsum((-1.0) ** np.arange(24) / (np.arange(24) + 1))),
            ("divergent", np.cumsum(1.0 / (np.arange(24) + 1.0))),
            ("negative", -4.0 + 2.0 * 0.85 ** np.arange(20)),
            ("constant", np.full(14, 3.25)),
        ],
    )
    def test_step_for_step(self, name, seq):
        got = accelerate(seq)
        for k, ((ref, _), state) in enumerate(zip(dqelg_reference(seq), got)):
            np.testing.assert_allclose(
                float(state.result),
                ref,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"{name}: diverged from QUADPACK at term {k}",
            )


class TestKnownLimits:
    """Sequences whose limits are known in closed form."""

    def test_single_geometric_is_annihilated(self):
        """One geometric error mode is exactly what a Shanks transform removes."""
        seq = 10.0 - 3.0 * 0.8 ** np.arange(20)
        assert best_relerr(seq, 10.0) < 1e-14

    @pytest.mark.parametrize("r", [0.7071, 0.9330, 0.9931])
    def test_slow_geometric(self, r):
        """The ratios a ``x**-alpha`` endpoint singularity actually produces.

        ``r = 2**-(1-alpha)`` for alpha = 0.5, 0.9, 0.99. The last is the case no
        sampling based method can reach at all, which is the point of the exercise.
        """
        seq = 10.0 - 3.0 * r ** np.arange(30)
        plain = abs(seq[-1] - 10.0) / 10.0
        best = best_relerr(seq, 10.0)
        assert best < 1e-10
        assert best < plain / 100  # a large gain, not a marginal one

    def test_two_geometric_modes(self):
        """Column 2k is exact for a sum of k modes, so two need a deeper table."""
        n = np.arange(30)
        assert best_relerr(10.0 - 3.0 * 0.7**n - 1.5 * 0.9**n, 10.0) < 1e-12

    def test_alternating_series(self):
        """``sum (-1)^n/(n+1) -> log 2``, slowly convergent and strongly alternating."""
        n = np.arange(30)
        assert best_relerr(np.cumsum((-1.0) ** n / (n + 1)), np.log(2)) < 1e-12


class TestDegenerateInput:
    """The guards exist for these; none may produce a NaN or an exception."""

    def _finite(self, states):
        assert all(np.all(np.isfinite(np.asarray(st.result))) for st in states)

    def test_constant_sequence(self):
        """Every delta is exactly zero, so every denominator would be too."""
        states = accelerate(np.full(12, 3.25))
        self._finite(states)
        np.testing.assert_allclose(float(states[-1].result), 3.25)

    def test_already_converged(self):
        """Agreement to machine precision trips the short circuit, not a division."""
        seq = 5.0 + np.array([1e-16, -1e-16, 0.0, 1e-16, 0.0, -1e-16, 0.0, 1e-16])
        states = accelerate(seq)
        self._finite(states)
        np.testing.assert_allclose(float(states[-1].result), 5.0, atol=1e-14)

    def test_divergent_sequence(self):
        """``sum 1/n`` has no limit; the table must not invent one or blow up."""
        states = accelerate(np.cumsum(1.0 / (np.arange(30) + 1.0)))
        self._finite(states)

    def test_structureless_noise(self):
        """No asymptotic structure at all: stay finite, do not fabricate a limit."""
        rng = np.random.default_rng(0)
        self._finite(accelerate(10.0 + 1e-3 * rng.standard_normal(30)))

    @pytest.mark.parametrize("n_terms", [1, 2, 3])
    def test_too_few_terms(self, n_terms):
        """Under three terms there is nothing to extrapolate from."""
        states = accelerate(np.arange(n_terms) + 1.0)
        self._finite(states)
        np.testing.assert_allclose(float(states[-1].result), float(n_terms))

    def test_more_terms_than_the_table_holds(self):
        """Past ``MAX_TABLE_SIZE`` the table recycles rather than overrunning."""
        n = np.arange(2 * MAX_TABLE_SIZE)
        states = accelerate(10.0 - 3.0 * 0.9**n)
        self._finite(states)
        assert min(abs(float(st.result) - 10.0) for st in states) < 1e-8

    def test_negative_values(self):
        """The sentinel marking the newest slot must not break on negative sums."""
        states = accelerate(-4.0 + 2.0 * 0.85 ** np.arange(20))
        self._finite(states)
        assert min(abs(float(st.result) + 4.0) for st in states) < 1e-10


class TestTransforms:
    """The table has to survive every JAX transformation the integrator applies."""

    @staticmethod
    def _run(seq):
        state = init_table((), seq.dtype)
        return jax.lax.fori_loop(
            0,
            seq.shape[0],
            lambda i, st: extrapolate(st, seq[i], _inf_norm),
            state,
        ).result

    def test_jit(self):
        seq = jnp.asarray(10.0 - 3.0 * 0.8 ** np.arange(20))
        np.testing.assert_allclose(
            float(jax.jit(self._run)(seq)), float(self._run(seq))
        )

    def test_vmap(self):
        """Batched sequences must each get their own table, not a shared one."""
        n = np.arange(20)
        seqs = jnp.stack([jnp.asarray(10.0 - 3.0 * r**n) for r in (0.7, 0.8, 0.9)])
        got = jax.vmap(self._run)(seqs)
        one_at_a_time = jnp.stack([self._run(s) for s in seqs])
        np.testing.assert_allclose(np.asarray(got), np.asarray(one_at_a_time))

    def test_gradients_are_finite(self):
        """The denominators sit at roundoff by construction, so this is the real risk.

        A NaN here would mean a division was not sanitized before being selected away.
        """
        n = np.arange(14)
        seq = jnp.asarray(10.0 - 3.0 * 0.9**n + 1e-13 * np.sin(7.0 * n))
        assert np.all(np.isfinite(np.asarray(jax.grad(self._run)(seq))))

    def test_forward_and_reverse_agree(self):
        """A fixed-size table is a fixed rational map, so the modes must match.

        Checked this way rather than against a finite difference: the map amplifies
        by up to ~1e4 and a central difference on it is itself wrong by up to 100%,
        so an FD comparison would fail a correct implementation.
        """
        n = np.arange(14)
        seq = jnp.asarray(10.0 - 3.0 * 0.9**n + 1e-13 * np.sin(7.0 * n))
        tangent = jnp.asarray(np.random.default_rng(0).standard_normal(len(seq)))
        fwd = jax.jvp(self._run, (seq,), (tangent,))[1]
        rev = jnp.dot(jax.grad(self._run)(seq), tangent)
        np.testing.assert_allclose(float(fwd), float(rev), rtol=1e-10)

    def test_gradient_matches_complex_step(self):
        """A derivative of the same map obtained without autodiff or cancelling.

        A finite difference cannot check this map, as ``test_forward_and_reverse_agree``
        notes. A complex step can: perturbing one entry by ``i*h`` and reading the
        derivative off the imaginary part involves no subtraction of nearly-equal
        numbers, so it stays exact for an ``h`` far below the roundoff scale.

        The premise is that such an ``h`` cannot move any of the guards, since every
        one of them tests a magnitude and ``|x + i*h| == |x|`` to working precision.
        The complex run therefore takes the same branches as the real one and the two
        derivatives are of the same rational map; the equality of the two results
        asserted first is what confirms it.
        """
        n = np.arange(14)
        seq = 10.0 - 3.0 * 0.9**n + 1e-13 * np.sin(7.0 * n)
        h = 1e-100

        steps = np.empty(len(seq))
        for i in range(len(seq)):
            perturbed = jnp.asarray(seq, jnp.complex128).at[i].add(1j * h)
            result = complex(self._run(perturbed))
            assert result.real == float(self._run(jnp.asarray(seq)))
            steps[i] = result.imag / h

        grad = np.asarray(jax.grad(self._run)(jnp.asarray(seq)))
        # Loose for a derivative comparison, and necessarily so: the two runs divide by
        # the same near-zero differences along different arithmetic paths, real and
        # complex, and the table amplifies the discrepancy between them.
        np.testing.assert_allclose(grad, steps, rtol=1e-8)


class TestDTypes:
    """The table is a scan carry, so its dtype has to be stable and correct."""

    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_dtype_is_preserved(self, dtype):
        state = accelerate(10.0 - 3.0 * 0.8 ** np.arange(12), dtype=dtype)[-1]
        assert state.result.dtype == dtype
        assert state.table.dtype == dtype
        assert state.abserr.dtype == dtype
        assert np.isfinite(float(state.result))

    @pytest.mark.parametrize("dtype", real_dtypes)
    def test_does_no_harm_at_any_precision(self, dtype):
        """What is reachable depends on the dtype, so this is not a fixed tolerance.

        At float16 and bfloat16 the adaptive loop hits its sub-interval width floor
        after only a handful of bisections, so in practice the table never receives
        enough terms to do much. The requirement here is that it does no harm.
        """
        seq = 10.0 - 3.0 * 0.8 ** np.arange(20)
        eps = float(jnp.finfo(dtype).eps)
        plain = abs(float(jnp.asarray(seq[-1], dtype)) - 10.0) / 10.0
        assert best_relerr(seq, 10.0, dtype=dtype) <= max(plain, 100 * eps)

    @pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
    def test_precision_buys_accuracy(self, dtype):
        """Where there are enough terms, the result tracks the working precision."""
        seq = 10.0 - 3.0 * 0.8 ** np.arange(20)
        assert best_relerr(seq, 10.0, dtype=dtype) < 100 * float(jnp.finfo(dtype).eps)

    @pytest.mark.parametrize("dtype", complex_dtypes)
    def test_complex_integrand(self, dtype):
        """Complex values, real error estimate."""
        n = np.arange(20)
        seq = (10.0 - 3.0 * 0.8**n) + 1j * (2.0 - 0.5 * 0.8**n)
        state = accelerate(seq, dtype=dtype)[-1]
        assert state.result.dtype == dtype
        assert not jnp.iscomplexobj(state.abserr)
        assert best_relerr(seq, 10.0 + 2.0j, dtype=dtype) < 1e-6


class TestVectorValued:
    """Arithmetic is per component; the structural decisions are shared."""

    def test_independent_components(self):
        n = np.arange(20)
        seq = np.stack(
            [10.0 - 3.0 * 0.8**n, 5.0 + 2.0 * 0.7**n, -1.0 - 0.5 * 0.9**n], axis=1
        )
        states = accelerate(seq, shape=(3,))
        best = min(
            states,
            key=lambda st: np.abs(np.asarray(st.result) - [10.0, 5.0, -1.0]).max(),
        )
        np.testing.assert_allclose(
            np.asarray(best.result), [10.0, 5.0, -1.0], rtol=1e-8
        )

    def test_error_estimate_is_scalar(self):
        n = np.arange(20)
        seq = np.stack([10.0 - 3.0 * 0.8**n, 5.0 + 2.0 * 0.7**n], axis=1)
        assert jnp.ndim(accelerate(seq, shape=(2,))[-1].abserr) == 0

    @pytest.mark.parametrize(
        "settled", [0.0, 7.0, -2.5], ids=["zero", "positive", "negative"]
    )
    def test_a_component_that_has_already_converged_exactly(self, settled):
        """One component at its limit must not spoil the ones still moving.

        The structural decisions go through the norm, which is right for them - they
        are single integers - but the divisions are per component. A component that has
        reached its limit exactly has differences of exactly zero while the norm, driven
        by whichever component is still moving, says they are comfortably above
        tolerance. Dividing on that verdict produces an infinity, `inf - inf` turns the
        whole diagonal into NaN, and no element of it is ever kept: the table stops
        producing anything at all rather than failing loudly.

        Reached by any vector valued integrand with a component the local rule
        integrates without error.
        """
        n = np.arange(20)
        moving = 10.0 - 3.0 * 0.8**n
        seq = np.stack([moving, np.full_like(moving, settled)], axis=1)
        states = accelerate(seq, shape=(2,))
        assert all(np.all(np.isfinite(np.asarray(st.result))) for st in states)
        best = min(states, key=lambda st: abs(float(np.asarray(st.result)[0]) - 10.0))
        np.testing.assert_allclose(np.asarray(best.result), [10.0, settled], atol=1e-10)
        # and it does as well as it would have on its own
        alone = min(
            accelerate(moving), key=lambda st: abs(float(np.asarray(st.result)) - 10.0)
        )
        assert abs(float(np.asarray(best.result)[0]) - 10.0) <= abs(
            float(np.asarray(alone.result)) - 10.0
        )


class TestSequencesThatDiscriminateAlgorithms:
    """Sequences where the choice of acceleration method visibly matters.

    Wynn's epsilon was selected over Brezinski's theta, Levin t/u/v and Weniger's delta
    by measurement on the sequences the integrator actually produces. These pin the
    properties that decided it, and the one place epsilon is known to be the wrong tool,
    so that swapping the algorithm cannot quietly change the trade.
    """

    def test_the_model_sequence_of_bisecting_toward_a_singularity(self):
        """The exact running totals for ``int_0^1 x**-alpha`` under bisection.

        ``s_n = int_{2**-n}^1 x**-a dx`` has error exactly ``C r**n`` with
        ``r = 2**-(1-a)`` - one pure geometric mode, which is precisely what a Shanks
        transform annihilates. This is the structure the whole method rests on, in
        closed form and free of any quadrature error.
        """
        for a, target in [(0.5, 1e-15), (0.9, 1e-14), (0.99, 1e-13)]:
            exact = 1 / (1 - a)
            seq = [(1 - 2.0 ** (-n * (1 - a))) / (1 - a) for n in range(1, 15)]
            assert best_relerr(seq, exact) < target

    def test_logarithmic_convergence_is_not_accelerated(self):
        """``sum 1/n**2 -> pi**2/6`` is the case Wynn's epsilon cannot do.

        Its error decays like ``1/n`` rather than geometrically, and epsilon assumes the
        latter. Levin's u transform reaches 1.8e-11 on this sequence where epsilon
        manages 1.5e-03. That is a real limitation and is recorded rather than hidden,
        but it does not matter here, because no adaptive quadrature sequence in the
        suite converges logarithmically (logarithmic singularities are integrated to
        machine precision already without extrapolation). If a future change makes this
        case work, the algorithm has been swapped and the rest of these trade-offs need
        revisiting.
        """
        n = np.arange(60)
        got = best_relerr(np.cumsum(1.0 / (n + 1) ** 2), np.pi**2 / 6)
        assert got > 1e-6  # not accelerated
        assert got < 1e-1  # but not made worse either

    def test_three_geometric_modes(self):
        """Column ``2k`` is exact for ``k`` modes, so three need a deeper table."""
        n = np.arange(40)
        seq = 10.0 - 3.0 * 0.7**n - 1.5 * 0.9**n - 0.4 * 0.95**n
        assert best_relerr(seq, 10.0) < 1e-10

    def test_the_best_estimate_is_not_the_last_one(self):
        """Why the integrator keeps a running best instead of the final call.

        The table shifts entries out between calls and its error estimate is not
        monotone, so a later diagonal is not reliably a better one. On a clean geometric
        sequence the final value is orders of magnitude worse than the best seen along
        the way; reporting the last one would throw away almost all of the benefit.
        """
        seq = 10.0 - 3.0 * 0.8 ** np.arange(30)
        states = accelerate(seq)
        best = min(abs(float(st.result) - 10.0) for st in states)
        final = abs(float(states[-1].result) - 10.0)
        assert best < final / 100

    def test_a_sequence_that_stalls_into_roundoff(self):
        """Geometric while it can be, then flat once the terms hit the noise floor.

        This is what the tail of a real solve looks like once sub-intervals stop being
        resolvable. The table must hold whatever it had rather than extrapolate the
        noise that follows.
        """
        n = np.arange(30)
        seq = np.where(n < 12, 10.0 - 3.0 * 0.8**n, 10.0 - 3.0 * 0.8**12)
        states = accelerate(seq)
        assert all(np.all(np.isfinite(np.asarray(st.result))) for st in states)
        assert min(abs(float(st.result) - 10.0) for st in states) < 1e-9

    def test_alternating_and_monotone_sequences_both_work(self):
        """Levin's t and u transforms differ sharply on these; epsilon should not.

        The remainder estimate a Levin transform needs is modelled on one or the other,
        so picking wrongly costs decades. Epsilon estimates nothing and handles both.
        """
        n = np.arange(30)
        monotone = 10.0 - 3.0 * 0.8**n
        alternating = 10.0 - 3.0 * (-0.8) ** n
        assert best_relerr(monotone, 10.0) < 1e-13
        assert best_relerr(alternating, 10.0) < 1e-13
