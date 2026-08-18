"""Tests for Romberg's method and its tanh-sinh variant.

These live apart from the adaptive solvers because ``extrapolate`` means something
different here. On ``romberg`` it selects Richardson extrapolation, and turning it off
leaves plain trapezoidal quadrature rather than a tuned version of the same method; on
the adaptive routines it is Wynn's epsilon algorithm over the running totals, layered
on a rule that converges without it. Romberg also takes ``divmax`` where they take
``max_ninter``, and rejects breakpoints outright.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

import quadax
from quadax import romberg, rombergts

from .problems import (
    NO_BREAKPOINTS,
    PROBLEMS,
    RICHARDSON_MODEL,
    ROMBERG_CONVERGES,
    ROMBERGTS_CONVERGES,
    SMOOTH_FINITE,
    TOLS,
    ULP_ATOL,
    ULP_RTOL,
    assert_contract,
    problem_id,
    xfail_if_known,
)

config.update("jax_enable_x64", True)

METHODS = [romberg, rombergts]
METHOD_IDS = ["romberg", "ts"]

# Which problems each variant is expected to solve. Unlike the adaptive routines there
# is no setting under which Romberg solves the whole suite, so the claim is made per
# method rather than keyed off a flag.
EXPECTED_TO_CONVERGE = {romberg: ROMBERG_CONVERGES, rombergts: ROMBERGTS_CONVERGES}


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize("tol", TOLS, ids=[f"{t:g}" for t in TOLS])
@pytest.mark.parametrize("i", NO_BREAKPOINTS, ids=problem_id)
class TestRomberg:
    """Every problem Romberg accepts, at every tolerance.

    The contract is the one the adaptive routines are held to: reaching the tolerance is
    required only of the problems the method is for, and everything else has merely to
    report its failure with an error estimate that does not understate the true error.
    """

    def test_value_and_error(self, request, method, tol, i):
        """The answer is good to the tolerance asked for, and the error is honest."""
        prob = PROBLEMS[i]
        xfail_if_known(request, method, prob, tol)
        y, info = method(
            prob["fun"],
            jnp.asarray(prob["interval"], float),
            epsabs=jnp.asarray(tol),
            epsrel=jnp.asarray(tol),
        )
        if i in EXPECTED_TO_CONVERGE[method]:
            assert int(info.status) == 0, (
                f"{prob['name']} at tol={tol:g}: {quadax.STATUS[int(info.status)]}"
            )
        assert_contract(y, info, prob, tol, model=RICHARDSON_MODEL)


class TestRichardsonFlag:
    """Richardson extrapolation, on and off, over the whole example suite.

    Romberg's method *is* the trapezoidal rule plus Richardson, so this flag turns it
    into something else rather than merely tuning it: the same nodes and the same
    halving schedule, reading the un-extrapolated column. The two methods want opposite
    things from it, which is why it is a flag and not a fixed choice.

    ``divmax`` is well below the default here so the plain mode is affordable to test.
    Without extrapolation the trapezoidal rule needs O(h**2) refinement, so its
    evaluation count runs to millions on the harder problems, and the contrast the tests
    below check for is visible long before that.
    """

    DIVMAX = 14
    TOL = 1e-8

    def _run(self, method, i, extrapolate, tol=None, divmax=None):
        prob = PROBLEMS[i]
        y, info = method(
            prob["fun"],
            jnp.asarray(prob["interval"], float),
            epsabs=tol or self.TOL,
            epsrel=tol or self.TOL,
            divmax=divmax or self.DIVMAX,
            full_output=True,
            extrapolate=extrapolate,
        )
        exact = np.asarray(prob["val"])
        # Floored at one, so a problem whose exact value is zero measures absolute
        # error rather than dividing by it.
        scale = max(np.max(np.abs(exact)), 1.0)
        err = float(np.max(np.abs(np.asarray(y) - exact)) / scale)
        return y, err, int(info.status), int(info.neval)

    def test_richardson_is_what_makes_romberg_work(self):
        """Without it the trapezoidal rule cannot keep up on a smooth integrand.

        This is the justification for the flag defaulting to on. The gap is not
        marginal: Richardson reaches machine precision in tens of evaluations where
        bisection alone is still several digits short after tens of thousands.

        Finite limits only, since the map onto an infinite range already buys the
        trapezoidal rule exponential convergence and leaves Richardson little to add.
        Problems the trapezoidal rule already integrates to roundoff are excluded as
        well: with both settings sitting on the floor there is no gap for the flag to
        open and which of the two is nearer is noise, so the claim is made only where
        accuracy has to be earned.
        """
        for i in SMOOTH_FINITE:
            _, err_on, _, neval_on = self._run(romberg, i, True)
            _, err_off, _, neval_off = self._run(romberg, i, False)
            if err_off < 1e-15:
                continue
            assert err_on < err_off, f"problem {i}: {err_on:.2e} vs {err_off:.2e}"
            assert neval_on < neval_off, f"problem {i}"

    def test_tanh_sinh_gains_nothing_from_richardson(self):
        """On tanh-sinh it is at best neutral, and usually just costs evaluations.

        The rule already converges doubly exponentially, so there is no expansion in
        powers of the step for Richardson to cancel. Measured over the suite it helps
        nothing, is slightly worse on a few problems, and reaches the same accuracy in
        fewer evaluations on around half of them.
        """
        for i in NO_BREAKPOINTS:
            _, err_on, _, _ = self._run(rombergts, i, True)
            _, err_off, _, _ = self._run(rombergts, i, False)
            floor = 1e-14 * max(np.max(np.abs(np.asarray(PROBLEMS[i]["val"]))), 1)
            assert err_off <= max(10 * err_on, floor), (
                f"problem {i}: turning extrapolation off made it worse, "
                f"{err_off:.2e} vs {err_on:.2e}"
            )

    @pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
    def test_the_flag_does_not_change_the_table_shape(self, method):
        """``full_output`` keeps its contract either way, column 0 always filled.

        The two settings do not generally stop at the same level (they are comparing
        different estimates from one refinement to the next, so they meet the tolerance
        at different depths) but the trapezoidal column is the same computation in
        both, so wherever they both reached it holds the same numbers. Only what is
        built on top of it differs.
        """
        prob = PROBLEMS[0]
        tables = {}
        for extrapolate in (False, True):
            _, info = method(
                prob["fun"],
                jnp.asarray(prob["interval"], float),
                epsabs=self.TOL,
                epsrel=self.TOL,
                divmax=self.DIVMAX,
                full_output=True,
                extrapolate=extrapolate,
            )
            tables[extrapolate] = np.asarray(info.info)
        assert tables[False].shape == tables[True].shape
        columns = [tables[e][:, 0] for e in (False, True)]
        # How far each column got: the length of its leading run of nonzeros. Counted
        # as a run rather than located as the first zero, because a column filled to
        # the end has no zero in it and a search would have to report its absence.
        depth = min(*(int((c != 0).cumprod().sum()) for c in columns), self.DIVMAX)
        assert depth > 1, "neither setting filled the trapezoidal column"
        np.testing.assert_allclose(
            columns[0][:depth], columns[1][:depth], rtol=ULP_RTOL, atol=ULP_ATOL
        )


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
class TestBatchSize:
    """Evaluating a level's new points in parallel rather than one at a time.

    A level places ``2**(k-1)`` new points, a count only known at run time, so they are
    evaluated in fixed size batches with the last one padded. The padding is what lets a
    single batch shape serve every level; the price is that early levels, which place
    fewer points than a batch holds, still pay for a whole one.

    Batching reorders the sum over a level, from strictly sequential to pairwise
    within each batch, so the answers move in the last bits. That is why the comparisons
    here are to a tolerance rather than exact.
    """

    DIVMAX = 8
    BATCH_SIZES = [1, 3, 8, 16, 64]

    def _run(self, method, i, batch_size, divmax=None):
        """Run to a fixed depth, so every batch size does the same amount of work.

        ``epsabs=epsrel=0`` stops the tolerance from being met, so the loop always
        exhausts ``divmax``. Without that a batch size could stop a level earlier or
        later than the serial run purely on the last-bit differences above, and the
        comparison would be between two different discretizations.
        """
        prob = PROBLEMS[i]
        return method(
            prob["fun"],
            jnp.asarray(prob["interval"], float),
            epsabs=0.0,
            epsrel=0.0,
            divmax=divmax or self.DIVMAX,
            full_output=True,
            batch_size=batch_size,
        )

    def _depth(self, info, batch_size):
        """How many refinement levels a run performed, read off its evaluation count.

        For a fixed batch size ``neval`` is strictly increasing in the depth, so it
        identifies it. Taken from the count rather than from where the table stops being
        filled, because a problem whose integral is zero fills it with zeros and there
        nothing to find.
        """
        b = batch_size or 1
        total = 2
        for level in range(self.DIVMAX + 1):
            if total == int(info.neval):
                return level
            total += -(-(2**level) // b) * b
        raise AssertionError(f"neval={int(info.neval)} matches no depth at b={b}")

    @pytest.mark.parametrize("i", SMOOTH_FINITE, ids=problem_id)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=str)
    def test_matches_the_serial_run(self, method, i, batch_size):
        """At the same depth, batching changes only the rounding of the sums.

        The trapezoidal column is compared as well as the value, since it is what the
        batched sum feeds directly and the rest of the table is built from it by a
        deterministic recurrence.

        Only as far as both runs got. Zeroing the tolerance does not quite pin depth:
        the loop also stops when two successive estimates agree to the last bit, on a
        problem this converges on, whether that happens at the second to last level is
        decided by a difference of a few ulp that batching is entitled to move. So the
        depths can differ by one, and ``status`` (which under a zero tolerance reports
        only whether that difference was nonzero) is not compared at all.
        """
        want, want_info = self._run(method, i, None)
        got, got_info = self._run(method, i, batch_size)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=1e-11, atol=1e-13
        )
        columns = [np.asarray(o.info)[:, 0] for o in (got_info, want_info)]
        depth = min(self._depth(got_info, batch_size), self._depth(want_info, None))
        assert depth > 1, "neither run filled the trapezoidal column"
        np.testing.assert_allclose(
            columns[0][:depth], columns[1][:depth], rtol=1e-11, atol=1e-13
        )

    @pytest.mark.parametrize("i", SMOOTH_FINITE, ids=problem_id)
    def test_serial_is_the_default(self, method, i):
        """``batch_size=None`` and ``batch_size=1`` are the same computation.

        One point per batch leaves the mask always true and the sum over a single row,
        so no term is reordered and the two agree to roundoff rather than merely to
        quadrature accuracy. Not bitwise: the batched form still evaluates the integrand
        under a ``vmap`` of width one, and XLA is free to fuse that differently.
        """
        want, _ = self._run(method, i, None)
        got, _ = self._run(method, i, 1)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=ULP_RTOL, atol=ULP_ATOL
        )

    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=str)
    def test_neval_counts_the_padding(self, method, batch_size):
        """Padded lanes are real calls to the integrand and are reported as such."""
        _, info = self._run(method, 0, batch_size)
        expected = 2 + sum(
            -(-(2 ** (k - 1)) // batch_size) * batch_size
            for k in range(1, self.DIVMAX + 1)
        )
        assert int(info.neval) == expected

    def test_batch_size_is_clipped_to_the_deepest_level(self, method):
        """No level places more than ``2**(divmax-1)`` points, so nothing above helps.

        Without the clip a generous ``batch_size`` would be padding on every level of a
        shallow run, and ``neval`` would grow without the run doing any more work.
        """
        divmax = 4
        deepest = 2 ** (divmax - 1)
        _, clipped = self._run(method, 0, 10**6, divmax=divmax)
        _, exact = self._run(method, 0, deepest, divmax=divmax)
        assert int(clipped.neval) == int(exact.neval)

    def test_vector_valued(self, method):
        """The mask has to broadcast against the integrand's own trailing axes."""
        fun = lambda x: jnp.array([jnp.sin(x), jnp.cos(x), x**2])  # noqa: E731
        interval = jnp.array([0.0, 1.0])
        want, _ = method(fun, interval, divmax=self.DIVMAX, full_output=True)
        got, _ = method(
            fun, interval, divmax=self.DIVMAX, full_output=True, batch_size=8
        )
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize("batch_size", [0, -1, 2.5], ids=["zero", "negative", "float"])
def test_bad_batch_size_rejected(method, batch_size):
    """A batch size that is not a positive integer is a mistake, not a default."""
    with pytest.raises(ValueError, match="batch_size"):
        method(PROBLEMS[0]["fun"], jnp.array([0.0, 1.0]), batch_size=batch_size)
