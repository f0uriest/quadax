"""Tests for Romberg's method and its tanh-sinh variant.

These live apart from the adaptive solvers because ``extrapolate`` means something
different here. On ``romberg`` it selects Richardson extrapolation, defaults to
``True``, and turning it off leaves plain trapezoidal quadrature rather than a tuned
version of the same method; on the adaptive routines it is Wynn's epsilon algorithm
over the running totals and defaults to ``False``. Romberg also takes ``divmax`` where
they take ``max_ninter``, and rejects breakpoints outright.
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
