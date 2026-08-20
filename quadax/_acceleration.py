"""Convergence acceleration for the sequence of running totals.

A globally adaptive integrator bisecting towards a singularity produces a sequence of
running totals whose error decays geometrically but slowly: for an endpoint
``x**-alpha`` singularity the ratio is ``2**-(1-alpha)``, which tends to 1 as ``alpha``
does. Bisection alone cannot reach the limit, because it can only sample where the
abscissae are still distinguishable from the singular point, and below that width the
mass it is missing is ``h**(1-alpha)`` of the total however good the local rule is.

Wynn's epsilon algorithm reaches the limit anyway, by inferring the tail from the trend
of the sequence rather than sampling it. It computes Shanks transforms iteratively:
given partial sums ``S_n``,

    eps_{-1}^(n) = 0,   eps_0^(n) = S_n
    eps_{k+1}^(n) = eps_{k-1}^(n+1) + 1 / (eps_k^(n+1) - eps_k^(n))

The even columns are the accelerated estimates, equivalently the diagonal Pade
approximants of the generating series, and column ``2k`` is exact for a sequence whose
error is a sum of ``k`` geometric terms.

The catch is that those denominators are differences of nearly-equal numbers by
construction, so a usable implementation is mostly guards. Three of them shape the code
below:

- a short circuit, which stops as soon as three consecutive entries agree to machine
  accuracy, since past that point the recursion is doing arithmetic on noise;
- a truncation, which throws the older part of the table away when a difference is too
  small to divide by, or when the reciprocals so nearly cancel that the correction the
  step would apply dwarfs the value it corrects;
- and reporting the *best* element of a diagonal rather than its last. Each step divides
  by a smaller difference than the one before, so the higher order transforms are more
  accurate in exact arithmetic and noisier in floating point, and the trade-off can turn
  over partway along.

Notes
-----
The value returned by an extrapolation carries no rigorous error bound, and the estimate
reported with it is a heuristic that callers should treat as a guide rather than a
guarantee. It is built to err towards over-reporting: the spread of the three most
recent extrapolants says how far the sequence has moved, and on its own that is
optimistic, because a slowly converging sequence still will have a small spread even
when far from the limit.

The reported estimate is where this departs furthest from QUADPACK [2], which reports
the spread alone. The recursion itself, its guards and their constants are from
QUADPACK. The additions are the tail term, the separate ``abserr_sharp`` that callers
rank by, and in the adaptive loop a widening of the kept estimate when later
extrapolations disagree with the value it belongs to; each is marked where it appears.
They exist because the spread understates the error by more the longer a run goes on,
without limit, on problems whose asymptotics the algorithm cannot fit.

References
----------
.. [1] P. Wynn. "On a Device for Computing the e_m(S_n) Transformation". Mathematical
       Tables and Other Aids to Computation, vol. 10, no. 54, 1956, pp. 91-96.
       doi:10.2307/2002183
.. [2] R. Piessens, E. de Doncker-Kapenga, C. W. Uberhuber, D. K. Kahaner. "QUADPACK: A
       Subroutine Package for Automatic Integration". Springer Series in Computational
       Mathematics, vol. 1. Springer-Verlag, Berlin, 1983. doi:10.1007/978-3-642-61786-7
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .utils import _real_dtype, tree_where

# Cap on the number of partial sums the table holds. Never reached in practice: the
# adaptive loop stops at the sub-interval width floor after ~48 bisections at float64,
# and sooner at coarser precision.
MAX_TABLE_SIZE = 50

# Truncate the table when ``|inv_delta_sum * e1|`` falls below this. The Shanks step
# adds ``1/inv_delta_sum`` to ``e1``, so a small product says the correction is larger
# than the value it corrects by a factor of ``1/IRREGULAR`` or more: the three
# reciprocals have very nearly cancelled (``1/d1 + 1/d2 ~ 1/d3``), and what survives the
# cancellation is roundoff rather than a geometric mode. The value is QUADPACK's [2],
# empirical rather than derived, and is a rarely taken safety valve rather than a tuned
# parameter: changing by several orders of magnitude has no effect on any problems
# tested
IRREGULAR = 1e-4

# Floor under the reported error, as a multiple of eps times the result. No
# extrapolation can be more accurate than the arithmetic that built the table it came
# from. The multiplier is QUADPACK's [2].
ERR_FLOOR = 5.0

# The three constants below have no counterpart in QUADPACK [2], which reports the
# spread of the last three extrapolants and nothing else. They parameterize the tail
# term `_record` adds to that spread, and are set from the behavior of the endpoint
# singularities in the test suite rather than derived.
#
# The remaining tail of a converging extrapolant sequence, as a multiple of the sum of
# its future steps. A ratio estimated from three steps is itself noisy, and for a
# sequence whose steps shrink like a power of the index rather than geometrically the
# geometric sum understates the tail by a factor between one and three; two covers the
# powers that arise from endpoint singularities. See `_record`.
TAIL_SAFETY = 2.0

# Largest step ratio the tail estimate acts on. A sequence whose steps are not shrinking
# has no bounded tail to estimate, so the correction is capped here rather than allowed
# to run to infinity; what the cap says is "much larger than the last step", and the
# caller decides whether that is worse than its own mesh estimate.
RHO_MAX = 0.99

# A step below this multiple of eps times the result is roundoff rather than movement,
# and the ratios formed from it carry no information. Set well above the point where the
# steps are pure noise: a sequence whose extrapolants still disagree, but only in the
# last few digits, has no tail worth estimating, and applying the correction there costs
# a converged answer its status for nothing.
STEP_NOISE = 1e3


class EpsilonTable(NamedTuple):
    """State of an in-progress epsilon algorithm.

    Carried across iterations of the adaptive loop, one new diagonal computed per
    appended partial sum. All fields are fixed-shape.

    Parameters
    ----------
    table : Array
        The working table, ``(MAX_TABLE_SIZE + 2, *shape)``. Holds one diagonal at a
        time; entries the recursion has consumed are shifted out after each call.
    n : int
        Number of entries currently in ``table``.
    n_calls : int
        How many extrapolations have been performed, ie calls to ``extrapolate`` rather
        than appends. The error estimate needs three of them before it means anything.
    last_results : Array
        The three most recent extrapolated values, ``(3, *shape)``, whose spread is the
        error estimate.
    result : Array
        Value from the most recent extrapolation, shape ``shape``. Whether it is an
        improvement on the one before is for the caller to decide and remember.
    abserr : Array
        Estimated absolute error in ``result``, a real scalar.
    abserr_sharp : Array
        How tightly this extrapolation settled, ie the spread of the last three
        extrapolants alone. It is the right quantity for ranking one extrapolation
        against another and the wrong one to report, since it measures how far the
        sequence has moved rather than how far it has left to go.
    """

    table: jax.Array
    n: jax.Array
    n_calls: jax.Array
    last_results: jax.Array
    result: jax.Array
    abserr: jax.Array
    abserr_sharp: jax.Array


def init_table(shape: tuple[int, ...], ytype: Any) -> EpsilonTable:
    """Empty table for an integrand of the given shape and dtype."""
    etype = _real_dtype(ytype)
    return EpsilonTable(
        table=jnp.zeros((MAX_TABLE_SIZE + 2, *shape), ytype),
        n=jnp.zeros((), int),
        n_calls=jnp.zeros((), int),
        last_results=jnp.zeros((3, *shape), ytype),
        result=jnp.zeros(shape, ytype),
        # Starts at infinity so that the first genuine estimate always improves on it.
        abserr=jnp.array(jnp.inf, etype),
        abserr_sharp=jnp.array(jnp.inf, etype),
    )


def append(state: EpsilonTable, value: jax.Array) -> EpsilonTable:
    """Record a partial sum without extrapolating from it."""
    return state._replace(table=state.table.at[state.n].set(value), n=state.n + 1)


def ready(state: EpsilonTable) -> jax.Array:
    """Whether the next partial sum fed to the table can be extrapolated from."""
    # The recursion needs three entries before it can take a Shanks step at all.
    return state.n > 1


def step(state: EpsilonTable, value: jax.Array, norm: Callable):
    """Feed a partial sum to the table, extrapolating from it once there is enough.

    Until the table holds enough entries the value is only recorded, and those calls
    must not count as extrapolations for the purpose of the error estimate; ``ready``
    says which of the two happened.
    """
    return tree_where(
        ready(state), extrapolate(state, value, norm), append(state, value)
    )


def _gather(table, idx):
    """Reorder the table's leading axis, clipped so no index runs off the end."""
    return jnp.take(table, jnp.clip(idx, 0, table.shape[0] - 1), axis=0)


def _safe_reciprocal(delta: jax.Array, ok: jax.Array) -> jax.Array:
    """``1/delta`` where ``ok``, and zero elsewhere."""
    safe = jnp.where(ok, delta, 1.0)
    return jnp.where(ok, 1.0 / safe, 0.0)


@eqx.filter_jit
def extrapolate(state: EpsilonTable, value: jax.Array, norm: Callable) -> EpsilonTable:
    """Append a partial sum and compute one new diagonal of the epsilon table.

    Parameters
    ----------
    state : EpsilonTable
        Table state from the previous call, or ``init_table``.
    value : Array
        The new partial sum, ie the integrator's running total.
    norm : callable
        Reduction used to compare vector valued quantities, as elsewhere in quadax. The
        table's arithmetic is per component, but its structural decisions (where to
        truncate, when to stop) are single integers, so the comparisons driving them go
        through ``norm``.

    Returns
    -------
    state : EpsilonTable
        Updated state. ``state.result`` and ``state.abserr`` are the current best
        extrapolated value and its estimated error.
    """
    table = state.table.at[state.n].set(value)
    n_entries = state.n + 1
    n_calls = state.n_calls + 1
    ytype = table.dtype
    etype = _real_dtype(ytype)
    epmach = float(jnp.finfo(etype).eps)
    # ``oflow`` marks a slot as unusable
    oflow = float(jnp.finfo(etype).max)

    # 0-based index of the newest element, one less than the number of entries. Both the
    # truncation inside the loop and the parity of the sublattice the shift at the end
    # walks are stated in terms of this index rather than of the count.
    n = n_entries - 1
    result = table[n]
    abserr = jnp.array(oflow, etype)

    # Fewer than three entries: nothing to extrapolate from yet, just record the value.
    too_short = n < 2

    # Stash the newest entry two slots up and mark its own slot unusable, so that the
    # recursion reads it as `e2` and can never compare against it. Both writes are
    # skipped while the table is too short: the only entries it holds then are the
    # partial sums themselves, and the marker would overwrite one of them.
    table = table.at[n + 2].set(jnp.where(too_short, table[n + 2], table[n]))
    newelm = n // 2
    table = table.at[n].set(jnp.where(too_short, table[n], jnp.array(oflow, ytype)))
    num = n

    def body(i, carry):
        table, k1, result, abserr, done, n_out = carry
        # Iterations past `newelm`, and everything after a short circuit or truncation,
        # are no-ops. `fori_loop` runs the full static range and masking stands in for
        # the jumps; the range is at most 25, so the cost is nothing next to an
        # integrand evaluation.
        active = (i <= newelm) & ~done & ~too_short
        # Clamped so the gathers below stay in bounds on the inactive iterations. The
        # real recursion never reaches k1 < 2, because `newelm = n // 2` bounds it.
        k1 = jnp.maximum(k1, 2)

        e0, e1, e2 = table[k1 - 2], table[k1 - 1], table[k1 + 2]
        e1abs = norm(e1)
        delta2, delta3 = e2 - e1, e1 - e0
        err2, err3 = norm(delta2), norm(delta3)
        tol2 = jnp.maximum(norm(e2), e1abs) * epmach
        tol3 = jnp.maximum(e1abs, norm(e0)) * epmach
        # The same closeness tests again, per component. The recursion's arithmetic is
        # component-wise while its structural decisions are single integers and so go
        # through `norm`, and a division guarded only by the norm is a division a
        # component can still fail: one component of a vector valued integrand can reach
        # its limit exactly - a component that is identically zero, or one the local
        # rule integrates without error - while another is still moving. Its
        # differences are then exactly zero, the norm is whatever the moving component
        # says, and the guard waves through a `1/0` that turns the whole diagonal into
        # NaN. For a scalar integrand these are the same tests as above and nothing
        # changes.
        c_tol2 = jnp.maximum(jnp.abs(e2), jnp.abs(e1)) * epmach
        c_tol3 = jnp.maximum(jnp.abs(e1), jnp.abs(e0)) * epmach

        # Three consecutive entries agreeing to machine accuracy: the table has
        # converged, and anything further would be arithmetic on noise.
        converged = active & (err2 <= tol2) & (err3 <= tol3)

        e3 = table[k1]
        table = table.at[k1].set(jnp.where(active, e1, table[k1]))
        delta1 = e1 - e3
        err1 = norm(delta1)
        tol1 = jnp.maximum(e1abs, norm(e3)) * epmach
        c_tol1 = jnp.maximum(jnp.abs(e1), jnp.abs(e3)) * epmach

        # On the first iteration of a diagonal there is no earlier diagonal to reach
        # back to, so `e3` is the marker written above rather than a real table entry
        # and its term has to drop out of `inv_delta_sum`. Testing for that iteration
        # directly is exact at every precision. QUADPACK gets it from the arithmetic
        # instead: `e1 - oflow` is enormous, so `1/delta1` underflows to nothing and
        # the closeness test passes, but that relies on the subtraction not overflowing,
        # which it does in half precision whenever `e1` is negative, leaving
        # `inf <= inf` and truncating the table on the spot.
        first = i == 1
        delta1_ok = first | (err1 > tol1)

        # Two entries too close to divide by, or a near-degenerate `inv_delta_sum`:
        # truncate the table rather than amplify noise.
        too_close = ~delta1_ok | (err2 <= tol2) | (err3 <= tol3)
        inv_delta_sum = (
            _safe_reciprocal(
                delta1, ~first & (err1 > tol1) & (jnp.abs(delta1) > c_tol1)
            )
            + _safe_reciprocal(delta2, (err2 > tol2) & (jnp.abs(delta2) > c_tol2))
            - _safe_reciprocal(delta3, (err3 > tol3) & (jnp.abs(delta3) > c_tol3))
        )
        irregular = norm(inv_delta_sum * e1) <= IRREGULAR
        truncate = active & ~converged & (too_close | irregular)

        step_ok = active & ~converged & ~truncate
        # A component all three of whose differences vanished contributes nothing to the
        # sum, and the Shanks step for it degenerates to its own current value: it has
        # already converged, so there is no tail left to infer.
        res = e1 + _safe_reciprocal(inv_delta_sum, step_ok & (inv_delta_sum != 0))
        table = table.at[k1].set(jnp.where(step_ok, res, table[k1]))
        error = err2 + norm(res - e2) + err3

        # Keep the best element of the diagonal, which is not always the last one: the
        # later steps are higher order but divide by smaller differences, so accuracy
        # improves along the diagonal until cancellation takes over. Ties go to the
        # later element, which is the higher order transform.
        better = step_ok & (error <= abserr)
        abserr = jnp.where(better, error, abserr)
        result = jnp.where(better, res, result)

        # The convergence short circuit reports the entry it agreed on, not the best
        # diagonal element, and overrides it.
        result = jnp.where(converged, e2, result)
        abserr = jnp.where(converged, err2 + err3, abserr)

        # Keep only what this diagonal has already worked through, discarding the older
        # entries the truncation has just declared unusable. `i` steps have been taken
        # and each consumes two slots, so `2i - 2` is the index of the newest survivor.
        n_out = jnp.where(truncate, 2 * i - 2, n_out)
        return (
            table,
            jnp.where(step_ok, k1 - 2, k1),
            result,
            abserr,
            done | converged | truncate,
            n_out,
        )

    table, _, result, abserr, _, n = jax.lax.fori_loop(
        1,
        MAX_TABLE_SIZE // 2 + 1,
        body,
        (table, n, result, abserr, jnp.zeros((), bool), n),
    )

    n = jnp.asarray(
        jnp.where(n == MAX_TABLE_SIZE - 1, 2 * (MAX_TABLE_SIZE // 2) - 2, n)
    )

    # Shift the table so the next call sees a compacted diagonal. Two passes, both
    # expressed as gathers rather than loops: the first drops the entries the recursion
    # consumed, on whichever of the odd/even sublattices this call used; the second
    # closes the gap a truncation left behind.
    #
    # Which of the two sublattices was used is set by the parity of the entry count: the
    # recursion walked `k1` down from the newest slot in steps of two, so the entries it
    # consumed start at index 0 for an odd count and index 1 for an even one.
    idx = jnp.arange(MAX_TABLE_SIZE + 2)
    start = jnp.where(num % 2 == 0, 0, 1)
    on_lattice = (idx >= start) & (idx <= start + 2 * newelm) & ((idx - start) % 2 == 0)
    table = _gather(table, jnp.where(on_lattice & ~too_short, idx + 2, idx))

    dropped = num - n
    table = _gather(
        table, jnp.where((idx <= n) & (dropped > 0) & ~too_short, idx + dropped, idx)
    )
    n_entries = jnp.where(too_short, n_entries, n + 1)

    return _record(
        state._replace(table=table, n=n_entries, n_calls=n_calls),
        result,
        abserr,
        norm,
        epmach,
    )


def _record(
    state: EpsilonTable,
    result: jax.Array,
    abserr: jax.Array,
    norm: Callable,
    epmach: float,
) -> EpsilonTable:
    """Store the extrapolation and estimate how far it still is from the limit.

    QUADPACK [2] reports the spread of the three most recent extrapolated values. That
    says how far the sequence has recently *moved*, which is not the same as how far it
    still has to *go*, and using it as if it were is optimistic by exactly the amount
    that matters: a sequence whose steps are still shrinking slowly has its whole
    remaining tail ahead of it. If the extrapolants converge with step ratio ``rho``
    that tail sums to ``step / (1 - rho)``, and that correction is added here. Where the
    table is working ``rho`` is small and the correction changes nothing; where the
    acceleration cannot fit the problem's asymptotics ``rho`` approaches one and the
    estimate grows to say so, which is the case the spread alone gets wrong and gets
    wrong by more the longer the run goes on.

    Neither part makes the result a bound; the extrapolated value still has none. Both
    are chosen to fail towards over-reporting, since a run that overstates its error
    declines to claim an accuracy it reached, while one that understates it claims an
    accuracy it did not.

    The uncorrected spread is kept as ``abserr_sharp`` for the caller to rank
    extrapolations by. QUADPACK ranks on the same number it reports, which is safe only
    while the two are the same; once the tail is added they are not, and ranking on the
    corrected figure would let the tail term decide which value is kept rather than only
    what is said about it. Ranking wants the sequence that settled tightest, which is
    what the spread measures; only the reported figure needs the tail added.
    """
    have_three = state.n_calls >= 4
    spread = sum(norm(result - r) for r in state.last_results)
    est = jnp.where(have_three, spread, abserr)
    sharp = jnp.maximum(est, ERR_FLOOR * epmach * norm(result))

    # Successive steps of the extrapolant sequence, newest first. Two ratios are
    # available from three steps; the larger is taken, so that a sequence which has
    # merely paused is not mistaken for one that has arrived.
    #
    # A step of exactly zero is two extrapolations that returned the same value, which
    # happens while the table is still filling and its unused slots read as equal. The
    # flags below say only that a denominator is safe to divide by; a ratio whose
    # denominator vanishes drops out of the maximum rather than suppressing the whole
    # correction, which is the right way round, since a sequence that stood still and
    # then moved is the least converged of all and wants the largest tail, not none.
    d2 = jnp.asarray(norm(result - state.last_results[2]))
    d1 = jnp.asarray(norm(state.last_results[2] - state.last_results[1]))
    d0 = jnp.asarray(norm(state.last_results[1] - state.last_results[0]))
    d1_ok, d0_ok = d1 > 0, d0 > 0
    rho = jnp.maximum(
        jnp.where(d1_ok, d2 / jnp.where(d1_ok, d1, 1.0), 0.0),
        jnp.where(d0_ok, d1 / jnp.where(d0_ok, d0, 1.0), 0.0),
    )
    tail = TAIL_SAFETY * d2 / (1.0 - jnp.clip(rho, 0.0, RHO_MAX))
    # Only while the steps still carry signal. Once the extrapolants agree to roundoff
    # the ratio between their differences is noise, and the floor below is what applies.
    # The spread and the tail are two readings of the same quantity, how far the
    # extrapolant is from the limit, taken from the same three steps, so the larger is
    # kept rather than the two added: summing would count one movement twice.
    signal = d2 > STEP_NOISE * epmach * norm(result)
    est = jnp.where(have_three & signal, jnp.maximum(est, tail), est)
    est = jnp.maximum(est, ERR_FLOOR * epmach * norm(result))

    # Roll the newest in and the oldest out.
    last = jnp.where(
        have_three,
        jnp.concatenate([state.last_results[1:], result[jnp.newaxis]]),
        state.last_results.at[jnp.minimum(state.n_calls - 1, 2)].set(result),
    )
    return state._replace(
        last_results=last, result=result, abserr=est, abserr_sharp=sharp
    )
