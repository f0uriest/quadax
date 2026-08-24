"""Why a quadrature stopped, and the message that explains it."""

import equinox as eqx
import jax


class STATUS(eqx.Enumeration):
    """Reason a quadrature terminated.

    Members are declared from least to most severe, and a run that meets more than one
    condition reports the most severe of them. The order is what a reader should do
    about each.

    The message is the member's value, and is what ``repr`` shows::

        >>> y, info = quadgk(fun, [0, 1])
        >>> print(info.status)
        >>> print(STATUS[info.status])  # the message on its own

    Indexing works on a batch of statuses too, which is how a ``vmap``-ed quadrature is
    read back.
    """

    normal = (
        "Normal convergence. The estimated error fell below the requested tolerance, "
    )

    max_ninter = (
        "The subdivision used all max_ninter sub-intervals without reaching the "
        "tolerance. The value returned is the total over the mesh it ended with, and "
        "the reported error is that mesh's own estimate for it. Raising max_ninter "
        "helps if the subdivision was still making progress; where the integrand has "
        "a singularity or a jump at a known point, passing that point in `interval` "
        "as a breakpoint is worth more, since the mesh no longer has to find it."
    )

    max_divisions = (
        "The schedule used all divmax refinement levels without reaching the "
        "tolerance. A Romberg mesh halves the whole interval uniformly and cannot "
        "refine near a difficulty, so an integrand with a local feature is better "
        "given to quadgk or quadcc."
    )

    no_converge = (
        "The convergence acceleration stopped making progress: six extrapolations in "
        "a row produced nothing better than the one already held, while claiming an "
        "error far below what the subdivision reports. The best of them is returned, "
        "with its error widened to cover how far the later ones moved away from it. "
        "The running totals most likely have no asymptotic form for the extrapolation "
        "to fit."
    )

    truncation = (
        "The tanh-sinh map leaves more of the integral outside the range its abscissae "
        "cover than the tolerance allows, so refining further cannot reach it. This is "
        "a property of the map in finite precision rather than of the mesh, and more "
        "levels would only spend evaluations arriving at the same answer. The reported "
        "error includes an estimate of what the map omits and remains a bound on the "
        "value returned; it is the tolerance that is out of reach. Using higher "
        "precision may allow lower error."
    )

    bad_integrand = (
        "Subdivision drove sub-intervals down to a width of order the floating point "
        "spacing across the interval, where the abscissae within one can no longer be "
        "told apart, so the mesh cannot localize the difficulty any further. This is "
        "what a non-integrable singularity looks like, or a feature narrower than the "
        "abscissae can resolve. Check that the integral exists, and that any "
        "singularity or discontinuity sits at a limit or a breakpoint rather than in "
        "the interior of a sub-interval."
    )

    roundoff = (
        "The achievable accuracy is limited by roundoff. Subdivision has stopped "
        "buying accuracy: the error estimate has reached the floor the arithmetic "
        "imposes, or the total stopped moving while its error stayed where it was. The "
        "tolerance asked for is below what the integrand can be summed to at this "
        "precision. The reported error may be understated, being built from the same "
        "arithmetic that has bottomed out. Try loosening tolerances, or use higher "
        "precision."
    )

    divergent = (
        "The integral is suspected to be divergent. The extrapolated value bears no "
        "relation to the running total it was built from. A finite value may still be "
        "returned, but do not use it without establishing that the integral converges."
    )


def _index(status: STATUS) -> jax.Array:
    """The member's position in the declaration order, ie its severity.

    ``Enumeration`` holds it as the item's single pytree leaf, which is the same value
    its message lookup and ``where`` address it by. Nothing public exposes it, and the
    ordering comparison below is the whole reason it is wanted.
    """
    (value,) = jax.tree.leaves(status)
    return value


def escalate(status: STATUS, flag: STATUS, cond) -> STATUS:
    """Raise ``status`` to ``flag`` where ``cond`` holds and ``flag`` is more severe.

    A quadrature can meet several termination conditions at once, and reports the worst
    of them. Taking the more severe rather than the first one raised is what lets a
    condition detected after the loop has ended -- divergence, which is only testable
    against the finished result -- override one the loop itself recorded.

    ``cond`` is a scalar boolean, and ``flag`` a member of :class:`STATUS`.
    """
    return STATUS.where(cond & (_index(flag) > _index(status)), flag, status)


def withdraw(status: STATUS, flag: STATUS, cond) -> STATUS:
    """Clear ``flag`` where ``cond`` holds, leaving any other status alone.

    A flag describing how the answer was reached rather than the answer itself can stop
    applying once the run has finished: the tolerance may be met in the end by a route
    that stalled on the way. Only the flag named is cleared, so a more severe one raised
    since still stands.
    """
    return STATUS.where(cond & (status == flag), STATUS.normal, status)
