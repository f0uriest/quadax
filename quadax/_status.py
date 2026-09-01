"""Why a quadrature stopped, and the message that explains it."""

from enum import EnumMeta, IntEnum
from typing import cast

import jax
import jax.numpy as jnp
from equinox.internal import branched_error_if


class _StatusMeta(EnumMeta):
    """Metaclass letting ``STATUS`` be indexed by a reported code as well as by name.

    A routine reports its status as the integer the member carries, so looking the
    message back up starts from that integer rather than from a name.
    """

    def __getitem__(cls, key) -> "STATUS":
        member = super().__getitem__(key) if isinstance(key, str) else cls(int(key))
        return cast("STATUS", member)


class STATUS(IntEnum, metaclass=_StatusMeta):
    """Reason a quadrature terminated.

    Each member is the integer code the routine reports, and carries the message
    explaining it. ``QuadratureInfo.status`` is that integer, so it can be compared
    against a member, read as an ``int``, or held in an array under ``vmap``.

    Codes run from least to most severe, and a run that meets more than one condition
    reports the most severe of them. The order is what a reader should do about each.

    The message is the member's ``message``, and is what printing the member shows::

        >>> y, info = quadgk(fun, [0, 1])
        >>> if info.status != STATUS.normal:
        ...     print(STATUS[info.status])

    """

    message: str

    def __new__(cls, code, message):
        member = int.__new__(cls, code)
        member._value_ = code
        member.message = message
        return member

    def __str__(self):
        return self.message

    def __format__(self, spec):
        return format(str(self), spec)

    normal = (
        0,
        "Normal convergence. The estimated error fell below the requested tolerance.",
    )

    max_ninter = (
        1,
        "The subdivision used all max_ninter sub-intervals without reaching the "
        "tolerance. The value returned is the total over the mesh it ended with, and "
        "the reported error is that mesh's own estimate for it. Raising max_ninter "
        "helps if the subdivision was still making progress; where the integrand has "
        "a singularity or a jump at a known point, passing that point in `interval` "
        "as a breakpoint is worth more, since the mesh no longer has to find it.",
    )

    max_divisions = (
        2,
        "The schedule used all divmax refinement levels without reaching the "
        "tolerance. A Romberg mesh halves the whole interval uniformly and cannot "
        "refine near a difficulty, so an integrand with a local feature is better "
        "given to quadgk or quadcc.",
    )

    no_converge = (
        3,
        "The convergence acceleration stopped making progress: six extrapolations in "
        "a row produced nothing better than the one already held, while claiming an "
        "error far below what the subdivision reports. The best of them is returned, "
        "with its error widened to cover how far the later ones moved away from it. "
        "The running totals most likely have no asymptotic form for the extrapolation "
        "to fit.",
    )

    truncation = (
        4,
        "The tanh-sinh map leaves more of the integral outside the range its abscissae "
        "cover than the tolerance allows, so refining further cannot reach it. This is "
        "a property of the map in finite precision rather than of the mesh, and more "
        "levels would only spend evaluations arriving at the same answer. The reported "
        "error includes an estimate of what the map omits and remains a bound on the "
        "value returned; it is the tolerance that is out of reach. Using higher "
        "precision may allow lower error.",
    )

    bad_integrand = (
        5,
        "Subdivision drove sub-intervals down to a width of order the floating point "
        "spacing across the interval, where the abscissae within one can no longer be "
        "told apart, so the mesh cannot localize the difficulty any further. This is "
        "what a non-integrable singularity looks like, or a feature narrower than the "
        "abscissae can resolve. Check that the integral exists, and that any "
        "singularity or discontinuity sits at a limit or a breakpoint rather than in "
        "the interior of a sub-interval.",
    )

    roundoff = (
        6,
        "The achievable accuracy is limited by roundoff. Subdivision has stopped "
        "buying accuracy: the error estimate has reached the floor the arithmetic "
        "imposes, or the total stopped moving while its error stayed where it was. The "
        "tolerance asked for is below what the integrand can be summed to at this "
        "precision. The reported error may be understated, being built from the same "
        "arithmetic that has bottomed out. Try loosening tolerances, or use higher "
        "precision.",
    )

    divergent = (
        7,
        "The integral is suspected to be divergent. The extrapolated value bears no "
        "relation to the running total it was built from. A finite value may still be "
        "returned, but do not use it without establishing that the integral converges.",
    )


def escalate(status: int | jax.Array, flag: STATUS, cond) -> jax.Array:
    """Raise ``status`` to ``flag`` where ``cond`` holds and ``flag`` is more severe.

    A quadrature can meet several termination conditions at once, and reports the worst
    of them. Taking the more severe rather than the first one raised is what lets a
    condition detected after the loop has ended -- divergence, which is only testable
    against the finished result -- override one the loop itself recorded.

    ``cond`` is a scalar boolean, and ``flag`` a member of :class:`STATUS`.
    """
    return jnp.where(cond & (flag > status), flag, status)


def withdraw(status: int | jax.Array, flag: STATUS, cond) -> jax.Array:
    """Clear ``flag`` where ``cond`` holds, leaving any other status alone.

    A flag describing how the answer was reached rather than the answer itself can stop
    applying once the run has finished: the tolerance may be met in the end by a route
    that stalled on the way. Only the flag named is cleared, so a more severe one raised
    since still stands.
    """
    return jnp.where(cond & (status == flag), STATUS.normal, status)


def error_if_flagged(y, status: int | jax.Array):
    """Raise the message ``status`` carries, on anything but ``STATUS.normal``.

    This is what ``throw=True`` does with the status a run would otherwise have only
    reported. The message is selected under the trace by the code itself, so the error
    names the condition actually met rather than a generic failure.
    """
    messages = [member.message for member in STATUS]
    return branched_error_if(y, status != STATUS.normal, status, messages)
