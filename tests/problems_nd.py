"""Test problems for cubature over a box.

The n dimensional counterpart of ``problems.py``. The dict keys are the same, plus
``ndim``, and the integrand takes an abscissa of shape ``(ndim,)`` rather than a scalar.
``ndim`` of one is among them: the routines accept a box of one axis, and the contract
they are held to there is the same one.
The tags, the tolerance constants and the ``assert_honest`` / ``assert_converged``
contract are imported from there unchanged, because they are the same contract: a
routine is free to say it could not solve a problem, and is never free to claim an
accuracy it did not have.

Each problem covers something the others do not. They are kept few on purpose, and a
new one is worth adding only where it exercises a path none of these reach.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from .problems import (  # noqa: F401
    CONVERGENT_TOLS,
    INDUCED_TAG,
    LOCALIZED_TAG,
    OSCILLATORY_TAG,
    PIECEWISE_SMOOTH_TAG,
    QUADPACK_MODEL,
    SINGULAR_TAG,
    SLOP,
    SMOOTH_TAG,
    TOLS,
    ErrorModel,
    assert_converged,
    assert_honest,
    real_dtypes,
)


def _abs_mean(c):
    """Integral of ``|x - c|`` over the unit interval, for ``0 <= c <= 1``."""
    return (c**2 + (1 - c) ** 2) / 2


def _gauss_1d(s):
    """Integral of ``exp(-(x/s)**2)`` over ``[-1, 1]``."""
    return s * math.sqrt(math.pi) * math.erf(1 / s)


PROBLEMS = [
    # problem 0
    {
        "name": "cos-product",
        "tags": {SMOOTH_TAG},
        "ndim": 2,
        "fun": lambda x: jnp.cos(x[0]) * jnp.cos(x[1]),
        "interval": [[0.0, 1.0], [0.0, 1.0]],
        "val": math.sin(1.0) ** 2,
    },
    # problem 1 - separable, so a wrong per-axis Jacobian shows up as a wrong power
    {
        "name": "exp-sum-3d",
        "tags": {SMOOTH_TAG},
        "ndim": 3,
        "fun": lambda x: jnp.exp(-jnp.sum(x)),
        "interval": [[0.0, 1.0]] * 3,
        "val": (1 - math.exp(-1)) ** 3,
    },
    # problem 2 - anisotropic: the feature is 12x narrower along axis 0 than axis 1,
    # which is what the local rule's split indicator has to notice
    {
        "name": "gauss-ridge",
        "tags": {LOCALIZED_TAG},
        "ndim": 2,
        "fun": lambda x: jnp.exp(-((x[0] / 0.05) ** 2) - (x[1] / 0.6) ** 2),
        "interval": [[-1.0, 1.0], [-1.0, 1.0]],
        "val": _gauss_1d(0.05) * _gauss_1d(0.6),
    },
    # problem 3 - singular at one corner of the box, and integrable there
    {
        "name": "corner-sqrt",
        "tags": {SINGULAR_TAG},
        "ndim": 2,
        "fun": lambda x: 1 / jnp.sqrt(x[0] + x[1]),
        "interval": [[0.0, 1.0], [0.0, 1.0]],
        "val": (8 / 3) * (math.sqrt(2) - 1),
    },
    # problem 4 - a kink on an axis aligned plane in each axis, marked as a breakpoint
    {
        "name": "kink-marked",
        "tags": {PIECEWISE_SMOOTH_TAG},
        "ndim": 2,
        "fun": lambda x: jnp.abs(x[0] - 1 / 3) * jnp.abs(x[1] - 2 / 3),
        "interval": [[0.0, 1 / 3, 1.0], [0.0, 2 / 3, 1.0]],
        "val": _abs_mean(1 / 3) * _abs_mean(2 / 3),
    },
    # problem 5 - the same integrand with the kinks left for the mesh to find, which is
    # what makes the pair a measurement of what a breakpoint is worth
    {
        "name": "kink-unmarked",
        "tags": {PIECEWISE_SMOOTH_TAG},
        "ndim": 2,
        "fun": lambda x: jnp.abs(x[0] - 1 / 3) * jnp.abs(x[1] - 2 / 3),
        "interval": [[0.0, 1.0], [0.0, 1.0]],
        "val": _abs_mean(1 / 3) * _abs_mean(2 / 3),
    },
    # problem 6 - oscillation running diagonally, so it is not resolved by refining
    # either axis alone
    {
        "name": "osc-diagonal",
        "tags": {OSCILLATORY_TAG},
        "ndim": 2,
        "fun": lambda x: jnp.cos(10 * (x[0] + x[1])),
        "interval": [[0.0, 1.0], [0.0, 1.0]],
        "val": (-math.cos(20.0) + 2 * math.cos(10.0) - 1) / 100,
    },
    # problem 7 - two components of very different difficulty sharing one mesh and one
    # error estimate, which is what charges the whole run its worst component
    {
        "name": "vector-mixed",
        "tags": {SINGULAR_TAG},
        "ndim": 2,
        "fun": lambda x: jnp.array(
            [jnp.cos(x[0]) * jnp.cos(x[1]), 1 / jnp.sqrt(x[0] + x[1])]
        ),
        "interval": [[0.0, 1.0], [0.0, 1.0]],
        "val": np.array([math.sin(1.0) ** 2, (8 / 3) * (math.sqrt(2) - 1)]),
    },
    # problem 8 - complex valued, where the error and the auxiliary sums stay real
    {
        "name": "complex-phase",
        "tags": {SMOOTH_TAG},
        "ndim": 2,
        "fun": lambda x: jnp.exp(1j * (x[0] + x[1])),
        "interval": [[0.0, 1.0], [0.0, 1.0]],
        "val": ((math.e ** (1j) - 1) / 1j) ** 2,
    },
    # problem 9 - every axis unbounded on one side
    {
        "name": "exp-semi-infinite",
        "tags": {INDUCED_TAG},
        "ndim": 2,
        "fun": lambda x: jnp.exp(-jnp.sum(x)),
        "interval": [[0.0, np.inf], [0.0, np.inf]],
        "val": 1.0,
    },
    # problem 10 - unbounded on both sides of both axes, with a mixed pair of limits so
    # that more than one branch of the map is exercised at once
    {
        "name": "gauss-infinite",
        "tags": {INDUCED_TAG},
        "ndim": 2,
        "fun": lambda x: jnp.exp(-jnp.sum(x**2)),
        "interval": [[-np.inf, np.inf], [-np.inf, 1.0]],
        "val": math.pi * (1 + math.erf(1.0)) / 2,
    },
    # problem 11 - algebraic decay in |x| over the whole plane, the case where the
    # separable map's Jacobian and the integrand's decay disagree at the corners of the
    # reference box. The value is right and the convergence rate is what suffers.
    {
        "name": "algebraic-infinite",
        "tags": {INDUCED_TAG},
        "ndim": 2,
        "fun": lambda x: 1 / (1 + jnp.sum(x**2)) ** 2,
        "interval": [[-np.inf, np.inf], [-np.inf, np.inf]],
        "val": math.pi,
    },
    # problem 12 - smooth, but at a dimension where the node count per region is what
    # dominates the cost rather than the number of regions
    {
        "name": "gauss-5d",
        "tags": {SMOOTH_TAG},
        "ndim": 5,
        "fun": lambda x: jnp.exp(-jnp.sum(x**2)),
        "interval": [[-1.0, 1.0]] * 5,
        "val": _gauss_1d(1.0) ** 5,
    },
    # problem 13 - a cusp ridge on an axis aligned plane in every axis, unmarked. The
    # integrand does not vanish at the cusp the way a product of kinks does, so the
    # mesh can refine away from it while the error estimate keeps falling: the true
    # error here stops improving around 3e-6 whatever tolerance is asked for.
    # The parameters were chosen as the worst behaving over a random draw.
    {
        "name": "cusp-ridge",
        "tags": {PIECEWISE_SMOOTH_TAG},
        "ndim": 3,
        "fun": lambda x: jnp.exp(
            -jnp.sum(
                jnp.array([0.2660088941584163, 0.4430043456880922, 0.4918098187571035])
                * jnp.abs(
                    x
                    - jnp.array(
                        [
                            0.25354825920421914,
                            0.7997477160722586,
                            0.19929440330793777,
                        ]
                    )
                )
            )
        ),
        "interval": [[0.02, 0.97]] * 3,
        "val": 0.5882689256199701,
    },
    # problem 14 - one axis, where the split indicator has nothing to choose between
    # and every orbit of the local rule collapses onto the same two points. The peak is
    # narrow enough that the box has to be subdivided rather than resolved outright.
    {
        "name": "peak-1d",
        "tags": {LOCALIZED_TAG},
        "ndim": 1,
        "fun": lambda x: 1 / (0.05**2 + (x[0] - 0.3) ** 2),
        "interval": [[0.0, 1.0]],
        "val": (math.atan(0.7 / 0.05) + math.atan(0.3 / 0.05)) / 0.05,
    },
    # problem 15 - one axis, unbounded and carrying a breakpoint, which is the domain
    # map and the seeding of the initial mesh over a single axis
    {
        "name": "exp-semi-infinite-1d",
        "tags": {INDUCED_TAG},
        "ndim": 1,
        "fun": lambda x: x[0] ** 2 * jnp.exp(-x[0]),
        "interval": [[0.0, 2.0, np.inf]],
        "val": 2.0,
    },
]

ALL = list(range(len(PROBLEMS)))


def tagged(*tags):
    """Indices of the problems carrying every one of ``tags``."""
    return [i for i, p in enumerate(PROBLEMS) if set(tags) <= p["tags"]]


SMOOTH = tagged(SMOOTH_TAG)
FINITE = [
    i
    for i, p in enumerate(PROBLEMS)
    if np.all(
        np.isfinite(
            np.concatenate([np.asarray(axis, dtype=float) for axis in p["interval"]])
        )
    )
]


def problem_id(i):
    """Name of a problem index: the slug used as its pytest id."""
    return PROBLEMS[i]["name"]


# The region budget the convergence battery runs at. Generous: the point of the sweep
# is whether a routine converges and tells the truth about it, not how cheaply, and a
# budget tight enough to make ordinary problems fail would fill the table below with
# entries that describe the budget rather than the method.
BATTERY_MAX_NREGION = 4000

# Problems a rule does not converge on, keyed by (rule name, problem name). A rule that
# reports failure here is behaving correctly; what it is never allowed to do is claim an
# accuracy it did not reach, which `KNOWN_DISHONEST` below is for.
#
# `gauss-5d` is the dimensional cost. A degree 9 rule in five dimensions compares a
# degree 9 against a degree 7 rule, and the exponent relating the two error rates is
# (9+1+5)/(7+1+5) = 1.15 rather than the 1.5 the one dimensional routines enjoy, so each
# region's estimate comes down slowly.
KNOWN_FAILURES: dict[tuple[str, str], set[float]] = {
    ("genz-malik", "gauss-5d"): {1e-8},
    # not a separate defect from the `KNOWN_DISHONEST` entry below: a run that
    # understates its error is by the same token one that did not reach the tolerance
    # it claims, so it fails both halves of the contract at once.
    ("genz-malik", "cusp-ridge"): {1e-8},
}

# Cases where the reported error came out below the true error. Meant to empty: every
# other problem here leaves the reported value a bound rather than an estimate.
#
# `cusp-ridge` is a real gap rather than a tuning miss. Where the mesh lands badly
# against an unmarked cusp it refines away from it, the estimate falls while the true
# error does not, and the run reports success having understated by some three orders of
# magnitude. Giving the cusp coordinates as breakpoints fixes it completely, which is
# what the `cubgm` docstring recommends, but a caller who does not know where the cusp
# is gets no warning. The parameters are the ones a search over draws found worst;
# nearby ones are honest, so the entry records a reachable failure rather than the
# typical case.
KNOWN_DISHONEST: dict[tuple[str, str], set[float]] = {
    ("genz-malik", "cusp-ridge"): {1e-8},
}


def _xfail_from(table, what, request, rule_name, prob, tol):
    """Mark the running test xfail if ``table`` lists this case.

    A case in a table is reported as an expected failure and one that starts passing is
    reported as ``XPASS`` rather than silently dropping out of the record.
    """
    tols = table.get((rule_name, prob["name"]))
    if tols is not None and float(tol) in tols:
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{rule_name} {what} on {prob['name']} at tol={tol:g}",
                strict=False,
            )
        )


def xfail_if_known(request, rule_name, prob, tol):
    """Mark the running test xfail if ``KNOWN_FAILURES`` lists this case."""
    _xfail_from(KNOWN_FAILURES, "does not converge", request, rule_name, prob, tol)


def xfail_if_dishonest(request, rule_name, prob, tol):
    """Mark the running test xfail if ``KNOWN_DISHONEST`` lists this case."""
    _xfail_from(KNOWN_DISHONEST, "understates its error", request, rule_name, prob, tol)
