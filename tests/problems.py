"""The example integrals the quadrature tests are run against, and shared constants.

Each problem is a dict with

- ``fun``     the integrand,
- ``interval`` the limits, with interior entries taken as breakpoints and ``+/-inf``
  allowed at either end,
- ``val``     the exact integral, used as the reference every accuracy check measures
  against,
- ``name``    a slug used as the pytest id, so a failure names the integrand,
- ``tags``    the set of difficulty tags below that the problem carries.

Tags describe what the integrator faces rather than what the integrand looks like, and a
problem carries as many as apply: an oscillation accumulating at a singular endpoint is
tagged both ways, and the routines meet both difficulties at once. The first four tags
name what makes a problem hard, and in practice a problem carries exactly one of them,
since they are four different answers to that question; the last two are properties that
cut across the first four, and are what the tag set exists for.

Whether a problem is vector valued, complex, or has breakpoints is not a difficulty tag
and is derived from the entry itself rather than tracked by hand.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
from jax import config

config.update("jax_enable_x64", True)

# What makes the problem hard. One of these four, since they are four answers to that
# one question.
#
# The local rule resolves the integrand, after whatever coordinate map is applied. The
# subdivision converges on its own and convergence acceleration should never fire.
SMOOTH_TAG = "smooth"
# A genuine singularity of the integrand somewhere on a finite interval.
SINGULAR_TAG = "singular"
# An infinite range over which the integrand decays algebraically. The map onto the
# reference interval turns that decay into an endpoint singularity, so the difficulty is
# manufactured by the coordinate rather than present in the integrand.
INDUCED_TAG = "induced"
# The integrand is bounded and has no singularity, but carries a feature - an
# oscillation, a spike, a break in smoothness - on a length scale far below the width of
# the interval. The local rule cannot see it until the mesh has isolated it, so the
# difficulty is one of finding the feature rather than of resolving it once found.
# Nothing here needs convergence acceleration, and none of it would help: what these
# problems test is the subdivision's search, which no amount of extrapolation replaces.
LOCALIZED_TAG = "localized"

# Properties cutting across the four above, which is what the tag set is for: either can
# sit on a problem that is otherwise smooth, singular, induced or localized, and each
# names a distinct thing that can go wrong.
#
# The integrand changes sign many times across the region carrying the integral, so a
# panel spanning several of those crossings averages them away and both rules of a
# nested pair can agree on a value neither has resolved.
OSCILLATORY_TAG = "oscillatory"
# The integrand is smooth on each of finitely many pieces but not across their joins:
# somewhere strictly inside the interval it, or one of its derivatives, jumps. Says
# nothing about whether the break is bounded - pair it with `SINGULAR_TAG` or
# `LOCALIZED_TAG` for that. A break at an endpoint is not this: the interval never has
# to be split.
PIECEWISE_SMOOTH_TAG = "piecewise-smooth"

# The four that answer "what makes this hard", as against the two that cut across them.
# Every problem carries exactly one, which `tests/test_problems.py` holds it to: the
# derived lists below are unions over these, so a problem carrying none would appear in
# no list and quietly never run.
DIFFICULTY_TAGS = {SMOOTH_TAG, SINGULAR_TAG, INDUCED_TAG, LOCALIZED_TAG}
# Every tag a problem may carry. Tags are plain strings, so without this a misspelled
# one would form an empty group rather than an error.
TAGS = DIFFICULTY_TAGS | {OSCILLATORY_TAG, PIECEWISE_SMOOTH_TAG}

PROBLEMS = [
    # problem 0
    {
        "name": "t-log1p",
        "tags": {SMOOTH_TAG},
        "fun": lambda t: t * jnp.log(1 + t),
        "interval": [0, 1],
        "val": 1 / 4,
    },
    # problem 1
    {
        "name": "t2-arctan",
        "tags": {SMOOTH_TAG},
        "fun": lambda t: t**2 * jnp.arctan(t),
        "interval": [0, 1],
        "val": (jnp.pi - 2 + 2 * jnp.log(2)) / 12,
    },
    # problem 2
    {
        "name": "exp-cos",
        "tags": {SMOOTH_TAG},
        "fun": lambda t: jnp.exp(t) * jnp.cos(t),
        "interval": [0, jnp.pi / 2],
        "val": (jnp.exp(jnp.pi / 2) - 1) / 2,
    },
    # problem 3
    {
        "name": "arctan-sqrt",
        "tags": {SMOOTH_TAG},
        "fun": lambda t: (
            jnp.arctan(jnp.sqrt(2 + t**2)) / ((1 + t**2) * jnp.sqrt(2 + t**2))
        ),
        "interval": [0, 1],
        "val": 5 * jnp.pi**2 / 96,
    },
    # problem 4
    {
        "name": "sqrt-log",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: jnp.sqrt(t) * jnp.log(t),
        "interval": [0, 1],
        "val": -4 / 9,
    },
    # problem 5 - looks smooth and is not: the semicircle has an infinite derivative at
    # the endpoint, which is exactly the endpoint behavior the acceleration exists for
    {
        "name": "semicircle",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: jnp.sqrt(1 - t**2),
        "interval": [0, 1],
        "val": jnp.pi / 4,
    },
    # problem 6
    {
        "name": "sqrt-over-semicircle",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: jnp.sqrt(t) / jnp.sqrt(1 - t**2),
        "interval": [0, 1],
        "val": 2
        * jnp.sqrt(jnp.pi)
        * scipy.special.gamma(3 / 4)
        / scipy.special.gamma(1 / 4),
    },
    # problem 7
    {
        "name": "log-squared",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: jnp.log(t) ** 2,
        "interval": [0, 1],
        "val": 2,
    },
    # problem 8
    {
        "name": "log-cos",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: jnp.log(jnp.cos(t)),
        "interval": [0, jnp.pi / 2],
        "val": -jnp.pi * jnp.log(2) / 2,
    },
    # problem 9
    {
        "name": "sqrt-tan",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: jnp.sqrt(jnp.tan(t)),
        "interval": [0, jnp.pi / 2],
        "val": jnp.pi * jnp.sqrt(2) / 2,
    },
    # problem 10 - decays as t**-2, so the induced exponent is zero and the mapped
    # integrand is already smooth. Grouped with the infinite ranges because that is
    # where its difficulty comes from, not because it has a singularity.
    {
        "name": "lorentzian-halfline",
        "tags": {INDUCED_TAG},
        "fun": lambda t: 1 / (1 + t**2),
        "interval": [0, jnp.inf],
        "val": jnp.pi / 2,
    },
    # problem 11
    {
        "name": "exp-over-sqrt",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: jnp.exp(-t) / jnp.sqrt(t),
        "interval": [0, jnp.inf],
        "val": jnp.sqrt(jnp.pi),
    },
    # problem 12 - exponential decay, so the map leaves a smooth integrand behind
    {
        "name": "gaussian-line",
        "tags": {SMOOTH_TAG},
        "fun": lambda t: jnp.exp(-(t**2) / 2),
        "interval": [-jnp.inf, jnp.inf],
        "val": jnp.sqrt(2 * jnp.pi),
    },
    # problem 13 - likewise
    {
        "name": "exp-cos-halfline",
        "tags": {SMOOTH_TAG},
        "fun": lambda t: jnp.exp(-t) * jnp.cos(t),
        "interval": [0, jnp.inf],
        "val": 1 / 2,
    },
    # problem 14 - vector valued integrand made up of problems 0 and 1
    {
        "name": "vector",
        "tags": {SMOOTH_TAG},
        "fun": lambda t: jnp.array([t * jnp.log(1 + t), t**2 * jnp.arctan(t)]),
        "interval": [0, 1],
        "val": jnp.array([1 / 4, (jnp.pi - 2 + 2 * jnp.log(2)) / 12]),
    },
    # problem 15 - integral with breakpoints
    {
        "name": "log-breakpoint",
        "tags": {SINGULAR_TAG, PIECEWISE_SMOOTH_TAG},
        "fun": lambda t: jnp.log((t - 1) ** 2),
        "interval": [0, 1, 2],
        "val": -4,
    },
    # problem 16 - complex function
    {
        "name": "complex",
        "tags": {SMOOTH_TAG},
        "fun": lambda t: t * jnp.log(1 + t) * 1j,
        "interval": [0, 1],
        "val": 0.25j,
    },
    # Problems 17-25 are algebraic singularities, the family bisection alone cannot
    # resolve: for `t**-alpha` the mass below a panel of width h is exactly
    # `h**(1-alpha)` of the total, so an integrator that never samples closer than h to
    # the singular point carries that much relative error whatever its local rule does.
    # They span the axes that turn out to matter: how strong the singularity is, which
    # end it sits at, whether there is one or several, and whether the caller marked it.
    #
    # problem 17 - mild endpoint algebraic singularity
    {
        "name": "pow-0.5",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: t**-0.5,
        "interval": [0, 1],
        "val": 2.0,
    },
    # problem 18 - strong endpoint algebraic singularity
    {
        "name": "pow-0.9",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: t**-0.9,
        "interval": [0, 1],
        "val": 10.0,
    },
    # problem 19 - extreme endpoint singularity, near the divergence at alpha=1
    {
        "name": "pow-0.99",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: t**-0.99,
        "interval": [0, 1],
        "val": 100.0,
    },
    # problem 20 - the same strength at the *right* endpoint, which the mapping onto the
    # reference interval treats differently from the left
    {
        "name": "pow-0.9-right",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: (1 - t) ** -0.9,
        "interval": [0, 1],
        "val": 10.0,
    },
    # problem 21 - both endpoints singular, to different strengths. Two decaying modes
    # arriving at different rates, which is the case a single running total cannot
    # separate. Beta(3/4, 1/4) = gamma(3/4)gamma(1/4) = pi/sin(pi/4).
    {
        "name": "beta-both-ends",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: t**-0.25 * (1 - t) ** -0.75,
        "interval": [0, 1],
        "val": jnp.pi * jnp.sqrt(2),
    },
    # problem 22 - logarithmic times algebraic
    {
        "name": "log-over-sqrt",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: jnp.log(t) / jnp.sqrt(t),
        "interval": [0, 1],
        "val": -4.0,
    },
    # problem 23 - interior singularity, not marked
    {
        "name": "interior-unmarked",
        "tags": {SINGULAR_TAG, PIECEWISE_SMOOTH_TAG},
        "fun": lambda t: jnp.abs(t - 0.3) ** -0.5,
        "interval": [0, 1],
        "val": 2 * (jnp.sqrt(0.3) + jnp.sqrt(0.7)),
    },
    # problem 24 - the same one, marked as a breakpoint so it lands on a panel end
    {
        "name": "interior-marked",
        "tags": {SINGULAR_TAG, PIECEWISE_SMOOTH_TAG},
        "fun": lambda t: jnp.abs(t - 0.3) ** -0.5,
        "interval": [0, 0.3, 1],
        "val": 2 * (jnp.sqrt(0.3) + jnp.sqrt(0.7)),
    },
    # problem 25 - two interior singularities of different strengths
    {
        "name": "two-interior",
        "tags": {SINGULAR_TAG, PIECEWISE_SMOOTH_TAG},
        "fun": lambda t: jnp.abs(t - 0.3) ** -0.5 + jnp.abs(t - 0.7) ** -0.25,
        "interval": [0, 1],
        "val": 2 * (jnp.sqrt(0.3) + jnp.sqrt(0.7)) + (0.7**0.75 + 0.3**0.75) / 0.75,
    },
    # Problems 26-32 decay algebraically over an infinite range, which the mapping onto
    # the reference interval turns into an endpoint singularity, so they exercise the
    # same machinery as 17-25 but with the difficulty manufactured by the transform
    # rather than present in the integrand.
    #
    # For [a, inf) the map is `x = a - 1 + 2/(1-t)` with weight `2/(1-t)**2`, so an
    # integrand falling off as `x**-p` becomes `(1-t)**(p-2)`: the induced singularity
    # has strength `alpha = 2 - p`. The doubly infinite `tan` map gives the same law.
    # That makes these an exact counterpart of the finite cases: p = 1.1 induces the
    # same alpha = 0.9 as problem 18, and the pair measures how much of the difficulty
    # is the singularity itself and how much is the coordinate it is expressed in.
    #
    # problem 26 - semi-infinite, induced alpha = 0.5
    {
        "name": "decay-1.5",
        "tags": {INDUCED_TAG},
        "fun": lambda t: t**-1.5,
        "interval": [1, jnp.inf],
        "val": 2.0,
    },
    # problem 27 - semi-infinite, induced alpha = 0.9
    {
        "name": "decay-1.1",
        "tags": {INDUCED_TAG},
        "fun": lambda t: t**-1.1,
        "interval": [1, jnp.inf],
        "val": 10.0,
    },
    # problem 28 - semi-infinite, induced alpha = 0.99, the slowest decay that converges
    {
        "name": "decay-1.01",
        "tags": {INDUCED_TAG},
        "fun": lambda t: t**-1.01,
        "interval": [1, jnp.inf],
        "val": 100.0,
    },
    # problem 29 - the same decay from a finite left end, so the map is exercised
    # without the integrand also being singular at the start of the range
    {
        "name": "decay-1.5-from-0",
        "tags": {INDUCED_TAG},
        "fun": lambda t: (1 + t) ** -1.5,
        "interval": [0, jnp.inf],
        "val": 2.0,
    },
    # problem 30 - the mirror image, which uses the other one sided map
    {
        "name": "decay-1.5-mirrored",
        "tags": {INDUCED_TAG},
        "fun": lambda t: (1 - t) ** -1.5,
        "interval": [-jnp.inf, 0],
        "val": 2.0,
    },
    # problem 31 - logarithmic factor on top of the algebraic decay. Decays fast enough
    # that the induced exponent is zero and the mapped integrand is already smooth.
    {
        "name": "log-decay",
        "tags": {INDUCED_TAG},
        "fun": lambda t: jnp.log(t) / t**2,
        "interval": [1, jnp.inf],
        "val": 1.0,
    },
    # problem 32 - doubly infinite with algebraic decay, so the transform induces a
    # singularity at *both* ends at once
    {
        "name": "decay-line",
        "tags": {INDUCED_TAG},
        "fun": lambda t: (1 + t**2) ** -0.75,
        "interval": [-jnp.inf, jnp.inf],
        "val": jnp.sqrt(jnp.pi) * scipy.special.gamma(0.25) / scipy.special.gamma(0.75),
    },
    # Problems 33-38 are bounded and free of singularities, and carry a feature on a
    # length scale far below the interval: the routine has to find it before any rule
    # can resolve it. They cover the shapes that feature takes - oscillation, an
    # isolated spike, a break in smoothness - and are the only problems here whose
    # difficulty lies in the subdivision's search rather than in what the local rule
    # or the acceleration can do once the mesh is right.
    #
    # problem 33 - the standard oscillatory test integral, around 32 oscillations across
    # the range, where a rule spanning several of them at once averages them away
    {
        "name": "osc-bessel",
        "tags": {LOCALIZED_TAG, OSCILLATORY_TAG},
        "fun": lambda t: jnp.cos(100 * jnp.sin(t)),
        "interval": [0, jnp.pi],
        # pi*J0(100). Written out rather than computed, because `scipy.special.j0` is
        # only good to ~50 ulp at this argument and the reference has to be tighter
        # than the smallest tolerance the suite asks for.
        "val": 0.06278740049149269565503282,
    },
    # problem 34 - oscillation carried onto an infinite range, where the map compresses
    # the periods towards the endpoint rather than leaving them evenly spaced
    {
        "name": "osc-exp-decay",
        "tags": {LOCALIZED_TAG, OSCILLATORY_TAG},
        "fun": lambda t: jnp.exp(-t) * jnp.sin(10 * t),
        "interval": [0, jnp.inf],
        "val": 10 / 101,
    },
    # problem 35 - two Lorentzian spikes two orders of magnitude apart in width, on a
    # constant background subtracted off so the peaks carry the whole integral
    {
        "name": "two-peaks",
        "tags": {LOCALIZED_TAG},
        "fun": lambda t: 1 / ((t - 0.3) ** 2 + 1e-2) + 1 / ((t - 0.9) ** 2 + 1e-4) - 6,
        "interval": [0, 1],
        "val": (jnp.arctan(0.7 / 0.1) + jnp.arctan(0.3 / 0.1)) / 0.1
        + (jnp.arctan(0.1 / 0.01) + jnp.arctan(0.9 / 0.01)) / 0.01
        - 6,
    },
    # problem 36 - a single narrow peak, smooth to every order but with a standard
    # deviation of about 0.06, so the initial nodes can straddle it entirely
    {
        "name": "narrow-gauss",
        "tags": {LOCALIZED_TAG},
        "fun": lambda t: jnp.sqrt(50.0) * jnp.exp(-50 * jnp.pi * t**2),
        "interval": [0, 1],
        # the half line integral is exactly 1/2 and the tail past t=1 is O(1e-69)
        "val": 0.5,
    },
    # problem 37 - a jump, whose panel error falls off as h rather than as any power
    # the local rule's order buys
    {
        "name": "jump",
        "tags": {LOCALIZED_TAG, PIECEWISE_SMOOTH_TAG},
        "fun": lambda t: jnp.where(t < 1 / 3, 1.0, 0.0),
        "interval": [0, 1],
        "val": 1 / 3,
    },
    # problem 38 - continuous but not differentiable at an interior point, one order
    # smoother than the jump and still short of what the rule's order assumes
    {
        "name": "kink",
        "tags": {LOCALIZED_TAG, PIECEWISE_SMOOTH_TAG},
        "fun": lambda t: jnp.abs(t - 1 / 3),
        "interval": [0, 1],
        "val": 5 / 18,
    },
    # Problems 39-42 sit outside the algebraic and logarithmic endpoint behavior that
    # problems 4-25 cover. Both the epsilon table and Richardson's diagonal fit an
    # expansion in powers of the step, so an endpoint whose asymptotic is not of that
    # form is where the acceleration has nothing correct to converge to; these are the
    # cases that say what it does when its premise fails.
    #
    # problem 39 - bounded with a bounded first derivative, but a singular second, so
    # only the leading term of Richardson's even-power expansion is valid
    {
        "name": "sqrt-cubed",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: t**1.5,
        "interval": [0, 1],
        "val": 2 / 5,
    },
    # problem 40 - bounded, with infinitely many oscillations accumulating at the left
    # endpoint. Neither singular in magnitude nor of bounded variation, and no mesh
    # reaches the accumulation point, so the honest outcome is a reported failure.
    {
        "name": "sin-inverse",
        "tags": {SINGULAR_TAG, OSCILLATORY_TAG},
        "fun": lambda t: jnp.sin(1 / t),
        "interval": [0, 1],
        # sin(1) minus the cosine integral evaluated at 1
        "val": 0.5040670619069283719898561,
    },
    # problem 41 - problem 40 in the other coordinate: the same integral written over
    # an infinite range, where the oscillation sits in the decaying tail instead of at
    # a finite endpoint. Decays as t**-2, so the induced exponent is zero and what is
    # left after the map is the oscillation alone, which is the point of the pair.
    {
        "name": "osc-tail",
        "tags": {INDUCED_TAG, OSCILLATORY_TAG},
        "fun": lambda t: jnp.sin(t) / t**2,
        "interval": [1, jnp.inf],
        "val": 0.5040670619069283719898561,
    },
    # problem 42 - integrable, but only just: the antiderivative -1/log(t) approaches
    # its endpoint value logarithmically, which is not a power of the step at all. The
    # integrand is also NaN rather than infinite at the endpoint, since it is a zero
    # times an infinity there.
    {
        "name": "loglog",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: 1 / (t * jnp.log(t) ** 2),
        "interval": [0, 0.5],
        "val": 1 / jnp.log(2),
    },
    # problem 43 - the integral is zero, so a relative tolerance says nothing and the
    # accuracy check falls entirely on the absolute one. Every other problem here has a
    # value bounded away from zero, which leaves that branch untested.
    {
        "name": "zero-value",
        "tags": {SMOOTH_TAG},
        "fun": lambda t: jnp.cos(t),
        "interval": [0, jnp.pi],
        "val": 0.0,
    },
    # problem 44 - vector valued with the components at different difficulties, unlike
    # problem 14 where both are smooth. One mesh and one error estimate serve both, so
    # the easy component must not be allowed to talk the routine into stopping early.
    {
        "name": "vector-mixed",
        "tags": {SINGULAR_TAG},
        "fun": lambda t: jnp.array([t**-0.5, t**1.5]),
        "interval": [0, 1],
        "val": jnp.array([2.0, 2 / 5]),
    },
]

ALL = list(range(len(PROBLEMS)))


def tagged(*tags):
    """Indices of the problems carrying any of ``tags``, in problem order.

    Takes several tags rather than one so that a union is expressed once and comes back
    deduplicated: a problem carrying two of the tags asked for still appears once, which
    a concatenation of the separate lists would not give.
    """
    wanted = set(tags)
    return [i for i, p in enumerate(PROBLEMS) if wanted & p["tags"]]


SMOOTH = tagged(SMOOTH_TAG)
SINGULAR = tagged(SINGULAR_TAG)
INDUCED = tagged(INDUCED_TAG)
LOCALIZED = tagged(LOCALIZED_TAG)
OSCILLATORY = tagged(OSCILLATORY_TAG)
PIECEWISE_SMOOTH = tagged(PIECEWISE_SMOOTH_TAG)

# Romberg rejects breakpoints, so its suite is the problems whose interval is just
# the two limits.
NO_BREAKPOINTS = [i for i, p in enumerate(PROBLEMS) if len(p["interval"]) == 2]

# The smooth problems on finite limits. An infinite range is reached through a map that
# already gives the trapezoidal rule exponential convergence - the same mechanism
# tanh-sinh is built on - so those problems say nothing about what Richardson's
# expansion in powers of the step is worth, and are excluded where that is the question.
SMOOTH_FINITE = [
    i for i in SMOOTH if np.all(np.isfinite(np.asarray(PROBLEMS[i]["interval"], float)))
]

# The problems bisection alone cannot resolve, where convergence acceleration is what
# makes the difference. Narrower than SINGULAR + INDUCED, for two separate reasons, and
# the distinction between them matters: one set is above the acceleration and the other
# below it.
#
# The mesh already gets there on its own, so there is no tail left to remove:
#
# - 10 and 31 decay fast enough that the induced exponent is zero, so the mapped
#   integrand is already smooth and there is no asymptotic tail for the table to fit;
# - 4, 7 and 8 are purely logarithmic rather than algebraic, and bisection alone
#   reaches machine precision on them, so there is nothing left to remove. Problem 22
#   stays in: its log sits on top of an algebraic singularity, which does not resolve;
# - 5, 11, 15 and 39 are likewise resolved by the subdivision on its own.
_NO_TAIL_TO_ACCELERATE = (4, 5, 7, 8, 10, 11, 15, 31, 39)
# The acceleration's premise does not hold, so it has nothing correct to converge to.
# Wynn's epsilon algorithm fits an expansion in powers of the panel width, and on these
# the true asymptotic is not of that form: 40 and 41 are the same oscillation
# accumulating at a point no mesh reaches, and 42 approaches its endpoint value
# logarithmically. Feeding the table anyway makes the answer slightly worse rather than
# better, which is the honest outcome and not a defect to be tuned away.
_ACCELERATION_DOES_NOT_APPLY = (40, 41, 42)
RESOLVED_BY_ACCELERATION = [
    i
    for i in tagged(SINGULAR_TAG, INDUCED_TAG)
    if i not in _NO_TAIL_TO_ACCELERATE + _ACCELERATION_DOES_NOT_APPLY
]

# Problems each Romberg variant is expected to solve at every tolerance in TOLS. Not
# a record of what currently happens: the claim is that these are the integrands the
# method is *for*, and losing one should fail rather than quietly change a status.
#
# Richardson's expansion in even powers of the step needs the Euler-Maclaurin series to
# exist, which the SMOOTH tag gives outright. It also survives a feature the mesh has
# merely to find, since once the step is below that feature's scale the expansion holds
# again, so the localized problems belong here too - except the jump, where the
# expansion has no valid term at all and refinement buys one order in the step rather
# than two.
ROMBERG_CONVERGES = [
    i for i in tagged(SMOOTH_TAG, LOCALIZED_TAG) if PROBLEMS[i]["name"] != "jump"
]
# The tanh-sinh substitution flattens an endpoint singularity into the exponentially
# decaying tail the trapezoidal rule wants, so the variant reaches well past SMOOTH.
ROMBERGTS_CONVERGES = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    10,
    11,
    12,
    13,
    14,
    16,
    17,
    26,
    29,
    30,
    31,
    33,
    34,
    35,
    36,
    39,
    43,
    44,
]


def problem_id(i):
    """Name of a problem index: the slug used as its pytest id."""
    return PROBLEMS[i]["name"]


# The solvers are wrapped in `eqx.filter_jit`, which traces array leaves and treats
# every other leaf as static. Anything a test varies that is not an array therefore
# joins the compilation key, and a sweep pays for a separate trace and XLA compile of
# each value. This is why the sweeps pass their tolerances as `jnp.asarray(tol)` and
# share the integrand below rather than writing a fresh lambda per case.


def exp_neg(x):
    """``exp(-x)``, the integrand the dtype and precision tests share.

    Shared rather than written inline to avoid recompiling lambda functions.
    """
    return jnp.exp(-x)


# The tolerances every value test sweeps.
TOLS = (1e-4, 1e-8, 1e-12)

# 1e-12 sits below the accumulated `50*eps*integral_abs` floor of the near-divergent
# problems, where declining to converge is the correct answer rather than a failure, so
# convergence is only required of the two tolerances that are actually reachable here.
CONVERGENT_TOLS = (1e-4, 1e-8)


class ErrorModel(NamedTuple):
    """How much margin a routine's error estimate carries, on each branch.

    ``slack`` applies where the routine reported success, ``honesty`` where it reported
    failure. Both are the factor by which the true error may exceed the reported one, so
    ``slack = 1`` is the statement that the estimate is a genuine bound.

    The two families need different numbers because they estimate the error in
    genuinely different ways, not because one of them is held to a lower standard.
    """

    slack: float
    honesty: float


# The adaptive routines inflate each panel's error by the QUADPACK model and floor it at
# `50*eps*integral_abs`, which makes the reported value a bound rather than an estimate:
# on a converged run it is never optimistic, measured worst case 0.9997x, on quadgk at
# `vector-mixed`. Once the routine has given up that guarantee lapses, but the result
# must still be in the right league; measured worst case 21.3x, on quadts at `sqrt-tan`.
# Both numbers sit just above their measurement rather than at a round safe distance, so
# that an estimator getting looser shows up here as a failure. The slack figure now sits
# so close to the bound that `vector-mixed` is effectively the case defining it: the two
# components share one mesh and one estimate, which stretches the bound to its limit,
# and a change that loosened the estimator at all would fail there first.
#
# A margin this tight is only worth holding against a case whose ratio is reproducible.
# Where the acceleration cannot fit a problem's asymptotics, the epsilon table turns
# last-bit differences in the mesh sums into order-of-magnitude swings in both the
# extrapolated value and the reported error, so a ratio measured there samples a chaotic
# quantity rather than describing the estimator, and will differ between machines and
# between versions of the underlying libraries. Those cases are in `KNOWN_FAILURES` at
# the tolerances where the acceleration decides the answer; the honesty figure is
# anchored on `sqrt-tan`, whose ratio is unmoved by perturbing the integrand.
QUADPACK_MODEL = ErrorModel(slack=1.0, honesty=25)

# Romberg reports the difference between successive Richardson diagonals. That is an
# indicator of convergence rather than a bound, and carries no safety margin of any
# kind, so it can come in under the true error even on a converged run - measured worst
# case 1.500x, on rombergts at `sqrt-cubed`, which is the slack figure exactly rather
# than merely close to it. On a failed run it degrades much further than the QUADPACK
# estimate does, to a measured 144x. Set at the same fraction above their measurements
# as the QUADPACK pair, for the same reason.
RICHARDSON_MODEL = ErrorModel(slack=2.0, honesty=175)

# Tolerance for two routes to the same computation - extrapolation on against off, a
# different adjoint, jit against eager - which should agree to ~1 ULP. Tight enough
# that any real difference in what is being computed shows up, loose enough not to
# depend on the operation order XLA happens to choose.
ULP_RTOL = 1e-13
ULP_ATOL = 1e-15

# The dtypes the working precision plumbing is checked at. `interval` is always real;
# complex is a property of the integrand's values.
real_dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
complex_dtypes = [jnp.complex128, jnp.complex64]
real_of = {jnp.complex128: jnp.float64, jnp.complex64: jnp.float32}

# How much worse than sqrt(eps) a converged result is allowed to be in the dtype tests.
# Generous, because the point of those tests is dtype plumbing, not accuracy.
SLOP = 50


def assert_contract(y, info, prob, tol, model=QUADPACK_MODEL):
    """Assert what a quadrature routine promises about ``y`` and its error estimate.

    A status of 0 is a claim that the reported error met the request, and the reported
    error is a bound rather than an estimate: it is allowed to be loose, but never
    optimistic. The two together say the answer is good to the tolerance asked for.

    A run that reports failure makes no accuracy claim, and is only required to be in
    the right league.

    ``model`` is the `ErrorModel` of the routine under test, which decides how much the
    true error may exceed the reported one on each branch.
    """
    exact = np.asarray(prob["val"])
    value = np.asarray(y)
    true_err = float(np.max(np.abs(value - exact)))
    reported = float(np.max(np.asarray(info.err)))

    if int(info.status) == 0:
        assert true_err <= model.slack * reported, (
            f"{prob['name']} at tol={tol:g}: reported {reported:.3e} "
            f"< true {true_err:.3e}"
        )
        assert reported <= max(tol, tol * float(np.max(np.abs(value))))
        np.testing.assert_allclose(
            value, exact, rtol=tol, atol=tol, err_msg=f"{prob['name']}, tol={tol:g}"
        )
    else:
        assert true_err <= model.honesty * reported, (
            f"{prob['name']} at tol={tol:g}: failed with reported {reported:.3e} "
            f"but true error {true_err:.3e}"
        )


# Problems the routines do not currently deliver on, as (routine, problem) pointing at
# the tolerances that fail. Every entry is one of the two contract breaches
# `assert_contract` checks for, or a non-zero status on a problem the suite expects the
# routine to solve; which one it is varies with the tolerance, so the tolerances are
# listed rather than the failure mode.
#
# Three groups sit here, with separate causes.
#
# The largest is the double-exponential map, which breaks where the subdivision and the
# acceleration wrapped around it do not: the Clenshaw-Curtis entries are problems whose
# singularity the map moves onto a node, and the tanh-sinh entries are dominated by
# algebraic endpoint singularities and slowly decaying tails, where the map's own
# truncation error is the floor. Fixing that floor should retire whole blocks at once,
# so those entries are expected to be removed in groups rather than one at a time.
#
# The second is `loglog` and the `sin-inverse`/`osc-tail` pair, which every routine here
# fails, and which no amount of subdivision or extrapolation is expected to fix at the
# tighter tolerances: their asymptotics are not of the form either acceleration fits.
# The loosest tolerance is not out of reach - `scipy.integrate.quad` does land inside
# 1e-4 on `sin-inverse`, which is why the entry for it is not marked below - but nothing
# reaches 1e-8. What is wanted from these is an honest report of failure, and the
# entries record where even that is not met.
#
# The third is the Romberg pair reporting success on an answer it never sampled: the
# level 0 and level 1 estimates can agree by accident - because the integrand is NaN at
# an endpoint, or because a narrow feature falls between the three points those levels
# use - and nothing requires a minimum depth before that agreement is believed. These
# are the most serious entries in the table, being wrong answers returned with a status
# of 0 rather than shortfalls in accuracy.
#
# `# scipy too` marks the entries where the nearest scipy routine does not deliver the
# tolerance either, measured against scipy 1.17.1: `scipy.integrate.tanhsinh` for
# `quadts` and `rombergts`, `scipy.integrate.quad` for the rest, since scipy has no
# Clenshaw-Curtis or Romberg to compare against. A routine is counted as delivering only
# if it both reports success and lands inside the tolerance. Where the note names
# tolerances, scipy fails at those and delivers at the others.
#
# The split is roughly half: 29 of the 54 entries carry the note at one tolerance or
# more. That is the useful part of the annotation. An unmarked entry is one where a
# widely used routine does solve the problem, so the shortfall is quadax's; a marked one
# says the integrand is hard for the method rather than badly implemented here, and
# `scipy.integrate.tanhsinh` failing on much the same set as `quadts` and `rombergts` is
# what the double-exponential reading above rests on. Marked entries are still worth
# fixing, but they are evidence about the method and not a defect report.
KNOWN_FAILURES = {
    ("quadcc", "decay-line"): {1e-4, 1e-8},
    ("quadcc", "interior-marked"): {1e-4, 1e-8},
    ("quadcc", "loglog"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("quadcc", "osc-tail"): {1e-4, 1e-8},  # scipy too
    ("quadcc", "sin-inverse"): {1e-4, 1e-8},  # scipy too at 1e-8
    ("quadcc", "sqrt-tan"): {1e-4, 1e-8, 1e-12},
    ("quadgk", "loglog"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("quadgk", "osc-tail"): {1e-4, 1e-8},  # scipy too
    ("quadgk", "sin-inverse"): {1e-4, 1e-8},  # scipy too at 1e-8
    ("quadts", "beta-both-ends"): {1e-4, 1e-8, 1e-12},  # scipy too at 1e-8, 1e-12
    ("quadts", "decay-1.01"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("quadts", "decay-1.1"): {1e-4, 1e-8, 1e-12},
    ("quadts", "decay-1.5"): {1e-4, 1e-8, 1e-12},
    ("quadts", "decay-1.5-from-0"): {1e-4, 1e-8, 1e-12},
    ("quadts", "decay-1.5-mirrored"): {1e-4, 1e-8, 1e-12},
    ("quadts", "decay-line"): {1e-4, 1e-8},  # scipy too at 1e-8
    ("quadts", "exp-over-sqrt"): {1e-8, 1e-12},  # scipy too at 1e-12
    ("quadts", "interior-marked"): {1e-4, 1e-8},  # scipy too
    ("quadts", "log-over-sqrt"): {1e-4, 1e-8},
    ("quadts", "loglog"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("quadts", "osc-tail"): {1e-4, 1e-8},  # scipy too
    ("quadts", "pow-0.5"): {1e-4, 1e-8},
    ("quadts", "pow-0.9-right"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("quadts", "pow-0.99"): {1e-4, 1e-8},  # scipy too
    ("quadts", "sin-inverse"): {1e-4, 1e-8},  # scipy too
    ("quadts", "sqrt-over-semicircle"): {1e-4, 1e-8, 1e-12},  # scipy too at 1e-12
    ("quadts", "sqrt-tan"): {1e-4, 1e-8},
    ("quadts", "vector-mixed"): {1e-4, 1e-8},
    ("romberg", "loglog"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("romberg", "osc-exp-decay"): {1e-4},
    ("romberg", "osc-tail"): {1e-4},  # scipy too
    ("romberg", "sin-inverse"): {1e-4},
    ("rombergts", "beta-both-ends"): {1e-4, 1e-8, 1e-12},  # scipy too at 1e-8, 1e-12
    ("rombergts", "decay-1.01"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("rombergts", "decay-1.1"): {1e-4, 1e-8, 1e-12},
    ("rombergts", "decay-1.5"): {1e-8, 1e-12},
    ("rombergts", "decay-1.5-from-0"): {1e-8, 1e-12},
    ("rombergts", "decay-1.5-mirrored"): {1e-8, 1e-12},
    ("rombergts", "decay-line"): {1e-8, 1e-12},  # scipy too
    ("rombergts", "exp-over-sqrt"): {1e-8, 1e-12},  # scipy too at 1e-12
    ("rombergts", "jump"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("rombergts", "log-decay"): {1e-12},
    ("rombergts", "log-over-sqrt"): {1e-8, 1e-12},
    ("rombergts", "log-squared"): {1e-12},
    ("rombergts", "loglog"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("rombergts", "lorentzian-halfline"): {1e-4},
    ("rombergts", "narrow-gauss"): {1e-4, 1e-8, 1e-12},
    ("rombergts", "pow-0.5"): {1e-8, 1e-12},
    ("rombergts", "pow-0.9"): {1e-4, 1e-8, 1e-12},
    ("rombergts", "pow-0.9-right"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("rombergts", "pow-0.99"): {1e-4, 1e-8, 1e-12},  # scipy too
    ("rombergts", "sqrt-over-semicircle"): {1e-8, 1e-12},  # scipy too at 1e-12
    ("rombergts", "sqrt-tan"): {1e-8, 1e-12},  # scipy too at 1e-12
    ("rombergts", "vector-mixed"): {1e-8, 1e-12},
}


def xfail_if_known(request, method, prob, tol):
    """Mark the running test xfail if ``KNOWN_FAILURES`` lists this case.

    Applied by the tests that sweep the example problems, so that a case in the table
    is reported as an expected failure and one that starts passing is reported as
    ``XPASS`` rather than silently dropping out of the record.
    """
    tols = KNOWN_FAILURES.get((method.__name__, prob["name"]))
    if tols is not None and float(tol) in tols:
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{method.__name__} does not meet the contract on "
                f"{prob['name']} at tol={tol:g}",
                strict=False,
            )
        )
