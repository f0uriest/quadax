========================================
Integrating over more than one dimension
========================================

:func:`~quadax.cubegm` is an n dimensional counterpart of :func:`~quadax.quadgk`: a
globally adaptive routine that subdivides where the difficulty is and returns an error
estimate alongside the value. Its first two arguments read the same way, except that the
integrand takes an abscissa of shape ``(ndim,)`` and the limits cover every axis::

    from quadax import cubegm

    fun = lambda x: jnp.exp(-jnp.sum(x**2))
    y, info = cubegm(fun, jnp.array([[0.0, 1.0], [0.0, 2.0]]))

with ``ndim`` inferred from the length of the limits. Where a cubature differs from a
quadrature is in the region it integrates over, and in which local rule is worth its
cost, both discussed below.


Describing the region
=====================

A box
-----

Limits can be specified in two forms, the first is similar to :func:`~quadax.quadgk`
but lifted to n axes. An array of shape ``(ndim, 2)`` gives the two limits of each axis
and nothing else::

    cubegm(fun, jnp.array([[0.0, 1.0], [0.0, 2.0]]))

Note this is the transpose of stacking the two corners; if you have those, pass
``jnp.stack([a, b], axis=-1)``.

In the second form, a list or tuple gives one entry per axis, and is the form that
carries anything more than two limits. The two mean the same thing where they overlap:
an ``(ndim, 2)`` array and the sequence obtained by iterating over it describe the same
box.

Reversing an axis' limits negates the integral, as it does in one dimension.

Breakpoints
-----------

An axis' entry may hold breakpoints between its two limits, and the axes need not carry
the same number of them::

    fun = lambda x: jnp.abs(x[0] - 0.3) * (x[1] - 0.7)**2
    # breakpoint at the kink along axis 0 at x[0] = 0.3
    cubegm(fun, [jnp.array([0.0, 0.3, 1.0]), jnp.array([0.0, 1.0])])

The initial mesh is the full grid the breakpoints cut, so ``max_nregion`` has to be at
least that many regions. As in one dimension, marking a feature that is really there is
worth far more than any change of rule, since the subdivision no longer has to find it.
In n dimensions a breakpoint marks a whole *plane*, ``x[0] == 0.3``, not a point. Only
axis aligned features can be marked this way. A feature lying along a curve is handled
differently; see `Marking a curved feature`_.

Unbounded axes
--------------

Use ``np.inf`` for an axis that is unbounded on either side or both, mixing bounded and
unbounded axes freely::

    cubegm(fun, jnp.array([[-jnp.inf, jnp.inf], [0.0, jnp.inf]]))

Each axis is mapped to a finite one by the same substitution the one dimensional
routines use. Because that map is separable there is one case where it converges badly,
covered under `Where the cost goes`_.

Limits that depend on the other coordinates
-------------------------------------------

An entry of the list or tuple form may be a **callable** giving that axis' limits, in
the manner of :func:`scipy.integrate.nquad`. It is called as ``lim(x_prev, *args)``,
where ``x_prev`` has shape ``(k,)`` and holds the coordinates of the axes before it, and
it returns that axis' two limits::

    # the triangle 0 <= x <= 1, 0 <= y <= 1 - x
    cubegm(lambda x: x[0] * x[1],
          [jnp.array([0.0, 1.0]),
           lambda xp: jnp.array([0.0, 1.0 - xp[0]])])

Axis ``k`` may depend only on axes ``0`` through ``k-1``, so **the order of the axes is
significant**: the region is described from the outside in. Axis ``0`` may still be a
callable, of ``args`` alone. Limits may depend on limits, one level at a time::

    # the simplex x + y + z <= 1
    cubegm(lambda x: x[0] * x[1] * x[2],
          [jnp.array([0.0, 1.0]),
           lambda xp: jnp.array([0.0, 1.0 - xp[0]]),
           lambda xp: jnp.array([0.0, 1.0 - xp[0] - xp[1]])])

A boundary need not be straight::

    # the unit disc
    cubegm(lambda x: jnp.ones_like(x[0]),
          [jnp.array([-1.0, 1.0]),
           lambda xp: jnp.sqrt(1 - xp[0]**2) * jnp.array([-1.0, 1.0])])

and a limit a callable returns may be unbounded like any other::

    # 0 <= x <= 1, y from x to infinity
    cubegm(lambda x: jnp.exp(-x[1]),
          [jnp.array([0.0, 1.0]),
           lambda xp: jnp.array([xp[0], jnp.inf])])

Under the hood such an axis is integrated over a fixed interval and mapped onto the
limits the callable returns, with the Jacobian folded into the integrand. So the region
is a box again by the time anything integrates over it, and nothing else about the
routine changes. What it costs is discussed under `Where the cost goes`_.

Marking a curved feature
------------------------

A callable may return more than two limits, and the extra ones are breakpoints of that
axis that are free to move with the coordinates before it. That is how a feature lying
along a *curve* gets marked: it becomes a plane of the box actually integrated over,
which the subdivision cuts along like any other breakpoint.

Take an integrand with a kink along ``y = c(x)``, which no entry of a constant
``interval`` can describe::

    c = lambda x: 0.3 + 0.4 * x**2
    fun = lambda x: jnp.abs(x[1] - c(x[0]))

    # unmarked: the kink has to be found by subdivision
    cubegm(fun, [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])])

    # marked: c(x) is returned as a breakpoint of the y axis
    cubegm(fun, [jnp.array([0.0, 1.0]),
                lambda xp: jnp.array([0.0, c(xp[0]), 1.0])])

At ``epsabs = epsrel = 1e-10`` the marked run converges in **2 regions**. The unmarked
one does not converge at all within 4000, and returns an answer good to about 1e-8. The
curve is a co-dimension one feature, and resolving one by subdivision alone is expensive
in a way that grows with dimension; marking it removes the problem rather than paying
for it.


Choosing a rule
===============

:func:`~quadax.cubegm` uses a :class:`~quadax.GenzMalikRule`, a fully symmetric rule
that spends a handful of nodes per region and refines a lot. It is usually the right
default at every dimension, and the only affordable option above three dimensions.
``degree`` selects between the rules of degree 7, 9 (the default), 11 and 13.

The alternative is a :class:`~quadax.TensorProductRule`, which applies a one dimensional
rule along every axis. It spends many nodes per region and refines less, which is the
opposite trade. Reach it through :func:`~quadax.adaptive_cubature`, which is
:func:`~quadax.cubegm` with the rule left to the caller::

    from quadax import adaptive_cubature, TensorProductRule, GaussKronrodRule

    rule = TensorProductRule(GaussKronrodRule(21), ndim=2)
    y, info = adaptive_cubature(rule, fun, interval)

A tensor product rule may also be given a different rule per axis, by passing a list of
them instead of one rule and an ``ndim``, which is worth doing when one axis is far
harder than the others.

Which rule, measured
--------------------

Evaluations of the integrand needed to reach ``epsabs = epsrel = 1e-8``, over the some
representative n dimensional problems, for the four Genz-Malik degrees and for a tensor
product of the fifteen point Gauss-Kronrod rule. Lower is better.

============================================  =====  =====  =====  =====  =====
Integrand over a two dimensional box          d=7    d=9    d=11   d=13   GK15
============================================  =====  =====  =====  =====  =====
``cos(x) cos(y)``, smooth                     147    99     49     69     225
``exp(1j (x + y))``, smooth                   315    99     49     69     225
``cos(10 (x + y))``, oscillatory              21483  9339   9457   3519   1575
a Gaussian ridge 12x narrower along one axis  12495  7557   6027   4623   6075
``1/sqrt(x + y)``, singular at a corner       4053   4521   3773   2277   9225
``|x-a| |y-b|``, kinks marked                 84     132    196    276    900
``|x-a| |y-b|``, kinks not marked             2331   2409   3577   6279   16425
``1/(1+|x|**2)**2`` over the whole plane      90909  66297  49049  35121  28575
============================================  =====  =====  =====  =====  =====

Two things to note:

**Degree tracks smoothness, in both directions.** On the smooth rows degree 11 is three
to six times cheaper than degree 7; on the kinked one with its breakpoints marked,
degree 7 is three times cheaper than degree 13. A higher degree rule places more nodes
per region and is repaid for them only if the integrand is smooth enough over a region
to reward the extra order. The effect grows with dimension: on a three dimensional
smooth integrand degree 11 costs 143 evaluations against degree 7's 4875, and on a three
dimensional cusped ridge degree 7 costs 17199 against degree 13's 82075.

**A tensor product rule is worth it for oscillation.** Oscillatory integrands generally
want high order local rules, and the Genz-Malik family tops out at degree 13, while the
Gauss-Kronrod 15 point rule is exact up to degree 21.

Cost against dimension
----------------------

The same measurement on one smooth integrand, ``exp(-|x|**2)`` over the unit box, as the
dimension grows. Starred entries are not measurements but the size of a *single*
application of the rule, which is the floor on what it could cost:

====  ======  =====  ====  ====  =========
ndim  d=7     d=9    d=11  d=13  GK15
====  ======  =====  ====  ====  =========
2     651     231    245   69    225
3     4875    539    143   245   3375
4     30095   2295   2415  705   50625*
5     150895  8463   5061  1725  759375*
6     598115  14043  9555  3729  11390625*
====  ======  =====  ====  ====  =========

A Genz-Malik rule places a number of nodes that grows like a polynomial in ``ndim``:
69, 245, 705, 1725, 3729 for degree 13 at two through six dimensions, while a tensor
product places ``n**ndim`` of them. At four dimensions one region of a fifteen point
tensor rule already costs more than the entire adaptive solve using a Genz-Malik rule.

Rules of thumb
--------------

* **Start with** :func:`~quadax.cubegm`. It is competitive everywhere and it is the only
  thing that scales past three dimensions.
* **Move** ``degree`` **with the smoothness of the integrand.** Raise it to 11 or 13 for
  a smooth one, especially above three dimensions; drop it to 7 for one with kinks or
  cusps. This is worth more than any other adjustment to the routine, and worth several
  factors either way. The exception is an algebraically decaying integrand on an
  unbounded box, where the map rather than the integrand sets the convergence rate and
  no degree helps; see `Algebraic decay on an unbounded box`_.
* **Consider a** :class:`~quadax.TensorProductRule` **only at two or three dimensions**,
  and mainly for an oscillatory integrand, or where one axis is much harder than the
  others and an anisotropic tensor product can help.
* **Mark what you know.** Giving the kinked integrand above its two breakpoints takes it
  from 2331 evaluations to 84. A breakpoint, on an axis or returned by a limit callable,
  is worth more than any choice of rule.
* **On an accelerator, weigh the node count differently.** These are evaluation counts,
  not wall times. A higher degree rule, or a tensor product one, evaluates more points
  per region and therefore vectorizes better, so it can win on wall time somewhere it
  loses here; ``batch_size`` controls how much of a region's node set is evaluated at
  once. See :doc:`performance`.


Where the cost goes
===================

Curved boundaries
-----------------

Collapsing a region onto a box is exact, but the integrand it produces is only as smooth
as the limit functions are: a curved boundary moves its curvature into the axes before
it. The disc's ``sqrt(1 - x**2)`` has an infinite derivative at ``x = ±1``, and that is
the whole reason the disc costs about thirty regions where the triangle (the same kind
of region, with a straight boundary) costs one. This is inherent to describing a curved
region by its limits, :func:`scipy.integrate.nquad` pays it too, and no breakpoint
helps, the difficulty being a square root rather than a kink.

Where the two limits of an axis meet, the Jacobian vanishes and the contribution goes to
zero, which is harmless: the triangle has such an apex and still converges in one
region.

The limit functions are evaluated once per node per axis they govern, which is cheap for
the usual algebraic ones but worth knowing if a limit is expensive to compute.

Algebraic decay on an unbounded box
-----------------------------------

Each unbounded axis is mapped to a finite one on its own, so the map is separable and
its Jacobian is a product over axes. For an integrand decaying **algebraically** in
``|x|`` that product disagrees with the decay along different directions into a corner
of the mapped box, and the mesh converges there at first order rather than at the rule's
degree. Such a problem costs far more regions than its smoothness suggests, and no
breakpoint helps, the difficulty lying at a point the map sends to infinity.

**No choice of rule or degree helps**, because the difficulty is in the integrand the
rule is handed rather than in the rule. A rule with more nodes does buy a better
constant, a tensor product reaching about twenty five times the accuracy of Genz-Malik
at comparable node counts, but it cannot buy back the order, and under *adaptive*
subdivision even that advantage goes away: the mesh concentrates on the corner, where
what matters is the cost of a region rather than its accuracy, and Genz-Malik is then
several times cheaper. Raising ``degree`` is likewise no help here, which makes this the
one case where the advice above misleads: the integrand is perfectly smooth, and the
extra order still buys nothing.

An integrand decaying exponentially is not affected, nor is one over an axis unbounded
on only one side. Where it does bite, a change of variables in the integrand is the
answer: in spherical coordinates the domain is already a box (radius semi-infinite,
angles finite), which :func:`~quadax.cubegm` integrates over directly, and for a radially
decaying integrand that removes the problem entirely.

Anisotropy
----------

The subdivision cuts one axis of one region at a time, so a difficulty lying along an
axis aligned plane is refined without the axes it spans being refined with it. That is
what makes an axis aligned ridge affordable at all, though not cheap: the Gaussian ridge
in the table above is twelve times narrower along one axis than the other and costs
4623 evaluations where a smooth integrand over the same box costs 49. A feature aligned
with none of the axes is worse still, since resolving it means refining every axis it
crosses. Where the anisotropy or the misalignment is known, a linear change of variables
in the integrand to line the feature up with an axis, or that rescales the axes to make
the integrand roughly isotropic, costs only a constant Jacobian and can be worth more
than any change of rule.


Derivatives
===========

Both adjoints work over a box, and :class:`~quadax.DirectAdjoint` is the default and
usually the cheaper one here: :class:`~quadax.LeibnizAdjoint`'s boundary term is an
integral over a face, so it costs up to two adaptive solves per axis whose limits move,
plus a rule one dimension down. Both shipped rule families can build one; a rule that
cannot has to be handed a face rule as ``options_face={"rule": ...}``, and says so.

A parameter appearing in a limit *callable* is not a moving limit at all as far as the
adjoint is concerned (it is treated as an ordinary parameter of the integrand, since
that is where the transform put it) so it needs no boundary term and no face rule::

    # area of {0 <= x <= 1, 0 <= y <= k(1 - x)}, which is k/2
    area = lambda k: cubegm(
        lambda x: jnp.ones_like(x[0]),
        [jnp.array([0.0, 1.0]),
         lambda xp: jnp.array([0.0, k * (1 - xp[0])])],
    )[0]

    jax.grad(area)(0.8)      # 0.5

A parameter closed over by a limit callable is hoisted out for AD exactly as one closed
over by the integrand is, so it needs no special handling at the call site. Everything
in :doc:`differentiation` otherwise applies unchanged.
