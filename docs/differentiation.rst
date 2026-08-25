===============
Differentiation
===============

Every quadrature in quadax is differentiable in both forward and reverse mode.
This page covers how that derivative is computed, and the cases where getting a
correct one takes more than calling :func:`jax.grad`.

.. _adjoints:

Adjoints
--------

Adjoints control how derivatives of a quadrature are computed, without changing what
the quadrature itself returns. Pass one as the ``adjoint`` argument. The classes
themselves are listed under :ref:`adjoints-api`.

Choosing an adjoint
~~~~~~~~~~~~~~~~~~~

There are two, both supporting forward and reverse mode.

:class:`~quadax.DirectAdjoint` is the default. It differentiates the discretization the
primal solve settled on, so the derivative costs no error control of its own, and for a
cheap integrand it is usually the cheaper option in either forward or reverse mode.

:class:`~quadax.LeibnizAdjoint` instead evaluates the derivative with a second adaptive
solve, giving it its own error control rather than inheriting the subdivision chosen for
the integral. This buys:

* **Accuracy.** When the derivative of the integrand is sharply peaked somewhere the
  integrand itself is smooth, the subdivision that resolves the integral need not
  resolve its derivative. On such a problem at a loose tolerance the difference can be
  several orders of magnitude; at a tolerance tight enough that the integral's
  subdivision resolves the derivative anyway, it may be less of an issue.
* **Speed when the integrand is expensive.** The derivative solve stops as soon as the
  derivative has converged, rather than covering the subdivision the integral needed, so
  the more one evaluation of the integrand costs, the more there is to save. This
  applies to both forward and reverse modes.

Against that, its reverse pass carries the workspace of the second solve, so on a scalar
integrand it generally costs more memory than the default; on a vector or matrix valued
integrand, where the stored subdivision is what dominates, it can cost less.

Both pick up the jump term from differentiating with respect to a breakpoint that sits
on a discontinuity. Neither can see a discontinuity that has no breakpoint at it, so be
sure to declare a breakpoint, see :ref:`derivative-sharp-edges` below.

:func:`~quadax.romberg` and :func:`~quadax.tanhsinh` have no subdivision to reuse -
:class:`~quadax.DirectAdjoint` freezes the number of Richardson levels instead - and
there the two cost about the same, so the choice is about accuracy alone.

These are rules of thumb, not laws. The balance shifts with how expensive the integrand
is relative to the quadrature around it, how hard its derivative is to integrate
compared to the integrand, and how many parameters are involved. Time both on your own
problem before caring much about the difference.

:class:`~quadax.DirectAdjoint` takes two options controlling the memory a derivative
needs. ``checkpoint`` (off by default) recomputes each block of sub-intervals during the
backward pass instead of storing it, rather than replaying the frozen subdivision.
``chunk_size`` sets how many sub-intervals of that subdivision are evaluated at once,
and multiplies with the ``batch_size`` of the routine itself: a gradient evaluates the
integrand at up to ``chunk_size * batch_size`` points at a time.
:class:`~quadax.LeibnizAdjoint` takes neither. It replays no subdivision, its backward
pass being an error controlled solve, and the derivative with respect to the limits is
a boundary term rather than a mesh evaluation.

.. _derivative-solve-options:

Options for the derivative solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~quadax.LeibnizAdjoint` runs a solve of its own, and it does not have to be the
solve the integral got. Give it ``options`` to override what the routine was called
with::

    quadgk(fun, interval, args, epsabs=1e-6,
           adjoint=LeibnizAdjoint(options={"epsabs": 1e-10, "max_ninter": 200}))

The integral is then computed to 1e-6 and its derivative to 1e-10, each stopping when it
has converged rather than sharing a budget. :class:`~quadax.DirectAdjoint` takes no
options of this kind, having no solve of its own to configure.

Which vector the norm measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two directions of the derivative do not integrate the same vector, which is what
``options_fwd`` and ``options_rev`` are for. They take the same names as ``options`` and
take precedence over it, for one direction alone.

Forward mode integrates the tangent of the integrand, a vector of the integrand's own
shape. Reverse mode integrates the cotangent of the arguments being differentiated,
raveled into a single flat vector. Its length is the total number of differentiated
parameter components, which has nothing to do with the integrand's shape, and its
entries are parameters rather than components of the integral. They appear in the order
the quadrature's own arguments do: the limits, ``args``, and whatever the integrand
closes over. Each contributes ``size`` entries if it is being differentiated and none
at all if it is not. Differentiating a two point ``interval`` and a length three
``args[0]`` gives a vector of five with the limits first; differentiating ``args[0]``
alone gives a vector of three::

    weights = jnp.array([1.0, 10.0, 100.0])          # one per parameter
    norm = lambda x: jnp.linalg.norm(x * weights, ord=2)
    jax.grad(lambda c: quadgk(fun, interval, (c,),
                              adjoint=LeibnizAdjoint(options_rev={"norm": norm}))[0])(c)

Putting that norm in ``options_rev`` rather than ``options`` is what stops it from being
handed a tangent instead, which the two vectors being the same length by coincidence
would otherwise hide.

.. _derivative-sharp-edges:

Sharp edges
-----------

Marking jumps and singularities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Mark a jump or a singularity with a breakpoint, and build that breakpoint from the same
parameter that positions the feature::

    lambda s: quadgk(fun, jnp.array([lo, s, hi]), (jnp.array([s]),))

Marking a feature that is genuinely there is never worse than leaving it unmarked. It
helps the value, often a lot, and for derivatives it is frequently the difference
between a correct answer and a silently wrong one. The rest of this section is what goes
wrong without it.

There is one way marking can hurt, and it is marking something that is not there: a
breakpoint that *moves* while the feature stays put is a false declaration, and is
discussed below.

Differentiating a moving discontinuity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Differentiating a jump gives a delta, and no quadrature of the integrand's tangent can
represent one, so the jump is recovered from the motion of the breakpoint instead. That
only works if the breakpoint moves with the discontinuity::

    step = lambda t, z: jnp.where(t > z[0], 1.0, 0.0)   # jumps at t = z[0]

    # correct: one parameter `s` positions both the jump and the breakpoint
    f = lambda s: quadgk(step, jnp.array([-1.0, s, 1.0]), (jnp.array([s]),))[0]
    jax.grad(f)(0.3)            # -1.0

Both of the following return the same primal *value*, 0.7, and both give zero where the
derivative is ``-1``::

    # WRONG: the jump is unmarked, so nothing tracks it
    f = lambda s: quadgk(step, jnp.array([-1.0, 1.0]), (jnp.array([s]),))[0]

    # WRONG: marked, but with a constant, which carries no derivative.
    f = lambda s: quadgk(step, jnp.array([-1.0, 0.3, 1.0]), (jnp.array([s]),))[0]

Splitting the feature across two parameters is what the *same parameter* requirement
rules out. The derivative with respect to a breakpoint on its own is not a well posed
question: the value of the integral does not depend on where the mesh is cut, so a
breakpoint only means anything in combination with the integrand it marks. Ask for it
anyway, by differentiating with respect to ``[breakpoint, jump location]`` at
``[0.3, 0.3]``, and the answer is ``[-1, 0]``. The total over the two is right, and how
it is divided between them is an artifact of having written one feature as two
parameters, not a property of the quadrature.

The parameter may reach the breakpoint through any expression, not only directly::

    # also correct: the jump sits at sin(2s), and so does the breakpoint
    f = lambda s: quadgk(lambda t, z: jnp.where(t > jnp.sin(2 * z[0]), 2 * z[0], z[0]),
                         jnp.stack([-jnp.ones_like(s), jnp.sin(2 * s),
                                    jnp.ones_like(s)]),
                         (jnp.atleast_1d(s),))[0]

Differentiating a moving singularity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The same rule applies, for the same reason. An unmarked singularity slides across a
subdivision that does not follow it, and the derivative picks up the mesh rather than
the integral::

    f = lambda s: quadgk(lambda t, z: 1 / jnp.sqrt(jnp.abs(t - z[0])),
                         jnp.array([-1.0, 1.0]), (jnp.array([s]),))[0]
    jax.grad(f)(0.3)      # -807.3 under DirectAdjoint; the answer is -0.318

    # correct under both adjoints: mark it, tied to the same `s`
    f = lambda s: quadgk(lambda t, z: 1 / jnp.sqrt(jnp.abs(t - z[0])),
                         jnp.array([-1.0, s, 1.0]), (jnp.array([s]),))[0]
    jax.grad(f)(0.3)      # -0.31817059

:class:`~quadax.LeibnizAdjoint` happens to give the correct answer in the unmarked case,
its own error-controlled solve regularizes the divergent tangent integral, but that is a
property of that adjoint rather than something to rely on.

Only a feature that is steep but *continuous* needs none of this: a sharp ``tanh``
differentiates correctly unmarked under either adjoint. Marking one anyway is harmless,
the breakpoint then only helps the subdivision, and contributes nothing to the
derivative since the integrand has the same limit from both sides of it.
