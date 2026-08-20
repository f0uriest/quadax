.. include:: ../README.rst


Which method should I choose?
=============================
Can you evaluate the integrand at an arbitary point?
----------------------------------------------------

To start, ``quadgk`` or ``quadcc`` are probably your best options, and are similar to
methods in QUADPACK (or ``scipy.integrate.quad``). ``quadgk`` is usually the most efficient
for very smooth integrands (well approximated by a high degree polynomial), ``quadcc``
tends to be slightly more efficient for less smooth integrands. If both of those don't
perform well, you should think about your integrand a bit more:

- Does your integrand have badly behaved singularites at the endpoints? Use ``quadts`` or ``rombergts``
- Is your integrand only piecewise smooth or piecewise continuous? Use ``romberg`` or ``rombergts``

Do you only know your integrand at discrete points?
---------------------------------------------------
- Use ``trapezoid`` or ``simspson``


Precision and dtypes
====================
The dtype of ``interval`` is how you ask for a precision, and quadax works in it
throughout. Everything else follows from that and from the integrand:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - what
     - dtype
   * - the ``x`` your integrand is called with, and the sub-interval endpoints
     - the dtype of ``interval``
   * - the returned integral
     - ``jnp.result_type(interval.dtype, fun(x).dtype)``
   * - the reported ``err``, and ``epsabs``/``epsrel``
     - the real counterpart of the above
   * - the default ``epsabs``/``epsrel``
     - ``sqrt(eps)`` of whichever of the first and third is coarser

So to integrate in single precision, say so::

    y, info = quadgk(fun, jnp.array([0.0, 1.0], dtype=jnp.float32))

Note that JAX itself defaults to 32 bit, and silently narrows a float64 array to float32
unless x64 is enabled::

    import jax
    jax.config.update("jax_enable_x64", True)

A plain python list or an integer ``interval`` carries no dtype of its own, and falls
back to the JAX default, so ``quadgk(fun, [0.0, 1.0])`` is float64 with x64 on and
float32 without.

An integrand that upcasts on purpose is respected. It is still *called* with an ``x`` at
the precision you asked for, but its own output dtype is carried through, so this returns
float64 while only ever evaluating ``fun`` at float32 abscissae::

    quadgk(lambda x: g(x.astype(jnp.float64)), jnp.array([0.0, 1.0], dtype=jnp.float32))

Complex integrands are supported. The limits themselves must be real -- the subdivision
has to be able to order the breakpoints -- and ``err`` stays real.

Reduced precision
-----------------
``float16`` and ``bfloat16`` work and return valid results, with two caveats worth
knowing before you rely on them.

The error estimate stops being sharp. quadax uses QUADPACK's estimator, which floors
each sub-interval's error at ``50*eps`` times the integral of ``|f|`` there. In half
precision that floor sits above the point where the estimator would otherwise refine, so
it saturates and reports the total variation of the integrand over the sub-interval. That
is a genuine upper bound on the error, and never an under-estimate, but it is a loose
one, and it will make the adaptive methods subdivide more than they need to.

``quadts`` and ``rombergts`` lose most of what makes them special, and say so with a
``UserWarning``. Their double exponential clustering can only place a node about
``10*eps`` of the half width away from an endpoint -- 2.2e-15 at float64, but 7.8e-2 at
bfloat16 -- so at half precision there is no longer any real clustering near the
singularity. Use ``quadgk`` or ``quadcc`` there, or move to float32.


Notes on parallel efficiency
============================
Adaptive algorithms are inherently somewhat sequential, so perfect parallelism is
generally not achievable. What parallelism there is happens at a local level: for each
sub-interval, the ``quad*`` methods can vectorize the integrand evaluation over all of
the local rule's nodes, controlled via ``batch_size``. The default is to vectorize over
all the local nodes at once, so higher order methods tend to be more efficient on
GPU/TPU. However, if the integrand is not sufficiently smooth, a higher order method
can be less efficient, but often still wins on wall time on accelerators.

``romberg`` and ``rombergts`` are sequential by default. The number of new points a
refinement level places is only known at run time, so there is no single shape for
JAX to vectorize over. Setting ``batch_size`` will vectorize over the points on each
level, but will pad the coarser levels, so the total number of function evaluations
will slightly increase.

In reverse mode there is a second axis to parallelize over, since the sub-intervals of
the converged subdivision are independent. ``DirectAdjoint`` evaluates them
``chunk_size`` at a time, set on the adjoint rather than on the routine::

    quadgk(fun, interval, batch_size=8, adjoint=DirectAdjoint(chunk_size=4))

The two multiply: a gradient evaluates the integrand at up to ``chunk_size`` times
``batch_size`` points at once.

``LeibnizAdjoint`` takes no ``chunk_size``, and no ``checkpoint`` either. It evaluates
no frozen subdivision, its backward pass is an error-controlled solve, and the
derivative with respect to the limits is a boundary term, so ``batch_size`` is the
only knob that applies to it.



Sharp edges
===========

Marking jumps and singularities
-------------------------------
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
--------------------------------------
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
------------------------------------
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

``LeibnizAdjoint`` happens to give the correct answer in the unmarked case, its own
error-controlled solve regularizes the divergent tangent integral, but that is a
property of that adjoint rather than something to rely on.

Only a feature that is steep but *continuous* needs none of this: a sharp ``tanh``
differentiates correctly unmarked under either adjoint. Marking one anyway is harmless,
the breakpoint then only helps the subdivision, and contributes nothing to the
derivative since the integrand has the same limit from both sides of it.

Non-finite integrand values
---------------------------
Points where the integrand evaluates to ``inf`` or ``nan`` are masked to zero rather
than poisoning the result, so an integrable singularity that a node happens to land on
does not spoil the integral. Derivatives are masked too, and correctly in reverse mode
as well as forward, but what is dropped is that point's *contribution* to the
derivative: the quadrature is linearized only where the integrand is finite. For an
integrable singularity that is the right trade, since a single abscissa carries no
weight in the exact integral. It does mean an integrand that is non-finite over a whole
region will quietly integrate that region as zero.

The masking costs a second evaluation of the integrand, on the evaluations that are
actually differentiated. Values, and derivatives taken in forward mode, are unaffected.


.. toctree::
   :maxdepth: 4
   :caption: Public API

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
