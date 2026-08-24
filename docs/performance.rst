Parallel efficiency
===================
Adaptive algorithms are inherently somewhat sequential, so perfect parallelism is
generally not achievable. What parallelism there is happens at a local level: for each
sub-interval, the ``quad*`` methods can vectorize the integrand evaluation over all of
the local rule's nodes, controlled via ``batch_size``. The default is to vectorize over
all the local nodes at once, so higher order methods tend to be more efficient on
GPU/TPU. However, if the integrand is not sufficiently smooth, a higher order method
can be less efficient, but often still wins on wall time on accelerators.

:func:`~quadax.romberg` and :func:`~quadax.tanhsinh` are sequential by default. The
number of new points a refinement level places is only known at run time, so there is no
single shape for JAX to vectorize over. Setting ``batch_size`` will vectorize over the
points on each level, but will pad the coarser levels, so the total number of function
evaluations will slightly increase. Raising ``divmin`` shifts that trade the other way:
the first level then places that many points at once, so a run can start wide enough to
keep an accelerator busy instead of waiting for its levels to grow into it.

In reverse mode there is a second axis to parallelize over, since the sub-intervals of
the converged subdivision are independent. :class:`~quadax.DirectAdjoint` evaluates them
``chunk_size`` at a time, set on the adjoint rather than on the routine::

    quadgk(fun, interval, batch_size=8, adjoint=DirectAdjoint(chunk_size=4))

The two multiply: a gradient evaluates the integrand at up to ``chunk_size`` times
``batch_size`` points at once.

:class:`~quadax.LeibnizAdjoint` takes no ``chunk_size``, and no ``checkpoint`` either.
It evaluates no frozen subdivision, its backward pass is an error-controlled solve, and
the derivative with respect to the limits is a boundary term, so ``batch_size`` is the
only knob of the routine's that applies to it. What it does take is ``options``, giving
that solve its own tolerances, its own ``max_ninter``, even its own local rule; see
:ref:`derivative-solve-options`.
