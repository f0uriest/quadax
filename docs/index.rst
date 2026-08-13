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
Adaptive algorithms are inherently somewhat sequential, so perfect parallelism
is generally not achievable. ``romberg`` and ``rombergts`` are fully sequential, due to
limitiations on dynamically sized arrays in JAX. All of the ``quad*`` methods are parallelized
on a local level (ie, for each sub-interval, the function evaluations are vectorized).
This means that ``quad*`` methods will evaluate the integrand in batch sizes of ``order``,
and hence higher order methods will tend to be more efficient on GPU/TPU. However, if the
integrand is not sufficiently smooth, using a higher order method can slow down convergence,
particularly for ``quadgk``, ``quadts`` and ``quadcc`` are somewhat less sensitive to the
smoothness of the integrand.



.. toctree::
   :maxdepth: 4
   :caption: Public API

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
