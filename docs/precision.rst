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
the precision you asked for, but its own output dtype is carried through, so this
returns float64 while only ever evaluating ``fun`` at float32 abscissae::

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
it saturates and reports the total variation of the integrand over the sub-interval.
That is a genuine upper bound on the error, and never an under-estimate, but it is a
loose one, and it will make the adaptive methods subdivide more than they need to.

:func:`~quadax.quadts` and :func:`~quadax.rombergts` lose most of what makes them
special, and say so with a :exc:`UserWarning`. Their double exponential clustering can
only place a node about ``eps`` of the half width away from an endpoint -- 2.2e-16 at
float64, but 7.8e-3 at bfloat16 -- so at half precision there is no longer any real
clustering near the singularity. Use :func:`~quadax.quadgk` or :func:`~quadax.quadcc`
there, or move to float32.
