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
