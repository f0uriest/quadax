Which method should I choose?
=============================
Can you evaluate the integrand at an arbitrary point?
-----------------------------------------------------

To start, :func:`~quadax.quadgk` or :func:`~quadax.quadcc` are probably your best
options, and are similar to methods in QUADPACK (or :func:`scipy.integrate.quad`).
:func:`~quadax.quadgk` is usually the most efficient for very smooth integrands (well
approximated by a high degree polynomial), :func:`~quadax.quadcc` tends to be slightly
more efficient for less smooth integrands. If both of those don't perform well, you
should think about your integrand a bit more:

- Does your integrand have badly behaved singularities at the endpoints? Use
  :func:`~quadax.quadts` or :func:`~quadax.rombergts`
- Is your integrand only piecewise smooth or piecewise continuous? Use
  :func:`~quadax.romberg` or :func:`~quadax.rombergts`

Do you only know your integrand at discrete points?
---------------------------------------------------
- Use :func:`~quadax.trapezoid` or :func:`~quadax.simpson`
