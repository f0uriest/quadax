Which method should I choose?
=============================

Can you evaluate the integrand at an arbitrary point?
-----------------------------------------------------

Start with :func:`~quadax.quadgk`. It is the closest thing here to a general purpose
integrator, similar to QUADPACK or :func:`scipy.integrate.quad`, and across the example
integrals the test suite covers it is both the most reliable and usually the cheapest,
on smooth and non-smooth integrands alike. Reach for something else only when it has a
specific reason to be a poor fit:

- **Does the integrand have a singularity at an endpoint?** Stay with
  :func:`~quadax.quadgk`. Its convergence acceleration (``extrapolate=True``, the
  default) is built for algebraic and logarithmic endpoint singularities, and generally
  reaches the tolerance in fewer evaluations than a double exponential method such as
  :func:`~quadax.quadts`, which also has a truncation floor that no amount of
  refinement gets past.

- **Is the interval infinite and** :func:`~quadax.quadgk` **struggling?**
  :func:`~quadax.tanhsinh` is the most robust option on infinite intervals, at several
  times the cost. It applies a double exponential substitution and refines a uniform
  mesh rather than subdividing, which lets it keep going where adaptive subdivision
  stops short.

- **Do you want a method with minimal overhead for smooth integrands?**
  :func:`~quadax.romberg` refines the whole interval uniformly rather than subdividing
  where the difficulty is. That makes its cost predictable and its control flow
  independent of the integrand, which can make it the fastest in terms of wall clock
  time on cheap integrands. Setting a large ``divmin`` can also make it very efficient
  on accelerators.

- **Is the integrand piecewise smooth, or singular in the interior?** Stay with
  :func:`~quadax.quadgk` or :func:`~quadax.quadcc`, and pass the location of the break
  as a breakpoint in ``interval``. Marking the break is worth far more than any change
  of method, since the mesh then never has to find it. The Romberg routines are the
  wrong choice here and do not accept breakpoints at all.

:func:`~quadax.quadcc` is a reasonable alternative to :func:`~quadax.quadgk` rather than
a specialist. The two behave similarly; measured over the test suite's integrands
Gauss-Kronrod is somewhat cheaper on most of them, including the less smooth ones, so
prefer it unless you have a reason not to. Which of :func:`~quadax.quadcc`'s two rules
to use depends on the integrand: the default closed rule is cheaper on smooth and
peaked ones, while the open rule (``closed=False``) reaches the requested tolerance on
more integrands, and is the one to use for an endpoint singularity or an infinite
interval, since it never evaluates at the limits. One place :func:`~quadax.quadcc` has
an advantage over :func:`~quadax.quadgk` is on smooth but highly oscillatory integrands,
where using a high order rule pays off. Choosing the order to have ~7-8 points per
period is usually optimal.

Do you only know your integrand at discrete points?
---------------------------------------------------
- Use :func:`~quadax.trapezoid` or :func:`~quadax.simpson`
