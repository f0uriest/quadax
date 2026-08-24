===========
Diagnostics
===========

The info object
---------------

Every adaptive routine returns ``(y, info)``. The integral is ``y``; ``info`` is a
:class:`~quadax.QuadratureInfo` carrying what the solve can tell you about it:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - field
     - meaning
   * - ``err``
     - Estimate of the error in ``y``, in the real counterpart of its dtype.
   * - ``neval``
     - Number of evaluations of the integrand.
   * - ``status``
     - Why the routine stopped, as a member of :class:`~quadax.STATUS`.
   * - ``info``
     - The solver's own state, present only with ``full_output=True``. What it
       holds differs between the adaptive and Romberg routines; see the individual
       docstrings.

Check ``status`` before trusting ``err``, and ``err`` before trusting ``y``::

    y, info = quadgk(fun, interval, epsabs=1e-10)
    assert info.status == quadax.STATUS.normal
    assert info.err < 1e-10

Termination status
------------------

``STATUS.normal`` means the requested tolerances were reached; every other member of
:class:`~quadax.STATUS` names a difficulty and prints as the message explaining it::

    if info.status != quadax.STATUS.normal:
        print(info.status)

Pass ``throw=True`` to have the routine raise that message itself rather than report
it, for code that should not carry on with an unconverged answer::

    y, info = quadgk(fun, interval, epsabs=1e-10, throw=True)

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
