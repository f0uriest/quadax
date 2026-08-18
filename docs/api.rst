=================
API Documentation
=================

.. currentmodule:: quadax

Adaptive integration of a callable function or method
-----------------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    quadgk    -- General purpose integration using Gauss-Kronrod scheme
    quadcc    -- General purpose integration using Clenshaw-Curtis scheme
    quadts    -- General purpose integration using tanh-sinh (aka double exponential) scheme
    romberg   -- Adaptive trapezoidal integration with Richardson extrapolation
    rombergts -- Adaptive tanh-sinh integration with Richardson extrapolation


Quadrature Rules
----------------

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: class.rst

    AbstractQuadratureRule -- Abstract base class for all quadrature rules
    GaussKronrodRule       -- Fixed order integration over finite interval using Gauss-Kronrod scheme
    ClenshawCurtisRule     -- Fixed order integration over finite interval using Clenshaw-Curtis scheme
    TanhSinhRule           -- Fixed order integration over finite interval using tanh-sinh (aka double exponential) scheme


.. _adjoints:

Adjoints
--------

Adjoints control how derivatives of a quadrature are computed, without changing what the
quadrature itself returns. Pass one as the ``adjoint`` argument.

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: class.rst

    AbstractAdjoint       -- Abstract base class for all adjoint methods
    DirectAdjoint         -- Exact derivative of discretized problem on the converged subdivision, aka "discretize then optimize" or "discrete adjoint" (default)
    LeibnizAdjoint        -- Leibniz rule with an error controlled derivative solve, aka "optimize then discretize" or "continuous adjoint"

Choosing an adjoint
~~~~~~~~~~~~~~~~~~~

There are two, both supporting forward and reverse mode.

:class:`~quadax.DirectAdjoint` is the default. It differentiates the discretization the
primal solve settled on, so the derivative costs no error control of its own, and for a
cheap integrand it is usually the cheaper option in either mode.

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
on a discontinuity. Neither can see a discontinuity that has no breakpoint at it -
declare one there.

:func:`~quadax.romberg` and :func:`~quadax.rombergts` have no subdivision to reuse -
``DirectAdjoint`` freezes the number of Richardson levels instead - and there the two
cost about the same, so the choice is about accuracy alone.

These are rules of thumb, not laws. The balance shifts with how expensive the integrand
is relative to the quadrature around it, how hard its derivative is to integrate
compared to the integrand, and how many parameters are involved. Time both on your own
problem before caring much about the difference.

Both take two options controlling the memory a derivative needs. ``checkpoint`` (on by
default) recomputes each block of sub-intervals during the backward pass instead of
storing it, and is where nearly all of the saving is. ``chunk_size`` sets how many
sub-intervals of the frozen subdivision are evaluated at once, and multiplies with the
``batch_size`` of the routine itself: a gradient evaluates the integrand at up to
``chunk_size * batch_size`` points at a time.


Integrating function from sampled values
----------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    trapezoid            -- Use trapezoidal rule to approximate definite integral.
    cumulative_trapezoid -- Use trapezoidal rule to approximate indefinite integral.
    simpson              -- Use Simpson's rule to compute integral from samples.
    cumulative_simpson   -- Use Simpson's rule to approximate indefinite integral.


Low level routines and wrappers
-------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    adaptive_quadrature -- Custom h-adaptive quadrature using user specified local rule.
