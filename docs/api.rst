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
