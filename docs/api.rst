=================
API Documentation
=================

.. currentmodule:: quadax

Adaptive integration of a callable function or method
-----------------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    quadgk    -- General purpose integration using Gauss-Konrod scheme
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
    GaussKronrodRule       -- Fixed order integration over finite interval using Gauss-Konrod scheme
    ClenshawCurtisRule     -- Fixed order integration over finite interval using Clenshaw-Curtis scheme
    TanhSinhRule           -- Fixed order integration over finite interval using tanh-sinh (aka double exponential) scheme


Integrating function from sampled values
----------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    trapezoid            -- Use trapezoidal rule to approximate definite integral.
    cumulative_trapezoid -- Use trapezoidal rule to approximate indefinite integral.
    simpson              -- Use Simpson's rule to compute integral from samples.


Low level routines and wrappers
-------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    adaptive_quadrature -- Custom h-adaptive quadrature using user specified local rule.
