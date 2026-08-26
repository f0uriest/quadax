=================
API Documentation
=================

.. currentmodule:: quadax

Adaptive integration of a callable function or method
-----------------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    quadgk
    quadcc
    quadts
    romberg
    tanhsinh
    adaptive_quadrature


Quadrature Rules
----------------

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: class.rst

    AbstractQuadratureRule
    NestedRule
    GaussKronrodRule
    ClenshawCurtisRule
    TanhSinhRule


.. _adjoints-api:

Adjoints
--------

Adjoints control how derivatives of a quadrature are computed, without changing what the
quadrature itself returns. Pass one as the ``adjoint`` argument. See
:doc:`differentiation` for how to choose between them, how to give the derivative solve
options of its own, and which vector its norm measures.

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: class.rst

    AbstractAdjoint
    DirectAdjoint
    LeibnizAdjoint


Results and termination status
------------------------------

Every iterative routine returns its result alongside a :class:`~quadax.QuadratureInfo`,
whose ``status`` says why it stopped. See :doc:`diagnostics` for how to read one and
what to do about each code.

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: plain.rst

    QuadratureInfo

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: plain.rst

    STATUS


Integrating function from sampled values
----------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    trapezoid
    cumulative_trapezoid
    simpson
    cumulative_simpson
