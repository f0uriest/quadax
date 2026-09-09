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


Adaptive integration over a region in several dimensions
--------------------------------------------------------

See :doc:`cubature` for which rule to use at which dimension, and for the regions
``interval`` can describe.

.. autosummary::
    :toctree: _api/
    :recursive:

    cubegm
    adaptive_cubature


Quadrature Rules
----------------

One dimensional rules, taken by :func:`~quadax.adaptive_quadrature` and used to build
tensor product cubature rules.

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: class.rst

    AbstractQuadratureRule
    NestedRule
    GaussKronrodRule
    ClenshawCurtisRule
    TanhSinhRule


Cubature Rules
--------------

Rules over a box in several dimensions, taken by
:func:`~quadax.adaptive_cubature`.

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: class.rst

    AbstractCubatureRule
    GenzMalikRule
    TensorProductRule


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
