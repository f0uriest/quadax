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

:class:`~quadax.DirectAdjoint` takes two options controlling the memory a derivative
needs. ``checkpoint`` (off by default) recomputes each block of sub-intervals during the
backward pass instead of storing it, rather than replaying the frozen subdivision.
``chunk_size`` sets how many sub-intervals of that subdivision are evaluated at once,
and multiplies with the ``batch_size`` of the routine itself: a gradient evaluates the
integrand at up to ``chunk_size * batch_size`` points at a time.
:class:`~quadax.LeibnizAdjoint` takes neither. It replays no subdivision, its backward
pass being an error controlled solve, and the derivative with respect to the limits is
a boundary term rather than a mesh evaluation.

Options for the derivative solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~quadax.LeibnizAdjoint` runs a solve of its own, and it does not have to be the
solve the integral got. Give it ``options`` to override what the routine was called
with::

    quadgk(fun, interval, args, epsabs=1e-6,
           adjoint=LeibnizAdjoint(options={"epsabs": 1e-10, "max_ninter": 200}))

The integral is then computed to 1e-6 and its derivative to 1e-10, each stopping when it
has converged rather than sharing a budget. :class:`~quadax.DirectAdjoint` takes no
options of this kind, having no solve of its own to configure.

Which vector the norm measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two directions of the derivative do not integrate the same vector, which is what
``options_fwd`` and ``options_rev`` are for. They take the same names as ``options`` and
take precedence over it, for one direction alone.

Forward mode integrates the tangent of the integrand, a vector of the integrand's own
shape. Reverse mode integrates the cotangent of the arguments being differentiated,
raveled into a single flat vector. Its length is the total number of differentiated
parameter components, which has nothing to do with the integrand's shape, and its
entries are parameters rather than components of the integral. They appear in the order
the quadrature's own arguments do: the limits, ``args``, and whatever the integrand
closes over. Each contributes ``size`` entries if it is being differentiated and none
at all if it is not. Differentiating a two point ``interval`` and a length three
``args[0]`` gives a vector of five with the limits first; differentiating ``args[0]``
alone gives a vector of three::

    weights = jnp.array([1.0, 10.0, 100.0])          # one per parameter
    norm = lambda x: jnp.linalg.norm(x * weights, ord=2)
    jax.grad(lambda c: quadgk(fun, interval, (c,),
                              adjoint=LeibnizAdjoint(options_rev={"norm": norm}))[0])(c)

Putting that norm in ``options_rev`` rather than ``options`` is what stops it from being
handed a tangent instead, which the two vectors being the same length by coincidence
would otherwise hide.


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
