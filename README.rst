
.. image:: https://raw.githubusercontent.com/f0uriest/quadax/main/docs/_static/images/logo.png
    :alt: quadax
    :width: 420

|License| |DOI| |Issues| |Pypi|

|Docs| |UnitTests| |Codecov|

.. docs-include-start

quadax is a library for numerical quadrature and integration using JAX.

- ``vmap``-able, ``jit``-able, differentiable in both forward and reverse mode.
- Pluggable ``adjoint`` methods controlling how derivatives are computed, including
  with respect to the limits of integration and to moving discontinuities.
- Scalar, vector, or matrix valued integrands, real or complex.
- Finite or infinite domains, with discontinuities or singularities inside the domain
  of integration.
- Globally adaptive Gauss-Kronrod and Clenshaw-Curtis quadrature for general
  integrands (similar to ``scipy.integrate.quad``), with rules of arbitrary order.
- tanh-sinh and Romberg quadrature using uniform refinement.
- Globally adaptive cubature in several dimensions, over boxes or over regions whose
  limits depend on the other coordinates (similar to ``scipy.integrate.nquad``).
- Quadrature from sampled values using trapezoidal and Simpson's rules.
- Every routine reports an error estimate, the number of evaluations, and a status
  code saying why it stopped.

Full documentation, including a user guide and API reference, is at
`quadax.readthedocs.io <https://quadax.readthedocs.io/en/latest/>`__.

Coming soon:

- QMC methods
- Integration with weight functions
- Sparse grids (maybe, need to play with data structures and JAX)

Installation
============

quadax is installable with ``pip``:

.. code-block:: console

    pip install quadax


Usage
=====

JAX uses 32 bit floats by default. The examples below turn on 64 bit precision to
show what the methods can reach:

.. code-block:: python

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

Basic integration
-----------------

Every routine returns the integral along with an ``info`` tuple holding an error
estimate, the number of integrand evaluations, and a status code.

.. code-block:: python

    from quadax import quadgk

    y, info = quadgk(lambda t: t * jnp.log(1 + t), [0, 1], epsabs=1e-12, epsrel=1e-12)
    # y = 0.25, info.err ~ 3e-15, info.neval = 21, info.status = 0 (converged)

Infinite intervals and breakpoints
----------------------------------

Use ``jnp.inf`` for an unbounded limit. If the integrand has a jump, kink, or
singularity at a known interior point, pass that point in ``interval``:

.. code-block:: python

    y, info = quadgk(lambda x: jnp.exp(-x**2), [-jnp.inf, jnp.inf])  # sqrt(pi)

    f = lambda x: jnp.where(x < 1, x**2, jnp.cos(x))
    y, info = quadgk(f, [0, 1, 3])  # 42 evaluations; [0, 3] takes 399

Vector valued integrands and extra arguments
--------------------------------------------

An integrand can return an array of any shape, and gets extra arguments through
``args``. The whole array is integrated over one shared adaptive mesh:

.. code-block:: python

    k = jnp.arange(1, 6)
    y, info = quadgk(lambda x, k: jnp.exp(-x) * jnp.cos(k * x), [0, jnp.inf], args=(k,))
    # y = 1 / (1 + k**2)

Composing with ``jit`` and ``vmap``
-----------------------------------

All routines work under JAX transformations, so you can batch over parameters or
limits and compile the whole thing:

.. code-block:: python

    @jax.jit
    @jax.vmap
    def laplace_sin(s):
        y, _ = quadgk(lambda t: jnp.exp(-s * t) * jnp.sin(t), [0, jnp.inf])
        return y

    laplace_sin(jnp.linspace(0.5, 3, 4))  # 1 / (1 + s**2)

Differentiation
---------------

Integrals are differentiable with respect to parameters and limits, in forward and
reverse mode:

.. code-block:: python

    from quadax import LeibnizAdjoint

    def Gamma(s):
        integrand = lambda t, s: t ** (s - 1) * jnp.exp(-t)
        return quadgk(integrand, [0, jnp.inf], args=(s,))[0]

    jax.grad(Gamma)(3.5)  # Gamma(3.5) * digamma(3.5)
    jax.grad(lambda b: quadgk(jnp.sin, [0, b])[0])(1.2)  # sin(1.2)

The ``adjoint`` argument controls how the derivative is computed. The default,
``DirectAdjoint``, differentiates the mesh the solve converged on, while
``LeibnizAdjoint`` runs a second adaptive solve with its own error control:

.. code-block:: python

    def Gamma(s):
        integrand = lambda t, s: t ** (s - 1) * jnp.exp(-t)
        return quadgk(integrand, [0, jnp.inf], args=(s,), adjoint=LeibnizAdjoint())[0]

If a discontinuity moves with a parameter, mark it with a breakpoint that depends on
that same parameter, and the derivative includes the contribution from the jump:

.. code-block:: python

    def area(c):
        f = lambda x, c: jnp.where(x < c, jnp.exp(x), 0.0)
        return quadgk(f, [0, c, 2], args=(c,))[0]

    jax.grad(area)(1.0)  # e; integrating over [0, 2] with no breakpoint gives 0

Multidimensional integrals
--------------------------

``cubegm`` does globally adaptive cubature with Genz-Malik rules. The region can be a
box, or each axis' limits can be a function of the coordinates before it:

.. code-block:: python

    from quadax import cubegm

    # Gaussian over the cube [-1, 1]^3
    y, info = cubegm(lambda x: jnp.exp(-x @ x), [[-1, 1]] * 3)

    # Same integrand over the unit disk: the y limits depend on x
    disk = [[-1, 1], lambda x: jnp.sqrt(1 - x[0] ** 2) * jnp.array([-1, 1])]
    y, info = cubegm(lambda x: jnp.exp(-x @ x), disk)  # pi * (1 - 1/e)

Sampled data
------------

When the integrand is only known at a set of points:

.. code-block:: python

    from quadax import simpson, cumulative_trapezoid

    x = jnp.linspace(0, jnp.pi, 101)
    simpson(jnp.sin(x), x=x)  # 2
    cumulative_trapezoid(jnp.sin(x), x=x, initial=0)  # running integral

Checking the result
-------------------

A run that stops before reaching the tolerance still returns a value, and
``info.status`` says why it stopped. ``quadax.STATUS`` turns the code into a message,
or pass ``throw=True`` to raise an error instead:

.. code-block:: python

    from quadax import STATUS

    y, info = quadgk(lambda x: 1 / jnp.sqrt(jnp.abs(jnp.sin(50 * x))), [0, 10], max_ninter=10)
    if info.status != STATUS.normal:
        print(STATUS[info.status])  # explains the failure and what to try next

Which method?
-------------

===================================================  ================================
Situation                                            Start with
===================================================  ================================
Anything, if you're not sure                         ``quadgk``
Infinite interval where ``quadgk`` struggles         ``tanhsinh``
Smooth, cheap integrand where wall time matters      ``romberg``
Highly oscillatory, smooth integrand                 ``quadcc`` with a high ``order``
Integrals in several dimensions                      ``cubegm``
Only sampled values                                  ``simpson``, ``trapezoid``
===================================================  ================================

The `user guide <https://quadax.readthedocs.io/en/latest/>`__ has more on choosing a
method, precision, differentiation, performance, and diagnosing failures.

.. docs-include-end

For full details of various options see the `API documentation <https://quadax.readthedocs.io/en/latest/api.html>`__


.. |License| image:: https://img.shields.io/github/license/f0uriest/quadax?color=blue&logo=open-source-initiative&logoColor=white
    :target: https://github.com/f0uriest/quadax/blob/master/LICENSE
    :alt: License

.. |DOI| image:: https://zenodo.org/badge/709132830.svg
    :target: https://zenodo.org/doi/10.5281/zenodo.10035983
    :alt: DOI

.. |Docs| image:: https://img.shields.io/readthedocs/quadax?logo=Read-the-Docs
    :target: https://quadax.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation

.. |UnitTests| image:: https://github.com/f0uriest/quadax/actions/workflows/unittest.yml/badge.svg
    :target: https://github.com/f0uriest/quadax/actions/workflows/unittest.yml
    :alt: UnitTests

.. |Codecov| image:: https://codecov.io/github/f0uriest/quadax/graph/badge.svg?token=MB11I7WE3I
    :target: https://codecov.io/github/f0uriest/quadax
    :alt: Coverage

.. |Issues| image:: https://img.shields.io/github/issues/f0uriest/quadax
    :target: https://github.com/f0uriest/quadax/issues
    :alt: GitHub issues

.. |Pypi| image:: https://img.shields.io/pypi/v/quadax
    :target: https://pypi.org/project/quadax/
    :alt: Pypi
