
########
quadax
########
|License| |DOI| |Issues| |Pypi|

|Docs| |UnitTests| |Codecov|

quadax is a library for numerical quadrature and integration using JAX.

- ``vmap``-able, ``jit``-able, differentiable.
- Scalar or vector valued integrands.
- Finite or infinite domains with discontinuities or singularities within the domain of integration.
- Globally adaptive Gauss-Kronrod and Clenshaw-Curtis quadrature for smooth integrands (similar to ``scipy.integrate.quad``)
- Adaptive tanh-sinh quadrature for singular or near singular integrands.
- Quadrature from sampled values using trapezoidal and Simpsons methods.

Coming soon:

- Custom JVP/VJP rules (currently AD works by differentiating the loop which isn't the most efficient.)
- N-D quadrature (cubature)
- QMC methods
- Integration with weight functions
- Sparse grids (maybe, need to play with data structures and JAX)

Installation
============

quadax is installable with `pip`:

.. code-block:: console

    pip install quadax



Usage
=====

.. code-block:: python

    import jax.numpy as jnp
    import numpy as np
    from quadax import quadgk

    fun = lambda t: t * jnp.log(1 + t)

    epsabs = epsrel = 1e-5 # by default jax uses 32 bit, higher accuracy requires going to 64 bit
    a, b = 0, 1
    y, info = quadgk(fun, [a, b], epsabs=epsabs, epsrel=epsrel)
    assert info.err < max(epsabs, epsrel*abs(y))
    np.testing.assert_allclose(y, 1/4, rtol=epsrel, atol=epsabs)


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
