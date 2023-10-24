
########
quadax
########
|License| |DOI| |Issues| |Pypi|

|Docs| |UnitTests| |Codecov|

quadax is a library for numerical quadrature and integration using JAX.

- Globally adaptive Gauss-Konrod quadrature for smooth integrands (similar to ``scipy.integrate.quad``)
- Adaptive tanh-sinh quadrature for singular or near singular integrands.
- Quadrature from sampled values using trapezoidal, simpsons, and higher order rules.

Coming soon:

- N-D quadrature (cubature) via iterated 1-D rules, sparse grids, and QMC methods
- Integration with weight functions.

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

    f = lambda t: t * jnp.log(1 + t)

    y, err = quadgk(fun, 0, 1, epsabs=1e-14, epsrel=1e-14)

    np.testing.assert_allclose(y, 1/4, rtol=1e-14, atol=1e-14)


For full details of various options see the `API documentation <https://quadax.readthedocs.io/en/latest/api.html>`__


.. |License| image:: https://img.shields.io/github/license/f0uriest/quadax?color=blue&logo=open-source-initiative&logoColor=white
    :target: https://github.com/f0uriest/quadax/blob/master/LICENSE
    :alt: License

.. |DOI| image:: https://zenodo.org/badge/706703896.svg
    :target: https://zenodo.org/doi/10.5281/zenodo.10028967
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
