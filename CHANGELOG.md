Changelog
=========


v0.3.0
------
- **Breaking**: removed ``fixed_quadgk``, ``fixed_quadcc``, and ``fixed_quadts``,
  deprecated since v0.2.2. Use ``GaussKronrodRule``, ``ClenshawCurtisRule``, and
  ``TanhSinhRule`` instead, eg
  ``quadax.GaussKronrodRule(n, norm).integrate(fun, a, b, args)``.
- **Breaking**: ``adaptive_quadrature`` no longer accepts a callable for ``rule``,
  deprecated since v0.2.2. It now raises a ``TypeError``. Custom rules should subclass
  ``quadax.AbstractQuadratureRule``.
- **Breaking**: removed the unused ``norm`` argument to ``adaptive_quadrature``. It had
  no effect; the norm is taken from the rule, ie ``GaussKronrodRule(order, norm)``.


v0.2.13
-------
- Bumped maximum jax version to 0.10.0.


v0.2.12
-------
- Use better mapping for doubly infinite domains
  ([#116](https://github.com/f0uriest/quadax/pull/116)).
- Bumped maximum jax version.


v0.2.11
-------
- Bumped maximum jax version to 0.9.0.


v0.2.10
-------
- Added type hints and static type checking
  ([#86](https://github.com/f0uriest/quadax/pull/86)).


v0.2.9
------
- Bumped maximum jax version to 0.8.
- Bumped maximum equinox version to 0.14.


v0.2.8
------
- Updated min and max jax version. Min is now v0.4.36, max is v0.6.*.


v0.2.7
------
- Bumped maximum jax version to 0.5.3.
- Fixed issue preventing sdist from installing due to missing requirements files
  ([#50](https://github.com/f0uriest/quadax/pull/50)).


v0.2.6
------
- Fixed setup.cfg for installation on windows
  ([#47](https://github.com/f0uriest/quadax/pull/47)).
- Fixed leaked tracers ([#48](https://github.com/f0uriest/quadax/pull/48)).


v0.2.5
------
- Maintenance release for compatibility with jax 0.5.0, numpy 2.2.2, scipy 1.15.1,
  equinox 0.11.11.


v0.2.4
------
- Introduced preliminary support for integrating complex valued functions of a real
  variable.


v0.2.3
------
- Maintenance release, bumping allowed numpy version.


v0.2.2
------
- Added abstract classes for quadrature rules
  ([#6](https://github.com/f0uriest/quadax/pull/6)). ``fixed_quadgk``, ``fixed_quadcc``,
  and ``fixed_quadts``, along with passing a callable for ``rule`` to
  ``adaptive_quadrature``, are deprecated in favor of these.
- Added cumulative Simpson integration
  ([#12](https://github.com/f0uriest/quadax/pull/12)).


v0.2.1
------
- Updated default tolerances to work in either 32 or 64 bit
  ([#3](https://github.com/f0uriest/quadax/pull/3)).
- Bumped requirements ([#4](https://github.com/f0uriest/quadax/pull/4)).


v0.2.0
------
- Input/Output API for most functions is now consistent.
- Breakpoints or locations of discontinuities and singularities in the domain can now be
  specified.
- Vector valued integrands now supported.
- Added Clenshaw-Curtis and tanh-sinh methods.
- Fixed a bug with semi-infinite domains.
- Forward and reverse mode AD tested and working.


v0.1.0
------
Initial release
