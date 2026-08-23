Changelog
=========


v0.3.0
------
- Added convergence acceleration to the adaptive integrators. The sequence of running
  totals is accelerated using Wynn's epsilon algorithm, which can greatly reduce the
  work needed for integrands with algebraic singularities or on infinite intervals.
  ``quadgk`` is then the same algorithm as ``scipy.integrate.quad``.
  - ``quadgk``, ``quadcc``, ``quadts`` and ``adaptive_quadrature`` take a new
    ``extrapolate`` argument controlling it, on by default everywhere except
    ``quadts``. The tanh-sinh rule converges doubly exponentially, so its running
    totals have no geometric tail for the epsilon algorithm to sum and the acceleration
    rarely helps there.
  - The extrapolated value is only returned when its error estimate beats the one from
    the subdivision, so the accuracy is never worse than with ``extrapolate=False``.
  - Smooth integrands on finite domains don't need it, but the additional cost when it
    doesn't help is small and constant.
  - Derivatives are supported as usual, with either adjoint.
  - Results with ``extrapolate=False`` are unchanged.
- Added pluggable adjoints, controlling how derivatives of a quadrature are computed.
 ``quadgk``, ``quadcc``, ``quadts``, ``romberg``, ``rombergts``, and
 ``adaptive_quadrature`` all take a new ``adjoint`` argument, and ``AbstractAdjoint``, ``DirectAdjoint``, and ``LeibnizAdjoint`` are exported at the top level.
  - ``DirectAdjoint`` (the default) differentiates the discretization, reusing the
    subdivision the primal solve converged to. Matches but uses a much faster
    implementation.
  - ``LeibnizAdjoint`` gives the derivative its own adaptive solve, so it gets its own
    error control rather than inheriting the subdivision chosen for the integral. Often
    several times faster for a gradient of a scalar-valued integral.
  - That solve can be configured separately from the integral's, via
    ``LeibnizAdjoint(options={...}, options_fwd={...}, options_rev={...})``.
    ``options_fwd`` and ``options_rev`` configure one direction alone, taking precedence
    over ``options``. This matters most for ``norm``, because the two directions measure
    different vectors: forward mode integrates the tangent of the integrand, of the
    integrand's own shape, while reverse mode integrates the cotangent of the arguments
    being differentiated, whose layout is documented on ``LeibnizAdjoint``.
- Added control over how many integrand evaluations are made in parallel.
  - ``quadgk``, ``quadcc`` and ``quadts`` take a new ``batch_size``, as do the rule
    classes ``GaussKronrodRule``, ``ClenshawCurtisRule`` and ``TanhSinhRule``. It bounds
    how many of the local rule's nodes are evaluated at once, and is clipped at the
    number of nodes. Lowering it bounds the memory an expensive integrand needs, which
    previously could only be done by dropping the order and losing accuracy with it.
  - ``romberg`` and ``rombergts`` take the same argument, where it instead raises the
    width from one point at a time. A level places ``2**(k-1)`` new points, a count only
    known at run time, so the last batch of a level is padded up to a full one and one
    batch shape is traced for every level. ``batch_size=None`` remains the sequential
    loop.
- Fixed a bug in the sub-interval bookkeeping causing wrong results when the number of
  iterations reaches the max allowed (though in this case the solution is marked as
  un-converged anyways)
- ``vmap``-ed integrations should now be faster, by stopping evaluation once every batch
  element has converged, rather than running for the full ``max_ninter`` iterations
  whenever any single element still needs them. Results are unchanged.
- Improved the accuracy of the error estimates used by all the adaptive integrators.
  Reported ``err`` values and iteration counts change throughout, and integrands whose
  error was previously under-estimated now take more work to reach a given tolerance.
  Some integrations that used to report ``status == 0`` while quietly missing the
  requested tolerance now report a non-zero status instead; the values they return are
  more accurate than before, not less.
  - ``status`` is no longer set on an iteration that reaches the requested tolerance. An
    integration that converged just as it ran out of sub-intervals previously reported
    ``MAX_NINTER`` despite having succeeded.
  - Integrands that are simply unresolved, rather than limited by roundoff, are no
    longer written off as ``ROUNDOFF`` while they are still converging.
  - Asking for a tolerance tighter than the arithmetic can deliver is now reported as
    ``ROUNDOFF``, rather than subdividing until the ``max_ninter`` limit is reached.
    Such integrations also finish sooner.
  - The reported error estimate can no longer come back negative.
  - Documented when the nested-rule error estimate can under-state the true error: on
    integrands sampled at fewer than about three points per oscillation (any rule, any
    order), on endpoint singularities under ``ClenshawCurtisRule``, and for
    ``TanhSinhRule`` below order 15. Behaviour is unchanged; this is guidance only.
  - ``y_abs`` and ``y_mmn`` returned by ``AbstractQuadratureRule.integrate`` are now the
    integrals over ``[a, b]`` that their docstrings describe; they were previously
    scaled by a factor of ``2 / (b - a)``. Relevant when calling a rule directly or
    implementing a custom one.
  - ``quadts`` and ``TanhSinhRule`` now account for the mass lying beyond their
    outermost node. A tanh-sinh rule is the trapezoidal rule for an integral over the
    whole real line in the mapped variable, cut off at a finite range, and the terms
    past that cutoff carry mass no other weight compensates for. On a bounded integrand
    the omitted mass is at the level of roundoff and nothing changes; on one singular
    at an endpoint it can be the whole of the error. Reported ``err`` values and
    sub-interval counts change on such integrands, and runs that used to report
    success while missing the requested tolerance now report a non-zero status.
  - ``rombergts``'s reported ``err`` now accounts for the mass its tanh-sinh map leaves
    outside the range it integrates over. That mass is fixed by the map rather than by
    the mesh, so refining converges onto it and the level-to-level movement the estimate
    was built from says nothing about it. Runs that cannot reach the requested tolerance
    because of it now say so rather than reporting success, and stop once refining can
    no longer help instead of spending the rest of ``divmax`` budget.
- Added an open variant of the Clenshaw-Curtis rule. ``ClenshawCurtisRule`` and
  ``quadcc`` take a new ``closed`` argument, defaulting to ``True``, which keeps the
  existing closed rule. With ``closed=False`` the rule uses the Fejer-2 nodes: the same
  ``cos(k*pi/order)`` family with the two endpoints dropped, ``order - 1`` points exact
  to degree ``order - 1``, and the same 2:1 nesting against an embedded ``order // 2``
  rule.
  - The open variant is much cheaper on infinite intervals whose integrand decays
  algebraically or for integrands that are singular at an endpoint.
  - The closed rule remains the default and is the cheaper of the two on smooth, peaked
    and oscillatory integrands, by up to about a factor of two in evaluations.
- ``ClenshawCurtisRule``, ``TanhSinhRule``, ``quadcc``, and ``quadts`` now raise an
  error on an order that would build a malformed rule, rather than silently changing
  the order.
- **Breaking**: removed ``fixed_quadgk``, ``fixed_quadcc``, and ``fixed_quadts``,
  deprecated since v0.2.2. Use ``GaussKronrodRule``, ``ClenshawCurtisRule``, and
  ``TanhSinhRule`` instead, eg
  ``quadax.GaussKronrodRule(n, norm).integrate(fun, a, b, args)``.
- **Breaking**: ``adaptive_quadrature`` no longer accepts a callable for ``rule``,
  deprecated since v0.2.2. It now raises a ``TypeError``. Custom rules should subclass
  ``quadax.AbstractQuadratureRule``.
- **Breaking**: removed the unused ``norm`` argument to ``adaptive_quadrature``. It had
  no effect; the norm is taken from the rule, ie ``GaussKronrodRule(order, norm)``.
- quadax now works in the precision you ask for, rather than always in whatever
  `jax_enable_x64` happens to make the default. The dtype of `interval` is how you ask:
  the integrand is called with an `x` of that dtype, and the result follows it unless
  the integrand upcasts internally, in which case that is respected too. `float16`,
  `bfloat16`, `float32`, `float64` and complex integrands are all supported, and the
  default `epsabs`/`epsrel` follow the working dtype. See "Precision and dtypes" in the
  documentation.
  - Fixes a `TypeError` from `map_interval` that made *any* explicitly-dtyped `interval`
    other than the default unusable, on all of `quadgk`, `quadcc`, `quadts`, `romberg`
    and `rombergts`, for finite and infinite intervals alike.
  - Fixes a `TypeError` from `GaussKronrodRule`/`ClenshawCurtisRule`/`TanhSinhRule`'s
    `integrate` when the integrand's dtype was not the JAX default.
  - The node and weight tables are now built in float64 on the host and rounded once to
    the working dtype. Previously the Clenshaw-Curtis and tanh-sinh tables were
    *computed* in float32 whenever x64 was off, which is less accurate. Building them
    with numpy also makes them independent of the backend's transcendental functions,
    which are not all as accurate as the host's; as a result `quadts` and `rombergts`
    results move in the last ulp or two. `quadgk` and `quadcc` are bit-identical.
  - `simpson` and `cumulative_simpson` no longer return the default float dtype
    regardless of their inputs.
  - No change for float64, or for any configuration that worked before: the default
    tolerances resolve to exactly what they used to in each case.
- `quadts` and `rombergts` now warn when used at `float16`/`bfloat16`, where the
  tanh-sinh clustering can no longer get close enough to an endpoint to be worth having.
- Finite intervals are no longer mapped twice - this helps to reduce roundoff error
- Semi-infinite intervals now use a more numerically stable map to similarly avoid
  roundoff
- Tanh-sinh nodes now sit closer to the endpoints of the domain, improving convergence
  for integrands that are singular at an endpoint. Additional care is also taken when
  constructing tanh-sinh nodes in reduced precision to ensure the nodes are distinct.
- ``romberg`` and ``rombergts`` take a new ``extrapolate`` argument, on by default.
  ``extrapolate=False`` keeps the same nodes and the same halving schedule but returns
  the un-extrapolated estimate instead of the Richardson-extrapolated one. Worth having
  for integrands not smooth enough for the extrapolation's error expansion to hold,
  where it amplifies the error rather than cancelling it. The convergence check and the
  reported ``err`` follow whichever estimate is in use, and the ``table`` returned by
  ``full_output`` then has only its first column filled.
- ``romberg`` and ``rombergts`` take a new ``divmin`` argument, default 4. It sets what
  refinement level the solver starts at (number of initial intervals = ``2**divmin``),
  with the old behavior corresponding to ``divmin=0``. The new default is more efficient
  on accelerators, and is generally more robust against false early termination, with a
  small increase in cost on extremely simple integrands.
- Packaging metadata moved from ``setup.py``/``setup.cfg`` into ``pyproject.toml``.
  Development dependencies are now declared as extras rather than in requirements
  files, so use ``pip install -e ".[dev]"`` (or the narrower ``test``, ``docs``, and
  ``lint`` extras) when working from a checkout.


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
