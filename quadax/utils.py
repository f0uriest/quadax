"""Utility functions for parsing inputs, mapping coordinates etc."""

import functools
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.internal import unvmap_any
from jax.typing import ArrayLike


def errorif(cond: bool | jax.Array, err: type[Exception] = ValueError, msg: str = ""):
    """Raise an error if condition is met.

    Similar to assert but allows wider range of Error types, rather than
    just AssertionError.
    """
    if cond:
        raise err(msg)


class DTypes(NamedTuple):
    """The working dtypes of a quadrature.

    quadax takes the dtype of ``interval`` as the statement of what precision the user
    wants, and derives everything else from it and from the integrand. An integrand that
    deliberately upcasts is respected: it is still *called* with an abscissa at the
    requested precision, but its own output dtype is carried through to the result.

    Parameters
    ----------
    xtype : dtype
        Abscissae. The sub-interval endpoints, the node tables, and the ``x`` the user's
        integrand is called with. Taken from ``interval``.
    ytype : dtype
        Integrand values, the per-interval contributions, and the returned integral. May
        be complex.
    etype : dtype
        Real. Weight tables, error estimates, tolerances. The real counterpart of
        ``ytype``, so that a complex integrand contracted with real weights promotes to
        complex on its own.
    toltype : dtype
        Real. Sets the default ``epsabs``/``epsrel`` of ``sqrt(eps)``. The coarser of
        ``xtype`` and ``etype``: a float32 abscissa limits the achievable accuracy
        however precisely the integrand itself is evaluated.

    """

    xtype: Any
    ytype: Any
    etype: Any
    toltype: Any


def _real_dtype(dtype) -> Any:
    """Real counterpart of ``dtype`` (float32 for complex64, and itself for real)."""
    return jnp.finfo(dtype).dtype


def tree_where(cond, new, old):
    """``jnp.where(cond, new, old)`` leafwise over two pytrees of the same structure."""
    return jax.tree_util.tree_map(lambda n, o: jnp.where(cond, n, o), new, old)


def _coarser_dtype(dtype1, dtype2) -> Any:
    """Whichever of the two has the larger machine epsilon."""
    if float(jnp.finfo(dtype1).eps) >= float(jnp.finfo(dtype2).eps):
        return dtype1
    return dtype2


def resolve_dtypes(
    interval: jax.Array, fun: Callable[..., jax.Array], args: tuple[Any, ...] = ()
) -> DTypes:
    """Work out the dtypes of a quadrature from its limits and its integrand.

    The single point at which quadax decides what precision it is working in. See
    :class:`DTypes` for what each one governs.
    """
    xtype = jnp.asarray(interval).dtype
    # `jnp.zeros((), xtype)` rather than `jnp.array(0.0)`: the latter is *weakly* typed,
    # which both hides the requested precision and lets different expressions involving
    # it settle on different dtypes. See the note on `MAPFUNS`.
    f = jax.eval_shape(fun, jnp.zeros((), xtype), *args)
    ytype = jnp.result_type(xtype, f.dtype)
    etype = _real_dtype(ytype)
    return DTypes(xtype, ytype, etype, _coarser_dtype(xtype, etype))


def _map_identity(t: jax.Array, a: jax.Array, b: jax.Array):
    """For finite intervals no mapping is needed.

    Mapping twice introduces extra roundoff error.
    """
    del a, b
    return t.squeeze(), jnp.ones_like(t).squeeze()


def _map_identity_inv(x: jax.Array, a: jax.Array, b: jax.Array):
    """Leave a point in [a, b] where it is."""
    del a, b
    return x.squeeze()


def _map_linear(t: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point t in [-1, 1] to x in [a, b]."""
    c = (b - a) / 2
    d = (b + a) / 2
    x = d + c * t
    w = c * jnp.ones_like(t)
    return x.squeeze(), w.squeeze()


def _map_linear_inv(x: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point x in [a, b] to t in [-1, 1]."""
    c = (b - a) / 2
    d = (b + a) / 2
    t = (x - d) / c
    return t.squeeze()


def _map_ninfinf(t: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point t in [-1, 1] to x in [-inf, inf]."""
    x = jnp.tan(t * jnp.pi / 2)
    w = jnp.pi / 2 / jnp.cos(jnp.pi * t / 2) ** 2
    return x.squeeze(), w.squeeze()


def _map_ninfinf_inv(x: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point x in [-inf, inf] to t in [-1, 1]."""
    t = jnp.arctan(x) / (jnp.pi / 2)
    return t.squeeze()


def _map_ainf(t: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point t in [-1, 1] to x in [a, inf]."""
    # The distance from the finite endpoint is (1+t)/(1-t), and writing it that way
    # keeps every bit `t` has: `1+t` is exact for `t` near -1. The algebraically equal
    # `a - 1 + 2/(1-t)` instead forms it as the difference of two numbers near 1 and
    # keeps only `d/eps` of it, so the nodes closest to the endpoint, ie the ones that
    # decide the answer when the integrand is singular there, come out wrong by a factor
    # of two at the last of them.
    x = a + (1 + t) / (1 - t)
    w = 2 / (1 - t) ** 2
    return x.squeeze(), w.squeeze()


def _map_ainf_inv(x: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point x in [a, inf] to t in [-1, 1]."""
    t = (a - x + 1) / (a - x - 1)
    return t.squeeze()


def _map_ninfb(t: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point t in [-1, 1] to x in [-inf, b]."""
    # Distance from the finite endpoint as (1-t)/(1+t) rather than 1 - 2/(t+1); see
    # ``_map_ainf``, which this mirrors.
    x = b - (1 - t) / (1 + t)
    w = 2 / (t + 1) ** 2
    return x.squeeze(), w.squeeze()


def _map_ninfb_inv(x: jax.Array, a: jax.Array, b: jax.Array):
    """Map a point x in [-inf, b] to t in [-1, 1]."""
    t = (x - b + 1) / (b - x + 1)
    return t.squeeze()


# A finite interval stays where it is; the three infinite cases have to be brought into
# [-1, 1] because there is no other way to subdivide them. ``MAPFUNS_REF`` normalizes
# the finite case too, for the one caller that needs a function on the reference
# interval rather than one it can integrate directly: ``tanhsinh_transform`` substitutes
# ``x = tanh(pi/2 sinh u)``, which produces points in [-1, 1] by construction.
MAPFUNS = [_map_identity, _map_ninfb, _map_ainf, _map_ninfinf]
MAPFUNS_INV = [_map_identity_inv, _map_ninfb_inv, _map_ainf_inv, _map_ninfinf_inv]
MAPFUNS_REF = [_map_linear, _map_ninfb, _map_ainf, _map_ninfinf]
MAPFUNS_REF_INV = [_map_linear_inv, _map_ninfb_inv, _map_ainf_inv, _map_ninfinf_inv]

# These are the branches of a `lax.switch`, so all four have to return the same dtypes,
# and they only do so if `t`, `a` and `b` agree. Note in particular that they must not
# be given a *weakly* typed `t`: the four differ in which of `a`/`b` they use, and a
# weak `t` lets each branch settle on whichever of the two is present. `_map_linear`'s
# `c * jnp.ones_like(t)` would follow a strong float32 `a`, while `_map_ninfb`'s
# `2 / (t + 1) ** 2` would stay at the weak default, and the switch would not build.
# This is why the integrand is probed with `jnp.zeros((), xtype)`, not `jnp.array(0.0)`.


def map_interval(
    fun: Callable[..., jax.Array], interval: ArrayLike, *, reference: bool = False
):
    """Map a function over an arbitrary interval [a, b] to one that can be subdivided.

    Transform a function such that integral(fun) on interval is the same as
    integral(fun_t) on interval_t

    Parameters
    ----------
    fun : callable
        Integrand to transform.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals.
    reference : bool
        Whether a finite interval should be normalized to [-1, 1] rather than left
        alone. Only for callers that need the integrand on the reference interval
        specifically; leaving it alone is more accurate near the endpoints. Infinite
        intervals are mapped to [-1, 1] either way.

    Returns
    -------
    fun_t : callable
        Transformed integrand.
    interval_t : float
        New lower and upper limits of integration with possible breakpoints.
    """
    interval = jnp.asarray(interval)
    errorif(
        not jnp.issubdtype(interval.dtype, jnp.floating),
        TypeError,
        "integration limits must be real floating point, got dtype "
        f"{interval.dtype}. Complex limits are not supported: the subdivision has to "
        "order the breakpoints, which complex numbers do not admit.",
    )
    a, b = interval[0], interval[-1]
    # An `xtype` scalar rather than the integer `(-1) ** (a > b)`, so that it cannot
    # participate in promotion downstream.
    sgn = jnp.where(a > b, -1, 1).astype(interval.dtype)
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    # catch breakpoints that are outside the domain, replace with endpoints
    # this creates intervals of 0 length which will be ignored later
    interval = jnp.where(interval < a, a, interval)
    interval = jnp.where(interval > b, b, interval)
    interval = jnp.sort(interval)

    # bit mask to select mapping case
    # 0 : both sides finite
    # 1 : a = -inf, b finite
    # 2 : a finite, b = inf
    # 3 : both infinite
    bitmask = jnp.isinf(a) + 2 * jnp.isinf(b)

    fun_mapped = _MappedFunction(fun, bitmask, sgn, a, b, reference)
    # map original breakpoints to new domain
    inv = MAPFUNS_REF_INV if reference else MAPFUNS_INV
    interval_t: jax.Array = jax.lax.switch(bitmask, inv, interval, a, b)
    # +/-inf gets mapped to +/-1 but numerically evaluates to nan so we replace that.
    interval_t = jnp.where(interval == jnp.inf, 1, interval_t)
    interval_t = jnp.where(interval == -jnp.inf, -1, interval_t)
    return fun_mapped, interval_t


class _MappedFunction(eqx.Module):
    """Function mapped to an interval a fixed rule can be applied over."""

    fun: Callable[..., jax.Array]
    bitmask: jax.Array
    sgn: jax.Array
    a: jax.Array
    b: jax.Array
    reference: bool = eqx.field(static=True, default=False)

    @eqx.filter_jit
    def __call__(self, t: jax.Array, *args):
        mapfuns = MAPFUNS_REF if self.reference else MAPFUNS
        x, w = jax.lax.switch(self.bitmask, mapfuns, t, self.a, self.b)
        return self.sgn * w * self.fun(x, *args)


def tanhsinh_tmax(dtype, order: int | None = None) -> float:
    """Largest ``t`` whose tanh-sinh node is still distinct from the endpoint.

    The tanh-sinh nodes ``x = tanh(pi/2 sinh(t))`` cluster double-exponentially at the
    endpoints, which is what lets the rule handle an endpoint singularity. The cutoff is
    the last node that survives being written down: ``x = 1 - eps`` is two ulps below
    the endpoint, and one step further ``1 - d`` rounds to the endpoint itself, at which
    point an integrand singular there is evaluated at the singularity and returns a
    non-finite value rather than merely a useless one. That bound is set by the
    precision the nodes will be used at, which is why this takes a dtype rather than
    being a constant.

    This is a representability bound, not an accuracy one, but the two coincide: the
    truncation error of the trapezoidal rule in ``t`` falls monotonically as the range
    grows, so the best reachable cutoff is the largest one, and measured optima across
    dtypes and rule orders sit on this bound rather than inside it. A margin costs
    nothing at float64, where truncation is far below eps either way, and a great deal
    at half precision, where it is not.

    The bound assumes the reference interval. Composing with the map onto ``[a, b]``
    needs ``d > eps*|a + b| / |b - a|``, so a sub-interval far from the origin relative
    to its own width loses its outermost nodes to the same rounding regardless.

    Given ``order``, the range is additionally cut back until all nodes are unique
    in the given dtype. Reaching the endpoint is worth nothing if the last two nodes
    reach it together: the rule would spend two evaluations on one point and quietly
    lose an order. That constraint is the one place the range depends on how many nodes
    are being spread over it, and it only binds where the mantissa is short enough that
    the clustering is already marginal.

    Warns when the precision is coarse enough that the clustering has essentially been
    lost. Computed in float64 on the host; the result is a compile time constant.
    """
    eps = float(jnp.finfo(dtype).eps)
    closest = eps
    if closest > 1e-4:
        warnings.warn(
            f"tanh-sinh quadrature in {jnp.dtype(dtype).name} can place a node no "
            f"closer than {closest:.1e} of the half width from an endpoint (float64 "
            "reaches 2.2e-16), so the double exponential clustering that makes the "
            "method good at endpoint singularities is largely gone. Results are still "
            "valid but no better than a plain trapezoidal rule near the endpoints; use "
            "float32 or better, or use quadgk/quadcc instead.",
            UserWarning,
            stacklevel=2,
        )
    tanhinv = lambda x: 0.5 * np.log((1 + x) / (1 - x))
    sinhinv = lambda x: np.log(x + np.sqrt(x**2 + 1))
    tmax = float(sinhinv(2 / np.pi * tanhinv(1.0 - closest)))
    if order is None or order < 2:
        return tmax

    def nodes_resolve(t):
        """Whether an ``order`` point rule's nodes are all distinct, inside (-1, 1)."""
        nodes = np.tanh(np.pi / 2 * np.sinh(np.linspace(-t, t, order)))
        # `jnp.dtype` resolves bfloat16 to the ml_dtypes scalar numpy understands, so
        # the round trip stays on the host: this runs while a trace may be open, and a
        # jnp cast would be staged into it rather than evaluated.
        cast = nodes.astype(jnp.dtype(dtype)).astype(np.float64)
        return bool(len(np.unique(cast)) == order and np.max(np.abs(cast)) < 1.0)

    # Shrink until they do. float32 and above pass on the first test at every order, so
    # this costs nothing where the clustering is healthy. It binds only on the short
    # mantissas, and more as the order rises and the nodes crowd: bfloat16 gives up 2%
    # of the range at order 61 and 18% at order 121.
    while tmax > 0.5 and not nodes_resolve(tmax):
        tmax *= 0.98
    return tmax


def tanhsinh_transform(fun, interval):
    """Transform a function by mapping with tanh-sinh.

    Transform a function such that integral(fun) on interval is the same as
    integral(fun_t) on interval_t

    Parameters
    ----------
    fun : callable
        Integrand to transform.
    interval : array-like
        Lower and upper limits of integration. Use np.inf to denote infinite intervals.

    Returns
    -------
    fun_t : callable
        Transformed integrand.
    interval_t : float
        New lower and upper limits.
    """
    errorif(
        len(interval) != 2,
        NotImplementedError,
        "tanh-sinh transformation with breakpoints not supported",
    )
    xtype = jnp.asarray(interval).dtype
    # map a, b -> [-1, 1]. The substitution below produces points in [-1, 1] whatever
    # the original limits were, so this one caller needs the reference interval rather
    # than the interval it was handed.
    fun, interval = map_interval(fun, interval, reference=True)

    func = _TanhSinhTransformedFunction(fun)

    # we generally only need to integrate ~[-3, 3] or ~[-4, 4]
    # we don't want to include the endpoint that maps to x==1 to avoid
    # possible singularities, so we find the largest t s.t. x(t) < 1
    # and use that as our interval. How large that is depends on the precision the
    # abscissae are carried at, so it follows `interval`.
    tmax = tanhsinh_tmax(xtype)
    interval_t = jnp.array([-tmax, tmax], dtype=xtype)
    return func, interval_t


# map [-1, 1] to [-inf, inf], but with mass concentrated near 0
tanhsinh_x = lambda t: jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
tanhsinh_w = lambda t: (
    jnp.pi / 2 * jnp.cosh(t) / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2
)


class _TanhSinhTransformedFunction(eqx.Module):
    """Function transformed by tanh-sinh transformation."""

    fun: Callable[..., jax.Array]

    @eqx.filter_jit
    def __call__(self, t, *args):
        x = tanhsinh_x(t)
        w = tanhsinh_w(t)
        return self.fun(x, *args) * w


messages = {
    # NORMAL_EXIT
    0: "Algorithm terminated normally, desired tolerances assumed reached",
    # MAX_NINTER
    1: (
        "Maximum number of subdivisions allowed has been achieved. One can allow more "
        + "subdivisions by increasing the value of max_ninter. However,if this yields "
        + "no improvement it is advised to analyze the integrand in order to determine "
        + "the integration difficulties. If the position of a local difficulty can be "
        + "determined (e.g. singularity, discontinuity within the interval) one will "
        + "probably gain from splitting up the interval at this point and calling the "
        + "integrator on the sub-ranges. If possible, an appropriate special-purpose "
        + "integrator should be used, which is designed for handling the type of "
        + "difficulty involved."
    ),
    # ROUNDOFF
    2: (
        "The occurrence of roundoff error is detected, which prevents the requested "
        + "tolerance from being achieved. The error may be under-estimated."
    ),
    # BAD_INTEGRAND
    3: (
        "Extremely bad integrand behavior occurs at some points of the integration "
        + "interval."
    ),
    # NO_CONVERGE
    4: (
        "The algorithm does not converge. Roundoff error is detected in the "
        + "extrapolation table. It is assumed that the requested tolerance cannot be "
        + "achieved, and that the returned result is the best which can be obtained."
    ),
    # DIVERGENT
    5: "The integral is probably divergent, or slowly convergent.",
}


def _decode_status(status):
    if status == 0:
        msg = messages[0]
    else:
        status = f"{status:06b}"[::-1]
        msg = ""
        for s, m in zip(status, messages.values()):
            if int(s):
                msg += m + "\n\n"
    return msg


STATUS = {i: _decode_status(i) for i in range(int(2**6))}


def wrap_func(
    fun: Callable[..., jax.Array],
    args: tuple[Any, ...],
    xtype,
    batch_size: int | None = None,
):
    """Vectorize, jit, and mask out inf/nan.

    ``xtype`` is the dtype the integrand will be called at, and the integrand is probed
    at that dtype rather than at a weakly typed default. See the note on ``MAPFUNS``.

    ``batch_size`` bounds how many points the returned function evaluates at once. The
    default evaluates however many it is given.
    """
    f = jax.eval_shape(fun, jnp.zeros((), xtype), *args)
    # need to make sure we get the correct shape for array valued integrands
    outsig = "(" + ",".join("n" + str(i) for i in range(len(f.shape))) + ")"

    return _WrappedFunction(fun, args, outsig, batch_size)


class _WrappedFunction(eqx.Module):
    """Wraps a function in jit/vectorize and masks out inf/nans.

    Evaluates at most ``batch_size`` points at once, scanning over the batches. The
    number of points is fixed at trace time here, so the batches are cut to fit rather
    than the points being padded up to a whole number of batches: the leftovers are
    evaluated together in one smaller batch. That costs one extra tracing of the
    integrand and never an extra evaluation of it, which is the right way round when an
    evaluation is the expensive part, which is the case ``batch_size`` exists for.

    Callers whose point count is only known at run time cannot do this, and pad instead;
    see ``_level_sum`` in the Romberg solver.
    """

    fun: Callable[..., jax.Array]
    args: tuple[Any, ...]
    outsig: str
    batch_size: int | None = None

    def _vectorize(self, x: jax.Array) -> jax.Array:
        return jnp.vectorize(
            self.fun,
            excluded=tuple(range(1, len(self.args) + 1)),
            signature="()->" + self.outsig,
        )(x, *self.args)

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        b = self.batch_size
        # A scalar abscissa has nothing to batch: Romberg calls the integrand one point
        # at a time, and every caller probes it with a scalar under eval_shape.
        if b is None or x.ndim == 0 or x.shape[0] <= b:
            f: jax.Array = self._vectorize(x)
        else:
            n = x.shape[0]
            nfull = n // b
            full = jax.lax.map(self._vectorize, x[: nfull * b].reshape(nfull, b))
            parts = [full.reshape(-1, *full.shape[2:])]
            if n % b:
                parts.append(self._vectorize(x[nfull * b :]))
            f = jnp.concatenate(parts)
        return jnp.where(jnp.isfinite(f), f, 0.0)


def check_size(size: int | None, name: str = "batch_size") -> None:
    """Raise if ``size`` is neither ``None`` nor a positive integer.

    Shared by the options that set how many evaluations are grouped together:
    ``batch_size`` on the quadrature routines and ``chunk_size`` on the adjoints.
    """
    errorif(
        size is not None and (not isinstance(size, (int, np.integer)) or size < 1),
        ValueError,
        f"{name} must be None or a positive integer, got {size}",
    )


class QuadratureInfo(NamedTuple):
    """Information about quadrature.

    Parameters
    ----------
    err : float
        Estimate of the error in the quadrature result.
    neval : int
        Number of evaluations of the integrand.
    status : int
        Flag indicating reason for termination. status of 0 means normal termination,
        any other value indicates a possible error. A human readable message can be
        obtained by ``print(quadax.STATUS[status])``
    info : dict or None
        Other information returned by the algorithm. See specific algorithm for
        details. Only present if ``full_output`` is True.
    """

    err: float | jax.Array
    neval: int | jax.Array
    status: int | jax.Array
    info: Any


def bounded_while_loop(condfun, bodyfun, init_val, bound):
    """While loop for bounded number of iterations, implemented using cond and scan.

    Implemented with ``scan`` rather than ``lax.while_loop`` so that it can be reverse
    mode differentiated.

    Each iteration is gated twice. The outer gate is
    ``unvmap_any(condfun(state))``, a scalar, so it stays a real branch under ``vmap``
    and the loop stops doing work once *every* batch element has converged. Without it
    the raw predicate is per-element, the branch degrades to a ``select``, and the body
    runs for all ``bound`` iterations however few elements still need it. The inner gate
    is the raw predicate, which unbatched is a second cheap branch and batched is the
    per-element select that leaves already-converged elements untouched. Results are
    unchanged either way.
    """
    # could do some fancy stuff with checkpointing here like in equinox but the loops
    # in quadax usually only do ~100 iterations max so probably not worth it.

    def scanfun(state, *args):
        keep = condfun(state)

        def stepfun(state):
            # Inner branch on the raw predicate. Unbatched this is a second real branch
            # and costs almost nothing; batched it becomes a select, which is the
            # per-element masking that keeps already-converged elements untouched.
            return jax.lax.cond(keep, bodyfun, lambda x: x, state)

        return jax.lax.cond(unvmap_any(keep), stepfun, lambda x: x, state), None

    return jax.lax.scan(scanfun, init_val, None, bound)[0]


def _pnorm(x: jax.Array, p: int | float | jax.Array) -> jax.Array:
    return jnp.linalg.norm(x.flatten(), ord=p)


def wrap_jit(*args, **kwargs):
    """Wrap a function with jit with optional extra args.

    This is a helper to ensure docstrings and type hints are correctly propagated
    to the wrapped function, bc vscode seems to have issues with regular jitted funcs.
    """

    def wrapper(fun):
        foo = jax.jit(fun, *args, **kwargs)
        foo = functools.wraps(fun)(foo)
        return foo

    return wrapper
