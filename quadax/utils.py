"""Utility functions for parsing inputs, mapping coordinates etc."""

from typing import Any, NamedTuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp


def errorif(cond, err=ValueError, msg=""):
    """Raise an error if condition is met.

    Similar to assert but allows wider range of Error types, rather than
    just AssertionError.
    """
    if cond:
        raise err(msg)


def _map_linear(t, a, b):
    """Map a point t in [-1, 1] to x in [a, b]."""
    c = (b - a) / 2
    d = (b + a) / 2
    x = d + c * t
    w = c * jnp.ones_like(t)
    return x.squeeze(), w.squeeze()


def _map_linear_inv(x, a, b):
    """Map a point x in [a, b] to t in [-1, 1]."""
    c = (b - a) / 2
    d = (b + a) / 2
    t = (x - d) / c
    return t.squeeze()


def _map_ninfinf(t, a, b):
    """Map a point t in [-1, 1] to x in [-inf, inf]."""
    x = t / jnp.sqrt(1 - t**2)
    w = 1 / jnp.sqrt(1 - t**2) ** 3
    return x.squeeze(), w.squeeze()


def _map_ninfinf_inv(x, a, b):
    """Map a point x in [-inf, inf] to t in [-1, 1]."""
    t = x / jnp.sqrt(x**2 + 1)
    return t.squeeze()


def _map_ainf(t, a, b):
    """Map a point t in [-1, 1] to x in [a, inf]."""
    x = a - 1 + 2 / (1 - t)
    w = 2 / (1 - t) ** 2
    return x.squeeze(), w.squeeze()


def _map_ainf_inv(x, a, b):
    """Map a point x in [a, inf] to t in [-1, 1]."""
    t = (a - x + 1) / (a - x - 1)
    return t.squeeze()


def _map_ninfb(t, a, b):
    """Map a point t in [-1, 1] to x in [-inf, b]."""
    x = b + 1 - 2 / (t + 1)
    w = 2 / (t + 1) ** 2
    return x.squeeze(), w.squeeze()


def _map_ninfb_inv(x, a, b):
    """Map a point x in [-inf, b] to t in [-1, 1]."""
    t = (x - b + 1) / (b - x + 1)
    return t.squeeze()


MAPFUNS = [_map_linear, _map_ninfb, _map_ainf, _map_ninfinf]
MAPFUNS_INV = [_map_linear_inv, _map_ninfb_inv, _map_ainf_inv, _map_ninfinf_inv]


def map_interval(fun, interval):
    """Map a function over an arbitrary interval [a, b] to the interval [-1, 1].

    Transform a function such that integral(fun) on interval is the same as
    integral(fun_t) on interval_t

    Parameters
    ----------
    fun : callable
        Integrand to transform.
    interval : array-like
        Lower and upper limits of integration with possible breakpoints. Use np.inf to
        denote infinite intervals.

    Returns
    -------
    fun_t : callable
        Transformed integrand.
    interval_t : float
        New lower and upper limits of integration with possible breakpoints.
    """
    interval = jnp.asarray(interval)
    a, b = interval[0], interval[-1]
    sgn = (-1) ** (a > b)
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

    fun_mapped = _MappedFunction(fun, bitmask, sgn, a, b)
    # map original breakpoints to new domain
    interval_t = jax.lax.switch(bitmask, MAPFUNS_INV, interval, a, b)
    # +/-inf gets mapped to +/-1 but numerically evaluates to nan so we replace that.
    interval_t = jnp.where(interval == jnp.inf, 1, interval_t)
    interval_t = jnp.where(interval == -jnp.inf, -1, interval_t)
    return fun_mapped, interval_t


class _MappedFunction(eqx.Module):
    """Function mapped to unit interval [-1,1]."""

    fun: callable
    bitmask: int
    sgn: int
    a: float
    b: float

    @eqx.filter_jit
    def __call__(self, t, *args):
        x, w = jax.lax.switch(self.bitmask, MAPFUNS, t, self.a, self.b)
        return self.sgn * w * self.fun(x, *args)


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
    # map a, b -> [-1, 1]
    fun, interval = map_interval(fun, interval)

    func = _TanhSinhTransformedFunction(fun)

    # we generally only need to integrate ~[-3, 3] or ~[-4, 4]
    # we don't want to include the endpoint that maps to x==1 to avoid
    # possible singularities, so we find the largest t s.t. x(t) < 1
    # and use that as our interval
    def get_tmax(xmax):
        """Inverse of tanh-sinh transform."""
        tanhinv = lambda x: 1 / 2 * jnp.log((1 + x) / (1 - x))
        sinhinv = lambda x: jnp.log(x + jnp.sqrt(x**2 + 1))
        return sinhinv(2 / jnp.pi * tanhinv(xmax))

    # inverse of tanh-sinh transformation for x = 1-eps
    tmax = get_tmax(jnp.array(1.0) - 10 * jnp.finfo(jnp.array(1.0)).eps)
    interval_t = jnp.array([-tmax, tmax])
    return func, interval_t


# map [-1, 1] to [-inf, inf], but with mass concentrated near 0
tanhsinh_x = lambda t: jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
tanhsinh_w = (
    lambda t: jnp.pi / 2 * jnp.cosh(t) / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2
)


class _TanhSinhTransformedFunction(eqx.Module):
    """Function transformed by tanh-sinh transformation."""

    fun: callable

    @eqx.filter_jit
    def __call__(self, t, *args):
        x = tanhsinh_x(t)
        w = tanhsinh_w(t)
        return self.fun(x, *args) * w


messages = {
    0: "Algorithm terminated normally, desired tolerances assumed reached",
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
    2: (
        "The occurrence of roundoff error is detected, which prevents the requested "
        + "tolerance from being achieved. The error may be under-estimated."
    ),
    3: (
        "Extremely bad integrand behavior occurs at some points of the integration "
        + "interval."
    ),
    4: (
        "The algorithm does not converge. Roundoff error is detected in the "
        + "extrapolation table. It is assumed that the requested tolerance cannot be "
        + "achieved, and that the returned result is the best which can be obtained."
    ),
    5: "The integral is probably divergent, or slowly convergent.",
}


def _decode_status(status):
    if status == 0:
        msg = messages[0]
    else:
        status = "{:05b}".format(status)[::-1]
        msg = ""
        for s, m in zip(status, messages.values()):
            if int(s):
                msg += m + "\n\n"
    return msg


STATUS = {i: _decode_status(i) for i in range(int(2**5))}


def wrap_func(fun, args):
    """Vectorize, jit, and mask out inf/nan."""
    f = jax.eval_shape(fun, jnp.array(0.0), *args)
    # need to make sure we get the correct shape for array valued integrands
    outsig = "(" + ",".join("n" + str(i) for i in range(len(f.shape))) + ")"

    return _WrappedFunction(fun, args, outsig)


class _WrappedFunction(eqx.Module):
    """Wraps a function in jit/vectorize and masks out inf/nans."""

    fun: callable
    args: Any
    outsig: str

    @eqx.filter_jit
    def __call__(self, x):
        f = jnp.vectorize(
            self.fun,
            excluded=tuple(range(1, len(self.args) + 1)),
            signature="()->" + self.outsig,
        )(x, *self.args)
        return jnp.where(jnp.isfinite(f), f, 0.0)


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

    err: float
    neval: int
    status: int
    info: Union[dict, None]


def bounded_while_loop(condfun, bodyfun, init_val, bound):
    """While loop for bounded number of iterations, implemented using cond and scan."""
    # could do some fancy stuff with checkpointing here like in equinox but the loops
    # in quadax usually only do ~100 iterations max so probably not worth it.

    def scanfun(state, *args):
        return jax.lax.cond(condfun(state), bodyfun, lambda x: x, state), None

    return jax.lax.scan(scanfun, init_val, None, bound)[0]


def setdefault(val, default, cond=None):
    """Return val if condition is met, otherwise default.

    If cond is None, then it checks if val is not None, returning val
    or default accordingly.
    """
    return val if cond or (cond is None and val is not None) else default
