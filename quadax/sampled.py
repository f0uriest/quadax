"""Quadrature of functions using known sample values."""

import functools
from typing import Callable, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def _tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


@functools.partial(jax.jit, static_argnames="axis")
def trapezoid(
    y: ArrayLike, *, x: Union[None, ArrayLike] = None, dx: float = 1.0, axis: int = -1
) -> jax.Array:
    r"""
    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapezoid : float or ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.

    Examples
    --------
    Use the trapezoidal rule on evenly spaced points:

    >>> import jax.numpy as jnp
    >>> from quadax import trapezoid
    >>> trapezoid([1, 2, 3])
    4.0

    The spacing between sample points can be selected by either the
    ``x`` or ``dx`` arguments:

    >>> trapezoid([1, 2, 3], x=[4, 6, 8])
    8.0
    >>> trapezoid([1, 2, 3], dx=2)
    8.0

    Using a decreasing ``x`` corresponds to integrating in reverse:

    >>> trapezoid([1, 2, 3], x=[8, 6, 4])
    -8.0

    More generally ``x`` is used to integrate along a parametric curve. We can
    estimate the integral :math:`\int_0^1 x^2 = 1/3` using:

    >>> x = jnp.linspace(0, 1, num=50)
    >>> y = x**2
    >>> trapezoid(y, x)
    0.33340274885464394

    Or estimate the area of a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = jnp.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> trapezoid(jnp.cos(theta), x=jnp.sin(theta))
    3.141571941375841

    ``trapezoid`` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = jnp.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> trapezoid(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> trapezoid(a, axis=1)
    array([2.,  8.])
    """
    if x is None:
        y_arr, dx_array = jnp.asarray(y), jnp.asarray(dx)
    else:
        y_arr, x_arr = jnp.asarray(y), jnp.asarray(x)
        if x_arr.ndim == 1:
            dx_array = jnp.diff(x_arr)
        else:
            dx_array = jnp.moveaxis(jnp.diff(x_arr, axis=axis), axis, -1)
    y_arr = jnp.moveaxis(y_arr, axis, -1)
    return 0.5 * (dx_array * (y_arr[..., 1:] + y_arr[..., :-1])).sum(-1)


@functools.partial(jax.jit, static_argnames="axis")
def cumulative_trapezoid(
    y: ArrayLike,
    *,
    x: Union[None, ArrayLike] = None,
    dx: float = 1.0,
    axis: int = -1,
    initial: Union[ArrayLike, None] = None,
) -> jax.Array:
    """Cumulatively integrate y(x) using the composite trapezoidal rule.

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        Typically this value should be 0. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    Examples
    --------
    >>> from quadax import cumulative_trapezoid
    >>> import jax.numpy as jnp
    >>> import matplotlib.pyplot as plt

    >>> x = jnp.linspace(-2, 2, num=20)
    >>> y = x
    >>> y_int = cumulative_trapezoid(y, x, initial=0)
    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
    >>> plt.show()

    """
    y = jnp.asarray(y)
    if x is None:
        d = dx
    else:
        x = jnp.asarray(x)
        if x.ndim == 1:
            d = jnp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the " "same as y.")
        else:
            d = jnp.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError(
                "If given, length of x along axis must be the " "same as y."
            )

    nd = len(y.shape)
    slice1 = _tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = _tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = jnp.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if not jnp.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = jnp.concatenate(
            [jnp.full(shape, initial, dtype=res.dtype), res], axis=axis
        )

    return res


def _basic_simpson(
    y: jax.Array, start: int, stop: int, x: Union[jax.Array, None], dx: float, axis: int
) -> jax.Array:
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),) * nd
    slice0 = _tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = _tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
    slice2 = _tupleset(slice_all, axis, slice(start + 2, stop + 2, step))

    if x is None:  # Even-spaced Simpson's rule.
        result = jnp.sum(y[slice0] + 4.0 * y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = jnp.diff(x, axis=axis)
        sl0 = _tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = _tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        h0 = h[sl0].astype(float)
        h1 = h[sl1].astype(float)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = jnp.where(h1 == 0, 0, h0 / h1)
        tmp = (
            hsum
            / 6.0
            * (
                y[slice0] * (2.0 - jnp.where(h0divh1 == 0, 0, 1 / h0divh1))
                + y[slice1] * (hsum * jnp.where(hprod == 0, 0, hsum / hprod))
                + y[slice2] * (2.0 - h0divh1)
            )
        )
        result = jnp.sum(tmp, axis=axis)
    return result


@functools.partial(jax.jit, static_argnames="axis")
def simpson(
    y: ArrayLike, *, x: Union[None, ArrayLike] = None, dx: float = 1.0, axis: int = -1
) -> jax.Array:
    """Integrate y(x) from samples using the composite Simpson's rule.

    If x is None, spacing of dx is assumed.

    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `x`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.

    Returns
    -------
    float
        The estimated integral computed with the composite Simpson's rule.

    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less. If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.

    """
    y = jnp.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    returnshape = 0
    if x is not None:
        x = jnp.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the " "same as y.")
        if x.shape[axis] != N:
            raise ValueError(
                "If given, length of x along axis must be the " "same as y."
            )

    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd

        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = _tupleset(slice_all, axis, -1)
            slice2 = _tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])

        else:
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N - 3, x, dx, axis)

            slice1 = _tupleset(slice_all, axis, -1)
            slice2 = _tupleset(slice_all, axis, -2)
            slice3 = _tupleset(slice_all, axis, -3)

            h = jnp.asarray([dx, dx])
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = _tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = _tupleset(slice_all, axis, slice(-1, None, 1))

                diffs = jnp.diff(x, axis=axis)
                h = [
                    jnp.squeeze(diffs[hm2], axis=axis),
                    jnp.squeeze(diffs[hm1], axis=axis),
                ]

            # This is the correction for the last interval according to
            # Cartwright.
            # However, I used the equations given at
            # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
            # A footnote on Wikipedia says:
            # Cartwright 2017, Equation 8. The equation in Cartwright is
            # calculating the first interval whereas the equations in the
            # Wikipedia article are adjusting for the last integral. If the
            # proper algebraic substitutions are made, the equation results in
            # the values shown.
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = jnp.where(den == 0, 0, num / den)

            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = jnp.where(den == 0, 0, num / den)

            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = jnp.where(den == 0, 0, num / den)

            result += alpha * y[slice1] + beta * y[slice2] - eta * y[slice3]

        result = result + val
    else:
        result = _basic_simpson(y, 0, N - 2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result


def cumulative_simpson(
    y: ArrayLike,
    *,
    x: Union[None, ArrayLike] = None,
    dx: float = 1.0,
    axis: int = -1,
    initial: Union[ArrayLike, None] = None,
) -> jax.Array:
    r"""Cumulatively integrate y(x) using the composite Simpson's 1/3 rule.

    The integral of the samples at every point is calculated by assuming a
    quadratic relationship between each point and the two adjacent points.

    Parameters
    ----------
    y : array_like
        Values to integrate. Requires at least one point along `axis`. If two or fewer
        points are provided along `axis`, Simpson's integration is not possible and the
        result is calculated with `cumulative_trapezoid`.
    x : array_like, optional
        The coordinate to integrate along. Must have the same shape as `y` or
        must be 1D with the same length as `y` along `axis`. `x` must also be
        strictly increasing along `axis`.
        If `x` is None (default), integration is performed using spacing `dx`
        between consecutive elements in `y`.
    dx : scalar or array_like, optional
        Spacing between elements of `y`. Only used if `x` is None. Can either
        be a float, or an array with the same shape as `y`, but of length one along
        `axis`. Default is 1.0.
    axis : int, optional
        Specifies the axis to integrate along. Default is -1 (last axis).
    initial : scalar or array_like, optional
        If given, insert this value at the beginning of the returned result,
        and add it to the rest of the result. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration. Can either be a float, or an array with
        the same shape as `y`, but of length one along `axis`.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less. If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.

    """
    y = _ensure_float_array(y)

    # validate `axis` and standardize to work along the last axis
    original_y = y
    original_shape = y.shape
    try:
        y = jnp.swapaxes(y, axis, -1)
    except IndexError as e:
        message = f"`axis={axis}` is not valid for `y` with `y.ndim={y.ndim}`."
        raise ValueError(message) from e
    if y.shape[-1] < 3:
        res = cumulative_trapezoid(original_y, x, dx=dx, axis=axis, initial=None)
        res = jnp.swapaxes(res, axis, -1)

    elif x is not None:
        x = _ensure_float_array(x)
        message = (
            "If given, shape of `x` must be the same as `y` or 1-D with "
            "the same length as `y` along `axis`."
        )
        if not (
            x.shape == original_shape
            or (x.ndim == 1 and len(x) == original_shape[axis])
        ):
            raise ValueError(message)

        x = jnp.broadcast_to(x, y.shape) if x.ndim == 1 else jnp.swapaxes(x, axis, -1)
        dx = jnp.diff(x, axis=-1)
        dx = eqx.error_if(dx, dx <= 0, "Input x must be strictly increasing.")
        res = _cumulatively_sum_simpson_integrals(
            y, dx, _cumulative_simpson_unequal_intervals
        )

    else:
        dx = _ensure_float_array(dx)
        final_dx_shape = _tupleset(original_shape, axis, original_shape[axis] - 1)
        alt_input_dx_shape = _tupleset(original_shape, axis, 1)
        message = (
            "If provided, `dx` must either be a scalar or have the same "
            "shape as `y` but with only 1 point along `axis`."
        )
        if not (dx.ndim == 0 or dx.shape == alt_input_dx_shape):
            raise ValueError(message)
        dx = jnp.broadcast_to(dx, final_dx_shape)
        dx = jnp.swapaxes(dx, axis, -1)
        res = _cumulatively_sum_simpson_integrals(
            y, dx, _cumulative_simpson_equal_intervals
        )

    if initial is not None:
        initial = _ensure_float_array(initial)
        alt_initial_input_shape = _tupleset(original_shape, axis, 1)
        message = (
            "If provided, `initial` must either be a scalar or have the "
            "same shape as `y` but with only 1 point along `axis`."
        )
        if not (initial.ndim == 0 or initial.shape == alt_initial_input_shape):
            raise ValueError(message)
        initial = jnp.broadcast_to(initial, alt_initial_input_shape)
        initial = jnp.swapaxes(initial, axis, -1)

        res += initial
        res = jnp.concatenate((initial, res), axis=-1)

    res = jnp.swapaxes(res, -1, axis)
    return res


def _cumulatively_sum_simpson_integrals(
    y: jax.Array,
    dx: jax.Array,
    integration_func: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Calculate cumulative sum of Simpson integrals.

    Takes as input the integration function to be used.
    The integration_func is assumed to return the cumulative sum using
    composite Simpson's rule. Assumes the axis of summation is -1.
    """
    sub_integrals_h1 = integration_func(y, dx)
    sub_integrals_h2 = integration_func(y[..., ::-1], dx[..., ::-1])[..., ::-1]

    shape = list(sub_integrals_h1.shape)
    shape[-1] += 1
    sub_integrals = jnp.empty(shape)
    sub_integrals = sub_integrals.at[..., :-1:2].set(sub_integrals_h1[..., ::2])
    sub_integrals = sub_integrals.at[..., 1::2].set(sub_integrals_h2[..., ::2])
    # Integral over last subinterval can only be calculated from
    # formula for h2
    sub_integrals = sub_integrals.at[..., -1].set(sub_integrals_h2[..., -1])
    res = jnp.cumsum(sub_integrals, axis=-1)
    return res


def _cumulative_simpson_equal_intervals(y: jax.Array, dx: jax.Array) -> jax.Array:
    """Calculate the Simpson integrals assuming equal interval widths."""
    d = dx[..., :-1]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]

    return d / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4)


def _cumulative_simpson_unequal_intervals(y: jax.Array, dx: jax.Array) -> jax.Array:
    """Calculate the Simpson integrals assuming unequal interval widths."""
    x21 = dx[..., :-1]
    x32 = dx[..., 1:]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]

    x31 = x21 + x32
    x21_x31 = x21 / x31
    x21_x32 = x21 / x32
    x21x21_x31x32 = x21_x31 * x21_x32

    coeff1 = 3 - x21_x31
    coeff2 = 3 + x21x21_x31x32 + x21_x31
    coeff3 = -x21x21_x31x32

    return x21 / 6 * (coeff1 * f1 + coeff2 * f2 + coeff3 * f3)


def _ensure_float_array(arr):
    arr = jnp.asarray(arr)
    if jnp.issubdtype(arr.dtype, jnp.integer):
        arr = arr.astype(float, copy=False)
    return arr
