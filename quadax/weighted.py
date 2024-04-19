"""Weighted quadrature."""

import functools
from collections.abc import Callable
from typing import Any, Union

import jax
import jax.numpy as jnp

from .fixed_order import NestedRule, _dot
from .quad_weights import _chebmom_alg_plus, _chebmom_alg_log_plus
from .utils import wrap_func


class WeightedRule(NestedRule):
    """Base class for weighted quadrature rules."""

    def apply_weighted_rule(self, x1: float, x2: float) -> bool:
        """Check if weighted rule should be applied.

        Determines if the weighted quadrature rule should be applied in the interval
        defined by [x1, x2].
        """
        pass


class AlgLogWeightRule(WeightedRule):
    r"""Quadrature for integrands with algebraic-logarithmic weight functions.

    Weight functions are of the form

    .. math ::

        w(x) = (x - a)**\gamma  \log^i(x - a)
        
    or
        
    .. math ::
        
        w(x) = (b - x)**\gamma \log^i(b - x)

    where ..math::`\gamma > -1`, ..math::`i = \mathrm{0 or 1}`, and ...math::`a, b`
    are the end points of the integration region [a, b].

    Parameters
    ----------
    singularpoint: float
        The singular point of the weight function.
    algpower: float > -1
        The power of the algebraic term.
    logpower: bool
        Determines if the logarithmic term is included.
    order: float, deafult 32
        Number of points used for fixed-order quadrature rules.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    """

    def __init__(
        self,
        singularpoint: float,
        algpower: float,
        logpower: bool,
        order: int = 32,
        norm: Union[Callable, int] = jnp.inf,
    ):
        self.singularpoint = singularpoint
        self.algpower = algpower
        self.logpower = logpower
        self.norm = (
            norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
        )
        
        if self.algpower <= -1:
            raise NotImplementedError(
                f"Power of algebraic term must be > -1. It is {self.algpower}"
            )

        # compute integration weights
        def _modcc_weights(N):
            """Compute modified Clenshaw-Curtis nodes and weights for order N.

            N must be even.
            """
            d_alg = _chebmom_alg_plus(N + 1, self.algpower)[::2]
            d_alg_log = _chebmom_alg_log_plus(N + 1, self.algpower)[::2]
            k = jnp.arange(N // 2 + 1)
            n = jnp.arange(N // 2 + 1)
            D = 2 / N * jnp.cos(k[:, None] * n[None, :] * jnp.pi / (N // 2))
            D = jnp.where((n == 0) | (n == N // 2), D * 1 / 2, D)
            t = jnp.arange(0, 1 + N // 2) * jnp.pi / N
            x = jnp.cos(t)

            algweights = D.T @ d_alg
            alglogweights = D.T @ d_alg_log

            return x, algweights, alglogweights

        order = 2 * (order // 2)  # make sure order is even
        self.xh, self.walgh, self.walglogh = _modcc_weights(order)
        _, walgl, walglogl = _modcc_weights(order // 2)
        self.walgl = jnp.zeros_like(self.walgh).at[::2].set(walgl)
        self.walglogl = jnp.zeros_like(self.walglogh).at[::2].set(walgl)

    def integrate(self, fun, a, b, args=()):
        r"""Algebraic-logarithmic weighted quadrature.

        Uses a modified Clenshaw-Curtis fixed order rule to compute the integral
        from a to b of f(x) * w(x) where

        .. math ::

        w(x) = (x - a)**\gamma  \log^i(x - a)
        
        or
        
        .. math ::
        
        w(x) = (b - x)**\gamma \log^i(b - x)

        where ..math::`\gamma > -1`, and ..math::`i = \mathrm{0 or 1}`.

        Parameters
        ----------
        fun : callable
            Function to integrate, should have a signature of the form
            ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
        a, b : float
            Lower and upper limits of integration. Must be finite.
        args : tuple, optional
            Extra arguments passed to fun.

        Returns
        -------
        y : float, Array
            Estimate of the integral of fun from a to b
        err : float
            Estimate of the absolute error in y from nested rule.
        y_abs : float, Array
            Estimate of the integral of abs(fun) from a to b
        y_mmn : float, Array
            Estimate of the integral of abs(fun - <fun>) from a to b, where <fun>
            is the mean value of fun over the interval.
        """
        halflength = (b - a) / 2

        self.wh = (
            self.walgh
            * halflength**self.algpower
            * (
                jnp.log(b - a)
                + (self.walglogh if self.logpower else 1)
            )
        )
        self.wl = (
            self.walgl
            * halflength**self.algpower
            * (
                jnp.log(b - a)
                + (self.walglogl if self.logpower else 1)
            )
        )

        return super.integrate(fun, a, b, args)

    def apply_weighted_rule(self, x1: float, x2: float) -> bool:
        """Check if weighted rule should be applied.

        Determines if the weighted quadrature rule should be applied in the interval
        defined by [x1, x2].
        
        Parameters
        ----------
        x1, x2: floats
            Endpoints of integration region.
            
        Returns
        -------
            : bool
        """
        if self.singularpoint == x1 or self.singularpoint == x2:
            return True
        return False

class AlgLogWeight(WeightedRule):
    """Quadrature for integrands with algebraic-logarithmic weight functions.

    Parameters
    ----------
    singular1: float
        First singular point of the function.
    singular2: float
        Second singular point of the function.
    alg1power: float > -1
        The power of the first algebraic term
    alg2power: float > -1
        The power of the second algebraic term
    log1power: bool
        Determines if the first logarithmic term is included.
    log2power: bool
        Determines if the second logarithmic term is included.
    n_order: float, deafult 32
        Number of points used for fixed-order quadrature rules.
    """

    singular1: float
    singular2: float
    alg1power: float
    alg2power: float
    log1power: bool
    log2power: bool
    n: int = 32

    def __init__(self):
        if self.alg1power <= -1:
            raise NotImplementedError(
                f"Power of first algebraic term must be > -1. It is {self.alg1power}"
            )
        if self.alg2power <= -1:
            raise NotImplementedError(
                f"Power of second algebraic term must be > -1. It is {self.alg2power}"
            )
        if self.sing1 >= self.sing2:
            raise NotImplementedError(
                f"""The first singular point {self.sing1} must be less than the
                 second singular point {self.sing2}."""
            )

        # compute integration weights
        self.intweights_sing1 = ccmod_weights(self.n_order, self.alg1power)
        self.intweights_sing2 = ccmod_weights(self.n_order, self.alg2power)

    def evaluate(self, x):
        r"""Evaluate the weight function.

        .. math ::

        w(x) = (x - s_1)^\alpha (s_2 - x)^\beta \log^i(x - s_1) \log^j(s_2 - x)
        where ..math::`\alpha, \beta > -1`, ..math::`i, j = \mathrm{0 or 1}`, and
        ..math::`s_1, s_2` are the singular points of the function.
        """
        log1 = jnp.log(x - self.sing1) if self.log1power else 1
        log2 = jnp.log(self.sing2 - x) if self.log2power else 1

        return (
            (x - self.sing1) ** self.alg1power
            * (self.sing2 - x) ** self.alg2power
            * log1
            * log2
        )

    @functools.partial(
        jax.jit,
        static_argnums=(
            0,
            1,
            5,
        ),
    )
    def fixed_quad_leftsingular(self, fun, a, b, args=(), norm=jnp.inf):
        r"""Algebraic-Logarithmic weighted quadrature with a singular point at a.

        Uses a modified Clenshaw-Curtis fixed order rule to compute the integral
        from a to b of f(x) * w(x) where

        .. math ::

        w(x) = (x - a)**\alpha * (s_2 - x)**\beta * \log^i(x - a) * \log^j(s_2 - x)

        where a < b < s_2, ..math::`\alpha, \beta > -1`, and
        ..math::`i, j = \mathrm{0 or 1}`.

        """
        # modify function with nonsingular parts of the weight function
        fun = lambda x, args: (
            fun(x, *args)
            * (self.sing2 - x) ** self.alg2power
            * (jnp.log(self.sing2 - x) if self.log2power else 1)
        )

        halflength = (b - a) / 2

        self.xh = self.intweights_sing1["xc"]
        self.wh = (
            self.intweights_sing1["walgc"]
            * halflength**self.alg1power
            * (
                jnp.log(b - a)
                + (self.intweights_sing1["walglogc"] if self.log1power else 1)
            )
        )
        self.wl = (
            self.intweights_sing1["walge"]
            * halflength**self.alg1power
            * (
                jnp.log(b - a)
                + (self.intweights_sing1["walgloge"] if self.log1power else 1)
            )
        )

        return _fixed_quadcc_base(fun, a, b, xc, wc, we, args, norm, self.n_order)

    @functools.partial(
        jax.jit,
        static_argnums=(
            0,
            1,
            5,
        ),
    )
    def fixed_quad_rightsingular(self, fun, a, b, args=(), norm=jnp.inf):
        r"""Algebraic-Logarithmic weighted quadrature with a singular point at b.

        Uses a modified Clenshaw-Curtis fixed order rule to compute the integral
        from a to b of f(x) * w(x) where

        .. math ::

        w(x) = (x - s_1)**\alpha * (b - x)**\beta * \log^i(x - s_1) * \log^j(b - x)

        where s_1 < a < b, ..math::`\alpha, \beta > -1`, and
        ..math::`i, j = \mathrm{0 or 1}`.
        """
        # modify function with nonsingular parts of the weight function
        fun = lambda x, args: (
            fun(x, *args)
            * (x - self.sing1) ** self.alg1power
            * (jnp.log(x - self.sing1) if self.log1power else 1)
        )

        halflength = (b - a) / 2

        xc = self.intweights_sing2["xc"]
        wc = (
            self.intweights_sing2["walgc"]
            * halflength**self.alg2power
            * (
                jnp.log(b - a)
                + (self.intweights_sing2["walglogc"] if self.log2power else 1)
            )
        )
        we = (
            self.intweights_sing2["walge"]
            * halflength**self.alg2power
            * (
                jnp.log(b - a)
                + (self.intweights_sing2["walgloge"] if self.log2power else 1)
            )
        )

        return _fixed_quadcc_base(fun, a, b, xc, wc, we, args, norm, self.n_order)

    def quad(self, fun, interval, args=()):
        """Global adaptive weighted quadrature using a modified Clenshaw-Curtis rule.

        etc.
        """
        interval = jnp.asarray(interval)
        if (
            len(interval) == 2
            and interval[0] == self.sing1
            and interval[1] == self.sing2
        ):
            # insert breakpoint if there are two endpoint singularities present
            interval = jnp.insert(interval, 1, (interval[0] + interval[1]) / 2)

        return 0


def _fixed_quadcc_base(
    fun, a, b, intnodes, intweights, interrweights, args=(), norm=jnp.inf, n=32
):
    """Helper function for Clenshaw-Curtis-based fixed-order quadrature."""
    _norm = norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
    vfun = wrap_func(fun, args)

    halflength = (b - a) / 2
    center = (b + a) / 2

    fp = vfun(center + halflength * intnodes)
    fm = vfun(center - halflength * intnodes)

    result_2 = _dot(intweights, (fp + fm)) * halflength
    result_1 = _dot(interrweights, (fp + fm)) * halflength

    integral_abs = _dot(
        intweights, (jnp.abs(fp) + jnp.abs(fm))
    )  # ~integral of abs(fun)
    integral_mmn = _dot(
        interrweights, jnp.abs(fp + fm - result_2 / (b - a))
    )  # ~ integral of abs(fun - mean(fun))

    result = result_2

    uflow = jnp.finfo(fp.dtype).tiny
    eps = jnp.finfo(fp.dtype).eps
    abserr = jnp.abs(result_2 - result_1)
    abserr = jnp.where(
        (integral_mmn != 0.0) & (abserr != 0.0),
        integral_mmn * jnp.minimum(1.0, (200.0 * abserr / integral_mmn) ** 1.5),
        abserr,
    )
    abserr = jnp.where(
        (integral_abs > uflow / (50.0 * eps)),
        jnp.maximum((eps * 50.0) * integral_abs, abserr),
        abserr,
    )
    return result, _norm(abserr), integral_abs, integral_mmn


def _check_singularities(self, a, b):
    if self.sing1 == a and self.sing2 == b:
        raise NotImplementedError(
            """Fixed-order modified Clenshaw-Curtis quadrature can only be called
            with one singular point at a time."""
        )
    if a <= self.sing1:
        raise NotImplementedError(
            f"The first singular point {self.sing1} must be < a = {a}"
        )
    if b >= self.sing2:
        raise NotImplementedError(
            f"The second singular point {self.sing2} must be > b = {b}"
        )


'''
@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8))
def fixed_quadweighted(
    fun, a, b, args=(), norm=jnp.inf, n=32, weight=None, weightargs=None
):
    """Weighted quadrature."""
    # if the singular point x is a <= x < b or a < x <=b then call the correct
    # modified cc quad rule
    # if not, use a regular purpose integrator (gk or cc)
    # what if singularities at both end points? When we write the adaptive version
    # of this function, we can have it introduce a breakpoint (if it doesn't have one)
    # in between (the original) a and b so this function is not called with a weight
    # function that has singularities at the endpoints of its current interval.
    pass


def alglogweightfn(
    x, alg1power=0, alg2power=0, log1present=False, log2present=False, s1=-1, s2=1
):
    r"""Algebraic-Logarithmic weight function.

    .. math ::

        w(x) = (x - s_1)^\alpha (s_2 - x)^\beta \log^i(x - s_1) \log^j(s_2 - x)
    where ..math::`\alpha, \beta > -1`, ..math::`i, j = \mathrm{0 or 1}`, and
    ..math::`s_1, s_2` are the singular points of the function.

    Parameters
    ----------
    x: float
        argument of the function
    alg1power: float > -1, default is 0
        The power of the first algebraic term (x - s1)
    alg2power: float > -1, default is 0
        The power of the second algebraic term (s2 - x)
    log1present: bool, default is False
        Determines if the first logarithmic term log(x - s1) is included.
    log2present: bool, default is False
        Determines if the second logarithmic term log(s2 - x) is included.
    s1: float, default is -1
        First singular point of the function. Must have s1 < s2.
    s2: float, default is 1
        Second singular point of the function. Must have s1 < s2.
    """
    log1 = jnp.log(x - s1) if log1present else 1
    log2 = jnp.log(s2 - x) if log2present else 1

    return (x - s1) ** alg1power * (s2 - x) ** alg2power * log1 * log2


@functools.partial(jax.jit, static_argnums=(0, 3, 5, 6, 7))
def fixed_quadcc_alglogweight(
    fun,
    a,
    b,
    weightargs,
    args=(),
    norm=jnp.inf,
    n=32,
    intweights=None,
):
    r"""Integrate a weighted function from a to b with a modified Clenshaw-Curtis rule.

    .. math ::
        \int_a^b f(x) w(x) dx

    where ..math::`w(x) = (x - c1)^\alpha (c2 - x)^\beta \log^i(x - c1)
    \log^j(c2 - x)`, ..math::`\alpha > -1,\ \beta > -1, i,\ j = \mathrm{0 or 1}`,
    and c1 = a < b < c2 or c1 < a < b =c2, but not c1 = a < b = c2 (only one end-point
    singularity possible).

    If the singularities (c1, c2) are not a or b, then defaults to normal
    Clenshaw-Curtis!

    Integration is performed using a modified Clenshaw-Curtis order n rule with error
    estimated using an embedded n//2 order rule.

    Parameters
    ----------
    fun : callable
        Function to integrate, should have a signature of the form
        ``fun(x, *args)`` -> float, Array. Should be JAX transformable.
    a, b : float
        Lower and upper limits of integration. Must be finite.
    weightargs : dict
        dictionary of arguments that go into the weight function:
        "s1": float, location of first singular point s1
        "s2": float, location of second singular point s2
        "alg1power": float, power of first algebraic weight, must be > -1.
        "alg2power": float, power of second algebraic weight, must be > -1.
        "log1present": bool, whether to include first logarithmic weight
        "log2present": bool, whether to include second logarithmic weight
    args : tuple, optional
        Extra arguments passed to fun.
    norm : int, callable
        Norm to use for measuring error for vector valued integrands. No effect if the
        integrand is scalar valued. If an int, uses p-norm of the given order, otherwise
        should be callable.
    n : {8, 16, 32, 64, 128, 256}
        Order of integration scheme.
    intweights : dict, optional
        Weights used in quadrature scheme. If None then these are calculated based on
        `weightargs`.

    Returns
    -------
    y : float, Array
        Estimate of the integral of fun from a to b
    err : float
        Estimate of the absolute error in y from nested rule.
    y_abs : float, Array
        Estimate of the integral of abs(fun) from a to b
    y_mmn : float, Array
        Estimate of the integral of abs(fun - <fun>) from a to b, where <fun>
        is the mean value of fun over the interval.

    """
    # check validity of singular points and integration range
    jax.debug.callback(_check_alglog_singularities, (a, b, weightargs))
    # check that the algebraic powers in the weight function are > -1
    if weightargs["alg1power"] <= -1:
        raise NotImplementedError(
            f"weightargs['alg1power'] = {weightargs['alg1power']} must be > -1"
        )
    if weightargs["alg2power"] <= -1:
        raise NotImplementedError(
            f"weightargs['alg2power'] = {weightargs['alg2power']} must be > -1"
        )

    def truefun():
        vfun = wrap_func(fun, args)
        f = jax.eval_shape(vfun, jnp.array(0.0))
        z = jnp.zeros(f.shape, f.dtype)
        return z, 0.0, z, z

    def falsefun():
        def basecase(fun, a, b, args, norm, n):
            # no singularities at either endpoint, fall back to normal Clenshaw-Curtis
            fun = lambda x, args: fun(x, *args) * alglogweightfn(x, **weightargs)
            return fixed_quadcc(fun, a, b, args, norm, n)

        def leftsingularcase(fun, a, b, args, norm, n):
            return _fixed_quadcc_alglogweight_leftsingular(
                fun, a, b, weightargs, args, norm, n, intweights
            )

        def rightsingularcase(fun, a, b, args, norm, n):
            return _fixed_quadcc_alglogweight_rightsingular(
                fun, a, b, weightargs, args, norm, n, intweights
            )

        leftsingular = int(weightargs["s1"] == a)
        rightsingular = int(weightargs["s2"] == b) * 2
        index = leftsingular + rightsingular

        return jax.lax.switch(
            index,
            [basecase, rightsingularcase, leftsingularcase],
            (fun, a, b, args, norm, n),
        )

    return jax.lax.cond(a == b, truefun, falsefun)


def _check_alglog_singularities(a, b, weightargs):
    if weightargs["s1"] == a and weightargs["s2"] == b:
        raise NotImplementedError(
            """Fixed-order modified Clenshaw-Curtis quadrature can only be called with
            one singular point at a time."""
        )
    if b >= weightargs["s2"]:
        raise NotImplementedError(
            f"weightargs['s2'] = {weightargs['s2']} must be > b = {b}"
        )
    if a <= weightargs["s1"]:
        raise NotImplementedError(
            f"weightargs['s1'] = {weightargs['s1']} must be < a = {a}"
        )


def _fixed_quadcc_alglogweight_leftsingular(
    fun, a, b, weightargs, args=(), norm=jnp.inf, n=32, intweights=None
):
    r"""Algebraic-Logarithmic weighted quadrature with a singular point at a.

    Uses a modified Clenshaw-Curtis fixed order rule to compute the integral
    from a to b of f(x) * w(x) where

    .. math ::

    w(x) = (x - a)**\alpha * (s_2 - x)**\beta * \log^i(x - a) * \log^j(s_2 - x)

    where a < b < s_2, ..math::`\alpha, \beta > -1`, and
    ..math::`i, j = \mathrm{0 or 1}`.

    """
    # modify function with nonsingular parts of the weight function
    fun = lambda x, args: (
        fun(x, *args)
        * (
            alglogweightfn(
                x,
                alg1power=0,
                alg2power=weightargs["alg2power"],
                log1present=False,
                log2present=weightargs["log2present"],
                s1=0,
                s2=weightargs["s2"],
            )
        )
    )
    if intweights is None:
        # compute weights for integration
        intweights = ccmod_get_weights(n, weightargs["alg1power"])

    halflength = (b - a) / 2

    xc = intweights["xc"]
    wc = (
        intweights["walgc"]
        * halflength ** weightargs["alg1power"]
        * (jnp.log(b - a) + intweights["walglogc"] if weightargs["log1present"] else 1)
    )
    we = (
        intweights["walge"]
        * halflength ** weightargs["alg1power"]
        * (jnp.log(b - a) + intweights["walgloge"] if weightargs["log1present"] else 1)
    )

    return _fixed_quadcc_base(fun, a, b, xc, wc, we, args, norm, n)


def _fixed_quadcc_alglogweight_rightsingular(
    fun, a, b, weightargs, args=(), norm=jnp.inf, n=32, intweights=None
):
    r"""Algebraic-Logarithmic weighted quadrature with a singular point at b.

    Uses a modified Clenshaw-Curtis fixed order rule to compute the integral
    from a to b of f(x) * w(x) where

    .. math ::

    w(x) = (x - s_1)**\alpha * (b - x)**\beta * \log^i(x - s_1) * \log^j(b - x)

    where s_1 < a < b, ..math::`\alpha, \beta > -1`, and
    ..math::`i, j = \mathrm{0 or 1}`.

    """
    # modify function with nonsingular parts of the weight function
    fun = lambda x, args: (
        fun(x, *args)
        * (
            alglogweightfn(
                x,
                alg1power=weightargs["alg1power"],
                alg2power=0,
                log1present=weightargs["log1present"],
                log2present=False,
                s1=weightargs["s1"],
                s2=0,
            )
        )
    )
    if intweights is None:
        # compute weights for integration
        intweights = ccmod_get_weights(n, weightargs["alg2power"])

    halflength = (b - a) / 2

    xc = intweights["xc"]
    wc = (
        intweights["walgc"]
        * halflength ** weightargs["alg2power"]
        * (jnp.log(b - a) + intweights["walglogc"] if weightargs["log2present"] else 1)
    )
    we = (
        intweights["walge"]
        * halflength ** weightargs["alg2power"]
        * (jnp.log(b - a) + intweights["walgloge"] if weightargs["log2present"] else 1)
    )

    return _fixed_quadcc_base(fun, a, b, xc, wc, we, args, norm, n)
'''
