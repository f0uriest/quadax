"""Weighted quadrature."""

import abc
from collections.abc import Callable
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp

from .fixed_order import NestedRule
from .quad_weights import _chebmom_alg_log_plus, _chebmom_alg_plus


class AbstractWeightedRule(eqx.Module):
    """Base class for weighted quadrature rules."""

    @abc.abstractmethod
    def apply_weighted_rule(self, x1: float, x2: float) -> bool:
        """Check if weighted rule should be applied in the interval [x1, x2]."""
        pass

    @abc.abstractmethod
    def evaluate_weight(self, x: float) -> float:
        """Evaluate the weight function at x."""
        pass


class AlgLogWeightRule(NestedRule):
    r"""Base class for quadrature rules with algebraic-logarithmic weight functions.

    Weight functions are of the form

    .. math ::

        w(x) = (x - a)**\gamma  \log^i(x - a)

    or

    .. math ::

        w(x) = (b - x)**\gamma \log^i(b - x)

    where ..math::`\gamma > -1`, ..math::`i = \mathrm{0 or 1}`, and ...math::`a, b`
    are the singular points of the weight function and represent the end points of the
    integration region [a, b], so ..math::`a <= x <= b`.

    Parameters
    ----------
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

    algpower: float
    logpower: bool
    algweights: jax.Array
    alglogweights: jax.Array
    algweights_half_ord: jax.Array
    alglogweights_half_ord: jax.Array

    def __init__(
        self,
        algpower: float,
        logpower: bool,
        order: int = 32,
        norm: Union[Callable, int] = jnp.inf,
    ):
        self.algpower = algpower
        self.logpower = logpower
        self.norm = (
            norm if callable(norm) else lambda x: jnp.linalg.norm(x.flatten(), ord=norm)
        )

        if self.algpower <= -1:
            raise NotImplementedError(
                f"Power of algebraic term must be > -1. It is {self.algpower}"
            )

        # compute integration nodes and (partial) weights
        def _mod_cc_partial_weights(N):
            """Modified Clenshaw-Curtis quadrature nodes and partial weights.

            Full weights depend on the integration interval.

            Parameters
            ----------
            N: int (assumed to be even)
                Order of the quadrature rule.
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
        # integration nodes and partial weights for higher order rule
        self.xh, self.algweights, self.alglogweights = _mod_cc_partial_weights(order)
        # partial weights for lower (half) order rule
        _, algweights_half_ord, alglogweights_half_ord = _mod_cc_partial_weights(
            order // 2
        )
        self.algweights_half_ord = (
            jnp.zeros_like(self.algweights).at[::2].set(algweights_half_ord)
        )
        self.alglogweights_half_ord = (
            jnp.zeros_like(self.alglogweights).at[::2].set(alglogweights_half_ord)
        )

        # complete integration weights computed during integration
        self.wl = None
        self.wh = None

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

        # update integration weights
        new_wh = halflength**self.algpower * (
            (jnp.log(b - a) if self.logpower else 1) * self.algweights
            + (self.alglogweights if self.logpower else 0)
        )
        self = eqx.tree_at(lambda t: t.wh, self, new_wh, is_leaf=lambda x: x is None)
        new_wl = halflength**self.algpower * (
            (jnp.log(b - a) if self.logpower else 1) * self.algweights_half_ord
            + (self.alglogweights_half_ord if self.logpower else 0)
        )
        self = eqx.tree_at(lambda t: t.wl, self, new_wl, is_leaf=lambda x: x is None)

        return super().integrate(fun, a, b, args)


class AlgLogLeftSingularRule(AbstractWeightedRule, AlgLogWeightRule):
    r"""Quadrature for left-endpoint singular algebraic-logarithmic weight functions.

    Weight functions are of the form

    .. math ::

        w(x) = (x - a)**\gamma  \log^i(x - a)


    where ..math::`\gamma > -1`, ..math::`i = \mathrm{0 or 1}`, and ...math::`a <= x`.

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

    singularpoint: float

    def __init__(
        self,
        singularpoint: float,
        algpower: float,
        logpower: bool,
        order: int = 32,
        norm: Union[Callable, int] = jnp.inf,
    ):
        self.singularpoint = singularpoint
        AlgLogWeightRule.__init__(self, algpower, logpower, order, norm)

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
        return self.singularpoint == x1

    def evaluate_weight(self, x):
        """Evaluate the weight function at x."""
        log = jnp.log(x - self.singularpoint) if self.logpower else 1
        return (x - self.singularpoint) ** self.algpower * log


class AlgLogRightSingularRule(AbstractWeightedRule, AlgLogWeightRule):
    r"""Quadrature for right-endpoint singular algebraic-logarithmic weight functions.

    Weight functions are of the form

    .. math ::

        w(x) = (b - x)**\gamma  \log^i(b - x)


    where ..math::`\gamma > -1`, ..math::`i = \mathrm{0 or 1}`, and ...math::`b >= x`.

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
        AlgLogWeightRule.__init__(algpower, logpower, order, norm)

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
        return self.singularpoint == x2

    def evaluate_weight(self, x):
        """Evaluate the weight function at x."""
        log = jnp.log(self.singularpoint - x) if self.logpower else 1
        return (self.singularpoint - x) ** self.algpower * log
