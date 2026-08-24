"""quadax : numerical quadrature with JAX."""

from . import _version
from ._status import STATUS
from .adaptive import adaptive_quadrature, quadcc, quadgk, quadts
from .adjoint import AbstractAdjoint, DirectAdjoint, LeibnizAdjoint
from .fixed_order import (
    AbstractQuadratureRule,
    ClenshawCurtisRule,
    GaussKronrodRule,
    NestedRule,
    TanhSinhRule,
)
from .romberg import romberg, rombergts, tanhsinh
from .sampled import cumulative_simpson, cumulative_trapezoid, simpson, trapezoid
from .utils import QuadratureInfo

__all__ = [
    "adaptive_quadrature",
    "quadcc",
    "quadgk",
    "quadts",
    "AbstractQuadratureRule",
    "ClenshawCurtisRule",
    "GaussKronrodRule",
    "NestedRule",
    "TanhSinhRule",
    "AbstractAdjoint",
    "DirectAdjoint",
    "LeibnizAdjoint",
    "romberg",
    "rombergts",
    "tanhsinh",
    "cumulative_simpson",
    "cumulative_trapezoid",
    "simpson",
    "trapezoid",
    "STATUS",
    "QuadratureInfo",
]

__version__ = _version.get_versions()["version"]
