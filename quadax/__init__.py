"""quadax : numerical quadrature with JAX."""

from . import _version
from .adaptive import adaptive_quadrature, quadcc, quadgk, quadts
from .fixed_order import fixed_quadcc, fixed_quadgk, fixed_quadts
from .romberg import romberg, rombergts
from .sampled import cumulative_trapezoid, simpson, trapezoid
from .utils import STATUS

__version__ = _version.get_versions()["version"]
