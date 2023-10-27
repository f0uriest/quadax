"""quadax : numerical quadrature with JAX."""

from . import _version
from .adaptive import adaptive_quadrature, quadcc, quadgk
from .fixed_order import fixed_quadcc, fixed_quadgk
from .romberg import romberg
from .sampled import cumulative_trapezoid, simpson, trapezoid
from .tanhsinh import quadts
from .utils import STATUS

__version__ = _version.get_versions()["version"]
