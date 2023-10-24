"""quadax : numerical quadrature with JAX."""

from . import _version
from .adaptive import adaptive_quadrature, quadgk
from .fixed_qk import fixed_quadgk
from .romberg import romberg
from .sampled import cumulative_trapezoid, simpson, trapezoid
from .tanhsinh import quadts

__version__ = _version.get_versions()["version"]
