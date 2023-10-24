from .adaptive import adaptive_quadrature, quadgk
from .fixed_qk import fixed_quadgk
from .romberg import romberg
from .tanhsinh import quadts
from .sampled import trapezoid, cumulative_trapezoid, simpson

from . import _version

__version__ = _version.get_versions()["version"]
