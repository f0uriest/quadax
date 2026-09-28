"""quadax : numerical quadrature with JAX."""

from . import _version
from ._adaptive import adaptive_quadrature, quadcc, quadgk, quadts
from ._adaptive_cube import adaptive_cubature, cubegm
from ._adjoint import AbstractAdjoint, DirectAdjoint, LeibnizAdjoint
from ._fixed_cubature import (
    AbstractCubatureRule,
    GenzMalikRule,
    TensorProductRule,
)
from ._fixed_order import (
    AbstractQuadratureRule,
    ClenshawCurtisRule,
    GaussKronrodRule,
    NestedRule,
    TanhSinhRule,
)
from ._romberg import romberg, rombergts, tanhsinh
from ._sampled import cumulative_simpson, cumulative_trapezoid, simpson, trapezoid
from ._status import STATUS
from ._utils import QuadratureInfo

__all__ = [
    "adaptive_quadrature",
    "quadcc",
    "quadgk",
    "quadts",
    "cubegm",
    "adaptive_cubature",
    "AbstractQuadratureRule",
    "ClenshawCurtisRule",
    "GaussKronrodRule",
    "NestedRule",
    "TanhSinhRule",
    "AbstractCubatureRule",
    "GenzMalikRule",
    "TensorProductRule",
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


def _deprecated_submodules(names):
    """Make each ``quadax.<name>`` a deprecated alias of ``quadax._<name>``."""
    import importlib
    import sys
    import types
    import warnings

    def warn(name):
        warnings.warn(
            f"`{__name__}.{name}` is not part of the public API and importing it is "
            f"deprecated. Import from `{__name__}` instead; the old module path will "
            "be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )

    for name in names:
        private = importlib.import_module(f"{__name__}._{name}")
        alias = types.ModuleType(f"{__name__}.{name}")

        def alias_getattr(attr, name=name, private=private):
            # Tools that scan sys.modules, such as inspect.getmodule, probe dunder
            # attributes of every module, so those must not warn or be forwarded.
            if attr[:2] == attr[-2:] == "__":
                raise AttributeError(attr)
            warn(name)
            return getattr(private, attr)

        setattr(alias, "__all__", [k for k in vars(private) if not k.startswith("_")])
        setattr(alias, "__getattr__", alias_getattr)
        # Since the alias is already in sys.modules, importing it never binds it as an
        # attribute of the package, so it cannot shadow a public name like `romberg`.
        sys.modules[alias.__name__] = alias

    def package_getattr(name):
        if name not in names:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        warn(name)
        return sys.modules[f"{__name__}.{name}"]

    return package_getattr


# Assigned through globals() so that static type checkers do not treat arbitrary
# attributes of the package as valid.
globals()["__getattr__"] = _deprecated_submodules(
    [
        "adaptive",
        "adaptive_cube",
        "adjoint",
        "fixed_cubature",
        "fixed_order",
        "quad_weights",
        "romberg",
        "sampled",
        "utils",
    ]
)
