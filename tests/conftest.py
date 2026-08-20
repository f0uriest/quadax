"""Fixtures etc for testing."""

import ctypes
import ctypes.util
import gc
import inspect
import os
import warnings

_original_warn = warnings.warn


def smart_warn(message, category=None, stacklevel=1, source=None):
    """Ignore deprecation warnings from libs, if we aren't calling them directly."""
    # ie, if we call foo, which calls bar, and bar emits a deprecation warning, we
    # should ignore that since it's really foo's problem. If we call bar directly
    # we want to see the warning since that's on us to fix.
    cat = category or (
        message.__class__ if isinstance(message, Warning) else UserWarning
    )

    if issubclass(cat, DeprecationWarning):
        # get the name of all the modules in the stack at the time of the warning
        stack = inspect.stack()
        modules_in_stack = [
            frame.frame.f_globals.get("__name__", "") for frame in stack
        ]

        # Filter out warnings, conftest, and python's internal stuff
        call_chain = [
            m
            for m in modules_in_stack
            if m
            and not m.startswith("warnings")
            and not m.startswith("importlib")
            and not m.startswith("_pytest")
            and not m.startswith("pluggy")
            and "conftest" not in m
        ]
        # get just the package names, don't care about specific modules
        call_chain = [c.split(".")[0] for c in call_chain]

        if call_chain:
            emitter = call_chain[0]
            for caller in call_chain[1:]:
                if caller == emitter:
                    # internal to emitting library, assume it's on us to fix
                    continue
                elif caller != emitter and caller != "quadax":
                    # warning is caused by intermediate party, ignore it
                    return

    # otherwise, fall back to original behavior (which pytest turns into an error)
    return _original_warn(message, category, stacklevel, source)


# Need to do this here before any other imports in order to catch import time
# deprecation warnings
warnings.warn = smart_warn


import pytest  # noqa:E402


@pytest.fixture
def quiet_tanhsinh():
    """Let the half precision tanh-sinh warning through without failing the test.

    ``pyproject.toml`` turns warnings into errors, which is right for the rest of the
    suite. The warning itself is asserted on separately in ``TestTanhSinhPrecision``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tanh-sinh quadrature in.*")
        yield


# JAX and equinox cache a compiled executable per distinct integrand, and the suite
# passes a fresh closure in nearly every test, so almost nothing is ever a cache hit.
# The caches hang off module level functions in quadax that live for the whole session,
# so nothing is evicted either and the process grows without bound - a full run reaches
# a working set large enough to be killed on a CI runner. Dropping the caches when the
# process gets large bounds the high water mark; the cost is recompiling the shared
# machinery afterwards, which is why this triggers on size rather than after every test.
#
# Freeing the caches is not sufficient on its own. glibc keeps the freed pages on its
# own free lists instead of returning them, so RSS - which is what a runner's memory
# limit measures - only falls once malloc_trim hands them back.
SWEEP_RSS_MB = 1500

# Used only when RSS cannot be read, so that the sweep still happens on platforms
# without procfs.
SWEEP_EVERY = 50

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_tests_since_sweep = 0


def _rss_mb():
    """Resident set size in MB, or None where it cannot be read."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE_SIZE / 1e6
    except OSError:
        return None


def _release_memory():
    """Drop the compilation caches and return the freed pages to the OS."""
    import jax

    jax.clear_caches()
    gc.collect()
    try:
        ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        # malloc_trim is glibc's; elsewhere the allocator decides on its own.
        pass


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Bound the process working set over a long run."""
    global _tests_since_sweep

    rss = _rss_mb()
    if rss is None:
        _tests_since_sweep += 1
        if _tests_since_sweep < SWEEP_EVERY:
            return
    elif rss < SWEEP_RSS_MB:
        return

    _tests_since_sweep = 0
    _release_memory()
