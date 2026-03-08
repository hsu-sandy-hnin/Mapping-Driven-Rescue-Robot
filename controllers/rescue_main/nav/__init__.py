"""Navigation package scaffolding for Webots rescue controller.

Includes a small dataclass compatibility shim so the package can import under
Python 3.8/3.9 where `dataclass(slots=True)` is not supported.
"""

import dataclasses as _dataclasses
import inspect as _inspect


def _patch_dataclass_slots_compat() -> None:
    """Drop unsupported `slots=` kwarg on older Python versions.

    The nav modules use `from dataclasses import dataclass` widely. Because this
    package `__init__` runs before submodule imports, patching the stdlib module
    here ensures those imports receive the compatibility wrapper.
    """
    try:
        params = _inspect.signature(_dataclasses.dataclass).parameters
    except Exception:
        return
    if "slots" in params:
        return

    _orig_dataclass = _dataclasses.dataclass

    def _dataclass_compat(*args, **kwargs):
        kwargs.pop("slots", None)
        return _orig_dataclass(*args, **kwargs)

    _dataclasses.dataclass = _dataclass_compat


_patch_dataclass_slots_compat()

from .config import (
    ControllerGains,
    GridConfig,
    MotionLimits,
    NavigationConfig,
    PlannerConfig,
)
from .logger import NavLogger
from .types import GridData, GridIndex, GridMap, Path, PlannerStatus, Pose2D, Twist, Waypoint

__all__ = [
    "ControllerGains",
    "GridConfig",
    "GridData",
    "GridIndex",
    "GridMap",
    "MotionLimits",
    "NavigationConfig",
    "NavLogger",
    "Path",
    "PlannerConfig",
    "PlannerStatus",
    "Pose2D",
    "Twist",
    "Waypoint",
]
