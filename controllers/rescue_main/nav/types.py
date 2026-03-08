"""Shared navigation types and lightweight data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

# Grid cell coordinates as integer indices (row, col).
GridIndex = Tuple[int, int]
# Occupancy grid data (0=free, 100=occupied, or project-defined equivalent).
GridData = List[List[int]]


@dataclass(slots=True)
class Pose2D:
    """Robot pose in a planar world frame."""

    x: float
    y: float
    theta: float = 0.0


@dataclass(slots=True)
class Waypoint:
    """Path waypoint in world coordinates."""

    x: float
    y: float
    theta: float = 0.0
    speed_hint: float = 0.0


# Ordered waypoint list.
Path = List[Waypoint]


@dataclass(slots=True)
class Twist:
    """Velocity command for differential/unicycle robot models."""

    v: float
    omega: float


@dataclass(slots=True)
class GridMap:
    """Occupancy grid metadata and cells."""

    resolution_m: float
    width: int
    height: int
    origin: Pose2D = field(default_factory=lambda: Pose2D(0.0, 0.0, 0.0))
    data: GridData = field(default_factory=list)

    @classmethod
    def empty(
        cls,
        width: int,
        height: int,
        resolution_m: float,
        origin: Pose2D | None = None,
        fill_value: int = 0,
    ) -> "GridMap":
        """Build an empty grid initialized to `fill_value`."""
        grid = [[fill_value for _ in range(width)] for _ in range(height)]
        return cls(
            resolution_m=resolution_m,
            width=width,
            height=height,
            origin=origin or Pose2D(0.0, 0.0, 0.0),
            data=grid,
        )

    def in_bounds(self, cell: GridIndex) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width


@dataclass(slots=True)
class PlannerStatus:
    """Runtime planner state snapshot for debugging/telemetry."""

    mode: str = "IDLE"
    replans: int = 0
    has_path: bool = False
    message: str = ""
