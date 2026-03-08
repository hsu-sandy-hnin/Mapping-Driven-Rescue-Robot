"""CSV logger for navigation telemetry."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from .types import Pose2D, Waypoint

GoalLike = Union[Pose2D, Waypoint]


@dataclass(slots=True)
class NavLogger:
    """Lightweight CSV logger writing to `./logs` by default."""

    log_dir: Union[str, Path] = Path("logs")
    file_name: str = "navigation.csv"
    file_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.log_dir / self.file_name
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            return
        with self.file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "timestamp",
                    "pose_x",
                    "pose_y",
                    "pose_theta",
                    "goal_x",
                    "goal_y",
                    "goal_theta",
                    "mode",
                    "replans",
                    "collisions_count",
                    "stuck_events_count",
                ]
            )

    def append_row(
        self,
        pose: Pose2D,
        goal: GoalLike,
        mode: str,
        replans: int,
        collisions_count: int,
        stuck_events_count: int,
        timestamp: datetime | None = None,
    ) -> None:
        """Append one telemetry row to CSV."""
        ts = timestamp or datetime.now(timezone.utc)
        with self.file_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    ts.isoformat(),
                    pose.x,
                    pose.y,
                    pose.theta,
                    goal.x,
                    goal.y,
                    goal.theta,
                    mode,
                    replans,
                    collisions_count,
                    stuck_events_count,
                ]
            )
