"""Replanning manager for deciding when to rerun global A*."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from math import floor, hypot
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .a_star import plan_a_star_world
from .grid_utils import world_to_grid
from .logger import NavLogger
from .types import PlannerStatus, Pose2D

GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]

UNKNOWN = -1
OCCUPIED_ALT = 1


def _cfg_get(config: object | None, name: str, default):
    if config is None:
        return default
    if hasattr(config, name):
        return getattr(config, name)
    if hasattr(config, "planner") and hasattr(config.planner, name):
        return getattr(config.planner, name)
    return default


def _cfg_get_grid(config: object | None, name: str, default):
    if config is None:
        return default
    if hasattr(config, name):
        return getattr(config, name)
    if hasattr(config, "grid") and hasattr(config.grid, name):
        return getattr(config.grid, name)
    return default


def _grid_origin_xy(grid: object) -> Tuple[float, float]:
    if hasattr(grid, "origin_x_m") and hasattr(grid, "origin_y_m"):
        return float(getattr(grid, "origin_x_m")), float(getattr(grid, "origin_y_m"))
    if hasattr(grid, "origin"):
        origin = getattr(grid, "origin")
        if hasattr(origin, "x") and hasattr(origin, "y"):
            return float(getattr(origin, "x")), float(getattr(origin, "y"))
    return 0.0, 0.0


def _obstacle_threshold(config: object | None) -> int:
    if config is None:
        return 50
    if hasattr(config, "obstacle_threshold"):
        return int(getattr(config, "obstacle_threshold"))
    if hasattr(config, "grid") and hasattr(config.grid, "obstacle_threshold"):
        return int(getattr(config.grid, "obstacle_threshold"))
    return 50


def _is_grid_like(obj: object) -> bool:
    return isinstance(obj, np.ndarray) or hasattr(obj, "data")


@dataclass(slots=True)
class _PlanningGrid:
    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    data: np.ndarray


def _normalize_grid_like(grid_like: object, config: object | None) -> _PlanningGrid:
    if isinstance(grid_like, np.ndarray):
        data = grid_like
        if data.ndim != 2:
            raise ValueError("Grid ndarray must be 2D")
        height, width = data.shape
        resolution_m = float(_cfg_get_grid(config, "resolution_m", 1.0))
        origin_x_m = float(_cfg_get_grid(config, "origin_x_m", 0.0))
        origin_y_m = float(_cfg_get_grid(config, "origin_y_m", 0.0))
        return _PlanningGrid(
            width=width,
            height=height,
            resolution_m=resolution_m,
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
            data=np.asarray(data),
        )

    if not hasattr(grid_like, "data"):
        raise TypeError("Grid must be ndarray or expose `.data`")
    data = np.asarray(getattr(grid_like, "data"))
    if data.ndim != 2:
        raise ValueError("Grid data must be 2D")

    height = int(getattr(grid_like, "height", data.shape[0]))
    width = int(getattr(grid_like, "width", data.shape[1]))
    if (height, width) != data.shape:
        height, width = data.shape

    resolution_m = float(getattr(grid_like, "resolution_m", _cfg_get_grid(config, "resolution_m", 1.0)))
    if resolution_m <= 0.0:
        raise ValueError("Grid resolution_m must be > 0")
    origin_x_m, origin_y_m = _grid_origin_xy(grid_like)

    return _PlanningGrid(
        width=width,
        height=height,
        resolution_m=resolution_m,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        data=data,
    )


def _select_grid_and_version(source: object, config: object | None) -> tuple[_PlanningGrid, object]:
    """Pick planning grid (prefer inflated) and map version from callback output."""
    version: object | None = None
    chosen: object | None = None

    if isinstance(source, dict):
        for key in ("inflated_grid", "inflated", "grid", "raw_grid", "map"):
            if key in source and _is_grid_like(source[key]):
                chosen = source[key]
                break
        for key in ("map_version", "map_update_counter", "map_hash", "version"):
            if key in source:
                version = source[key]
                break

    elif isinstance(source, (tuple, list)):
        items = list(source)
        if len(items) >= 2 and _is_grid_like(items[0]) and _is_grid_like(items[1]):
            # (raw_grid, inflated_grid, [optional version])
            chosen = items[1]
            if len(items) >= 3 and not _is_grid_like(items[2]):
                version = items[2]
        elif len(items) >= 2 and _is_grid_like(items[0]) and not _is_grid_like(items[1]):
            # (grid, version)
            chosen = items[0]
            version = items[1]
        elif items and _is_grid_like(items[0]):
            chosen = items[0]

    elif _is_grid_like(source):
        chosen = source
        if hasattr(source, "map_update_counter"):
            version = getattr(source, "map_update_counter")
        elif hasattr(source, "map_hash"):
            version = getattr(source, "map_hash")
        elif hasattr(source, "version"):
            version = getattr(source, "version")

    if chosen is None:
        raise TypeError(
            "get_current_grid() must return a grid-like object, dict, or tuple/list with grid data"
        )

    grid = _normalize_grid_like(chosen, config)

    if version is None:
        if config is not None and hasattr(config, "map_update_counter"):
            version = getattr(config, "map_update_counter")
        elif config is not None and hasattr(config, "map_hash"):
            version = getattr(config, "map_hash")
        else:
            # Fallback deterministic fingerprint when explicit version is unavailable.
            version = hashlib.sha1(np.ascontiguousarray(grid.data).tobytes()).hexdigest()

    return grid, version


def _pose_xy(pose: Pose2D) -> Tuple[float, float]:
    if hasattr(pose, "x") and hasattr(pose, "y"):
        return float(pose.x), float(pose.y)
    raise TypeError("Pose must expose x/y attributes")


def _world_to_cell_clamped(x_m: float, y_m: float, grid: _PlanningGrid) -> GridCell:
    try:
        return world_to_grid(x_m, y_m, grid)
    except ValueError:
        ix = int(floor((x_m - grid.origin_x_m) / grid.resolution_m))
        iy = int(floor((y_m - grid.origin_y_m) / grid.resolution_m))
        ix = min(max(ix, 0), grid.width - 1)
        iy = min(max(iy, 0), grid.height - 1)
        return ix, iy


def _cell_blocked(value: int, allow_unknown: bool, obstacle_threshold: int) -> bool:
    if value == OCCUPIED_ALT or value >= obstacle_threshold:
        return True
    if value == UNKNOWN and not allow_unknown:
        return True
    return False


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return hypot(px - qx, py - qy)


def _distance_to_polyline(px: float, py: float, path: Sequence[WorldPoint]) -> float:
    if not path:
        return float("inf")
    if len(path) == 1:
        return hypot(px - path[0][0], py - path[0][1])
    best = float("inf")
    for i in range(1, len(path)):
        ax, ay = path[i - 1]
        bx, by = path[i]
        d = _point_segment_distance(px, py, ax, ay, bx, by)
        if d < best:
            best = d
    return best


def _closest_waypoint_index(path: Sequence[WorldPoint], x_m: float, y_m: float) -> int:
    best_i = 0
    best_d = float("inf")
    for i, (wx, wy) in enumerate(path):
        d = hypot(wx - x_m, wy - y_m)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _path_blocked_ahead(
    path_world: Sequence[WorldPoint],
    pose: Pose2D,
    grid: _PlanningGrid,
    lookahead_waypoints: int,
    allow_unknown: bool,
    obstacle_threshold: int,
) -> bool:
    if not path_world or lookahead_waypoints <= 0:
        return False
    px, py = _pose_xy(pose)
    start_i = _closest_waypoint_index(path_world, px, py)
    end_i = min(len(path_world), start_i + lookahead_waypoints)
    for i in range(start_i, end_i):
        wx, wy = path_world[i]
        try:
            ix, iy = world_to_grid(wx, wy, grid)
        except ValueError:
            return True
        v = int(grid.data[iy, ix])
        if _cell_blocked(v, allow_unknown, obstacle_threshold):
            return True
    return False


@dataclass(slots=True)
class ReplanningManager:
    """Decides when to rerun A* and stores the active global path."""

    get_current_grid: Callable[[], object]
    get_pose: Callable[[], Pose2D]
    get_goal: Callable[[], Tuple[float, float]]
    config: object
    logger: NavLogger | None = None
    time_fn: Callable[[], float] = time.monotonic

    last_plan_time: float | None = None
    last_start_cell: GridCell | None = None
    last_goal_cell: GridCell | None = None
    last_map_version: object | None = None
    current_path: List[WorldPoint] = field(default_factory=list)
    status: PlannerStatus = field(default_factory=PlannerStatus)

    _force_replan_flag: bool = field(default=False, init=False, repr=False)
    _last_plan_pose: Pose2D | None = field(default=None, init=False, repr=False)

    def force_replan(self) -> None:
        """Mark planner for forced replanning at next update."""
        self._force_replan_flag = True

    def _should_replan(
        self,
        now_s: float,
        pose: Pose2D,
        goal_cell: GridCell,
        map_version: object,
        grid: _PlanningGrid,
    ) -> List[str]:
        reasons: List[str] = []
        replanning_period_s = float(_cfg_get(self.config, "replanning_period_s", 0.5))
        min_pose_change_m = float(_cfg_get(self.config, "min_pose_change_to_replan_m", 0.08))
        path_deviation_m = float(_cfg_get(self.config, "path_deviation_m", 0.40))
        lookahead_wp = int(_cfg_get(self.config, "blocked_lookahead_waypoints", 8))
        map_change_replan_min_period_s = float(
            _cfg_get(self.config, "map_change_replan_min_period_s", 1.20)
        )
        allow_unknown = bool(_cfg_get(self.config, "allow_unknown", True))
        obstacle_threshold = _obstacle_threshold(self.config)

        if self._force_replan_flag:
            reasons.append("forced")
        if self.last_plan_time is None:
            reasons.append("initial")
        elif now_s - self.last_plan_time >= replanning_period_s:
            reasons.append("periodic")

        if self.last_goal_cell is not None and goal_cell != self.last_goal_cell:
            reasons.append("goal_changed")

        if self.last_map_version is not None and map_version != self.last_map_version:
            map_period_ok = (
                self.last_plan_time is None
                or (now_s - self.last_plan_time) >= map_change_replan_min_period_s
            )
            if map_period_ok:
                reasons.append("map_changed")

        if not self.current_path:
            reasons.append("no_path")
        else:
            px, py = _pose_xy(pose)
            moved_enough = True
            if self._last_plan_pose is not None:
                moved_enough = (
                    hypot(px - self._last_plan_pose.x, py - self._last_plan_pose.y) >= min_pose_change_m
                )
            if moved_enough:
                deviation = _distance_to_polyline(px, py, self.current_path)
                if deviation > path_deviation_m:
                    reasons.append("path_deviation")

            if _path_blocked_ahead(
                self.current_path,
                pose,
                grid,
                lookahead_wp,
                allow_unknown=allow_unknown,
                obstacle_threshold=obstacle_threshold,
            ):
                reasons.append("path_blocked")

        # Deterministic de-dup preserving order.
        seen = set()
        deduped = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped

    def _append_log(self, pose: Pose2D, goal: Pose2D) -> None:
        if self.logger is None:
            return
        collisions_count = int(getattr(self.status, "collisions_count", 0))
        stuck_events_count = int(getattr(self.status, "stuck_events_count", 0))
        self.logger.append_row(
            pose=pose,
            goal=goal,
            mode=self.status.mode,
            replans=self.status.replans,
            collisions_count=collisions_count,
            stuck_events_count=stuck_events_count,
        )

    def update(self, now_s: float | None = None) -> tuple[List[WorldPoint], PlannerStatus]:
        """Update replanning state and return `(current_path, status)`."""
        now = float(now_s if now_s is not None else self.time_fn())

        grid_source = self.get_current_grid()
        planning_grid, map_version = _select_grid_and_version(grid_source, self.config)

        pose = self.get_pose()
        goal_x_m, goal_y_m = self.get_goal()
        goal_pose = Pose2D(goal_x_m, goal_y_m, 0.0)

        start_x_m, start_y_m = _pose_xy(pose)
        start_cell = _world_to_cell_clamped(start_x_m, start_y_m, planning_grid)
        goal_cell = _world_to_cell_clamped(goal_x_m, goal_y_m, planning_grid)

        reasons = self._should_replan(now, pose, goal_cell, map_version, planning_grid)
        if reasons:
            self.status.replans += 1
            new_path = plan_a_star_world(
                planning_grid,
                pose,
                goal_pose,
                self.config,
            )
            self.last_plan_time = now
            self.last_start_cell = start_cell
            self.last_goal_cell = goal_cell
            self.last_map_version = map_version
            self._last_plan_pose = Pose2D(pose.x, pose.y, pose.theta)
            self._force_replan_flag = False

            if new_path:
                self.current_path = list(new_path)
                self.status.mode = "REPLAN_OK"
                self.status.has_path = True
                self.status.message = ",".join(reasons)
            else:
                self.status.mode = "REPLAN_FAIL"
                self.status.has_path = bool(self.current_path)
                self.status.message = ",".join(reasons)
        else:
            if self.current_path:
                self.status.mode = "FOLLOW_PATH"
                self.status.has_path = True
                self.status.message = "reuse_path"
            else:
                self.status.mode = "IDLE"
                self.status.has_path = False
                self.status.message = "no_path"

        self._append_log(pose, goal_pose)
        return list(self.current_path), self.status


Replanner = ReplanningManager


@dataclass(slots=True)
class _SimGrid:
    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    data: np.ndarray
    map_update_counter: int = 0


def _world_to_grid_for_demo(x_m: float, y_m: float, grid: _SimGrid) -> GridCell:
    ix = int(floor((x_m - grid.origin_x_m) / grid.resolution_m))
    iy = int(floor((y_m - grid.origin_y_m) / grid.resolution_m))
    ix = min(max(ix, 0), grid.width - 1)
    iy = min(max(iy, 0), grid.height - 1)
    return ix, iy


def _grid_to_world_for_demo(ix: int, iy: int, grid: _SimGrid) -> WorldPoint:
    x_m = grid.origin_x_m + (ix + 0.5) * grid.resolution_m
    y_m = grid.origin_y_m + (iy + 0.5) * grid.resolution_m
    return x_m, y_m


def _step_pose_toward(path: Sequence[WorldPoint], pose_state: dict, speed_m_per_step: float) -> None:
    if not path:
        return
    target = path[min(1, len(path) - 1)]
    tx, ty = target
    px = pose_state["x"]
    py = pose_state["y"]
    dx = tx - px
    dy = ty - py
    d = hypot(dx, dy)
    if d < 1e-9:
        return
    s = min(speed_m_per_step, d) / d
    pose_state["x"] = px + dx * s
    pose_state["y"] = py + dy * s


def _demo() -> None:
    from .config import NavigationConfig

    rng = np.random.default_rng(21)
    width, height = 28, 18
    data = np.zeros((height, width), dtype=np.int16)
    obstacle_mask = rng.random((height, width)) < 0.14
    data[obstacle_mask] = 100

    grid = _SimGrid(
        width=width,
        height=height,
        resolution_m=0.2,
        origin_x_m=0.0,
        origin_y_m=0.0,
        data=data,
        map_update_counter=0,
    )

    start_cell = (1, 1)
    goal_cell = (width - 2, height - 2)
    # Carve a deterministic corridor so a path exists.
    for t in np.linspace(0.0, 1.0, 80):
        ix = int(round(start_cell[0] * (1.0 - t) + goal_cell[0] * t))
        iy = int(round(start_cell[1] * (1.0 - t) + goal_cell[1] * t))
        for yy in range(max(0, iy - 1), min(height, iy + 2)):
            for xx in range(max(0, ix - 1), min(width, ix + 2)):
                grid.data[yy, xx] = 0

    start_x_m, start_y_m = _grid_to_world_for_demo(start_cell[0], start_cell[1], grid)
    goal_x_m, goal_y_m = _grid_to_world_for_demo(goal_cell[0], goal_cell[1], grid)
    pose_state = {"x": start_x_m, "y": start_y_m}
    goal_state = {"x": goal_x_m, "y": goal_y_m}

    cfg = NavigationConfig()
    cfg.grid.resolution_m = grid.resolution_m
    cfg.planner.replanning_period_s = 2.0
    cfg.planner.min_pose_change_to_replan_m = 0.10
    cfg.planner.path_deviation_m = 0.45
    cfg.planner.blocked_lookahead_waypoints = 6
    cfg.planner.allow_unknown = True
    cfg.planner.unknown_penalty = 2.0
    cfg.planner.heuristic_weight = 1.0
    cfg.planner.start_goal_search_radius_m = 1.0

    sim_time_s = 0.0

    def _time_fn() -> float:
        return sim_time_s

    def get_current_grid():
        # Demo returns both raw and inflated entries (identical here).
        return {
            "raw_grid": grid,
            "inflated_grid": grid,
            "map_update_counter": grid.map_update_counter,
        }

    def get_pose() -> Pose2D:
        return Pose2D(pose_state["x"], pose_state["y"], 0.0)

    def get_goal() -> tuple[float, float]:
        return goal_state["x"], goal_state["y"]

    logger = NavLogger(log_dir="logs", file_name="replanner_demo.csv")
    manager = ReplanningManager(
        get_current_grid=get_current_grid,
        get_pose=get_pose,
        get_goal=get_goal,
        config=cfg,
        logger=logger,
        time_fn=_time_fn,
    )

    print("Simulated replanning loop:")
    for step in range(16):
        sim_time_s += 0.5

        if step == 4:
            # Goal moved to a different cell.
            gx, gy = _grid_to_world_for_demo(width - 3, height - 4, grid)
            goal_state["x"], goal_state["y"] = gx, gy
        if step == 7:
            # Map changed + version updated.
            ix, iy = _world_to_grid_for_demo(pose_state["x"] + 1.0, pose_state["y"], grid)
            grid.data[iy, max(0, ix - 1) : min(width, ix + 2)] = 100
            grid.map_update_counter += 1
        if step == 10:
            # Force off-path jump to trigger deviation-based replanning.
            pose_state["x"] += 0.9
            pose_state["y"] -= 0.6
        if step == 12:
            # Block next corridor without bumping version (tests path_blocked rule).
            if manager.current_path:
                wx, wy = manager.current_path[min(2, len(manager.current_path) - 1)]
                ix, iy = _world_to_grid_for_demo(wx, wy, grid)
                grid.data[max(0, iy - 1) : min(height, iy + 2), ix] = 100
        if step == 14:
            manager.force_replan()

        path, status = manager.update(now_s=sim_time_s)
        _step_pose_toward(path, pose_state, speed_m_per_step=0.22)
        print(
            f"t={sim_time_s:4.1f}s mode={status.mode:>10} "
            f"replans={status.replans:2d} path={len(path):2d} reason={status.message}"
        )

    print("Demo finished. Logs written to logs/replanner_demo.csv")


if __name__ == "__main__":
    _demo()
