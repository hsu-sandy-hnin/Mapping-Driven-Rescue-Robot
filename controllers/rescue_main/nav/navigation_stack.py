"""End-to-end navigation stack orchestration for Webots controllers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import cos, hypot, sin
from typing import Callable, Sequence

import numpy as np

from .config import NavigationConfig
from .inflation import inflate_from_config
from .local_controller import LocalController
from .logger import NavLogger
from .obstacle_avoidance import adjust_twist_with_avoidance
from .path_smoothing import path_to_waypoints, smooth_path_to_waypoints
from .replanner import ReplanningManager
from .stuck_recovery import RecoveryState, StuckRecoveryManager
from .types import PlannerStatus, Pose2D, Twist, Waypoint

MODE_IDLE = "IDLE"
MODE_PLANNING = "PLANNING"
MODE_FOLLOWING = "FOLLOWING"
MODE_RECOVERY = "RECOVERY"
MODE_GOAL_REACHED = "GOAL_REACHED"


@dataclass(slots=True)
class _GridView:
    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    data: np.ndarray

    @property
    def origin(self) -> Pose2D:
        return Pose2D(self.origin_x_m, self.origin_y_m, 0.0)


def _is_grid_like(obj: object) -> bool:
    return isinstance(obj, np.ndarray) or hasattr(obj, "data")


def _coerce_pose2d(value: object) -> Pose2D:
    if isinstance(value, Pose2D):
        return value
    if hasattr(value, "x") and hasattr(value, "y"):
        theta = float(getattr(value, "theta", 0.0))
        return Pose2D(float(getattr(value, "x")), float(getattr(value, "y")), theta)
    if isinstance(value, Sequence) and len(value) >= 2:
        theta = float(value[2]) if len(value) >= 3 else 0.0
        return Pose2D(float(value[0]), float(value[1]), theta)
    raise TypeError("Pose must be Pose2D-like or a sequence [x, y, (theta)]")


def _coerce_goal_xy(value: object | None) -> tuple[float, float] | None:
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(getattr(value, "x")), float(getattr(value, "y"))
    if isinstance(value, Sequence) and len(value) >= 2:
        return float(value[0]), float(value[1])
    raise TypeError("Goal must be None, Pose2D-like, or a sequence [x, y]")


def _extract_origin_xy(grid: object) -> tuple[float, float]:
    if hasattr(grid, "origin_x_m") and hasattr(grid, "origin_y_m"):
        return float(getattr(grid, "origin_x_m")), float(getattr(grid, "origin_y_m"))
    if hasattr(grid, "origin"):
        origin = getattr(grid, "origin")
        if hasattr(origin, "x") and hasattr(origin, "y"):
            return float(getattr(origin, "x")), float(getattr(origin, "y"))
    return 0.0, 0.0


def _coerce_grid_view(grid: object) -> _GridView:
    if isinstance(grid, np.ndarray):
        data = grid
        if data.ndim != 2:
            raise ValueError("Grid ndarray must be 2D")
        height, width = data.shape
        return _GridView(
            width=width,
            height=height,
            resolution_m=1.0,
            origin_x_m=0.0,
            origin_y_m=0.0,
            data=data,
        )

    if not hasattr(grid, "data"):
        raise TypeError("Grid object must expose `data`")
    data = np.asarray(getattr(grid, "data"))
    if data.ndim != 2:
        raise ValueError("Grid data must be 2D")

    height = int(getattr(grid, "height", data.shape[0]))
    width = int(getattr(grid, "width", data.shape[1]))
    if (height, width) != data.shape:
        height, width = data.shape

    resolution_m = float(getattr(grid, "resolution_m", 1.0))
    if resolution_m <= 0.0:
        raise ValueError("Grid resolution_m must be > 0")
    origin_x_m, origin_y_m = _extract_origin_xy(grid)
    return _GridView(
        width=width,
        height=height,
        resolution_m=resolution_m,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        data=data,
    )


def _hash_grid_data(data: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(data).tobytes()).hexdigest()


def _path_signature(path_world: Sequence[tuple[float, float]]) -> str:
    if not path_world:
        return "EMPTY"
    arr = np.asarray(path_world, dtype=np.float64)
    digest = hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()
    return f"{len(path_world)}:{digest}"


class NavigationStack:
    """Coordinates mapping, planning, control, avoidance, recovery, and logging."""

    def __init__(
        self,
        get_grid: Callable[[], object],
        get_pose: Callable[[], object],
        get_goal: Callable[[], object | None],
        get_lidar: Callable[[], Sequence[float] | None],
        send_cmd: Callable[[Twist], None],
        config: NavigationConfig | None = None,
        logger: NavLogger | None = None,
        enable_inflation: bool = True,
    ) -> None:
        self._get_grid_cb = get_grid
        self._get_pose_cb = get_pose
        self._get_goal_cb = get_goal
        self._get_lidar_cb = get_lidar
        self._send_cmd_cb = send_cmd

        self.config = config or NavigationConfig()
        self.logger = logger or NavLogger(file_name=self.config.log_file_name)
        self.enable_inflation = bool(enable_inflation)

        # Goal state.
        self._manual_goal_xy: tuple[float, float] | None = None
        self._active_goal_xy: tuple[float, float] | None = None
        self._last_goal_xy: tuple[float, float] | None = None
        self._goal_reached = False

        # Grid/inflation cache.
        self._step_token = 0
        self._grid_cache_token = -1
        self._cached_map_version: object | None = None
        self._raw_grid: _GridView | None = None
        self._inflated_grid: _GridView | None = None
        self._cost_grid: np.ndarray | None = None
        self._latest_map_version: object | None = None

        # Planning/control state.
        self._replanner = ReplanningManager(
            get_current_grid=self._replanner_get_current_grid,
            get_pose=self._replanner_get_pose,
            get_goal=self._replanner_get_goal,
            config=self.config,
            logger=None,
        )
        self._local_controller = LocalController(self.config)
        self._stuck_recovery = StuckRecoveryManager(config=self.config, replanner=self._replanner)

        self._current_path_world: list[tuple[float, float]] = []
        self._current_waypoints: list[Waypoint] = []
        self._path_sig = "EMPTY"

        # Debug/telemetry state.
        self._mode = MODE_IDLE
        self._emergency_stop = False
        self._stuck_state = RecoveryState.NORMAL.value
        self._last_replan_time: float | None = None
        self._last_cmd = Twist(0.0, 0.0)
        self._waypoint_index = -1
        self._collisions_count = 0
        self._stuck_events_count = 0

    def set_goal(self, x: float, y: float) -> None:
        """Set/override active goal in world meters."""
        self._manual_goal_xy = (float(x), float(y))
        self._goal_reached = False
        self._mode = MODE_PLANNING
        self._current_path_world = []
        self._current_waypoints = []
        self._path_sig = "EMPTY"
        self._waypoint_index = -1
        self._local_controller.reset(0)
        self._replanner.force_replan()

    def get_debug(self) -> dict:
        return {
            "mode": self._mode,
            "current_path_len": len(self._current_path_world),
            "waypoint_index": self._waypoint_index,
            "last_replan_time": self._last_replan_time,
            "emergency_stop": self._emergency_stop,
            "stuck_state": self._stuck_state,
        }

    def get_path_snapshot(self) -> dict:
        """Return lightweight copies of the current global path and waypoints."""
        return {
            "path_world": list(self._current_path_world),
            "waypoints": [(float(wp.x), float(wp.y)) for wp in self._current_waypoints],
            "waypoint_index": self._waypoint_index,
            "mode": self._mode,
        }

    def step(self, now_s: float) -> None:
        """Run one navigation tick and send a motor command."""
        now = float(now_s)
        self._step_token += 1

        pose = _coerce_pose2d(self._get_pose_cb())
        goal_xy = self._resolve_goal()
        self._active_goal_xy = goal_xy

        if goal_xy is None:
            self._set_idle(now, pose)
            return

        goal_pose = Pose2D(goal_xy[0], goal_xy[1], 0.0)
        self._handle_goal_change(goal_xy)

        path_world, planner_status = self._replanner.update(now_s=now)
        self._last_replan_time = self._replanner.last_plan_time
        self._update_path_and_waypoints(path_world)

        reached_goal = False
        if self._current_waypoints:
            base_cmd, _, reached_goal = self._local_controller.update(pose, self._current_waypoints)
            self._waypoint_index = min(
                self._local_controller.current_index,
                max(0, len(self._current_waypoints) - 1),
            )
        else:
            base_cmd = Twist(0.0, 0.0)
            self._waypoint_index = -1

        if reached_goal:
            self._goal_reached = True
            final_cmd = Twist(0.0, 0.0)
            self._mode = MODE_GOAL_REACHED
            self._emergency_stop = False
            self._stuck_state = RecoveryState.NORMAL.value
            self._stuck_recovery.reset()
            self._send_and_log(final_cmd, pose, goal_pose, planner_status)
            return

        lidar_ranges = self._get_lidar_cb()
        adjusted_cmd, emergency_stop = adjust_twist_with_avoidance(
            current_cmd=base_cmd,
            config=self.config,
            lidar_ranges_m=lidar_ranges,
        )
        if emergency_stop and not self._emergency_stop:
            self._collisions_count += 1
        self._emergency_stop = emergency_stop

        # Preserve "intent to move" for stuck detection during emergency-stop
        # phases; otherwise low/zero adjusted commands can delay recovery entry.
        cmd_for_stuck = base_cmd if emergency_stop else adjusted_cmd
        override_cmd, stuck_state, _did_trigger_replan = self._stuck_recovery.update(
            pose=pose,
            cmd=cmd_for_stuck,
            now_s=now,
        )
        if self._stuck_state != RecoveryState.TURN_IN_PLACE.value and stuck_state == RecoveryState.TURN_IN_PLACE.value:
            self._stuck_events_count += 1
        self._stuck_state = stuck_state

        final_cmd = override_cmd if override_cmd is not None else adjusted_cmd
        self._mode = self._select_mode(planner_status, reached_goal=False)
        self._send_and_log(final_cmd, pose, goal_pose, planner_status)

    # Internal helpers.
    def _resolve_goal(self) -> tuple[float, float] | None:
        if self._manual_goal_xy is not None:
            return self._manual_goal_xy
        return _coerce_goal_xy(self._get_goal_cb())

    def _handle_goal_change(self, goal_xy: tuple[float, float]) -> None:
        if self._last_goal_xy is None:
            changed = True
        else:
            changed = hypot(goal_xy[0] - self._last_goal_xy[0], goal_xy[1] - self._last_goal_xy[1]) > 1e-6
        if changed:
            self._goal_reached = False
            self._current_path_world = []
            self._current_waypoints = []
            self._path_sig = "EMPTY"
            self._waypoint_index = -1
            self._local_controller.reset(0)
            self._replanner.force_replan()
            self._last_goal_xy = goal_xy

    def _set_idle(self, now_s: float, pose: Pose2D) -> None:
        self._mode = MODE_IDLE
        self._goal_reached = False
        self._current_path_world = []
        self._current_waypoints = []
        self._waypoint_index = -1
        self._path_sig = "EMPTY"
        self._emergency_stop = False
        self._stuck_state = RecoveryState.NORMAL.value
        self._local_controller.reset(0)
        self._stuck_recovery.reset()
        cmd = Twist(0.0, 0.0)
        self._send_cmd_cb(cmd)
        self._last_cmd = cmd
        self._append_log(pose, None, PlannerStatus(mode=MODE_IDLE, replans=self._replanner.status.replans))

    def _select_mode(self, planner_status: PlannerStatus, reached_goal: bool) -> str:
        if self._active_goal_xy is None:
            return MODE_IDLE
        if reached_goal or self._goal_reached:
            return MODE_GOAL_REACHED
        if self._stuck_state != RecoveryState.NORMAL.value:
            return MODE_RECOVERY
        if not self._current_waypoints:
            return MODE_PLANNING
        if planner_status.mode.startswith("REPLAN"):
            return MODE_PLANNING
        return MODE_FOLLOWING

    def _update_path_and_waypoints(self, path_world: Sequence[tuple[float, float]]) -> None:
        sig = _path_signature(path_world)
        if sig == self._path_sig:
            return
        self._path_sig = sig
        self._current_path_world = list(path_world)

        if not path_world:
            self._current_waypoints = []
            self._waypoint_index = -1
            self._local_controller.reset(0)
            return

        self._refresh_grid_cache()
        assert self._inflated_grid is not None
        try:
            waypoints = smooth_path_to_waypoints(path_world, self._inflated_grid, self.config)
            if not waypoints:
                waypoints = path_to_waypoints(path_world, self.config)
        except Exception:
            waypoints = path_to_waypoints(path_world, self.config)
        self._current_waypoints = waypoints
        self._local_controller.reset(0)
        self._waypoint_index = 0 if waypoints else -1

    def _send_and_log(self, cmd: Twist, pose: Pose2D, goal_pose: Pose2D, planner_status: PlannerStatus) -> None:
        self._send_cmd_cb(cmd)
        self._last_cmd = cmd
        self._append_log(pose, goal_pose, planner_status)

    def _append_log(self, pose: Pose2D, goal_pose: Pose2D | None, planner_status: PlannerStatus) -> None:
        if self.logger is None:
            return
        goal_for_log = goal_pose if goal_pose is not None else Pose2D(pose.x, pose.y, pose.theta)
        self.logger.append_row(
            pose=pose,
            goal=goal_for_log,
            mode=self._mode,
            replans=planner_status.replans,
            collisions_count=self._collisions_count,
            stuck_events_count=self._stuck_events_count,
        )

    def _replanner_get_current_grid(self) -> dict:
        self._refresh_grid_cache()
        assert self._raw_grid is not None and self._inflated_grid is not None
        return {
            "raw_grid": self._raw_grid,
            "inflated_grid": self._inflated_grid,
            "map_update_counter": self._latest_map_version,
        }

    def _replanner_get_pose(self) -> Pose2D:
        return _coerce_pose2d(self._get_pose_cb())

    def _replanner_get_goal(self) -> tuple[float, float]:
        if self._active_goal_xy is None:
            return 0.0, 0.0
        return self._active_goal_xy

    def _parse_grid_source(self, source: object) -> tuple[object, object]:
        if isinstance(source, dict):
            grid = None
            for key in ("grid", "raw_grid", "map"):
                if key in source and _is_grid_like(source[key]):
                    grid = source[key]
                    break
            if grid is None:
                raise TypeError("get_grid() dict output must include grid under `grid`/`raw_grid`/`map`")
            version = source.get("map_version", source.get("map_update_counter", source.get("version")))
            return grid, version

        if isinstance(source, (tuple, list)):
            items = list(source)
            if not items:
                raise TypeError("get_grid() returned an empty tuple/list")
            grid = items[0]
            if not _is_grid_like(grid):
                raise TypeError("First get_grid() tuple/list element must be grid-like")
            version = items[1] if len(items) >= 2 and not _is_grid_like(items[1]) else None
            return grid, version

        if _is_grid_like(source):
            version = getattr(source, "map_update_counter", getattr(source, "map_version", None))
            return source, version

        raise TypeError("Unsupported get_grid() return type")

    def _refresh_grid_cache(self) -> None:
        if self._grid_cache_token == self._step_token and self._inflated_grid is not None:
            return

        source = self._get_grid_cb()
        raw_obj, map_version = self._parse_grid_source(source)
        raw_grid = _coerce_grid_view(raw_obj)

        if map_version is None:
            map_version = _hash_grid_data(raw_grid.data)

        if map_version != self._cached_map_version or self._inflated_grid is None:
            if self.enable_inflation:
                inflated_data, cost_grid = inflate_from_config(raw_grid, self.config)
            else:
                inflated_data = np.asarray(raw_grid.data).copy()
                cost_grid = np.zeros(raw_grid.data.shape, dtype=np.float32)
            self._cost_grid = cost_grid
            self._inflated_grid = _GridView(
                width=raw_grid.width,
                height=raw_grid.height,
                resolution_m=raw_grid.resolution_m,
                origin_x_m=raw_grid.origin_x_m,
                origin_y_m=raw_grid.origin_y_m,
                data=np.asarray(inflated_data),
            )
            self._cached_map_version = map_version

        self._raw_grid = raw_grid
        self._latest_map_version = map_version
        self._grid_cache_token = self._step_token


def _demo() -> None:
    """Minimal fake harness for local execution/testing."""

    @dataclass(slots=True)
    class _FakeGrid:
        width: int
        height: int
        resolution_m: float
        origin_x_m: float
        origin_y_m: float
        data: np.ndarray

    cfg = NavigationConfig()
    cfg.grid.resolution_m = 0.10
    cfg.inflation_radius_m = 0.18
    cfg.planner.replanning_period_s = 0.6
    cfg.planner.waypoint_spacing_m = 0.35
    cfg.max_v = 0.40
    cfg.max_omega = 1.8
    cfg.goal_tolerance_m = 0.12
    cfg.stop_distance_m = 0.18

    width, height = 80, 60
    base = np.zeros((height, width), dtype=np.int16)
    # Vertical obstacle wall with a gap.
    base[5:50, 35] = 100
    base[27:33, 35] = 0

    grid = _FakeGrid(
        width=width,
        height=height,
        resolution_m=0.10,
        origin_x_m=0.0,
        origin_y_m=0.0,
        data=base,
    )
    map_version = {"value": 0}

    pose_state = {"x": 0.5, "y": 0.5, "theta": 0.0}
    goal_state = {"x": 6.8, "y": 4.8}
    cmd_state = {"cmd": Twist(0.0, 0.0)}
    sim_time = {"t": 0.0}

    def get_grid():
        return grid, map_version["value"]

    def get_pose():
        return Pose2D(pose_state["x"], pose_state["y"], pose_state["theta"])

    def get_goal():
        return goal_state["x"], goal_state["y"]

    def get_lidar():
        n = 181
        scan = np.full((n,), 5.0, dtype=np.float64)
        # Inject a temporary frontal obstacle in the middle of the run.
        if 6.0 < sim_time["t"] < 7.0:
            scan[n // 2] = 0.14
        return scan.tolist()

    def send_cmd(cmd: Twist):
        cmd_state["cmd"] = cmd

    stack = NavigationStack(
        get_grid=get_grid,
        get_pose=get_pose,
        get_goal=get_goal,
        get_lidar=get_lidar,
        send_cmd=send_cmd,
        config=cfg,
    )

    dt = 0.1
    print("NavigationStack fake harness:")
    for step in range(220):
        now_s = step * dt
        sim_time["t"] = now_s

        # Change map once to trigger replanning.
        if step == 90:
            grid.data[20:42, 35] = 100
            grid.data[38:44, 35] = 0
            map_version["value"] += 1

        stack.step(now_s)
        cmd = cmd_state["cmd"]

        # Fake differential-drive kinematics.
        pose_state["theta"] += cmd.omega * dt
        pose_state["x"] += cmd.v * cos(pose_state["theta"]) * dt
        pose_state["y"] += cmd.v * sin(pose_state["theta"]) * dt

        if step % 10 == 0:
            dbg = stack.get_debug()
            print(
                f"t={now_s:4.1f}s mode={dbg['mode']:>12} "
                f"path={dbg['current_path_len']:3d} wp_idx={dbg['waypoint_index']:3d} "
                f"estop={dbg['emergency_stop']} stuck={dbg['stuck_state']} "
                f"cmd=({cmd.v:.2f},{cmd.omega:.2f})"
            )

        if stack.get_debug()["mode"] == MODE_GOAL_REACHED:
            print(f"Goal reached at t={now_s:.1f}s")
            break


if __name__ == "__main__":
    _demo()
