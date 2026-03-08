"""Local waypoint-following controller for differential-drive robots."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from typing import List, Sequence, Tuple

from .types import Pose2D, Twist, Waypoint

WorldPoint = Tuple[float, float]


def _cfg_get(config: object | None, name: str, default):
    if config is None:
        return default
    if isinstance(config, dict) and name in config:
        return config[name]
    if hasattr(config, name):
        return getattr(config, name)
    for section in ("controller", "planner", "limits"):
        if hasattr(config, section):
            sec = getattr(config, section)
            if isinstance(sec, dict) and name in sec:
                return sec[name]
            if hasattr(sec, name):
                return getattr(sec, name)
    return default


def _max_v(config: object | None) -> float:
    v = _cfg_get(config, "max_v", None)
    if v is None:
        v = _cfg_get(config, "max_speed_mps", 0.45)
    return max(0.0, float(v))


def _max_omega(config: object | None) -> float:
    omega = _cfg_get(config, "max_omega", None)
    if omega is None:
        omega = _cfg_get(config, "max_omega_radps", 2.2)
    return max(0.0, float(omega))


def _goal_tolerance(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "goal_tolerance_m", 0.15)))


def _lookahead(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "lookahead_m", 0.50)))


def _slow_down_radius(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "slow_down_radius_m", 0.70)))


def _heading_kp(config: object | None) -> float:
    # Reuse existing controller gain naming when available.
    if _cfg_get(config, "heading_kp", None) is not None:
        return float(_cfg_get(config, "heading_kp", 2.4))
    return float(_cfg_get(config, "k_alpha", 2.4))


def _wrap_to_pi(angle_rad: float) -> float:
    while angle_rad > pi:
        angle_rad -= 2.0 * pi
    while angle_rad < -pi:
        angle_rad += 2.0 * pi
    return angle_rad


def _point_xy(point: Waypoint | Sequence[float]) -> WorldPoint:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(getattr(point, "x")), float(getattr(point, "y"))
    if isinstance(point, Sequence) and len(point) >= 2:
        return float(point[0]), float(point[1])
    raise TypeError("Waypoint must expose x/y or be a sequence with at least two values")


def _normalize_waypoints(waypoints: Sequence[Waypoint | Sequence[float]]) -> List[WorldPoint]:
    return [_point_xy(wp) for wp in waypoints]


def _interp_point(a: WorldPoint, b: WorldPoint, t: float) -> WorldPoint:
    t = min(1.0, max(0.0, t))
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _project_point_to_segment(px: float, py: float, a: WorldPoint, b: WorldPoint) -> tuple[float, WorldPoint, float]:
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        q = a
        return 0.0, q, hypot(px - q[0], py - q[1])
    t = ((px - a[0]) * abx + (py - a[1]) * aby) / ab2
    t = min(1.0, max(0.0, t))
    q = (a[0] + t * abx, a[1] + t * aby)
    return t, q, hypot(px - q[0], py - q[1])


def select_lookahead_target(
    pose: Pose2D,
    waypoints: Sequence[Waypoint | Sequence[float]],
    lookahead_m: float,
    start_index: int = 0,
) -> tuple[WorldPoint, int]:
    """Select lookahead target from path and return `(target_xy, target_index)`."""
    points = _normalize_waypoints(waypoints)
    if not points:
        raise ValueError("Waypoints list is empty")

    n = len(points)
    start_index = min(max(0, int(start_index)), n - 1)
    if start_index >= n - 1:
        return points[-1], n - 1

    # Precompute cumulative arclength along polyline.
    arc = [0.0] * n
    for i in range(1, n):
        arc[i] = arc[i - 1] + hypot(
            points[i][0] - points[i - 1][0],
            points[i][1] - points[i - 1][1],
        )

    px, py = pose.x, pose.y
    best_seg = start_index
    best_t = 0.0
    best_dist = float("inf")
    for i in range(start_index, n - 1):
        t, _, d = _project_point_to_segment(px, py, points[i], points[i + 1])
        key = (d, i)
        if key < (best_dist, best_seg):
            best_dist = d
            best_seg = i
            best_t = t

    lookahead_m = max(0.0, float(lookahead_m))
    s_on_path = arc[best_seg] + best_t * (arc[best_seg + 1] - arc[best_seg])
    target_s = s_on_path + lookahead_m
    if target_s >= arc[-1]:
        return points[-1], n - 1

    for i in range(best_seg, n - 1):
        seg_len = arc[i + 1] - arc[i]
        if seg_len <= 1e-12:
            continue
        if target_s <= arc[i + 1]:
            t = (target_s - arc[i]) / seg_len
            return _interp_point(points[i], points[i + 1], t), i + 1

    return points[-1], n - 1


def compute_cmd(
    pose: Pose2D,
    target: Waypoint | Sequence[float],
    config: object | None = None,
    distance_to_goal_m: float | None = None,
) -> Twist:
    """Compute differential-drive command toward a target point."""
    tx, ty = _point_xy(target)
    dx = tx - pose.x
    dy = ty - pose.y
    target_heading = atan2(dy, dx)
    heading_error = _wrap_to_pi(target_heading - pose.theta)

    max_v = _max_v(config)
    max_omega = _max_omega(config)
    kp = _heading_kp(config)
    slow_radius = _slow_down_radius(config)

    # Linear speed attenuation:
    # 1) large heading errors -> slow down to avoid oscillation
    # 2) near-goal slowdown inside configured radius
    heading_scale = max(0.0, cos(heading_error))
    if abs(heading_error) > 0.85 * pi:
        heading_scale = 0.0

    if distance_to_goal_m is None:
        distance_to_goal_m = hypot(dx, dy)
    if slow_radius > 1e-9:
        near_goal_scale = min(1.0, max(0.0, distance_to_goal_m / slow_radius))
    else:
        near_goal_scale = 1.0

    v = max_v * heading_scale * near_goal_scale
    omega = max(-max_omega, min(max_omega, kp * heading_error))
    return Twist(v=v, omega=omega)


def detect_goal_reached(
    pose: Pose2D,
    final_goal: Waypoint | Sequence[float],
    goal_tolerance_m: float,
) -> bool:
    gx, gy = _point_xy(final_goal)
    return hypot(gx - pose.x, gy - pose.y) <= max(0.0, float(goal_tolerance_m))


def controller_step(
    pose: Pose2D,
    waypoints: Sequence[Waypoint | Sequence[float]],
    config: object | None = None,
    current_index: int = 0,
) -> tuple[Twist, int, bool]:
    """Single local-control update.

    Returns:
      `(cmd, reached_waypoint_index, reached_goal_bool)`
    """
    points = _normalize_waypoints(waypoints)
    if not points:
        return Twist(0.0, 0.0), -1, True

    goal_tol = _goal_tolerance(config)
    current_index = min(max(0, int(current_index)), len(points) - 1)
    reached_idx = current_index - 1

    # Consume waypoints reached by distance, and waypoints passed along segment direction.
    while current_index < len(points):
        wx, wy = points[current_index]
        if hypot(wx - pose.x, wy - pose.y) <= goal_tol:
            reached_idx = current_index
            current_index += 1
            continue
        if current_index < len(points) - 1:
            nx, ny = points[current_index + 1]
            sx = nx - wx
            sy = ny - wy
            seg2 = sx * sx + sy * sy
            if seg2 > 1e-12:
                proj = ((pose.x - wx) * sx + (pose.y - wy) * sy) / seg2
                if proj > 1.0:
                    reached_idx = current_index
                    current_index += 1
                    continue
        break

    final_goal = points[-1]
    reached_goal = current_index >= len(points) or detect_goal_reached(pose, final_goal, goal_tol)
    if reached_goal:
        return Twist(0.0, 0.0), max(reached_idx, len(points) - 1), True

    target, _ = select_lookahead_target(
        pose=pose,
        waypoints=points,
        lookahead_m=_lookahead(config),
        start_index=current_index,
    )
    distance_to_goal = hypot(final_goal[0] - pose.x, final_goal[1] - pose.y)
    cmd = compute_cmd(pose, target, config=config, distance_to_goal_m=distance_to_goal)
    return cmd, reached_idx, False


@dataclass(slots=True)
class LocalController:
    """Stateful wrapper around `controller_step` that tracks waypoint progress."""

    config: object | None = None
    current_index: int = 0
    last_target_index: int = -1

    def reset(self, current_index: int = 0) -> None:
        self.current_index = max(0, int(current_index))
        self.last_target_index = -1

    def update(
        self,
        pose: Pose2D,
        waypoints: Sequence[Waypoint | Sequence[float]],
    ) -> tuple[Twist, int, bool]:
        points = _normalize_waypoints(waypoints)
        if not points:
            self.current_index = 0
            self.last_target_index = -1
            return Twist(0.0, 0.0), -1, True

        self.current_index = min(max(0, self.current_index), len(points) - 1)
        goal_tol = _goal_tolerance(self.config)

        reached_idx = self.current_index - 1
        while self.current_index < len(points):
            wx, wy = points[self.current_index]
            if hypot(wx - pose.x, wy - pose.y) <= goal_tol:
                reached_idx = self.current_index
                self.current_index += 1
                continue
            if self.current_index < len(points) - 1:
                nx, ny = points[self.current_index + 1]
                sx = nx - wx
                sy = ny - wy
                seg2 = sx * sx + sy * sy
                if seg2 > 1e-12:
                    proj = ((pose.x - wx) * sx + (pose.y - wy) * sy) / seg2
                    if proj > 1.0:
                        reached_idx = self.current_index
                        self.current_index += 1
                        continue
            break

        final_goal = points[-1]
        reached_goal = self.current_index >= len(points) or detect_goal_reached(
            pose, final_goal, goal_tol
        )
        if reached_goal:
            self.current_index = len(points)
            self.last_target_index = len(points) - 1
            return Twist(0.0, 0.0), max(reached_idx, len(points) - 1), True

        target, target_idx = select_lookahead_target(
            pose=pose,
            waypoints=points,
            lookahead_m=_lookahead(self.config),
            start_index=self.current_index,
        )
        self.last_target_index = target_idx
        distance_to_goal = hypot(final_goal[0] - pose.x, final_goal[1] - pose.y)
        cmd = compute_cmd(
            pose=pose,
            target=target,
            config=self.config,
            distance_to_goal_m=distance_to_goal,
        )
        return cmd, reached_idx, False


def _simulate_pose_step(pose: Pose2D, cmd: Twist, dt_s: float) -> Pose2D:
    theta = _wrap_to_pi(pose.theta + cmd.omega * dt_s)
    x = pose.x + cmd.v * cos(pose.theta) * dt_s
    y = pose.y + cmd.v * sin(pose.theta) * dt_s
    return Pose2D(x=x, y=y, theta=theta)


def _count_zero_crossings(values: Sequence[float]) -> int:
    prev_sign = 0
    crossings = 0
    for v in values:
        sign = 1 if v > 0.0 else (-1 if v < 0.0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            crossings += 1
        prev_sign = sign
    return crossings


def _demo() -> None:
    from .config import NavigationConfig

    cfg = NavigationConfig()
    cfg.max_v = 0.45
    cfg.max_omega = 1.8
    cfg.lookahead_m = 0.55
    cfg.goal_tolerance_m = 0.12
    cfg.slow_down_radius_m = 0.65
    cfg.controller.heading_kp = 2.2

    # Waypoint set with a long straight segment then a turn.
    waypoints = [
        (0.0, 0.0),
        (1.5, 0.0),
        (3.0, 0.0),
        (4.5, 0.0),
        (6.0, 0.0),
        (7.0, 0.8),
        (7.5, 1.6),
    ]

    pose = Pose2D(x=-0.4, y=-0.35, theta=0.35)
    controller = LocalController(cfg)

    dt_s = 0.1
    track: List[Pose2D] = [pose]
    reached = False
    for step in range(500):
        cmd, reached_wp_idx, reached_goal = controller.update(pose, waypoints)
        pose = _simulate_pose_step(pose, cmd, dt_s)
        track.append(pose)
        reached = reached_goal
        if step % 25 == 0 or reached_goal:
            print(
                f"step={step:03d} pose=({pose.x:.2f},{pose.y:.2f},{pose.theta:.2f}) "
                f"cmd=({cmd.v:.2f},{cmd.omega:.2f}) reached_wp={reached_wp_idx} goal={reached_goal}"
            )
        if reached_goal:
            break

    # Basic straight-path stability check: during the straight segment we expect
    # at most one lateral zero-crossing (converge without oscillating back/forth).
    straight_errors = [p.y for p in track if p.x <= 6.0]
    crossings = _count_zero_crossings(straight_errors)
    assert crossings <= 1, f"Too many lateral oscillations on straight path: {crossings}"
    assert reached, "Controller failed to reach final goal in demo"

    print(f"Simulation finished in {len(track) - 1} steps.")
    print(f"Straight-segment zero-crossings: {crossings}")
    print("Local controller demo passed.")


if __name__ == "__main__":
    _demo()
