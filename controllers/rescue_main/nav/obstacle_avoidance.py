"""Reactive obstacle-avoidance layer for differential-drive commands."""

from __future__ import annotations

from math import pi
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from .types import Twist

SectorName = str
_ESCAPE_MODE = "none"  # none | side_follow | backup
_ESCAPE_TURN_SIGN = 0.0  # +1 left, -1 right
_ESCAPE_FRONT_CLEAR_HYST_M = 0.10
_ESCAPE_SIDE_HYST_M = 0.06


def _cfg_get(config: object | None, name: str, default):
    if config is None:
        return default
    if isinstance(config, dict):
        if name in config:
            return config[name]
        for section in ("controller", "planner", "limits", "avoidance"):
            if section in config and isinstance(config[section], dict) and name in config[section]:
                return config[section][name]
        return default
    if hasattr(config, name):
        return getattr(config, name)
    for section in ("controller", "planner", "limits", "avoidance"):
        if hasattr(config, section):
            sec = getattr(config, section)
            if isinstance(sec, dict) and name in sec:
                return sec[name]
            if hasattr(sec, name):
                return getattr(sec, name)
    return default


def _max_omega(config: object | None, fallback: float = 2.2) -> float:
    val = _cfg_get(config, "max_omega", None)
    if val is None:
        val = _cfg_get(config, "max_omega_radps", fallback)
    return max(0.0, float(val))


def _max_v(config: object | None, fallback: float = 0.45) -> float:
    val = _cfg_get(config, "max_v", None)
    if val is None:
        val = _cfg_get(config, "max_speed_mps", fallback)
    return max(0.0, float(val))


def _safety_distance(config: object | None) -> float:
    return max(0.01, float(_cfg_get(config, "safety_distance_m", 0.55)))


def _side_clearance(config: object | None) -> float:
    return max(0.01, float(_cfg_get(config, "side_clearance_m", 0.45)))


def _avoid_gain(config: object | None) -> float:
    return float(_cfg_get(config, "avoid_gain", 1.20))


def _stop_distance(config: object | None) -> float:
    return max(0.01, float(_cfg_get(config, "stop_distance_m", 0.18)))


def _angles_for_scan(num: int, angle_min_rad: float, angle_max_rad: float) -> np.ndarray:
    if num <= 0:
        return np.empty((0,), dtype=np.float64)
    if num == 1:
        return np.array([0.5 * (angle_min_rad + angle_max_rad)], dtype=np.float64)
    return np.linspace(angle_min_rad, angle_max_rad, num, dtype=np.float64)


def _points_to_ranges_angles(
    obstacle_points_rf: Sequence[Sequence[float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not obstacle_points_rf:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    pts = np.asarray(obstacle_points_rf, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("obstacle_points_rf must be Nx2 or NxM with x,y in first two columns")
    x = pts[:, 0]
    y = pts[:, 1]
    ranges = np.sqrt(x * x + y * y)
    angles = np.arctan2(y, x)
    valid = np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0.0)
    return ranges[valid], angles[valid]


def _concat_measurements(
    lidar_ranges_m: Sequence[float] | None,
    obstacle_points_rf: Sequence[Sequence[float]] | None,
    angle_min_rad: float,
    angle_max_rad: float,
    range_max_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    scan_ranges = np.empty((0,), dtype=np.float64)
    scan_angles = np.empty((0,), dtype=np.float64)
    if lidar_ranges_m is not None:
        raw = np.asarray(lidar_ranges_m, dtype=np.float64).reshape(-1)
        all_angles = _angles_for_scan(raw.size, angle_min_rad, angle_max_rad)
        valid = np.isfinite(raw) & (raw > 0.0)
        if range_max_m is not None and range_max_m > 0.0:
            valid &= raw <= float(range_max_m)
        scan_ranges = raw[valid]
        scan_angles = all_angles[valid]

    pt_ranges, pt_angles = _points_to_ranges_angles(obstacle_points_rf)

    if scan_ranges.size == 0 and pt_ranges.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    if scan_ranges.size == 0:
        return pt_ranges, pt_angles
    if pt_ranges.size == 0:
        return scan_ranges, scan_angles
    return (
        np.concatenate([scan_ranges, pt_ranges]),
        np.concatenate([scan_angles, pt_angles]),
    )


def _sector_mask(angles: np.ndarray, min_deg: float, max_deg: float) -> np.ndarray:
    min_rad = np.deg2rad(min_deg)
    max_rad = np.deg2rad(max_deg)
    return (angles >= min_rad) & (angles <= max_rad)


def segment_lidar_sectors(
    lidar_ranges_m: Sequence[float] | None = None,
    obstacle_points_rf: Sequence[Sequence[float]] | None = None,
    angle_min_rad: float = -pi,
    angle_max_rad: float = pi,
    range_max_m: float | None = None,
) -> Dict[SectorName, np.ndarray]:
    """Segment measurements into front/side sectors.

    Sector definitions (degrees):
    - `front`: [-20, +20]
    - `front_left`: (+20, +60]
    - `front_right`: [-60, -20)
    - `left`: (+60, +120]
    - `right`: [-120, -60)
    """
    ranges, angles = _concat_measurements(
        lidar_ranges_m=lidar_ranges_m,
        obstacle_points_rf=obstacle_points_rf,
        angle_min_rad=angle_min_rad,
        angle_max_rad=angle_max_rad,
        range_max_m=range_max_m,
    )
    if ranges.size == 0:
        return {
            "front": np.empty((0,), dtype=np.float64),
            "front_left": np.empty((0,), dtype=np.float64),
            "front_right": np.empty((0,), dtype=np.float64),
            "left": np.empty((0,), dtype=np.float64),
            "right": np.empty((0,), dtype=np.float64),
        }

    return {
        "front": ranges[_sector_mask(angles, -20.0, 20.0)],
        "front_left": ranges[_sector_mask(angles, 20.0, 60.0)],
        "front_right": ranges[_sector_mask(angles, -60.0, -20.0)],
        "left": ranges[_sector_mask(angles, 60.0, 120.0)],
        "right": ranges[_sector_mask(angles, -120.0, -60.0)],
    }


def _sector_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size > 0 else float("inf")


def _repulsive_pressure(values: np.ndarray, clearance_m: float) -> float:
    if values.size == 0:
        return 0.0
    clearance = max(1e-6, float(clearance_m))
    norm = np.clip((clearance - values) / clearance, 0.0, 1.0)
    close = norm[norm > 0.0]
    if close.size == 0:
        return 0.0
    return float(np.mean(close * close))


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def adjust_twist_with_avoidance(
    current_cmd: Twist,
    config: object | None = None,
    lidar_ranges_m: Sequence[float] | None = None,
    obstacle_points_rf: Sequence[Sequence[float]] | None = None,
    angle_min_rad: float = -pi,
    angle_max_rad: float = pi,
    range_max_m: float | None = None,
) -> tuple[Twist, bool]:
    """Adjust local-controller command with reactive obstacle avoidance.

    Returns `(adjusted_cmd, is_emergency_stop)`.
    """
    global _ESCAPE_MODE
    global _ESCAPE_TURN_SIGN

    all_ranges_raw, _ = _concat_measurements(
        lidar_ranges_m=lidar_ranges_m,
        obstacle_points_rf=obstacle_points_rf,
        angle_min_rad=angle_min_rad,
        angle_max_rad=angle_max_rad,
        range_max_m=range_max_m,
    )
    if all_ranges_raw.size == 0:
        return Twist(current_cmd.v, current_cmd.omega), False

    sectors = segment_lidar_sectors(
        lidar_ranges_m=lidar_ranges_m,
        obstacle_points_rf=obstacle_points_rf,
        angle_min_rad=angle_min_rad,
        angle_max_rad=angle_max_rad,
        range_max_m=range_max_m,
    )

    safety_distance = _safety_distance(config)
    side_clearance = _side_clearance(config)
    avoid_gain = _avoid_gain(config)
    stop_distance = _stop_distance(config)
    max_omega = _max_omega(config, fallback=max(abs(current_cmd.omega), 2.2))
    max_v = _max_v(config, fallback=max(abs(current_cmd.v), 0.45))

    left_pressure = _repulsive_pressure(
        np.concatenate([sectors["left"], sectors["front_left"]]),
        side_clearance,
    )
    right_pressure = _repulsive_pressure(
        np.concatenate([sectors["right"], sectors["front_right"]]),
        side_clearance,
    )
    left_min = min(_sector_min(sectors["left"]), _sector_min(sectors["front_left"]))
    right_min = min(_sector_min(sectors["right"]), _sector_min(sectors["front_right"]))
    angular_bias = avoid_gain * (right_pressure - left_pressure)

    front_min = min(
        _sector_min(sectors["front"]),
        _sector_min(sectors["front_left"]),
        _sector_min(sectors["front_right"]),
    )

    # Emergency close-obstacle branch.
    closest_range = min(front_min, float(np.min(all_ranges_raw)))
    if closest_range < stop_distance:
        # Default non-zero turn direction to avoid deadlock when left/right are perfectly symmetric.
        turn_sign = 1.0
        if abs(right_pressure - left_pressure) > 1e-6:
            turn_sign = 1.0 if right_pressure > left_pressure else -1.0
        else:
            fl = _sector_min(sectors["front_left"])
            fr = _sector_min(sectors["front_right"])
            if fl < fr:
                turn_sign = -1.0
            elif fr < fl:
                turn_sign = 1.0
        emergency_turn = turn_sign * max(0.35, 0.4 * max_omega)
        emergency_turn = _clamp(emergency_turn, -max_omega, max_omega)
        # Back-and-turn is more reliable than pure turn-in-place when the robot
        # is physically touching cylindrical obstacles.
        _ESCAPE_MODE = "backup"
        _ESCAPE_TURN_SIGN = turn_sign
        backup_v = -_clamp(0.32 * max_v, 0.08, 0.22)
        return Twist(v=backup_v, omega=emergency_turn), True

    # Front blocked: prefer changing path left/right if a side has space.
    side_follow_front_m = max(stop_distance + 0.08, 0.72 * safety_distance)
    side_space_needed_m = max(0.35, 0.90 * side_clearance)
    front_clear_threshold = side_follow_front_m + _ESCAPE_FRONT_CLEAR_HYST_M
    if _ESCAPE_MODE != "none" and front_min >= front_clear_threshold:
        _ESCAPE_MODE = "none"
        _ESCAPE_TURN_SIGN = 0.0

    if front_min < side_follow_front_m:
        turn_sign = 0.0
        left_ok = left_min > side_space_needed_m
        right_ok = right_min > side_space_needed_m

        # Keep previous side-follow direction while front is blocked to avoid
        # left-right oscillation near pillars/walls.
        if _ESCAPE_MODE == "side_follow" and abs(_ESCAPE_TURN_SIGN) > 0.5:
            if _ESCAPE_TURN_SIGN > 0.0 and left_min > (side_space_needed_m - _ESCAPE_SIDE_HYST_M):
                turn_sign = 1.0
            elif _ESCAPE_TURN_SIGN < 0.0 and right_min > (side_space_needed_m - _ESCAPE_SIDE_HYST_M):
                turn_sign = -1.0

        if turn_sign == 0.0:
            if left_ok and right_ok:
                turn_sign = 1.0 if left_min >= right_min else -1.0
            elif left_ok:
                turn_sign = 1.0
            elif right_ok:
                turn_sign = -1.0

        # Move in the chosen side direction while front is constrained.
        if turn_sign != 0.0:
            _ESCAPE_MODE = "side_follow"
            _ESCAPE_TURN_SIGN = turn_sign
            side_turn = turn_sign * max(0.28, 0.28 * max_omega)
            side_turn = _clamp(side_turn + 0.40 * angular_bias, -max_omega, max_omega)
            front_scale = _clamp(
                (front_min - stop_distance) / max(1e-6, side_follow_front_m - stop_distance),
                0.10,
                0.55,
            )
            side_v_cap = _clamp(0.45 * max_v, 0.10, 0.30)
            desired_v = max(0.0, float(current_cmd.v))
            side_v = _clamp(max(0.10, desired_v * front_scale), 0.10, side_v_cap)
            return Twist(v=side_v, omega=side_turn), False

        # No-space branch: cannot go left/right, so back-and-turn until space reappears.
        if _ESCAPE_MODE == "backup" and abs(_ESCAPE_TURN_SIGN) > 0.5:
            turn_sign = _ESCAPE_TURN_SIGN
        elif abs(left_min - right_min) > 1e-6:
            turn_sign = 1.0 if left_min > right_min else -1.0
        else:
            turn_sign = 1.0 if right_pressure < left_pressure else -1.0
        _ESCAPE_MODE = "backup"
        _ESCAPE_TURN_SIGN = turn_sign
        backup_turn = turn_sign * max(0.30, 0.32 * max_omega)
        backup_turn = _clamp(backup_turn, -max_omega, max_omega)
        backup_v = -_clamp(0.32 * max_v, 0.10, 0.22)
        return Twist(v=backup_v, omega=backup_turn), True

    # Reactive non-emergency branch.
    _ESCAPE_MODE = "none"
    _ESCAPE_TURN_SIGN = 0.0
    span = max(1e-6, safety_distance - stop_distance)
    front_slow = _clamp((front_min - stop_distance) / span, 0.0, 1.0)
    adjusted_v = current_cmd.v * front_slow
    adjusted_omega = _clamp(current_cmd.omega + angular_bias, -max_omega, max_omega)
    return Twist(v=adjusted_v, omega=adjusted_omega), False


def _build_scan_with_obstacle(
    num: int,
    obstacle_angle_deg: float | None,
    obstacle_range_m: float,
    default_range_m: float = 5.0,
    angle_min_rad: float = -pi / 2.0,
    angle_max_rad: float = pi / 2.0,
) -> np.ndarray:
    scan = np.full((num,), default_range_m, dtype=np.float64)
    if obstacle_angle_deg is None:
        return scan
    angle = np.deg2rad(obstacle_angle_deg)
    if num == 1:
        idx = 0
    else:
        ratio = (angle - angle_min_rad) / (angle_max_rad - angle_min_rad)
        idx = int(np.clip(round(ratio * (num - 1)), 0, num - 1))
    scan[idx] = obstacle_range_m
    return scan


def _demo() -> None:
    from .config import NavigationConfig

    cfg = NavigationConfig()
    cfg.stop_distance_m = 0.20
    cfg.safety_distance_m = 0.60
    cfg.side_clearance_m = 0.50
    cfg.avoid_gain = 1.6
    cfg.max_omega = 2.0

    base_cmd = Twist(v=0.40, omega=0.00)
    n = 181
    amin = -pi / 2.0
    amax = pi / 2.0

    # 1) Emergency escape case: very close obstacle in front.
    scan_stop = _build_scan_with_obstacle(
        num=n,
        obstacle_angle_deg=0.0,
        obstacle_range_m=0.12,
        angle_min_rad=amin,
        angle_max_rad=amax,
    )
    cmd_stop, emergency = adjust_twist_with_avoidance(
        current_cmd=base_cmd,
        config=cfg,
        lidar_ranges_m=scan_stop,
        angle_min_rad=amin,
        angle_max_rad=amax,
    )
    print("Scenario A (front obstacle very close):")
    print(f"  base={base_cmd} adjusted={cmd_stop} emergency={emergency}")
    assert emergency and cmd_stop.v < 0.0

    # 2) Obstacle on left: should bias turn to the right (negative omega).
    scan_left = _build_scan_with_obstacle(
        num=n,
        obstacle_angle_deg=40.0,
        obstacle_range_m=0.30,
        angle_min_rad=amin,
        angle_max_rad=amax,
    )
    cmd_left, emergency_left = adjust_twist_with_avoidance(
        current_cmd=base_cmd,
        config=cfg,
        lidar_ranges_m=scan_left,
        angle_min_rad=amin,
        angle_max_rad=amax,
    )
    print("Scenario B (obstacle on left):")
    print(f"  base={base_cmd} adjusted={cmd_left} emergency={emergency_left}")
    assert not emergency_left
    assert cmd_left.omega < 0.0

    # 3) Obstacle points in robot frame on right: should bias turn left.
    points_right = [(0.30, -0.20), (0.35, -0.24), (0.40, -0.18)]
    cmd_right, emergency_right = adjust_twist_with_avoidance(
        current_cmd=base_cmd,
        config=cfg,
        obstacle_points_rf=points_right,
    )
    print("Scenario C (obstacle points on right):")
    print(f"  base={base_cmd} adjusted={cmd_right} emergency={emergency_right}")
    assert not emergency_right
    assert cmd_right.omega > 0.0

    print("Obstacle avoidance demo passed.")


if __name__ == "__main__":
    _demo()
