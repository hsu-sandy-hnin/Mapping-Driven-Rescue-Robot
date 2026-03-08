"""Path simplification, spacing, and collision-aware smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, hypot
from typing import List, Sequence, Tuple

import numpy as np

from .grid_utils import bresenham_line, in_bounds, is_occupied, is_unknown, world_to_grid
from .types import Waypoint

WorldPoint = Tuple[float, float]
GridCell = Tuple[int, int]


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


def _cfg_get_limits(config: object | None, name: str, default):
    if config is None:
        return default
    if hasattr(config, name):
        return getattr(config, name)
    if hasattr(config, "limits") and hasattr(config.limits, name):
        return getattr(config.limits, name)
    return default


def _obstacle_threshold(config: object | None) -> int:
    if config is None:
        return 50
    if hasattr(config, "obstacle_threshold"):
        return int(getattr(config, "obstacle_threshold"))
    if hasattr(config, "grid") and hasattr(config.grid, "obstacle_threshold"):
        return int(getattr(config.grid, "obstacle_threshold"))
    return 50


@dataclass(slots=True)
class _ArrayGrid:
    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    data: np.ndarray


def _normalize_grid(grid_or_data: object, config: object | None) -> object:
    if isinstance(grid_or_data, np.ndarray):
        if grid_or_data.ndim != 2:
            raise ValueError("Grid ndarray must be 2D")
        height, width = grid_or_data.shape
        resolution_m = float(_cfg_get_grid(config, "resolution_m", 1.0))
        origin_x_m = float(_cfg_get_grid(config, "origin_x_m", 0.0))
        origin_y_m = float(_cfg_get_grid(config, "origin_y_m", 0.0))
        return _ArrayGrid(
            width=width,
            height=height,
            resolution_m=resolution_m,
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
            data=grid_or_data,
        )
    return grid_or_data


def _grid_data(grid: object) -> np.ndarray:
    if not hasattr(grid, "data"):
        raise AttributeError("Grid must provide data")
    data = np.asarray(getattr(grid, "data"))
    if data.ndim != 2:
        raise ValueError("Grid data must be 2D")
    return data


def _point_xy(point: Sequence[float] | object) -> WorldPoint:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(getattr(point, "x")), float(getattr(point, "y"))
    if isinstance(point, Sequence) and len(point) >= 2:
        return float(point[0]), float(point[1])
    raise TypeError("Point must expose x/y or be a sequence with at least two values")


def _dedupe_points(points: Sequence[WorldPoint], eps: float = 1e-9) -> List[WorldPoint]:
    if not points:
        return []
    deduped = [points[0]]
    for p in points[1:]:
        if hypot(p[0] - deduped[-1][0], p[1] - deduped[-1][1]) > eps:
            deduped.append(p)
    return deduped


def _cell_value(ix: int, iy: int, grid: object) -> int:
    data = _grid_data(grid)
    return int(data[iy, ix])


def _is_cell_blocked(
    ix: int,
    iy: int,
    grid: object,
    allow_unknown: bool,
    obstacle_threshold: int,
) -> bool:
    if not in_bounds(ix, iy, grid):
        return True
    if is_occupied(ix, iy, grid):
        return True
    if not allow_unknown and is_unknown(ix, iy, grid):
        return True
    # Supports non-default occupancy thresholds when needed.
    if _cell_value(ix, iy, grid) >= obstacle_threshold:
        return True
    return False


def _point_in_free_space(
    point: WorldPoint,
    grid: object,
    allow_unknown: bool,
    obstacle_threshold: int,
) -> bool:
    try:
        ix, iy = world_to_grid(point[0], point[1], grid)
    except ValueError:
        return False
    return not _is_cell_blocked(ix, iy, grid, allow_unknown, obstacle_threshold)


def _segment_collision_free(
    a: WorldPoint,
    b: WorldPoint,
    grid: object,
    allow_unknown: bool,
    obstacle_threshold: int,
) -> bool:
    try:
        start = world_to_grid(a[0], a[1], grid)
        end = world_to_grid(b[0], b[1], grid)
    except ValueError:
        return False
    for ix, iy in bresenham_line(start, end):
        if _is_cell_blocked(ix, iy, grid, allow_unknown, obstacle_threshold):
            return False
    return True


def simplify_path_world(
    path_world: Sequence[Sequence[float] | object],
    grid: object,
    allow_unknown: bool = False,
    obstacle_threshold: int = 50,
) -> List[WorldPoint]:
    """Greedy line-of-sight simplification on a world-coordinate path."""
    points = _dedupe_points([_point_xy(p) for p in path_world])
    if len(points) <= 2:
        return points

    simplified: List[WorldPoint] = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        candidate = len(points) - 1
        while candidate > anchor + 1:
            if _segment_collision_free(
                points[anchor],
                points[candidate],
                grid,
                allow_unknown,
                obstacle_threshold,
            ):
                break
            candidate -= 1
        simplified.append(points[candidate])
        anchor = candidate
    return simplified


def resample_path_world(
    path_world: Sequence[Sequence[float] | object],
    spacing_m: float,
) -> List[WorldPoint]:
    """Resample path points so adjacent samples are approximately `spacing_m` apart."""
    points = _dedupe_points([_point_xy(p) for p in path_world])
    if len(points) <= 1:
        return points
    if spacing_m <= 0.0:
        return points

    cumulative = [0.0]
    for i in range(1, len(points)):
        cumulative.append(
            cumulative[-1] + hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        )
    total_length = cumulative[-1]
    if total_length <= 1e-12:
        return [points[0]]

    samples: List[WorldPoint] = [points[0]]
    target = spacing_m
    seg = 1
    while target < total_length:
        while seg < len(cumulative) - 1 and cumulative[seg] < target:
            seg += 1
        prev_len = cumulative[seg - 1]
        next_len = cumulative[seg]
        if next_len <= prev_len:
            break
        t = (target - prev_len) / (next_len - prev_len)
        x0, y0 = points[seg - 1]
        x1, y1 = points[seg]
        samples.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        target += spacing_m

    if hypot(samples[-1][0] - points[-1][0], samples[-1][1] - points[-1][1]) > 1e-9:
        samples.append(points[-1])
    return samples


def smooth_path_world(
    path_world: Sequence[Sequence[float] | object],
    grid: object,
    passes: int = 3,
    alpha: float = 0.35,
    max_shift_m: float = 0.10,
    allow_unknown: bool = False,
    obstacle_threshold: int = 50,
) -> List[WorldPoint]:
    """Smooth path with a collision-checked Laplacian update.

    For each interior point, the candidate update is:
      p' = p + alpha * ((p_prev + p_next)/2 - p)
    If either adjacent segment collides after the update, the edit is rejected.
    """
    points = _dedupe_points([_point_xy(p) for p in path_world])
    if len(points) <= 2 or passes <= 0:
        return points

    alpha = float(np.clip(alpha, 0.0, 1.0))
    max_shift_m = max(0.0, float(max_shift_m))
    current = list(points)

    for _ in range(passes):
        changed = False
        for i in range(1, len(current) - 1):
            prev_pt = current[i - 1]
            curr_pt = current[i]
            next_pt = current[i + 1]

            target_x = 0.5 * (prev_pt[0] + next_pt[0])
            target_y = 0.5 * (prev_pt[1] + next_pt[1])
            cand_x = curr_pt[0] + alpha * (target_x - curr_pt[0])
            cand_y = curr_pt[1] + alpha * (target_y - curr_pt[1])

            dx = cand_x - curr_pt[0]
            dy = cand_y - curr_pt[1]
            d = hypot(dx, dy)
            if max_shift_m > 0.0 and d > max_shift_m and d > 1e-12:
                s = max_shift_m / d
                cand_x = curr_pt[0] + dx * s
                cand_y = curr_pt[1] + dy * s

            candidate = (cand_x, cand_y)
            if not _point_in_free_space(candidate, grid, allow_unknown, obstacle_threshold):
                continue
            if not _segment_collision_free(prev_pt, candidate, grid, allow_unknown, obstacle_threshold):
                continue
            if not _segment_collision_free(candidate, next_pt, grid, allow_unknown, obstacle_threshold):
                continue

            if hypot(candidate[0] - curr_pt[0], candidate[1] - curr_pt[1]) > 1e-9:
                current[i] = candidate
                changed = True

        if not changed:
            break
    return current


def _compute_headings(path_world: Sequence[WorldPoint]) -> List[float]:
    if not path_world:
        return []
    headings: List[float] = []
    for i in range(len(path_world)):
        if i < len(path_world) - 1:
            x0, y0 = path_world[i]
            x1, y1 = path_world[i + 1]
        elif len(path_world) >= 2:
            x0, y0 = path_world[i - 1]
            x1, y1 = path_world[i]
        else:
            headings.append(0.0)
            continue
        headings.append(float(np.arctan2(y1 - y0, x1 - x0)))
    return headings


def _compute_speed_hints(path_world: Sequence[WorldPoint], base_speed_mps: float) -> List[float]:
    if not path_world:
        return []
    n = len(path_world)
    base = max(0.01, float(base_speed_mps))
    min_speed = 0.35 * base
    hints: List[float] = []
    for i in range(n):
        if i == 0 or i == n - 1 or n < 3:
            hints.append(min_speed)
            continue
        x0, y0 = path_world[i - 1]
        x1, y1 = path_world[i]
        x2, y2 = path_world[i + 1]
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        n1 = hypot(v1x, v1y)
        n2 = hypot(v2x, v2y)
        if n1 <= 1e-12 or n2 <= 1e-12:
            hints.append(min_speed)
            continue
        c = np.clip((v1x * v2x + v1y * v2y) / (n1 * n2), -1.0, 1.0)
        turn = acos(float(c))
        factor = max(0.0, 1.0 - (turn / np.pi))
        hints.append(min_speed + (base - min_speed) * factor)
    return hints


def path_to_waypoints(
    path_world: Sequence[Sequence[float] | object],
    config: object | None = None,
) -> List[Waypoint]:
    """Convert world points to `Waypoint` list with heading and speed hints."""
    points = [_point_xy(p) for p in path_world]
    if not points:
        return []
    base_speed = float(_cfg_get_limits(config, "max_speed_mps", 0.35))
    headings = _compute_headings(points)
    speed_hints = _compute_speed_hints(points, base_speed)
    return [
        Waypoint(x=points[i][0], y=points[i][1], theta=headings[i], speed_hint=speed_hints[i])
        for i in range(len(points))
    ]


def smooth_path_to_waypoints(
    raw_path_world: Sequence[Sequence[float] | object],
    grid: object,
    config: object | None = None,
    apply_smoothing: bool = True,
) -> List[Waypoint]:
    """Full pipeline: LOS simplify -> spacing -> optional smoothing -> waypoints."""
    if not raw_path_world:
        return []

    grid = _normalize_grid(grid, config)
    allow_unknown = bool(_cfg_get(config, "allow_unknown", False))
    obstacle_threshold = _obstacle_threshold(config)
    spacing_m = float(_cfg_get(config, "waypoint_spacing_m", 0.25))
    smoothing_passes = int(_cfg_get(config, "smoothing_passes", 3))
    smoothing_alpha = float(_cfg_get(config, "smoothing_alpha", 0.35))
    smoothing_max_shift_m = float(
        _cfg_get(config, "smoothing_max_shift_m", max(0.05, 0.5 * spacing_m))
    )
    smoothing_enabled = bool(_cfg_get(config, "smoothing_enabled", True))

    simplified = simplify_path_world(
        raw_path_world,
        grid,
        allow_unknown=allow_unknown,
        obstacle_threshold=obstacle_threshold,
    )
    spaced = resample_path_world(simplified, spacing_m=spacing_m)
    if apply_smoothing and smoothing_enabled:
        smoothed = smooth_path_world(
            spaced,
            grid,
            passes=smoothing_passes,
            alpha=smoothing_alpha,
            max_shift_m=smoothing_max_shift_m,
            allow_unknown=allow_unknown,
            obstacle_threshold=obstacle_threshold,
        )
    else:
        smoothed = spaced

    # Safety fallback: if smoothing produced an invalid segment, keep the spaced path.
    valid = True
    for p in smoothed:
        if not _point_in_free_space(p, grid, allow_unknown, obstacle_threshold):
            valid = False
            break
    if valid:
        for i in range(1, len(smoothed)):
            if not _segment_collision_free(
                smoothed[i - 1],
                smoothed[i],
                grid,
                allow_unknown,
                obstacle_threshold,
            ):
                valid = False
                break
    final_path = smoothed if valid else spaced
    return path_to_waypoints(final_path, config=config)


@dataclass(slots=True)
class _DemoGrid:
    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    data: np.ndarray


def _grid_to_world(ix: int, iy: int, grid: _DemoGrid) -> WorldPoint:
    return (
        grid.origin_x_m + (ix + 0.5) * grid.resolution_m,
        grid.origin_y_m + (iy + 0.5) * grid.resolution_m,
    )


def _demo() -> None:
    from .config import NavigationConfig

    grid_data = np.zeros((18, 30), dtype=np.int16)
    # Inflated obstacle block in the middle.
    grid_data[7:12, 12:18] = 100
    grid = _DemoGrid(
        width=30,
        height=18,
        resolution_m=0.20,
        origin_x_m=0.0,
        origin_y_m=0.0,
        data=grid_data,
    )

    # Zig-zag path around the obstacle.
    raw_cells = [
        (2, 10),
        (4, 8),
        (6, 10),
        (8, 8),
        (10, 10),
        (12, 6),
        (14, 5),
        (16, 5),
        (18, 6),
        (20, 8),
        (22, 10),
        (24, 8),
        (26, 10),
    ]
    raw_path = [_grid_to_world(ix, iy, grid) for ix, iy in raw_cells]

    cfg = NavigationConfig()
    cfg.grid.resolution_m = grid.resolution_m
    cfg.planner.waypoint_spacing_m = 0.80
    cfg.planner.allow_unknown = False
    cfg.planner.smoothing_passes = 4  # Optional field read by `_cfg_get`.
    cfg.planner.smoothing_alpha = 0.40
    cfg.planner.smoothing_max_shift_m = 0.18

    simplified = simplify_path_world(raw_path, grid, allow_unknown=False, obstacle_threshold=50)
    spaced = resample_path_world(simplified, spacing_m=cfg.planner.waypoint_spacing_m)
    smoothed = smooth_path_world(
        spaced,
        grid,
        passes=cfg.planner.smoothing_passes,
        alpha=cfg.planner.smoothing_alpha,
        max_shift_m=cfg.planner.smoothing_max_shift_m,
        allow_unknown=False,
        obstacle_threshold=50,
    )
    waypoints = smooth_path_to_waypoints(raw_path, grid, cfg, apply_smoothing=True)

    # Acceptance checks: all waypoint points and segments must be collision free.
    for wp in waypoints:
        assert _point_in_free_space((wp.x, wp.y), grid, allow_unknown=False, obstacle_threshold=50)
    for i in range(1, len(waypoints)):
        a = (waypoints[i - 1].x, waypoints[i - 1].y)
        b = (waypoints[i].x, waypoints[i].y)
        assert _segment_collision_free(a, b, grid, allow_unknown=False, obstacle_threshold=50)

    print("Path smoothing demo:")
    print(f"raw points:        {len(raw_path)}")
    print(f"LOS simplified:    {len(simplified)}")
    print(f"resampled points:  {len(spaced)}")
    print(f"smoothed points:   {len(smoothed)}")
    print(f"output waypoints:  {len(waypoints)}")
    print("First 5 waypoints (x, y, speed_hint):")
    for wp in waypoints[:5]:
        print(f"  ({wp.x:.2f}, {wp.y:.2f}, {wp.speed_hint:.2f})")
    print("All waypoints validated in free space.")


if __name__ == "__main__":
    _demo()
