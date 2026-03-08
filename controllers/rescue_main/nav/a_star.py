"""A* global planner for occupancy grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import ceil, floor, hypot, sqrt
from typing import List, Sequence, Tuple

import numpy as np

from .grid_utils import bresenham_line, grid_to_world, world_to_grid
from .types import Pose2D

GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]

UNKNOWN = -1
OCCUPIED_ALT = 1
SQRT2 = sqrt(2.0)


def _cfg_get(config: object, name: str, default):
    if config is None:
        return default
    if hasattr(config, name):
        return getattr(config, name)
    if hasattr(config, "planner") and hasattr(config.planner, name):
        return getattr(config.planner, name)
    return default


def _cfg_get_grid(config: object, name: str, default):
    if config is None:
        return default
    if hasattr(config, name):
        return getattr(config, name)
    if hasattr(config, "grid") and hasattr(config.grid, name):
        return getattr(config.grid, name)
    return default


def _obstacle_threshold(config: object) -> int:
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


def _normalize_grid(grid: object, config: object | None) -> object:
    """Accept either grid-like object or ndarray + config metadata."""
    if isinstance(grid, np.ndarray):
        if grid.ndim != 2:
            raise ValueError("Grid ndarray must be 2D")
        height, width = grid.shape
        resolution_m = float(_cfg_get_grid(config, "resolution_m", 1.0))
        origin_x_m = float(_cfg_get_grid(config, "origin_x_m", 0.0))
        origin_y_m = float(_cfg_get_grid(config, "origin_y_m", 0.0))
        return _ArrayGrid(
            width=width,
            height=height,
            resolution_m=resolution_m,
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
            data=grid,
        )
    return grid


def _grid_data_array(grid_or_data: object) -> np.ndarray:
    if isinstance(grid_or_data, np.ndarray):
        data = grid_or_data
    elif hasattr(grid_or_data, "data"):
        data = np.asarray(getattr(grid_or_data, "data"))
    else:
        raise TypeError("Expected grid object with .data or a numpy array")
    if data.ndim != 2:
        raise ValueError("Grid data must be 2D")
    if data.dtype.kind not in ("i", "u"):
        data = data.astype(np.int16, copy=False)
    return data


def _grid_origin_xy(grid: object) -> Tuple[float, float]:
    if hasattr(grid, "origin_x_m") and hasattr(grid, "origin_y_m"):
        return float(getattr(grid, "origin_x_m")), float(getattr(grid, "origin_y_m"))
    if hasattr(grid, "origin"):
        origin = getattr(grid, "origin")
        if hasattr(origin, "x") and hasattr(origin, "y"):
            return float(getattr(origin, "x")), float(getattr(origin, "y"))
    raise AttributeError(
        "Grid must provide origin via origin_x_m/origin_y_m or origin.x/origin.y"
    )


def _pose_xy(pose: Pose2D | Sequence[float]) -> Tuple[float, float]:
    if hasattr(pose, "x") and hasattr(pose, "y"):
        return float(getattr(pose, "x")), float(getattr(pose, "y"))
    if isinstance(pose, Sequence) and len(pose) >= 2:
        return float(pose[0]), float(pose[1])
    raise TypeError("Pose must expose x,y or be a sequence with at least two values")


def _is_occupied(value: int, obstacle_threshold: int) -> bool:
    return value == OCCUPIED_ALT or value >= obstacle_threshold


def _is_unknown(value: int) -> bool:
    return value == UNKNOWN


def _is_traversable(value: int, allow_unknown: bool, obstacle_threshold: int) -> bool:
    if _is_occupied(value, obstacle_threshold):
        return False
    if _is_unknown(value) and not allow_unknown:
        return False
    return True


def _fallback_radius_cells(grid: object, config: object) -> float:
    if not hasattr(grid, "resolution_m"):
        raise AttributeError("Grid must provide resolution_m")
    resolution = float(getattr(grid, "resolution_m"))
    if resolution <= 0.0:
        raise ValueError("Grid resolution_m must be > 0")
    radius_m = float(_cfg_get(config, "start_goal_search_radius_m", 0.75))
    return max(0.0, radius_m / resolution)


def _to_grid_clamped(x_m: float, y_m: float, grid: object, width: int, height: int) -> GridCell:
    """Convert world->grid and clamp when outside map bounds."""
    try:
        return world_to_grid(x_m, y_m, grid)
    except ValueError:
        origin_x_m, origin_y_m = _grid_origin_xy(grid)
        resolution_m = float(getattr(grid, "resolution_m"))
        ix = int(floor((x_m - origin_x_m) / resolution_m))
        iy = int(floor((y_m - origin_y_m) / resolution_m))
        ix = min(max(ix, 0), width - 1)
        iy = min(max(iy, 0), height - 1)
        return ix, iy


def _nearest_traversable(
    start: GridCell,
    data: np.ndarray,
    allow_unknown: bool,
    obstacle_threshold: int,
    radius_cells: float,
) -> GridCell | None:
    sx, sy = start
    height, width = data.shape
    if 0 <= sx < width and 0 <= sy < height:
        if _is_traversable(int(data[sy, sx]), allow_unknown, obstacle_threshold):
            return sx, sy

    max_r = max(0, int(ceil(radius_cells)))
    r2_limit = radius_cells * radius_cells
    best: tuple[float, int, int, int] | None = None
    best_cell: GridCell | None = None

    x_min = max(0, sx - max_r)
    x_max = min(width - 1, sx + max_r)
    y_min = max(0, sy - max_r)
    y_max = min(height - 1, sy + max_r)

    for iy in range(y_min, y_max + 1):
        for ix in range(x_min, x_max + 1):
            dx = ix - sx
            dy = iy - sy
            d2 = float(dx * dx + dy * dy)
            if d2 > r2_limit:
                continue
            if not _is_traversable(int(data[iy, ix]), allow_unknown, obstacle_threshold):
                continue
            # Deterministic tie-break: euclidean, manhattan, row, col.
            key = (d2, abs(dx) + abs(dy), iy, ix)
            if best is None or key < best:
                best = key
                best_cell = (ix, iy)
    return best_cell


def _heuristic(ix: int, iy: int, gx: int, gy: int, allow_diagonal: bool) -> float:
    dx = abs(gx - ix)
    dy = abs(gy - iy)
    if allow_diagonal:
        return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)
    return float(dx + dy)


def _neighbor_steps(allow_diagonal: bool) -> List[Tuple[int, int, float]]:
    steps: List[Tuple[int, int, float]] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
    ]
    if allow_diagonal:
        steps.extend(
            [
                (1, 1, SQRT2),
                (1, -1, SQRT2),
                (-1, 1, SQRT2),
                (-1, -1, SQRT2),
            ]
        )
    return steps


def plan_a_star_grid(
    grid: object,
    start_pose: Pose2D | Sequence[float],
    goal_pose: Pose2D | Sequence[float],
    config: object | None = None,
) -> List[GridCell]:
    """Plan a path with A* and return grid cells [(ix, iy), ...].

    Returns an empty list when no path is found.
    """
    grid = _normalize_grid(grid, config)
    data = _grid_data_array(grid)
    height, width = data.shape

    allow_diagonal = bool(_cfg_get(config, "allow_diagonal", True))
    allow_unknown = bool(_cfg_get(config, "allow_unknown", True))
    unknown_penalty = max(0.0, float(_cfg_get(config, "unknown_penalty", 2.5)))
    heuristic_weight = max(0.0, float(_cfg_get(config, "heuristic_weight", 1.0)))
    obstacle_threshold = _obstacle_threshold(config)

    sx_m, sy_m = _pose_xy(start_pose)
    gx_m, gy_m = _pose_xy(goal_pose)
    start = _to_grid_clamped(sx_m, sy_m, grid, width, height)
    goal = _to_grid_clamped(gx_m, gy_m, grid, width, height)

    radius_cells = _fallback_radius_cells(grid, config)
    start = _nearest_traversable(start, data, allow_unknown, obstacle_threshold, radius_cells)
    goal = _nearest_traversable(goal, data, allow_unknown, obstacle_threshold, radius_cells)
    if start is None or goal is None:
        return []

    sx, sy = start
    gx, gy = goal
    if (sx, sy) == (gx, gy):
        return [(sx, sy)]

    g_score = np.full((height, width), np.inf, dtype=np.float64)
    parent_x = np.full((height, width), -1, dtype=np.int32)
    parent_y = np.full((height, width), -1, dtype=np.int32)

    steps = _neighbor_steps(allow_diagonal)
    open_heap: list[tuple[float, float, int, int, int]] = []
    push_id = 0

    g_score[sy, sx] = 0.0
    h0 = _heuristic(sx, sy, gx, gy, allow_diagonal)
    heappush(open_heap, (heuristic_weight * h0, 0.0, push_id, sx, sy))
    push_id += 1

    found = False
    while open_heap:
        _, g_curr, _, x, y = heappop(open_heap)
        if g_curr > g_score[y, x] + 1e-12:
            continue
        if (x, y) == (gx, gy):
            found = True
            break

        for dx, dy, move_cost in steps:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            v = int(data[ny, nx])
            if not _is_traversable(v, allow_unknown, obstacle_threshold):
                continue

            step_cost = move_cost + (unknown_penalty if _is_unknown(v) else 0.0)
            tentative_g = g_curr + step_cost
            if tentative_g + 1e-12 >= g_score[ny, nx]:
                continue

            g_score[ny, nx] = tentative_g
            parent_x[ny, nx] = x
            parent_y[ny, nx] = y
            h = _heuristic(nx, ny, gx, gy, allow_diagonal)
            f = tentative_g + heuristic_weight * h
            heappush(open_heap, (f, tentative_g, push_id, nx, ny))
            push_id += 1

    if not found:
        return []

    path_rev: List[GridCell] = [(gx, gy)]
    cx, cy = gx, gy
    while (cx, cy) != (sx, sy):
        px = int(parent_x[cy, cx])
        py = int(parent_y[cy, cx])
        if px < 0 or py < 0:
            return []
        path_rev.append((px, py))
        cx, cy = px, py
    path_rev.reverse()
    return path_rev


def grid_path_to_world(path_cells: Sequence[GridCell], grid: object) -> List[WorldPoint]:
    """Convert a grid-cell path to world coordinates at cell centers."""
    return [grid_to_world(ix, iy, grid) for ix, iy in path_cells]


def plan_a_star_world(
    grid: object,
    start_pose: Pose2D | Sequence[float],
    goal_pose: Pose2D | Sequence[float],
    config: object | None = None,
) -> List[WorldPoint]:
    """Plan a path and return world points [(x_m, y_m), ...]."""
    grid = _normalize_grid(grid, config)
    path_cells = plan_a_star_grid(grid, start_pose, goal_pose, config)
    return grid_path_to_world(path_cells, grid)


def _line_of_sight_free(
    a: GridCell,
    b: GridCell,
    data: np.ndarray,
    allow_unknown: bool,
    obstacle_threshold: int,
) -> bool:
    height, width = data.shape
    for ix, iy in bresenham_line(a, b):
        if ix < 0 or ix >= width or iy < 0 or iy >= height:
            return False
        v = int(data[iy, ix])
        if not _is_traversable(v, allow_unknown, obstacle_threshold):
            return False
    return True


def simplify_path_grid(
    path_cells: Sequence[GridCell],
    grid_or_data: object,
    allow_unknown: bool = True,
    obstacle_threshold: int = 50,
) -> List[GridCell]:
    """Simplify a grid path using Bresenham line-of-sight shortcutting."""
    if len(path_cells) <= 2:
        return list(path_cells)

    data = _grid_data_array(grid_or_data)
    simplified: List[GridCell] = [path_cells[0]]
    anchor = 0

    while anchor < len(path_cells) - 1:
        candidate = len(path_cells) - 1
        while candidate > anchor + 1:
            if _line_of_sight_free(
                path_cells[anchor],
                path_cells[candidate],
                data,
                allow_unknown,
                obstacle_threshold,
            ):
                break
            candidate -= 1
        simplified.append(path_cells[candidate])
        anchor = candidate
    return simplified


def path_length(path_world: Sequence[WorldPoint]) -> float:
    """Return total Euclidean length of a world-coordinate path."""
    if len(path_world) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path_world)):
        x0, y0 = path_world[i - 1]
        x1, y1 = path_world[i]
        total += hypot(x1 - x0, y1 - y0)
    return total


@dataclass(slots=True)
class _DemoGrid:
    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    data: np.ndarray


def _print_map(grid: _DemoGrid, path_cells: Sequence[GridCell], start: GridCell, goal: GridCell) -> None:
    canvas = np.full(grid.data.shape, ".", dtype="<U1")
    canvas[grid.data == UNKNOWN] = "?"
    canvas[(grid.data == OCCUPIED_ALT) | (grid.data >= 50)] = "#"
    for ix, iy in path_cells:
        canvas[iy, ix] = "o"
    sx, sy = start
    gx, gy = goal
    canvas[sy, sx] = "S"
    canvas[gy, gx] = "G"
    for y in range(grid.height):
        print(" ".join(canvas[y, :].tolist()))


def _demo() -> None:
    rng = np.random.default_rng(7)
    width, height = 40, 24
    data = np.zeros((height, width), dtype=np.int16)

    obstacle_mask = rng.random((height, width)) < 0.23
    unknown_mask = (rng.random((height, width)) < 0.08) & ~obstacle_mask
    data[obstacle_mask] = 100
    data[unknown_mask] = UNKNOWN

    start_cell = (1, 1)
    goal_cell = (width - 2, height - 2)

    # Carve a free corridor so the map remains random but path existence is guaranteed.
    corridor = bresenham_line(start_cell, goal_cell)
    for ix, iy in corridor:
        for ny in range(max(0, iy - 1), min(height, iy + 2)):
            for nx in range(max(0, ix - 1), min(width, ix + 2)):
                data[ny, nx] = 0

    grid = _DemoGrid(
        width=width,
        height=height,
        resolution_m=0.10,
        origin_x_m=0.0,
        origin_y_m=0.0,
        data=data,
    )

    @dataclass(slots=True)
    class _PlannerCfg:
        allow_diagonal: bool = True
        allow_unknown: bool = True
        unknown_penalty: float = 3.0
        heuristic_weight: float = 1.0
        start_goal_search_radius_m: float = 0.5

    @dataclass(slots=True)
    class _GridCfg:
        obstacle_threshold: int = 50

    @dataclass(slots=True)
    class _Config:
        planner: _PlannerCfg = field(default_factory=_PlannerCfg)
        grid: _GridCfg = field(default_factory=_GridCfg)

    cfg = _Config()

    sx_m, sy_m = grid_to_world(start_cell[0], start_cell[1], grid)
    gx_m, gy_m = grid_to_world(goal_cell[0], goal_cell[1], grid)
    start_pose = Pose2D(sx_m, sy_m, 0.0)
    goal_pose = Pose2D(gx_m, gy_m, 0.0)

    path_cells = plan_a_star_grid(grid, start_pose, goal_pose, cfg)
    if not path_cells:
        print("No path found.")
        return

    simplified = simplify_path_grid(
        path_cells,
        grid,
        allow_unknown=cfg.planner.allow_unknown,
        obstacle_threshold=cfg.grid.obstacle_threshold,
    )
    path_world = grid_path_to_world(simplified, grid)

    print(f"Path cells: {len(path_cells)}")
    print(f"Simplified cells: {len(simplified)}")
    print(f"Path length (m): {path_length(path_world):.2f}")
    print("Map preview:")
    _print_map(grid, simplified, path_cells[0], path_cells[-1])


if __name__ == "__main__":
    _demo()
