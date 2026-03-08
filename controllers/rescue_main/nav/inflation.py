"""Obstacle inflation utilities for occupancy grids.

This module inflates occupied cells by a metric radius and computes an optional
soft cost field near obstacles. It uses a deterministic multi-source Dijkstra
expansion (queue-based) and does not depend on SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import sqrt
from typing import Tuple

import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED_ALT = 1


def _grid_size(grid: object) -> Tuple[int, int]:
    if not hasattr(grid, "width") or not hasattr(grid, "height"):
        raise AttributeError("Grid must provide width and height")
    width = int(getattr(grid, "width"))
    height = int(getattr(grid, "height"))
    if width <= 0 or height <= 0:
        raise ValueError("Grid width and height must be > 0")
    return width, height


def _grid_resolution(grid: object) -> float:
    if not hasattr(grid, "resolution_m"):
        raise AttributeError("Grid must provide resolution_m")
    resolution = float(getattr(grid, "resolution_m"))
    if resolution <= 0:
        raise ValueError("Grid resolution_m must be > 0")
    return resolution


def _grid_data_as_array(grid: object) -> np.ndarray:
    if not hasattr(grid, "data"):
        raise AttributeError("Grid must provide data")
    data = np.asarray(getattr(grid, "data"))
    if data.ndim != 2:
        raise ValueError("Grid data must be 2D")
    width, height = _grid_size(grid)
    if data.shape != (height, width):
        raise ValueError(
            f"Grid data shape {data.shape} does not match (height, width)=({height}, {width})"
        )
    if data.dtype.kind not in ("i", "u"):
        data = data.astype(np.int16, copy=False)
    return data


def _resolve_inflation_radius_m(config: object) -> float:
    if hasattr(config, "inflation_radius_m"):
        return float(getattr(config, "inflation_radius_m"))
    if hasattr(config, "grid") and hasattr(config.grid, "inflation_radius_m"):
        return float(getattr(config.grid, "inflation_radius_m"))
    raise AttributeError("Config must provide inflation_radius_m (directly or under config.grid)")


def _resolve_obstacle_threshold(config: object, default: int = 50) -> int:
    if hasattr(config, "obstacle_threshold"):
        return int(getattr(config, "obstacle_threshold"))
    if hasattr(config, "grid") and hasattr(config.grid, "obstacle_threshold"):
        return int(getattr(config.grid, "obstacle_threshold"))
    return default


def _occupied_write_value(data: np.ndarray, obstacle_threshold: int) -> int:
    # Preserve common map convention when possible.
    return 100 if np.any(data >= obstacle_threshold) else OCCUPIED_ALT


def _occupied_mask(data: np.ndarray, obstacle_threshold: int) -> np.ndarray:
    return (data == OCCUPIED_ALT) | (data >= obstacle_threshold)


def _compute_obstacle_distance_cells(
    occupied_mask: np.ndarray, max_radius_cells: float
) -> np.ndarray:
    height, width = occupied_mask.shape
    inf = np.float64(np.inf)
    dist = np.full((height, width), inf, dtype=np.float64)

    heap: list[tuple[float, int, int]] = []
    ys, xs = np.where(occupied_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        dist[y, x] = 0.0
        heappush(heap, (0.0, y, x))

    if not heap:
        return dist

    # 8-connected expansion with Euclidean step costs.
    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, sqrt(2.0)),
        (-1, 1, sqrt(2.0)),
        (1, -1, sqrt(2.0)),
        (1, 1, sqrt(2.0)),
    )
    max_expand = max(0.0, float(max_radius_cells))

    while heap:
        d, y, x = heappop(heap)
        if d != dist[y, x]:
            continue
        if d > max_expand:
            continue
        for dy, dx, w in neighbors:
            ny = y + dy
            nx = x + dx
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            nd = d + w
            if nd < dist[ny, nx] and nd <= max_expand:
                dist[ny, nx] = nd
                heappush(heap, (nd, ny, nx))
    return dist


def inflate_obstacles(
    grid: object,
    inflation_radius_m: float,
    obstacle_threshold: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Inflate occupied cells and return `(inflated_grid, cost_grid)`.

    - `inflated_grid`: same shape as `grid.data`, with inflated obstacles marked occupied.
    - `cost_grid`: float32 soft costs in [0, 1], decaying with distance from obstacles.
    """
    if inflation_radius_m < 0:
        raise ValueError("inflation_radius_m must be >= 0")

    data = _grid_data_as_array(grid)
    resolution_m = _grid_resolution(grid)
    radius_cells = float(inflation_radius_m) / resolution_m

    occupied_mask = _occupied_mask(data, obstacle_threshold)
    unknown_mask = data == UNKNOWN

    dist_cells = _compute_obstacle_distance_cells(occupied_mask, radius_cells)
    within_radius = dist_cells <= radius_cells

    inflated = data.copy()
    if occupied_mask.any():
        write_value = _occupied_write_value(data, obstacle_threshold)
        to_inflate = within_radius & ~occupied_mask
        if np.any(to_inflate):
            # Conservative safety rule:
            # Unknown cells normally stay unknown, but if obstacle inflation reaches
            # an unknown cell we mark it occupied to avoid planning through uncertainty.
            inflated[to_inflate] = write_value

    cost = np.zeros(data.shape, dtype=np.float32)
    if occupied_mask.any():
        if radius_cells <= 0.0:
            cost[occupied_mask] = 1.0
        else:
            # Linear decay from obstacle surface to inflation boundary.
            norm = np.clip(1.0 - (dist_cells / radius_cells), 0.0, 1.0)
            cost = norm.astype(np.float32, copy=False)
            cost[~within_radius] = 0.0
            cost[occupied_mask] = 1.0

    # Keep unknown cells unchanged unless explicitly overridden by inflation.
    untouched_unknown = unknown_mask & ~within_radius
    inflated[untouched_unknown] = UNKNOWN

    return inflated, cost


def inflate_from_config(grid: object, config: object) -> tuple[np.ndarray, np.ndarray]:
    """Inflate occupancy grid using `config.inflation_radius_m` (+ optional threshold)."""
    inflation_radius_m = _resolve_inflation_radius_m(config)
    obstacle_threshold = _resolve_obstacle_threshold(config)
    return inflate_obstacles(
        grid=grid,
        inflation_radius_m=inflation_radius_m,
        obstacle_threshold=obstacle_threshold,
    )


@dataclass(slots=True)
class _DemoGrid:
    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    data: np.ndarray


def _print_map(data: np.ndarray) -> None:
    symbols = {UNKNOWN: "?", FREE: ".", OCCUPIED_ALT: "#", 100: "#"}
    for y in range(data.shape[0]):
        row = []
        for x in range(data.shape[1]):
            v = int(data[y, x])
            row.append(symbols.get(v, "#"))
        print(" ".join(row))


def _demo() -> None:
    base = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, 0, 0],
            [0, 0, -1, 100, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    grid = _DemoGrid(
        width=7,
        height=5,
        resolution_m=0.1,
        origin_x_m=0.0,
        origin_y_m=0.0,
        data=base,
    )

    inflated, cost = inflate_obstacles(grid, inflation_radius_m=0.21)

    print("Before:")
    _print_map(base)
    print("\nAfter Inflation:")
    _print_map(inflated)
    print("\nCost Grid (rounded):")
    print(np.round(cost, 2))


if __name__ == "__main__":
    _demo()
