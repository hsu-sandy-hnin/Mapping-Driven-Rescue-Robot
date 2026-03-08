"""Occupancy grid helper functions for navigation.

The utilities in this module are intentionally lightweight and numpy-friendly.
Expected grid fields:
- `width`: number of columns
- `height`: number of rows
- `resolution_m`: meters per grid cell
- `origin_x_m`, `origin_y_m`: world origin of grid cell (0, 0)
- `data`: occupancy array/list indexed as [iy, ix]

Data convention:
- `-1`: unknown
- `0`: free
- `100` or `1`: occupied (and values >= 50 are also treated as occupied)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterator, List, Tuple

import numpy as np

GridCell = Tuple[int, int]

UNKNOWN_VALUE = -1
FREE_VALUE = 0
OCCUPIED_VALUE_ALT = 1
OCCUPIED_THRESHOLD = 50


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


def _grid_resolution(grid: object) -> float:
    if not hasattr(grid, "resolution_m"):
        raise AttributeError("Grid must provide resolution_m")
    resolution_m = float(getattr(grid, "resolution_m"))
    if resolution_m <= 0.0:
        raise ValueError("Grid resolution_m must be > 0")
    return resolution_m


def _grid_size(grid: object) -> Tuple[int, int]:
    if not hasattr(grid, "width") or not hasattr(grid, "height"):
        raise AttributeError("Grid must provide width and height")
    width = int(getattr(grid, "width"))
    height = int(getattr(grid, "height"))
    if width <= 0 or height <= 0:
        raise ValueError("Grid width and height must be > 0")
    return width, height


def _grid_data(grid: object):
    if not hasattr(grid, "data"):
        raise AttributeError("Grid must provide data")
    return getattr(grid, "data")


def in_bounds(ix: int, iy: int, grid: object) -> bool:
    """Return True if (ix, iy) is inside the grid bounds."""
    width, height = _grid_size(grid)
    return 0 <= ix < width and 0 <= iy < height


def world_to_grid(x_m: float, y_m: float, grid: object) -> GridCell:
    """Convert world coordinates to grid indices with bounds validation.

    Returns:
        (ix, iy), where ix is the x/column index and iy is the y/row index.

    Raises:
        ValueError if converted cell is outside grid bounds.
    """
    origin_x_m, origin_y_m = _grid_origin_xy(grid)
    resolution_m = _grid_resolution(grid)
    ix = int(floor((x_m - origin_x_m) / resolution_m))
    iy = int(floor((y_m - origin_y_m) / resolution_m))
    if not in_bounds(ix, iy, grid):
        raise ValueError(f"World point ({x_m}, {y_m}) maps outside grid: ({ix}, {iy})")
    return ix, iy


def grid_to_world(ix: int, iy: int, grid: object) -> Tuple[float, float]:
    """Convert grid indices to world coordinates at cell center.

    Raises:
        ValueError if (ix, iy) is outside grid bounds.
    """
    if not in_bounds(ix, iy, grid):
        raise ValueError(f"Grid index outside bounds: ({ix}, {iy})")
    origin_x_m, origin_y_m = _grid_origin_xy(grid)
    resolution_m = _grid_resolution(grid)
    x_m = origin_x_m + (ix + 0.5) * resolution_m
    y_m = origin_y_m + (iy + 0.5) * resolution_m
    return x_m, y_m


def _cell_value(ix: int, iy: int, grid: object) -> int:
    if not in_bounds(ix, iy, grid):
        raise ValueError(f"Grid index outside bounds: ({ix}, {iy})")
    data = _grid_data(grid)
    if isinstance(data, np.ndarray):
        return int(data[iy, ix])
    return int(data[iy][ix])


def is_free(ix: int, iy: int, grid: object) -> bool:
    """Return True when occupancy value is free (0)."""
    return _cell_value(ix, iy, grid) == FREE_VALUE


def is_unknown(ix: int, iy: int, grid: object) -> bool:
    """Return True when occupancy value is unknown (-1)."""
    return _cell_value(ix, iy, grid) == UNKNOWN_VALUE


def is_occupied(ix: int, iy: int, grid: object) -> bool:
    """Return True when occupancy value represents an occupied cell."""
    value = _cell_value(ix, iy, grid)
    return value == OCCUPIED_VALUE_ALT or value >= OCCUPIED_THRESHOLD


def set_cell(ix: int, iy: int, value: int, grid: object) -> None:
    """Set a cell value in the occupancy data structure."""
    if not in_bounds(ix, iy, grid):
        raise ValueError(f"Grid index outside bounds: ({ix}, {iy})")
    data = _grid_data(grid)
    if isinstance(data, np.ndarray):
        data[iy, ix] = value
    else:
        data[iy][ix] = int(value)


def bresenham_line(start: GridCell, end: GridCell) -> List[GridCell]:
    """Return integer grid cells along line from start to end (inclusive)."""
    x0, y0 = start
    x1, y1 = end

    cells: List[GridCell] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


def raycast_occupied(
    start_ixiy: GridCell, end_ixiy: GridCell, grid: object
) -> GridCell | None:
    """Return first occupied cell hit along the discrete line, else None.

    Traversal stops if the ray exits grid bounds.
    """
    for ix, iy in bresenham_line(start_ixiy, end_ixiy):
        if not in_bounds(ix, iy, grid):
            break
        if is_occupied(ix, iy, grid):
            return ix, iy
    return None


def neighbors4(ix: int, iy: int, grid: object) -> Iterator[GridCell]:
    """Yield in-bounds 4-connected neighbors."""
    candidates = ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1))
    for nx, ny in candidates:
        if in_bounds(nx, ny, grid):
            yield nx, ny


def neighbors8(ix: int, iy: int, grid: object) -> Iterator[GridCell]:
    """Yield in-bounds 8-connected neighbors."""
    for ny in (iy - 1, iy, iy + 1):
        for nx in (ix - 1, ix, ix + 1):
            if nx == ix and ny == iy:
                continue
            if in_bounds(nx, ny, grid):
                yield nx, ny


@dataclass(slots=True)
class _DemoGrid:
    """Minimal grid type for self-checks."""

    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    data: np.ndarray


def _self_check() -> None:
    grid = _DemoGrid(
        width=5,
        height=4,
        resolution_m=0.5,
        origin_x_m=-1.0,
        origin_y_m=2.0,
        data=np.zeros((4, 5), dtype=np.int16),
    )

    # Conversion checks.
    ix, iy = world_to_grid(-0.75, 2.25, grid)
    assert (ix, iy) == (0, 0)
    x_m, y_m = grid_to_world(0, 0, grid)
    assert abs(x_m - (-0.75)) < 1e-9
    assert abs(y_m - 2.25) < 1e-9

    # Occupancy checks.
    set_cell(2, 1, 100, grid)
    assert is_occupied(2, 1, grid)
    set_cell(3, 1, -1, grid)
    assert is_unknown(3, 1, grid)
    assert is_free(0, 0, grid)

    # Bresenham checks.
    line = bresenham_line((0, 0), (3, 3))
    assert line[0] == (0, 0)
    assert line[-1] == (3, 3)
    assert len(line) == 4

    # Raycast and neighbors checks.
    hit = raycast_occupied((0, 1), (4, 1), grid)
    assert hit == (2, 1)
    n4 = set(neighbors4(1, 1, grid))
    assert n4 == {(0, 1), (2, 1), (1, 0), (1, 2)}
    n8 = set(neighbors8(1, 1, grid))
    assert len(n8) == 8

    print("grid_utils self-checks passed.")


if __name__ == "__main__":
    _self_check()
