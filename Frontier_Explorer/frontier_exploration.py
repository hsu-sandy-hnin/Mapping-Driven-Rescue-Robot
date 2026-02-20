import math
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

GridIndex = Tuple[int, int]


class FrontierExplorer:
    def __init__(
        self,
        w_size: float = 1.0,
        w_dist: float = 1.5,
        w_safety: float = 2.0,
        revisit_threshold: float = 3.0,
        history_limit: int = 50,
        safety_radius: int = 1,
    ) -> None:
        self.w_size = float(w_size)
        self.w_dist = float(w_dist)
        self.w_safety = float(w_safety)
        self.revisit_threshold = float(revisit_threshold)
        self.history_limit = int(history_limit)
        self.safety_radius = int(safety_radius)
        self._goal_history: Deque[GridIndex] = deque(maxlen=self.history_limit)

    def select_next_goal(
        self, grid: np.ndarray, robot_position: GridIndex
    ) -> Optional[GridIndex]:
        if grid.size == 0:
            return None

        frontiers = self.detect_frontiers(grid)
        if not frontiers:
            return None

        clusters = self.cluster_frontiers(frontiers, grid.shape)
        if not clusters:
            return None

        scored = self.score_clusters(clusters, robot_position, grid)
        if not scored:
            return None

        # Sort by score descending
        scored.sort(key=lambda item: item["score"], reverse=True)

        # Try each cluster in order of score
        for item in scored:
            cluster_cells = item["cells"]
            
            # Find non-revisit cells first
            non_revisit_cells = []
            revisit_cells = []
            
            for cell in cluster_cells:
                dist = self.compute_distance(robot_position, cell)
                if self._is_revisit(cell):
                    revisit_cells.append((cell, dist))
                else:
                    non_revisit_cells.append((cell, dist))
            
            # Prefer non-revisit cells, sorted by distance
            if non_revisit_cells:
                non_revisit_cells.sort(key=lambda x: x[1])  # Sort by distance
                best_cell = non_revisit_cells[0][0]
                self._record_goal(best_cell)
                return best_cell
            
            # If all cells in this cluster are revisits, try next cluster
            # (don't select from this cluster)
            continue
        
        # If all clusters had only revisit cells, we need to select something
        # Pick the best cell from the best cluster (even if it's a revisit)
        # This prevents getting stuck when exploration is nearly complete
        if scored:
            best_item = scored[0]
            cluster_cells = best_item["cells"]
            if cluster_cells:
                # Select closest cell to robot
                best_cell = min(cluster_cells, 
                              key=lambda c: self.compute_distance(robot_position, c))
                self._record_goal(best_cell)
                return best_cell
        
        return None

    def detect_frontiers(self, grid: np.ndarray) -> Set[GridIndex]:
        rows, cols = grid.shape
        frontiers: Set[GridIndex] = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    continue
                for nr, nc in self.get_neighbors((r, c), (rows, cols)):
                    if grid[nr, nc] == -1:
                        frontiers.add((r, c))
                        break
        return frontiers

    def cluster_frontiers(
        self, frontiers: Set[GridIndex], shape: Tuple[int, int]
    ) -> List[List[GridIndex]]:
        clusters: List[List[GridIndex]] = []
        visited: Set[GridIndex] = set()

        for cell in frontiers:
            if cell in visited:
                continue
            cluster: List[GridIndex] = []
            queue: Deque[GridIndex] = deque([cell])
            visited.add(cell)

            while queue:
                current = queue.popleft()
                cluster.append(current)
                for neighbor in self.get_neighbors(current, shape):
                    if neighbor in frontiers and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if cluster:
                clusters.append(cluster)

        return clusters

    def score_clusters(
        self, clusters: List[List[GridIndex]], robot_position: GridIndex, grid: Optional[np.ndarray] = None
    ) -> List[Dict[str, object]]:
        """
        Score clusters based on size, distance, and safety.
        Uses a normalized scoring formula: score = (size * w_size + safety * w_safety) / (1 + distance * w_dist)
        This ensures positive scores and balances information gain vs travel cost.
        """
        scored: List[Dict[str, object]] = []
        for cluster in clusters:
            size = len(cluster)
            centroid = self.compute_centroid(cluster)
            distance = self.compute_distance(robot_position, centroid)
            
            # Compute safety score (distance to nearest obstacle)
            safety = self.compute_safety(centroid, grid) if grid is not None else 1.0
            
            # Normalized scoring: information gain divided by travel cost
            # This ensures positive scores and balances exploration vs distance
            # Formula: (information_gain) / (1 + travel_cost)
            information_gain = self.w_size * size + self.w_safety * max(safety, 0.1)  # Ensure positive
            travel_cost = 1.0 + self.w_dist * distance
            score = information_gain / travel_cost
            
            # Additional penalty for revisits (reduces score significantly)
            if self._is_revisit(centroid):
                score *= 0.1  # Reduce score by 90% for revisits

            scored.append(
                {
                    "cells": cluster,
                    "size": size,
                    "centroid": centroid,
                    "distance": distance,
                    "safety": safety,
                    "score": score,
                }
            )
        return scored

    def compute_centroid(self, cluster: List[GridIndex]) -> GridIndex:
        if not cluster:
            return (0, 0)
        row_sum = sum(cell[0] for cell in cluster)
        col_sum = sum(cell[1] for cell in cluster)
        r = int(round(row_sum / len(cluster)))
        c = int(round(col_sum / len(cluster)))
        return (r, c)

    def compute_distance(self, a: GridIndex, b: GridIndex) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def compute_safety(self, position: GridIndex, grid: np.ndarray) -> float:
        """
        Compute safety score for a position based on proximity to obstacles.
        Returns a positive value for safe positions (far from obstacles),
        and a negative value for unsafe positions (near obstacles).
        
        Args:
            position: Grid position to evaluate
            grid: Occupancy grid (0=free, 1=occupied, -1=unknown)
        
        Returns:
            Safety score: higher is safer, lower/negative is unsafe
        """
        if grid is None:
            return 1.0
        
        r, c = position
        rows, cols = grid.shape
        
        # Check neighbors within safety radius for obstacles
        min_dist_to_obstacle = float('inf')
        
        for dr in range(-self.safety_radius, self.safety_radius + 1):
            for dc in range(-self.safety_radius, self.safety_radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] == 1:  # Obstacle found
                        dist = math.hypot(dr, dc)
                        min_dist_to_obstacle = min(min_dist_to_obstacle, dist)
        
        # If no obstacle found nearby, return positive safety score
        if min_dist_to_obstacle == float('inf'):
            return 1.0
        
        # Penalize positions close to obstacles
        # Safety score decreases as distance to obstacle decreases
        # Returns negative values for very unsafe positions
        safety_score = min_dist_to_obstacle - 1.0
        
        return safety_score

    def get_neighbors(
        self, cell: GridIndex, shape: Tuple[int, int]
    ) -> Iterable[GridIndex]:
        r, c = cell
        rows, cols = shape
        if r > 0:
            yield (r - 1, c)
        if r + 1 < rows:
            yield (r + 1, c)
        if c > 0:
            yield (r, c - 1)
        if c + 1 < cols:
            yield (r, c + 1)

    def _is_revisit(self, goal: GridIndex) -> bool:
        if not self._goal_history:
            return False
        for prev in self._goal_history:
            if self.compute_distance(goal, prev) < self.revisit_threshold:
                return True
        return False

    def _record_goal(self, goal: GridIndex) -> None:
        self._goal_history.append(goal)


if __name__ == "__main__":
    grid = np.array(
        [
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, -1, 1, -1],
            [-1, 0, 1, 0, -1, 1, -1],
            [-1, 0, 0, 0, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    robot_pos = (3, 3)

    explorer = FrontierExplorer()
    frontiers = explorer.detect_frontiers(grid)
    print("Detected frontiers:", sorted(frontiers))
    goal = explorer.select_next_goal(grid, robot_pos)
    print("Selected goal:", goal)
