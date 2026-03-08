import numpy as np
import matplotlib.pyplot as plt

from frontier_exploration import FrontierExplorer

class FrontierVisualizer:

    def __init__(self, explorer):
        self.explorer = explorer
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6,6))

    def draw(self, grid, robot_pos, goal=None):
        self.ax.clear()

        # Base map
        color_map = {
            -1: [0.3, 0.3, 0.3],   # unknown
             0: [1.0, 1.0, 1.0],   # free
             1: [0.0, 0.0, 0.0]    # obstacle
        }

        img = np.zeros((grid.shape[0], grid.shape[1], 3))

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                img[r, c] = color_map[grid[r, c]]

        self.ax.imshow(img)

        # --- FRONTIERS ---
        frontiers = self.explorer.detect_frontiers(grid)
        for r,c in frontiers:
            self.ax.scatter(c, r, color='blue', s=30)

        # --- CLUSTERS ---
        clusters = self.explorer.cluster_frontiers(frontiers, grid.shape)
        scored = self.explorer.score_clusters(clusters, robot_pos, grid)

        for item in scored:
            cr, cc = item["centroid"]
            self.ax.scatter(cc, cr, color='yellow', s=120, marker='x')

        # --- HISTORY ---
        for pr,pc in self.explorer._goal_history:
            self.ax.scatter(pc, pr, color='pink', s=50)

        # --- ROBOT ---
        rr, rc = robot_pos
        self.ax.scatter(rc, rr, color='green', s=200)

        # --- GOAL ---
        if goal:
            gr, gc = goal
            self.ax.scatter(gc, gr, color='red', s=200)

        self.ax.set_title("Frontier Exploration Debug View")
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        plt.pause(0.4)

from demo import FrontierVisualizer
import time

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
    vis = FrontierVisualizer(explorer)

    for step in range(15):

        goal = explorer.select_next_goal(grid, robot_pos)
        print("STEP", step, "GOAL:", goal)

        vis.draw(grid, robot_pos, goal)

        if goal is None:
            break

        # simulate robot moving
        robot_pos = goal

        # simulate exploration (reveal around robot)
        r,c = robot_pos
        for dr in range(-1,2):
            for dc in range(-1,2):
                nr, nc = r+dr, c+dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr,nc] == -1:
                        grid[nr,nc] = 0

        time.sleep(0.3)

    input("Press Enter to exit...")