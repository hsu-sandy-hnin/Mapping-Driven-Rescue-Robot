import numpy as np
from frontier_exploration import FrontierExplorer

grid = np.array([
    [-1,-1,-1,-1,-1],
    [-1, 0, 0, 0,-1],
    [-1, 0, 1, 0,-1],
    [-1, 0, 0, 0,-1],
    [-1,-1,-1,-1,-1],
])

robot_pos = (2,2)

explorer = FrontierExplorer()

for step in range(5):
    goal = explorer.select_next_goal(grid, robot_pos)
    print("Step", step, "Goal:", goal)

    # simulate exploration
    if goal:
        grid[goal] = 0