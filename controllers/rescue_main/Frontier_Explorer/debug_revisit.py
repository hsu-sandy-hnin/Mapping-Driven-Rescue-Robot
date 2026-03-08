"""Debug revisit logic"""
import numpy as np
from frontier_exploration import FrontierExplorer

grid = np.full((20, 20), -1, dtype=int)
grid[5:15, 5:15] = 0
grid[7:9, 7:9] = 1
grid[11:13, 11:13] = 1
grid[8:12, 3:5] = 0
grid[8:12, 15:17] = 0
grid[3:5, 8:12] = 0
grid[15:17, 8:12] = 0

robot_pos = (10, 10)
explorer = FrontierExplorer(revisit_threshold=2.5)

print("Testing revisit logic:")
for step in range(5):
    goal = explorer.select_next_goal(grid, robot_pos)
    print(f"\nStep {step+1}:")
    print(f"  Selected goal: {goal}")
    print(f"  History: {list(explorer._goal_history)}")
    
    if goal:
        # Check if goal would be a revisit
        is_revisit = explorer._is_revisit(goal)
        print(f"  Is revisit: {is_revisit}")
        
        # Check distances to history
        for i, prev in enumerate(explorer._goal_history[:-1]):  # Exclude just-added goal
            dist = explorer.compute_distance(goal, prev)
            print(f"    Distance to history[{i}] {prev}: {dist:.2f}")
        
        robot_pos = goal
