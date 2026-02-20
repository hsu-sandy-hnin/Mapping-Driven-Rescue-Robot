"""
Test oscillation avoidance more thoroughly
"""
import numpy as np
from frontier_exploration import FrontierExplorer

def test_oscillation():
    """Test that we don't oscillate between similar goals"""
    print("=" * 70)
    print("OSCILLATION AVOIDANCE TEST")
    print("=" * 70)
    
    # Create a grid with multiple distinct frontier clusters
    grid = np.full((15, 15), -1, dtype=int)
    
    # Explored center area
    grid[5:10, 5:10] = 0
    
    # Four distinct frontier areas
    grid[2:4, 2:4] = 0  # Top-left
    grid[2:4, 11:13] = 0  # Top-right
    grid[11:13, 2:4] = 0  # Bottom-left
    grid[11:13, 11:13] = 0  # Bottom-right
    
    robot_pos = (7, 7)  # Center
    explorer = FrontierExplorer(revisit_threshold=2.5)
    
    print(f"Grid shape: {grid.shape}")
    print(f"Robot start position: {robot_pos}")
    
    frontiers = explorer.detect_frontiers(grid)
    print(f"\nDetected {len(frontiers)} frontier cells")
    
    clusters = explorer.cluster_frontiers(frontiers, grid.shape)
    print(f"Found {len(clusters)} clusters")
    
    goals = []
    print("\nGoal selection sequence:")
    print("-" * 70)
    
    for step in range(15):
        goal = explorer.select_next_goal(grid, robot_pos)
        if goal is None:
            print(f"Step {step+1}: No more goals")
            break
        
        goals.append(goal)
        
        # Check for oscillation
        if len(goals) >= 2:
            last_goal = goals[-1]
            prev_goal = goals[-2]
            dist = explorer.compute_distance(last_goal, prev_goal)
            
            # Check if we're oscillating (same goal or very close)
            if dist < 1.0:
                print(f"⚠️  Step {step+1}: Goal {goal} - OSCILLATION DETECTED! (distance from previous: {dist:.2f})")
            else:
                print(f"✓ Step {step+1}: Goal {goal} (distance from previous: {dist:.2f})")
        else:
            print(f"✓ Step {step+1}: Goal {goal}")
        
        # Simulate moving towards goal
        robot_pos = goal
        
        # Simulate exploration: mark area around goal as explored
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = goal[0] + dr, goal[1] + dc
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    if grid[r, c] == -1:
                        grid[r, c] = 0
    
    print("-" * 70)
    print(f"\nTotal goals: {len(goals)}")
    print(f"Unique goals: {len(set(goals))}")
    
    # Check for repeated consecutive goals
    consecutive_repeats = 0
    for i in range(1, len(goals)):
        if goals[i] == goals[i-1]:
            consecutive_repeats += 1
    
    if consecutive_repeats > 0:
        print(f"⚠️  Found {consecutive_repeats} consecutive repeated goals")
    else:
        print("✓ No consecutive repeated goals")

if __name__ == "__main__":
    test_oscillation()
