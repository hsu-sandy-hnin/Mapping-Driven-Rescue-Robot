"""
Demonstration of safety scoring functionality
Shows that frontiers near obstacles get penalized compared to safe frontiers
"""
import numpy as np
from frontier_exploration import FrontierExplorer


def test_safety_scoring_demo():
    """Demonstrate that safety scoring penalizes frontiers near obstacles"""
    print("=" * 70)
    print("SAFETY SCORING DEMONSTRATION")
    print("=" * 70)
    
    # Create a grid with two frontier clusters:
    # - Cluster 1: Near obstacles (unsafe)
    # - Cluster 2: Far from obstacles (safe)
    grid = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  1,  1,  1,  0,  0, -1],
        [-1,  0,  1,  1,  1,  1,  1,  0, -1],
        [-1,  0,  0,  1,  1,  1,  0,  0, -1],
        [-1,  0,  0,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ], dtype=int)
    
    robot_pos = (4, 4)  # Robot at bottom center
    explorer = FrontierExplorer(w_safety=2.0)
    
    frontiers = explorer.detect_frontiers(grid)
    clusters = explorer.cluster_frontiers(frontiers, grid.shape)
    scored = explorer.score_clusters(clusters, robot_pos, grid)
    
    print("\nGrid (0=free, 1=obstacle, -1=unknown):")
    print(grid)
    print(f"\nRobot position: {robot_pos}")
    print(f"\nDetected {len(frontiers)} frontier cells")
    print(f"Clustered into {len(clusters)} clusters\n")
    
    # Sort by score
    scored.sort(key=lambda x: x['score'], reverse=True)
    
    print("Cluster Scoring Results:")
    print("-" * 70)
    for i, item in enumerate(scored):
        centroid = item['centroid']
        # Check if centroid is near obstacles
        is_near_obstacle = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = centroid[0] + dr, centroid[1] + dc
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    if grid[r, c] == 1:
                        is_near_obstacle = True
                        break
            if is_near_obstacle:
                break
        
        safety_status = "UNSAFE (near obstacle)" if is_near_obstacle else "SAFE (far from obstacles)"
        
        print(f"\nCluster {i+1}:")
        print(f"  Centroid: {centroid}")
        print(f"  Size: {item['size']} cells")
        print(f"  Distance from robot: {item['distance']:.2f}")
        print(f"  Safety score: {item['safety']:.2f} ({safety_status})")
        print(f"  Total score: {item['score']:.2f}")
    
    print("\n" + "-" * 70)
    print("Expected: Safe clusters should have higher scores than unsafe ones")
    print("=" * 70)


def test_oscillation_with_exploration():
    """Test oscillation avoidance with simulated exploration progress"""
    print("\n" + "=" * 70)
    print("OSCILLATION AVOIDANCE WITH EXPLORATION SIMULATION")
    print("=" * 70)
    
    # Start with a partially explored grid
    grid = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1,  0,  1,  1,  1,  0, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ], dtype=int)
    
    robot_pos = (2, 3)
    explorer = FrontierExplorer(revisit_threshold=2.5)
    
    print("\nInitial grid:")
    print(grid)
    print(f"Robot position: {robot_pos}\n")
    
    selected_goals = []
    for step in range(5):
        goal = explorer.select_next_goal(grid, robot_pos)
        if goal is None:
            print(f"Step {step+1}: No more frontiers!")
            break
        
        selected_goals.append(goal)
        print(f"Step {step+1}: Selected goal {goal}")
        
        # Simulate exploration: mark area around goal as explored
        # In reality, this would happen as the robot moves and scans
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = goal[0] + dr, goal[1] + dc
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    if grid[r, c] == -1:  # Mark unknown as explored (free)
                        grid[r, c] = 0
        
        # Move robot towards goal (simplified)
        robot_pos = goal
    
    print(f"\nTotal unique goals: {len(set(selected_goals))}")
    print(f"Total goals selected: {len(selected_goals)}")
    
    if len(selected_goals) > 1:
        print("\nGoal sequence:")
        for i, goal in enumerate(selected_goals):
            if i > 0:
                dist = explorer.compute_distance(selected_goals[i-1], goal)
                print(f"  {i}. {goal} (distance from previous: {dist:.2f})")
            else:
                print(f"  {i+1}. {goal}")
    
    print("=" * 70)


if __name__ == "__main__":
    test_safety_scoring_demo()
    test_oscillation_with_exploration()
