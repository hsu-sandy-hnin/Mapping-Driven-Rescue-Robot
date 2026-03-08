"""
Test with larger grid to identify issues
"""
import numpy as np
from frontier_exploration import FrontierExplorer

def create_large_grid():
    """Create a larger test grid"""
    grid = np.full((20, 20), -1, dtype=int)  # Start with all unknown
    
    # Create explored area in center
    grid[5:15, 5:15] = 0  # Free space
    
    # Add some obstacles
    grid[7:9, 7:9] = 1  # Obstacle block 1
    grid[11:13, 11:13] = 1  # Obstacle block 2
    
    # Add some explored corridors
    grid[8:12, 3:5] = 0  # Left corridor
    grid[8:12, 15:17] = 0  # Right corridor
    grid[3:5, 8:12] = 0  # Top corridor
    grid[15:17, 8:12] = 0  # Bottom corridor
    
    return grid

def test_large_grid():
    print("=" * 70)
    print("LARGE GRID TEST")
    print("=" * 70)
    
    grid = create_large_grid()
    robot_pos = (10, 10)  # Center of explored area
    explorer = FrontierExplorer()
    
    print(f"Grid shape: {grid.shape}")
    print(f"Robot position: {robot_pos}")
    print(f"\nGrid visualization (showing center 12x12 area):")
    print(grid[4:16, 4:16])
    
    # Detect frontiers
    frontiers = explorer.detect_frontiers(grid)
    print(f"\nDetected {len(frontiers)} frontier cells")
    
    # Cluster
    clusters = explorer.cluster_frontiers(frontiers, grid.shape)
    print(f"Found {len(clusters)} clusters")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {len(cluster)} cells")
    
    # Score
    scored = explorer.score_clusters(clusters, robot_pos, grid)
    scored.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nScored clusters (top 5):")
    for i, item in enumerate(scored[:5]):
        print(f"\nCluster {i+1}:")
        print(f"  Size: {item['size']}")
        print(f"  Centroid: {item['centroid']}")
        print(f"  Distance: {item['distance']:.2f}")
        print(f"  Safety: {item.get('safety', 'N/A')}")
        print(f"  Score: {item['score']:.2f}")
        # Check if centroid is actually a frontier
        centroid = item['centroid']
        is_frontier = centroid in frontiers
        print(f"  Centroid is frontier: {is_frontier}")
    
    # Select goal
    print(f"\n" + "=" * 70)
    print("GOAL SELECTION TEST")
    print("=" * 70)
    
    goals = []
    for step in range(10):
        goal = explorer.select_next_goal(grid, robot_pos)
        if goal is None:
            print(f"Step {step+1}: No goal available")
            break
        
        goals.append(goal)
        print(f"Step {step+1}: Goal = {goal}, is_frontier = {goal in frontiers}")
        
        # Check distance from previous goals
        if len(goals) > 1:
            dist = explorer.compute_distance(goals[-2], goal)
            print(f"         Distance from previous: {dist:.2f}")
        
        # Simulate moving towards goal (simplified)
        robot_pos = goal

if __name__ == "__main__":
    test_large_grid()
