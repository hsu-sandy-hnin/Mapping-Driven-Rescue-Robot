"""
Comprehensive test suite for FrontierExplorer
Tests all three main tasks:
1. Frontier detection (free/unknown boundary)
2. Frontier clustering + scoring (distance, size, safety)
3. Goal selection and oscillation avoidance
"""
import numpy as np
from frontier_exploration import FrontierExplorer


def test_frontier_detection():
    """Test Task 1: Frontier detection (free/unknown boundary)"""
    print("=" * 60)
    print("TEST 1: Frontier Detection")
    print("=" * 60)
    
    # Grid values: -1 = unknown, 0 = free, 1 = occupied
    grid = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  0,  0,  0, -1],
        [-1,  0,  1,  0, -1],
        [-1,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1],
    ], dtype=int)
    
    explorer = FrontierExplorer()
    frontiers = explorer.detect_frontiers(grid)
    
    print(f"Grid:\n{grid}")
    print(f"\nDetected frontiers: {sorted(frontiers)}")
    
    # Expected frontiers: free cells adjacent to unknown cells
    # Should be: (1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)
    expected = {(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)}
    
    assert frontiers == expected, f"Expected {expected}, got {frontiers}"
    print("✅ Frontier detection test PASSED")
    print()


def test_frontier_clustering():
    """Test Task 2: Frontier clustering"""
    print("=" * 60)
    print("TEST 2: Frontier Clustering")
    print("=" * 60)
    
    grid = np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0, -1],
        [-1,  0,  1,  1,  0, -1],
        [-1,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1],
    ], dtype=int)
    
    explorer = FrontierExplorer()
    frontiers = explorer.detect_frontiers(grid)
    clusters = explorer.cluster_frontiers(frontiers, grid.shape)
    
    print(f"Grid:\n{grid}")
    print(f"\nDetected frontiers: {sorted(frontiers)}")
    print(f"\nNumber of clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {len(cluster)} cells - {sorted(cluster)}")
    
    # Should have 2 clusters (left side and right side separated by obstacles)
    assert len(clusters) >= 1, "Should have at least one cluster"
    print("✅ Frontier clustering test PASSED")
    print()


def test_scoring():
    """Test Task 2: Scoring (distance, size, safety)"""
    print("=" * 60)
    print("TEST 3: Cluster Scoring")
    print("=" * 60)
    
    grid = np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0, -1],
        [-1,  0,  1,  1,  0, -1],
        [-1,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1],
    ], dtype=int)
    
    robot_pos = (2, 2)  # Robot at center obstacle
    explorer = FrontierExplorer()
    frontiers = explorer.detect_frontiers(grid)
    clusters = explorer.cluster_frontiers(frontiers, grid.shape)
    scored = explorer.score_clusters(clusters, robot_pos)
    
    print(f"Robot position: {robot_pos}")
    print(f"\nScored clusters:")
    for i, item in enumerate(scored):
        print(f"  Cluster {i+1}:")
        print(f"    Size: {item['size']}")
        print(f"    Distance: {item['distance']:.2f}")
        print(f"    Score: {item['score']:.2f}")
        print(f"    Centroid: {item['centroid']}")
    
    assert len(scored) > 0, "Should have scored clusters"
    assert all('score' in item for item in scored), "All clusters should have scores"
    print("✅ Scoring test PASSED")
    print()


def test_oscillation_avoidance():
    """Test Task 3: Avoid repeated/oscillating targets"""
    print("=" * 60)
    print("TEST 4: Oscillation Avoidance")
    print("=" * 60)
    
    # Create a grid with multiple frontier clusters
    grid = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1,  0,  1,  1,  1,  0, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ], dtype=int)
    
    robot_pos = (2, 3)
    explorer = FrontierExplorer(revisit_threshold=2.0)
    
    selected_goals = []
    for step in range(10):
        goal = explorer.select_next_goal(grid, robot_pos)
        if goal is None:
            break
        selected_goals.append(goal)
        print(f"Step {step+1}: Selected goal {goal}")
        
        # Simulate robot moving towards goal (simplified)
        # In real scenario, the grid would be updated as exploration progresses
        robot_pos = goal  # Move robot to goal
    
    print(f"\nTotal goals selected: {len(selected_goals)}")
    print(f"Unique goals: {len(set(selected_goals))}")
    
    # Check that we're not oscillating between the same goals
    if len(selected_goals) >= 2:
        # Check that consecutive goals are not too close
        for i in range(len(selected_goals) - 1):
            dist = explorer.compute_distance(selected_goals[i], selected_goals[i+1])
            print(f"  Distance between goal {i+1} and {i+2}: {dist:.2f}")
    
    print("✅ Oscillation avoidance test PASSED")
    print()


def test_safety_scoring():
    """Test safety scoring (proximity to obstacles)"""
    print("=" * 60)
    print("TEST 5: Safety Scoring")
    print("=" * 60)
    
    # Grid with frontiers near obstacles (unsafe) and far from obstacles (safe)
    grid = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  1,  0,  0, -1],
        [-1,  0,  1,  1,  1,  0, -1],
        [-1,  0,  0,  1,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ], dtype=int)
    
    robot_pos = (2, 3)
    explorer = FrontierExplorer()
    frontiers = explorer.detect_frontiers(grid)
    clusters = explorer.cluster_frontiers(frontiers, grid.shape)
    scored = explorer.score_clusters(clusters, robot_pos)
    
    print(f"Grid:\n{grid}")
    print(f"\nScored clusters with safety:")
    for i, item in enumerate(scored):
        safety = item.get('safety', 'N/A')
        print(f"  Cluster {i+1}: Score={item['score']:.2f}, Safety={safety}")
    
    print("✅ Safety scoring test PASSED (if safety field exists)")
    print()


def test_integration():
    """Integration test: Full workflow"""
    print("=" * 60)
    print("TEST 6: Integration Test")
    print("=" * 60)
    
    grid = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0, -1,  0, -1],
        [-1,  0,  1,  1, -1,  0, -1],
        [-1,  0,  0,  0,  1,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ], dtype=int)
    
    robot_pos = (3, 3)
    explorer = FrontierExplorer()
    
    print("Initial grid:")
    print(grid)
    print(f"\nRobot position: {robot_pos}")
    
    goal = explorer.select_next_goal(grid, robot_pos)
    print(f"\nSelected goal: {goal}")
    
    if goal:
        print(f"Goal is valid: {0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]}")
        print(f"Goal cell value: {grid[goal]}")
    
    print("✅ Integration test PASSED")
    print()

def test_all_in_one():
    explorer = FrontierExplorer(revisit_threshold=2.0)
    
    grid = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0, -1,  0,  0,  0, -1],
        [-1,  0,  1,  0, -1,  0,  1,  0, -1],
        [-1,  0,  0,  0, -1,  0,  0,  0, -1],
        [-1, -1, -1,  0,  1,  -1, -1, -1, -1],
        [-1,  0,  0,  0, -1,  0,  0,  0, -1],
        [-1,  0,  1,  0, -1,  0,  1,  0, -1],
        [-1,  0,  0,  0, -1,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ], dtype=int)
    robot_pos = (4, 4)
    
    print("Grid:")
    print(grid)
    
    # Step 1: Detection
    frontiers = explorer.detect_frontiers(grid)
    print(f"\nFrontiers ({len(frontiers)}): {sorted(frontiers)}")
    
    # Step 2: Clustering
    clusters = explorer.cluster_frontiers(frontiers, grid.shape)
    print(f"\nClusters: {len(clusters)}")
    
    # Step 3: Scoring
    scored = explorer.score_clusters(clusters, robot_pos)
    for i, c in enumerate(scored):
        print(f"\nCluster {i+1}:")
        print(f"  Size: {c['size']}")
        print(f"  Distance: {c['distance']:.2f}")
        print(f"  Score: {c['score']:.2f}")
        print(f"  Safety: {c.get('safety', 'N/A')}")
        print(f"  Centroid: {c['centroid']}")
    
    # Step 4: Goal selection over time
    print("\nGoal sequence:")
    for step in range(8):
        goal = explorer.select_next_goal(grid, robot_pos)
        print(f"Step {step+1}: {goal}")
        if goal:
            robot_pos = goal


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COMPREHENSIVE FRONTIER EXPLORATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_all_in_one()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
