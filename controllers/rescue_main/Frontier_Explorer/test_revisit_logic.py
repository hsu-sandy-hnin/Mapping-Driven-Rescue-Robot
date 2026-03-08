"""Test revisit detection logic"""
import numpy as np
from frontier_exploration import FrontierExplorer

# Simple test to verify revisit detection
explorer = FrontierExplorer(revisit_threshold=2.0)

# Record a goal
goal1 = (5, 5)
explorer._record_goal(goal1)
print(f"Recorded goal: {goal1}")
print(f"History: {list(explorer._goal_history)}")

# Test revisit detection
test_goals = [
    (5, 5),   # Same goal
    (5, 6),   # Very close (distance 1.0)
    (6, 5),   # Very close (distance 1.0)
    (7, 7),   # Close (distance ~2.83)
    (10, 10), # Far (distance ~7.07)
]

print("\nRevisit detection test:")
for goal in test_goals:
    dist = explorer.compute_distance(goal1, goal)
    is_revisit = explorer._is_revisit(goal)
    print(f"  Goal {goal}: distance={dist:.2f}, is_revisit={is_revisit}")
