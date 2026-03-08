# Member 4 Navigation Stack (Webots, Python)

## 1) Overview
Member 4 provides the navigation pipeline in `/nav`:

- Global planning with A* on occupancy grid
- Obstacle inflation (safety margin)
- Path smoothing and waypoint generation
- Local waypoint follower for differential drive
- Reactive obstacle avoidance (LiDAR)
- Stuck detection and recovery (turn/backup/replan)
- Replanning manager (time, goal, map, deviation, blocked path triggers)
- Per-tick CSV logging for evaluation

`nav/navigation_stack.py` is the main orchestrator for the Webots controller.

## 2) Folder Structure
- `nav/config.py`: dataclass config and integration-friendly aliases
- `nav/types.py`: `Pose2D`, `Twist`, `Waypoint`, `GridMap`, `PlannerStatus`
- `nav/grid_utils.py`: grid/world conversion, occupancy checks, raycast, neighbors
- `nav/inflation.py`: inflate occupied cells + soft `cost_grid`
- `nav/a_star.py`: global A* planner and path simplification helpers
- `nav/replanner.py`: manages when to run/re-run A*
- `nav/path_smoothing.py`: LOS simplify + spacing + smoothing -> `Waypoint`s
- `nav/local_controller.py`: local differential-drive waypoint tracking
- `nav/obstacle_avoidance.py`: reactive LiDAR/point-cloud command adjustment
- `nav/stuck_recovery.py`: stuck detection + recovery state machine
- `nav/navigation_stack.py`: end-to-end orchestration API (`step`, `set_goal`, `get_debug`)
- `nav/logger.py`: CSV nav telemetry logger (`./logs/*.csv`)
- `nav/__init__.py`: package exports

## 3) Coordinate Frames & Units
- Units:
- `meters` for position and distances
- `radians` for yaw and angular velocity
- Webots world frame:
- Ground motion is in `X-Z` plane
- `Y` axis is up
- Nav 2D frame convention used here:
- `nav_x = webots_x`
- `nav_y = webots_z`
- `nav_yaw` is heading in the `X-Z` plane (radians)
- Yaw sign:
- Positive yaw is counterclockwise in the nav 2D plane
- If robot rotates opposite of commanded `omega`, flip sign once in command mapping
- LiDAR assumptions for avoidance:
- Ranges are in meters
- Angle `0` points forward in robot frame
- Angle ordering is increasing from `angle_min` to `angle_max`
- If your sensor order is reversed, reverse ranges in `get_lidar()`

## 4) Required Webots Devices & Setup
- Motors (required):
- Two wheel motors (`left`, `right`) in velocity control mode
- Call `motor.setPosition(float("inf"))`, then command velocities
- LiDAR (recommended):
- Enable with controller timestep
- Use horizontal scan (`getRangeImage()`)
- Keep horizontal resolution and FOV known by integration layer
- Pose source (required):
- Either:
- GPS + Compass (or Supervisor pose), or
- Wheel odometry/localization output from Member 2
- Optional comms devices:
- `Emitter`/`Receiver`/`Supervisor` can be used by Member 1 orchestration
- Nav stack does not require them directly

## 5) Integration Contract (Most Important)
### Member 1 Shared Interfaces (`/shared`)
If `/shared/map_store.py`, `/shared/pose.py`, `/shared/goal_store.py` are not created yet, use these expected signatures:

```python
def get_grid() -> tuple[object, int]:
    """Return (grid, map_version)."""
    # grid fields required:
    # resolution_m, origin_x_m, origin_y_m, width, height, data
    # data shape: (height, width), values in {-1, 0, 100}

def get_pose() -> Pose2D:
    """Return current robot pose in nav frame (meters, radians)."""

def get_goal() -> tuple[float, float] | None:
    """Return current world goal (x, y), or None if no goal."""
```

### Member 2 Outputs (Mapping/Localization)
- Map output to nav:
- Occupancy convention: `-1` unknown, `0` free, `100` occupied
- Metadata required: `resolution_m`, `origin_x_m`, `origin_y_m`, `width`, `height`
- Pose output to nav:
- Convert Webots translation to nav:
- `Pose2D(x=webots_x, y=webots_z, theta=yaw_rad)`
- Ensure yaw is radians and consistent sign

### Member 3 Output (Frontier Explorer)
- Frontier module provides next goal `(x, y)` in world meters
- Integration options:
- Push goal directly: `stack.set_goal(x, y)`
- Pull goal each tick via `get_goal()`
- If no valid frontier goal, return `None` (stack enters `IDLE`)

### NavigationStack Input/Output Contract
- `NavigationStack` constructor expects:
- `get_grid`, `get_pose`, `get_goal`, `get_lidar`, `send_cmd`
- `step(now_s)` does in order:
- inflation (if enabled)
- replanning / A*
- smoothing + waypoints
- local control
- obstacle avoidance
- stuck recovery (may override command + trigger replan)
- logging
- Command output:
- `send_cmd(Twist(v, omega))` where:
- `v` in m/s
- `omega` in rad/s

### Twist -> Wheel Velocity Mapping (Differential Drive)
Use wheel radius `R` and axle length `L`:

```python
left_rad_s = (v - 0.5 * L * omega) / R
right_rad_s = (v + 0.5 * L * omega) / R
```

### Member 5 Logging/Evaluation Hooks
- Built-in nav CSV logging path:
- `./logs/navigation.csv` (or `config.log_file_name`)
- Fields:
- `timestamp, pose_x, pose_y, pose_theta, goal_x, goal_y, goal_theta, mode, replans, collisions_count, stuck_events_count`
- Useful evaluation hooks from `stack.get_debug()`:
- `mode`, `current_path_len`, `waypoint_index`, `last_replan_time`, `emergency_stop`, `stuck_state`
- Collision/stuck proxy events:
- `emergency_stop == True` (near-collision safety event)
- `stuck_state != "NORMAL"` (recovery event)
- Screenshot trigger points (Member 5):
- when `mode` enters `RECOVERY`
- when `emergency_stop` becomes `True`
- when `mode` becomes `GOAL_REACHED`

## 6) Minimal Wiring Example
```python
# controllers/rescue_controller/rescue_controller.py
from controller import Robot
from nav.config import NavigationConfig
from nav.navigation_stack import NavigationStack
from nav.types import Pose2D, Twist

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)

WHEEL_RADIUS_M = 0.025  # placeholder
AXLE_LENGTH_M = 0.090   # placeholder

# Placeholder shared state (replace with /shared/map_store.py, /shared/pose.py, /shared/goal_store.py)
shared_grid = None
shared_map_version = 0
shared_goal = None
shared_pose = Pose2D(0.0, 0.0, 0.0)

def get_grid():
    return shared_grid, shared_map_version

def get_pose():
    return shared_pose

def get_goal():
    return shared_goal  # (x, y) or None

def get_lidar():
    return list(lidar.getRangeImage())  # meters

def send_cmd(cmd: Twist):
    left = (cmd.v - 0.5 * AXLE_LENGTH_M * cmd.omega) / WHEEL_RADIUS_M
    right = (cmd.v + 0.5 * AXLE_LENGTH_M * cmd.omega) / WHEEL_RADIUS_M
    left_motor.setVelocity(left)
    right_motor.setVelocity(right)

config = NavigationConfig()
stack = NavigationStack(get_grid, get_pose, get_goal, get_lidar, send_cmd, config=config)

while robot.step(timestep) != -1:
    # update shared_pose/shared_grid/shared_map_version/shared_goal before step()
    stack.step(robot.getTime())
```

## 7) Common Pitfalls / Debugging
- Wrong axis mapping (`x/z`):
- If nav appears mirrored or rotated, verify `nav_y = webots_z` mapping and yaw sign
- Map origin/resolution mismatch:
- Wrong `origin_*` or `resolution_m` causes path offsets and fake collisions
- Inflation too large:
- If no path found often, reduce `inflation_radius_m`
- Replanning too frequent / too slow:
- Tune `replanning_period_s` and map version updates
- Goal inside obstacle:
- A* has fallback to nearest free cell, but repeated bad goals can stall behavior
- LiDAR ordering mismatch:
- If avoidance turns wrong way, verify front index and angle ordering

## 8) How to Run
- In Webots:
- Open the rescue world
- Assign controller `controllers/rescue_controller/rescue_controller.py`
- Run simulation
- Verify quickly in a simple world:
- Start with one reachable goal and sparse obstacles
- Confirm modes transition: `IDLE -> PLANNING -> FOLLOWING -> GOAL_REACHED`
- Confirm logs:
- Check `./logs/navigation.csv`
- Inspect `mode`, `replans`, `collisions_count`, `stuck_events_count`

## 9) Optional ROS2 Note
ROS2 is not required for this stack, but mapping is straightforward:

- Occupancy map equivalent: `/map` (`nav_msgs/OccupancyGrid`)
- Pose/odometry equivalent: `/odom` (`nav_msgs/Odometry`) or TF-based pose
- Velocity command equivalent: `/cmd_vel` (`geometry_msgs/Twist`)
- Goal equivalent: `/goal` (project-defined, e.g., `geometry_msgs/PoseStamped`)

Use ROS2 bridges only if your team decides to expose Webots data over topics.
