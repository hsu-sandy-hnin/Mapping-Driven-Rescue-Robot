import os, math, json
from datetime import datetime
from controller import Robot

# =========================================================
# Member 1: Shared contracts / conventions
# =========================================================
# Pose: (x, y, yaw) in meters, radians.
# Map (stub): occupancy grid values convention:
#   -1 unknown, 0 free, 100 occupied (Member 2 can implement)
# Goal: goal in map/world plane (x,y)
# Path: list of waypoints [(x,y), ...]
# Logs: pose.csv + events.jsonl + optional screenshots/

# =========================================================
# Braitenberg (matches your C constants)
# =========================================================
TIME_STEP = 32
MAX_SPEED = 6.4
CRUISING_SPEED = 5.0
OBSTACLE_THRESHOLD = 0.1
DECREASE_FACTOR = 0.9
BACK_SLOWDOWN = 0.9

# Enable/disable algorithm modules (Member 1 integration switches)
USE_MAPPING = False         # Member 2 will set True when ready
USE_EXPLORATION = False     # Member 3 will set True when ready
USE_PLANNING = False        # Member 4 will set True when ready
USE_VICTIM_DETECTION = False  # Member 5 will set True when ready

# =========================================================
# Helpers
# =========================================================
def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def mkdirp(p):
    os.makedirs(p, exist_ok=True)
    return p

def is_inf(x):
    return x == float("inf") or x > 1e9

def try_get(robot, names):
    for n in names:
        try:
            d = robot.getDevice(n)
            if d is not None:
                return d
        except:
            pass
    return None

def gaussian(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-((x - mu) * (x - mu)) / (2.0 * sigma * sigma))

# =========================================================
# Interfaces (Member 1 responsibilities)
# =========================================================
class SharedState:
    def __init__(self):
        self.pose = {"x": float("nan"), "y": float("nan"), "yaw": float("nan")}
        self.map = None  # occupancy grid dict or numpy later
        self.goal = None # (gx, gy)
        self.goal_status = "none"  # none/active/reached/failed
        self.path = None # [(x,y),...]
        self.path_status = "none"  # none/active/done/failed
        self.victims = [] # list of detections

STATE = SharedState()

def set_goal(gx, gy):
    STATE.goal = (float(gx), float(gy))
    STATE.goal_status = "active"
    log_event("goal_set", {"gx": gx, "gy": gy})

def clear_goal():
    STATE.goal = None
    STATE.goal_status = "none"
    log_event("goal_cleared", {})

def set_path(waypoints):
    STATE.path = [(float(x), float(y)) for x, y in waypoints]
    STATE.path_status = "active"
    log_event("path_set", {"n": len(STATE.path)})

def clear_path():
    STATE.path = None
    STATE.path_status = "none"
    log_event("path_cleared", {})

# ---- Stubs for other members to implement later
def update_map(lidar_ranges, pose, map_state):
    """
    Member 2 will implement:
      - occupancy grid update using lidar + pose
      - return updated map object
    For now: passthrough.
    """
    return map_state

def choose_frontier_goal(map_state, pose):
    """
    Member 3 will implement frontier detection/clustering/scoring.
    Return (gx, gy) or None.
    """
    return None

def plan_path(map_state, start_pose, goal_xy):
    """
    Member 4 will implement A* on occupancy grid.
    Return list of waypoints [(x,y), ...] or None.
    """
    return None

def follow_path(path, pose):
    """
    Member 4 can implement waypoint follower to output (lv, rv) wheel velocities.
    Return (lv, rv) or None to fallback.
    """
    return None

def detect_victim(camera_dev, pose):
    """
    Member 5 will implement detection. Should return list of detections.
    Each detection can include: {"x":..., "y":..., "confidence":..., "note":...}
    """
    return []

# =========================================================
# Webots init
# =========================================================
robot = Robot()
basic_ts = int(robot.getBasicTimeStep())
ts = TIME_STEP if TIME_STEP > 0 else basic_ts

# ---- Logs
RUN_DIR = mkdirp(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs", f"run_{now_stamp()}")))
SCR_DIR = mkdirp(os.path.join(RUN_DIR, "screenshots"))
pose_csv = open(os.path.join(RUN_DIR, "pose.csv"), "w", encoding="utf-8")
pose_csv.write("t,x,y,yaw\n")
events = open(os.path.join(RUN_DIR, "events.jsonl"), "w", encoding="utf-8")

def log_event(evt_type, data=None):
    if data is None:
        data = {}
    rec = {"t": robot.getTime(), "type": evt_type, **data}
    events.write(json.dumps(rec) + "\n")
    events.flush()

print("✅ rescue_main started | logs:", RUN_DIR)
log_event("run_start", {"run_dir": RUN_DIR})

# ---- Motors (Pioneer3at = 4 wheels)
motors_left  = [robot.getDevice("front left wheel"), robot.getDevice("back left wheel")]
motors_right = [robot.getDevice("front right wheel"), robot.getDevice("back right wheel")]

for m in motors_left + motors_right:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

def set_wheels(lv, rv):
    for m in motors_left:
        m.setVelocity(lv)
    for m in motors_right:
        m.setVelocity(rv)

# ---- LiDAR
lidar = try_get(robot, ["Sick LMS 291", "lidar", "laser", "Lidar"])
if lidar is None:
    raise RuntimeError("LiDAR not found. Ensure device name is 'Sick LMS 291' or update try_get list.")

lidar.enable(ts)
try:
    lidar.enablePointCloud()
except:
    pass
print("✅ LiDAR:", lidar.getName())
log_event("device_ok", {"device": "lidar", "name": lidar.getName()})

lms_width = lidar.getHorizontalResolution()
half_width = lms_width // 2
max_range = float(lidar.getMaxRange())
range_threshold = max_range / 20.0

# braitenberg coefficients (same idea as C: gaussian centered at half_width, sigma = width/5)
brait = [gaussian(i, half_width, lms_width / 5.0) for i in range(lms_width)]

# ---- Optional camera + pose sensors
camera = try_get(robot, ["camera", "Camera", "cam", "front_camera"])
if camera is not None:
    camera.enable(ts)
    print("✅ Camera:", camera.getName())
    log_event("device_ok", {"device": "camera", "name": camera.getName()})
else:
    print("⚠️ Camera not found (ok for now). Add Camera name='camera' if needed.")

gps = try_get(robot, ["gps", "GPS"])
compass = try_get(robot, ["compass", "Compass"])
if gps is not None:
    gps.enable(ts)
    log_event("device_ok", {"device": "gps", "name": gps.getName()})
else:
    print("⚠️ GPS not found (pose will be NaN). Add GPS name='gps' for stable pose.")

if compass is not None:
    compass.enable(ts)
    log_event("device_ok", {"device": "compass", "name": compass.getName()})
else:
    print("⚠️ Compass not found (yaw will be NaN). Add Compass name='compass' for stable yaw.")

def get_pose():
    # Uses GPS+Compass if available; otherwise NaN.
    x = y = yaw = float("nan")
    if gps is not None:
        p = gps.getValues()   # [x, y, z]
        x, y = float(p[0]), float(p[2])  # ground plane
    if compass is not None:
        n = compass.getValues()
        yaw = math.atan2(n[0], n[2])
    return x, y, yaw

# ---- Screenshot helper
_last_shot = -1e9
SHOT_PERIOD = 3.0
def maybe_screenshot():
    global _last_shot
    t = robot.getTime()
    if camera is None:
        return
    if t - _last_shot < SHOT_PERIOD:
        return
    _last_shot = t
    fn = os.path.join(SCR_DIR, f"cam_{t:.2f}.png")
    try:
        camera.saveImage(fn, 100)
        log_event("screenshot", {"file": os.path.basename(fn)})
    except:
        pass

# =========================================================
# Main loop (Integration priority order)
# 1) Update pose/log
# 2) Mapping (if enabled)
# 3) Exploration goal (if enabled)
# 4) Planning path (if enabled)
# 5) Follow path (if enabled) else fallback to Braitenberg
# 6) Victim detection hook (if enabled)
# =========================================================
pose_flush_counter = 0

while robot.step(ts) != -1:
    t = robot.getTime()

    # ---- Pose
    x, y, yaw = get_pose()
    STATE.pose = {"x": x, "y": y, "yaw": yaw}
    pose_csv.write(f"{t:.3f},{x},{y},{yaw}\n")
    pose_flush_counter += 1
    if pose_flush_counter >= 10:
        pose_flush_counter = 0
        pose_csv.flush()

    # ---- LiDAR ranges (for mapping + behavior)
    ranges = lidar.getRangeImage()

    # ---- Mapping (Member 2)
    if USE_MAPPING:
        STATE.map = update_map(ranges, STATE.pose, STATE.map)

    # ---- Exploration (Member 3)
    if USE_EXPLORATION and STATE.goal is None and STATE.map is not None:
        g = choose_frontier_goal(STATE.map, STATE.pose)
        if g is not None:
            set_goal(g[0], g[1])

    # ---- Planning (Member 4)
    if USE_PLANNING and STATE.goal is not None and STATE.map is not None and STATE.path is None:
        wp = plan_path(STATE.map, STATE.pose, STATE.goal)
        if wp:
            set_path(wp)
        else:
            log_event("path_failed", {"gx": STATE.goal[0], "gy": STATE.goal[1]})
            STATE.goal_status = "failed"
            clear_goal()

    # ---- Control: follow path if available, else Braitenberg fallback
    cmd = None
    if USE_PLANNING and STATE.path:
        cmd = follow_path(STATE.path, STATE.pose)  # expected (lv, rv)
        if cmd is None:
            # follower not implemented yet -> fallback
            cmd = None

    if cmd is not None:
        lv, rv = cmd
        set_wheels(lv, rv)
    else:
        # -------- Braitenberg-like obstacle avoidance (matches your C)
        left_obs = 0.0
        right_obs = 0.0

        # same structure: scan half, mirror index for right
        for i in range(half_width):
            rL = ranges[i]
            if rL < range_threshold:
                left_obs += brait[i] * (1.0 - rL / max_range)

            j = lms_width - i - 1
            rR = ranges[j]
            if rR < range_threshold:
                right_obs += brait[i] * (1.0 - rR / max_range)

        obstacle = left_obs + right_obs

        if obstacle > OBSTACLE_THRESHOLD:
            speed_factor = (1.0 - DECREASE_FACTOR * obstacle) * MAX_SPEED / obstacle
            fl = speed_factor * left_obs
            fr = speed_factor * right_obs
            bl = BACK_SLOWDOWN * fl
            br = BACK_SLOWDOWN * fr
        else:
            fl = fr = bl = br = CRUISING_SPEED

        # Pioneer3at: set left side with "front-left/back-left", right side with "front-right/back-right"
        # Use front speeds as primary; back slowed in obstacle case like C.
        motors_left[0].setVelocity(fl)   # front left
        motors_left[1].setVelocity(bl)   # back left
        motors_right[0].setVelocity(fr)  # front right
        motors_right[1].setVelocity(br)  # back right

    # ---- Victim detection hook (Member 5)
    if USE_VICTIM_DETECTION and camera is not None:
        dets = detect_victim(camera, STATE.pose)
        if dets:
            for d in dets:
                STATE.victims.append(d)
                log_event("victim", d)
            maybe_screenshot()

    # ---- Optional: periodic screenshot even without detection (proof pipeline)
    maybe_screenshot()