import os, math, json, sys, csv, zlib, random
import numpy as np
import cv2
from datetime import datetime
from controller import Robot, Supervisor

# Add paths for modules
base_path = os.path.dirname(__file__)
sys.path.append(os.path.join(base_path, "nav"))
sys.path.append(os.path.join(base_path, "Frontier_Explorer"))

from nav.navigation_stack import NavigationStack
from nav.types import Pose2D, Twist
from Frontier_Explorer.frontier_exploration import FrontierExplorer
from victim.event_logger import VictimEventWriter, VictimMetricsTracker


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

# Enable/disable algorithm modules
USE_MAPPING = True         
USE_EXPLORATION = True     
USE_PLANNING = True        
USE_VICTIM_DETECTION = True   # Member 5 — enabled

# =========================================================
# Member 5 — Victim Detection Parameters
# =========================================================
# HSV ranges (tune to match simulation victim colours)
M5_RED_LO1  = np.array([0,   120, 80],  np.uint8)
M5_RED_HI1  = np.array([10,  255, 255], np.uint8)
M5_RED_LO2  = np.array([170, 120, 80],  np.uint8)
M5_RED_HI2  = np.array([180, 255, 255], np.uint8)
M5_GRN_LO   = np.array([40,  80,  80],  np.uint8)
M5_GRN_HI   = np.array([90,  255, 255], np.uint8)
M5_MIN_AREA      = 120    # px²  — keep long-range victims detectable
M5_MAX_ASPECT    = 5.0    # discard long thin shapes
M5_DEDUP_RADIUS  = 0.8   # metres — skip if already detected within this range
M5_FWD_OFFSET    = 1.0   # metres — assume victim is this far ahead of robot
M5_MAX_TARGET_DETECTION_RANGE_M = 8.0  # metres — ignore very far camera hits

# Mission objectives (scan -> detect -> rescue -> finish)
MISSION_REQUIRED_RESCUES = 4
MISSION_RESCUE_RADIUS_M = 1.10
# With pillar layouts, center-to-center distance often bottoms out near ~0.53m.
# Keep a small margin so proximity rescue can complete once the robot has approached.
RESCUE_RADIUS_M = 0.65  # Final close-range rescue trigger radius (meters), center-to-center.
MATCH_RADIUS_M = 0.50   # Deterministic detection<->world-target association radius (meters).
RECENT_SEEN_TICKS = 180  # Camera evidence validity window (ticks), ~5.7s at 32 ms step.
RESCUE_GOAL_MATCH_TOL_M = 0.75  # Goal-to-target XY match tolerance for evidence gate.
MISSION_NAV_MAX_V_MPS = 0.95
MISSION_NAV_MAX_OMEGA_RADPS = 1.85
MISSION_NAV_SPEED_GAIN_PER_RESCUE_MPS = 0.14
MISSION_NAV_MAX_V_CAP_MPS = 1.60
# Do not mark rescued just by camera sighting; require actual rescue action.
AUTO_RESCUE_ON_DETECTION = False
WORLD_TARGET_PROXIMITY_RESCUE_RADIUS_M = 1.85
WORLD_TARGET_MATCH_RADIUS_M = 2.5
WORLD_TARGET_DETECTION_MATCH_RADIUS_M = MATCH_RADIUS_M  # Final detection->world target association radius.
WORLD_TARGET_RESCUE_FALLBACK_MATCH_RADIUS_M = MATCH_RADIUS_M  # Fallback rematch radius inside rescue_victim.
WORLD_TARGET_HIDE_BASE_X = 1000.0
WORLD_TARGET_HIDE_BASE_Y = 1100.0
DEBUG_RESCUE = False  # Verbose rescue diagnostics / traces / decision tables.
DEBUG_WORLD_TARGETS = True  # Set False to silence world-target integrity audits.
RESCUE_ATTEMPT_SUMMARY_RADIUS_M = 1.0  # Emit one-line rescue attempt summary only within this distance.
EXPLORATION_GOAL_JITTER_M = 0.0
ARENA_HALF_EXTENT_M = 30.0
EXPLORATION_WALL_MARGIN_M = 4.0
EXPLORATION_GOAL_MAX_ABS_M = ARENA_HALF_EXTENT_M - EXPLORATION_WALL_MARGIN_M
EXPLORATION_MIN_GOAL_TRAVEL_M = 1.0  # Avoid zero-distance explore goals at current pose.
EXPLORATION_VICTIM_EXCLUSION_RADIUS_M = 1.20  # Keep explore goals away from victim locations.
RESCUE_TARGET_MAP_CLEAR_RADIUS_M = 0.45
RESCUE_TARGET_LIDAR_IGNORE_RADIUS_M = 0.14
RESCUE_TARGET_LIDAR_IGNORE_MAX_DIST_M = 1.6
RESCUE_TARGET_LIDAR_IGNORE_NEAR_DELTA_M = 0.10
RESCUE_TARGET_LIDAR_IGNORE_FAR_DELTA_M = 0.22
FINAL_APPROACH_ENTER_DISTANCE_M = 1.00
FINAL_APPROACH_EXIT_DISTANCE_M = 1.20
FINAL_APPROACH_MAX_V_MPS = 0.75
FINAL_APPROACH_MAX_OMEGA_RADPS = 1.90
FINAL_APPROACH_LOOKAHEAD_M = 0.70
FINAL_APPROACH_AVOID_GAIN = 0.85
FINAL_APPROACH_SAFETY_DISTANCE_M = 0.72
FINAL_APPROACH_SIDE_CLEARANCE_M = 0.50
FINAL_APPROACH_STOP_DISTANCE_M = 0.24
FINAL_APPROACH_INFLATION_RADIUS_CELLS = 1
EXPLORATION_MIN_KNOWN_FRACTION = 0.90
EXPLORATION_NO_FRONTIER_HOLD_S = 12.0
EXPLORATION_STABLE_MAP_HOLD_S = 10.0
# Keep a short fallback cadence so the robot does not pause long between goals.
EXPLORATION_FALLBACK_GOAL_PERIOD_S = 1.2
EXPLORATION_FALLBACK_MIN_TRAVEL_M = 2.0
KEEP_RUNNING_AFTER_MISSION_DONE = False
ENABLE_STARTUP_DIRECT_RESCUE = False  # Keep camera-only detection flow; do not pre-seed victim locations.
STARTUP_DIRECT_RESCUE_WINDOW_S = 55.0
STARTUP_DIRECT_RESCUE_MIN_CLEARANCE_M = 1.10
STARTUP_DIRECT_RESCUE_CORRIDOR_HALF_DEG = 16.0
STARTUP_DIRECT_RESCUE_GLOBAL_MIN_M = 0.52
CAMERA_SIGHTING_TTL_S = 2.5
CAMERA_SIGHTING_LOG_PERIOD_S = 0.40
FORCE_WORLD_TARGET_RESCUE_TASK = False
WORLD_TARGET_TASK_START_DELAY_S = 0.5
WORLD_TARGET_TASK_GOAL_PERIOD_S = 0.8
# Camera-driven flow: do not auto-rescue from pre-setup world target proximity.
USE_WORLD_TARGET_PROXIMITY_RESCUE = False

# =========================================================
# Member 2 tuning: mapping + pose fusion
# =========================================================
MAP_UPDATE_HZ = 8.0
MAP_INFLATION_RADIUS_CELLS = 2
LIDAR_DOWNSAMPLE = 2
MAP_CHANGE_CELL_THRESHOLD = 100

POSE_WHEEL_RADIUS_M = 0.1
POSE_AXLE_LENGTH_M = 0.4
POSE_GPS_CORR_ALPHA = 0.15
POSE_COMPASS_CORR_ALPHA = 0.12
POSE_GPS_OUTLIER_JUMP_M = 2.0
POSE_GPS_MAX_CORR_STEP_M = 0.5
POSE_COMPASS_MAX_CORR_RAD = 0.45

# =========================================================
# Member 1 software checkpoints (no world edits required)
# =========================================================
CHECKPOINTS = [
    {"name": "CP_CENTER", "x": 0.0, "y": 0.0},
    {"name": "CP_NE", "x": 4.5, "y": 4.5},
    {"name": "CP_NW", "x": -4.5, "y": 4.5},
    {"name": "CP_SE", "x": 4.5, "y": -4.5},
    {"name": "CP_SW", "x": -4.5, "y": -4.5},
]
CHECKPOINT_ENTER_THRESHOLD_M = 1.25
CHECKPOINT_EXIT_HYSTERESIS_M = 0.5
CHECKPOINT_POLL_HZ = 4.0

# =========================================================
# Member 5 non-ROS evaluation + unified victim logging
# =========================================================
VICTIM_METRICS_WRITE_PERIOD_S = 10.0
VICTIM_UNIQUE_CLUSTER_RADIUS_M = 0.75
VICTIM_DUPLICATE_RADIUS_M = 0.35

# Victim state (Member 5)
_m5_victims   = []   # list of (wx, wy) already logged
_m5_next_id   = 1
VICTIM_EVENT_WRITER = None
VICTIM_METRICS = None
VICTIM_METRICS_SUMMARY_PATH = None

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

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def _wrap_angle(a):
    """Wrap angle to [-pi, pi)."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

def _angle_blend(current, target, alpha):
    """Blend two angles while respecting wrap-around."""
    alpha = _clamp(float(alpha), 0.0, 1.0)
    return _wrap_angle(float(current) + alpha * _wrap_angle(float(target) - float(current)))


class PoseFusionEstimator:
    """Complementary fusion of wheel odom (or cmd integration) with GPS/Compass."""

    def __init__(self, wheel_radius_m, axle_length_m):
        self.wheel_radius_m = float(wheel_radius_m)
        self.axle_length_m = max(1e-6, float(axle_length_m))
        self.left_sensors = []
        self.right_sensors = []
        self.encoder_available = False
        self._prev_left_angle = None
        self._prev_right_angle = None
        self._cmd_v = 0.0
        self._cmd_omega = 0.0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._last_t = None
        self._last_accepted_gps = None
        self._initialized = False

    def attach_wheel_sensors(self, left_sensors, right_sensors):
        """Register enabled wheel position sensors for differential odometry."""
        self.left_sensors = [s for s in left_sensors if s is not None]
        self.right_sensors = [s for s in right_sensors if s is not None]
        self.encoder_available = bool(self.left_sensors and self.right_sensors)
        return self.encoder_available

    def set_command(self, v_mps, omega_radps):
        """Store the last commanded twist for fallback integration."""
        self._cmd_v = float(v_mps)
        self._cmd_omega = float(omega_radps)

    def _avg_sensor_angle(self, sensors):
        vals = []
        for s in sensors:
            try:
                vals.append(float(s.getValue()))
            except Exception:
                continue
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _integrate(self, dt_s):
        source = "command"
        if self.encoder_available:
            left = self._avg_sensor_angle(self.left_sensors)
            right = self._avg_sensor_angle(self.right_sensors)
            if left is not None and right is not None:
                if self._prev_left_angle is None or self._prev_right_angle is None:
                    self._prev_left_angle = left
                    self._prev_right_angle = right
                    return "encoder_init"
                dl = (left - self._prev_left_angle) * self.wheel_radius_m
                dr = (right - self._prev_right_angle) * self.wheel_radius_m
                self._prev_left_angle = left
                self._prev_right_angle = right
                d_center = 0.5 * (dl + dr)
                d_yaw = (dr - dl) / self.axle_length_m
                mid_yaw = self._yaw + 0.5 * d_yaw
                self._x += d_center * math.cos(mid_yaw)
                self._y += d_center * math.sin(mid_yaw)
                self._yaw = _wrap_angle(self._yaw + d_yaw)
                return "encoders"

        d_center = self._cmd_v * float(dt_s)
        d_yaw = self._cmd_omega * float(dt_s)
        mid_yaw = self._yaw + 0.5 * d_yaw
        self._x += d_center * math.cos(mid_yaw)
        self._y += d_center * math.sin(mid_yaw)
        self._yaw = _wrap_angle(self._yaw + d_yaw)
        return source

    def update(self, now_s, gps_xy=None, compass_yaw=None):
        """Advance odometry and apply slow GPS/Compass correction."""
        now_s = float(now_s)
        diag = {"odom_source": "none", "gps_outlier": False}

        if self._last_t is None:
            self._last_t = now_s
            if gps_xy is not None:
                self._x, self._y = float(gps_xy[0]), float(gps_xy[1])
                self._last_accepted_gps = (self._x, self._y)
            if compass_yaw is not None:
                self._yaw = _wrap_angle(compass_yaw)
            self._initialized = True
            diag["odom_source"] = "init"
            return self._x, self._y, self._yaw, diag

        dt_s = _clamp(now_s - self._last_t, 0.0, 0.25)
        self._last_t = now_s
        if dt_s > 0.0:
            diag["odom_source"] = self._integrate(dt_s)

        if gps_xy is not None:
            gx, gy = float(gps_xy[0]), float(gps_xy[1])
            if self._last_accepted_gps is not None:
                jump = math.hypot(gx - self._last_accepted_gps[0], gy - self._last_accepted_gps[1])
                if jump > POSE_GPS_OUTLIER_JUMP_M:
                    diag["gps_outlier"] = True
                else:
                    self._last_accepted_gps = (gx, gy)
            else:
                self._last_accepted_gps = (gx, gy)

            if not diag["gps_outlier"]:
                dx = gx - self._x
                dy = gy - self._y
                corr_mag = math.hypot(dx, dy)
                if corr_mag > POSE_GPS_MAX_CORR_STEP_M and corr_mag > 1e-9:
                    scale = POSE_GPS_MAX_CORR_STEP_M / corr_mag
                    dx *= scale
                    dy *= scale
                self._x += POSE_GPS_CORR_ALPHA * dx
                self._y += POSE_GPS_CORR_ALPHA * dy

        if compass_yaw is not None:
            d_yaw = _wrap_angle(float(compass_yaw) - self._yaw)
            d_yaw = _clamp(d_yaw, -POSE_COMPASS_MAX_CORR_RAD, POSE_COMPASS_MAX_CORR_RAD)
            self._yaw = _angle_blend(self._yaw, self._yaw + d_yaw, POSE_COMPASS_CORR_ALPHA)

        return self._x, self._y, self._yaw, diag

# =========================================================
# Interfaces (Member 1 responsibilities)
# =========================================================
class SharedState:
    def __init__(self):
        self.pose = {"x": float("nan"), "y": float("nan"), "yaw": float("nan")}
        self.map_raw = None  # uninflated occupancy grid from LiDAR updates
        self.map = None  # occupancy grid dict or numpy later
        self.map_version = 0
        self.last_grid_hash = None
        self.last_map_changed_cells = 0
        self.last_map_update_t = -1e9
        self.last_map_version_bump_t = -1e9
        self.goal = None # (gx, gy)
        self.goal_kind = "none"  # none/explore/rescue
        self.goal_status = "none"  # none/active/reached/failed
        self.path = None # [(x,y),...]
        self.path_signature = None
        self.path_status = "none"  # none/active/done/failed
        self.victims = [] # list of detections
        self.active_rescue_id = None
        self.rescued_victim_ids = set()
        self.rescued_sites = []  # [(x, y), ...] for duplicate suppression after rescue
        self.rescued_world_target_defs = set()
        self.rescued_target_positions = {}  # DEF -> (x, y) stable marker locations for map overlays
        self.world_target_to_victim_id = {}  # DEF -> victim_id
        self.last_seen_tick_by_def = {}  # DEF -> last camera-seen loop tick (canonical key for evidence gate)
        self.camera_sightings_by_key = {}  # sighting_key -> {"x","y","t","matched","target_def","source"}
        self.last_camera_sighting_log_t_by_key = {}  # sighting_key -> last event timestamp
        # Backward-compatible alias used by older call sites.
        self.world_target_last_seen_tick = self.last_seen_tick_by_def
        self.required_victims = MISSION_REQUIRED_RESCUES
        self.mission_done = False
        self.mission_done_reason = "none"
        self.no_frontier_since_t = None
        self.last_explore_fallback_goal_t = -1e9
        self.last_world_target_task_goal_t = -1e9
        self.next_explore_checkpoint_idx = 0
        self.current_checkpoint_name = None
        self.last_checkpoint_check_t = -1e9

STATE = SharedState()

def log_event(evt_type, data=None): 
    if data is None: 
        data = {} 
    rec = {"t": robot.getTime(), "type": evt_type, **data} 
    events.write(json.dumps(rec) + "\n") 
    events.flush()

def set_goal(gx, gy, goal_kind="explore", rescue_id=None):
    gx = float(gx)
    gy = float(gy)
    goal_kind = str(goal_kind)
    if goal_kind == "explore":
        block_reason = _explore_goal_block_reason(gx, gy)
        if block_reason is not None:
            log_event(
                "goal_rejected",
                {"goal_kind": "explore", "gx": gx, "gy": gy, "reason": str(block_reason)},
            )
            return False
    prev_goal = STATE.goal
    STATE.goal = (gx, gy)
    STATE.goal_kind = goal_kind
    if STATE.goal_kind != "explore":
        STATE.no_frontier_since_t = None
    STATE.active_rescue_id = None if rescue_id is None else int(rescue_id)
    STATE.goal_status = "active"
    payload = {"gx": gx, "gy": gy, "goal_kind": STATE.goal_kind}
    if STATE.active_rescue_id is not None:
        payload["rescue_id"] = STATE.active_rescue_id
    if prev_goal is None or _dist_xy(prev_goal[0], prev_goal[1], STATE.goal[0], STATE.goal[1]) > 1e-6:
        clear_path(reason="new_goal")
    log_event("goal_set", payload)
    return True

def _path_points_signature(points):
    if not points:
        return None
    return tuple((round(float(x), 3), round(float(y), 3)) for x, y in points)

def clear_goal(reason="completed"):
    prev_kind = STATE.goal_kind
    prev_rescue_id = STATE.active_rescue_id
    STATE.goal = None
    STATE.goal_kind = "none"
    STATE.active_rescue_id = None
    STATE.goal_status = "none"
    payload = {"goal_kind": prev_kind, "reason": reason}
    if prev_rescue_id is not None:
        payload["rescue_id"] = prev_rescue_id
    clear_path(reason=f"goal_cleared:{reason}")
    log_event("goal_cleared", payload)

def set_path(waypoints, reason="unspecified"):
    """Update the shared path contract when nav produces a new route."""
    pts = [(float(x), float(y)) for x, y in waypoints]
    if not pts:
        clear_path(reason=reason)
        return
    sig = _path_points_signature(pts)
    if sig == STATE.path_signature and STATE.path_status == "active":
        return
    STATE.path = pts
    STATE.path_signature = sig
    STATE.path_status = "active"
    print(f"PATH_SET length={len(STATE.path)} reason={reason}")
    log_event("path_set", {"n": len(STATE.path), "reason": str(reason)})

def clear_path(reason="unspecified"):
    """Clear the shared path contract when goal/path becomes invalid."""
    had_path = STATE.path is not None or STATE.path_status != "none"
    STATE.path = None
    STATE.path_signature = None
    STATE.path_status = "none"
    if had_path:
        print(f"PATH_CLEARED reason={reason}")
        log_event("path_cleared", {"reason": str(reason)})

def _pose_xy_valid(pose):
    return not (np.isnan(pose["x"]) or np.isnan(pose["y"]))

def _dist_xy(x1, y1, x2, y2):
    return math.hypot(float(x2) - float(x1), float(y2) - float(y1))

def _explore_goal_block_reason(gx, gy, radius_m=EXPLORATION_VICTIM_EXCLUSION_RADIUS_M):
    """Return reason string when an exploration goal is too close to victim locations."""
    gx = float(gx)
    gy = float(gy)
    radius_m = max(0.05, float(radius_m))

    # Block around currently known unrescued victim detections.
    for victim in STATE.victims:
        try:
            vid = int(victim.get("id", -1))
            if bool(victim.get("rescued", False)) or vid in STATE.rescued_victim_ids:
                continue
            vx = float(victim["x"])
            vy = float(victim["y"])
        except Exception:
            continue
        if _dist_xy(gx, gy, vx, vy) <= radius_m:
            return f"near_detected_victim:{vid}"

    # Also block around active pre-setup world targets (camera-independent locations).
    world_xy = globals().get("_world_target_xy", None)
    target_defs = globals().get("RESCUE_TARGET_DEFS", [])
    hidden_defs = globals().get("_WORLD_TARGET_HIDDEN", set())
    if callable(world_xy):
        for def_name in target_defs:
            if def_name in STATE.rescued_world_target_defs or def_name in hidden_defs:
                continue
            pos = world_xy(def_name)
            if pos is None:
                continue
            if _dist_xy(gx, gy, float(pos[0]), float(pos[1])) <= radius_m:
                return f"near_world_target:{def_name}"
    return None

def _rescue_trace_value(value):
    """Format structured rescue-trace values as compact single-token strings."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.3f}"
    return str(value).replace(" ", "_")

def _rescue_trace(func_name, event, **fields):
    """Single-line structured trace logger for rescue/hide control paths."""
    if not DEBUG_RESCUE:
        return
    parts = ["RESCUE_TRACE", f"fn={func_name}", f"event={event}"]

    tick = globals().get("loop_debug_tick", None)
    if tick is not None:
        try:
            parts.append(f"tick={int(tick)}")
        except Exception:
            parts.append(f"tick={_rescue_trace_value(tick)}")

    try:
        now_t = robot.getTime()
        parts.append(f"time={float(now_t):.3f}")
    except Exception:
        parts.append("time=na")

    pose = getattr(globals().get("STATE", None), "pose", None)
    if isinstance(pose, dict):
        try:
            px = float(pose.get("x", float("nan")))
            py = float(pose.get("y", float("nan")))
            yaw = float(pose.get("yaw", float("nan")))
            parts.append(f"robot_pose=({px:.2f},{py:.2f},{yaw:.2f})")
        except Exception:
            parts.append("robot_pose=(na,na,na)")

    for key, value in fields.items():
        parts.append(f"{key}={_rescue_trace_value(value)}")
    print(" ".join(parts))

def _world_target_trace(event, **fields):
    """Single-line structured logger for world-target handle/iterator audits."""
    if not (DEBUG_RESCUE and DEBUG_WORLD_TARGETS):
        return
    _rescue_trace("world_targets", event, **fields)

def _rescue_gate_status(value):
    """Normalize gate values to PASS/FAIL/NA for one-line decision tables."""
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    s = str(value).strip()
    if not s:
        return "NA"
    up = s.upper()
    if up in {"PASS", "FAIL", "NA"}:
        return up
    return s

def _rescue_gate_table(func_name, event, gates, **fields):
    """Emit a one-line rescue decision table with the first failing gate.

    Canonical gates (when applicable):
    - supervisor_available
    - handles_initialized
    - candidate_target_exists
    - matching_success
    - already_rescued
    - cooldown_throttle
    - distance_le_rescue_radius
    """
    if not DEBUG_RESCUE:
        return
    ordered = []
    first_fail = "none"
    for name, raw in gates:
        status = _rescue_gate_status(raw)
        ordered.append((name, status))
        if first_fail == "none" and status == "FAIL":
            first_fail = name

    parts = ["RESCUE_GATES", f"fn={func_name}", f"event={event}", f"first_fail={first_fail}"]
    tick = globals().get("loop_debug_tick", None)
    if tick is not None:
        try:
            parts.append(f"tick={int(tick)}")
        except Exception:
            parts.append(f"tick={_rescue_trace_value(tick)}")
    try:
        parts.append(f"time={float(robot.getTime()):.3f}")
    except Exception:
        parts.append("time=na")
    pose = getattr(globals().get("STATE", None), "pose", None)
    if isinstance(pose, dict):
        try:
            parts.append(
                f"robot_pose=({float(pose.get('x', float('nan'))):.2f},"
                f"{float(pose.get('y', float('nan'))):.2f},"
                f"{float(pose.get('yaw', float('nan'))):.2f})"
            )
        except Exception:
            parts.append("robot_pose=(na,na,na)")

    for name, status in ordered:
        parts.append(f"{name}={status}")
    for key, value in fields.items():
        parts.append(f"{key}={_rescue_trace_value(value)}")
    print(" ".join(parts))

_LAST_RESCUE_ATTEMPT_SUMMARY_KEY = None

def _rescue_attempt_summary(func_name, victim, target_def, gate_result):
    """Single-line rescue attempt summary kept when DEBUG_RESCUE is disabled."""
    global _LAST_RESCUE_ATTEMPT_SUMMARY_KEY
    if DEBUG_RESCUE:
        return
    if not isinstance(gate_result, dict):
        return
    dist_m = gate_result.get("dist_m")
    try:
        dist_ok = math.isfinite(float(dist_m))
    except Exception:
        dist_ok = False
    if not dist_ok:
        return
    if float(dist_m) > float(RESCUE_ATTEMPT_SUMMARY_RADIUS_M):
        return

    tick = _current_loop_tick()
    victim_id = None if victim is None else int(victim.get("id", -1))
    target_def_str = None if not target_def else str(target_def)
    key = (tick, victim_id, target_def_str)
    if key == _LAST_RESCUE_ATTEMPT_SUMMARY_KEY:
        return
    _LAST_RESCUE_ATTEMPT_SUMMARY_KEY = key

    robot_xy = gate_result.get("robot_xy")
    target_xy = gate_result.get("target_xy")
    if isinstance(robot_xy, tuple) and len(robot_xy) >= 2:
        robot_xy_s = f"({float(robot_xy[0]):.2f},{float(robot_xy[1]):.2f})"
    else:
        robot_xy_s = "none"
    if isinstance(target_xy, tuple) and len(target_xy) >= 2:
        target_xy_s = f"({float(target_xy[0]):.2f},{float(target_xy[1]):.2f})"
    else:
        target_xy_s = "none"

    print(
        "RESCUE_ATTEMPT "
        f"tick={_rescue_trace_value(tick)} "
        f"fn={func_name} "
        f"victim_id={_rescue_trace_value(victim_id)} "
        f"victim_def={_rescue_trace_value(target_def_str)} "
        f"robot_xy={robot_xy_s} "
        f"target_xy={target_xy_s} "
        f"dist_m={_rescue_trace_value(dist_m)} "
        f"threshold_m={_rescue_trace_value(gate_result.get('threshold_m'))} "
        f"recent_seen={_rescue_trace_value(gate_result.get('recent_seen'))} "
        f"goal_evidence={_rescue_trace_value(gate_result.get('goal_evidence'))} "
        f"lidar_evidence={_rescue_trace_value(gate_result.get('lidar_evidence'))} "
        f"evidence_ok={_rescue_trace_value(gate_result.get('evidence_ok', gate_result.get('evidence_gate')))} "
        f"decision={'PASS' if gate_result.get('ok') else 'FAIL'} "
        f"first_fail={_rescue_trace_value(gate_result.get('first_fail'))}"
    )

def find_victim_by_id(victim_id):
    for v in STATE.victims:
        if int(v.get("id", -1)) == int(victim_id):
            return v
    return None

def _current_loop_tick():
    """Best-effort current loop tick for temporal evidence windows."""
    try:
        return int(globals().get("loop_debug_tick", 0))
    except Exception:
        return None

def _mark_world_target_seen(target_def, source="camera"):
    """Record camera evidence for a world target and linked victim, if any."""
    if not target_def:
        return
    tick = _current_loop_tick()
    if tick is None:
        return
    target_def = str(target_def)
    # Canonical write path used by rescue gating.
    STATE.last_seen_tick_by_def[target_def] = int(tick)
    # Keep legacy alias in sync if state was reconstructed differently.
    STATE.world_target_last_seen_tick[target_def] = int(tick)
    victim_id = STATE.world_target_to_victim_id.get(target_def)
    if victim_id is not None:
        v = find_victim_by_id(victim_id)
        if v is not None:
            v["last_seen_tick"] = int(tick)
            v["last_seen_source"] = str(source)
    if DEBUG_RESCUE:
        print(f"DETECT_SEEN tick={int(tick)} def={target_def} last_seen_tick={int(tick)}")
    _rescue_trace(
        "evidence",
        "TARGET_SEEN",
        candidate_world_DEF=target_def,
        seen_tick=int(tick),
        source=source,
    )

def _active_goal_supports_target(victim=None, target_def=None, target_xy=None):
    """Return `(bool, reason)` when current rescue goal supports rescuing this target."""
    if STATE.goal is None or STATE.goal_kind != "rescue":
        return False, "no_rescue_goal"

    if victim is not None and STATE.active_rescue_id is not None:
        try:
            if int(victim.get("id", -1)) == int(STATE.active_rescue_id):
                return True, "active_rescue_id_matches_victim"
        except Exception:
            pass

    if target_def:
        mapped_id = STATE.world_target_to_victim_id.get(str(target_def))
        if mapped_id is not None and STATE.active_rescue_id is not None:
            try:
                if int(mapped_id) == int(STATE.active_rescue_id):
                    return True, "active_rescue_id_matches_target_def"
            except Exception:
                pass

    if target_xy is not None and STATE.goal is not None:
        try:
            gx, gy = float(STATE.goal[0]), float(STATE.goal[1])
            tx, ty = float(target_xy[0]), float(target_xy[1])
            if math.hypot(gx - tx, gy - ty) <= float(RESCUE_GOAL_MATCH_TOL_M):
                return True, "goal_xy_matches_target"
        except Exception:
            pass

    return False, "goal_not_matching_target"

def _evaluate_close_range_rescue_trigger(
    caller_fn,
    target_xy,
    *,
    victim=None,
    target_def=None,
    target_xy_source="unknown",
    reason="",
):
    """Single source of truth for final close-range rescue decision (distance + evidence)."""
    result = {
        "ok": False,
        "first_fail": "unknown",
        "dist_m": float("nan"),
        "threshold_m": float(RESCUE_RADIUS_M),
        "dx": float("nan"),
        "dy": float("nan"),
        "recent_seen": False,
        "recent_seen_age_ticks": None,
        "goal_evidence": False,
        "lidar_evidence": False,
        "goal_evidence_reason": "unknown",
        "matching_success": False,
        "already_rescued_gate": True,
        "evidence_gate": False,
        "evidence_ok": False,
    }

    pose_valid = _pose_xy_valid(STATE.pose)
    tx_valid = target_xy is not None
    if tx_valid:
        try:
            tx = float(target_xy[0])
            ty = float(target_xy[1])
            tx_valid = math.isfinite(tx) and math.isfinite(ty)
        except Exception:
            tx_valid = False
            tx, ty = float("nan"), float("nan")
    else:
        tx, ty = float("nan"), float("nan")

    if pose_valid:
        rx, ry = float(STATE.pose["x"]), float(STATE.pose["y"])
    else:
        rx, ry = float("nan"), float("nan")

    if pose_valid and tx_valid:
        dx = tx - rx
        dy = ty - ry
        dist_m = math.hypot(dx, dy)
    else:
        dx = dy = dist_m = float("nan")

    result.update({"dx": dx, "dy": dy, "dist_m": dist_m, "robot_xy": (rx, ry), "target_xy": (tx, ty)})

    matching_success = bool(target_def) or (victim is not None and tx_valid) or (target_xy_source == "victim_registry" and tx_valid)
    result["matching_success"] = bool(matching_success)

    recent_seen = False
    recent_seen_age = None
    seen_tick = None
    stable_def_key = None
    if target_def:
        stable_def_key = str(target_def)
    elif isinstance(victim, dict) and victim.get("world_target_def"):
        stable_def_key = str(victim.get("world_target_def"))
    elif isinstance(victim, dict):
        # Fallback: resolve DEF through stable registry mapping DEF -> victim_id.
        try:
            victim_id = int(victim.get("id", -1))
            for _def_name, _mapped_id in STATE.world_target_to_victim_id.items():
                if int(_mapped_id) == victim_id:
                    stable_def_key = str(_def_name)
                    break
        except Exception:
            stable_def_key = stable_def_key

    if stable_def_key:
        seen_dict = getattr(STATE, "last_seen_tick_by_def", STATE.world_target_last_seen_tick)
        seen_tick = seen_dict.get(stable_def_key)
    now_tick = _current_loop_tick()
    if seen_tick is not None and now_tick is not None:
        recent_seen_age = int(now_tick) - int(seen_tick)
        recent_seen = (recent_seen_age >= 0) and (recent_seen_age <= int(RECENT_SEEN_TICKS))
    result["recent_seen"] = bool(recent_seen)
    result["recent_seen_age_ticks"] = recent_seen_age

    goal_evidence, goal_evidence_reason = _active_goal_supports_target(victim=victim, target_def=target_def, target_xy=(tx, ty) if tx_valid else None)
    result["goal_evidence"] = bool(goal_evidence)
    result["goal_evidence_reason"] = str(goal_evidence_reason)

    # Optional LiDAR evidence hook: if a caller attaches `lidar_evidence` to the
    # current victim record, it participates in the OR evidence gate.
    lidar_evidence = False
    if isinstance(victim, dict):
        lidar_evidence = bool(victim.get("lidar_evidence", False))
    result["lidar_evidence"] = bool(lidar_evidence)

    evidence_gate = bool(recent_seen or goal_evidence or lidar_evidence)
    result["evidence_gate"] = evidence_gate
    result["evidence_ok"] = evidence_gate

    already_rescued_gate = True
    if victim is not None:
        try:
            already_rescued_gate = already_rescued_gate and (int(victim.get("id", -1)) not in STATE.rescued_victim_ids)
        except Exception:
            already_rescued_gate = False
    if target_def:
        td = str(target_def)
        already_rescued_gate = already_rescued_gate and (td not in STATE.rescued_world_target_defs) and (td not in _WORLD_TARGET_HIDDEN)
    result["already_rescued_gate"] = bool(already_rescued_gate)

    distance_ok = bool(math.isfinite(dist_m) and dist_m <= float(RESCUE_RADIUS_M))

    # Final gate order:
    # 1) distance must pass
    # 2) evidence_any (recent OR goal OR lidar) must pass
    # 3) already_rescued must pass
    # Keep candidate/matching checks explicit as preconditions, but do not let
    # `recent_seen=false` fail when other evidence is true.
    gates = [
        ("supervisor_available", _SUPERVISOR_ENABLED),
        ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS) if _SUPERVISOR_ENABLED else "NA"),
        ("candidate_target_exists", tx_valid),
        ("matching_success", matching_success),
        ("cooldown_throttle", "PASS(no_gate)"),
        ("distance", distance_ok),
        ("evidence_any", evidence_gate),
        ("already_rescued", already_rescued_gate),
    ]

    first_fail = "none"
    if not tx_valid:
        first_fail = "candidate_target_exists"
    elif not matching_success:
        first_fail = "matching_success"
    elif not distance_ok:
        first_fail = "distance"
    elif not evidence_gate:
        first_fail = "evidence_any"
    elif not already_rescued_gate:
        first_fail = "already_rescued"

    result["first_fail"] = first_fail
    result["ok"] = (first_fail == "none")

    _rescue_trace(
        caller_fn,
        "DIST_CHECK",
        reason=reason,
        candidate_victim_id=(None if victim is None else int(victim.get("id", -1))),
        candidate_world_DEF=target_def,
        robot_xy=f"({rx:.2f},{ry:.2f})" if math.isfinite(rx) and math.isfinite(ry) else "none",
        target_xy=f"({tx:.2f},{ty:.2f})" if tx_valid else "none",
        dx=dx,
        dy=dy,
        dist_m=dist_m,
        threshold_m=RESCUE_RADIUS_M,
        frame="world_xy_m",
        target_xy_source=target_xy_source,
        recent_seen=recent_seen,
        recent_seen_age_ticks=recent_seen_age,
        last_seen_tick=seen_tick,
        goal_evidence=goal_evidence,
        lidar_evidence=lidar_evidence,
        goal_evidence_reason=goal_evidence_reason,
        evidence_ok=evidence_gate,
        stable_def_key=stable_def_key,
        decision=("PASS" if result["ok"] else "FAIL"),
    )
    _rescue_gate_table(
        caller_fn,
        "EVAL",
        gates,
        reason=reason,
        candidate_victim_id=(None if victim is None else int(victim.get("id", -1))),
        candidate_world_DEF=target_def,
        dist_m=dist_m,
        threshold_m=RESCUE_RADIUS_M,
        target_xy_source=target_xy_source,
        recent_seen_age_ticks=recent_seen_age,
        recent_seen=recent_seen,
        goal_evidence=goal_evidence,
        lidar_evidence=lidar_evidence,
        evidence_ok=evidence_gate,
        decision=("PASS" if result["ok"] else "FAIL"),
        goal_evidence_reason=goal_evidence_reason,
    )
    _rescue_attempt_summary(caller_fn, victim, target_def, result)
    return result

def _check_rescue_contract_after_success(victim_id, rescued_before, target_def, rescue_dist_m=None):
    """Post-rescue invariant check; logs only on failure."""
    fail_reasons = []
    victim = find_victim_by_id(victim_id)
    victim_exists = victim is not None
    victim_rescued_flag = bool(victim.get("rescued", False)) if victim_exists else False

    rescued_after = len(STATE.rescued_victim_ids)
    count_delta_ok = (rescued_after == int(rescued_before) + 1)

    target_def_str = str(target_def) if target_def else None
    handle_exists = bool(target_def_str and (target_def_str in _WORLD_TARGET_TRANS_FIELDS))
    target_marked_hidden_state = bool(target_def_str and (target_def_str in _WORLD_TARGET_HIDDEN))
    target_marked_rescued_state = bool(target_def_str and (target_def_str in STATE.rescued_world_target_defs))

    target_xyz_post = None
    target_removed_by_pose = False
    if target_def_str:
        try:
            pos = _world_target_xy(target_def_str)
        except Exception:
            pos = None
        if pos is not None:
            try:
                target_xyz_post = (float(pos[0]), float(pos[1]), float(pos[2]))
                if all(math.isfinite(v) for v in target_xyz_post):
                    # hide_world_rescue_target moves target far away and below ground
                    target_removed_by_pose = (
                        (target_xyz_post[2] <= -1.0)
                        or (
                            target_xyz_post[0] >= float(WORLD_TARGET_HIDE_BASE_X) - 5.0
                            and target_xyz_post[1] >= float(WORLD_TARGET_HIDE_BASE_Y) - 5.0
                        )
                    )
            except Exception:
                target_xyz_post = None

    target_hidden_or_removed_ok = bool(target_marked_hidden_state or target_removed_by_pose)

    yielded_defs = []
    target_not_yielded_ok = False if target_def_str else False
    try:
        yielded_defs = [str(name) for name, _ in iter_unrescued_world_targets()]
        if target_def_str:
            target_not_yielded_ok = (target_def_str not in yielded_defs)
    except Exception as e:
        fail_reasons.append(f"iter_unrescued_exception:{e}")

    if not victim_exists:
        fail_reasons.append("victim_record_missing")
    if victim_exists and not victim_rescued_flag:
        fail_reasons.append("victim_rescued_flag_false")
    if not count_delta_ok:
        fail_reasons.append(f"rescued_count_delta_not_1:{rescued_after-int(rescued_before)}")
    if not target_def_str:
        fail_reasons.append("victim_def_missing")
    if target_def_str and not target_hidden_or_removed_ok:
        fail_reasons.append("target_not_hidden_or_removed")
    if target_def_str and not target_not_yielded_ok:
        fail_reasons.append("target_still_yielded")

    if not fail_reasons:
        return True

    tick = _current_loop_tick()
    rescued_ids_sorted = ",".join(str(v) for v in sorted(int(v) for v in STATE.rescued_victim_ids)) if STATE.rescued_victim_ids else "none"
    msg_parts = [
        "RESCUE_CONTRACT_FAIL",
        f"tick={_rescue_trace_value(tick)}",
        f"time={_rescue_trace_value(robot.getTime())}",
        f"victim_id={_rescue_trace_value(victim_id)}",
        f"victim_def={_rescue_trace_value(target_def_str)}",
        f"dist_m={_rescue_trace_value(rescue_dist_m)}",
        f"supervisor={_rescue_trace_value(_SUPERVISOR_ENABLED)}",
        f"has_handles={_rescue_trace_value(bool(_WORLD_TARGET_TRANS_FIELDS))}",
        f"def_handle={_rescue_trace_value(handle_exists)}",
        f"victim_exists={_rescue_trace_value(victim_exists)}",
        f"victim_rescued={_rescue_trace_value(victim_rescued_flag)}",
        f"rescued_before={_rescue_trace_value(rescued_before)}",
        f"rescued_after={_rescue_trace_value(rescued_after)}",
        f"hidden_state={_rescue_trace_value(target_marked_hidden_state)}",
        f"rescued_mark_state={_rescue_trace_value(target_marked_rescued_state)}",
        f"removed_by_pose={_rescue_trace_value(target_removed_by_pose)}",
        f"not_yielded={_rescue_trace_value(target_not_yielded_ok)}",
        f"current_rescued={rescued_ids_sorted}",
        f"fail_reasons={','.join(fail_reasons)}",
    ]
    if target_xyz_post is not None:
        msg_parts.append(f"target_xyz_post=({target_xyz_post[0]:.2f},{target_xyz_post[1]:.2f},{target_xyz_post[2]:.2f})")
    if yielded_defs:
        msg_parts.append(f"yielded_defs={','.join(yielded_defs)}")
    print(" ".join(msg_parts))
    log_event(
        "rescue_contract_fail",
        {
            "tick": tick,
            "victim_id": int(victim_id),
            "victim_def": target_def_str,
            "distance_at_rescue_m": rescue_dist_m,
            "has_supervisor": bool(_SUPERVISOR_ENABLED),
            "has_handles": bool(_WORLD_TARGET_TRANS_FIELDS),
            "def_handle_exists": handle_exists,
            "victim_exists": victim_exists,
            "victim_rescued": victim_rescued_flag,
            "rescued_before": int(rescued_before),
            "rescued_after": int(rescued_after),
            "hidden_state": target_marked_hidden_state,
            "rescued_mark_state": target_marked_rescued_state,
            "removed_by_pose": target_removed_by_pose,
            "not_yielded": target_not_yielded_ok,
            "current_rescued_ids": [int(v) for v in sorted(int(v) for v in STATE.rescued_victim_ids)],
            "yielded_defs": yielded_defs,
            "target_xyz_post": list(target_xyz_post) if target_xyz_post is not None else None,
            "fail_reasons": fail_reasons,
        },
    )
    return False

def _ensure_victim_for_world_target(def_name):
    """Return/create a victim record mapped to a specific world rescue target DEF."""
    global _m5_next_id
    if not def_name:
        return None
    mapped_id = STATE.world_target_to_victim_id.get(def_name)
    if mapped_id is not None:
        return find_victim_by_id(mapped_id)
    pos = _world_target_xy(def_name)
    if pos is None:
        return None
    victim_id = int(_m5_next_id)
    _m5_next_id += 1
    det = {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "id": victim_id,
        "note": "world_target_proxy",
        "confidence": 1.0,
        "victim_type": "world_target",
        "world_target_def": str(def_name),
        "rescued": False,
        "detected_at_t": robot.getTime(),
        "rescued_at_t": None,
    }
    STATE.victims.append(det)
    STATE.world_target_to_victim_id[str(def_name)] = victim_id
    log_event("victim", {"id": victim_id, "x": det["x"], "y": det["y"], "source": "world_target_proxy"})
    print(f"Victim target mapped: {def_name} -> victim #{victim_id}")
    return det

def maybe_rescue_near_world_target(reason):
    """Proximity rescue fallback using actual world target positions (Supervisor mode)."""
    global _LAST_WORLD_TARGET_HANDLE_RETRY_T
    _rescue_trace(
        "maybe_rescue_near_world_target",
        "ENTER",
        reason=reason,
        has_supervisor=_SUPERVISOR_ENABLED,
        has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        rescued_count_before=len(STATE.rescued_victim_ids),
    )
    if not _pose_xy_valid(STATE.pose):
        _rescue_gate_table(
            "maybe_rescue_near_world_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", None),
                ("matching_success", None),
                ("already_rescued", None),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
                ("pose_valid", False),
            ],
            reason="invalid_pose",
        )
        _rescue_trace(
            "maybe_rescue_near_world_target",
            "RETURN",
            reason="invalid_pose",
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        )
        return False
    if not _WORLD_TARGET_TRANS_FIELDS:
        # Lazy retry in case supervisor handles were not ready at startup or world was reset.
        if _SUPERVISOR_ENABLED and (robot.getTime() - _LAST_WORLD_TARGET_HANDLE_RETRY_T) > 1.0:
            _LAST_WORLD_TARGET_HANDLE_RETRY_T = robot.getTime()
            _init_world_target_handles()
        _rescue_gate_table(
            "maybe_rescue_near_world_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", False),
                ("candidate_target_exists", None),
                ("matching_success", None),
                ("already_rescued", None),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
            ],
            reason="no_handles",
        )
        _rescue_trace(
            "maybe_rescue_near_world_target",
            "RETURN",
            reason="no_handles",
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=False,
        )
        return False
    rx, ry = float(STATE.pose["x"]), float(STATE.pose["y"])
    nearest_def = None
    nearest_pos = None
    nearest_dist = float("inf")
    for def_name in RESCUE_TARGET_DEFS:
        if def_name in STATE.rescued_world_target_defs or def_name in _WORLD_TARGET_HIDDEN:
            continue
        pos = _world_target_xy(def_name)
        if pos is None:
            continue
        d = _dist_xy(rx, ry, pos[0], pos[1])
        if d < nearest_dist:
            nearest_dist = d
            nearest_def = def_name
            nearest_pos = pos
    rescue_radius = max(float(MISSION_RESCUE_RADIUS_M), float(WORLD_TARGET_PROXIMITY_RESCUE_RADIUS_M))
    if nearest_def is None:
        _rescue_gate_table(
            "maybe_rescue_near_world_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", False),
                ("matching_success", None),
                ("already_rescued", None),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
            ],
            reason="no_candidate_target",
            threshold_m=rescue_radius,
        )
        _rescue_trace(
            "maybe_rescue_near_world_target",
            "RETURN",
            reason="no_candidate_target",
            threshold_m=rescue_radius,
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            match_result="none",
        )
        return False
    if nearest_pos is not None:
        tx, ty = float(nearest_pos[0]), float(nearest_pos[1])
    else:
        tx, ty = float("nan"), float("nan")
    dx = tx - rx
    dy = ty - ry
    _rescue_trace(
        "maybe_rescue_near_world_target",
        "DIST_CHECK",
        candidate_world_DEF=nearest_def,
        robot_xy=f"({rx:.2f},{ry:.2f})",
        target_xy=f"({tx:.2f},{ty:.2f})",
        dx=dx,
        dy=dy,
        dist_m=nearest_dist,
        threshold_m=rescue_radius,
        frame="world_xy_m",
    )
    if nearest_dist > rescue_radius:
        _rescue_gate_table(
            "maybe_rescue_near_world_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", True),
                ("matching_success", True),
                ("already_rescued", True),  # candidate list already excludes rescued/hidden targets
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", False),
            ],
            reason="too_far",
            candidate_world_DEF=nearest_def,
            dist_m=nearest_dist,
            threshold_m=rescue_radius,
            match_result="nearest_target",
        )
        _rescue_trace(
            "maybe_rescue_near_world_target",
            "RETURN",
            reason="too_far",
            candidate_world_DEF=nearest_def,
            robot_xy=f"({rx:.2f},{ry:.2f})",
            target_xy=f"({tx:.2f},{ty:.2f})",
            dx=dx,
            dy=dy,
            dist_m=nearest_dist,
            threshold_m=rescue_radius,
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            match_result="nearest_target",
        )
        return False
    victim = _ensure_victim_for_world_target(nearest_def)
    if victim is None:
        _rescue_gate_table(
            "maybe_rescue_near_world_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", True),
                ("matching_success", False),
                ("already_rescued", True),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", nearest_dist <= rescue_radius),
            ],
            reason="no_victim_proxy",
            candidate_world_DEF=nearest_def,
            dist_m=nearest_dist,
            threshold_m=rescue_radius,
            match_result="nearest_target",
        )
        _rescue_trace(
            "maybe_rescue_near_world_target",
            "RETURN",
            reason="no_victim_proxy",
            candidate_world_DEF=nearest_def,
            dist_m=nearest_dist,
            threshold_m=rescue_radius,
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            match_result="nearest_target",
        )
        return False
    gate_eval = _evaluate_close_range_rescue_trigger(
        "maybe_rescue_near_world_target",
        (tx, ty),
        victim=victim,
        target_def=nearest_def,
        target_xy_source="world_target_xy",
        reason=f"{reason}:final_gate",
    )
    if not bool(gate_eval.get("ok", False)):
        _rescue_trace(
            "maybe_rescue_near_world_target",
            "RETURN",
            reason=f"close_range_gate_failed:{gate_eval.get('first_fail', 'unknown')}",
            candidate_victim_id=int(victim["id"]),
            candidate_world_DEF=nearest_def,
            robot_xy=f"({rx:.2f},{ry:.2f})",
            target_xy=f"({tx:.2f},{ty:.2f})",
            dx=gate_eval.get("dx"),
            dy=gate_eval.get("dy"),
            dist_m=gate_eval.get("dist_m"),
            threshold_m=gate_eval.get("threshold_m"),
            recent_seen=gate_eval.get("recent_seen"),
            recent_seen_age_ticks=gate_eval.get("recent_seen_age_ticks"),
            goal_evidence=gate_eval.get("goal_evidence"),
            goal_evidence_reason=gate_eval.get("goal_evidence_reason"),
            evidence_any=gate_eval.get("evidence_gate"),
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        )
        return False
    rescued_before = len(STATE.rescued_victim_ids)
    rescued = rescue_victim(victim["id"], reason)
    if rescued and STATE.goal is not None and STATE.goal_kind == "rescue":
        clear_goal(reason="world_target_proximity_rescue")
    if rescued:
        _rescue_gate_table(
            "maybe_rescue_near_world_target",
            "SUCCESS",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", True),
                ("matching_success", True),
                ("already_rescued", True),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", True),
                ("rescue_victim_call", True),
            ],
            candidate_victim_id=int(victim["id"]),
            candidate_world_DEF=nearest_def,
            dist_m=gate_eval.get("dist_m", nearest_dist),
            threshold_m=gate_eval.get("threshold_m", RESCUE_RADIUS_M),
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
        _rescue_trace(
            "maybe_rescue_near_world_target",
            "SUCCESS",
            candidate_victim_id=int(victim["id"]),
            candidate_world_DEF=nearest_def,
            robot_xy=f"({rx:.2f},{ry:.2f})",
            target_xy=f"({tx:.2f},{ty:.2f})",
            dx=gate_eval.get("dx", dx),
            dy=gate_eval.get("dy", dy),
            dist_m=gate_eval.get("dist_m", nearest_dist),
            threshold_m=gate_eval.get("threshold_m", RESCUE_RADIUS_M),
            match_result="nearest_target",
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
    else:
        _rescue_gate_table(
            "maybe_rescue_near_world_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", True),
                ("matching_success", True),
                ("already_rescued", True),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", True),
                ("rescue_victim_call", False),
            ],
            reason="rescue_victim_failed",
            candidate_victim_id=int(victim["id"]),
            candidate_world_DEF=nearest_def,
            dist_m=gate_eval.get("dist_m", nearest_dist),
            threshold_m=gate_eval.get("threshold_m", RESCUE_RADIUS_M),
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
        _rescue_trace(
            "maybe_rescue_near_world_target",
            "RETURN",
            reason="rescue_victim_failed",
            candidate_victim_id=int(victim["id"]),
            candidate_world_DEF=nearest_def,
            robot_xy=f"({rx:.2f},{ry:.2f})",
            target_xy=f"({tx:.2f},{ty:.2f})",
            dx=gate_eval.get("dx", dx),
            dy=gate_eval.get("dy", dy),
            dist_m=gate_eval.get("dist_m", nearest_dist),
            threshold_m=gate_eval.get("threshold_m", RESCUE_RADIUS_M),
            match_result="nearest_target",
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
    return bool(rescued)

def nearest_unrescued_victim(pose):
    pending = [v for v in STATE.victims if int(v["id"]) not in STATE.rescued_victim_ids]
    if not pending:
        return None
    if not _pose_xy_valid(pose):
        return min(pending, key=lambda v: int(v["id"]))
    return min(pending, key=lambda v: _dist_xy(pose["x"], pose["y"], v["x"], v["y"]))

def rescue_victim(victim_id, reason):
    rescued_before = len(STATE.rescued_victim_ids)
    _rescue_trace(
        "rescue_victim",
        "ENTER",
        candidate_victim_id=victim_id,
        has_supervisor=_SUPERVISOR_ENABLED,
        has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        rescued_count_before=rescued_before,
        reason=reason,
    )
    victim = find_victim_by_id(victim_id)
    if victim is None:
        _rescue_gate_table(
            "rescue_victim",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", False),
                ("matching_success", None),
                ("already_rescued", None),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
            ],
            reason="victim_not_found",
            candidate_victim_id=victim_id,
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
        _rescue_trace(
            "rescue_victim",
            "RETURN",
            reason="victim_not_found",
            candidate_victim_id=victim_id,
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
        return False
    victim_id = int(victim["id"])
    if victim_id in STATE.rescued_victim_ids:
        _rescue_gate_table(
            "rescue_victim",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", True),
                ("matching_success", True),
                ("already_rescued", False),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", True),
            ],
            reason="already_rescued",
            candidate_victim_id=victim_id,
            candidate_world_DEF=victim.get("world_target_def"),
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
        _rescue_trace(
            "rescue_victim",
            "RETURN",
            reason="already_rescued",
            candidate_victim_id=victim_id,
            candidate_world_DEF=victim.get("world_target_def"),
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
        return False

    STATE.rescued_victim_ids.add(victim_id)
    STATE.rescued_sites.append((float(victim["x"]), float(victim["y"])))
    target_def = victim.get("world_target_def")
    if target_def is not None:
        target_def = str(target_def)
        victim["world_target_def"] = target_def
    match_result = "provided" if target_def else "none"
    if target_def and (target_def not in RESCUE_TARGET_DEFS):
        _rescue_trace(
            "rescue_victim",
            "ASSOC_WARN",
            reason="invalid_stored_world_target_def",
            candidate_victim_id=victim_id,
            candidate_world_DEF=target_def,
        )
        target_def = None
        victim["world_target_def"] = None
        match_result = "invalid_stored_def"
    if not target_def:
        nearest_target = match_world_rescue_target(
            float(victim["x"]),
            float(victim["y"]),
            max_dist_m=WORLD_TARGET_RESCUE_FALLBACK_MATCH_RADIUS_M,
            query_source="rescue_victim:fallback_xy",
        )
        if nearest_target is not None:
            target_def = nearest_target["def_name"]
            victim["world_target_def"] = target_def
            match_result = "matched_nearest"
            _rescue_trace(
                "rescue_victim",
                "ASSOC_FALLBACK_MATCH",
                candidate_victim_id=victim_id,
                candidate_world_DEF=target_def,
                dist_m=nearest_target.get("distance_m"),
                threshold_m=WORLD_TARGET_RESCUE_FALLBACK_MATCH_RADIUS_M,
            )
        else:
            match_result = "unmatched"
    rescue_dist_m = None
    if _pose_xy_valid(STATE.pose):
        try:
            tx_for_dist = float(victim["x"])
            ty_for_dist = float(victim["y"])
            if target_def:
                live_target_pos = _world_target_xy(str(target_def))
                if live_target_pos is not None:
                    tx_for_dist = float(live_target_pos[0])
                    ty_for_dist = float(live_target_pos[1])
            rescue_dist_m = _dist_xy(float(STATE.pose["x"]), float(STATE.pose["y"]), tx_for_dist, ty_for_dist)
        except Exception:
            rescue_dist_m = None
    hide_ok = None
    if target_def:
        target_xy_for_viz = _world_target_xy(str(target_def))
        if target_xy_for_viz is None:
            target_xy_for_viz = (float(victim["x"]), float(victim["y"]), 0.0)
        STATE.rescued_target_positions[str(target_def)] = (float(target_xy_for_viz[0]), float(target_xy_for_viz[1]))
        STATE.rescued_world_target_defs.add(str(target_def))
        hide_ok = hide_world_rescue_target(str(target_def))
        if not hide_ok:
            print(f"WORLD_TARGET_NOT_HIDDEN def={target_def}")
    victim["rescued"] = True
    victim["rescued_at_t"] = robot.getTime()
    log_event("victim_rescued", {
        "id": victim_id,
        "x": victim["x"],
        "y": victim["y"],
        "world_target_def": target_def,
        "rescued_count": len(STATE.rescued_victim_ids),
        "required": STATE.required_victims,
        "reason": reason,
    })
    print(f"Rescued victim #{victim_id} ({len(STATE.rescued_victim_ids)}/{STATE.required_victims})")
    _rescue_gate_table(
        "rescue_victim",
        "SUCCESS",
        [
            ("supervisor_available", _SUPERVISOR_ENABLED),
            ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
            ("candidate_target_exists", True),
            ("matching_success", match_result != "unmatched"),
            ("already_rescued", True),
            ("cooldown_throttle", True),
            ("distance_le_rescue_radius", True),
            ("hide_success_if_attempted", (hide_ok is True) if target_def else True),
        ],
        candidate_victim_id=victim_id,
        candidate_world_DEF=target_def,
        match_result=match_result,
        hide_attempted=bool(target_def),
        hide_result=hide_ok,
        rescued_count_before=rescued_before,
        rescued_count_after=len(STATE.rescued_victim_ids),
    )
    _rescue_trace(
        "rescue_victim",
        "SUCCESS",
        candidate_victim_id=victim_id,
        candidate_world_DEF=target_def,
        match_result=match_result,
        has_supervisor=_SUPERVISOR_ENABLED,
        has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        rescued_count_before=rescued_before,
        rescued_count_after=len(STATE.rescued_victim_ids),
        hide_attempted=bool(target_def),
        hide_result=hide_ok,
    )
    _check_rescue_contract_after_success(
        victim_id=victim_id,
        rescued_before=rescued_before,
        target_def=target_def,
        rescue_dist_m=rescue_dist_m,
    )
    return True

# ---- Stubs for other members to implement later
#member 2 start here
MAP_RESOLUTION = 0.1   # meters per cell
MAP_SIZE = 640         # 64m x 64m coverage (fits the 60m arena walls)

def init_map():
    return np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int8)

def world_to_grid(x, y):
    gx = int(x / MAP_RESOLUTION + MAP_SIZE // 2)
    gy = int(y / MAP_RESOLUTION + MAP_SIZE // 2)
    return gx, gy

def update_map(lidar_ranges, pose, map_state, ray_stride=1):
    if map_state is None:
        map_state = init_map()

    x, y, yaw = pose["x"], pose["y"], pose["yaw"]
    if np.isnan(x) or np.isnan(y) or np.isnan(yaw):
        return map_state

    stride = max(1, int(ray_stride))
    n_ranges = len(lidar_ranges)
    for i in range(0, n_ranges, stride):
        r = lidar_ranges[i]
        if r >= max_range:
            continue
        angle = yaw + lidar.getFov() * (i / n_ranges - 0.5)
        end_x = x + r * math.cos(angle)
        end_y = y + r * math.sin(angle)

        gx, gy = world_to_grid(end_x, end_y)

        steps = int(r / MAP_RESOLUTION)
        for s in range(steps):
            fx = x + s * MAP_RESOLUTION * math.cos(angle)
            fy = y + s * MAP_RESOLUTION * math.sin(angle)
            ix, iy = world_to_grid(fx, fy)
            if 0 <= ix < MAP_SIZE and 0 <= iy < MAP_SIZE:
                map_state[ix, iy] = 0  # free

        if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
            map_state[gx, gy] = 100  # occupied

    return map_state

def inflate_obstacles(map_state, inflation_radius=2):
    """
    Inflate occupied cells by a given radius (in grid cells).
    Ensures path planning avoids obstacles with a safety margin.
    """
    inflated = map_state.copy()
    occ_cells = np.argwhere(map_state == 100)

    for (x, y) in occ_cells:
        for dx in range(-inflation_radius, inflation_radius + 1):
            for dy in range(-inflation_radius, inflation_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                    inflated[nx, ny] = 100
    return inflated
    
#member 2 done here

EXPLORER = FrontierExplorer(safety_radius=3)

def _grid_to_world(ix, iy):
    return (float(ix) - MAP_SIZE // 2) * MAP_RESOLUTION, (float(iy) - MAP_SIZE // 2) * MAP_RESOLUTION

def _clamp_exploration_goal_xy(wx, wy):
    lim = float(EXPLORATION_GOAL_MAX_ABS_M)
    gx = _clamp(float(wx), -lim, lim)
    gy = _clamp(float(wy), -lim, lim)
    # Avoid sticky corner goals where both coordinates are at boundary.
    corner_band_m = 0.6
    corner_pull_m = 1.4
    if abs(gx) >= (lim - corner_band_m) and abs(gy) >= (lim - corner_band_m):
        gx = math.copysign(max(0.0, lim - corner_pull_m), gx)
        gy = math.copysign(max(0.0, lim - corner_pull_m), gy)
    return gx, gy

def _sanitize_exploration_goal(goal_xy, map_state):
    """Keep exploration goals inside arena-safe bounds and off occupied cells."""
    if goal_xy is None:
        return None
    gx, gy = _clamp_exploration_goal_xy(goal_xy[0], goal_xy[1])
    ix, iy = world_to_grid(gx, gy)
    if 0 <= ix < MAP_SIZE and 0 <= iy < MAP_SIZE and int(map_state[ix, iy]) != 100:
        return (gx, gy)

    # If clamped cell is occupied, search nearby free cells.
    search_cells = max(1, int(round(1.2 / MAP_RESOLUTION)))
    best = None
    best_d = float("inf")
    for dx in range(-search_cells, search_cells + 1):
        for dy in range(-search_cells, search_cells + 1):
            nx, ny = ix + dx, iy + dy
            if not (0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE):
                continue
            if int(map_state[nx, ny]) == 100:
                continue
            wx2, wy2 = _grid_to_world(nx, ny)
            wx2, wy2 = _clamp_exploration_goal_xy(wx2, wy2)
            d = math.hypot(wx2 - gx, wy2 - gy)
            if d < best_d:
                best_d = d
                best = (wx2, wy2)
    return best

def choose_frontier_goal(map_state, pose):
    """Score-based frontier policy with a minimum travel guard."""
    if map_state is None:
        return None
    if not _pose_xy_valid(pose):
        return None

    rx = float(pose["x"])
    ry = float(pose["y"])
    min_travel_m = max(0.05, float(EXPLORATION_MIN_GOAL_TRAVEL_M))

    # Map robot pose from world to grid coordinates.
    gx, gy = world_to_grid(pose["x"], pose["y"])
    robot_cell = (gx, gy)

    # Use FrontierExplorer's score-based selection (balances size/safety/distance).
    best_grid_goal = EXPLORER.select_next_goal(map_state, robot_cell)
    if best_grid_goal is None:
        return None

    wx_raw, wy_raw = _grid_to_world(best_grid_goal[0], best_grid_goal[1])
    candidate_goal = _sanitize_exploration_goal((wx_raw, wy_raw), map_state)
    if candidate_goal is None:
        return None

    # Prevent zero-distance/near-zero goals that can stall the robot at startup.
    if _dist_xy(rx, ry, float(candidate_goal[0]), float(candidate_goal[1])) >= min_travel_m:
        if _explore_goal_block_reason(candidate_goal[0], candidate_goal[1]) is None:
            return candidate_goal

    # Fallback: choose nearest frontier cell that is not near victim locations.
    frontiers = EXPLORER.detect_frontiers(map_state)
    if not frontiers:
        return None
    ranked = sorted(frontiers, key=lambda cell: EXPLORER.compute_distance(robot_cell, cell))
    for cell in ranked:
        wx_raw, wy_raw = _grid_to_world(cell[0], cell[1])
        candidate = _sanitize_exploration_goal((wx_raw, wy_raw), map_state)
        if candidate is None:
            continue
        if _dist_xy(rx, ry, float(candidate[0]), float(candidate[1])) < min_travel_m:
            continue
        if _explore_goal_block_reason(candidate[0], candidate[1]) is not None:
            continue
        return candidate
    return None

def jitter_exploration_goal(goal_xy, map_state, max_jitter_m=EXPLORATION_GOAL_JITTER_M):
    """Add bounded randomness to exploration goals so trajectories are less rigid."""
    if goal_xy is None or map_state is None:
        return goal_xy
    base_safe = _sanitize_exploration_goal(goal_xy, map_state)
    if base_safe is None:
        return None
    if float(max_jitter_m) <= 1e-6:
        return base_safe
    gx, gy = float(base_safe[0]), float(base_safe[1])
    best = base_safe
    # Try several random offsets and keep the first non-occupied in-bounds sample.
    for _ in range(8):
        r = random.uniform(0.2, max(0.2, float(max_jitter_m)))
        a = random.uniform(-math.pi, math.pi)
        cx = gx + r * math.cos(a)
        cy = gy + r * math.sin(a)
        candidate = _sanitize_exploration_goal((cx, cy), map_state)
        if candidate is not None:
            return candidate
    return best

def plan_path(map_state, start_pose, goal_xy):
    # NavigationStack handles planning
    return None

def follow_path(path, pose):
    # NavigationStack handles following
    return None

# =========================================================
# Member 5 — Victim Detection Implementation
# =========================================================
def _m5_webots_frame(camera_dev):
    """Convert Webots camera image → OpenCV BGR numpy array."""
    w   = camera_dev.getWidth()
    h   = camera_dev.getHeight()
    raw = camera_dev.getImage()          # bytes: BGRA order in Webots
    img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def _m5_detect_blobs(frame):
    """HSV colour filtering + contour detection. Returns list of (cx_norm, cy_norm)."""
    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = (
        cv2.inRange(hsv, M5_RED_LO1, M5_RED_HI1)
        | cv2.inRange(hsv, M5_RED_LO2, M5_RED_HI2)
        | cv2.inRange(hsv, M5_GRN_LO,  M5_GRN_HI)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        if cv2.contourArea(cnt) < M5_MIN_AREA:
            continue
        _x, _y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        asp = bw / bh
        if asp > M5_MAX_ASPECT or (1.0 / asp) > M5_MAX_ASPECT:
            continue
        M  = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        blobs.append((M["m10"] / M["m00"] / w,   # cx normalised
                      M["m01"] / M["m00"] / h))   # cy normalised
    return blobs


def _m5_project(cx_norm, pose):
    """Convert image x-offset → approximate world (wx, wy) ahead of robot."""
    rx, ry, yaw = pose["x"], pose["y"], pose["yaw"]
    lateral     = (cx_norm - 0.5) * 2.0   # ±1 m lateral span
    wx = rx + M5_FWD_OFFSET * math.cos(yaw) - lateral * math.sin(yaw)
    wy = ry + M5_FWD_OFFSET * math.sin(yaw) + lateral * math.cos(yaw)
    return wx, wy

def _camera_sighting_key(wx, wy, target_def=None):
    """Stable key for short-lived camera sightings in visualization."""
    if target_def:
        return f"def:{str(target_def)}"
    return f"xy:{round(float(wx), 1)}:{round(float(wy), 1)}"

def _prune_camera_sightings(now_t):
    """Drop stale camera sightings so visualization shows only current camera-visible objects."""
    expiry_t = float(now_t) - float(CAMERA_SIGHTING_TTL_S)
    stale_keys = [k for k, v in STATE.camera_sightings_by_key.items() if float(v.get("t", -1e9)) < expiry_t]
    for k in stale_keys:
        STATE.camera_sightings_by_key.pop(k, None)
    stale_log_keys = [k for k, ts in STATE.last_camera_sighting_log_t_by_key.items() if float(ts) < expiry_t]
    for k in stale_log_keys:
        STATE.last_camera_sighting_log_t_by_key.pop(k, None)

def _record_camera_sighting(wx, wy, now_t, target_def=None, matched=False, source="camera"):
    """Record a camera-visible object for live graph overlays and optional event stream."""
    if not (math.isfinite(float(wx)) and math.isfinite(float(wy))):
        return
    _prune_camera_sightings(now_t)
    key = _camera_sighting_key(wx, wy, target_def=target_def)
    STATE.camera_sightings_by_key[key] = {
        "x": float(wx),
        "y": float(wy),
        "t": float(now_t),
        "matched": bool(matched),
        "target_def": (None if target_def is None else str(target_def)),
        "source": str(source),
    }

    last_t = float(STATE.last_camera_sighting_log_t_by_key.get(key, -1e9))
    if (float(now_t) - last_t) >= float(CAMERA_SIGHTING_LOG_PERIOD_S):
        STATE.last_camera_sighting_log_t_by_key[key] = float(now_t)
        log_event(
            "camera_sighting",
            {
                "key": key,
                "x": float(wx),
                "y": float(wy),
                "matched": bool(matched),
                "world_target_def": (None if target_def is None else str(target_def)),
                "source": str(source),
            },
        )


def detect_victim(camera_dev, pose):
    """
    Member 5 — full OpenCV victim detection.
    Returns list of dicts: {"x", "y", "id", "note"}
    Also writes victims_log.csv and saves victim_<id>.png screenshots.
    """
    global _m5_victims, _m5_next_id

    if np.isnan(pose["x"]) or np.isnan(pose["y"]):
        return []

    frame  = _m5_webots_frame(camera_dev)
    blobs  = _m5_detect_blobs(frame)
    found  = []

    now_t = robot.getTime()

    for cx_norm, _ in blobs:
        wx, wy = _m5_project(cx_norm, pose)
        camera_target_hint = match_world_rescue_target_from_camera(cx_norm, pose, camera_dev)
        projected_target_match = match_world_rescue_target(
            wx, wy,
            max_dist_m=WORLD_TARGET_DETECTION_MATCH_RADIUS_M,
            query_source="detect:projected_xy",
        )
        world_target = projected_target_match
        assoc_source = "projected_xy"
        if world_target is None and camera_target_hint is not None:
            # Camera matcher is a visibility/angle hint; finalize using deterministic XY->nearest association.
            world_target = match_world_rescue_target(
                float(camera_target_hint["x"]),
                float(camera_target_hint["y"]),
                max_dist_m=WORLD_TARGET_DETECTION_MATCH_RADIUS_M,
                query_source="detect:camera_hint_xy",
            )
            assoc_source = "camera_hint_xy"
        if camera_target_hint is not None and world_target is not None:
            hint_def = str(camera_target_hint.get("def_name"))
            final_def = str(world_target.get("def_name"))
            if hint_def != final_def:
                _rescue_trace(
                    "detect_victim",
                    "ASSOC_CONFLICT",
                    cx_norm=float(cx_norm),
                    projected_wx=float(wx),
                    projected_wy=float(wy),
                    camera_hint_def=hint_def,
                    camera_hint_dist=camera_target_hint.get("distance_m"),
                    final_world_DEF=final_def,
                    final_dist=world_target.get("distance_m"),
                    assoc_source=assoc_source,
                )
        if world_target is None and _WORLD_TARGET_TRANS_FIELDS:
            # In supervisor mode, require a real target match to avoid projecting
            # a far visual hit to a fake nearby rescue coordinate.
            _record_camera_sighting(
                wx,
                wy,
                now_t,
                target_def=None,
                matched=False,
                source="camera_blob_unmatched",
            )
            continue
        if world_target is not None:
            if float(world_target.get("distance_m", 0.0)) > float(M5_MAX_TARGET_DETECTION_RANGE_M):
                continue
            target_def = str(world_target["def_name"])
            if target_def in STATE.rescued_world_target_defs:
                continue
            # Refresh temporal camera evidence even when this sighting deduplicates.
            _mark_world_target_seen(target_def, source="camera_match")
            _rescue_trace(
                "detect_victim",
                "ASSOC_MATCH",
                candidate_world_DEF=target_def,
                assoc_source=assoc_source,
                dist_m=world_target.get("distance_m"),
                cx_norm=float(cx_norm),
            )
            # Show camera-visible object immediately on live graph even before dedup/registration.
            _record_camera_sighting(
                float(world_target["x"]),
                float(world_target["y"]),
                now_t,
                target_def=target_def,
                matched=True,
                source="camera_match",
            )
            if target_def in STATE.world_target_to_victim_id:
                existing_id = int(STATE.world_target_to_victim_id[target_def])
                existing_victim = find_victim_by_id(existing_id)
                if existing_victim is not None and not bool(existing_victim.get("rescued", False)):
                    continue
            # Snap detections to actual target world position to stabilize dedup/rescue logic.
            wx, wy = float(world_target["x"]), float(world_target["y"])
        else:
            target_def = None
            _record_camera_sighting(
                wx,
                wy,
                now_t,
                target_def=None,
                matched=False,
                source="camera_projected",
            )

        # Deduplication
        duplicate = any(
            math.hypot(wx - px, wy - py) < M5_DEDUP_RADIUS
            for px, py in _m5_victims
        )
        if (not duplicate) and STATE.rescued_sites:
            duplicate = any(
                math.hypot(wx - px, wy - py) < max(M5_DEDUP_RADIUS, MISSION_RESCUE_RADIUS_M)
                for px, py in STATE.rescued_sites
            )
        if duplicate:
            if VICTIM_METRICS is not None:
                VICTIM_METRICS.record_duplicate_attempt(wx, wy, now_t)
            continue

        victim_id      = _m5_next_id
        _m5_next_id   += 1
        _m5_victims.append((wx, wy))
        if target_def is not None:
            STATE.world_target_to_victim_id[target_def] = int(victim_id)

        # CSV row — victims_log.csv in RUN_DIR
        csv_path = os.path.join(RUN_DIR, "victims_log.csv")
        write_header = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["ID", "timestamp", "x", "y"])
            w.writerow([victim_id, f"{now_t:.3f}",
                        f"{wx:.4f}", f"{wy:.4f}"])

        # Screenshot
        img_path = os.path.join(RUN_DIR, f"victim_{victim_id}.png")
        cv2.imwrite(img_path, frame)

        if VICTIM_EVENT_WRITER is not None:
            VICTIM_EVENT_WRITER.write_event(
                timestamp=now_t,
                robot_pose=pose,
                victim_type="color_blob",
                victim_class="unknown_color",
                confidence=0.8,
                source="webots_controller",
                image_path=img_path,
                world_coords={"x": wx, "y": wy},
                extra={"victim_id": int(victim_id), "note": "hsv_detection", "world_target_def": target_def},
            )
        if VICTIM_METRICS is not None:
            VICTIM_METRICS.record_detection(wx, wy, now_t)

        det = {
            "x": wx,
            "y": wy,
            "id": victim_id,
            "note": "hsv_detection",
            "image_path": img_path,
            "confidence": 0.8,
            "victim_type": "color_blob",
            "world_target_def": target_def,
        }
        found.append(det)
        print(f"\U0001f6a8 Victim #{victim_id} detected at ({wx:.2f}, {wy:.2f})")

    return found

def process_victim_detections(now_t):
    """Run camera detection and register new victims once per main-loop tick."""
    _prune_camera_sightings(now_t)
    if not (USE_VICTIM_DETECTION and camera is not None):
        return 0
    dets = detect_victim(camera, STATE.pose)
    new_count = 0
    for d in dets:
        d["rescued"] = False
        d["detected_at_t"] = now_t
        d["rescued_at_t"] = None
        if d.get("world_target_def") is not None:
            d["world_target_def"] = str(d["world_target_def"])
        if _WORLD_TARGET_TRANS_FIELDS and not d.get("world_target_def"):
            _rescue_trace(
                "process_victim_detections",
                "ASSOC_WARN",
                reason="missing_stable_world_target_def",
                candidate_victim_id=d.get("id"),
                x=d.get("x"),
                y=d.get("y"),
                has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            )
        STATE.victims.append(d)
        new_count += 1
        log_event("victim", {"id": d["id"], "x": d["x"], "y": d["y"], "world_target_def": d.get("world_target_def")})
        if AUTO_RESCUE_ON_DETECTION:
            rescued_now = rescue_victim(d["id"], "camera_auto_rescue")
            if rescued_now and STATE.goal is not None and STATE.goal_kind == "rescue":
                clear_goal(reason="camera_auto_rescue")
    return new_count

# =========================================================
# Webots init
# =========================================================
try:
    robot = Supervisor()
    _SUPERVISOR_ENABLED = True
except Exception:
    robot = Robot()
    _SUPERVISOR_ENABLED = False
basic_ts = int(robot.getBasicTimeStep())
ts = TIME_STEP if TIME_STEP > 0 else basic_ts

RESCUE_TARGET_DEFS = [f"RESCUE_TARGET_{i}" for i in range(1, 1 + MISSION_REQUIRED_RESCUES)]
_WORLD_TARGET_NODES = {}
_WORLD_TARGET_TRANS_FIELDS = {}
_WORLD_TARGET_HIDDEN = set()
_LAST_WORLD_TARGET_HANDLE_RETRY_T = -1e9
_LAST_WORLD_TARGET_AUDIT_T = -1e9
_WORLD_TARGET_ITER_CALLS = 0

def _init_world_target_handles():
    """Initialize supervisor handles for rescue target solids if available."""
    _world_target_trace(
        "INIT_ENTER",
        has_supervisor=_SUPERVISOR_ENABLED,
        expected_targets=len(RESCUE_TARGET_DEFS),
        existing_handles=len(_WORLD_TARGET_TRANS_FIELDS),
    )
    if not _SUPERVISOR_ENABLED:
        _world_target_trace(
            "INIT_RETURN",
            reason="supervisor_disabled",
            has_supervisor=False,
            expected_targets=len(RESCUE_TARGET_DEFS),
        )
        return
    try:
        # Reset caches before re-scan so stale handles are not treated as valid after world resets.
        _WORLD_TARGET_NODES.clear()
        _WORLD_TARGET_TRANS_FIELDS.clear()
        missing_nodes = []
        missing_translation_fields = []
        for def_name in RESCUE_TARGET_DEFS:
            node = robot.getFromDef(def_name)
            if node is None:
                missing_nodes.append(def_name)
                continue
            fld = node.getField("translation")
            if fld is None:
                missing_translation_fields.append(def_name)
                continue
            _WORLD_TARGET_NODES[def_name] = node
            _WORLD_TARGET_TRANS_FIELDS[def_name] = fld
        handle_count = len(_WORLD_TARGET_TRANS_FIELDS)
        invalid_xy_defs = []
        for def_name in RESCUE_TARGET_DEFS:
            if def_name not in _WORLD_TARGET_TRANS_FIELDS:
                continue
            pos = _world_target_xy(def_name)
            if pos is None:
                invalid_xy_defs.append(def_name)
        all_ok = (
            handle_count == len(RESCUE_TARGET_DEFS)
            and not missing_nodes
            and not missing_translation_fields
            and not invalid_xy_defs
        )
        if _WORLD_TARGET_TRANS_FIELDS:
            print(f"✅ Supervisor target handles: {len(_WORLD_TARGET_TRANS_FIELDS)}")
            log_event("device_ok", {"device": "supervisor_targets", "count": len(_WORLD_TARGET_TRANS_FIELDS)})
        _world_target_trace(
            "INIT_SUCCESS" if all_ok else "INIT_WARN",
            handle_count=handle_count,
            expected_targets=len(RESCUE_TARGET_DEFS),
            missing_nodes=",".join(missing_nodes) if missing_nodes else "none",
            missing_translation_fields=",".join(missing_translation_fields) if missing_translation_fields else "none",
            invalid_xy_defs=",".join(invalid_xy_defs) if invalid_xy_defs else "none",
        )
        if (not all_ok) and DEBUG_WORLD_TARGETS:
            log_event(
                "world_target_handle_audit",
                {
                    "handle_count": handle_count,
                    "expected_targets": len(RESCUE_TARGET_DEFS),
                    "missing_nodes": missing_nodes,
                    "missing_translation_fields": missing_translation_fields,
                    "invalid_xy_defs": invalid_xy_defs,
                },
            )
    except Exception as e:
        print(f"⚠️ Supervisor target handle init failed: {e}")
        log_event("device_warn", {"device": "supervisor_targets", "error": str(e)})
        _world_target_trace("INIT_RETURN", reason="exception", error=str(e))

def _world_target_xy(def_name):
    fld = _WORLD_TARGET_TRANS_FIELDS.get(def_name)
    if fld is None:
        _world_target_trace(
            "XY_RETURN",
            reason="no_field",
            candidate_world_DEF=def_name,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        )
        return None
    try:
        v = fld.getSFVec3f()
    except Exception as e:
        _world_target_trace(
            "XY_RETURN",
            reason="field_read_exception",
            candidate_world_DEF=def_name,
            error=str(e),
        )
        return None
    try:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
    except Exception:
        _world_target_trace(
            "XY_RETURN",
            reason="bad_field_value",
            candidate_world_DEF=def_name,
            raw_value=str(v),
        )
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        _world_target_trace(
            "XY_RETURN",
            reason="non_finite_xyz",
            candidate_world_DEF=def_name,
            x=x,
            y=y,
            z=z,
        )
        return None
    node = _WORLD_TARGET_NODES.get(def_name)
    node_type = "unknown"
    node_name = "none"
    if node is not None:
        try:
            if hasattr(node, "getTypeName"):
                node_type = str(node.getTypeName())
            elif hasattr(node, "getType"):
                node_type = str(node.getType())
        except Exception:
            node_type = "unknown"
        try:
            name_field = node.getField("name")
            if name_field is not None:
                node_name = str(name_field.getSFString())
        except Exception:
            node_name = "none"

    robot_xy = None
    dx = dy = dist_m = None
    if _pose_xy_valid(STATE.pose):
        try:
            rx = float(STATE.pose["x"])
            ry = float(STATE.pose["y"])
            robot_xy = (rx, ry)
            dx = x - rx
            dy = y - ry
            dist_m = math.hypot(dx, dy)
        except Exception:
            robot_xy = None
            dx = dy = dist_m = None
    _world_target_trace(
        "XY_READ",
        candidate_world_DEF=def_name,
        expected_target_def=(str(def_name) in RESCUE_TARGET_DEFS),
        node_handle_exists=bool(node is not None),
        node_type=node_type,
        node_name=node_name,
        translation=f"({x:.3f},{y:.3f},{z:.3f})",
        robot_xy=(f"({robot_xy[0]:.3f},{robot_xy[1]:.3f})" if robot_xy is not None else "none"),
        dx=dx,
        dy=dy,
        dist_m=dist_m,
        frame="world_xy_m",
    )
    return x, y, z

def _audit_world_target_registry_consistency(source):
    """Check consistency between victim.rescued and world-target hidden/marked state."""
    if not (DEBUG_RESCUE and DEBUG_WORLD_TARGETS):
        return True
    mismatches = []
    for v in STATE.victims:
        target_def = v.get("world_target_def")
        if not target_def:
            continue
        target_def = str(target_def)
        victim_id = int(v.get("id", -1))
        victim_rescued = bool(v.get("rescued", False)) or (victim_id in STATE.rescued_victim_ids)
        marked = target_def in STATE.rescued_world_target_defs
        hidden = target_def in _WORLD_TARGET_HIDDEN
        if victim_rescued and not (marked or hidden):
            mismatches.append(f"{target_def}:victim_rescued_but_unmarked")
        if (not victim_rescued) and (marked or hidden):
            mismatches.append(f"{target_def}:victim_active_but_marked_hidden")
    ok = not mismatches
    _world_target_trace(
        "REGISTRY_AUDIT_OK" if ok else "REGISTRY_AUDIT_FAIL",
        source=source,
        victim_count=len(STATE.victims),
        rescued_count=len(STATE.rescued_victim_ids),
        hidden_count=len(_WORLD_TARGET_HIDDEN),
        marked_defs=len(STATE.rescued_world_target_defs),
        mismatches=";".join(mismatches) if mismatches else "none",
    )
    if not ok:
        log_event("world_target_registry_inconsistent", {"source": str(source), "mismatches": mismatches})
    return ok

def _audit_world_target_runtime(reason, force=False):
    """Low-rate runtime audit proving handles/XY/iterator consistency for rescue targets."""
    global _LAST_WORLD_TARGET_AUDIT_T
    if not (DEBUG_RESCUE and DEBUG_WORLD_TARGETS):
        return
    now_t = robot.getTime()
    if (not force) and (now_t - _LAST_WORLD_TARGET_AUDIT_T) < 0.75:
        return
    _LAST_WORLD_TARGET_AUDIT_T = now_t

    if _SUPERVISOR_ENABLED and 0 < len(_WORLD_TARGET_TRANS_FIELDS) < len(RESCUE_TARGET_DEFS):
        _world_target_trace(
            "AUDIT_WARN",
            reason="partial_handles_detected_reinit",
            handle_count=len(_WORLD_TARGET_TRANS_FIELDS),
            expected_targets=len(RESCUE_TARGET_DEFS),
        )
        _init_world_target_handles()

    handle_defs = [d for d in RESCUE_TARGET_DEFS if _WORLD_TARGET_NODES.get(d) is not None and _WORLD_TARGET_TRANS_FIELDS.get(d) is not None]
    missing_defs = [d for d in RESCUE_TARGET_DEFS if d not in handle_defs]

    invalid_xy_defs = []
    finite_xy_defs = []
    for d in RESCUE_TARGET_DEFS:
        pos = _world_target_xy(d)
        if pos is None:
            invalid_xy_defs.append(d)
            continue
        finite_xy_defs.append(d)

    expected_unrescued = [
        d for d in RESCUE_TARGET_DEFS
        if d not in STATE.rescued_world_target_defs and d not in _WORLD_TARGET_HIDDEN
    ]
    actual_unrescued = []
    if _WORLD_TARGET_TRANS_FIELDS:
        for d in expected_unrescued:
            pos = _world_target_xy(d)
            if pos is not None:
                actual_unrescued.append(d)
    else:
        actual_unrescued = []

    iterator_consistent = (set(actual_unrescued) == set(expected_unrescued)) if _WORLD_TARGET_TRANS_FIELDS else True
    _world_target_trace(
        "AUDIT_OK" if (not missing_defs and not invalid_xy_defs and iterator_consistent) else "AUDIT_FAIL",
        reason=reason,
        has_supervisor=_SUPERVISOR_ENABLED,
        handle_count=len(handle_defs),
        expected_targets=len(RESCUE_TARGET_DEFS),
        missing_defs=",".join(missing_defs) if missing_defs else "none",
        finite_xy_defs=",".join(finite_xy_defs) if finite_xy_defs else "none",
        invalid_xy_defs=",".join(invalid_xy_defs) if invalid_xy_defs else "none",
        expected_unrescued=",".join(expected_unrescued) if expected_unrescued else "none",
        actual_unrescued=",".join(actual_unrescued) if actual_unrescued else "none",
        iterator_consistent=iterator_consistent,
    )
    _audit_world_target_registry_consistency(f"audit:{reason}")

def match_world_rescue_target(wx, wy, max_dist_m=WORLD_TARGET_MATCH_RADIUS_M, query_source="generic"):
    """Deterministically match world XY to the nearest non-rescued rescue target."""
    if not (math.isfinite(float(wx)) and math.isfinite(float(wy))):
        _world_target_trace(
            "MATCH_RETURN",
            reason="non_finite_query",
            qx=wx,
            qy=wy,
            threshold_m=max_dist_m,
            query_source=query_source,
        )
        return None
    scanned_defs = 0
    skipped_invalid_pos = []
    candidates = []
    for def_name in RESCUE_TARGET_DEFS:
        if def_name in STATE.rescued_world_target_defs or def_name in _WORLD_TARGET_HIDDEN:
            continue
        scanned_defs += 1
        pos = _world_target_xy(def_name)
        if pos is None:
            skipped_invalid_pos.append(def_name)
            continue
        d = _dist_xy(wx, wy, pos[0], pos[1])
        candidates.append((float(d), str(def_name), pos))

    if not candidates:
        _world_target_trace(
            "MATCH_RETURN",
            reason="no_match",
            qx=wx,
            qy=wy,
            threshold_m=max_dist_m,
            dist_m=None,
            scanned_defs=scanned_defs,
            candidate_count=0,
            skipped_invalid_pos=",".join(skipped_invalid_pos) if skipped_invalid_pos else "none",
            match_result="none",
            query_source=query_source,
        )
        return None

    # Deterministic ordering: smallest distance wins; DEF name breaks exact-distance ties.
    candidates.sort(key=lambda item: (item[0], item[1]))
    nearest_d, nearest_def, nearest_pos = candidates[0]
    within = [c for c in candidates if c[0] <= float(max_dist_m)]
    if not within:
        _world_target_trace(
            "MATCH_RETURN",
            reason="too_far",
            qx=wx,
            qy=wy,
            threshold_m=max_dist_m,
            dist_m=nearest_d,
            nearest_world_DEF=nearest_def,
            scanned_defs=scanned_defs,
            candidate_count=len(candidates),
            within_count=0,
            skipped_invalid_pos=",".join(skipped_invalid_pos) if skipped_invalid_pos else "none",
            match_result="none",
            query_source=query_source,
        )
        return None

    best_d, best_def, best_pos = within[0]
    tie_eps = 1e-6
    ties = [c for c in within if abs(c[0] - best_d) <= tie_eps]
    if len(ties) > 1:
        _world_target_trace(
            "MATCH_TIEBREAK",
            qx=wx,
            qy=wy,
            threshold_m=max_dist_m,
            candidate_count=len(candidates),
            within_count=len(within),
            tie_count=len(ties),
            tie_candidates=",".join(f"{d_name}:{d_val:.3f}" for d_val, d_name, _ in ties),
            chosen_def=best_def,
            chosen_dist=best_d,
            tie_break="min_distance_then_def",
            query_source=query_source,
        )

    _world_target_trace(
        "MATCH_SUCCESS",
        qx=wx,
        qy=wy,
        candidate_world_DEF=best_def,
        dist_m=best_d,
        threshold_m=max_dist_m,
        scanned_defs=scanned_defs,
        candidate_count=len(candidates),
        within_count=len(within),
        skipped_invalid_pos=",".join(skipped_invalid_pos) if skipped_invalid_pos else "none",
        match_result="nearest",
        query_source=query_source,
    )
    return {"def_name": best_def, "x": best_pos[0], "y": best_pos[1], "z": best_pos[2], "distance_m": best_d}

def match_world_rescue_target_from_camera(cx_norm, pose, camera_dev=None):
    """Return a camera-angle hint target; final association is resolved by deterministic XY matching."""
    if not _pose_xy_valid(pose):
        _world_target_trace("CAM_HINT_RETURN", reason="invalid_pose", cx_norm=cx_norm)
        return None
    if not _WORLD_TARGET_TRANS_FIELDS:
        _world_target_trace("CAM_HINT_RETURN", reason="no_handles", cx_norm=cx_norm)
        return None
    try:
        fov = float(camera_dev.getFov()) if camera_dev is not None else 1.05
    except Exception:
        fov = 1.05
    expected_yaw = _wrap_angle(float(pose["yaw"]) + (float(cx_norm) - 0.5) * fov)
    best = None
    best_score = float("inf")
    cand_scores = []
    for def_name in RESCUE_TARGET_DEFS:
        if def_name in STATE.rescued_world_target_defs or def_name in _WORLD_TARGET_HIDDEN:
            continue
        pos = _world_target_xy(def_name)
        if pos is None:
            continue
        dx = float(pos[0]) - float(pose["x"])
        dy = float(pos[1]) - float(pose["y"])
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        bearing = math.atan2(dy, dx)
        facing_err = abs(_wrap_angle(bearing - float(pose["yaw"])))
        # Keep camera gating tolerant so detections are not dropped by slight yaw/FOV mismatch.
        if facing_err > (0.5 * fov + 0.45):
            continue
        angle_err = abs(_wrap_angle(bearing - expected_yaw))
        score = angle_err + 0.02 * dist
        if angle_err <= 0.65:
            cand_scores.append((float(score), str(def_name), float(dist), float(angle_err), pos))
        if angle_err <= 0.65 and score < best_score:
            best_score = score
            best = {"def_name": def_name, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                    "distance_m": dist, "angle_err_rad": angle_err}
    if not cand_scores:
        _world_target_trace("CAM_HINT_RETURN", reason="no_candidate", cx_norm=cx_norm, expected_yaw=expected_yaw)
        return None
    cand_scores.sort(key=lambda c: (c[0], c[1]))
    if len(cand_scores) > 1:
        top = cand_scores[: min(3, len(cand_scores))]
        _world_target_trace(
            "CAM_HINT_TIEBREAK",
            cx_norm=cx_norm,
            expected_yaw=expected_yaw,
            candidate_count=len(cand_scores),
            top_candidates=",".join(f"{d}:{s:.3f}" for s, d, _, _, _ in top),
            chosen_def=cand_scores[0][1],
            tie_break="min_score_then_def",
        )
    if best is not None:
        _world_target_trace(
            "CAM_HINT_SUCCESS",
            cx_norm=cx_norm,
            candidate_world_DEF=best["def_name"],
            distance_m=best["distance_m"],
            angle_err_rad=best["angle_err_rad"],
            score=best_score,
        )
    return best

def hide_world_rescue_target(def_name):
    """Move a rescued target far away so it disappears and cannot be hit again."""
    hidden_before = len(_WORLD_TARGET_HIDDEN)
    def_name = str(def_name)
    rescued_count_now = len(STATE.rescued_victim_ids)
    action_name = "move_far_below_ground"
    cached_node = _WORLD_TARGET_NODES.get(def_name)
    cached_fld = _WORLD_TARGET_TRANS_FIELDS.get(def_name)

    def _hide_error(reason, **extra):
        msg_parts = [
            "WORLD_TARGET_HIDE_ERROR",
            f"def={def_name}",
            f"reason={reason}",
            f"supervisor={1 if _SUPERVISOR_ENABLED else 0}",
            f"cache_node={1 if cached_node is not None else 0}",
            f"cache_field={1 if cached_fld is not None else 0}",
            f"handles_total={len(_WORLD_TARGET_TRANS_FIELDS)}",
        ]
        for k, v in extra.items():
            msg_parts.append(f"{k}={_rescue_trace_value(v)}")
        print(" ".join(msg_parts))
        log_event(
            "world_target_hide_error",
            {
                "def_name": def_name,
                "reason": str(reason),
                "has_supervisor": bool(_SUPERVISOR_ENABLED),
                "cache_node_exists": bool(cached_node is not None),
                "cache_field_exists": bool(cached_fld is not None),
                "handles_total": int(len(_WORLD_TARGET_TRANS_FIELDS)),
                **{str(k): (v if isinstance(v, (int, float, bool, str)) or v is None else str(v)) for k, v in extra.items()},
            },
        )

    _rescue_trace(
        "hide_world_rescue_target",
        "ENTER",
        candidate_world_DEF=def_name,
        has_supervisor=_SUPERVISOR_ENABLED,
        has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        node_handle_exists=bool(cached_node is not None),
        translation_field_exists=bool(cached_fld is not None),
        rescued_count_before=rescued_count_now,
    )

    if not _SUPERVISOR_ENABLED:
        _hide_error("supervisor_disabled")
        _rescue_gate_table(
            "hide_world_rescue_target",
            "RETURN",
            [
                ("supervisor_available", False),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", def_name in RESCUE_TARGET_DEFS),
                ("matching_success", True),
                ("already_rescued", True),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
                ("node_handle_exists", False),
                ("translation_field_exists", False),
                ("hide_command_applied", False),
            ],
            reason="supervisor_disabled",
            candidate_world_DEF=def_name,
        )
        _rescue_trace(
            "hide_world_rescue_target",
            "RETURN",
            reason="supervisor_disabled",
            candidate_world_DEF=def_name,
            has_supervisor=False,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            rescued_count_before=rescued_count_now,
            rescued_count_after=rescued_count_now,
            hidden_count_before=hidden_before,
            hidden_count_after=len(_WORLD_TARGET_HIDDEN),
        )
        return False

    node = cached_node
    if node is None:
        try:
            node = robot.getFromDef(def_name)
            if node is not None:
                _WORLD_TARGET_NODES[def_name] = node
                _rescue_trace(
                    "hide_world_rescue_target",
                    "NODE_HANDLE_RECOVERED",
                    candidate_world_DEF=def_name,
                    action="robot.getFromDef",
                )
        except Exception as e:
            _hide_error("node_lookup_exception", error=str(e))
            node = None

    fld = cached_fld
    if fld is None and node is not None:
        try:
            fld = node.getField("translation")
            if fld is not None:
                _WORLD_TARGET_TRANS_FIELDS[def_name] = fld
                _rescue_trace(
                    "hide_world_rescue_target",
                    "FIELD_HANDLE_RECOVERED",
                    candidate_world_DEF=def_name,
                    action="node.getField:translation",
                )
        except Exception as e:
            _hide_error("translation_field_lookup_exception", error=str(e), node_handle_exists=True)
            fld = None

    if node is None:
        _hide_error("node_handle_missing", requested_def_known=(def_name in RESCUE_TARGET_DEFS))
        _rescue_gate_table(
            "hide_world_rescue_target",
            "RETURN",
            [
                ("supervisor_available", True),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", def_name in RESCUE_TARGET_DEFS),
                ("matching_success", True),
                ("already_rescued", True),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
                ("node_handle_exists", False),
                ("translation_field_exists", False),
                ("hide_command_applied", False),
            ],
            reason="node_handle_missing",
            candidate_world_DEF=def_name,
        )
        _rescue_trace(
            "hide_world_rescue_target",
            "RETURN",
            reason="node_handle_missing",
            candidate_world_DEF=def_name,
            has_supervisor=True,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            rescued_count_before=rescued_count_now,
            rescued_count_after=rescued_count_now,
            hidden_count_before=hidden_before,
            hidden_count_after=len(_WORLD_TARGET_HIDDEN),
        )
        return False

    if fld is None:
        _hide_error("translation_field_missing", requested_def_known=(def_name in RESCUE_TARGET_DEFS), node_handle_exists=True)
        _rescue_gate_table(
            "hide_world_rescue_target",
            "RETURN",
            [
                ("supervisor_available", True),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", def_name in RESCUE_TARGET_DEFS),
                ("matching_success", True),
                ("already_rescued", True),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
                ("node_handle_exists", True),
                ("translation_field_exists", False),
                ("hide_command_applied", False),
            ],
            reason="translation_field_missing",
            candidate_world_DEF=def_name,
        )
        _rescue_trace(
            "hide_world_rescue_target",
            "RETURN",
            reason="translation_field_missing",
            candidate_world_DEF=def_name,
            has_supervisor=True,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            rescued_count_before=rescued_count_now,
            rescued_count_after=rescued_count_now,
            hidden_count_before=hidden_before,
            hidden_count_after=len(_WORLD_TARGET_HIDDEN),
        )
        return False

    try:
        idx = max(0, RESCUE_TARGET_DEFS.index(def_name))
    except ValueError:
        idx = len(_WORLD_TARGET_HIDDEN)

    target_xyz = [float(WORLD_TARGET_HIDE_BASE_X + idx), float(WORLD_TARGET_HIDE_BASE_Y), -10.0]
    pre_xyz = None
    pre_read_ok = False
    try:
        pre_v = fld.getSFVec3f()
        pre_xyz = [float(pre_v[0]), float(pre_v[1]), float(pre_v[2])]
        if all(math.isfinite(v) for v in pre_xyz):
            pre_read_ok = True
        else:
            _hide_error("pre_read_non_finite", pre_xyz=pre_xyz, node_handle_exists=True, translation_field_exists=True)
    except Exception as e:
        if DEBUG_RESCUE:
            print(f"WORLD_TARGET_HIDE_WARN def={def_name} reason=pre_read_failed err={e}")
        log_event("world_target_hide_warn", {"def_name": def_name, "reason": "pre_read_failed", "error": str(e)})

    _rescue_trace(
        "hide_world_rescue_target",
        "ACTION",
        candidate_world_DEF=def_name,
        action=action_name,
        target_xyz=target_xyz,
        pre_xyz=pre_xyz,
        pre_read_ok=pre_read_ok,
        node_handle_exists=True,
        translation_field_exists=True,
    )

    try:
        fld.setSFVec3f(target_xyz)
        physics_reset_ok = None
        if node is not None:
            try:
                node.resetPhysics()
                physics_reset_ok = True
            except Exception as e:
                physics_reset_ok = False
                if DEBUG_RESCUE:
                    print(f"WORLD_TARGET_HIDE_WARN def={def_name} reason=reset_physics_failed err={e}")
                log_event("world_target_hide_warn", {"def_name": def_name, "reason": "reset_physics_failed", "error": str(e)})

        post_xyz = None
        post_read_ok = False
        try:
            post_v = fld.getSFVec3f()
            post_xyz = [float(post_v[0]), float(post_v[1]), float(post_v[2])]
            post_read_ok = all(math.isfinite(v) for v in post_xyz)
        except Exception as e:
            _hide_error("post_read_failed", error=str(e), action=action_name, target_xyz=target_xyz)
            _rescue_gate_table(
                "hide_world_rescue_target",
                "RETURN",
                [
                    ("supervisor_available", True),
                    ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                    ("candidate_target_exists", def_name in RESCUE_TARGET_DEFS),
                    ("matching_success", True),
                    ("already_rescued", True),
                    ("cooldown_throttle", True),
                    ("distance_le_rescue_radius", None),
                    ("node_handle_exists", True),
                    ("translation_field_exists", True),
                    ("hide_command_applied", True),
                    ("post_read_ok", False),
                ],
                reason="post_read_failed",
                candidate_world_DEF=def_name,
                action=action_name,
            )
            _rescue_trace(
                "hide_world_rescue_target",
                "RETURN",
                reason="post_read_failed",
                candidate_world_DEF=def_name,
                action=action_name,
                has_supervisor=True,
                has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
                node_handle_exists=True,
                translation_field_exists=True,
                rescued_count_before=rescued_count_now,
                rescued_count_after=rescued_count_now,
                hidden_count_before=hidden_before,
                hidden_count_after=len(_WORLD_TARGET_HIDDEN),
            )
            return False

        if not post_read_ok:
            _hide_error("post_read_non_finite", action=action_name, post_xyz=post_xyz, target_xyz=target_xyz)
            _rescue_gate_table(
                "hide_world_rescue_target",
                "RETURN",
                [
                    ("supervisor_available", True),
                    ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                    ("candidate_target_exists", def_name in RESCUE_TARGET_DEFS),
                    ("matching_success", True),
                    ("already_rescued", True),
                    ("cooldown_throttle", True),
                    ("distance_le_rescue_radius", None),
                    ("node_handle_exists", True),
                    ("translation_field_exists", True),
                    ("hide_command_applied", True),
                    ("post_read_ok", False),
                ],
                reason="post_read_non_finite",
                candidate_world_DEF=def_name,
                action=action_name,
            )
            _rescue_trace(
                "hide_world_rescue_target",
                "RETURN",
                reason="post_read_non_finite",
                candidate_world_DEF=def_name,
                action=action_name,
                post_xyz=post_xyz,
                target_xyz=target_xyz,
                has_supervisor=True,
                has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
                rescued_count_before=rescued_count_now,
                rescued_count_after=rescued_count_now,
                hidden_count_before=hidden_before,
                hidden_count_after=len(_WORLD_TARGET_HIDDEN),
            )
            return False

        changed = True
        if pre_read_ok and pre_xyz is not None:
            changed = any(abs(float(post_xyz[i]) - float(pre_xyz[i])) > 1e-6 for i in range(3))
        post_matches_target = all(abs(float(post_xyz[i]) - float(target_xyz[i])) <= 1e-4 for i in range(3))

        if pre_read_ok and not changed:
            _hide_error("translation_unchanged_after_set", action=action_name, pre_xyz=pre_xyz, post_xyz=post_xyz, target_xyz=target_xyz)
            _rescue_gate_table(
                "hide_world_rescue_target",
                "RETURN",
                [
                    ("supervisor_available", True),
                    ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                    ("candidate_target_exists", def_name in RESCUE_TARGET_DEFS),
                    ("matching_success", True),
                    ("already_rescued", True),
                    ("cooldown_throttle", True),
                    ("distance_le_rescue_radius", None),
                    ("node_handle_exists", True),
                    ("translation_field_exists", True),
                    ("hide_command_applied", True),
                    ("post_read_ok", True),
                    ("translation_changed", False),
                    ("post_matches_target", post_matches_target),
                ],
                reason="translation_unchanged_after_set",
                candidate_world_DEF=def_name,
                action=action_name,
            )
            _rescue_trace(
                "hide_world_rescue_target",
                "RETURN",
                reason="translation_unchanged_after_set",
                candidate_world_DEF=def_name,
                action=action_name,
                pre_xyz=pre_xyz,
                post_xyz=post_xyz,
                target_xyz=target_xyz,
                post_matches_target=post_matches_target,
                has_supervisor=True,
                has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
                rescued_count_before=rescued_count_now,
                rescued_count_after=rescued_count_now,
                hidden_count_before=hidden_before,
                hidden_count_after=len(_WORLD_TARGET_HIDDEN),
            )
            return False

        if not post_matches_target:
            _hide_error("post_translation_mismatch", action=action_name, pre_xyz=pre_xyz, post_xyz=post_xyz, target_xyz=target_xyz)
            _rescue_gate_table(
                "hide_world_rescue_target",
                "RETURN",
                [
                    ("supervisor_available", True),
                    ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                    ("candidate_target_exists", def_name in RESCUE_TARGET_DEFS),
                    ("matching_success", True),
                    ("already_rescued", True),
                    ("cooldown_throttle", True),
                    ("distance_le_rescue_radius", None),
                    ("node_handle_exists", True),
                    ("translation_field_exists", True),
                    ("hide_command_applied", True),
                    ("post_read_ok", True),
                    ("translation_changed", changed if pre_read_ok else "NA"),
                    ("post_matches_target", False),
                ],
                reason="post_translation_mismatch",
                candidate_world_DEF=def_name,
                action=action_name,
            )
            _rescue_trace(
                "hide_world_rescue_target",
                "RETURN",
                reason="post_translation_mismatch",
                candidate_world_DEF=def_name,
                action=action_name,
                pre_xyz=pre_xyz,
                post_xyz=post_xyz,
                target_xyz=target_xyz,
                translation_changed=(changed if pre_read_ok else None),
                has_supervisor=True,
                has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
                rescued_count_before=rescued_count_now,
                rescued_count_after=rescued_count_now,
                hidden_count_before=hidden_before,
                hidden_count_after=len(_WORLD_TARGET_HIDDEN),
            )
            return False

        _WORLD_TARGET_HIDDEN.add(def_name)
        print(
            f"WORLD_TARGET_HIDDEN def={def_name} action={action_name} "
            f"pre={pre_xyz} post={post_xyz} changed={changed if pre_read_ok else 'unknown'}"
        )
        log_event(
            "world_target_hidden",
            {
                "def_name": str(def_name),
                "action": action_name,
                "target_xyz": target_xyz,
                "pre_xyz": pre_xyz,
                "post_xyz": post_xyz,
                "pre_read_ok": pre_read_ok,
                "post_read_ok": post_read_ok,
                "translation_changed": (changed if pre_read_ok else None),
                "post_matches_target": post_matches_target,
                "physics_reset_ok": physics_reset_ok,
            },
        )
        _rescue_gate_table(
            "hide_world_rescue_target",
            "SUCCESS",
            [
                ("supervisor_available", True),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", def_name in RESCUE_TARGET_DEFS),
                ("matching_success", True),
                ("already_rescued", True),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
                ("node_handle_exists", True),
                ("translation_field_exists", True),
                ("hide_command_applied", True),
                ("post_read_ok", True),
                ("translation_changed", changed if pre_read_ok else "NA"),
                ("post_matches_target", True),
            ],
            candidate_world_DEF=def_name,
            action=action_name,
            hidden_count_before=hidden_before,
            hidden_count_after=len(_WORLD_TARGET_HIDDEN),
        )
        _rescue_trace(
            "hide_world_rescue_target",
            "SUCCESS",
            candidate_world_DEF=def_name,
            action=action_name,
            target_xyz=target_xyz,
            pre_xyz=pre_xyz,
            post_xyz=post_xyz,
            pre_read_ok=pre_read_ok,
            post_read_ok=post_read_ok,
            translation_changed=(changed if pre_read_ok else None),
            post_matches_target=post_matches_target,
            has_supervisor=True,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            node_handle_exists=True,
            translation_field_exists=True,
            rescued_count_before=rescued_count_now,
            rescued_count_after=rescued_count_now,
            hidden_count_before=hidden_before,
            hidden_count_after=len(_WORLD_TARGET_HIDDEN),
        )
        return True
    except Exception as e:
        _hide_error("exception", error=str(e), action=action_name)
        print(f"WORLD_TARGET_HIDE_FAILED def={def_name} err={e}")
        log_event("world_target_hide_failed", {"def_name": def_name, "error": str(e), "action": action_name})
        _rescue_gate_table(
            "hide_world_rescue_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", def_name in RESCUE_TARGET_DEFS),
                ("matching_success", True),
                ("already_rescued", True),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
                ("node_handle_exists", bool(node is not None)),
                ("translation_field_exists", bool(fld is not None)),
                ("hide_command_applied", False),
            ],
            reason="exception",
            candidate_world_DEF=def_name,
            error=str(e),
            action=action_name,
            hidden_count_before=hidden_before,
            hidden_count_after=len(_WORLD_TARGET_HIDDEN),
        )
        _rescue_trace(
            "hide_world_rescue_target",
            "RETURN",
            reason="exception",
            candidate_world_DEF=def_name,
            action=action_name,
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            node_handle_exists=bool(node is not None),
            translation_field_exists=bool(fld is not None),
            rescued_count_before=rescued_count_now,
            rescued_count_after=rescued_count_now,
            hidden_count_before=hidden_before,
            hidden_count_after=len(_WORLD_TARGET_HIDDEN),
        )
        return False

def iter_unrescued_world_targets():
    """Yield `(def_name, (x, y, z))` for currently active rescue targets."""
    global _WORLD_TARGET_ITER_CALLS
    _WORLD_TARGET_ITER_CALLS += 1

    if _SUPERVISOR_ENABLED and 0 < len(_WORLD_TARGET_TRANS_FIELDS) < len(RESCUE_TARGET_DEFS):
        _world_target_trace(
            "ITER_WARN",
            reason="partial_handles_retry_init",
            call_index=_WORLD_TARGET_ITER_CALLS,
            handle_count=len(_WORLD_TARGET_TRANS_FIELDS),
            expected_targets=len(RESCUE_TARGET_DEFS),
        )
        _init_world_target_handles()

    if _WORLD_TARGET_TRANS_FIELDS:
        expected_unrescued_defs = [
            def_name for def_name in RESCUE_TARGET_DEFS
            if def_name not in STATE.rescued_world_target_defs and def_name not in _WORLD_TARGET_HIDDEN
        ]
        yielded = []
        missing_xy_defs = []
        for def_name in expected_unrescued_defs:
            pos = _world_target_xy(def_name)
            if pos is None:
                missing_xy_defs.append(def_name)
                continue
            yielded.append((def_name, pos))
        yielded_defs = [d for d, _ in yielded]
        iterator_consistent = (set(yielded_defs) == set(expected_unrescued_defs))
        _world_target_trace(
            "ITER_YIELD",
            call_index=_WORLD_TARGET_ITER_CALLS,
            has_handles=True,
            expected_unrescued=",".join(expected_unrescued_defs) if expected_unrescued_defs else "none",
            yielded_defs=",".join(yielded_defs) if yielded_defs else "none",
            missing_xy_defs=",".join(missing_xy_defs) if missing_xy_defs else "none",
            iterator_consistent=iterator_consistent,
        )
        if not iterator_consistent:
            log_event(
                "world_target_iter_inconsistent",
                {
                    "expected_unrescued_defs": expected_unrescued_defs,
                    "yielded_defs": yielded_defs,
                    "missing_xy_defs": missing_xy_defs,
                },
            )
        _audit_world_target_registry_consistency("iter_unrescued_world_targets")
        for item in yielded:
            yield item
        return
    # Fallback if supervisor handles are unavailable: use known victim detections.
    fallback_items = []
    for v in STATE.victims:
        if bool(v.get("rescued", False)):
            continue
        vid = int(v.get("id", -1))
        if vid in STATE.rescued_victim_ids:
            continue
        fallback_items.append((f"victim_{vid}", (float(v["x"]), float(v["y"]), 0.0)))
    _world_target_trace(
        "ITER_FALLBACK",
        call_index=_WORLD_TARGET_ITER_CALLS,
        has_handles=False,
        has_supervisor=_SUPERVISOR_ENABLED,
        fallback_count=len(fallback_items),
        expected_targets=len(RESCUE_TARGET_DEFS),
    )
    _audit_world_target_registry_consistency("iter_unrescued_world_targets_fallback")
    for item in fallback_items:
        yield item

def _active_rescue_target_def():
    """Resolve the currently active rescue target DEF, if available."""
    if STATE.active_rescue_id is None:
        return None
    def _is_active_candidate(def_name):
        return (
            def_name in RESCUE_TARGET_DEFS
            and def_name not in STATE.rescued_world_target_defs
            and def_name not in _WORLD_TARGET_HIDDEN
        )
    victim = find_victim_by_id(STATE.active_rescue_id)
    if victim is not None:
        target_def = victim.get("world_target_def")
        if target_def is not None:
            target_def = str(target_def)
            if _is_active_candidate(target_def):
                return target_def
    for def_name, victim_id in STATE.world_target_to_victim_id.items():
        try:
            if int(victim_id) == int(STATE.active_rescue_id) and _is_active_candidate(str(def_name)):
                return str(def_name)
        except Exception:
            continue
    return None

def _target_defs_for_victim_filtering():
    """Return only the active rescue target DEF for map/lidar victim filtering."""
    active = _active_rescue_target_def()
    return [str(active)] if active else []

def _collect_world_target_positions(target_defs=None):
    """Return `(def_name, (x, y, z))` positions for selected rescue targets."""
    if target_defs is None:
        return list(iter_unrescued_world_targets())
    out = []
    seen = set()
    for def_name in target_defs:
        if not def_name:
            continue
        def_name = str(def_name)
        if def_name in seen:
            continue
        seen.add(def_name)
        if def_name in STATE.rescued_world_target_defs or def_name in _WORLD_TARGET_HIDDEN:
            continue
        pos = _world_target_xy(def_name)
        if pos is None:
            continue
        out.append((def_name, pos))
    return out

def _carve_rescue_targets_from_map(map_arr, radius_m=RESCUE_TARGET_MAP_CLEAR_RADIUS_M, target_defs=None):
    """Mark rescue-target cells as free so planner can approach them."""
    if map_arr is None:
        return
    targets = _collect_world_target_positions(target_defs=target_defs)
    if not targets:
        return
    radius_cells = max(1, int(math.ceil(float(radius_m) / float(MAP_RESOLUTION))))
    for _def_name, pos in targets:
        try:
            gx, gy = world_to_grid(pos[0], pos[1])
        except Exception:
            continue
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                ix, iy = gx + dx, gy + dy
                if 0 <= ix < MAP_SIZE and 0 <= iy < MAP_SIZE:
                    map_arr[ix, iy] = 0

def _filter_lidar_targets_from_avoidance(ranges, target_defs=None):
    """Ignore rescue-target returns in the reactive avoider while preserving walls/obstacles."""
    if ranges is None:
        return None
    if not _pose_xy_valid(STATE.pose):
        return ranges
    active_targets = _collect_world_target_positions(target_defs=target_defs)
    if not active_targets:
        return ranges
    arr = np.asarray(ranges, dtype=np.float64).copy()
    if arr.size == 0:
        return ranges
    try:
        fov = float(lidar.getFov())
    except Exception:
        return ranges
    n = int(arr.size)
    half_fov = 0.5 * fov
    for _def_name, pos in active_targets:
        dx = float(pos[0]) - float(STATE.pose["x"])
        dy = float(pos[1]) - float(STATE.pose["y"])
        dist = math.hypot(dx, dy)
        if dist <= 1e-6 or dist > float(RESCUE_TARGET_LIDAR_IGNORE_MAX_DIST_M):
            continue
        bearing = _wrap_angle(math.atan2(dy, dx) - float(STATE.pose["yaw"]))
        if abs(bearing) > half_fov + 0.2:
            continue
        half_width = math.atan2(float(RESCUE_TARGET_LIDAR_IGNORE_RADIUS_M), max(dist, 1e-6))
        start_ang = max(-half_fov, bearing - half_width)
        end_ang = min(half_fov, bearing + half_width)
        i0 = int(math.floor(((start_ang + half_fov) / fov) * n))
        i1 = int(math.ceil(((end_ang + half_fov) / fov) * n))
        i0 = max(0, min(n - 1, i0))
        i1 = max(0, min(n - 1, i1))
        if i1 < i0:
            i0, i1 = i1, i0
        segment = arr[i0:i1 + 1]
        valid = np.isfinite(segment) & (segment > 0.0)
        if not np.any(valid):
            continue
        near_bound = max(0.05, dist - float(RESCUE_TARGET_LIDAR_IGNORE_NEAR_DELTA_M))
        far_bound = dist + float(RESCUE_TARGET_LIDAR_IGNORE_FAR_DELTA_M)
        likely_target_echo = valid & (segment >= near_bound) & (segment <= far_bound)
        if not np.any(likely_target_echo):
            continue
        segment[likely_target_echo] = max_range
        arr[i0:i1 + 1] = segment
    return arr.tolist()

def _lidar_corridor_min_range(ranges, bearing_rad, half_width_deg=STARTUP_DIRECT_RESCUE_CORRIDOR_HALF_DEG):
    """Return minimum range in a narrow angular corridor around a bearing (robot frame)."""
    if ranges is None:
        return float("inf")
    arr = np.asarray(ranges, dtype=np.float64)
    if arr.size == 0:
        return float("inf")
    try:
        fov = float(lidar.getFov())
    except Exception:
        return float("inf")
    if fov <= 1e-9:
        return float("inf")
    half_fov = 0.5 * fov
    bearing = _wrap_angle(float(bearing_rad))
    if bearing < -half_fov or bearing > half_fov:
        return float("inf")

    half_width = abs(math.radians(float(half_width_deg)))
    start_ang = max(-half_fov, bearing - half_width)
    end_ang = min(half_fov, bearing + half_width)
    n = int(arr.size)
    i0 = int(math.floor(((start_ang + half_fov) / fov) * n))
    i1 = int(math.ceil(((end_ang + half_fov) / fov) * n))
    i0 = max(0, min(n - 1, i0))
    i1 = max(0, min(n - 1, i1))
    if i1 < i0:
        i0, i1 = i1, i0

    segment = arr[i0:i1 + 1]
    valid = segment[np.isfinite(segment) & (segment > 0.0)]
    if valid.size == 0:
        return float("inf")
    return float(np.min(valid))

# ---- Logs
RUN_DIR = mkdirp(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs", f"run_{now_stamp()}")))
SCR_DIR = mkdirp(os.path.join(RUN_DIR, "screenshots"))
pose_csv = open(os.path.join(RUN_DIR, "pose.csv"), "w", encoding="utf-8")
pose_csv.write("t,x,y,yaw\n")
events = open(os.path.join(RUN_DIR, "events.jsonl"), "w", encoding="utf-8")
VICTIM_EVENT_WRITER = VictimEventWriter(os.path.join(RUN_DIR, "victim_events.jsonl"))
VICTIM_METRICS = VictimMetricsTracker(
    cluster_radius_m=VICTIM_UNIQUE_CLUSTER_RADIUS_M,
    duplicate_radius_m=VICTIM_DUPLICATE_RADIUS_M,
)
VICTIM_METRICS_SUMMARY_PATH = os.path.join(RUN_DIR, "victim_metrics_summary.json")

print("✅ rescue_main started | logs:", RUN_DIR)
log_event("run_start", {"run_dir": RUN_DIR})
_init_world_target_handles()
_audit_world_target_runtime("startup_init", force=True)

# ---- Motors (Pioneer3at = 4 wheels)
motors_left  = [robot.getDevice("front left wheel"), robot.getDevice("back left wheel")]
motors_right = [robot.getDevice("front right wheel"), robot.getDevice("back right wheel")]

for m in motors_left + motors_right:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

try:
    wheel_speed_cap = min(float(m.getMaxVelocity()) for m in (motors_left + motors_right))
    if wheel_speed_cap > 0:
        MAX_SPEED = max(MAX_SPEED, wheel_speed_cap)
        print(f"✅ Wheel speed cap: {MAX_SPEED:.2f} rad/s")
        log_event("device_ok", {"device": "wheel_speed_cap", "max_radps": float(MAX_SPEED)})
except Exception:
    pass

def set_wheels(lv, rv):
    for m in motors_left:
        m.setVelocity(lv)
    for m in motors_right:
        m.setVelocity(rv)

def _try_enable_motor_pos_sensor(motor, timestep_ms):
    """Best-effort access to a wheel encoder exposed via the motor."""
    try:
        sensor = motor.getPositionSensor()
        if sensor is None:
            return None
        sensor.enable(int(timestep_ms))
        return sensor
    except Exception:
        return None

POSE_EST = PoseFusionEstimator(POSE_WHEEL_RADIUS_M, POSE_AXLE_LENGTH_M)
left_encoders = [_try_enable_motor_pos_sensor(m, ts) for m in motors_left]
right_encoders = [_try_enable_motor_pos_sensor(m, ts) for m in motors_right]
if POSE_EST.attach_wheel_sensors(left_encoders, right_encoders):
    print("✅ Wheel encoders enabled for odometry fusion.")
    log_event("device_ok", {"device": "wheel_encoders", "left": len([s for s in left_encoders if s]), "right": len([s for s in right_encoders if s])})
else:
    print("⚠️ Wheel encoders unavailable; using commanded-velocity odometry fallback.")
    log_event("device_warn", {"device": "wheel_encoders", "mode": "cmd_velocity_fallback"})

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

_last_gps_outlier_log_t = -1e9
def get_pose():
    """Return fused pose estimate (wheel odom + GPS/Compass correction)."""
    global _last_gps_outlier_log_t
    gps_xy = None
    compass_yaw = None
    if gps is not None:
        p = gps.getValues()   # [x, y, z]
        # Webots uses a Z-up world frame, so the ground plane is (x, y) in meters.
        gps_xy = (float(p[0]), float(p[1]))  # ground plane x/y
    if compass is not None:
        n = compass.getValues()
        # Match the same Z-up ground plane convention used by GPS/targets: yaw in x/y plane.
        compass_yaw = math.atan2(n[0], n[1])
    x, y, yaw, diag = POSE_EST.update(robot.getTime(), gps_xy=gps_xy, compass_yaw=compass_yaw)
    if diag.get("gps_outlier") and gps_xy is not None and (robot.getTime() - _last_gps_outlier_log_t) > 1.0:
        _last_gps_outlier_log_t = robot.getTime()
        log_event("gps_outlier_rejected", {"x": gps_xy[0], "y": gps_xy[1]})
    return x, y, yaw

class _NavGridAdapter:
    """Minimal grid object exposing metadata expected by NavigationStack."""
    def __init__(self, data_xy_indexed):
        # rescue_main map is indexed [ix, iy]; nav stack expects [iy, ix].
        data_yx = np.asarray(data_xy_indexed).T
        self.data = data_yx
        self.height, self.width = data_yx.shape
        self.resolution_m = float(MAP_RESOLUTION)
        half_span_m = 0.5 * float(MAP_SIZE) * float(MAP_RESOLUTION)
        self.origin_x_m = -half_span_m
        self.origin_y_m = -half_span_m

def _as_nav_grid(map_arr):
    if map_arr is None:
        return None
    return _NavGridAdapter(map_arr)

# ---- Navigation Stack Setup
def nav_get_grid():
    raw_for_nav = _as_nav_grid(STATE.map_raw if STATE.map_raw is not None else STATE.map)
    inflated_for_nav = _as_nav_grid(STATE.map)
    return {
        # NavigationStack inflates internally, so provide the raw LiDAR-updated grid.
        "raw_grid": raw_for_nav,
        "inflated_grid": inflated_for_nav,
        "map_update_counter": STATE.map_version,
        "map_timestamp": STATE.last_map_update_t,
    }

def nav_get_pose():
    return Pose2D(STATE.pose["x"], STATE.pose["y"], STATE.pose["yaw"])

def nav_get_goal():
    return STATE.goal

def nav_get_lidar():
    target_defs = _target_defs_for_victim_filtering()
    return _filter_lidar_targets_from_avoidance(
        lidar.getRangeImage(),
        target_defs=(target_defs if target_defs else None),
    )

def nav_send_cmd(twist):
    # Pioneer3at differential drive
    # Convert unicycle command (m/s, rad/s) -> wheel motor angular speed (rad/s).
    # Pioneer track is ~0.4m, wheel radius ~0.1m.
    L = 0.4
    R = 0.1
    lv = (twist.v - (twist.omega * L / 2.0)) / R
    rv = (twist.v + (twist.omega * L / 2.0)) / R
    lv = max(-MAX_SPEED, min(MAX_SPEED, lv))
    rv = max(-MAX_SPEED, min(MAX_SPEED, rv))
    POSE_EST.set_command(twist.v, twist.omega)
    set_wheels(lv, rv)

NAV = NavigationStack(
    get_grid=nav_get_grid,
    get_pose=nav_get_pose,
    get_goal=nav_get_goal,
    get_lidar=nav_get_lidar,
    send_cmd=nav_send_cmd
)
NAV.config.max_v = MISSION_NAV_MAX_V_MPS
NAV.config.max_omega = MISSION_NAV_MAX_OMEGA_RADPS
NAV.config.lookahead_m = 1.25
# Obstacle-first tuning for pillar maps: react around ~1m, but keep side-gap
# acceptance permissive enough so the robot does not freeze.
NAV.config.avoid_gain = 1.20
NAV.config.safety_distance_m = 1.05
NAV.config.side_clearance_m = 0.70
NAV.config.stop_distance_m = 0.30
NAV.config.goal_tolerance_m = 0.22
NAV.config.inflation_radius_m = 0.35
NAV.config.stuck_window_s = 3.0
NAV.config.min_progress_m = 0.015
NAV.config.recovery_turn_time_s = 0.60
NAV.config.recovery_backup_time_s = 1.10
try:
    NAV.config.controller.heading_kp = 1.85
    NAV.config.controller.slow_down_radius_m = 0.55
    NAV.config.planner.allow_unknown = False
    NAV.config.planner.replanning_period_s = 1.80
    NAV.config.planner.map_change_replan_min_period_s = 1.30
    NAV.config.planner.min_pose_change_to_replan_m = 0.35
    NAV.config.planner.path_deviation_m = 1.10
    NAV.config.planner.blocked_lookahead_waypoints = 6
    NAV.config.planner.start_goal_search_radius_m = 1.50
    NAV.config.planner.smoothing_passes = 7
    NAV.config.planner.smoothing_alpha = 0.50
    NAV.config.planner.smoothing_max_shift_m = 0.18
    NAV.config.planner.waypoint_spacing_m = 0.45
except Exception:
    pass
for _name, _value in (
    ("recovery_backup_speed_mps", 0.20),
    ("recovery_backup_turn_omega_radps", 0.92),
    ("recovery_forward_speed_mps", 0.12),
    ("recovery_forward_time_s", 0.30),
    ("recovery_turn_omega_radps", 0.95),
    ("stuck_cmd_omega_threshold_radps", 0.32),
    ("stuck_min_yaw_progress_rad", 0.10),
    ("oscillation_yaw_change_rad", 2.2),
):
    try:
        setattr(NAV.config, _name, _value)
    except Exception:
        pass
print(
    "NAV_TUNING "
    f"max_v={NAV.config.max_v:.2f} max_omega={NAV.config.max_omega:.2f} "
    f"lookahead={NAV.config.lookahead_m:.2f} backup_time={NAV.config.recovery_backup_time_s:.2f}"
)
log_event(
    "nav_tuning",
    {
        "max_v": float(NAV.config.max_v),
        "max_omega": float(NAV.config.max_omega),
        "lookahead_m": float(NAV.config.lookahead_m),
        "avoid_gain": float(NAV.config.avoid_gain),
        "safety_distance_m": float(NAV.config.safety_distance_m),
        "side_clearance_m": float(NAV.config.side_clearance_m),
        "stop_distance_m": float(NAV.config.stop_distance_m),
        "goal_tolerance_m": float(NAV.config.goal_tolerance_m),
        "stuck_window_s": float(NAV.config.stuck_window_s),
        "min_progress_m": float(NAV.config.min_progress_m),
        "recovery_turn_time_s": float(NAV.config.recovery_turn_time_s),
        "recovery_backup_time_s": float(NAV.config.recovery_backup_time_s),
        "map_change_replan_min_period_s": float(getattr(NAV.config.planner, "map_change_replan_min_period_s", 1.30)),
        "recovery_backup_speed_mps": 0.20,
        "recovery_backup_turn_omega_radps": 0.92,
        "recovery_forward_speed_mps": 0.12,
        "recovery_forward_time_s": 0.30,
        "recovery_turn_omega_radps": 0.95,
        "stuck_cmd_omega_threshold_radps": 0.32,
        "stuck_min_yaw_progress_rad": 0.10,
        "oscillation_yaw_change_rad": 2.2,
    },
)

NAV_BASELINE_TUNING = {
    "avoid_gain": float(NAV.config.avoid_gain),
    "safety_distance_m": float(NAV.config.safety_distance_m),
    "side_clearance_m": float(NAV.config.side_clearance_m),
    "stop_distance_m": float(NAV.config.stop_distance_m),
    "lookahead_m": float(NAV.config.lookahead_m),
    "max_omega": float(NAV.config.max_omega),
    "inflation_radius_m": float(NAV.config.inflation_radius_m),
}
_NAV_BASE_ALLOW_UNKNOWN = bool(getattr(NAV.config.planner, "allow_unknown", False))
_MAP_INFLATION_RADIUS_CELLS_RUNTIME = int(MAP_INFLATION_RADIUS_CELLS)
_FINAL_APPROACH_ACTIVE = False
_FINAL_APPROACH_TARGET_DEF = None
_STARTUP_DIRECT_ALLOW_UNKNOWN_ACTIVE = False

def maybe_update_startup_direct_planner_policy(now_t):
    """Temporarily allow planning through unknown cells for startup direct-rescue goals."""
    global _STARTUP_DIRECT_ALLOW_UNKNOWN_ACTIVE
    should_force_allow_unknown = bool(
        ENABLE_STARTUP_DIRECT_RESCUE
        and float(now_t) <= float(STARTUP_DIRECT_RESCUE_WINDOW_S)
        and STATE.goal_kind == "rescue"
        and STATE.goal is not None
        and STATE.active_rescue_id is not None
    )
    desired_allow_unknown = bool(_NAV_BASE_ALLOW_UNKNOWN or should_force_allow_unknown)
    try:
        current_allow_unknown = bool(getattr(NAV.config.planner, "allow_unknown", desired_allow_unknown))
    except Exception:
        return
    if current_allow_unknown != desired_allow_unknown:
        NAV.config.planner.allow_unknown = bool(desired_allow_unknown)
        state = "ON" if should_force_allow_unknown else "OFF"
        print(
            "STARTUP_DIRECT_POLICY "
            f"state={state} allow_unknown={NAV.config.planner.allow_unknown} "
            f"goal_kind={STATE.goal_kind} active_rescue_id={STATE.active_rescue_id}"
        )
        log_event(
            "startup_direct_planner_policy",
            {
                "state": state,
                "allow_unknown": bool(NAV.config.planner.allow_unknown),
                "goal_kind": str(STATE.goal_kind),
                "active_rescue_id": (
                    None if STATE.active_rescue_id is None else int(STATE.active_rescue_id)
                ),
            },
        )
    _STARTUP_DIRECT_ALLOW_UNKNOWN_ACTIVE = bool(should_force_allow_unknown)

def _active_rescue_distance_snapshot():
    """Return active target distance snapshot in world meters, or None."""
    if not _pose_xy_valid(STATE.pose):
        return None
    active_def = _active_rescue_target_def()
    if not active_def:
        return None
    target_pos = _world_target_xy(active_def)
    if target_pos is None:
        return None
    rx = float(STATE.pose["x"])
    ry = float(STATE.pose["y"])
    tx = float(target_pos[0])
    ty = float(target_pos[1])
    dx = tx - rx
    dy = ty - ry
    return {
        "def_name": str(active_def),
        "robot_x": rx,
        "robot_y": ry,
        "target_x": tx,
        "target_y": ty,
        "dx": dx,
        "dy": dy,
        "dist_m": math.hypot(dx, dy),
    }

def maybe_update_final_approach_mode(now_t):
    """Bias nav for close-range rescue approach without relaxing rescue distance gate."""
    del now_t  # kept for optional future time-based smoothing
    global _MAP_INFLATION_RADIUS_CELLS_RUNTIME
    global _FINAL_APPROACH_ACTIVE
    global _FINAL_APPROACH_TARGET_DEF

    snap = _active_rescue_distance_snapshot()
    target_def = None if snap is None else str(snap["def_name"])
    dist_m = None if snap is None else float(snap["dist_m"])

    should_enable = False
    if target_def is not None and dist_m is not None and math.isfinite(dist_m):
        if _FINAL_APPROACH_ACTIVE and _FINAL_APPROACH_TARGET_DEF == target_def:
            should_enable = (float(RESCUE_RADIUS_M) < dist_m <= float(FINAL_APPROACH_EXIT_DISTANCE_M))
        else:
            should_enable = (float(RESCUE_RADIUS_M) < dist_m <= float(FINAL_APPROACH_ENTER_DISTANCE_M))

    if should_enable:
        _MAP_INFLATION_RADIUS_CELLS_RUNTIME = min(
            int(MAP_INFLATION_RADIUS_CELLS),
            int(FINAL_APPROACH_INFLATION_RADIUS_CELLS),
        )
        NAV.config.avoid_gain = min(float(NAV.config.avoid_gain), float(FINAL_APPROACH_AVOID_GAIN))
        NAV.config.safety_distance_m = min(float(NAV.config.safety_distance_m), float(FINAL_APPROACH_SAFETY_DISTANCE_M))
        NAV.config.side_clearance_m = min(float(NAV.config.side_clearance_m), float(FINAL_APPROACH_SIDE_CLEARANCE_M))
        NAV.config.stop_distance_m = min(float(NAV.config.stop_distance_m), float(FINAL_APPROACH_STOP_DISTANCE_M))
        NAV.config.lookahead_m = min(float(NAV.config.lookahead_m), float(FINAL_APPROACH_LOOKAHEAD_M))
        NAV.config.max_v = min(float(NAV.config.max_v), float(FINAL_APPROACH_MAX_V_MPS))
        NAV.config.max_omega = max(float(NAV.config.max_omega), float(FINAL_APPROACH_MAX_OMEGA_RADPS))
        NAV.config.inflation_radius_m = min(
            float(NAV.config.inflation_radius_m),
            float(FINAL_APPROACH_INFLATION_RADIUS_CELLS) * float(MAP_RESOLUTION),
        )
    else:
        _MAP_INFLATION_RADIUS_CELLS_RUNTIME = int(MAP_INFLATION_RADIUS_CELLS)
        NAV.config.avoid_gain = float(NAV_BASELINE_TUNING["avoid_gain"])
        NAV.config.safety_distance_m = float(NAV_BASELINE_TUNING["safety_distance_m"])
        NAV.config.side_clearance_m = float(NAV_BASELINE_TUNING["side_clearance_m"])
        NAV.config.stop_distance_m = float(NAV_BASELINE_TUNING["stop_distance_m"])
        NAV.config.lookahead_m = float(NAV_BASELINE_TUNING["lookahead_m"])
        NAV.config.max_omega = float(NAV_BASELINE_TUNING["max_omega"])
        NAV.config.inflation_radius_m = float(NAV_BASELINE_TUNING["inflation_radius_m"])

    transition = (should_enable != _FINAL_APPROACH_ACTIVE) or (_FINAL_APPROACH_TARGET_DEF != target_def)
    if transition:
        state = "ON" if should_enable else "OFF"
        dist_txt = "na" if dist_m is None or not math.isfinite(dist_m) else f"{dist_m:.3f}"
        print(
            "FINAL_APPROACH_MODE "
            f"state={state} target_def={target_def or 'none'} dist_m={dist_txt} "
            f"infl_cells={_MAP_INFLATION_RADIUS_CELLS_RUNTIME} "
            f"avoid_gain={NAV.config.avoid_gain:.2f} safety_distance_m={NAV.config.safety_distance_m:.2f} "
            f"max_v={NAV.config.max_v:.2f} max_omega={NAV.config.max_omega:.2f}"
        )
        log_event(
            "final_approach_mode",
            {
                "state": state,
                "target_def": target_def,
                "dist_m": dist_m,
                "inflation_radius_cells": int(_MAP_INFLATION_RADIUS_CELLS_RUNTIME),
                "avoid_gain": float(NAV.config.avoid_gain),
                "safety_distance_m": float(NAV.config.safety_distance_m),
                "side_clearance_m": float(NAV.config.side_clearance_m),
                "stop_distance_m": float(NAV.config.stop_distance_m),
                "lookahead_m": float(NAV.config.lookahead_m),
                "max_v": float(NAV.config.max_v),
                "max_omega": float(NAV.config.max_omega),
            },
        )

    _FINAL_APPROACH_ACTIVE = bool(should_enable)
    _FINAL_APPROACH_TARGET_DEF = target_def

_last_nav_speed_profile = None
def update_nav_speed_profile(now_t):
    """Increase forward speed as more victims are rescued (mission progression)."""
    global _last_nav_speed_profile
    rescued_n = int(len(STATE.rescued_victim_ids))
    # Gentle time ramp plus reward for completed rescues.
    time_bonus = min(0.12, max(0.0, float(now_t)) * 0.003)
    target_v = min(
        float(MISSION_NAV_MAX_V_CAP_MPS),
        float(MISSION_NAV_MAX_V_MPS) + rescued_n * float(MISSION_NAV_SPEED_GAIN_PER_RESCUE_MPS) + time_bonus,
    )
    if abs(float(NAV.config.max_v) - target_v) > 1e-6:
        NAV.config.max_v = target_v
    speed_key = round(float(NAV.config.max_v), 3)
    if _last_nav_speed_profile != speed_key:
        _last_nav_speed_profile = speed_key
        print(f"NAV_SPEED_PROFILE max_v={NAV.config.max_v:.2f} rescued={rescued_n}")
        log_event("nav_speed_profile", {"max_v": float(NAV.config.max_v), "rescued_count": rescued_n})

print(f"Mission: scan map with camera, rescue {STATE.required_victims} objects, then stop.")
log_event("mission_start", {"required_victims": STATE.required_victims})

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

def nearest_checkpoint(robot_pose, checkpoints=None):
    """Return the nearest configured checkpoint to the robot pose."""
    checkpoints = CHECKPOINTS if checkpoints is None else checkpoints
    if not checkpoints or not _pose_xy_valid(robot_pose):
        return None
    return min(
        checkpoints,
        key=lambda cp: _dist_xy(robot_pose["x"], robot_pose["y"], cp["x"], cp["y"]),
    )

def checkpoint_reached(robot_pose, checkpoint, threshold_m=None):
    """Return True when the robot is within the checkpoint threshold."""
    if checkpoint is None or not _pose_xy_valid(robot_pose):
        return False
    threshold_m = CHECKPOINT_ENTER_THRESHOLD_M if threshold_m is None else float(threshold_m)
    return _dist_xy(robot_pose["x"], robot_pose["y"], checkpoint["x"], checkpoint["y"]) <= threshold_m

def maybe_update_checkpoint_tracking(now_t):
    """Software-only checkpoint enter/exit tracking (no world edits required)."""
    if not CHECKPOINTS or not _pose_xy_valid(STATE.pose):
        return
    poll_period = 1.0 / max(1e-6, CHECKPOINT_POLL_HZ)
    if (now_t - STATE.last_checkpoint_check_t) < poll_period:
        return
    STATE.last_checkpoint_check_t = now_t

    active = None
    if STATE.current_checkpoint_name is not None:
        for cp in CHECKPOINTS:
            if cp["name"] == STATE.current_checkpoint_name:
                active = cp
                break

    if active is not None:
        exit_threshold = float(active.get("threshold_m", CHECKPOINT_ENTER_THRESHOLD_M)) + CHECKPOINT_EXIT_HYSTERESIS_M
        if not checkpoint_reached(STATE.pose, active, threshold_m=exit_threshold):
            log_event("checkpoint_exit", {"name": active["name"], "x": active["x"], "y": active["y"]})
            print(f"CHECKPOINT_EXIT name={active['name']}")
            STATE.current_checkpoint_name = None

    if STATE.current_checkpoint_name is None:
        cp = nearest_checkpoint(STATE.pose)
        if cp is not None and checkpoint_reached(STATE.pose, cp, threshold_m=float(cp.get("threshold_m", CHECKPOINT_ENTER_THRESHOLD_M))):
            STATE.current_checkpoint_name = cp["name"]
            log_event("checkpoint_enter", {"name": cp["name"], "x": cp["x"], "y": cp["y"]})
            print(f"CHECKPOINT_ENTER name={cp['name']}")

def _map_known_fraction(map_arr):
    """Fraction of cells that are known (not -1)."""
    if map_arr is None:
        return 0.0
    data = np.asarray(map_arr)
    total = int(data.size)
    if total <= 0:
        return 0.0
    known = int(np.count_nonzero(data != -1))
    return float(known) / float(total)

def _exploration_completion_status(now_t):
    """Return map-exploration completion diagnostics."""
    grid = STATE.map_raw if STATE.map_raw is not None else STATE.map
    known_fraction = _map_known_fraction(grid)
    no_frontier_age_s = None
    if STATE.no_frontier_since_t is not None:
        no_frontier_age_s = max(0.0, float(now_t) - float(STATE.no_frontier_since_t))
    stable_map_age_s = max(0.0, float(now_t) - float(STATE.last_map_version_bump_t))
    coverage_gate = bool(known_fraction >= float(EXPLORATION_MIN_KNOWN_FRACTION))
    no_frontier_gate = bool(
        no_frontier_age_s is not None and no_frontier_age_s >= float(EXPLORATION_NO_FRONTIER_HOLD_S)
    )
    stable_map_gate = bool(stable_map_age_s >= float(EXPLORATION_STABLE_MAP_HOLD_S))
    complete = bool(coverage_gate and no_frontier_gate and stable_map_gate)
    return {
        "known_fraction": float(known_fraction),
        "no_frontier_age_s": no_frontier_age_s,
        "stable_map_age_s": float(stable_map_age_s),
        "coverage_gate": coverage_gate,
        "no_frontier_gate": no_frontier_gate,
        "stable_map_gate": stable_map_gate,
        "complete": complete,
    }

def _select_exploration_fallback_checkpoint():
    """Pick a checkpoint fallback goal when frontier detection yields no target."""
    if not CHECKPOINTS or not _pose_xy_valid(STATE.pose):
        return None
    n = len(CHECKPOINTS)
    if n <= 0:
        return None

    # Prefer round-robin checkpoints that are not too close to the current pose.
    for offset in range(n):
        idx = (int(STATE.next_explore_checkpoint_idx) + offset) % n
        cp = CHECKPOINTS[idx]
        d = _dist_xy(STATE.pose["x"], STATE.pose["y"], cp["x"], cp["y"])
        if d >= float(EXPLORATION_FALLBACK_MIN_TRAVEL_M):
            if _explore_goal_block_reason(cp["x"], cp["y"]) is not None:
                continue
            STATE.next_explore_checkpoint_idx = (idx + 1) % n
            return cp

    # If all checkpoints are close, choose the farthest one to force movement.
    candidates = [
        c for c in CHECKPOINTS
        if _explore_goal_block_reason(c["x"], c["y"]) is None
    ]
    if not candidates:
        return None
    cp = max(
        candidates,
        key=lambda c: _dist_xy(STATE.pose["x"], STATE.pose["y"], c["x"], c["y"]),
    )
    try:
        idx = CHECKPOINTS.index(cp)
        STATE.next_explore_checkpoint_idx = (idx + 1) % n
    except Exception:
        pass
    return cp

def maybe_queue_exploration_fallback_goal(now_t, reason="frontier_none", force=False):
    """Queue a checkpoint-based exploration goal when no frontiers are available."""
    if STATE.goal is not None:
        return False
    if (not bool(force)) and (
        (float(now_t) - float(STATE.last_explore_fallback_goal_t))
        < float(EXPLORATION_FALLBACK_GOAL_PERIOD_S)
    ):
        return False
    cp = _select_exploration_fallback_checkpoint()
    if cp is None:
        return False
    STATE.last_explore_fallback_goal_t = float(now_t)
    set_goal(cp["x"], cp["y"], goal_kind="explore")
    log_event(
        "explore_fallback_goal",
        {"name": cp["name"], "x": cp["x"], "y": cp["y"], "reason": str(reason)},
    )
    print(f"EXPLORE_FALLBACK_GOAL name={cp['name']} x={cp['x']:.2f} y={cp['y']:.2f}")
    return True

def _choose_shared_path_points(nav_snapshot):
    """Prefer smoothed waypoints for the shared path contract, fallback to raw path."""
    waypoints = nav_snapshot.get("waypoints") or []
    if waypoints:
        return waypoints, "waypoints"
    path_world = nav_snapshot.get("path_world") or []
    if path_world:
        return path_world, "global_path"
    return [], "empty"

def sync_shared_path_from_nav(reason="nav_step"):
    """Wire NavigationStack path output into the Member 1 shared path interface."""
    snapshot = NAV.get_path_snapshot()
    points, path_kind = _choose_shared_path_points(snapshot)
    nav_mode = str(snapshot.get("mode", "UNKNOWN"))
    if points and STATE.goal is not None:
        set_path(points, reason=f"{reason}:{path_kind}:{nav_mode}")
    else:
        clear_path(reason=f"{reason}:{path_kind}:{nav_mode}")

def _map_hash32(arr):
    """Cheap deterministic grid hash for map-version decisions."""
    if arr is None:
        return None
    # Hash a semantic grid (unknown/free/occupied) to avoid noisy value changes.
    grid_u8 = _semantic_grid_u8(arr)
    return int(zlib.adler32(np.ascontiguousarray(grid_u8).tobytes()) & 0xFFFFFFFF)

def _semantic_grid_u8(arr):
    """Map occupancy values to stable classes: unknown=0, free=1, occupied=2."""
    if arr is None:
        return None
    data = np.asarray(arr)
    semantic = np.zeros(data.shape, dtype=np.uint8)
    semantic[data == 0] = 1
    semantic[data > 0] = 2
    return semantic

def _occupied_grid_u8(arr):
    """Stable occupied-mask grid for replan versioning (0=not occupied, 1=occupied)."""
    if arr is None:
        return None
    data = np.asarray(arr)
    return (data > 0).astype(np.uint8, copy=False)

def _occupied_hash32(arr):
    """Cheap deterministic hash of occupied cells only."""
    occ_u8 = _occupied_grid_u8(arr)
    if occ_u8 is None:
        return None
    return int(zlib.adler32(np.ascontiguousarray(occ_u8).tobytes()) & 0xFFFFFFFF)

def _meaningful_map_change_count(prev_map, curr_map):
    """Count semantic changes that matter for replanning (newly known/newly occupied)."""
    curr_sem = _semantic_grid_u8(curr_map)
    if curr_sem is None:
        return 0
    if prev_map is None:
        return int(np.count_nonzero(curr_sem != 0))

    prev_sem = _semantic_grid_u8(prev_map)
    newly_known = np.count_nonzero((prev_sem == 0) & (curr_sem != 0))
    newly_occupied = np.count_nonzero((prev_sem != 2) & (curr_sem == 2))
    return int(newly_known + newly_occupied)

def _meaningful_obstacle_change_count(prev_map, curr_map):
    """Count occupied-cell changes only, reducing replans from free-space updates/noise."""
    curr_occ = _occupied_grid_u8(curr_map)
    if curr_occ is None:
        return 0
    if prev_map is None:
        return int(np.count_nonzero(curr_occ))
    prev_occ = _occupied_grid_u8(prev_map)
    return int(np.count_nonzero(prev_occ != curr_occ))

def maybe_update_mapping(ranges, now_t):
    """Throttled occupancy-grid update with deterministic map versioning."""
    if not USE_MAPPING:
        return
    period_s = 1.0 / max(1e-6, MAP_UPDATE_HZ)
    if (now_t - STATE.last_map_update_t) < period_s and STATE.map is not None:
        return

    prev_map = None if STATE.map is None else STATE.map.copy()
    prev_raw_map = None if STATE.map_raw is None else STATE.map_raw.copy()
    carve_defs = _target_defs_for_victim_filtering()
    if not carve_defs:
        carve_defs = None
    # Update on the raw map; inflating an already-inflated grid each cycle makes
    # obstacles grow over time and can keep forcing replans.
    new_raw_map = update_map(ranges, STATE.pose, STATE.map_raw, ray_stride=LIDAR_DOWNSAMPLE)
    _carve_rescue_targets_from_map(new_raw_map, target_defs=carve_defs)
    STATE.map_raw = new_raw_map
    inflation_cells = max(0, int(_MAP_INFLATION_RADIUS_CELLS_RUNTIME))
    new_map = inflate_obstacles(new_raw_map, inflation_radius=inflation_cells)
    _carve_rescue_targets_from_map(new_map, target_defs=carve_defs)
    STATE.map = new_map
    STATE.last_map_update_t = now_t

    semantic_changed_cells = _meaningful_map_change_count(prev_map, STATE.map)
    obstacle_changed_cells = _meaningful_obstacle_change_count(prev_raw_map, STATE.map_raw)
    changed_cells = obstacle_changed_cells
    STATE.last_map_changed_cells = changed_cells

    grid_hash = _occupied_hash32(STATE.map_raw)
    hash_changed = (grid_hash != STATE.last_grid_hash)
    meaningful_change = bool(changed_cells > MAP_CHANGE_CELL_THRESHOLD)
    if hash_changed and meaningful_change:
        prev_hash = STATE.last_grid_hash
        STATE.last_grid_hash = grid_hash
        STATE.map_version += 1
        STATE.last_map_version_bump_t = float(now_t)
        print(f"MAP_VERSION++ v={STATE.map_version}, changed_cells={changed_cells}, hash={grid_hash}")
        log_event(
            "map_version",
            {
                "version": STATE.map_version,
                "changed_cells": changed_cells,
                "semantic_changed_cells": semantic_changed_cells,
                "obstacle_changed_cells": obstacle_changed_cells,
                "hash": grid_hash,
                "prev_hash": prev_hash,
                "meaningful": meaningful_change,
            },
        )

_last_victim_metrics_write = -1e9
def maybe_write_victim_metrics(now_t, force=False):
    """Periodically write non-ROS victim metrics summary JSON."""
    global _last_victim_metrics_write
    if VICTIM_METRICS is None or not VICTIM_METRICS_SUMMARY_PATH:
        return
    if (not force) and (now_t - _last_victim_metrics_write) < VICTIM_METRICS_WRITE_PERIOD_S:
        return
    _last_victim_metrics_write = now_t
    summary = VICTIM_METRICS.write_summary(
        VICTIM_METRICS_SUMMARY_PATH,
        extra={
            "run_time_s": float(now_t),
            "rescued_count": int(len(STATE.rescued_victim_ids)),
            "mission_done": bool(STATE.mission_done),
        },
    )
    log_event(
        "victim_metrics_summary",
        {
            "accepted_detections": int(summary.get("accepted_detections", 0)),
            "unique_victim_locations": int(summary.get("unique_victim_locations", 0)),
            "duplicate_detection_candidates": int(summary.get("duplicate_detection_candidates", 0)),
        },
    )

# =========================================================
# Main loop (Integration priority order)
# 1) Update pose/log
# 2) Mapping (if enabled)
# 3) Exploration goal (if enabled)
# 4) Planning path (if enabled)
# 5) Follow path (if enabled) else fallback to Braitenberg
# 6) Victim detection hook (if enabled)
# =========================================================
def maybe_queue_rescue_goal(reason):
    """Prioritize detected victims over exploration goals."""
    if STATE.goal_kind == "rescue" and STATE.goal is not None and STATE.active_rescue_id is not None:
        return False

    victim = nearest_unrescued_victim(STATE.pose)
    if victim is None:
        return False

    victim_id = int(victim["id"])
    if STATE.goal_kind == "explore" and STATE.goal is not None:
        clear_goal(reason="preempt_for_rescue")

    if STATE.goal is None:
        set_goal(victim["x"], victim["y"], goal_kind="rescue", rescue_id=victim_id)
        print(f"Targeting victim #{victim_id} for rescue")
        log_event("rescue_target_selected", {
            "id": victim_id,
            "x": victim["x"],
            "y": victim["y"],
            "reason": reason,
        })
        return True
    return False

def maybe_queue_world_target_rescue_goal(now_t, reason="world_target_task"):
    """Guarantee rescue-task intent by queuing nearest unrescued world target when needed."""
    if not FORCE_WORLD_TARGET_RESCUE_TASK:
        return False
    if float(now_t) < float(WORLD_TARGET_TASK_START_DELAY_S):
        return False
    if not _pose_xy_valid(STATE.pose):
        return False

    # Keep current rescue objective stable; only preempt exploration/idle.
    if STATE.goal_kind == "rescue" and STATE.goal is not None and STATE.active_rescue_id is not None:
        return False
    if STATE.goal is not None and STATE.goal_kind not in ("none", "explore"):
        return False
    if (float(now_t) - float(STATE.last_world_target_task_goal_t)) < float(WORLD_TARGET_TASK_GOAL_PERIOD_S):
        return False

    nearest = None
    nearest_dist = float("inf")
    rx = float(STATE.pose["x"])
    ry = float(STATE.pose["y"])
    for def_name, pos in iter_unrescued_world_targets():
        if pos is None or len(pos) < 2:
            continue
        tx = float(pos[0])
        ty = float(pos[1])
        d = _dist_xy(rx, ry, tx, ty)
        if d < nearest_dist:
            nearest_dist = d
            nearest = (str(def_name), tx, ty)
    if nearest is None:
        return False

    def_name, tx, ty = nearest
    victim = _ensure_victim_for_world_target(def_name)
    if victim is None:
        return False

    # Seed short-lived evidence for close-range rescue gate, then set rescue goal.
    _mark_world_target_seen(def_name, source="task_seed")

    victim_id = int(victim["id"])
    if STATE.goal_kind == "explore" and STATE.goal is not None:
        clear_goal(reason="preempt_for_world_target_task")
    if STATE.goal is not None:
        return False

    set_goal(float(tx), float(ty), goal_kind="rescue", rescue_id=victim_id)
    STATE.last_world_target_task_goal_t = float(now_t)
    print(f"WORLD_TARGET_TASK rescue_goal def={def_name} victim_id={victim_id} dist_m={nearest_dist:.2f}")
    log_event(
        "rescue_target_selected",
        {
            "id": victim_id,
            "x": float(tx),
            "y": float(ty),
            "reason": str(reason),
            "source": "world_target_task",
            "world_target_def": str(def_name),
            "distance_m": float(nearest_dist),
        },
    )
    return True

def maybe_queue_startup_direct_rescue_goal(ranges, now_t, reason="startup_direct_rescue"):
    """During startup, preempt exploration and drive straight toward a visible, clear rescue target."""
    if not ENABLE_STARTUP_DIRECT_RESCUE:
        return False
    if float(now_t) > float(STARTUP_DIRECT_RESCUE_WINDOW_S):
        return False
    if not _pose_xy_valid(STATE.pose):
        return False
    if not _WORLD_TARGET_TRANS_FIELDS:
        return False

    # Allow preempting exploration, but do not override an active rescue intent.
    if STATE.goal is not None and STATE.goal_kind != "explore":
        return False

    target_defs = _target_defs_for_victim_filtering()
    if not target_defs:
        return False

    scan = _filter_lidar_targets_from_avoidance(ranges, target_defs=target_defs)
    if scan is None:
        scan = ranges
    if scan is None:
        return False

    arr = np.asarray(scan, dtype=np.float64)
    valid = arr[np.isfinite(arr) & (arr > 0.0)]
    if valid.size > 0 and float(np.min(valid)) < float(STARTUP_DIRECT_RESCUE_GLOBAL_MIN_M):
        return False

    try:
        half_fov = 0.5 * float(lidar.getFov())
    except Exception:
        half_fov = math.pi

    rx = float(STATE.pose["x"])
    ry = float(STATE.pose["y"])
    yaw = float(STATE.pose["yaw"])

    best = None
    for def_name, pos in iter_unrescued_world_targets():
        tx = float(pos[0])
        ty = float(pos[1])
        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)
        if dist <= float(RESCUE_RADIUS_M):
            continue

        bearing = _wrap_angle(math.atan2(dy, dx) - yaw)
        # Only pick targets visible in the current forward scan so robot does not choose a behind target.
        if abs(bearing) > (half_fov + 0.05):
            continue

        corridor_min = _lidar_corridor_min_range(
            scan,
            bearing,
            half_width_deg=STARTUP_DIRECT_RESCUE_CORRIDOR_HALF_DEG,
        )
        if corridor_min < float(STARTUP_DIRECT_RESCUE_MIN_CLEARANCE_M):
            continue

        # Prefer close targets with smaller heading deviation.
        score = float(dist) + 0.7 * abs(float(bearing))
        if best is None or score < best["score"]:
            best = {
                "def_name": str(def_name),
                "x": tx,
                "y": ty,
                "dist": float(dist),
                "bearing": float(bearing),
                "corridor_min": float(corridor_min),
                "score": score,
            }

    if best is None:
        return False

    victim = _ensure_victim_for_world_target(best["def_name"])
    if victim is None:
        return False
    victim_id = int(victim["id"])

    if STATE.goal is not None and STATE.goal_kind == "explore":
        clear_goal(reason="preempt_for_startup_direct_rescue")
    set_goal(best["x"], best["y"], goal_kind="rescue", rescue_id=victim_id)

    print(
        "STARTUP_DIRECT_RESCUE "
        f"def={best['def_name']} victim_id={victim_id} "
        f"dist_m={best['dist']:.2f} bearing_deg={math.degrees(best['bearing']):.1f} "
        f"corridor_min_m={best['corridor_min']:.2f}"
    )
    log_event(
        "startup_direct_rescue_goal",
        {
            "def_name": best["def_name"],
            "victim_id": victim_id,
            "x": float(best["x"]),
            "y": float(best["y"]),
            "dist_m": float(best["dist"]),
            "bearing_deg": float(math.degrees(best["bearing"])),
            "corridor_min_m": float(best["corridor_min"]),
            "reason": str(reason),
        },
    )
    return True

def maybe_rescue_active_target(reason):
    _rescue_trace(
        "maybe_rescue_active_target",
        "ENTER",
        reason=reason,
        candidate_victim_id=STATE.active_rescue_id,
        has_supervisor=_SUPERVISOR_ENABLED,
        has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        rescued_count_before=len(STATE.rescued_victim_ids),
    )
    if STATE.active_rescue_id is None:
        _rescue_gate_table(
            "maybe_rescue_active_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", False),
                ("matching_success", None),
                ("already_rescued", None),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
            ],
            reason="no_active_target_id",
        )
        _rescue_trace(
            "maybe_rescue_active_target",
            "RETURN",
            reason="no_active_target_id",
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        )
        return False
    if not _pose_xy_valid(STATE.pose):
        _rescue_gate_table(
            "maybe_rescue_active_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", bool(STATE.active_rescue_id)),
                ("matching_success", None),
                ("already_rescued", None),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
                ("pose_valid", False),
            ],
            reason="invalid_pose",
            candidate_victim_id=STATE.active_rescue_id,
        )
        _rescue_trace(
            "maybe_rescue_active_target",
            "RETURN",
            reason="invalid_pose",
            candidate_victim_id=STATE.active_rescue_id,
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        )
        return False
    victim = find_victim_by_id(STATE.active_rescue_id)
    if victim is None:
        _rescue_gate_table(
            "maybe_rescue_active_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", False),
                ("matching_success", False),
                ("already_rescued", None),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", None),
            ],
            reason="victim_not_found",
            candidate_victim_id=STATE.active_rescue_id,
        )
        _rescue_trace(
            "maybe_rescue_active_target",
            "RETURN",
            reason="victim_not_found",
            candidate_victim_id=STATE.active_rescue_id,
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        )
        return False
    target_def = victim.get("world_target_def")
    tx = float(victim["x"])
    ty = float(victim["y"])
    target_xy_source = "victim_registry"
    if target_def:
        target_pos = _world_target_xy(str(target_def))
        if target_pos is not None:
            tx = float(target_pos[0])
            ty = float(target_pos[1])
            target_xy_source = "world_target_xy"
    gate_eval = _evaluate_close_range_rescue_trigger(
        "maybe_rescue_active_target",
        (tx, ty),
        victim=victim,
        target_def=target_def,
        target_xy_source=target_xy_source,
        reason=f"{reason}:final_gate",
    )
    if not bool(gate_eval.get("ok", False)):
        robot_xy_tuple = gate_eval.get("robot_xy")
        target_xy_tuple = gate_eval.get("target_xy")
        _rescue_trace(
            "maybe_rescue_active_target",
            "RETURN",
            reason=f"close_range_gate_failed:{gate_eval.get('first_fail', 'unknown')}",
            candidate_victim_id=STATE.active_rescue_id,
            candidate_world_DEF=target_def,
            robot_xy=(
                f"({float(robot_xy_tuple[0]):.2f},{float(robot_xy_tuple[1]):.2f})"
                if isinstance(robot_xy_tuple, tuple) and len(robot_xy_tuple) >= 2
                else "none"
            ),
            target_xy=(
                f"({float(target_xy_tuple[0]):.2f},{float(target_xy_tuple[1]):.2f})"
                if isinstance(target_xy_tuple, tuple) and len(target_xy_tuple) >= 2
                else "none"
            ),
            dx=gate_eval.get("dx"),
            dy=gate_eval.get("dy"),
            dist_m=gate_eval.get("dist_m"),
            threshold_m=gate_eval.get("threshold_m"),
            target_xy_source=target_xy_source,
            recent_seen=gate_eval.get("recent_seen"),
            recent_seen_age_ticks=gate_eval.get("recent_seen_age_ticks"),
            goal_evidence=gate_eval.get("goal_evidence"),
            goal_evidence_reason=gate_eval.get("goal_evidence_reason"),
            evidence_any=gate_eval.get("evidence_gate"),
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
        )
        return False
    dx = gate_eval.get("dx")
    dy = gate_eval.get("dy")
    dist_m = gate_eval.get("dist_m")
    threshold_m = gate_eval.get("threshold_m")
    robot_xy_tuple = gate_eval.get("robot_xy")
    target_xy_tuple = gate_eval.get("target_xy")
    rescued_before = len(STATE.rescued_victim_ids)
    rescued = rescue_victim(STATE.active_rescue_id, reason)
    if rescued and STATE.goal is not None:
        clear_goal(reason="rescue_completed")
    if rescued:
        _rescue_gate_table(
            "maybe_rescue_active_target",
            "SUCCESS",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", True),
                ("matching_success", target_xy_source in {"world_target_xy", "victim_registry"}),
                ("already_rescued", int(victim["id"]) not in STATE.rescued_victim_ids or len(STATE.rescued_victim_ids) > rescued_before),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", True),
                ("rescue_victim_call", True),
            ],
            candidate_victim_id=int(victim["id"]),
            candidate_world_DEF=target_def,
            dist_m=dist_m,
            threshold_m=threshold_m,
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
            target_xy_source=target_xy_source,
        )
        _rescue_trace(
            "maybe_rescue_active_target",
            "SUCCESS",
            candidate_victim_id=int(victim["id"]),
            candidate_world_DEF=target_def,
            robot_xy=(
                f"({float(robot_xy_tuple[0]):.2f},{float(robot_xy_tuple[1]):.2f})"
                if isinstance(robot_xy_tuple, tuple) and len(robot_xy_tuple) >= 2
                else "none"
            ),
            target_xy=(
                f"({float(target_xy_tuple[0]):.2f},{float(target_xy_tuple[1]):.2f})"
                if isinstance(target_xy_tuple, tuple) and len(target_xy_tuple) >= 2
                else "none"
            ),
            dx=dx,
            dy=dy,
            dist_m=dist_m,
            threshold_m=threshold_m,
            target_xy_source=target_xy_source,
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
    else:
        _rescue_gate_table(
            "maybe_rescue_active_target",
            "RETURN",
            [
                ("supervisor_available", _SUPERVISOR_ENABLED),
                ("handles_initialized", bool(_WORLD_TARGET_TRANS_FIELDS)),
                ("candidate_target_exists", True),
                ("matching_success", target_xy_source in {"world_target_xy", "victim_registry"}),
                ("already_rescued", int(victim["id"]) not in STATE.rescued_victim_ids),
                ("cooldown_throttle", True),
                ("distance_le_rescue_radius", True),
                ("rescue_victim_call", False),
            ],
            reason="rescue_victim_failed",
            candidate_victim_id=int(victim["id"]),
            candidate_world_DEF=target_def,
            dist_m=dist_m,
            threshold_m=threshold_m,
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
            target_xy_source=target_xy_source,
        )
        _rescue_trace(
            "maybe_rescue_active_target",
            "RETURN",
            reason="rescue_victim_failed",
            candidate_victim_id=int(victim["id"]),
            candidate_world_DEF=target_def,
            robot_xy=(
                f"({float(robot_xy_tuple[0]):.2f},{float(robot_xy_tuple[1]):.2f})"
                if isinstance(robot_xy_tuple, tuple) and len(robot_xy_tuple) >= 2
                else "none"
            ),
            target_xy=(
                f"({float(target_xy_tuple[0]):.2f},{float(target_xy_tuple[1]):.2f})"
                if isinstance(target_xy_tuple, tuple) and len(target_xy_tuple) >= 2
                else "none"
            ),
            dx=dx,
            dy=dy,
            dist_m=dist_m,
            threshold_m=threshold_m,
            target_xy_source=target_xy_source,
            has_supervisor=_SUPERVISOR_ENABLED,
            has_handles=bool(_WORLD_TARGET_TRANS_FIELDS),
            rescued_count_before=rescued_before,
            rescued_count_after=len(STATE.rescued_victim_ids),
        )
    return rescued

def mission_done_reached(now_t):
    if len(STATE.victims) >= STATE.required_victims:
        STATE.mission_done_reason = "found_all"
        return True
    return False

pose_flush_counter = 0
last_nav_mode = None
loop_debug_tick = 0

while robot.step(ts) != -1:
    loop_debug_tick += 1
    t = robot.getTime()

    # ---- Pose
    x, y, yaw = get_pose()
    STATE.pose = {"x": x, "y": y, "yaw": yaw}
    pose_csv.write(f"{t:.3f},{x},{y},{yaw}\n")
    pose_flush_counter += 1
    if pose_flush_counter >= 10:
        pose_flush_counter = 0
        pose_csv.flush()

    # ---- Baseline debug heartbeat (grep: LOOP_STATUS)
    if (loop_debug_tick % 10) == 0:
        active_goal_str = "none"
        if STATE.goal is not None:
            active_goal_str = f"{STATE.goal_kind}:{STATE.goal[0]:.2f},{STATE.goal[1]:.2f}"

        active_target_id = STATE.active_rescue_id if STATE.active_rescue_id is not None else "none"
        active_target_def = "none"
        if STATE.active_rescue_id is not None:
            _active_victim = find_victim_by_id(STATE.active_rescue_id)
            if _active_victim is not None:
                active_target_def = str(_active_victim.get("world_target_def") or "none")

        nearest_def = "na"
        nearest_dist = "na"
        known_fraction = _map_known_fraction(STATE.map_raw if STATE.map_raw is not None else STATE.map)
        no_frontier_age = "na"
        if STATE.no_frontier_since_t is not None:
            no_frontier_age = f"{max(0.0, t - STATE.no_frontier_since_t):.1f}"
        if _WORLD_TARGET_TRANS_FIELDS and _pose_xy_valid(STATE.pose):
            _best_def = None
            _best_d = float("inf")
            for _def_name, _pos in iter_unrescued_world_targets():
                _d = _dist_xy(STATE.pose["x"], STATE.pose["y"], _pos[0], _pos[1])
                if _d < _best_d:
                    _best_d = _d
                    _best_def = _def_name
            if _best_def is not None:
                nearest_def = str(_best_def)
                nearest_dist = f"{_best_d:.2f}"

        print(
            "LOOP_STATUS "
            f"tick={loop_debug_tick} "
            f"pose=({STATE.pose['x']:.2f},{STATE.pose['y']:.2f},{STATE.pose['yaw']:.2f}) "
            f"active_goal={active_goal_str} "
            f"active_target_id={active_target_id} "
            f"active_target_def={active_target_def} "
            f"rescued_count={len(STATE.rescued_victim_ids)} "
            f"known_fraction={known_fraction:.3f} "
            f"no_frontier_age_s={no_frontier_age} "
            f"nearest_world_target_def={nearest_def} "
            f"nearest_world_target_distance_m={nearest_dist}"
        )
        _audit_world_target_runtime("loop_heartbeat", force=False)

    # ---- Software checkpoints (Member 1, low-rate)
    maybe_update_checkpoint_tracking(t)
    maybe_update_final_approach_mode(t)

    # ---- LiDAR ranges (for mapping + behavior)
    ranges = lidar.getRangeImage()

    # ---- Mapping (Member 2)
    maybe_update_mapping(ranges, t)

    # ---- If already close enough to a rescue target, complete rescue immediately
    maybe_rescue_active_target("within_rescue_radius")
    if USE_WORLD_TARGET_PROXIMITY_RESCUE:
        maybe_rescue_near_world_target("world_target_proximity_pre_nav")

    # ---- Camera victim detection before planning/nav so rescue can preempt motion
    process_victim_detections(t)

    # ---- Rescue targets take priority over frontier exploration
    maybe_queue_rescue_goal("pending_detected_victim")
    maybe_queue_world_target_rescue_goal(t, reason="pending_world_target_task")
    if ENABLE_STARTUP_DIRECT_RESCUE:
        maybe_queue_startup_direct_rescue_goal(ranges, t, reason="startup_clear_corridor")

    # ---- Exploration (Member 3): only when no rescue or active goal exists
    if USE_EXPLORATION and STATE.goal is None and STATE.map is not None:
        g = choose_frontier_goal(STATE.map, STATE.pose)
        if g is not None:
            if STATE.no_frontier_since_t is not None:
                no_frontier_dur = max(0.0, float(t) - float(STATE.no_frontier_since_t))
                log_event("frontier_reacquired", {"after_s": float(no_frontier_dur)})
            STATE.no_frontier_since_t = None
            g = jitter_exploration_goal(g, STATE.map)
            set_goal(g[0], g[1], goal_kind="explore")
        else:
            if STATE.no_frontier_since_t is None:
                STATE.no_frontier_since_t = float(t)
                status = _exploration_completion_status(t)
                log_event(
                    "frontier_none",
                    {
                        "known_fraction": float(status["known_fraction"]),
                        "stable_map_age_s": float(status["stable_map_age_s"]),
                    },
                )
            status = _exploration_completion_status(t)
            should_keep_patrolling = bool(KEEP_RUNNING_AFTER_MISSION_DONE and STATE.mission_done)
            if (not bool(status["complete"])) or should_keep_patrolling:
                maybe_queue_exploration_fallback_goal(t, reason="frontier_none")

    # ---- Navigation Stack (Member 4 replacing manual planning/following)
    if USE_PLANNING:
        update_nav_speed_profile(t)
        maybe_update_final_approach_mode(t)
        maybe_update_startup_direct_planner_policy(t)
        NAV.step(t)
        sync_shared_path_from_nav(reason="nav_step")
        nav_mode = NAV.get_debug().get("mode")
        if nav_mode == "GOAL_REACHED" and last_nav_mode != "GOAL_REACHED" and STATE.goal is not None:
            reached_kind = STATE.goal_kind
            reached_rescue_id = STATE.active_rescue_id
            if reached_kind == "rescue" and reached_rescue_id is not None:
                rescue_victim(reached_rescue_id, "nav_goal_reached")
            clear_goal(reason="nav_goal_reached")
            # Keep motion continuous: immediately queue the next task instead of idling for a cycle.
            maybe_queue_rescue_goal("post_nav_goal_reached")
            maybe_queue_world_target_rescue_goal(t, reason="post_nav_goal_reached")
            if USE_EXPLORATION and STATE.goal is None and STATE.map is not None:
                g = choose_frontier_goal(STATE.map, STATE.pose)
                if g is not None:
                    STATE.no_frontier_since_t = None
                    g = jitter_exploration_goal(g, STATE.map)
                    set_goal(g[0], g[1], goal_kind="explore")
                else:
                    if STATE.no_frontier_since_t is None:
                        STATE.no_frontier_since_t = float(t)
                    maybe_queue_exploration_fallback_goal(
                        t,
                        reason="post_nav_goal_reached",
                        force=True,
                    )
        last_nav_mode = nav_mode
    else:
        last_nav_mode = "MANUAL"
        clear_path(reason="planning_disabled")
        # Fallback to manual control if PLANNING is disabled
        # (Keeping existing logic for safety/reference)
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
        # Feed fallback odometry when the nav stack is disabled.
        POSE_EST.set_command(POSE_WHEEL_RADIUS_M * 0.5 * (fl + fr), POSE_WHEEL_RADIUS_M * (fr - fl) / POSE_AXLE_LENGTH_M)

    # ---- Post-nav rescue checks (detection already ran earlier this tick)
    maybe_queue_rescue_goal("new_camera_detection")
    maybe_queue_world_target_rescue_goal(t, reason="post_nav_world_target_task")
    maybe_rescue_active_target("post_detection_proximity")
    if USE_WORLD_TARGET_PROXIMITY_RESCUE:
        maybe_rescue_near_world_target("world_target_proximity_post_nav")
    maybe_write_victim_metrics(t, force=False)

    # ---- Mission completion
    if mission_done_reached(t):
        completion_status = _exploration_completion_status(t)
        if not STATE.mission_done:
            STATE.mission_done = True
            log_event("mission_done", {
                "found_count": len(STATE.victims),
                "rescued_count": len(STATE.rescued_victim_ids),
                "required": STATE.required_victims,
                "reason": STATE.mission_done_reason,
                "known_fraction": float(completion_status["known_fraction"]),
                "no_frontier_age_s": completion_status["no_frontier_age_s"],
                "stable_map_age_s": float(completion_status["stable_map_age_s"]),
                "keep_running": bool(KEEP_RUNNING_AFTER_MISSION_DONE),
            })
            print(
                f"Mission complete ({STATE.mission_done_reason}): "
                f"found={len(STATE.victims)}/{STATE.required_victims} "
                f"rescued={len(STATE.rescued_victim_ids)}/{STATE.required_victims} "
                f"known_fraction={completion_status['known_fraction']:.3f} "
                f"keep_running={KEEP_RUNNING_AFTER_MISSION_DONE}"
            )
        set_wheels(0.0, 0.0)
        clear_goal(reason="mission_done")
        clear_path(reason="mission_done")
        break

    # ---- Optional: periodic screenshot even without detection (proof pipeline)
    maybe_screenshot()

# ---- Shutdown / flush logs
set_wheels(0.0, 0.0)
clear_path(reason="shutdown")
maybe_write_victim_metrics(robot.getTime(), force=True)
log_event("run_end", {
    "reason": "mission_done" if STATE.mission_done else "simulation_stopped",
    "mission_done_reason": STATE.mission_done_reason,
    "found_count": len(STATE.victims),
    "rescued_count": len(STATE.rescued_victim_ids),
    "required": STATE.required_victims,
})
pose_csv.flush()
events.flush()
pose_csv.close()
events.close()
