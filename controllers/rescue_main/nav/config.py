"""Navigation configuration dataclasses.

These classes centralize tunable parameters for a Webots rescue robot
navigation stack. Values are intentionally conservative and can be adjusted
based on map scale, robot dynamics, and controller timestep.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GridConfig:
    """Occupancy grid and map inflation parameters."""

    # Meters per cell.
    resolution_m: float = 0.05
    # Obstacle inflation radius in meters (for robot footprint + safety margin).
    inflation_radius_m: float = 0.20
    # Cell value at/above this threshold is treated as occupied.
    obstacle_threshold: int = 50


@dataclass(slots=True)
class PlannerConfig:
    """Global and local path planning controls."""

    # Minimum seconds between planner runs.
    replanning_period_s: float = 0.50
    # Desired spacing between generated waypoints.
    waypoint_spacing_m: float = 0.25
    # Position tolerance for goal completion.
    goal_tolerance_m: float = 0.15
    # Heading tolerance at goal in radians.
    goal_heading_tolerance_rad: float = 0.20
    # Stop and flag "stuck" if movement under threshold for this duration.
    stuck_timeout_s: float = 2.00
    # Minimum displacement expected during stuck timeout window.
    stuck_min_progress_m: float = 0.03
    # Enable 8-connected A* motion (diagonal moves).
    allow_diagonal: bool = True
    # Allow traversing unknown cells (-1); otherwise unknown is treated as blocked.
    allow_unknown: bool = True
    # Extra traversal cost added when stepping into unknown cells.
    unknown_penalty: float = 2.5
    # Heuristic weighting for weighted A* (1.0 = standard A*).
    heuristic_weight: float = 1.0
    # Search radius for relocating blocked start/goal to nearest traversable cell.
    start_goal_search_radius_m: float = 0.75
    # Minimum robot displacement before deviation-based replanning is considered.
    min_pose_change_to_replan_m: float = 0.08
    # Maximum allowed distance from the current path before replanning.
    path_deviation_m: float = 0.40
    # Number of upcoming waypoints to inspect for newly blocked cells.
    blocked_lookahead_waypoints: int = 8
    # Minimum interval between map-change-triggered replans.
    map_change_replan_min_period_s: float = 1.20
    # Stuck detector window duration.
    stuck_window_s: float = 2.00
    # Minimum translation expected over stuck window.
    min_progress_m: float = 0.03
    # Duration of turn-in-place recovery phase.
    recovery_turn_time_s: float = 1.00
    # Duration of backup recovery phase.
    recovery_backup_time_s: float = 0.80
    # Duration of post-backup forward nudge phase.
    recovery_forward_time_s: float = 0.25
    # Optional fixed angular speed during turn-in-place recovery.
    recovery_turn_omega_radps: float | None = None
    # Optional fixed reverse speed during backup recovery.
    recovery_backup_speed_mps: float | None = None
    # Optional fixed angular speed during backup to reverse in an arc.
    recovery_backup_turn_omega_radps: float | None = None
    # Optional fixed linear speed during forward nudge recovery.
    recovery_forward_speed_mps: float | None = None
    # Yaw change threshold used by oscillation-based stuck detection.
    oscillation_yaw_change_rad: float = 1.0
    # Enable post-processing smoothing of sampled waypoints.
    smoothing_enabled: bool = True
    # Number of smoothing iterations.
    smoothing_passes: int = 3
    # Laplacian smoothing gain in [0, 1].
    smoothing_alpha: float = 0.35
    # Maximum shift allowed per smoothing edit.
    smoothing_max_shift_m: float = 0.12


@dataclass(slots=True)
class ControllerGains:
    """Unicycle pose-tracking gains (rho/alpha/beta form)."""

    k_rho: float = 1.20
    k_alpha: float = 2.50
    k_beta: float = -0.40
    # Heading-error proportional gain for waypoint tracking.
    heading_kp: float = 2.40
    # Lookahead distance for selecting target on current waypoint path.
    lookahead_m: float = 0.50
    # Radius near final goal where linear speed is ramped down.
    slow_down_radius_m: float = 0.70
    # Distance where reactive slowdown starts when obstacles are in front.
    safety_distance_m: float = 0.55
    # Side distance used to compute left/right repulsive pressure.
    side_clearance_m: float = 0.45
    # Gain applied to obstacle-driven angular bias.
    avoid_gain: float = 1.20
    # Emergency-stop distance threshold.
    stop_distance_m: float = 0.18


@dataclass(slots=True)
class MotionLimits:
    """Command limits for low-level velocity setpoints."""

    max_speed_mps: float = 0.45
    max_omega_radps: float = 2.20
    max_accel_mps2: float = 0.80
    max_alpha_radps2: float = 4.00


@dataclass(slots=True)
class NavigationConfig:
    """Top-level navigation config container."""

    grid: GridConfig = field(default_factory=GridConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    controller: ControllerGains = field(default_factory=ControllerGains)
    limits: MotionLimits = field(default_factory=MotionLimits)
    # Default CSV file name inside the `./logs` directory.
    log_file_name: str = "navigation.csv"

    @property
    def inflation_radius_m(self) -> float:
        """Compatibility alias to allow `config.inflation_radius_m` access."""
        return self.grid.inflation_radius_m

    @inflation_radius_m.setter
    def inflation_radius_m(self, value: float) -> None:
        self.grid.inflation_radius_m = float(value)

    @property
    def max_v(self) -> float:
        """Compatibility alias for local controller linear speed limit."""
        return self.limits.max_speed_mps

    @max_v.setter
    def max_v(self, value: float) -> None:
        self.limits.max_speed_mps = float(value)

    @property
    def max_omega(self) -> float:
        """Compatibility alias for local controller angular speed limit."""
        return self.limits.max_omega_radps

    @max_omega.setter
    def max_omega(self, value: float) -> None:
        self.limits.max_omega_radps = float(value)

    @property
    def lookahead_m(self) -> float:
        """Compatibility alias for local controller lookahead distance."""
        return self.controller.lookahead_m

    @lookahead_m.setter
    def lookahead_m(self, value: float) -> None:
        self.controller.lookahead_m = float(value)

    @property
    def slow_down_radius_m(self) -> float:
        """Compatibility alias for local controller slowdown radius."""
        return self.controller.slow_down_radius_m

    @slow_down_radius_m.setter
    def slow_down_radius_m(self, value: float) -> None:
        self.controller.slow_down_radius_m = float(value)

    @property
    def goal_tolerance_m(self) -> float:
        """Compatibility alias for goal distance tolerance."""
        return self.planner.goal_tolerance_m

    @goal_tolerance_m.setter
    def goal_tolerance_m(self, value: float) -> None:
        self.planner.goal_tolerance_m = float(value)

    @property
    def safety_distance_m(self) -> float:
        """Compatibility alias for obstacle-avoidance safety distance."""
        return self.controller.safety_distance_m

    @safety_distance_m.setter
    def safety_distance_m(self, value: float) -> None:
        self.controller.safety_distance_m = float(value)

    @property
    def side_clearance_m(self) -> float:
        """Compatibility alias for obstacle-avoidance side clearance."""
        return self.controller.side_clearance_m

    @side_clearance_m.setter
    def side_clearance_m(self, value: float) -> None:
        self.controller.side_clearance_m = float(value)

    @property
    def avoid_gain(self) -> float:
        """Compatibility alias for obstacle-avoidance angular gain."""
        return self.controller.avoid_gain

    @avoid_gain.setter
    def avoid_gain(self, value: float) -> None:
        self.controller.avoid_gain = float(value)

    @property
    def stop_distance_m(self) -> float:
        """Compatibility alias for obstacle-avoidance emergency threshold."""
        return self.controller.stop_distance_m

    @stop_distance_m.setter
    def stop_distance_m(self, value: float) -> None:
        self.controller.stop_distance_m = float(value)

    @property
    def stuck_window_s(self) -> float:
        """Compatibility alias for stuck detector window size."""
        return self.planner.stuck_window_s

    @stuck_window_s.setter
    def stuck_window_s(self, value: float) -> None:
        self.planner.stuck_window_s = float(value)

    @property
    def min_progress_m(self) -> float:
        """Compatibility alias for minimum progress threshold in stuck detection."""
        return self.planner.min_progress_m

    @min_progress_m.setter
    def min_progress_m(self, value: float) -> None:
        self.planner.min_progress_m = float(value)

    @property
    def recovery_turn_time_s(self) -> float:
        """Compatibility alias for turn-in-place recovery duration."""
        return self.planner.recovery_turn_time_s

    @recovery_turn_time_s.setter
    def recovery_turn_time_s(self, value: float) -> None:
        self.planner.recovery_turn_time_s = float(value)

    @property
    def recovery_backup_time_s(self) -> float:
        """Compatibility alias for backup recovery duration."""
        return self.planner.recovery_backup_time_s

    @recovery_backup_time_s.setter
    def recovery_backup_time_s(self, value: float) -> None:
        self.planner.recovery_backup_time_s = float(value)

    @property
    def recovery_forward_time_s(self) -> float:
        """Compatibility alias for forward-nudge recovery duration."""
        return self.planner.recovery_forward_time_s

    @recovery_forward_time_s.setter
    def recovery_forward_time_s(self, value: float) -> None:
        self.planner.recovery_forward_time_s = float(value)

    @property
    def recovery_turn_omega_radps(self) -> float | None:
        """Compatibility alias for fixed turn-in-place recovery omega."""
        return self.planner.recovery_turn_omega_radps

    @recovery_turn_omega_radps.setter
    def recovery_turn_omega_radps(self, value: float | None) -> None:
        self.planner.recovery_turn_omega_radps = None if value is None else float(value)

    @property
    def recovery_backup_speed_mps(self) -> float | None:
        """Compatibility alias for fixed backup recovery speed."""
        return self.planner.recovery_backup_speed_mps

    @recovery_backup_speed_mps.setter
    def recovery_backup_speed_mps(self, value: float | None) -> None:
        self.planner.recovery_backup_speed_mps = None if value is None else float(value)

    @property
    def recovery_backup_turn_omega_radps(self) -> float | None:
        """Compatibility alias for fixed backup-turn recovery omega."""
        return self.planner.recovery_backup_turn_omega_radps

    @recovery_backup_turn_omega_radps.setter
    def recovery_backup_turn_omega_radps(self, value: float | None) -> None:
        self.planner.recovery_backup_turn_omega_radps = None if value is None else float(value)

    @property
    def recovery_forward_speed_mps(self) -> float | None:
        """Compatibility alias for fixed forward-nudge recovery speed."""
        return self.planner.recovery_forward_speed_mps

    @recovery_forward_speed_mps.setter
    def recovery_forward_speed_mps(self, value: float | None) -> None:
        self.planner.recovery_forward_speed_mps = None if value is None else float(value)

    @property
    def oscillation_yaw_change_rad(self) -> float:
        """Compatibility alias for oscillation-based stuck threshold."""
        return self.planner.oscillation_yaw_change_rad

    @oscillation_yaw_change_rad.setter
    def oscillation_yaw_change_rad(self, value: float) -> None:
        self.planner.oscillation_yaw_change_rad = float(value)
