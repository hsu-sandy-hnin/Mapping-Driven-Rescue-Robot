"""Stuck detection and recovery state machine for navigation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import pi, hypot
from typing import Deque, Sequence

from .types import Pose2D, Twist


class RecoveryState(str, Enum):
    NORMAL = "NORMAL"
    TURN_IN_PLACE = "TURN_IN_PLACE"
    BACKUP = "BACKUP"
    FORWARD_NUDGE = "FORWARD_NUDGE"
    REPLAN = "REPLAN"


@dataclass(slots=True)
class PoseSample:
    x: float
    y: float
    yaw: float
    timestamp_s: float


@dataclass(slots=True)
class CmdSample:
    v: float
    omega: float
    timestamp_s: float


@dataclass(slots=True)
class StuckDiagnostics:
    is_stuck: bool
    reason: str
    displacement_m: float
    progress_metric_delta_m: float | None
    net_progress_m: float
    avg_cmd_v_mps: float
    max_cmd_v_mps: float
    yaw_change_abs_rad: float


def _cfg_get(config: object | None, name: str, default):
    if config is None:
        return default
    if isinstance(config, dict):
        if name in config:
            return config[name]
        for section in ("planner", "controller", "limits", "recovery"):
            if section in config and isinstance(config[section], dict) and name in config[section]:
                return config[section][name]
        return default
    if hasattr(config, name):
        return getattr(config, name)
    for section in ("planner", "controller", "limits", "recovery"):
        if hasattr(config, section):
            sec = getattr(config, section)
            if isinstance(sec, dict) and name in sec:
                return sec[name]
            if hasattr(sec, name):
                return getattr(sec, name)
    return default


def _wrap_to_pi(angle_rad: float) -> float:
    while angle_rad > pi:
        angle_rad -= 2.0 * pi
    while angle_rad < -pi:
        angle_rad += 2.0 * pi
    return angle_rad


def _max_omega(config: object | None) -> float:
    val = _cfg_get(config, "max_omega", None)
    if val is None:
        val = _cfg_get(config, "max_omega_radps", 2.2)
    return max(0.0, float(val))


def _max_v(config: object | None) -> float:
    val = _cfg_get(config, "max_v", None)
    if val is None:
        val = _cfg_get(config, "max_speed_mps", 0.45)
    return max(0.0, float(val))


def _stuck_window_s(config: object | None) -> float:
    return max(0.2, float(_cfg_get(config, "stuck_window_s", 2.0)))


def _min_progress_m(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "min_progress_m", 0.03)))


def _recovery_turn_time_s(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "recovery_turn_time_s", 1.0)))


def _recovery_backup_time_s(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "recovery_backup_time_s", 0.8)))


def _recovery_forward_time_s(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "recovery_forward_time_s", 0.25)))


def _cmd_v_threshold_mps(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "stuck_cmd_v_threshold_mps", 0.05)))


def _cmd_omega_threshold_radps(config: object | None) -> float:
    custom = _cfg_get(config, "stuck_cmd_omega_threshold_radps", None)
    if custom is not None:
        return max(0.0, float(custom))
    return max(0.25, 0.22 * _max_omega(config))


def _min_yaw_progress_rad(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "stuck_min_yaw_progress_rad", 0.12)))


def _oscillation_yaw_change_rad(config: object | None) -> float:
    return max(0.0, float(_cfg_get(config, "oscillation_yaw_change_rad", 1.0)))


def _turn_omega_radps(config: object | None) -> float:
    custom = _cfg_get(config, "recovery_turn_omega_radps", None)
    if custom is not None:
        return max(0.0, float(custom))
    return max(0.25, 0.55 * _max_omega(config))


def _backup_speed_mps(config: object | None) -> float:
    custom = _cfg_get(config, "recovery_backup_speed_mps", None)
    if custom is not None:
        return max(0.0, float(custom))
    return max(0.05, 0.35 * _max_v(config))


def _backup_turn_omega_radps(config: object | None) -> float:
    custom = _cfg_get(config, "recovery_backup_turn_omega_radps", None)
    if custom is not None:
        return max(0.0, float(custom))
    # Default to a gentle arc while backing up to escape tight frontal traps.
    return max(0.15, 0.35 * _turn_omega_radps(config))


def _forward_speed_mps(config: object | None) -> float:
    custom = _cfg_get(config, "recovery_forward_speed_mps", None)
    if custom is not None:
        return max(0.0, float(custom))
    return max(0.04, 0.22 * _max_v(config))


def _trim_history_by_time(samples: Deque, min_time_s: float) -> None:
    while len(samples) >= 2 and samples[0].timestamp_s < min_time_s:
        samples.popleft()


def _distance_xy(a: PoseSample, b: PoseSample) -> float:
    return hypot(b.x - a.x, b.y - a.y)


def _yaw_abs_change(samples: Sequence[PoseSample]) -> float:
    if len(samples) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(samples)):
        total += abs(_wrap_to_pi(samples[i].yaw - samples[i - 1].yaw))
    return total


def detect_stuck_from_histories(
    pose_history: Sequence[PoseSample],
    cmd_history: Sequence[CmdSample],
    now_s: float,
    config: object | None = None,
    progress_history: Sequence[tuple[float, float]] | None = None,
) -> StuckDiagnostics:
    """Run stuck detection over time-series history windows."""
    window_s = _stuck_window_s(config)
    start_t = now_s - window_s
    poses = [p for p in pose_history if p.timestamp_s >= start_t]
    cmds = [c for c in cmd_history if c.timestamp_s >= start_t]
    pose_span = poses[-1].timestamp_s - poses[0].timestamp_s if len(poses) >= 2 else 0.0
    cmd_span = cmds[-1].timestamp_s - cmds[0].timestamp_s if len(cmds) >= 2 else 0.0

    if len(poses) < 2 or len(cmds) < 1 or min(pose_span, cmd_span) < 0.75 * window_s:
        return StuckDiagnostics(
            is_stuck=False,
            reason="insufficient_history",
            displacement_m=0.0,
            progress_metric_delta_m=None,
            net_progress_m=0.0,
            avg_cmd_v_mps=0.0,
            max_cmd_v_mps=0.0,
            yaw_change_abs_rad=0.0,
        )

    displacement = _distance_xy(poses[0], poses[-1])
    progress_delta = None
    if progress_history is not None:
        pr = [item for item in progress_history if item[0] >= start_t]
        if len(pr) >= 2:
            progress_delta = float(pr[-1][1] - pr[0][1])

    # If a separate progress metric is available (wheel odometry or path-progress),
    # use the larger of pose displacement and metric delta as realized progress.
    net_progress = displacement
    if progress_delta is not None:
        net_progress = max(net_progress, max(0.0, progress_delta))

    v_values = [abs(c.v) for c in cmds]
    avg_cmd_v = float(sum(v_values) / max(1, len(v_values)))
    max_cmd_v = float(max(v_values)) if v_values else 0.0
    omega_values = [abs(c.omega) for c in cmds]
    avg_cmd_omega = float(sum(omega_values) / max(1, len(omega_values)))
    max_cmd_omega = float(max(omega_values)) if omega_values else 0.0
    yaw_change = _yaw_abs_change(poses)

    min_progress = _min_progress_m(config)
    v_threshold = _cmd_v_threshold_mps(config)
    omega_threshold = _cmd_omega_threshold_radps(config)

    forward_attempt = max_cmd_v > v_threshold
    turning_attempt = max_cmd_omega > omega_threshold
    rotation_blocked = turning_attempt and yaw_change < _min_yaw_progress_rad(config)
    low_translation = net_progress < min_progress

    stuck_progress = low_translation and (forward_attempt or rotation_blocked)
    stuck_oscillation = (
        low_translation
        and yaw_change >= _oscillation_yaw_change_rad(config)
        and ((avg_cmd_v > 0.5 * v_threshold) or (avg_cmd_omega > 0.5 * omega_threshold))
    )

    if rotation_blocked:
        reason = "turn_blocked_no_yaw_progress"
    elif stuck_progress and stuck_oscillation:
        reason = "low_progress+oscillation"
    elif stuck_progress:
        reason = "low_progress"
    elif stuck_oscillation:
        reason = "oscillation_no_translation"
    else:
        reason = "ok"

    return StuckDiagnostics(
        is_stuck=stuck_progress or stuck_oscillation,
        reason=reason,
        displacement_m=displacement,
        progress_metric_delta_m=progress_delta,
        net_progress_m=net_progress,
        avg_cmd_v_mps=avg_cmd_v,
        max_cmd_v_mps=max_cmd_v,
        yaw_change_abs_rad=yaw_change,
    )


@dataclass(slots=True)
class StuckRecoveryManager:
    """Detects stuck behavior and emits recovery override commands."""

    config: object | None = None
    replanner: object | None = None
    history_maxlen: int = 600

    state: RecoveryState = RecoveryState.NORMAL
    pose_history: Deque[PoseSample] = field(default_factory=deque, init=False)
    cmd_history: Deque[CmdSample] = field(default_factory=deque, init=False)
    progress_history: Deque[tuple[float, float]] = field(default_factory=deque, init=False)
    last_diagnostics: StuckDiagnostics | None = None

    _state_start_s: float | None = field(default=None, init=False, repr=False)
    _turn_sign: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.pose_history = deque(maxlen=self.history_maxlen)
        self.cmd_history = deque(maxlen=self.history_maxlen)
        self.progress_history = deque(maxlen=self.history_maxlen)

    def reset(self) -> None:
        self.state = RecoveryState.NORMAL
        self.pose_history.clear()
        self.cmd_history.clear()
        self.progress_history.clear()
        self.last_diagnostics = None
        self._state_start_s = None
        self._turn_sign = 1.0

    def _choose_turn_sign(self) -> float:
        if len(self.cmd_history) < 3:
            self._turn_sign *= -1.0
            return self._turn_sign
        mean_omega = sum(c.omega for c in self.cmd_history) / len(self.cmd_history)
        if abs(mean_omega) > 1e-6:
            return 1.0 if mean_omega >= 0.0 else -1.0
        self._turn_sign *= -1.0
        return self._turn_sign

    def _record_samples(
        self,
        pose: Pose2D,
        cmd: Twist,
        now_s: float,
        progress_metric_m: float | None,
    ) -> None:
        self.pose_history.append(PoseSample(pose.x, pose.y, pose.theta, now_s))
        self.cmd_history.append(CmdSample(cmd.v, cmd.omega, now_s))
        if progress_metric_m is not None:
            self.progress_history.append((now_s, float(progress_metric_m)))

        # Keep only ~2 windows to bound memory while maintaining enough context.
        keep_after = now_s - 2.0 * _stuck_window_s(self.config)
        _trim_history_by_time(self.pose_history, keep_after)
        _trim_history_by_time(self.cmd_history, keep_after)
        if self.progress_history:
            while len(self.progress_history) >= 2 and self.progress_history[0][0] < keep_after:
                self.progress_history.popleft()

    def _trigger_replan(self) -> bool:
        if self.replanner is None:
            return False
        force = getattr(self.replanner, "force_replan", None)
        if callable(force):
            force()
            return True
        return False

    def _enter_state(self, state: RecoveryState, now_s: float) -> None:
        self.state = state
        self._state_start_s = now_s

    def update(
        self,
        pose: Pose2D,
        cmd: Twist,
        now_s: float,
        progress_metric_m: float | None = None,
    ) -> tuple[Twist | None, str, bool]:
        """Update stuck recovery and return `(override_cmd_or_None, state, did_replan)`."""
        now = float(now_s)
        self._record_samples(pose, cmd, now, progress_metric_m)
        did_trigger_replan = False

        if self.state == RecoveryState.NORMAL:
            self.last_diagnostics = detect_stuck_from_histories(
                pose_history=list(self.pose_history),
                cmd_history=list(self.cmd_history),
                now_s=now,
                config=self.config,
                progress_history=list(self.progress_history) if self.progress_history else None,
            )
            if self.last_diagnostics.is_stuck:
                self._turn_sign = self._choose_turn_sign()
                self._enter_state(RecoveryState.TURN_IN_PLACE, now)

        if self.state == RecoveryState.TURN_IN_PLACE:
            elapsed = now - float(self._state_start_s or now)
            if elapsed < _recovery_turn_time_s(self.config):
                turn_cmd = Twist(v=0.0, omega=self._turn_sign * _turn_omega_radps(self.config))
                return turn_cmd, self.state.value, False
            self._enter_state(RecoveryState.BACKUP, now)

        if self.state == RecoveryState.BACKUP:
            elapsed = now - float(self._state_start_s or now)
            if elapsed < _recovery_backup_time_s(self.config):
                backup_cmd = Twist(
                    v=-_backup_speed_mps(self.config),
                    omega=self._turn_sign * _backup_turn_omega_radps(self.config),
                )
                return backup_cmd, self.state.value, False
            self._enter_state(RecoveryState.FORWARD_NUDGE, now)

        if self.state == RecoveryState.FORWARD_NUDGE:
            elapsed = now - float(self._state_start_s or now)
            if elapsed < _recovery_forward_time_s(self.config):
                forward_cmd = Twist(v=_forward_speed_mps(self.config), omega=0.0)
                return forward_cmd, self.state.value, False
            self._enter_state(RecoveryState.REPLAN, now)

        if self.state == RecoveryState.REPLAN:
            did_trigger_replan = self._trigger_replan()
            self._enter_state(RecoveryState.NORMAL, now)
            # Reset windows so stale pre-recovery samples do not retrigger immediately.
            self.pose_history.clear()
            self.cmd_history.clear()
            self.progress_history.clear()
            return None, RecoveryState.REPLAN.value, did_trigger_replan

        return None, self.state.value, False


def _demo() -> None:
    from .config import NavigationConfig

    @dataclass(slots=True)
    class _FakeReplanner:
        replan_calls: int = 0

        def force_replan(self) -> None:
            self.replan_calls += 1

    cfg = NavigationConfig()
    cfg.stuck_window_s = 1.5
    cfg.min_progress_m = 0.04
    cfg.recovery_turn_time_s = 0.8
    cfg.recovery_backup_time_s = 0.7
    cfg.recovery_backup_turn_omega_radps = 0.45
    cfg.recovery_forward_time_s = 0.3
    cfg.recovery_forward_speed_mps = 0.12
    cfg.max_v = 0.40
    cfg.max_omega = 1.60

    replanner = _FakeReplanner()
    recovery = StuckRecoveryManager(config=cfg, replanner=replanner)

    pose = Pose2D(0.0, 0.0, 0.0)
    local_cmd = Twist(0.25, 0.30)

    dt_s = 0.1
    sim_t = 0.0
    frozen = True
    states_seen: list[str] = []

    print("Stuck recovery simulation:")
    for step in range(120):
        override_cmd, state, did_replan = recovery.update(
            pose=pose,
            cmd=local_cmd,
            now_s=sim_t,
        )
        states_seen.append(state)

        # Robot remains stuck (no translation) until replanning is triggered once.
        if did_replan:
            frozen = False

        effective_cmd = override_cmd if override_cmd is not None else local_cmd
        if not frozen:
            # Simple kinematic forward motion after recovery/replan.
            pose = Pose2D(
                x=pose.x + effective_cmd.v * dt_s,
                y=pose.y,
                theta=_wrap_to_pi(pose.theta + effective_cmd.omega * dt_s),
            )
        else:
            # Stuck behavior: rotation can happen, translation is blocked.
            pose = Pose2D(
                x=pose.x,
                y=pose.y,
                theta=_wrap_to_pi(pose.theta + effective_cmd.omega * dt_s),
            )

        if step % 5 == 0 or did_replan:
            diag_reason = recovery.last_diagnostics.reason if recovery.last_diagnostics else "n/a"
            cmd_repr = (
                f"override(v={override_cmd.v:.2f},w={override_cmd.omega:.2f})"
                if override_cmd is not None
                else "override=None"
            )
            print(
                f"t={sim_t:4.1f}s state={state:>13} {cmd_repr} "
                f"replan={did_replan} reason={diag_reason}"
            )

        sim_t += dt_s
        if did_replan and pose.x > 0.30:
            break

    assert "TURN_IN_PLACE" in states_seen, "TURN_IN_PLACE not triggered"
    assert "BACKUP" in states_seen, "BACKUP not triggered"
    assert "FORWARD_NUDGE" in states_seen, "FORWARD_NUDGE not triggered"
    assert "REPLAN" in states_seen, "REPLAN not triggered"
    assert replanner.replan_calls >= 1, "Replanner was not triggered"
    print(f"Recovery sequence complete. replanner.force_replan calls={replanner.replan_calls}")


if __name__ == "__main__":
    _demo()
