import math
import os

import cv2
import numpy as np


class LiveMapVisualizer:
    def __init__(
        self,
        *,
        map_size,
        map_resolution,
        pose_trail_min_step_m=0.06,
        pose_trail_max_points=5000,
        enabled=True,
        window_name="Rescue Live Map",
        update_hz=8.0,
        scale=1.0,
        export_period_s=1.0,
        latest_file="map_live_latest.png",
    ):
        self.map_size = int(map_size)
        self.map_resolution = float(map_resolution)
        self.pose_trail_min_step_m = float(pose_trail_min_step_m)
        self.pose_trail_max_points = int(pose_trail_max_points)
        self.enabled = bool(enabled)
        self.window_name = str(window_name)
        self.update_hz = float(update_hz)
        self.scale = float(scale)
        self.export_period_s = float(export_period_s)
        self.latest_file = str(latest_file)

        self.last_live_viz_t = -1e9
        self.last_live_export_t = -1e9
        self.window_ready = False

    def _world_to_grid(self, x, y):
        gx = int(float(x) / self.map_resolution + self.map_size // 2)
        gy = int(float(y) / self.map_resolution + self.map_size // 2)
        return gx, gy

    def _grid_to_image_point(self, gx, gy):
        if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
            # Occupancy map is indexed [x, y]; OpenCV points are (col=x, row=y).
            return int(gy), int(gx)
        return None

    def _world_to_image_point(self, wx, wy):
        gx, gy = self._world_to_grid(float(wx), float(wy))
        return self._grid_to_image_point(gx, gy)

    def maybe_record_pose_trail(self, *, pose, pose_trail, pose_valid_fn, dist_fn):
        """Append the robot pose to trail if movement exceeds a small threshold."""
        if not pose_valid_fn(pose):
            return
        px = float(pose["x"])
        py = float(pose["y"])
        if pose_trail:
            lx, ly = pose_trail[-1]
            if dist_fn(px, py, lx, ly) < self.pose_trail_min_step_m:
                return
        pose_trail.append((px, py))
        overflow = len(pose_trail) - self.pose_trail_max_points
        if overflow > 0:
            del pose_trail[:overflow]

    def _draw_world_polyline(self, img, world_points, color, thickness=1):
        pts = []
        for wx, wy in world_points:
            p = self._world_to_image_point(wx, wy)
            if p is not None:
                pts.append(p)
        if len(pts) >= 2:
            cv2.polylines(img, [np.asarray(pts, dtype=np.int32)], False, color, int(thickness), lineType=cv2.LINE_AA)

    def _draw_world_cross(self, img, wx, wy, color, size=5, thickness=1):
        p = self._world_to_image_point(wx, wy)
        if p is None:
            return
        cx, cy = p
        s = int(size)
        cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), color, int(thickness), lineType=cv2.LINE_AA)
        cv2.line(img, (cx - s, cy + s), (cx + s, cy - s), color, int(thickness), lineType=cv2.LINE_AA)

    def _draw_world_square(self, img, wx, wy, color, half_size=4, thickness=1):
        p = self._world_to_image_point(wx, wy)
        if p is None:
            return
        cx, cy = p
        hs = int(half_size)
        cv2.rectangle(img, (cx - hs, cy - hs), (cx + hs, cy + hs), color, int(thickness), lineType=cv2.LINE_AA)

    def render(
        self,
        map_state,
        *,
        robot_time_s,
        map_version,
        pose,
        pose_trail,
        planned_path,
        world_targets,
        rescued_target_positions,
        victims,
        camera_sightings,
        rescued_victim_ids,
        rescued_sites,
        goal,
        required_victims,
    ):
        img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        img[map_state == -1] = (128, 128, 128)  # unknown = gray
        img[map_state == 0] = (255, 255, 255)   # free = white
        img[map_state == 100] = (0, 0, 0)       # occupied = black

        self._draw_world_polyline(img, pose_trail, color=(200, 0, 200), thickness=1)  # magenta
        if planned_path:
            self._draw_world_polyline(img, planned_path, color=(0, 180, 255), thickness=1)  # orange

        for _def_name, pos in world_targets:
            if pos is None or len(pos) < 2:
                continue
            self._draw_world_square(img, pos[0], pos[1], color=(0, 220, 255), half_size=4, thickness=1)

        for _def_name, (rx, ry) in rescued_target_positions.items():
            self._draw_world_square(img, rx, ry, color=(0, 170, 0), half_size=5, thickness=2)

        for victim in victims:
            try:
                vx = float(victim["x"])
                vy = float(victim["y"])
                victim_id = int(victim.get("id", -1))
            except Exception:
                continue
            rescued = bool(victim.get("rescued", False)) or (victim_id in rescued_victim_ids)
            p = self._world_to_image_point(vx, vy)
            if p is None:
                continue
            if rescued:
                self._draw_world_cross(img, vx, vy, color=(0, 200, 0), size=5, thickness=2)
            else:
                cv2.circle(img, p, 3, (0, 80, 255), -1, lineType=cv2.LINE_AA)

        for rx, ry in rescued_sites:
            self._draw_world_cross(img, rx, ry, color=(0, 220, 0), size=4, thickness=1)

        if goal is not None:
            self._draw_world_square(img, goal[0], goal[1], color=(255, 120, 0), half_size=6, thickness=2)

        pose_x = float(pose.get("x", float("nan")))
        pose_y = float(pose.get("y", float("nan")))
        pose_yaw = float(pose.get("yaw", float("nan")))
        if math.isfinite(pose_x) and math.isfinite(pose_y) and math.isfinite(pose_yaw):
            p0 = self._world_to_image_point(pose_x, pose_y)
            if p0 is not None:
                cv2.circle(img, p0, 4, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                p1 = self._world_to_image_point(
                    pose_x + 0.8 * math.cos(pose_yaw),
                    pose_y + 0.8 * math.sin(pose_yaw),
                )
                if p1 is not None:
                    cv2.arrowedLine(img, p0, p1, (255, 0, 0), 2, cv2.LINE_AA, 0, 0.35)

        rescued_count = len(rescued_victim_ids)
        planned_count = 0 if planned_path is None else len(planned_path)
        info_lines = [
            f"time={float(robot_time_s):.1f}s map_v={int(map_version)}",
            f"path_pts={len(pose_trail)} planned={planned_count}",
            f"victims={len(victims)} rescued={rescued_count}/{int(required_victims)}",
            "legend: black=obstacle yellow=obj orange=detected green=rescued magenta=trail blue=robot",
        ]
        box_h = 18 * len(info_lines) + 8
        cv2.rectangle(img, (5, 5), (520, 5 + box_h), (245, 245, 245), -1)
        cv2.rectangle(img, (5, 5), (520, 5 + box_h), (80, 80, 80), 1)
        for i, line in enumerate(info_lines):
            cv2.putText(
                img,
                line,
                (10, 22 + 18 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )
        return img

    def save_map_image(self, map_state, filename, *, render_kwargs):
        img = self.render(map_state, **render_kwargs)
        cv2.imwrite(filename, img)

    def maybe_update(self, *, now_t, map_state, run_dir, render_kwargs, is_enabled, log_event_cb):
        if (not self.enabled) or (not bool(is_enabled)) or map_state is None:
            return bool(is_enabled)
        if (float(now_t) - self.last_live_viz_t) < (1.0 / max(1e-6, self.update_hz)):
            return bool(is_enabled)
        self.last_live_viz_t = float(now_t)
        frame = self.render(map_state, **render_kwargs)

        if self.export_period_s > 0.0 and (float(now_t) - self.last_live_export_t) >= self.export_period_s:
            self.last_live_export_t = float(now_t)
            try:
                cv2.imwrite(os.path.join(run_dir, self.latest_file), frame)
            except Exception:
                pass

        try:
            if not self.window_ready:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.window_ready = True
            display = frame
            if abs(self.scale - 1.0) > 1e-6:
                display = cv2.resize(
                    frame,
                    None,
                    fx=self.scale,
                    fy=self.scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow(self.window_name, display)
            cv2.waitKey(1)
            return True
        except Exception as exc:
            if self.window_ready:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception:
                    pass
                self.window_ready = False
            if log_event_cb is not None:
                try:
                    log_event_cb("live_viz_disabled", {"error": str(exc)})
                except Exception:
                    pass
            print(f"LIVE_VIZ_DISABLED error={exc}")
            return False

    def close(self):
        if self.window_ready:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self.window_ready = False
