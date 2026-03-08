"""Victim event logging and lightweight evaluation metrics utilities.

This module is runtime-agnostic. The Webots controller uses it directly, and a
ROS2 node can optionally emit the same JSON schema if desired.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional


def _finite_or_none(value: float):
    """Return a JSON-safe finite float or None."""
    value = float(value)
    return value if math.isfinite(value) else None


class VictimEventWriter:
    """Append victim detection events to a JSONL file using a stable schema."""

    def __init__(self, out_path: str) -> None:
        self.out_path = str(out_path)
        parent = os.path.dirname(self.out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def write_event(
        self,
        *,
        timestamp: float,
        robot_pose: Dict[str, float],
        victim_type: str,
        victim_class: str,
        confidence: float,
        source: str,
        image_path: Optional[str] = None,
        world_coords: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Write one JSONL event and return the record for debugging/tests."""
        rec: Dict[str, object] = {
            "timestamp": float(timestamp),
            "robot_pose": {
                "x": _finite_or_none(robot_pose.get("x", float("nan"))),
                "y": _finite_or_none(robot_pose.get("y", float("nan"))),
                "yaw": _finite_or_none(robot_pose.get("yaw", float("nan"))),
            },
            "victim_type": str(victim_type),
            "victim_class": str(victim_class),
            "confidence": float(confidence),
            "source": str(source),
        }
        if image_path:
            rec["image_path"] = str(image_path)
        if world_coords is not None:
            rec["world_coords"] = {
                "x": _finite_or_none(world_coords.get("x", float("nan"))),
                "y": _finite_or_none(world_coords.get("y", float("nan"))),
            }
        if extra:
            rec.update(extra)

        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec


class VictimMetricsTracker:
    """Track simple non-ROS evaluation metrics for victim detections."""

    def __init__(self, cluster_radius_m: float = 0.75, duplicate_radius_m: float = 0.35) -> None:
        self.cluster_radius_m = float(cluster_radius_m)
        self.duplicate_radius_m = float(duplicate_radius_m)
        self._detections: List[Dict[str, float]] = []
        self._duplicate_attempts = 0
        self._candidate_count = 0

    @staticmethod
    def _dist(a: Dict[str, float], b: Dict[str, float]) -> float:
        return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

    def record_detection(self, x: float, y: float, timestamp: float) -> None:
        """Record an accepted detection."""
        self._candidate_count += 1
        self._detections.append({"x": float(x), "y": float(y), "t": float(timestamp)})

    def record_duplicate_attempt(self, x: float, y: float, timestamp: float) -> None:
        """Record a rejected near-duplicate detection candidate."""
        self._candidate_count += 1
        self._duplicate_attempts += 1
        # Keep duplicate coordinates for optional auditing without inflating unique count.
        self._detections.append({"x": float(x), "y": float(y), "t": float(timestamp), "duplicate": 1.0})

    def _cluster_unique(self) -> List[Dict[str, float]]:
        """Greedy spatial clustering for approximate unique victim locations."""
        accepted = [d for d in self._detections if not d.get("duplicate")]
        clusters: List[Dict[str, float]] = []
        for d in accepted:
            matched = None
            for c in clusters:
                if self._dist(d, c) <= self.cluster_radius_m:
                    matched = c
                    break
            if matched is None:
                clusters.append({"x": d["x"], "y": d["y"], "count": 1.0})
            else:
                n = matched["count"] + 1.0
                matched["x"] = (matched["x"] * matched["count"] + d["x"]) / n
                matched["y"] = (matched["y"] * matched["count"] + d["y"]) / n
                matched["count"] = n
        return clusters

    def build_summary(self) -> Dict[str, object]:
        """Return a JSON-serializable metrics summary."""
        accepted = [d for d in self._detections if not d.get("duplicate")]
        clusters = self._cluster_unique()
        denom = max(1, self._candidate_count)
        return {
            "total_detection_candidates": int(self._candidate_count),
            "accepted_detections": int(len(accepted)),
            "duplicate_detection_candidates": int(self._duplicate_attempts),
            "false_duplicate_rate": float(self._duplicate_attempts) / float(denom),
            "unique_victim_locations": int(len(clusters)),
            "unique_clusters": [
                {"x": round(c["x"], 4), "y": round(c["y"], 4), "count": int(c["count"])}
                for c in clusters
            ],
        }

    def write_summary(self, out_path: str, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """Write the current metrics summary to JSON and return it."""
        summary = self.build_summary()
        if extra:
            summary.update(extra)
        parent = os.path.dirname(str(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
