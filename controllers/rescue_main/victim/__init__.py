"""Victim logging helpers shared by the Webots controller and ROS2 adapters."""

from .event_logger import VictimEventWriter, VictimMetricsTracker

__all__ = ["VictimEventWriter", "VictimMetricsTracker"]
