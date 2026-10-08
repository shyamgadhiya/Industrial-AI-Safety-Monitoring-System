"""
Safety rules library for the Event/Rule Engine.
Evaluates domain conditions (Proximity, Falls/Ergonomics) over FramePackets.
"""

from __future__ import annotations
import time
import math
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.core.types import FramePacket, Event, Track, Pose

logger = logging.getLogger(__name__)


class BaseSafetyRule:
    """Base class for all safety evaluation rules."""

    def __init__(self, rule_id: str, name: str, severity: str = "WARNING", cooldown_seconds: float = 5.0):
        self.rule_id = rule_id
        self.name = name
        self.severity = severity
        self.cooldown_seconds = cooldown_seconds
        self.last_fired_time: float = 0.0

    def is_cooling_down(self, current_time: float) -> bool:
        return (current_time - self.last_fired_time) < self.cooldown_seconds

    def evaluate(self, packet: FramePacket) -> List[Event]:
        raise NotImplementedError


class WorkerForkliftProximityRule(BaseSafetyRule):
    """
    Triggers when a worker is within a hazardous distance of a forklift,
    especially if the forklift is moving or approaching the worker.
    """

    def __init__(
        self,
        rule_id: str = "RULE_001",
        name: str = "Worker-Forklift Proximity Violation",
        severity: str = "CRITICAL",
        min_distance_pixels: float = 180.0,
        cooldown_seconds: float = 5.0,
    ):
        super().__init__(rule_id, name, severity, cooldown_seconds)
        self.min_distance_pixels = min_distance_pixels

    def evaluate(self, packet: FramePacket) -> List[Event]:
        events = []
        if self.is_cooling_down(packet.timestamp):
            return events

        person_tracks = [t for t in packet.tracks if t.class_name in ("person", "worker")]
        forklift_tracks = [t for t in packet.tracks if t.class_name == "forklift"]

        for person in person_tracks:
            for forklift in forklift_tracks:
                dist = person.bbox.euclidean_distance(forklift.bbox)
                if dist < self.min_distance_pixels:
                    # Calculate proximity urgency
                    urgency = 1.0 - (dist / self.min_distance_pixels)
                    self.last_fired_time = packet.timestamp

                    desc = (
                        f"Unsafe proximity alert: Worker #{person.track_id} is only {dist:.1f}px "
                        f"from Forklift #{forklift.track_id} (threshold={self.min_distance_pixels}px)."
                    )

                    events.append(
                        Event(
                            type="UNSAFE_WORKER_FORKLIFT_PROXIMITY",
                            timestamp=packet.timestamp,
                            track_ids=[person.track_id, forklift.track_id],
                            severity=self.severity,
                            description=desc,
                            confidence_values={"distance_px": round(dist, 1), "urgency": round(urgency, 3)},
                            metadata={
                                "rule_id": self.rule_id,
                                "rule_name": self.name,
                                "person_velocity": person.velocity,
                                "forklift_velocity": forklift.velocity,
                            },
                        )
                    )
        return events


class WorkerFallErgonomicsRule(BaseSafetyRule):
    """
    Triggers when ViTPose posture analysis detects a fallen or collapsed worker.
    """

    def __init__(
        self,
        rule_id: str = "RULE_002",
        name: str = "Worker Fall / Collapse Detected",
        severity: str = "CRITICAL",
        fall_confidence_threshold: float = 0.65,
        cooldown_seconds: float = 8.0,
    ):
        super().__init__(rule_id, name, severity, cooldown_seconds)
        self.fall_confidence_threshold = fall_confidence_threshold

    def evaluate(self, packet: FramePacket) -> List[Event]:
        events = []
        if self.is_cooling_down(packet.timestamp):
            return events

        for pose in packet.poses:
            if pose.posture_label == "fallen" and pose.fall_confidence >= self.fall_confidence_threshold:
                self.last_fired_time = packet.timestamp
                desc = (
                    f"Man-down / Fall alert for Worker #{pose.track_id}: "
                    f"Spine angle={pose.spine_angle:.1f} deg, Fall Confidence={pose.fall_confidence:.2f}."
                )
                events.append(
                    Event(
                        type="WORKER_FALL_DETECTED",
                        timestamp=packet.timestamp,
                        track_ids=[pose.track_id],
                        severity=self.severity,
                        description=desc,
                        confidence_values={"fall_confidence": pose.fall_confidence, "spine_angle": pose.spine_angle},
                        metadata={
                            "rule_id": self.rule_id,
                            "rule_name": self.name,
                            "posture_label": pose.posture_label,
                        },
                    )
                )
        return events
