"""
Event and Safety Rule Engine.
Evaluates rules over streaming FramePackets, attaches model provenance,
triggers evidence clip recording, and publishes alerts to listeners.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Callable
from src.core.types import FramePacket, Event
from src.core.buffer import CircularRingBuffer
from src.rules.safety_rules import (
    BaseSafetyRule,
    WorkerForkliftProximityRule,
    WorkerFallErgonomicsRule,
)
from src.rules.evidence import EvidencePersistenceManager

logger = logging.getLogger(__name__)


class EventRuleEngine:
    """
    Central rule evaluator.
    Dispatches frame telemetry to registered safety rules and delegates evidence generation.
    """

    def __init__(
        self,
        evidence_manager: Optional[EvidencePersistenceManager] = None,
        model_versions: Optional[Dict[str, str]] = None,
        rules_path: str = "configs/rules.yaml",
    ):
        self.evidence_manager = evidence_manager or EvidencePersistenceManager()
        self.model_versions = model_versions or {
            "detector": "rf-detr-small:v1.2",
            "tracker": "bytetrack:v1.1",
            "pose": "vitpose-small:v1.0",
            "clip": "clip-vit-b32:v1.0",
        }
        self.rules: List[BaseSafetyRule] = []
        self.event_history: List[Event] = []
        self._listeners: List[Callable[[Event], None]] = []

        # Load rules from rules_path if available
        loaded_from_file = False
        import os
        if os.path.exists(rules_path):
            try:
                import yaml
                with open(rules_path, "r", encoding="utf-8") as f:
                    r_data = yaml.safe_load(f) or {}
                for r in r_data.get("rules", []):
                    if not r.get("enabled", True):
                        continue
                    r_type = r.get("type")
                    params = r.get("parameters", {})
                    if r_type == "proximity":
                        self.add_rule(WorkerForkliftProximityRule(
                            rule_id=r.get("rule_id", "RULE_001"),
                            name=r.get("name", "Worker-Forklift Proximity Violation"),
                            severity=r.get("severity", "CRITICAL"),
                            min_distance_pixels=float(params.get("min_distance_pixels", 160.0)),
                            cooldown_seconds=float(params.get("cooldown_seconds", 5.0)),
                        ))
                        loaded_from_file = True
                    elif r_type in ("ergonomics", "fall"):
                        self.add_rule(WorkerFallErgonomicsRule(
                            rule_id=r.get("rule_id", "RULE_002"),
                            name=r.get("name", "Worker Fall / Collapse Detected"),
                            severity=r.get("severity", "CRITICAL"),
                            fall_confidence_threshold=float(params.get("fall_confidence_threshold", 0.65)),
                            cooldown_seconds=float(params.get("cooldown_seconds", 10.0)),
                        ))
                        loaded_from_file = True
                if loaded_from_file:
                    logger.info(f"Loaded {len(self.rules)} safety rules from '{rules_path}'.")
            except Exception as e:
                logger.warning(f"Error loading rules from {rules_path}: {e}")

        if not loaded_from_file:
            # Fallback to default suite of safety rules
            self.add_rule(WorkerForkliftProximityRule())
            self.add_rule(WorkerFallErgonomicsRule())

    def add_rule(self, rule: BaseSafetyRule) -> None:
        self.rules.append(rule)

    def add_listener(self, callback: Callable[[Event], None]) -> None:
        self._listeners.append(callback)

    def process_packet(
        self,
        packet: FramePacket,
        ring_buffer: CircularRingBuffer,
        fps: Optional[float] = None,
    ) -> List[Event]:
        """
        Evaluate all active rules on the packet.
        Fires alerts and records evidence clips when rules trigger.
        """
        fired_events: List[Event] = []

        # Determine original video FPS: explicit argument > packet.fps > evidence_manager.fps
        video_fps = fps or getattr(packet, "fps", None) or self.evidence_manager.fps

        for rule in self.rules:
            try:
                events = rule.evaluate(packet)
                for evt in events:
                    # Inject active model versions for provenance
                    evt.model_versions = dict(self.model_versions)

                    # Persist evidence MP4 clip and JSON snapshot at original video FPS
                    self.evidence_manager.persist_incident_evidence(
                        event=evt,
                        packet=packet,
                        ring_buffer=ring_buffer,
                        fps=video_fps,
                    )

                    fired_events.append(evt)
                    self.event_history.append(evt)

                    # Notify subscribers (WebSocket, Dashboard, Logging)
                    for listener in self._listeners:
                        try:
                            listener(evt)
                        except Exception as cb_err:
                            logger.error(f"Error in event listener: {cb_err}")

            except Exception as e:
                logger.error(f"Error evaluating rule '{rule.name}': {e}")

        # Attach events to frame packet
        packet.events.extend(fired_events)
        return fired_events
