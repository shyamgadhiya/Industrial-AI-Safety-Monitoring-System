"""
Typed intermediate objects for Vision Pipeline Orchestrator.
Adheres strictly to the architectural design principles:
- Detection(id, class_name, bbox, score)
- Track(track_id, detection_id, age, velocity)
- Pose(track_id, keypoints, scores)
- Mask(track_id, rle_or_polygon, score)
- Event(type, timestamp, track_ids, evidence_uri, model_versions)
"""

from __future__ import annotations
import time
import uuid
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box representation [x1, y1, x2, y2] with convenience spatial methods."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Ground contact point estimation for distance computation."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def to_xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_int_xyxy(self) -> Tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def iou(self, other: BoundingBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def euclidean_distance(self, other: BoundingBox) -> float:
        c1 = self.bottom_center
        c2 = other.bottom_center
        return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))


@dataclass
class Detection:
    """Detection intermediate object: emitted by RF-DETR / detector models."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    class_name: str = "object"
    bbox: BoundingBox = field(default_factory=lambda: BoundingBox(0, 0, 0, 0))
    score: float = 0.0
    model_name: str = "rf-detr"
    model_version: str = "v1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "bbox": self.bbox.to_xyxy(),
            "score": round(self.score, 4),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }


@dataclass
class Track:
    """Track intermediate object: emitted by ByteTrack multi-object tracker."""
    track_id: int
    detection_id: str
    class_name: str
    bbox: BoundingBox
    score: float = 0.0
    age: int = 1
    hits: int = 1
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) pixels per second or frame
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    state: str = "Confirmed"  # "Tentative", "Confirmed", "Lost"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "detection_id": self.detection_id,
            "class_name": self.class_name,
            "bbox": self.bbox.to_xyxy(),
            "score": round(self.score, 4),
            "age": self.age,
            "velocity": [round(v, 2) for v in self.velocity],
            "trajectory_len": len(self.trajectory),
            "state": self.state,
        }


@dataclass
class Keypoint:
    """2D pose keypoint."""
    x: float
    y: float
    score: float
    name: str = ""

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.score)


# Standard 17 COCO Keypoints used by ViTPose
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON_PAIRS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]


@dataclass
class Pose:
    """Pose intermediate object: emitted by ViTPose posture analyzer."""
    track_id: int
    keypoints: List[Keypoint] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    posture_label: str = "standing"  # "standing", "bending", "fallen", "sitting", "awkward_reach"
    spine_angle: float = 0.0
    fall_confidence: float = 0.0
    model_name: str = "vitpose"
    model_version: str = "small-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "posture_label": self.posture_label,
            "spine_angle": round(self.spine_angle, 2),
            "fall_confidence": round(self.fall_confidence, 3),
            "keypoints_count": len(self.keypoints),
            "model_version": f"{self.model_name}:{self.model_version}",
        }


@dataclass
class Mask:
    """Mask intermediate object: emitted by SAM (Segment Anything) model."""
    track_id: int
    polygon: List[List[float]] = field(default_factory=list)  # [[x, y], ...]
    rle: Optional[Dict[str, Any]] = None
    bbox: Optional[BoundingBox] = None
    score: float = 0.0
    area: float = 0.0
    model_name: str = "sam3"
    model_version: str = "sam3-promptable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "score": round(self.score, 4),
            "area": round(self.area, 2),
            "polygon_vertices": len(self.polygon),
            "model_version": f"{self.model_name}:{self.model_version}",
        }


@dataclass
class Event:
    """Event record: emitted by the Event/Rule Engine when conditions or anomalies trigger."""
    type: str  # e.g., "UNSAFE_PROXIMITY", "WORKER_FALLEN", "NO_HELMET_IN_HAZARD_ZONE"
    timestamp: float = field(default_factory=time.time)
    track_ids: List[int] = field(default_factory=list)
    severity: str = "WARNING"  # "INFO", "WARNING", "CRITICAL"
    description: str = ""
    evidence_uri: Optional[str] = None  # Path to generated MP4 evidence clip
    snapshot_uri: Optional[str] = None  # Path to JPEG snapshot
    model_versions: Dict[str, str] = field(default_factory=dict)
    confidence_values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "track_ids": self.track_ids,
            "severity": self.severity,
            "description": self.description,
            "evidence_uri": self.evidence_uri,
            "snapshot_uri": self.snapshot_uri,
            "model_versions": self.model_versions,
            "confidence_values": {k: round(v, 4) for k, v in self.confidence_values.items()},
            "metadata": self.metadata,
        }


@dataclass
class FramePacket:
    """
    Standard data container passing through the orchestration pipeline.
    Carries the raw frame and the accumulated typed domain objects.
    """
    frame_id: int
    timestamp: float
    image: Optional[np.ndarray] = None  # BGR numpy array
    annotated_image: Optional[np.ndarray] = None  # Processed BGR numpy array with visual overlays
    camera_id: str = "cam_01"
    fps: Optional[float] = None
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    poses: List[Pose] = field(default_factory=list)
    masks: List[Mask] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)

    def total_latency_ms(self) -> float:
        return sum(self.stage_latencies_ms.values())
