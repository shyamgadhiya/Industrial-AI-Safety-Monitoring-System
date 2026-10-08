"""
ByteTrack Multi-Object Tracker Module using Supervision.
Integrates `supervision.ByteTrack` with persistent ID assignment, trajectory tracking,
and velocity estimation (vx, vy), returning typed intermediate Track records.
"""

from __future__ import annotations
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import supervision as sv

from src.core.types import Detection, Track, BoundingBox
from src.core.registry import ModelRegistry
from src.models.base import BaseVisionModel

logger = logging.getLogger(__name__)


@ModelRegistry.register("bytetrack")
@ModelRegistry.register("byte_track")
class ByteTrackTracker(BaseVisionModel):
    """
    ByteTrack Multi-Object Tracker.
    Wraps `supervision.ByteTrack` with trajectory history and spatial velocity tracking.
    Emits typed Track objects: Track(track_id, detection_id, age, velocity).
    """

    def __init__(
        self,
        name: str = "bytetrack",
        version: str = "v1.1",
        device: str = "cpu",
        options: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
    ):
        super().__init__(name=name, version=version, device=device, options=options or {})
        
        # Support both options dict and PipelineConfig object
        if config is not None:
            self.track_activation_threshold = float(getattr(config, "conf_threshold", 0.35))
            self.lost_track_buffer = int(getattr(config, "byte_track_buffer", 60))
            self.minimum_matching_threshold = float(getattr(config, "byte_match_threshold", 0.6))
            self.minimum_consecutive_frames = int(getattr(config, "byte_min_hits", 1))
            self.frame_rate = int(getattr(config, "frame_rate", 30))
        else:
            self.track_activation_threshold = float(self.options.get("track_thresh", 0.35))
            self.lost_track_buffer = int(self.options.get("track_buffer", 60))
            self.minimum_matching_threshold = float(self.options.get("match_thresh", 0.6))
            self.minimum_consecutive_frames = int(self.options.get("min_hits", 1))
            self.frame_rate = int(self.options.get("frame_rate", 30))

        self.tracker: Optional[sv.ByteTrack] = None
        self._history: Dict[int, Dict[str, Any]] = {}
        self._last_timestamp: float = time.time()
        self._id_map: Dict[int, int] = {}
        self._next_id: int = 1

    def initialize(self) -> None:
        """Initialize supervision ByteTrack instance and reset track IDs to start from 1."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            frame_rate=self.frame_rate,
        )
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        self._history.clear()
        self._id_map.clear()
        self._next_id = 1
        self._is_initialized = True
        logger.info(
            f"ByteTrack initialized (activation={self.track_activation_threshold}, "
            f"buffer={self.lost_track_buffer}, match_thresh={self.minimum_matching_threshold})"
        )

    def reset(self) -> None:
        """Reset internal tracker state and restart track IDs from 1."""
        if self.tracker and hasattr(self.tracker, "reset"):
            self.tracker.reset()
        self._history.clear()
        self._id_map.clear()
        self._next_id = 1

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Direct supervision update method.
        Accepts sv.Detections and returns tracked sv.Detections.
        """
        if not self._is_initialized or self.tracker is None:
            self.initialize()
        return self.tracker.update_with_detections(detections)

    def predict(
        self,
        detections: Union[List[Detection], sv.Detections],
        timestamp: Optional[float] = None,
    ) -> List[Track]:
        """
        Associate detections with existing tracks using Supervision ByteTrack.
        Returns a list of active confirmed Track objects with velocities and trajectories.
        """
        if not self._is_initialized:
            self.initialize()

        if timestamp is None:
            timestamp = time.time()

        dt = max(1e-3, timestamp - self._last_timestamp)
        self._last_timestamp = timestamp

        # 1. Convert input to sv.Detections if passed as List[Detection]
        class_name_map: Dict[int, str] = {}
        if isinstance(detections, list):
            if not detections:
                # Still update tracker with empty detections to increment age/lost buffer
                sv_dets = sv.Detections.empty()
            else:
                boxes = np.array([d.bbox.to_xyxy() for d in detections], dtype=np.float32)
                confs = np.array([d.score for d in detections], dtype=np.float32)
                # Map unique class names to integer IDs
                unique_classes = list({d.class_name for d in detections})
                cls_to_id = {cls: i for i, cls in enumerate(unique_classes)}
                class_name_map = {i: cls for cls, i in cls_to_id.items()}
                class_ids = np.array([cls_to_id[d.class_name] for d in detections], dtype=int)

                sv_dets = sv.Detections(
                    xyxy=boxes,
                    confidence=confs,
                    class_id=class_ids,
                    data={"class_name": np.array([d.class_name for d in detections])}
                )
        else:
            sv_dets = detections

        # 2. Run ByteTrack association
        tracked_sv = self.update(sv_dets)

        # 3. Process tracked results and update trajectories/velocities
        output_tracks: List[Track] = []
        if tracked_sv.tracker_id is not None and len(tracked_sv.tracker_id) > 0:
            for i, tid in enumerate(tracked_sv.tracker_id):
                raw_tid = int(tid)
                if raw_tid not in self._id_map:
                    self._id_map[raw_tid] = self._next_id
                    self._next_id += 1
                track_id = self._id_map[raw_tid]
                box = tracked_sv.xyxy[i]
                bbox = BoundingBox(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                score = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 1.0

                # Resolve class name
                if "class_name" in tracked_sv.data:
                    class_name = str(tracked_sv.data["class_name"][i])
                elif tracked_sv.class_id is not None and int(tracked_sv.class_id[i]) in class_name_map:
                    class_name = class_name_map[int(tracked_sv.class_id[i])]
                else:
                    class_name = "object"

                # Update trajectory and velocity history
                center = bbox.centroid
                if track_id not in self._history:
                    self._history[track_id] = {
                        "age": 1,
                        "hits": 1,
                        "trajectory": [center],
                        "velocity": (0.0, 0.0),
                        "last_center": center,
                    }
                else:
                    hist = self._history[track_id]
                    hist["age"] += 1
                    hist["hits"] += 1
                    prev_c = hist["last_center"]
                    vx = (center[0] - prev_c[0]) / dt
                    vy = (center[1] - prev_c[1]) / dt
                    # Smooth velocity
                    alpha = 0.6
                    sm_vx = alpha * vx + (1 - alpha) * hist["velocity"][0]
                    sm_vy = alpha * vy + (1 - alpha) * hist["velocity"][1]
                    hist["velocity"] = (sm_vx, sm_vy)
                    hist["last_center"] = center
                    hist["trajectory"].append(center)
                    if len(hist["trajectory"]) > 40:
                        hist["trajectory"].pop(0)

                hist = self._history[track_id]
                track = Track(
                    track_id=track_id,
                    detection_id=f"det_{track_id}",
                    class_name=class_name,
                    bbox=bbox,
                    score=round(score, 4),
                    age=hist["age"],
                    hits=hist["hits"],
                    velocity=(round(hist["velocity"][0], 2), round(hist["velocity"][1], 2)),
                    trajectory=list(hist["trajectory"]),
                    state="Confirmed",
                )
                output_tracks.append(track)

        return output_tracks
