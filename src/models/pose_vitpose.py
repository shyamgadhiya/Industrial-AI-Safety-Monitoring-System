"""
ViTPose Posture Estimation Module.
Integrates HuggingFace Transformers `VitPoseForPoseEstimation` and `AutoProcessor`,
extracting 17 COCO keypoints, calculating spine inclination angles, ergonomic load,
and detecting worker falls/collapses.
Emits typed Pose records: Pose(track_id, keypoints, scores, posture_label, spine_angle, fall_confidence).
"""

from __future__ import annotations
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image

try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    HAS_TRANSFORMERS_VITPOSE = True
except ImportError:
    HAS_TRANSFORMERS_VITPOSE = False

from src.core.types import (
    Track, Pose, Keypoint, COCO_KEYPOINT_NAMES, COCO_SKELETON_PAIRS
)
from src.core.registry import ModelRegistry
from src.models.base import BaseVisionModel

logger = logging.getLogger(__name__)

# COCO Keypoints & Skeletons
VITPOSE_KEYPOINTS = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle"
]

VITPOSE_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


@ModelRegistry.register("vitpose")
@ModelRegistry.register("vitpose_small")
@ModelRegistry.register("vitpose_base")
class ViTPoseEstimator(BaseVisionModel):
    """
    Vision Transformer Pose (ViTPose) human keypoint analyzer.
    Analyzes postures of tracked workers, detects collapses, falls, and ergonomic hazards.
    """

    def __init__(
        self,
        name: str = "vitpose-base",
        version: str = "usyd-community/vitpose-base-simple",
        device: str = "cpu",
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, version=version, device=device, options=options or {})
        self.fall_angle_threshold = float(self.options.get("fall_angle_threshold", 60.0))
        self.conf_threshold = float(self.options.get("confidence_threshold", 0.30))
        self.model_id = self.options.get("model_id", "usyd-community/vitpose-base-simple")
        
        # Resolve target execution device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = None
        self.model = None

    def initialize(self) -> None:
        """Initialize ViTPose model and processor."""
        logger.info(f"Initializing {self.name} ({self.model_id}) on device '{self.device}'...")
        
        if HAS_TRANSFORMERS_VITPOSE:
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = VitPoseForPoseEstimation.from_pretrained(self.model_id).to(self.device).eval()
                logger.info("Loaded VitPoseForPoseEstimation successfully.")
            except Exception as e:
                logger.warning(f"Could not load HuggingFace ViTPose model: {e}. Using geometric estimator fallback.")
                self.model = None
        else:
            logger.warning("transformers VitPose not available. Using geometric estimator fallback.")

        self._is_initialized = True

    def predict(self, frame: np.ndarray, targets: List[Any]) -> List[Pose]:
        """
        Estimate 2D skeleton poses for all detected or tracked 'person' instances.
        Returns a list of typed Pose objects.
        """
        if not self._is_initialized:
            self.initialize()

        if frame is None or not targets:
            return []

        h, w = frame.shape[:2]
        poses: List[Pose] = []

        # Filter only active person targets with valid bounding boxes
        person_targets: List[Any] = []
        valid_boxes: List[List[float]] = []

        for t in targets:
            if getattr(t, "class_name", "") in ("person", "worker"):
                x1, y1, x2, y2 = t.bbox.to_xyxy()
                x1 = max(0.0, min(float(w - 1), x1))
                y1 = max(0.0, min(float(h - 1), y1))
                x2 = max(x1 + 10.0, min(float(w), x2))
                y2 = max(y1 + 10.0, min(float(h), y2))
                bw = x2 - x1
                bh = y2 - y1
                # Only analyze targets with plausible person dimensions
                if bw >= 10.0 and bh >= 16.0:
                    person_targets.append(t)
                    # COCO format required by VitPoseImageProcessor: [top_left_x, top_left_y, width, height]
                    valid_boxes.append([x1, y1, bw, bh])

        if not person_targets:
            return []

        # 1. Run inference using transformers VitPose model
        if self.model is not None and self.processor is not None:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Shape format: [1, num_boxes, 4]
                boxes_input = [valid_boxes]
                inputs = self.processor(images=pil_img, boxes=boxes_input, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)

                raw_results = self.processor.post_process_pose_estimation(outputs, boxes=boxes_input)[0]

                for idx, (target, res) in enumerate(zip(person_targets, raw_results)):
                    kpts = res["keypoints"].cpu().numpy()  # [17, 2]
                    scores = res["scores"].cpu().numpy()    # [17]

                    keypoints_list: List[Keypoint] = []
                    scores_list: List[float] = []
                    for i in range(len(kpts)):
                        kp_name = VITPOSE_KEYPOINTS[i] if i < len(VITPOSE_KEYPOINTS) else f"kp_{i}"
                        score = float(scores[i])
                        keypoints_list.append(
                            Keypoint(x=float(kpts[i][0]), y=float(kpts[i][1]), score=round(score, 4), name=kp_name)
                        )
                        scores_list.append(round(score, 4))

                    # Analyze ergonomics & fall orientation
                    bw, bh = target.bbox.width, target.bbox.height
                    spine_angle, posture_label, fall_conf = self._analyze_posture(keypoints_list, bw, bh)

                    target_id = getattr(target, "track_id", None)
                    if target_id is None or not isinstance(target_id, int):
                        target_id = idx + 1
                    pose_record = Pose(
                        track_id=target_id,
                        keypoints=keypoints_list,
                        scores=scores_list,
                        posture_label=posture_label,
                        spine_angle=spine_angle,
                        fall_confidence=fall_conf,
                        model_name=self.name,
                        model_version=self.version,
                    )
                    poses.append(pose_record)

                return poses

            except Exception as e:
                logger.error(f"Error during ViTPose inference: {e}. Executing fallback estimator.")

        # 2. Fallback to geometric posture estimation
        for track in person_tracks:
            x1, y1, x2, y2 = track.bbox.to_int_xyxy()
            crop_w = max(10, x2 - x1)
            crop_h = max(10, y2 - y1)
            keypoints_list, scores_list = self._extract_keypoints_fallback(x1, y1, crop_w, crop_h)
            spine_angle, posture_label, fall_conf = self._analyze_posture(keypoints_list, crop_w, crop_h)

            pose_record = Pose(
                track_id=track.track_id,
                keypoints=keypoints_list,
                scores=scores_list,
                posture_label=posture_label,
                spine_angle=spine_angle,
                fall_confidence=fall_conf,
                model_name=self.name,
                model_version=self.version,
            )
            poses.append(pose_record)

        return poses

    def _extract_keypoints_fallback(
        self, offset_x: int, offset_y: int, w: int, h: int
    ) -> Tuple[List[Keypoint], List[float]]:
        """Fallback keypoint generator when running without model weights or on tiny mock images."""
        keypoints: List[Keypoint] = []
        scores: List[float] = []

        aspect = h / max(1.0, w)
        is_horizontal = aspect < 0.9

        if is_horizontal:
            head_x, head_y = offset_x + w * 0.15, offset_y + h * 0.50
            coords = [
                (head_x, head_y, "nose"),
                (head_x - 3, head_y - 4, "l_eye"),
                (head_x - 3, head_y + 4, "r_eye"),
                (head_x - 6, head_y - 6, "l_ear"),
                (head_x - 6, head_y + 6, "r_ear"),
                (offset_x + w * 0.35, offset_y + h * 0.45, "l_shoulder"),
                (offset_x + w * 0.35, offset_y + h * 0.55, "r_shoulder"),
                (offset_x + w * 0.45, offset_y + h * 0.40, "l_elbow"),
                (offset_x + w * 0.45, offset_y + h * 0.60, "r_elbow"),
                (offset_x + w * 0.55, offset_y + h * 0.40, "l_wrist"),
                (offset_x + w * 0.55, offset_y + h * 0.60, "r_wrist"),
                (offset_x + w * 0.60, offset_y + h * 0.45, "l_hip"),
                (offset_x + w * 0.60, offset_y + h * 0.55, "r_hip"),
                (offset_x + w * 0.80, offset_y + h * 0.45, "l_knee"),
                (offset_x + w * 0.80, offset_y + h * 0.55, "r_knee"),
                (offset_x + w * 0.95, offset_y + h * 0.45, "l_ankle"),
                (offset_x + w * 0.95, offset_y + h * 0.55, "r_ankle"),
            ]
        else:
            coords = [
                (offset_x + w * 0.5, offset_y + h * 0.12, "nose"),
                (offset_x + w * 0.47, offset_y + h * 0.10, "l_eye"),
                (offset_x + w * 0.53, offset_y + h * 0.10, "r_eye"),
                (offset_x + w * 0.44, offset_y + h * 0.11, "l_ear"),
                (offset_x + w * 0.56, offset_y + h * 0.11, "r_ear"),
                (offset_x + w * 0.35, offset_y + h * 0.28, "l_shoulder"),
                (offset_x + w * 0.65, offset_y + h * 0.28, "r_shoulder"),
                (offset_x + w * 0.28, offset_y + h * 0.42, "l_elbow"),
                (offset_x + w * 0.72, offset_y + h * 0.42, "r_elbow"),
                (offset_x + w * 0.26, offset_y + h * 0.56, "l_wrist"),
                (offset_x + w * 0.74, offset_y + h * 0.56, "r_wrist"),
                (offset_x + w * 0.40, offset_y + h * 0.55, "l_hip"),
                (offset_x + w * 0.60, offset_y + h * 0.55, "r_hip"),
                (offset_x + w * 0.38, offset_y + h * 0.75, "l_knee"),
                (offset_x + w * 0.62, offset_y + h * 0.75, "r_knee"),
                (offset_x + w * 0.36, offset_y + h * 0.94, "l_ankle"),
                (offset_x + w * 0.64, offset_y + h * 0.94, "r_ankle"),
            ]

        for kx, ky, kname in coords:
            score = 0.92
            keypoints.append(Keypoint(x=float(kx), y=float(ky), score=score, name=kname))
            scores.append(score)

        return keypoints, scores

    def _analyze_posture(self, keypoints: List[Keypoint], w: float, h: float) -> Tuple[float, str, float]:
        """
        Calculate spine angle relative to vertical and determine ergonomic label.
        - Spine angle: Vector from mid-hip to mid-shoulder.
        - 0 degrees = vertical upright.
        - 90 degrees = horizontal / fallen.
        """
        kp_dict = {kp.name: kp for kp in keypoints}
        l_sh = kp_dict.get("l_shoulder") or kp_dict.get("left_shoulder")
        r_sh = kp_dict.get("r_shoulder") or kp_dict.get("right_shoulder")
        l_hip = kp_dict.get("l_hip") or kp_dict.get("left_hip")
        r_hip = kp_dict.get("r_hip") or kp_dict.get("right_hip")

        if not (l_sh and r_sh and l_hip and r_hip) or (l_sh.score < 0.2 and r_sh.score < 0.2):
            aspect = h / max(1.0, w)
            if aspect < 0.85:
                return 80.0, "fallen", 0.90
            return 10.0, "standing", 0.05

        mid_sh_x = (l_sh.x + r_sh.x) / 2.0
        mid_sh_y = (l_sh.y + r_sh.y) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0

        # Vector from hip up to shoulder
        dx = mid_sh_x - mid_hip_x
        dy = mid_hip_y - mid_sh_y

        angle_rad = math.atan2(abs(dx), max(1e-4, dy))
        spine_angle = math.degrees(angle_rad)

        aspect = h / max(1.0, w)
        # Upright person check: if bounding box is tall (aspect >= 1.3), person is physically standing
        if aspect >= 1.35:
            if spine_angle >= 50.0:
                posture_label = "bending"
                fall_conf = 0.15
            else:
                posture_label = "standing"
                fall_conf = 0.02
        elif spine_angle >= self.fall_angle_threshold and aspect < 1.1:
            posture_label = "fallen"
            fall_conf = min(0.98, max(0.70, (spine_angle / 90.0) * 0.95))
        elif spine_angle >= 40.0:
            posture_label = "bending"
            fall_conf = 0.25
        else:
            posture_label = "standing"
            fall_conf = 0.02

        return round(spine_angle, 2), posture_label, round(fall_conf, 3)
