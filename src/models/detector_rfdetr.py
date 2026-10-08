"""
RF-DETR (Real-Time Detection Transformer) object detector module.
Fine-tuned warehouse detector strictly targeting Workers and Forklifts.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from rfdetr import RFDETRSmall
    from rfdetr.assets.coco_classes import COCO_CLASSES
    HAS_RFDETR_PKG = True
except ImportError:
    HAS_RFDETR_PKG = False
    COCO_CLASSES = ["person", "truck", "car"]

from src.core.registry import ModelRegistry
from src.core.types import BoundingBox, Detection
from src.models.base import BaseVisionModel

logger = logging.getLogger(__name__)

# Checkpoint class mappings from training dataset:
# 0: box, 1: worker, 2: forklift, 3: pallet, 4: safety helmet
WAREHOUSE_CLASSES = ["box", "worker", "forklift", "pallet", "safety helmet"]


@ModelRegistry.register("rf_detr")
@ModelRegistry.register("rf_detr_small")
@ModelRegistry.register("rfdetrsmall")
@ModelRegistry.register("rfdetr_small")
@ModelRegistry.register("rfdetrmedium")
class RFDETRSmallDetector(BaseVisionModel):
    """
    Industrial RF-DETR detector targeting Workers and Forklifts with
    per-class thresholding, geometric filtering, and temporal stabilization.
    """

    def __init__(
        self,
        name: str = "rfdetrSmall",
        version: str = "v1.2",
        device: str = "cpu",
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, version=version, device=device, options=options or {})

        self.checkpoint_path = self.options.get(
            "checkpoint_path",
            "src/models/rfdetr_checkpoint_best_total.pth",
        )
        self.default_conf_threshold = float(self.options.get("confidence_threshold", 0.30))

        # Dynamic class-wise threshold mapping
        user_thresholds = self.options.get("class_thresholds", {})
        self.class_thresholds: Dict[str, float] = {
            "worker": float(user_thresholds.get("worker", self.options.get("confidence_threshold", 0.25))),
            "forklift": float(user_thresholds.get("forklift", self.options.get("forklift_confidence_threshold", 0.30))),
        }

        # Temporal filter settings for forklift confirmation
        self.temporal_window: int = int(self.options.get("forklift_window", 4))
        self.temporal_min_hits: int = int(self.options.get("forklift_min_hits", 2))
        self._forklift_history: Dict[str, Deque[bool]] = {}

        self.target_classes = {"worker", "forklift"}
        self._model = None
        self._backend = "uninitialized"
        self.is_custom_checkpoint = False

    def initialize(self) -> None:
        """Loads custom checkpoint if present, falling back to COCO weights."""
        logger.info(f"Initializing {self.name} on device '{self.device}'...")

        if not HAS_RFDETR_PKG:
            logger.warning("rfdetr package not found. Running in mock/native mode.")
            self._backend = "native"
            self._is_initialized = True
            return

        ckpt = self.checkpoint_path
        if ckpt and os.path.exists(ckpt):
            try:
                logger.info(f"Loading custom RF-DETR checkpoint from '{ckpt}'...")
                self._model = RFDETRSmall(
                    num_classes=len(WAREHOUSE_CLASSES),
                    pretrain_weights=str(ckpt),
                    device=self.device or "cpu",
                )
                self._backend = "rfdetr_torch"
                self.is_custom_checkpoint = True
                logger.info("Custom warehouse RF-DETR model loaded successfully.")
            except Exception as err:
                logger.warning(f"Failed to load checkpoint '{ckpt}': {err}. Reverting to standard weights.")

        if self._model is None:
            try:
                self._model = RFDETRSmall()
                self._backend = "rfdetr_torch"
                self.is_custom_checkpoint = False
                logger.info("Loaded base RF-DETR model.")
            except Exception as err:
                logger.error(f"Failed to initialize standard RF-DETR: {err}")
                self._backend = "native"

        self._is_initialized = True

    # ------------------------------------------------------------------
    # Filtering & Mapping Helpers
    # ------------------------------------------------------------------

    def _resolve_canonical_class(self, cls_id: int) -> Optional[str]:
        """Translates raw checkpoint / COCO IDs to canonical targets ('worker', 'forklift')."""
        if self.is_custom_checkpoint:
            label = WAREHOUSE_CLASSES[cls_id] if cls_id < len(WAREHOUSE_CLASSES) else ""
        else:
            raw = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else ""
            label = "worker" if raw == "person" else "forklift" if raw in ("truck", "car") else ""

        return label if label in self.target_classes else None

    @staticmethod
    def _is_valid_forklift_geometry(bw: float, bh: float, frame_w: int, frame_h: int) -> bool:
        """Applies spatial and aspect-ratio validation rules for forklifts."""
        aspect_ratio = bh / bw if bw > 0 else 0.0
        return (
            aspect_ratio >= 1.35            # Forklifts with masts are vertically oriented
            and bw <= frame_w * 0.25        # Narrow profile constraint
            and (frame_h * 0.15 <= bh <= frame_h * 0.85)  # Scale boundary constraints
        )

    def _apply_temporal_consistency(
        self, candidates: List[Detection], frame_w: int, frame_h: int
    ) -> List[Detection]:
        """Filters out fleeting false positives using spatial grid hit counts."""
        grid_candidates: Dict[str, Detection] = {}
        active_keys = set()

        for det in candidates:
            cx = (det.bbox.x1 + det.bbox.x2) / 2.0
            cy = (det.bbox.y1 + det.bbox.y2) / 2.0
            key = f"{int(cx / frame_w * 8)}_{int(cy / frame_h * 8)}"
            active_keys.add(key)
            if key not in grid_candidates or det.score > grid_candidates[key].score:
                grid_candidates[key] = det

        # Update sliding window
        all_keys = set(self._forklift_history.keys()) | active_keys
        for k in all_keys:
            if k not in self._forklift_history:
                self._forklift_history[k] = deque(maxlen=self.temporal_window)
            self._forklift_history[k].append(k in active_keys)

        # Cleanup expired grid cells
        self._forklift_history = {k: dq for k, dq in self._forklift_history.items() if any(dq)}

        # Confirm candidates that meet hit requirement
        confirmed = []
        for k, det in grid_candidates.items():
            if sum(self._forklift_history.get(k, [])) >= self.temporal_min_hits:
                confirmed.append(det)
            else:
                logger.debug(f"Forklift candidate at cell {k} suppressed by temporal consistency.")
        return confirmed

    # ------------------------------------------------------------------
    # Inference Pipeline
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Executes inference and returns validated worker and forklift detections."""
        if not self._is_initialized:
            self.initialize()

        if frame is None or frame.size == 0 or self._model is None:
            return []

        h, w = frame.shape[:2]
        min_inference_thresh = min(self.class_thresholds.values(), default=self.default_conf_threshold)

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._model.predict(rgb_frame, threshold=min_inference_thresh)
            if not hasattr(results, "xyxy") or len(results.xyxy) == 0:
                self._apply_temporal_consistency([], w, h)
                return []

            confirmed_workers: List[Detection] = []
            candidate_forklifts: List[Detection] = []

            for i in range(len(results.xyxy)):
                cls_id = int(results.class_id[i])
                class_name = self._resolve_canonical_class(cls_id)
                if not class_name:
                    continue

                score = float(results.confidence[i])
                thresh = self.class_thresholds.get(class_name, self.default_conf_threshold)
                if score < thresh:
                    continue

                box = results.xyxy[i]
                bx1, by1 = max(0.0, float(box[0])), max(0.0, float(box[1]))
                bx2, by2 = min(float(w), float(box[2])), min(float(h), float(box[3]))
                bw, bh = bx2 - bx1, by2 - by1

                # Discard degenerate boxes (< 2% of frame size)
                if bw < w * 0.02 or bh < h * 0.02:
                    continue

                detection = Detection(
                    class_name=class_name,
                    bbox=BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2),
                    score=round(score, 4),
                    model_name=self.name,
                    model_version=self.version,
                    metadata={"class_id": cls_id},
                )

                if class_name == "worker":
                    confirmed_workers.append(detection)
                elif class_name == "forklift":
                    if self._is_valid_forklift_geometry(bw, bh, w, h):
                        candidate_forklifts.append(detection)

            confirmed_forklifts = self._apply_temporal_consistency(candidate_forklifts, w, h)
            return confirmed_workers + confirmed_forklifts

        except Exception as err:
            logger.error(f"Inference error in {self.name}: {err}", exc_info=True)
            return []