"""Core module initialization."""
from src.core.types import (
    BoundingBox,
    Detection,
    Track,
    Keypoint,
    Pose,
    Mask,
    Event,
    FramePacket,
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON_PAIRS,
)

__all__ = [
    "BoundingBox",
    "Detection",
    "Track",
    "Keypoint",
    "Pose",
    "Mask",
    "Event",
    "FramePacket",
    "COCO_KEYPOINT_NAMES",
    "COCO_SKELETON_PAIRS",
]
