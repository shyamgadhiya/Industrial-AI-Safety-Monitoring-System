"""Unit tests for Phase 5: ViTPose Posture Estimator."""
import cv2
from src.core.types import BoundingBox, Track
from src.core.registry import ModelRegistry
from src.models.pose_vitpose import ViTPoseEstimator


def test_vitpose_posture():
    estimator = ModelRegistry.create("vitpose_small")
    estimator.initialize()

    # Create dummy standing worker track
    track_standing = Track(
        track_id=1,
        detection_id="det_01",
        class_name="person",
        bbox=BoundingBox(100, 100, 160, 260),  # Upright box: height=160, width=60
        score=0.95
    )

    # Create dummy fallen worker track
    track_fallen = Track(
        track_id=2,
        detection_id="det_02",
        class_name="person",
        bbox=BoundingBox(300, 300, 440, 350),  # Horizontal box: width=140, height=50
        score=0.92
    )

    dummy_frame = cv2.imread("data/samples/sample_factory_feed.mp4") # Will be None or we can make a numpy frame
    import numpy as np
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    poses = estimator.predict(dummy_frame, [track_standing, track_fallen])
    assert len(poses) == 2, f"Expected 2 poses, got {len(poses)}"

    pose_standing = next(p for p in poses if p.track_id == 1)
    pose_fallen = next(p for p in poses if p.track_id == 2)

    print(f"Track 1 Posture: {pose_standing.posture_label}, Spine Angle={pose_standing.spine_angle} deg, Fall Conf={pose_standing.fall_confidence}")
    print(f"Track 2 Posture: {pose_fallen.posture_label}, Spine Angle={pose_fallen.spine_angle} deg, Fall Conf={pose_fallen.fall_confidence}")

    assert pose_standing.posture_label == "standing"
    assert pose_fallen.posture_label == "fallen"
    assert pose_fallen.fall_confidence > 0.60
    assert len(pose_standing.keypoints) == 17

    prov = estimator.get_provenance()
    assert "vitpose" in prov["model_name"]
    print("ViTPose posture estimation and fall detection verified successfully!")


if __name__ == "__main__":
    test_vitpose_posture()
