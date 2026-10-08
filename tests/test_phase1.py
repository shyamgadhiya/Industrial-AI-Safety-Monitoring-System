"""Unit tests for Phase 1: Types, Ring Buffer, Config, and Model Registry."""
import time
import numpy as np
from src.core.types import BoundingBox, Detection, Track, Pose, Mask, Event, FramePacket, Keypoint
from src.core.buffer import CircularRingBuffer
from src.core.config import OrchestratorConfig, CameraConfig, load_config_from_yaml, save_config_to_yaml
from src.core.registry import ModelRegistry
from src.models.base import BaseVisionModel


def test_bounding_box():
    b1 = BoundingBox(10, 10, 50, 50)
    b2 = BoundingBox(30, 30, 70, 70)
    assert b1.width == 40
    assert b1.height == 40
    assert b1.area == 1600
    assert b1.centroid == (30.0, 30.0)
    iou = b1.iou(b2)
    assert round(iou, 2) == 0.14  # 400 / 2800 = ~0.1428
    print("BoundingBox tests passed.")


def test_typed_records():
    det = Detection(class_name="person", bbox=BoundingBox(10, 10, 50, 100), score=0.92)
    track = Track(track_id=1, detection_id=det.id, class_name="person", bbox=det.bbox, score=0.92, velocity=(1.5, -0.5))
    pose = Pose(track_id=1, keypoints=[Keypoint(30, 20, 0.9, "nose")], posture_label="standing")
    mask = Mask(track_id=1, polygon=[[10, 10], [50, 10], [50, 100], [10, 100]], score=0.88)
    evt = Event(type="UNSAFE_PROXIMITY", track_ids=[1], severity="CRITICAL", confidence_values={"proximity": 0.95})

    packet = FramePacket(frame_id=1, timestamp=time.time(), detections=[det], tracks=[track], poses=[pose], masks=[mask], events=[evt])
    assert len(packet.detections) == 1
    assert len(packet.tracks) == 1
    assert packet.tracks[0].track_id == 1
    assert packet.events[0].severity == "CRITICAL"
    print("Typed domain records passed.")


def test_ring_buffer():
    buf = CircularRingBuffer(capacity=5)
    for i in range(10):
        frame = np.zeros((10, 10, 3), dtype=np.uint8) + i
        buf.append(frame_id=i, timestamp=float(i), image=frame)

    assert buf.size() == 5
    recent = buf.get_recent_frames(3)
    assert len(recent) == 3
    # Check chronological order (latest frames 7, 8, 9)
    assert recent[-1][0] == 9
    print("RingBuffer eviction and retrieval passed.")


def test_config():
    config = OrchestratorConfig()
    assert config.pipeline_name == "Factory Safety Orchestrator"
    assert len(config.cameras) == 1
    assert config.detector.options["model_type"] == "rf-detr-small"
    print("Config tests passed.")


class DummyDetector(BaseVisionModel):
    def __init__(self, name="dummy", version="v1", device="cpu", options=None):
        super().__init__(name, version, device, options)

    def initialize(self):
        self._is_initialized = True

    def predict(self, frame):
        return [Detection(class_name="person", score=0.95)]


def test_registry():
    ModelRegistry.register("dummy_det")(DummyDetector)
    det = ModelRegistry.create("dummy_det")
    res = det.predict(np.zeros((10, 10, 3)))
    assert len(res) == 1
    assert res[0].class_name == "person"
    print("Registry test passed.")


if __name__ == "__main__":
    test_bounding_box()
    test_typed_records()
    test_ring_buffer()
    test_config()
    test_registry()
    print("All Phase 1 tests passed successfully!")
