"""Unit tests for Phase 8: Event/Rule Engine and Evidence Persistence."""
import os
import time
import numpy as np
from src.core.types import BoundingBox, Track, Pose, FramePacket
from src.core.buffer import CircularRingBuffer
from src.rules.engine import EventRuleEngine
from src.rules.evidence import EvidencePersistenceManager


def test_rules_and_evidence():
    evidence_dir = "data/evidence/test_output"
    ev_mgr = EvidencePersistenceManager(output_dir=evidence_dir, fps=10.0)
    engine = EventRuleEngine(evidence_manager=ev_mgr)

    # Populate a ring buffer with dummy frames
    ring_buf = CircularRingBuffer(capacity=20)
    for i in range(15):
        frame = np.full((100, 100, 3), i * 10, dtype=np.uint8)
        ring_buf.append(frame_id=i, timestamp=float(i), image=frame)

    # 1. Test proximity violation: worker near forklift
    t_person = Track(
        track_id=1,
        detection_id="d1",
        class_name="person",
        bbox=BoundingBox(50, 50, 80, 90),
        score=0.95
    )
    t_forklift = Track(
        track_id=2,
        detection_id="d2",
        class_name="forklift",
        bbox=BoundingBox(85, 50, 120, 90),  # Close proximity
        score=0.91
    )

    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    packet_prox = FramePacket(
        frame_id=16,
        timestamp=16.0,
        image=test_img,
        tracks=[t_person, t_forklift]
    )

    events_prox = engine.process_packet(packet_prox, ring_buf)
    print("Fired Proximity Events:", len(events_prox))
    assert len(events_prox) >= 1
    evt1 = events_prox[0]
    assert evt1.type == "UNSAFE_WORKER_FORKLIFT_PROXIMITY"
    assert evt1.severity == "CRITICAL"
    assert "detector" in evt1.model_versions
    assert evt1.evidence_uri is not None

    # 2. Test fall rule: fallen worker pose
    pose_fallen = Pose(
        track_id=1,
        posture_label="fallen",
        spine_angle=82.0,
        fall_confidence=0.94
    )
    packet_fall = FramePacket(
        frame_id=17,
        timestamp=17.0,
        image=test_img,
        tracks=[t_person],
        poses=[pose_fallen]
    )

    events_fall = engine.process_packet(packet_fall, ring_buf)
    print("Fired Fall Events:", len(events_fall))
    assert len(events_fall) >= 1
    evt2 = events_fall[0]
    assert evt2.type == "WORKER_FALL_DETECTED"

    # Wait briefly for background evidence thread to write
    time.sleep(3.5)

    # Check evidence directory
    files = os.listdir(evidence_dir)
    print("Persisted Evidence Files:", files)
    assert any(f.endswith(".json") for f in files), "Expected metadata JSON file"
    assert any(f.endswith(".mp4") for f in files), "Expected evidence MP4 clip"

    print("Phase 8 Rule Engine and Evidence Persistence verified successfully!")


if __name__ == "__main__":
    test_rules_and_evidence()
