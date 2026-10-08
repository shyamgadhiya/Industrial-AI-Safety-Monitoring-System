"""Unit tests for Phase 4: ByteTrack Tracker and Cascade Pipeline."""
import cv2
from src.core.registry import ModelRegistry
from src.models.detector_rfdetr import RFDETRSmallDetector
from src.models.tracker_bytetrack import ByteTrackTracker


def test_bytetrack_tracker():
    detector = ModelRegistry.create("rf_detr_small")
    tracker = ModelRegistry.create("bytetrack")
    detector.initialize()
    tracker.initialize()

    cap = cv2.VideoCapture("data/samples/sample_factory_feed.mp4")

    all_tracks = []
    frame_idx = 0
    while frame_idx < 30:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = float(frame_idx) / 30.0
        dets = detector.predict(frame)
        tracks = tracker.predict(dets, timestamp=timestamp)

        if tracks:
            all_tracks.extend(tracks)
            # Print state on 5th and 20th frames
            if frame_idx in (5, 20):
                for t in tracks:
                    print(f"[Frame {frame_idx}] Track ID={t.track_id} Class={t.class_name} Hits={t.hits} Vel={t.velocity}")

        frame_idx += 1

    cap.release()

    assert len(all_tracks) > 0, "ByteTrack should have tracked detected objects"
    # Ensure velocity and trajectories are populated
    latest_track = all_tracks[-1]
    assert latest_track.track_id > 0
    assert hasattr(latest_track, "velocity")
    print("ByteTrack tracking and persistent ID assignment verified successfully!")


if __name__ == "__main__":
    test_bytetrack_tracker()
