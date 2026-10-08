"""Unit tests for Phase 3: RF-DETR Detector."""
import cv2
from src.core.registry import ModelRegistry
from src.models.detector_rfdetr import RFDETRSmallDetector


def test_rfdetr_detector():
    # Instantiate detector via Registry
    detector = ModelRegistry.create("rf_detr_small")
    detector.initialize()

    # Load a test frame from sample video
    cap = cv2.VideoCapture("data/samples/sample_factory_feed.mp4")
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None, "Failed to read test frame from video"

    # Run detection
    detections = detector.predict(frame)
    print(f"Total detections on frame 0: {len(detections)}")
    for d in detections:
        print(f"Detected: {d.class_name}, score={d.score:.2f}, bbox={d.bbox.to_int_xyxy()}")

    assert len(detections) >= 1, "Detector should find person or forklift"
    classes = [d.class_name for d in detections]
    assert "person" in classes or "forklift" in classes

    # Run benchmark
    bench = detector.benchmark(frame, warmup_runs=2, benchmark_runs=5)
    print("RF-DETR Benchmark Results:", bench)
    assert bench["avg_latency_ms"] >= 0.0

    # Verify provenance
    prov = detector.get_provenance()
    assert "rfdetr" in prov["model_name"].lower().replace("-", "")
    print("Phase 3 RF-DETR detector tests passed successfully!")


if __name__ == "__main__":
    test_rfdetr_detector()
