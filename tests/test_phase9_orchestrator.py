"""Unit test for Phase 9: Full VisionPipelineOrchestrator End-to-End Execution."""
import time
from src.core.config import OrchestratorConfig, CameraConfig
from src.pipeline.orchestrator import VisionPipelineOrchestrator


def test_full_pipeline_orchestration():
    # Setup test configuration
    config = OrchestratorConfig(
        pipeline_name="Test Industrial Pipeline",
        storage_dir="data/evidence/test_orch",
    )
    cam = CameraConfig(
        camera_id="cam_test",
        source="data/samples/sample_factory_feed.mp4",
        target_fps=20,
        enable_detector=True,
        enable_tracker=True,
        enable_vitpose=True,
        enable_clip=True,
        ring_buffer_seconds=5,
    )
    config.cameras = [cam]

    orchestrator = VisionPipelineOrchestrator(config)
    orchestrator.initialize_models()
    orchestrator.add_camera(cam)

    processed_packets = []

    def on_packet(pkt):
        processed_packets.append(pkt)

    orchestrator.add_packet_listener(on_packet)

    # Process 25 frames directly
    ctx = orchestrator.camera_contexts["cam_test"]
    ctx["reader"].start()
    time.sleep(0.1)  # Allow first frame capture
    sampler = ctx["sampler"]
    ring_buf = ctx["ring_buffer"]

    start_wait = time.time()
    while len(processed_packets) < 15 and time.time() - start_wait < 3.0:
        pkt = sampler.sample_packet()
        if pkt:
            orchestrator.process_single_frame(pkt, cam, ring_buf)
        time.sleep(0.01)

    ctx["reader"].stop()

    print(f"Total processed packets: {len(processed_packets)}")
    assert len(processed_packets) > 0

    last_pkt = processed_packets[-1]
    print("Stage Latencies (ms):", last_pkt.stage_latencies_ms)
    assert "rf_detr" in last_pkt.stage_latencies_ms
    assert "bytetrack" in last_pkt.stage_latencies_ms

    telemetry = orchestrator.metrics.get_telemetry_summary()
    print("Pipeline Telemetry Summary:", telemetry)
    assert telemetry["total_frames_processed"] > 0

    print("Phase 9 Full Vision Pipeline Orchestration verified successfully!")


if __name__ == "__main__":
    test_full_pipeline_orchestration()
