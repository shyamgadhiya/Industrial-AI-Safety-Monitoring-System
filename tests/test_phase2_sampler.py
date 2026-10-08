"""Unit tests for Phase 2: VideoStreamReader, FrameSampler, and Ring Buffer interaction."""
import time
from src.core.buffer import CircularRingBuffer
from src.ingestion.reader import VideoStreamReader
from src.ingestion.sampler import FrameSampler


def test_reader_and_sampler():
    video_path = "data/samples/sample_factory_feed.mp4"
    ring_buffer = CircularRingBuffer(capacity=50)

    reader = VideoStreamReader(source=video_path, camera_id="test_cam", loop_video=False)
    reader.start()

    sampler = FrameSampler(reader=reader, ring_buffer=ring_buffer, target_fps=30.0, stride=1)

    packets = []
    # Collect 15 packets
    start_t = time.time()
    while len(packets) < 15 and time.time() - start_t < 5.0:
        pkt = sampler.sample_packet()
        if pkt is not None:
            packets.append(pkt)
        time.sleep(0.01)

    reader.stop()

    assert len(packets) == 15, f"Expected 15 packets, got {len(packets)}"
    assert ring_buffer.size() > 0, "Ring buffer should have stored frames"
    first_pkt = packets[0]
    assert first_pkt.camera_id == "test_cam"
    assert first_pkt.image is not None
    assert first_pkt.image.shape == (720, 1280, 3)
    print("Phase 2 Reader and Sampler tests passed successfully!")


if __name__ == "__main__":
    test_reader_and_sampler()
