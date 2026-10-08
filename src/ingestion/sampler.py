"""
Adaptive frame sampler and preprocessor.
Pushes raw frames to the high-capacity CircularRingBuffer while yielding
subsampled FramePackets to the downstream detection & tracking pipeline.
"""

from __future__ import annotations
import time
from typing import Optional, Tuple, Generator
import cv2
import numpy as np

from src.core.types import FramePacket
from src.core.buffer import CircularRingBuffer
from src.ingestion.reader import VideoStreamReader


class FrameSampler:
    """
    Subsamples video frames for inference while preserving full-fidelity
    continuous recordings in a CircularRingBuffer for evidence generation.
    """

    def __init__(
        self,
        reader: VideoStreamReader,
        ring_buffer: CircularRingBuffer,
        target_fps: float = 1.0,
        target_resolution: Optional[Tuple[int, int]] = (1280, 720),
        stride: int = 1,
        sample_interval_sec: Optional[float] = None,
    ):
        self.reader = reader
        self.ring_buffer = ring_buffer
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.stride = max(1, stride)
        if sample_interval_sec is not None:
            self.sample_interval_sec = float(sample_interval_sec)
        elif target_fps > 0:
            self.sample_interval_sec = 1.0 / float(target_fps)
        else:
            self.sample_interval_sec = 1.0

        self._last_sampled_time: float = 0.0
        self._sampled_count: int = 0
        self._last_processed_frame_idx: int = -1

    def sample_packet(self) -> Optional[FramePacket]:
        """
        Polls the reader, appends to ring buffer, and returns a FramePacket
        if the sampling condition (stride/sample_interval_sec) is met.
        """
        success, frame_idx, timestamp, raw_frame = self.reader.read_latest()
        if not success or raw_frame is None or frame_idx == self._last_processed_frame_idx:
            return None

        self._last_processed_frame_idx = frame_idx

        # Ring buffer retention is performed post-inference with processed annotations
        # to ensure evidence clips contain bounding boxes, poses, and alerts.

        # 2. Check stride subsampling
        if frame_idx % self.stride != 0:
            return None

        # 3. Check interval throttle (e.g. process after 1 second)
        now = time.time()
        if self._last_sampled_time > 0 and (now - self._last_sampled_time) < self.sample_interval_sec:
            return None

        self._last_sampled_time = now
        self._sampled_count += 1

        # 4. Optional resolution resizing
        processed_frame = raw_frame
        if self.target_resolution is not None:
            target_w, target_h = self.target_resolution
            curr_h, curr_w = raw_frame.shape[:2]
            if (curr_w, curr_h) != (target_w, target_h):
                processed_frame = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 5. Pack into typed intermediate FramePacket with reader's native original video FPS
        orig_fps = getattr(self.reader, "fps", None)
        orig_fps = float(orig_fps) if orig_fps and orig_fps > 0 else 25.0

        packet = FramePacket(
            frame_id=frame_idx,
            timestamp=timestamp,
            image=processed_frame,
            camera_id=self.reader.camera_id,
            fps=orig_fps,
        )
        return packet

    def stream_packets(self, sleep_interval: float = 0.005) -> Generator[FramePacket, None, None]:
        """Generator yielding FramePackets continuously."""
        while self.reader._running:
            packet = self.sample_packet()
            if packet is not None:
                yield packet
            else:
                time.sleep(sleep_interval)
