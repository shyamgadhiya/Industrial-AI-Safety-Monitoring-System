"""
Thread-safe circular ring buffer for video frames.
Supports rolling pre-event frame retention and clip extraction when rules fire.
"""

from __future__ import annotations
import threading
import collections
from typing import List, Tuple, Optional
import numpy as np


class FrameBufferItem:
    """A single buffered frame with its capture metadata."""
    __slots__ = ("frame_id", "timestamp", "image")

    def __init__(self, frame_id: int, timestamp: float, image: np.ndarray):
        self.frame_id = frame_id
        self.timestamp = timestamp
        # Store copy or reference; keeping reference or low-res copy depending on memory
        self.image = image


class CircularRingBuffer:
    """
    Fixed-capacity circular queue for raw video frames.
    Oldest frames are automatically evicted when buffer reaches capacity.
    """

    def __init__(self, capacity: int = 300):
        """
        Args:
            capacity: Maximum number of frames to retain (e.g. 300 frames = 10s @ 30 FPS).
        """
        self.capacity = max(1, capacity)
        self._buffer: collections.deque = collections.deque(maxlen=self.capacity)
        self._lock = threading.Lock()

    def append(self, frame_id: int, timestamp: float, image: np.ndarray) -> None:
        """Add a frame to the circular buffer."""
        # Optional: copy image to prevent downstream modification
        item = FrameBufferItem(frame_id, timestamp, image.copy() if image is not None else None)
        with self._lock:
            self._buffer.append(item)

    def get_recent_frames(self, count: Optional[int] = None) -> List[Tuple[int, float, np.ndarray]]:
        """
        Retrieve the latest `count` frames in chronological order.
        Returns list of (frame_id, timestamp, image).
        """
        with self._lock:
            items = list(self._buffer)

        if count is not None and count < len(items):
            items = items[-count:]

        return [(item.frame_id, item.timestamp, item.image) for item in items if item.image is not None]

    def get_frames_around_timestamp(self, target_timestamp: float, seconds_before: float = 3.0, seconds_after: float = 3.0) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames falling within [target_timestamp - seconds_before, target_timestamp + seconds_after].
        """
        t_min = target_timestamp - seconds_before
        t_max = target_timestamp + seconds_after

        with self._lock:
            items = [item for item in self._buffer if t_min <= item.timestamp <= t_max]

        return [(item.frame_id, item.timestamp, item.image) for item in items if item.image is not None]

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
