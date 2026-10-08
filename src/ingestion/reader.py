"""
High-throughput, fault-tolerant video stream reader for RTSP feeds, webcams, and video files.
Supports threaded background frame acquisition and automatic reconnection.
"""

from __future__ import annotations
import os
import time
import logging
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStreamReader:
    """
    Threaded reader that continuously fetches the latest frame from an RTSP stream,
    webcam, or video file to eliminate camera frame buffer latency.
    """

    def __init__(
        self,
        source: str,
        camera_id: str = "cam_01",
        loop_video: bool = True,
        reconnect_delay_sec: float = 2.0,
        max_reconnect_attempts: int = 10,
    ):
        self.source = source
        self.camera_id = camera_id
        self.loop_video = loop_video
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_index: int = 0
        self._has_new_frame: bool = False
        self.fps: float = 25.0
        self.width: int = 0
        self.height: int = 0
        self.is_file: bool = False

        if isinstance(self.source, str) and self.source.isdigit():
            self.is_file = False
        else:
            self.is_file = isinstance(self.source, str) and not (self.source.startswith("rtsp://") or self.source.startswith("http://"))

        # Pre-probe source for native properties if it is a local file
        if self.is_file and os.path.exists(str(self.source)):
            try:
                probe_cap = cv2.VideoCapture(self.source)
                if probe_cap.isOpened():
                    p_fps = probe_cap.get(cv2.CAP_PROP_FPS)
                    if p_fps and p_fps > 0 and not np.isnan(p_fps):
                        self.fps = float(p_fps)
                    self.width = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    probe_cap.release()
            except Exception:
                pass

    def start(self) -> "VideoStreamReader":
        """Initialize connection and start background capture thread."""
        self._open_capture()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name=f"Reader-{self.camera_id}", daemon=True)
        self._thread.start()
        logger.info(f"Started video stream reader for [{self.camera_id}] from source '{self.source}' (native FPS: {self.fps})")
        return self

    def _open_capture(self) -> bool:
        """Open or reopen cv2.VideoCapture."""
        # Check if source is an integer string (webcam index)
        if isinstance(self.source, str) and self.source.isdigit():
            src_val = int(self.source)
        else:
            src_val = self.source
            self.is_file = isinstance(src_val, str) and not (src_val.startswith("rtsp://") or src_val.startswith("http://"))

        self._cap = cv2.VideoCapture(src_val)
        if not self._cap.isOpened():
            logger.warning(f"Failed to open video source: {self.source}")
            return False

        # Read native properties
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0 and not np.isnan(fps):
            self.fps = float(fps)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    def _capture_loop(self) -> None:
        """Continuous background thread fetching latest frame."""
        reconnect_count = 0
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"Exceeded max reconnect attempts for [{self.camera_id}]. Exiting reader thread.")
                    break
                reconnect_count += 1
                logger.info(f"Reconnecting to {self.source} (Attempt {reconnect_count}/{self.max_reconnect_attempts})...")
                time.sleep(self.reconnect_delay_sec)
                self._open_capture()
                continue

            # For video files, wait briefly if previous frame hasn't been read by consumer yet
            if self.is_file and self._has_new_frame:
                t_wait = time.time()
                while self._running and self._has_new_frame and (time.time() - t_wait < 0.08):
                    time.sleep(0.003)

            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self.is_file and self.loop_video:
                    # Rewind video file
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.warning(f"End of stream or read failure on [{self.camera_id}]")
                    self._cap.release()
                    self._cap = None
                    time.sleep(0.1)
                    continue

            reconnect_count = 0
            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = time.time()
                self._frame_index += 1
                self._has_new_frame = True

            # Prevent 100% CPU lock on file playback - pace at native FPS
            if self.is_file and self.fps > 0:
                time.sleep(1.0 / self.fps)

    def read_latest(self) -> Tuple[bool, int, float, Optional[np.ndarray]]:
        """
        Thread-safe fetch of the latest captured frame.
        Returns: (success, frame_index, timestamp, image_bgr)
        """
        with self._lock:
            if self._latest_frame is None:
                return False, 0, 0.0, None
            self._has_new_frame = False
            return True, self._frame_index, self._latest_timestamp, self._latest_frame.copy()

    def stop(self) -> None:
        """Stop background capture and release video hardware."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info(f"Stopped reader for [{self.camera_id}]")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
