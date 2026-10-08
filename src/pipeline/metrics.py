"""
Real-time pipeline telemetry, latency profiling, and throughput monitoring.
Tracks per-stage milliseconds (RF-DETR, ByteTrack, ViTPose, SAM, CLIP) and end-to-end FPS.
"""

from __future__ import annotations
import time
import collections
from typing import Dict, List, Any


class PipelineMetricsTracker:
    """Sliding-window performance metrics collector."""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._latencies: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.window_size)
        )
        self._frame_times: collections.deque = collections.deque(maxlen=self.window_size)
        self.total_processed_frames: int = 0
        self.total_dropped_frames: int = 0
        self.total_events_fired: int = 0

    def record_stage_latency(self, stage_name: str, latency_ms: float) -> None:
        self._latencies[stage_name].append(latency_ms)

    def record_frame_processed(self) -> None:
        self._frame_times.append(time.time())
        self.total_processed_frames += 1

    def record_dropped_frame(self) -> None:
        self.total_dropped_frames += 1

    def record_event_fired(self) -> None:
        self.total_events_fired += 1

    def get_current_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        time_span = self._frame_times[-1] - self._frame_times[0]
        if time_span <= 0:
            return 0.0
        return round((len(self._frame_times) - 1) / time_span, 1)

    def get_average_latencies(self) -> Dict[str, float]:
        stats = {}
        for stage, times in self._latencies.items():
            if times:
                stats[stage] = round(sum(times) / len(times), 2)
            else:
                stats[stage] = 0.0
        return stats

    def get_telemetry_summary(self) -> Dict[str, Any]:
        latencies = self.get_average_latencies()
        total_lat = sum(latencies.values())
        return {
            "fps": self.get_current_fps(),
            "total_frames_processed": self.total_processed_frames,
            "total_frames_dropped": self.total_dropped_frames,
            "total_events_fired": self.total_events_fired,
            "latencies_ms": latencies,
            "total_pipeline_latency_ms": round(total_lat, 2),
        }
