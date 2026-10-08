"""
Evidence recorder module.
Extracts rolling frames from CircularRingBuffer when rules fire, writes MP4 video clips,
and persists structured JSON incident metadata records.
"""

from __future__ import annotations
import os
import json
import time
import logging
import threading
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from src.core.types import Event, FramePacket
from src.core.buffer import CircularRingBuffer
from src.ui.renderer import FrameVisualizer

logger = logging.getLogger(__name__)


class EvidencePersistenceManager:
    """
    Manages the creation and storage of high-resolution evidence clips and incident JSON logs.
    Saves processed images and videos containing bounding boxes, keypoints/poses, and alerts.
    Executes disk writing on background threads so pipeline inference is never stalled.
    """

    def __init__(
        self,
        output_dir: str = "data/evidence",
        fps: float = 25.0,
        visualizer: Optional[FrameVisualizer] = None,
    ):
        self.output_dir = output_dir
        self.fps = float(fps if fps and fps > 0 else 25.0)
        self.visualizer = visualizer or FrameVisualizer()
        os.makedirs(self.output_dir, exist_ok=True)

    def persist_incident_evidence(
        self,
        event: Event,
        packet: FramePacket,
        ring_buffer: CircularRingBuffer,
        pre_seconds: float = 3.0,
        post_seconds: float = 3.0,
        fps: Optional[float] = None,
    ) -> Event:
        """
        Extracts pre-event processed frames from the ring buffer, captures current processed frame snapshot,
        schedules clip creation, and populates evidence URIs on the Event record.
        """
        timestamp_str = f"{int(event.timestamp)}"
        safe_type = event.type.lower().replace(" ", "_")
        base_name = f"evidence_{safe_type}_{timestamp_str}_{event.event_id[:6]}"

        snapshot_filename = f"{base_name}.jpg"
        clip_filename = f"{base_name}.mp4"
        meta_filename = f"{base_name}.json"

        snapshot_path = os.path.join(self.output_dir, snapshot_filename)
        clip_path = os.path.join(self.output_dir, clip_filename)
        meta_path = os.path.join(self.output_dir, meta_filename)

        # Determine original video FPS: explicit argument > packet.fps > self.fps
        if fps is not None and fps > 0:
            effective_fps = float(fps)
        elif getattr(packet, "fps", None) and packet.fps > 0:
            effective_fps = float(packet.fps)
        else:
            effective_fps = float(self.fps if self.fps > 0 else 25.0)

        event.metadata["video_fps"] = round(effective_fps, 2)

        # 1. Render and save instantaneous high-res PROCESSED snapshot (with detections, poses & alerts)
        annotated_snapshot = None
        if packet.image is not None:
            # Temporarily ensure event is present on packet so visualizer renders the incident alert banner
            if event not in packet.events:
                packet.events.append(event)

            annotated_snapshot = self.visualizer.render(
                packet,
                show_detections=True,
                show_tracks=True,
                show_poses=True,
                show_masks=True,
                show_events=True,
                show_hud=True,
            )
            cv2.imwrite(snapshot_path, annotated_snapshot)
            event.snapshot_uri = snapshot_path

            # Also buffer this processed frame immediately
            ring_buffer.append(packet.frame_id, packet.timestamp, annotated_snapshot)

        event.evidence_uri = clip_path

        # 2. Offload MP4 writing and JSON persistence to background worker thread
        threading.Thread(
            target=self._write_clip_and_meta_worker,
            args=(
                clip_path,
                meta_path,
                ring_buffer,
                event.timestamp,
                pre_seconds,
                post_seconds,
                event.to_dict(),
                effective_fps,
                annotated_snapshot,
            ),
            daemon=True,
        ).start()

        logger.info(f"Recorded processed incident evidence for [{event.type}] -> {clip_path} at original {effective_fps:.2f} FPS")
        return event

    @staticmethod
    def _write_clip_and_meta_worker(
        clip_path: str,
        meta_path: str,
        ring_buffer: CircularRingBuffer,
        event_timestamp: float,
        pre_seconds: float,
        post_seconds: float,
        event_dict: Dict[str, Any],
        fps: float,
        fallback_frame: Optional[np.ndarray] = None,
    ):
        """Worker function executed in background thread."""
        try:
            # Allow post-incident processed frames to accumulate in ring_buffer
            wait_time = min(max(0.5, post_seconds), 3.0)
            time.sleep(wait_time)

            target_fps = float(fps if fps and fps > 0 else 25.0)
            event_dict["video_fps"] = round(target_fps, 2)
            if "metadata" in event_dict and isinstance(event_dict["metadata"], dict):
                event_dict["metadata"]["video_fps"] = round(target_fps, 2)

            buffered_frames = ring_buffer.get_frames_around_timestamp(
                target_timestamp=event_timestamp,
                seconds_before=pre_seconds,
                seconds_after=post_seconds,
            )

            # Fallback if buffer is young: grab recent frames
            if len(buffered_frames) < 5:
                buffered_frames = ring_buffer.get_recent_frames(count=int(target_fps * (pre_seconds + post_seconds)))

            # If ring buffer had no frames, fallback to snapshot frame
            if not buffered_frames and fallback_frame is not None:
                buffered_frames = [(0, event_timestamp, fallback_frame)]

            # Filter valid frames with actual image arrays
            valid_frames = [f for f in buffered_frames if f[2] is not None]

            # Write processed video clip paced at original video FPS so it plays as the original video
            if valid_frames:
                h, w = valid_frames[0][2].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(clip_path, fourcc, target_fps, (w, h))

                n_frames = len(valid_frames)
                repeats_list: List[int] = []

                for i in range(n_frames):
                    if n_frames == 1:
                        # Single frame: hold for clip duration
                        rep = max(1, int(round(target_fps * max(2.5, pre_seconds + post_seconds))))
                    elif i < n_frames - 1:
                        # 1. First attempt pacing by frame ID index delta (exact for original video)
                        fid_curr = valid_frames[i][0]
                        fid_next = valid_frames[i + 1][0]
                        df = fid_next - fid_curr
                        if 0 < df <= int(target_fps * 3.0):
                            rep = df
                        else:
                            # 2. Timestamp delta fallback (for live streams or after loop rewinds)
                            t_curr = valid_frames[i][1]
                            t_next = valid_frames[i + 1][1]
                            dt = t_next - t_curr
                            if 0.0 < dt <= 3.0:
                                rep = max(1, min(int(round(dt * target_fps)), int(target_fps * 3.0)))
                            else:
                                rep = 1
                    else:
                        # Final frame: repeat matching the median of previous frames
                        if repeats_list:
                            rep = max(1, int(np.median(repeats_list)))
                        else:
                            rep = 1
                    repeats_list.append(rep)

                # Ensure minimum duration of at least 2.5 seconds at original fps
                total_frames = sum(repeats_list)
                min_frames = int(target_fps * 2.5)
                if total_frames < min_frames and n_frames > 0:
                    scale = min_frames / max(1, total_frames)
                    repeats_list = [max(1, int(round(r * scale))) for r in repeats_list]

                total_written = 0
                for (fid, ts, img), rep in zip(valid_frames, repeats_list):
                    if img.shape[0] != h or img.shape[1] != w:
                        img = cv2.resize(img, (w, h))
                    for _ in range(rep):
                        writer.write(img)
                        total_written += 1

                writer.release()
                actual_duration = total_written / target_fps
                logger.info(
                    f"Saved evidence clip -> {clip_path} "
                    f"({total_written} frames @ {target_fps:.2f} FPS = {actual_duration:.2f}s, source frames: {n_frames})"
                )

            # Write JSON metadata
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(event_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to write evidence clip {clip_path}: {e}")
