"""
Visualization Renderer for Vision Pipeline Orchestrator.
Draws bounding boxes, persistent IDs, velocity vectors, ViTPose skeletons,
SAM segmentation masks, hazard geofences, and latency telemetry HUD.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from src.core.types import (
    FramePacket, BoundingBox, Detection, Track, Pose, Mask, Event,
    COCO_SKELETON_PAIRS
)


class FrameVisualizer:
    """Renders all typed intermediate objects onto video frames."""

    CLASS_COLORS = {
        "person": (60, 220, 60),       # Vibrant Green
        "worker": (60, 220, 60),       # Vibrant Green
        "forklift": (25, 140, 240),    # Industrial Orange/Amber
        "box": (200, 160, 60),         # Amber/Gold Box
        "pallet": (180, 120, 70),      # Wood Pallet
        "safety helmet": (0, 230, 255),# Cyan Helmet
        "machinery": (200, 160, 40),   # Steel Blue
        "car": (180, 80, 220),         # Purple
        "truck": (220, 100, 140),      # Magenta
    }

    POSTURE_COLORS = {
        "standing": (50, 220, 50),
        "bending": (30, 210, 240),
        "fallen": (30, 30, 240),
    }

    def __init__(self):
        pass

    def render(
        self,
        packet: FramePacket,
        show_detections: bool = True,
        show_tracks: bool = True,
        show_poses: bool = True,
        show_masks: bool = True,
        show_events: bool = True,
        show_hud: bool = True,
    ) -> np.ndarray:
        """Render annotations onto a copy of the packet image."""
        if packet.image is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        annotated = packet.image.copy()

        # 1. Render SAM Segmentation Masks (alpha blended)
        if show_masks and packet.masks:
            mask_layer = annotated.copy()
            for mask in packet.masks:
                if mask.polygon and len(mask.polygon) >= 3:
                    pts = np.array(mask.polygon, dtype=np.int32)
                    cv2.fillPoly(mask_layer, [pts], (180, 100, 230))
                    cv2.polylines(annotated, [pts], True, (220, 140, 255), 2)
            cv2.addWeighted(mask_layer, 0.35, annotated, 0.65, 0, annotated)

        # 2A. Render Raw RF-DETR Detection Boxes
        if show_detections and packet.detections:
            for det in packet.detections:
                x1, y1, x2, y2 = det.bbox.to_int_xyxy()
                color = self.CLASS_COLORS.get(det.class_name, (200, 200, 200))
                # Draw detection bounding box (solid 2px)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw detection label badge (Class + Confidence)
                det_label = f"{det.class_name.upper()} {det.score:.2f}"
                (tw, th), _ = cv2.getTextSize(det_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                badge_y1 = max(0, y1 - th - 6)
                badge_y2 = y1
                cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)
                cv2.putText(annotated, det_label, (x1 + 3, badge_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1)

        # 2B. Render ByteTrack Tracks (IDs, velocities, trajectory trails)
        if show_tracks and packet.tracks:
            for track in packet.tracks:
                x1, y1, x2, y2 = track.bbox.to_int_xyxy()
                color = self.CLASS_COLORS.get(track.class_name, (200, 200, 200))

                # Draw velocity arrow
                vx, vy = track.velocity
                if abs(vx) > 1.0 or abs(vy) > 1.0:
                    cx, cy = int(track.bbox.centroid[0]), int(track.bbox.centroid[1])
                    end_pt = (int(cx + vx * 0.1), int(cy + vy * 0.1))
                    cv2.arrowedLine(annotated, (cx, cy), end_pt, (0, 240, 255), 2, tipLength=0.3)

                # Draw persistent track ID and velocity badge
                vel_mag = math.hypot(vx, vy)
                track_label = f"ID:#{track.track_id} | {vel_mag:.1f}px/s"
                (tw, th), _ = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)

                # Position track badge below the bbox (or above if at the bottom of the screen)
                h_img = annotated.shape[0]
                if y2 + th + 8 < h_img:
                    by1 = y2 + 2
                    by2 = y2 + th + 8
                else:
                    by1 = max(0, y1 - th - 8)
                    by2 = y1

                cv2.rectangle(annotated, (x1, by1), (x1 + tw + 6, by2), (15, 20, 30), -1)
                cv2.rectangle(annotated, (x1, by1), (x1 + tw + 6, by2), color, 1)
                cv2.putText(annotated, track_label, (x1 + 3, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)

        # 3. Render ViTPose Postures & Skeletons (For all detected persons)
        if show_poses and packet.poses:
            for pose in packet.poses:
                # Find matching person bounding box (from tracks or detections)
                matching_bbox = None
                if packet.tracks:
                    matched = next((t for t in packet.tracks if str(t.track_id) == str(pose.track_id) and t.class_name in ("person", "worker")), None)
                    if matched:
                        matching_bbox = matched.bbox
                if matching_bbox is None and packet.detections:
                    matched = next((d for d in packet.detections if str(getattr(d, "id", "")) == str(pose.track_id) and d.class_name in ("person", "worker")), None)
                    if matched:
                        matching_bbox = matched.bbox
                if matching_bbox is None:
                    # Match by spatial proximity to any detected person/worker in this frame
                    person_boxes = [t.bbox for t in packet.tracks if t.class_name in ("person", "worker")] + [d.bbox for d in packet.detections if d.class_name in ("person", "worker")]
                    if person_boxes:
                        kps_valid = [kp for kp in pose.keypoints if kp.score >= 0.25]
                        if kps_valid:
                            cx = sum(kp.x for kp in kps_valid) / len(kps_valid)
                            cy = sum(kp.y for kp in kps_valid) / len(kps_valid)
                            matching_bbox = min(person_boxes, key=lambda b: (b.centroid[0] - cx)**2 + (b.centroid[1] - cy)**2)

                if matching_bbox is None:
                    continue

                bx1, by1, bx2, by2 = matching_bbox.to_xyxy()
                bw, bh = max(10.0, bx2 - bx1), max(10.0, by2 - by1)
                margin_x, margin_y = bw * 0.35, bh * 0.35
                min_x, max_x = bx1 - margin_x, bx2 + margin_x
                min_y, max_y = by1 - margin_y, by2 + margin_y

                p_color = self.POSTURE_COLORS.get(pose.posture_label, (255, 255, 255))

                # Skeleton bones (connecting valid joints inside person bounds)
                for idx1, idx2 in COCO_SKELETON_PAIRS:
                    if idx1 < len(pose.keypoints) and idx2 < len(pose.keypoints):
                        kp1 = pose.keypoints[idx1]
                        kp2 = pose.keypoints[idx2]
                        if kp1.score >= 0.25 and kp2.score >= 0.25:
                            if (min_x <= kp1.x <= max_x and min_y <= kp1.y <= max_y and
                                min_x <= kp2.x <= max_x and min_y <= kp2.y <= max_y):
                                pt1 = (int(kp1.x), int(kp1.y))
                                pt2 = (int(kp2.x), int(kp2.y))
                                cv2.line(annotated, pt1, pt2, p_color, 2)

                # Keypoint joints
                for kp in pose.keypoints:
                    if kp.score >= 0.25:
                        if min_x <= kp.x <= max_x and min_y <= kp.y <= max_y:
                            cv2.circle(annotated, (int(kp.x), int(kp.y)), 3, (0, 240, 255), -1)

                # Posture label banner
                if pose.posture_label in ("fallen", "bending"):
                    ibx1, iby1 = matching_bbox.to_int_xyxy()[:2]
                    p_text = f"POSTURE: {pose.posture_label.upper()} ({pose.spine_angle:.0f} deg)"
                    cv2.putText(annotated, p_text, (ibx1, iby1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2)

        # 4. Render Triggered Event Alerts
        if show_events and packet.events:
            for i, evt in enumerate(packet.events):
                banner_y = 70 + (i * 36)
                bg_color = (0, 0, 200) if evt.severity == "CRITICAL" else (0, 140, 240)
                cv2.rectangle(annotated, (20, banner_y - 24), (720, banner_y + 8), bg_color, -1)
                text = f"[ALERT] {evt.severity}: {evt.type} - {evt.description[:60]}"
                cv2.putText(annotated, text, (28, banner_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 5. Render Latency and Performance HUD
        if show_hud:
            h, w = annotated.shape[:2]
            # Dark semi-transparent header bar
            header_bar = annotated[0:40, 0:w].copy()
            cv2.rectangle(header_bar, (0, 0), (w, 40), (20, 24, 30), -1)
            cv2.addWeighted(header_bar, 0.85, annotated[0:40, 0:w], 0.15, 0, annotated[0:40, 0:w])

            total_lat = sum(packet.stage_latencies_ms.values())
            hud_text = (
                f"CAM: {packet.camera_id} | FRAME: {packet.frame_id:04d} | INTERVAL: 1.0s | "
                f"RF-DETR: {packet.stage_latencies_ms.get('rf_detr', 0.0):.1f}ms | "
                f"BYTE: {packet.stage_latencies_ms.get('bytetrack', 0.0):.1f}ms | "
                f"VITPOSE: {packet.stage_latencies_ms.get('vitpose', 0.0):.1f}ms | "
                f"CLIP: {packet.stage_latencies_ms.get('clip', 0.0):.1f}ms | "
                f"TOTAL: {total_lat:.1f}ms"
            )
            cv2.putText(annotated, hud_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1)

        return annotated
