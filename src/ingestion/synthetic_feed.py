"""
Synthetic factory video generator utility for testing and demonstration.
Generates a realistic test video sequence featuring:
- Factory floor with painted safety lanes and hazard zone
- Moving worker walking toward a moving forklift
- Worker bending/kneeling to trigger ergonomic/fall detection
- Worker crossing the hazard perimeter to trigger geofence alarm
"""

import os
import cv2
import numpy as np


def generate_synthetic_factory_video(
    output_path: str = "data/samples/sample_factory_feed.mp4",
    num_frames: int = 180,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Generate and write a synthetic factory scenario MP4 video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # 1. Base Factory Floor (concrete gray)
        img = np.full((height, width, 3), (180, 185, 190), dtype=np.uint8)

        # 2. Draw Factory Machinery / Conveyor Belt on left
        cv2.rectangle(img, (50, 80), (350, 640), (80, 85, 90), -1)
        cv2.putText(img, "CONVEYOR CELL A", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)

        # 3. Hazard Zone (Yellow striped area on right)
        hazard_pts = np.array([[int(width * 0.65), int(height * 0.40)],
                               [int(width * 0.95), int(height * 0.40)],
                               [int(width * 0.95), int(height * 0.85)],
                               [int(width * 0.65), int(height * 0.85)]], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [hazard_pts], (40, 190, 230))  # Amber/yellow
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.polylines(img, [hazard_pts], True, (0, 140, 255), 3)
        cv2.putText(img, "HAZARD ZONE - FORKLIFT ONLY", (int(width * 0.66), int(height * 0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)

        # 4. Moving Forklift (Orange chassis, yellow mast, black wheels)
        # Forklift moves along the right lane horizontally & vertically
        forklift_progress = (frame_idx % 120) / 120.0
        fk_x = int(width * 0.80 - forklift_progress * 80)
        fk_y = int(height * 0.55 + np.sin(frame_idx * 0.05) * 40)
        
        # Forklift body
        cv2.rectangle(img, (fk_x - 60, fk_y - 40), (fk_x + 60, fk_y + 35), (25, 115, 230), -1) # Orange body
        # Forklift cabin
        cv2.rectangle(img, (fk_x - 30, fk_y - 90), (fk_x + 30, fk_y - 40), (40, 40, 40), -1)
        # Mast & forks
        cv2.line(img, (fk_x - 60, fk_y - 80), (fk_x - 60, fk_y + 35), (200, 200, 200), 6)
        cv2.line(img, (fk_x - 90, fk_y + 35), (fk_x - 60, fk_y + 35), (160, 160, 160), 6)
        # Wheels
        cv2.circle(img, (fk_x - 40, fk_y + 45), 18, (30, 30, 30), -1)
        cv2.circle(img, (fk_x + 40, fk_y + 45), 18, (30, 30, 30), -1)
        # Amber beacon light
        beacon_color = (0, 240, 255) if (frame_idx // 5) % 2 == 0 else (0, 100, 180)
        cv2.circle(img, (fk_x, fk_y - 95), 8, beacon_color, -1)
        cv2.putText(img, "FORKLIFT #2", (fk_x - 55, fk_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

        # 5. Moving Worker (Person walking toward forklift, then bending / slipping)
        worker_progress = min(1.0, frame_idx / 140.0)
        w_x = int(450 + worker_progress * 420)
        w_y = int(400 + np.sin(frame_idx * 0.1) * 10)

        # In later frames (frame 110+), simulate worker bending / falling
        is_falling = frame_idx > 120
        is_bending = 90 <= frame_idx <= 120

        if is_falling:
            # Person fallen horizontally on the floor
            cv2.ellipse(img, (w_x, w_y + 40), (45, 18), 0, 0, 360, (20, 120, 240), -1) # High-vis orange vest
            cv2.circle(img, (w_x - 50, w_y + 40), 14, (70, 150, 230), -1) # Head / hardhat
            cv2.line(img, (w_x - 30, w_y + 40), (w_x + 45, w_y + 40), (120, 50, 30), 10) # Body
            cv2.putText(img, "WORKER #1 (FALLEN)", (w_x - 70, w_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
        elif is_bending:
            # Person bending over (awkward ergonomics)
            cv2.ellipse(img, (w_x, w_y - 10), (25, 35), 45, 0, 360, (0, 200, 100), -1) # Green safety vest
            cv2.circle(img, (w_x + 25, w_y - 25), 14, (0, 230, 255), -1) # Yellow hardhat
            cv2.line(img, (w_x, w_y + 20), (w_x - 10, w_y + 80), (140, 60, 40), 8) # Legs
            cv2.putText(img, "WORKER #1 (BENDING)", (w_x - 50, w_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
        else:
            # Person standing and walking upright
            # Head + Hardhat
            cv2.circle(img, (w_x, w_y - 75), 16, (0, 220, 240), -1)
            # Torso + High-vis safety vest
            cv2.rectangle(img, (w_x - 22, w_y - 55), (w_x + 22, w_y + 15), (0, 190, 80), -1)
            # Vest reflective stripes
            cv2.line(img, (w_x - 20, w_y - 25), (w_x + 20, w_y - 25), (240, 240, 240), 3)
            # Legs
            leg_offset = int(np.sin(frame_idx * 0.4) * 12)
            cv2.line(img, (w_x - 10, w_y + 15), (w_x - 12 + leg_offset, w_y + 75), (80, 60, 50), 7)
            cv2.line(img, (w_x + 10, w_y + 15), (w_x + 12 - leg_offset, w_y + 75), (80, 60, 50), 7)
            cv2.putText(img, "WORKER #1", (w_x - 45, w_y - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

        # 6. Timestamp & HUD Overlay
        cv2.putText(img, f"CAMERA: FACTORY_FLOOR_CAM_01 | FRAME: {frame_idx:04d} | SIMULATED RTSP FEED",
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        out.write(img)

    out.release()
    return output_path


if __name__ == "__main__":
    path = generate_synthetic_factory_video()
    print(f"Generated synthetic test video at: {path}")
