"""
Vision Pipeline Orchestrator.
Coordinates video ingestion, frame sampling, model inference cascade,
rule evaluation, evidence persistence, and metrics collection.
"""

from __future__ import annotations
import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any

from src.core.types import FramePacket, Event
from src.core.config import OrchestratorConfig, CameraConfig
from src.core.buffer import CircularRingBuffer
from src.core.registry import ModelRegistry
from src.ingestion.reader import VideoStreamReader
from src.ingestion.sampler import FrameSampler
from src.rules.engine import EventRuleEngine
from src.rules.evidence import EvidencePersistenceManager
from src.pipeline.metrics import PipelineMetricsTracker
from src.pipeline.queue_manager import BoundedPipelineQueue
from src.ui.renderer import FrameVisualizer

# Import models to ensure registration in ModelRegistry
import os
import torch
if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(os.cpu_count() or 8)
import src.models

logger = logging.getLogger(__name__)


class VisionPipelineOrchestrator:
    """
    Main orchestrator managing the vision execution graph.
    Decoupled, event-driven, multi-camera, and configurable.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()

        # Telemetry & metrics
        self.metrics = PipelineMetricsTracker()

        # Visualization renderer
        self.visualizer = FrameVisualizer()

        # Evidence manager & rule engine
        default_fps = 25.0
        if self.config.cameras:
            default_fps = float(self.config.cameras[0].target_fps)

        self.evidence_manager = EvidencePersistenceManager(
            output_dir=self.config.storage_dir,
            fps=default_fps,
            visualizer=self.visualizer,
        )
        self.rule_engine = EventRuleEngine(evidence_manager=self.evidence_manager)

        # Vision model instances (lazily or eagerly initialized)
        self.detector = None
        self.tracker = None
        self.vitpose = None
        self.clip = None

        # Per-camera runtime state
        self.camera_contexts: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._worker_threads: List[threading.Thread] = []

        # Listeners for processed packets (e.g. UI stream, logging)
        self._packet_listeners: List[Callable[[FramePacket], None]] = []

    def initialize_models(self) -> None:
        """Instantiate all enabled vision models from ModelRegistry."""
        logger.info("Initializing Orchestrator Model Pipeline...")

        # 1. RF-DETR Detector
        if self.config.detector.enabled:
            det_opts = dict(self.config.detector.options)
            det_opts["confidence_threshold"] = self.config.detector.confidence_threshold
            self.detector = ModelRegistry.create(
                "rf_detr_small",
                device=self.config.detector.device,
                options=det_opts,
            )
            self.detector.initialize()

        # 2. ByteTrack Tracker
        if self.config.tracker.enabled:
            self.tracker = ModelRegistry.create(
                "bytetrack",
                device=self.config.tracker.device,
                options=self.config.tracker.options,
            )
            self.tracker.initialize()

        # 3. ViTPose Posture Analyzer
        if self.config.vitpose.enabled:
            self.vitpose = ModelRegistry.create(
                "vitpose_small",
                device=self.config.vitpose.device,
                options=self.config.vitpose.options,
            )
            self.vitpose.initialize()

        # 4. CLIP Zero-Shot Verifier
        if self.config.clip.enabled:
            self.clip = ModelRegistry.create(
                "clip",
                device=self.config.clip.device,
                options=self.config.clip.options,
            )
            self.clip.initialize()

        # Sync active model versions with RuleEngine
        versions = {}
        for m, name in [
            (self.detector, "detector"),
            (self.tracker, "tracker"),
            (self.vitpose, "pose"),
            (self.clip, "clip"),
        ]:
            if m:
                versions[name] = f"{m.name}:{m.version}"
        self.rule_engine.model_versions = versions

        logger.info(f"Models loaded: {list(versions.keys())}")

    def add_camera(self, cam_config: CameraConfig) -> None:
        """Register and initialize ingestion for a camera stream."""
        logger.info(f"Registering camera [{cam_config.camera_id}]: {cam_config.name}")
        reader = VideoStreamReader(
            source=cam_config.source,
            camera_id=cam_config.camera_id,
            loop_video=True,
        )
        orig_fps = reader.fps if getattr(reader, "fps", 0) > 0 else float(cam_config.target_fps)
        cam_fps = max(10, int(round(orig_fps)))
        ring_buffer = CircularRingBuffer(capacity=cam_config.ring_buffer_seconds * cam_fps)
        if getattr(reader, "fps", 0) > 0:
            self.evidence_manager.fps = reader.fps
        sampler = FrameSampler(
            reader=reader,
            ring_buffer=ring_buffer,
            target_fps=float(cam_config.target_fps),
            stride=cam_config.sampling_stride,
            target_resolution=tuple(cam_config.resolution) if cam_config.resolution else None,
            sample_interval_sec=getattr(cam_config, "sample_interval_sec", 0.0),
        )
        packet_queue = BoundedPipelineQueue(
            maxsize=self.config.max_queue_size,
            drop_oldest_on_full=self.config.drop_frames_on_backpressure,
        )

        camera_tracker = None
        if self.config.tracker.enabled:
            camera_tracker = ModelRegistry.create(
                "bytetrack",
                device=self.config.tracker.device,
                options=self.config.tracker.options,
            )
            camera_tracker.initialize()

        self.camera_contexts[cam_config.camera_id] = {
            "config": cam_config,
            "reader": reader,
            "sampler": sampler,
            "ring_buffer": ring_buffer,
            "queue": packet_queue,
            "tracker": camera_tracker,
        }

    def add_packet_listener(self, listener: Callable[[FramePacket], None]) -> None:
        self._packet_listeners.append(listener)

    def process_single_frame(self, packet: FramePacket, cam_config: CameraConfig, ring_buffer: CircularRingBuffer) -> FramePacket:
        """
        Execute full inference cascade on a single FramePacket.
        Adheres strictly to the architectural flowchart.
        """
        if packet.image is None:
            return packet

        frame = packet.image

        # 1. RF-DETR Detection Stage
        if cam_config.enable_detector and self.detector:
            t0 = time.perf_counter()
            packet.detections = self.detector.predict(frame)
            lat_det = (time.perf_counter() - t0) * 1000.0
            packet.stage_latencies_ms["rf_detr"] = round(lat_det, 2)
            self.metrics.record_stage_latency("rf_detr", lat_det)

        # 2. ByteTrack Tracking Stage
        cam_tracker = self.camera_contexts.get(packet.camera_id, {}).get("tracker", self.tracker)
        if cam_config.enable_tracker and cam_tracker and packet.detections:
            t0 = time.perf_counter()
            packet.tracks = cam_tracker.predict(packet.detections, timestamp=packet.timestamp)
            lat_track = (time.perf_counter() - t0) * 1000.0
            packet.stage_latencies_ms["bytetrack"] = round(lat_track, 2)
            self.metrics.record_stage_latency("bytetrack", lat_track)

        # 3. Parallel/Branching Stages: ViTPose & SAM
        # 3A. ViTPose Posture Analysis (on all detected persons)
        if cam_config.enable_vitpose and self.vitpose and (packet.tracks or packet.detections):
            person_targets = []
            seen_boxes = []

            if packet.tracks:
                for t in packet.tracks:
                    if t.class_name in ("person", "worker"):
                        person_targets.append(t)
                        seen_boxes.append(t.bbox)

            if packet.detections:
                for d in packet.detections:
                    if d.class_name in ("person", "worker"):
                        if not any(d.bbox.iou(sb) > 0.4 for sb in seen_boxes):
                            person_targets.append(d)
                            seen_boxes.append(d.bbox)

            if person_targets:
                t0 = time.perf_counter()
                packet.poses = self.vitpose.predict(frame, person_targets)
                lat_pose = (time.perf_counter() - t0) * 1000.0
                packet.stage_latencies_ms["vitpose"] = round(lat_pose, 2)
                self.metrics.record_stage_latency("vitpose", lat_pose)

        # 4. CLIP Semantic Verification (on tracks)
        if cam_config.enable_clip and self.clip and packet.tracks:
            t0 = time.perf_counter()
            clip_res = self.clip.predict(frame, packet.tracks)
            lat_clip = (time.perf_counter() - t0) * 1000.0
            packet.stage_latencies_ms["clip"] = round(lat_clip, 2)
            self.metrics.record_stage_latency("clip", lat_clip)

        # 5. Event & Safety Rule Engine
        t0 = time.perf_counter()
        cam_ctx = self.camera_contexts.get(packet.camera_id, {})
        reader = cam_ctx.get("reader")
        cam_orig_fps = None
        if reader and getattr(reader, "fps", 0) > 0:
            cam_orig_fps = reader.fps
        elif getattr(packet, "fps", None) and packet.fps > 0:
            cam_orig_fps = packet.fps

        events = self.rule_engine.process_packet(packet, ring_buffer, fps=cam_orig_fps)
        lat_rules = (time.perf_counter() - t0) * 1000.0
        packet.stage_latencies_ms["rules"] = round(lat_rules, 2)
        self.metrics.record_stage_latency("rules", lat_rules)

        if events:
            for _ in events:
                self.metrics.record_event_fired()

        # Update telemetry
        self.metrics.record_frame_processed()

        # 6. Render processed annotations onto frame (detections, tracks, poses, masks, alerts, hud)
        annotated_frame = self.visualizer.render(
            packet,
            show_detections=True,
            show_tracks=True,
            show_poses=True,
            show_masks=True,
            show_events=True,
            show_hud=True,
        )
        packet.annotated_image = annotated_frame

        # 7. Buffer processed frame into the circular ring buffer for evidence recording
        if ring_buffer is not None:
            ring_buffer.append(packet.frame_id, packet.timestamp, annotated_frame)

        # Notify downstream listeners (UI / websocket)
        for listener in self._packet_listeners:
            try:
                listener(packet)
            except Exception as e:
                logger.error(f"Error in packet listener: {e}")

        return packet

    def start(self) -> None:
        """Start all camera readers and processing threads."""
        self._running = True
        self.initialize_models()

        # Initialize cameras from config if none explicitly registered
        if not self.camera_contexts and self.config.cameras:
            for cam in self.config.cameras:
                self.add_camera(cam)

        for cam_id, ctx in self.camera_contexts.items():
            ctx["reader"].start()
            thread = threading.Thread(
                target=self._camera_processing_loop,
                args=(cam_id, ctx),
                name=f"Orchestrator-{cam_id}",
                daemon=True,
            )
            thread.start()
            self._worker_threads.append(thread)

        logger.info("Vision Pipeline Orchestrator started successfully.")

    def _camera_processing_loop(self, cam_id: str, ctx: Dict[str, Any]) -> None:
        """Processing loop for a single camera stream."""
        sampler: FrameSampler = ctx["sampler"]
        cam_config: CameraConfig = ctx["config"]
        ring_buffer: CircularRingBuffer = ctx["ring_buffer"]

        while self._running:
            packet = sampler.sample_packet()
            if packet is not None:
                self.process_single_frame(packet, cam_config, ring_buffer)
            else:
                time.sleep(0.005)

    def stop(self) -> None:
        """Stop all background workers and release camera resources."""
        self._running = False
        for cam_id, ctx in self.camera_contexts.items():
            ctx["reader"].stop()
        for t in self._worker_threads:
            if t.is_alive():
                t.join(timeout=1.0)
        logger.info("Vision Pipeline Orchestrator stopped.")

    def clean_db(self, clear_evidence: bool = True) -> Dict[str, Any]:
        """
        Clean and create new database for Milvus, clear old event logs,
        and reset tracking so track IDs restart from 1.
        """
        # 1. Clean Milvus vector database
        if self.clip and hasattr(self.clip, "clean_database"):
            self.clip.clean_database()
        elif hasattr(self, "vdb") and self.vdb:
            self.vdb.clean_and_recreate()

        # 2. Reset trackers
        if self.tracker and hasattr(self.tracker, "reset"):
            self.tracker.reset()
        for cam_ctx in self.camera_contexts.values():
            tr = cam_ctx.get("tracker")
            if tr and hasattr(tr, "reset"):
                tr.reset()

        # 3. Clear in-memory event history and telemetry counters
        if self.rule_engine:
            self.rule_engine.event_history.clear()
        self.metrics.total_events_fired = 0

        # 4. Remove persisted evidence files if requested
        deleted_count = 0
        if clear_evidence:
            evidence_dir = "data/evidence"
            if os.path.isdir(evidence_dir):
                for f in os.listdir(evidence_dir):
                    if f.startswith("evidence_") and (f.endswith(".json") or f.endswith(".jpg") or f.endswith(".mp4")):
                        try:
                            os.remove(os.path.join(evidence_dir, f))
                            deleted_count += 1
                        except Exception:
                            pass
            crops_dir = "data/evidence/crops"
            if os.path.isdir(crops_dir):
                for f in os.listdir(crops_dir):
                    if f.startswith("crop_") and f.endswith(".jpg"):
                        try:
                            os.remove(os.path.join(crops_dir, f))
                            deleted_count += 1
                        except Exception:
                            pass

        logger.info(f"[+] Cleaned database, reset trackers to 1, and removed {deleted_count} old evidence files.")
        return {"status": "ok", "deleted_evidence_count": deleted_count}
