"""
Configuration loader and Pydantic models for cameras, pipeline stages, and rules.
Allows different cameras to enable different model stages.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field


class ModelStageConfig(BaseModel):
    enabled: bool = True
    device: str = "cpu"  # "cpu", "cuda", "auto"
    confidence_threshold: float = 0.40
    batch_size: int = 1
    options: Dict[str, Any] = Field(default_factory=dict)


class CameraConfig(BaseModel):
    camera_id: str = "cam_01"
    name: str = "Factory Main Corridor"
    source: str = "0"  # RTSP URL, video path, or webcam index
    target_fps: float = 25.0  # target FPS for inference sampling (e.g. 25-30 FPS)
    sample_interval_sec: float = 0.0  # 0.0 = process every frame continuously
    sampling_stride: int = 1  # Sample every Nth frame
    resolution: List[int] = Field(default_factory=lambda: [1280, 720])
    
    # Per-camera stage toggles
    enable_detector: bool = True
    enable_tracker: bool = True
    enable_vitpose: bool = True
    enable_sam: bool = False  # By default, SAM runs conditionally or on incident
    enable_clip: bool = True
    
    # Ring buffer settings
    ring_buffer_seconds: int = 10
    evidence_pre_seconds: float = 4.0
    evidence_post_seconds: float = 4.0


class SafetyRuleConfig(BaseModel):
    rule_id: str
    name: str
    enabled: bool = True
    severity: str = "WARNING"  # INFO, WARNING, CRITICAL
    type: str  # "proximity", "ergonomics", "hazard_zone"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorConfig(BaseModel):
    pipeline_name: str = "Factory Safety Orchestrator"
    max_queue_size: int = 30
    drop_frames_on_backpressure: bool = True
    storage_dir: str = "data/evidence"
    
    # Model stage configs
    detector: ModelStageConfig = Field(default_factory=lambda: ModelStageConfig(
        options={"model_type": "rf-detr-small", "classes": ["person", "forklift", "truck"]}
    ))
    tracker: ModelStageConfig = Field(default_factory=lambda: ModelStageConfig(
        options={"track_thresh": 0.40, "track_buffer": 30, "match_thresh": 0.6}
    ))
    vitpose: ModelStageConfig = Field(default_factory=lambda: ModelStageConfig(
        options={"model_type": "vitpose-s", "fall_threshold_deg": 65.0}
    ))
    sam: Optional[ModelStageConfig] = None
    clip: ModelStageConfig = Field(default_factory=lambda: ModelStageConfig(
        options={"model_type": "vit-b-32", "labels": ["worker with helmet and vest", "worker without helmet", "unsafe posture"]}
    ))
    
    cameras: List[CameraConfig] = Field(default_factory=lambda: [CameraConfig()])
    rules: List[SafetyRuleConfig] = Field(default_factory=list)


def load_config_from_yaml(filepath: str) -> OrchestratorConfig:
    """Load and validate orchestrator config from a YAML file."""
    if not os.path.exists(filepath):
        # Return default config if file does not exist yet
        return OrchestratorConfig()
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        
    return OrchestratorConfig(**data)


def save_config_to_yaml(config: OrchestratorConfig, filepath: str) -> None:
    """Save an OrchestratorConfig object to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
