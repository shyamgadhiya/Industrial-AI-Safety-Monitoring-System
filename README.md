# 🏭 Industrial AI Safety Monitoring System

> **Industrial AI Safety Monitoring System** — Real-time multi-camera video analysis for warehouse and factory environments using state-of-the-art transformer-based computer vision models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [AI Model Pipeline](#ai-model-pipeline)
- [Safety Rules & Alerts](#safety-rules--alerts)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Web Dashboard](#web-dashboard)
- [Evidence System](#evidence-system)
- [Database Management](#database-management)
- [CLI Reference](#cli-reference)
- [Class Mappings](#class-mappings)

---

## Overview

The **Industrial AI Safety Monitoring System** is a production-grade, multi-stage computer vision pipeline designed for real-time industrial safety monitoring. It ingests live or recorded video feeds from multiple cameras, runs a cascade of AI models, evaluates domain-specific safety rules, and automatically saves forensic evidence when hazardous events are detected.

The system is built around a custom fine-tuned **RF-DETR Small** transformer detection model, augmented with **ByteTrack** multi-object tracking, **ViTPose** posture analysis, and **CLIP** zero-shot semantic scene verification — all coordinated by a central orchestrator engine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Video Source(s) (MP4 / RTSP)              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    VideoStreamReader
                    (per-camera thread)
                            │
                     FrameSampler
                   (target FPS / stride)
                            │
              ┌─────────────▼────────────┐
              │   Circular Ring Buffer   │  ← pre-event frames
              └─────────────┬────────────┘
                            │
              ┌─────────────▼────────────┐
              │  RF-DETR Small Detector  │  Stage 1 – Detection
              │  (custom warehouse ckpt) │
              └─────────────┬────────────┘
                            │
              ┌─────────────▼────────────┐
              │   ByteTrack Tracker      │  Stage 2 – Tracking
              │   (per-camera instance)  │
              └──────┬──────────┬────────┘
                     │          │
           ┌─────────▼──┐  ┌───▼──────────┐
           │  ViTPose   │  │  CLIP Model  │  Stage 3 – Posture & Semantics
           │  (Posture) │  │  (Zero-Shot) │
           └─────────┬──┘  └───┬──────────┘
                     │         │
              ┌──────▼─────────▼──────┐
              │  Safety Rule Engine   │  Stage 4 – Event Detection
              └──────────┬────────────┘
                         │
              ┌──────────▼────────────┐
              │  Evidence Manager     │  Stage 5 – Persistence
              │  (MP4 + JSON + JPEG)  │
              └──────────┬────────────┘
                         │
              ┌──────────▼────────────┐
              │  Web Dashboard (HUD)  │  Stage 6 – Monitoring UI
              └───────────────────────┘
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Custom RF-DETR Detector** | Fine-tuned transformer detector for warehouse objects (workers, forklifts, boxes, pallets, helmets) |
| 👤 **Multi-Object Tracking** | ByteTrack with persistent, sequentially-assigned track IDs starting from 1 |
| 🧍 **Posture Analysis** | ViTPose estimates human skeletal poses to detect fallen workers |
| 🔍 **Semantic Verification** | CLIP zero-shot classifier validates safety conditions (PPE compliance, fall detection) |
| 🚨 **Safety Rule Engine** | Configurable rules that trigger forensic evidence capture on safety violations |
| 📹 **Evidence Recording** | Auto-saves annotated MP4 clips, JPEG snapshots, and JSON metadata for each event |
| 🗄️ **Vector Search** | Milvus-Lite vector database enables semantic similarity search over all recorded incidents |
| 📊 **Live Dashboard** | Browser-based monitoring dashboard with real-time video feed, event timeline, and metrics |
| 🧹 **DB Management** | One-click or CLI reset of vector database, event logs, and evidence files |
| 🎥 **Multi-Camera** | Independent per-camera processing threads with shared model instances |

---

## AI Model Pipeline

### Stage 1 — RF-DETR Small Detector (`src/models/detector_rfdetr.py`)

A custom **Real-time Detection TRansformer (RF-DETR Small)** model fine-tuned on warehouse imagery.

- **Checkpoint**: `src/models/rfdetr_checkpoint_best_total.pth` (~122 MB)
- **Target Classes**: `worker` and `forklift` only (other classes filtered out)
- **Per-class confidence thresholds**:
  - Worker: `0.30`
  - Forklift: `0.50` (stricter to reduce false positives)
- **False-positive suppression**:
  - **Geometric filter** — forklifts must satisfy aspect-ratio >= 1.35, narrow profile (<= 25% frame width), and proper scale
  - **Temporal consistency** — a detection must appear in >= 2 of the last 4 frames (grid-cell sliding window) before being emitted

### Stage 2 — ByteTrack Tracker (`src/models/tracker_bytetrack.py`)

Multi-object tracker based on **ByteTrack** (via the `supervision` library).

- Per-camera isolated tracker instances
- Track IDs always start at `1` and increment sequentially
- IDs can be fully reset (e.g., after DB clean)
- Configurable: `track_buffer=60`, `match_thresh=0.60`, `track_thresh=0.25`

### Stage 3A — ViTPose Posture Analyzer (`src/models/pose_vitpose.py`)

Skeleton-based posture analysis using **ViTPose-Base** from Hugging Face.

- Model: `usyd-community/vitpose-base-simple`
- Detects: spine angle, torso orientation, keypoint positions
- Outputs: `posture_label` (`upright` / `fallen`), `fall_confidence`, `spine_angle`
- Fall threshold: spine angle > 60 degrees, rapid drop > 45 pixels

### Stage 3B — CLIP Semantic Verifier (`src/models/hf_clip.py`, `zero_shot_clip.py`)

**OpenAI CLIP** (`clip-vit-base-patch32`) for zero-shot classification of cropped worker/forklift regions.

- Semantic queries (configurable):
  - `"worker wearing high visibility safety vest and hardhat"`
  - `"worker not wearing safety vest"`
  - `"worker fallen on factory floor"`
  - `"forklift carrying heavy load"`
- Embeds crops into 512-dim vectors and stores them in **Milvus-Lite** for semantic search
- Database path: `data/milvus_orchestrator.db`

---

## Safety Rules & Alerts

The **Rule Engine** (`src/rules/engine.py`) evaluates `FramePacket` objects against configurable safety rules.

### Active Rules

| Rule ID | Name | Severity | Trigger Condition |
|---------|------|----------|-------------------|
| `RULE_001` | Worker-Forklift Proximity Violation | **CRITICAL** | A worker is detected within 180px of a forklift |
| `RULE_002` | Worker Fall / Collapse Detected | **CRITICAL** | ViTPose `posture_label == "fallen"` with confidence >= 0.65 |

Each rule has a cooldown period (5–8 seconds) to prevent alert flooding.

### Evidence on Alert

When a rule fires, the **Evidence Persistence Manager** (`src/rules/evidence.py`) automatically:
1. Captures annotated **pre-event frames** from the ring buffer (3 seconds before)
2. Continues recording **post-event frames** (3 seconds after)
3. Exports an **annotated MP4 video clip** (at source FPS)
4. Saves a **JPEG snapshot** of the peak event frame
5. Writes a **JSON metadata file** with timestamps, rule info, tracks, and model versions

---

## Project Structure

```
Project 8 Vision Pipeline Orchestrator/
│
├── main.py                         # CLI entrypoint (run / dashboard / benchmark / clean-db)
├── requirements.txt                # Python dependencies
├── configs/
│   └── default_pipeline.yaml       # Full pipeline configuration
│
├── src/
│   ├── core/
│   │   ├── config.py               # Pydantic config models (OrchestratorConfig, CameraConfig)
│   │   ├── types.py                # Core data types (FramePacket, Detection, Track, Pose, Event)
│   │   ├── registry.py             # ModelRegistry – factory for vision models
│   │   ├── buffer.py               # Circular ring buffer for pre-event frame storage
│   │   └── base_model.py           # Abstract base for all vision models
│   │
│   ├── models/
│   │   ├── detector_rfdetr.py      # RF-DETR Small warehouse detector
│   │   ├── tracker_bytetrack.py    # ByteTrack multi-object tracker
│   │   ├── pose_vitpose.py         # ViTPose human posture estimator
│   │   ├── hf_clip.py              # CLIP Hugging Face wrapper
│   │   ├── zero_shot_clip.py       # CLIP zero-shot + Milvus vector store
│   │   ├── base.py                 # BaseVisionModel interface
│   │   └── rfdetr_checkpoint_best_total.pth  # Custom trained checkpoint
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py         # VisionPipelineOrchestrator – main execution graph
│   │   ├── metrics.py              # Pipeline telemetry and FPS tracking
│   │   └── queue_manager.py        # Bounded FIFO queue with backpressure support
│   │
│   ├── ingestion/
│   │   ├── reader.py               # VideoStreamReader (threaded OpenCV reader)
│   │   ├── sampler.py              # FrameSampler (FPS control, resolution, stride)
│   │   └── synthetic_feed.py       # Synthetic factory video generator (for testing)
│   │
│   ├── rules/
│   │   ├── engine.py               # EventRuleEngine (evaluates rules per packet)
│   │   ├── evidence.py             # EvidencePersistenceManager (MP4/JSON/JPEG export)
│   │   └── safety_rules.py         # WorkerForkliftProximityRule, WorkerFallErgonomicsRule
│   │
│   └── ui/
│       ├── dashboard_server.py     # HTTP web server (live video + dashboard)
│       └── renderer.py             # FrameVisualizer (annotation overlay + HUD)
│
├── data/
│   ├── samples/                    # Input video files (sample_feed1.mp4, sample_feed2.mp4)
│   ├── evidence/                   # Auto-generated: annotated clips, snapshots, JSON logs
│   │   └── crops/                  # CLIP crop thumbnails per tracked object
│   └── milvus_orchestrator.db      # Milvus-Lite vector database
│
└── tests/                          # Unit and integration tests
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Windows / Linux / macOS
- (Optional) CUDA-enabled GPU for accelerated inference

### Setup

```bash
# 1. Clone or navigate to the project directory
cd "Project 8 Vision Pipeline Orchestrator"

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install RF-DETR package (required for detection)
pip install rfdetr

# 5. Install PyTorch (CPU or CUDA — see https://pytorch.org)
pip install torch torchvision

# 6. Install ViTPose dependencies (Hugging Face Transformers)
pip install transformers accelerate

# 7. Install Milvus-Lite for vector storage
pip install pymilvus

# 8. Install ByteTrack / Supervision for tracking
pip install supervision
```

### Place Sample Videos

Put your source video files in `data/samples/`:
```
data/samples/sample_feed1.mp4
data/samples/sample_feed2.mp4
```

Or generate a synthetic test video:
```bash
python main.py generate-sample
```

---

## Configuration

All settings are managed through `configs/default_pipeline.yaml`.

### Key Configuration Sections

```yaml
# Global pipeline settings
pipeline_name: "Industrial Vision Pipeline Orchestrator"
max_queue_size: 30
drop_frames_on_backpressure: true
storage_dir: "data/evidence"

# RF-DETR Detector
detector:
  enabled: true
  device: "cpu"           # Use "cuda" for GPU
  confidence_threshold: 0.30
  options:
    checkpoint_path: "src/models/rfdetr_checkpoint_best_total.pth"
    class_thresholds:
      worker: 0.25
      forklift: 0.30     # Stricter threshold to reduce false positives
    forklift_window: 4   # Temporal consistency window (frames)
    forklift_min_hits: 2 # Hits required to confirm a detection

# ByteTrack Tracker
tracker:
  enabled: true
  options:
    tracker_type: "bytetrack"
    track_buffer: 60     # Frames to keep lost tracks alive
    match_thresh: 0.60

# ViTPose Posture Analyzer
vitpose:
  enabled: true
  device: "auto"         # Auto-detects GPU/CPU
  options:
    model_id: "usyd-community/vitpose-base-simple"
    fall_angle_threshold: 60.0
    rapid_drop_px: 45.0

# CLIP Semantic Verifier
clip:
  enabled: true
  options:
    model_id: "openai/clip-vit-base-patch32"
    db_path: "data/milvus_orchestrator.db"

# Camera Sources
cameras:
  - camera_id: "cam_feed_01"
    source: "data/samples/sample_feed1.mp4"
    target_fps: 25.0
    ring_buffer_seconds: 10   # Pre-event recording buffer
    evidence_pre_seconds: 3.0
    evidence_post_seconds: 3.0
```

---

## Usage

### Run with Live Dashboard (Recommended)

```bash
python main.py dashboard
```

Opens the orchestrator and serves the web dashboard at **http://127.0.0.1:8501**

### Run in Console Mode (No Browser)

```bash
python main.py run
```

Prints telemetry (FPS, frame count, events, latency) every 2 seconds to the terminal.

### Run with Custom Config

```bash
python main.py dashboard --config configs/my_config.yaml --port 9000
```

### Start with a Clean Database

```bash
python main.py dashboard --clean-db
# or
python main.py run --clean-db
```

---

## Web Dashboard
<img width="1892" height="926" alt="Screenshot 2026-10-08 113800" src="https://github.com/user-attachments/assets/7c2b122f-ad2e-44d0-8733-e4af8d499464" />

Access at **http://127.0.0.1:8501** after starting with `dashboard` command.

### Dashboard Panels

| Panel | Description |
|-------|-------------|
| 📹 **Live Video Feed** | Annotated real-time stream with bounding boxes, track IDs, skeleton overlays, and HUD |
| 📊 **Pipeline Metrics** | FPS, total frames processed, events fired, per-stage latency |
| ⚠️ **Event Timeline** | Chronological log of all safety alerts with severity badges |
| 🗂️ **Evidence Browser** | View all saved evidence: MP4 clips (with video player), JPEG snapshots, JSON metadata |
| 🔍 **Semantic Search** | Search past incidents using natural language queries (powered by CLIP + Milvus) |
| 🧹 **Clean DB Button** | Reset the vector database, clear event logs, and remove old evidence files |

### Live Annotations (HUD Overlay)

The rendered video frame includes:
- **Bounding boxes** — color-coded by class (green = worker, orange = forklift)
- **Track IDs** — persistent identity labels per detected object
- **Skeleton overlay** — joint keypoints and bone connections for detected workers
- **Alert banners** — flashing red overlays when safety rules fire
- **Stage latency HUD** — real-time per-stage inference timing in the corner

---

## Evidence System

When a safety rule fires, evidence is automatically saved to `data/evidence/`:

```
data/evidence/
├── evidence_RULE001_cam_feed_01_1728290400.mp4     # Annotated video clip
├── evidence_RULE001_cam_feed_01_1728290400.jpg     # Peak-frame JPEG snapshot
├── evidence_RULE001_cam_feed_01_1728290400.json    # Metadata (tracks, rule, timestamps)
└── crops/
    └── crop_worker_1_1728290400.jpg                # CLIP crop thumbnail
```

### Evidence JSON Structure

```json
{
  "event_type": "UNSAFE_WORKER_FORKLIFT_PROXIMITY",
  "timestamp": 1728290400.123,
  "camera_id": "cam_feed_01",
  "severity": "CRITICAL",
  "description": "Unsafe proximity alert: Worker #1 is only 142.3px from Forklift #2",
  "track_ids": [1, 2],
  "confidence_values": { "distance_px": 142.3, "urgency": 0.789 },
  "model_versions": { "detector": "rfdetrSmall:v1.2", "tracker": "bytetrack:v1.0" }
}
```

---

## Database Management

The system uses **Milvus-Lite** (`data/milvus_orchestrator.db`) to store CLIP embeddings of all detected objects for semantic similarity search.

### Reset Database

**Via Dashboard**: Click the **🧹 Clean DB** button in the dashboard timeline panel.

**Via CLI**:
```bash
python main.py clean-db
```

**At startup** (fresh run):
```bash
python main.py dashboard --clean-db
```

### What gets cleaned:
- Milvus vector collection (`vision_events`) — dropped and recreated empty
- All `evidence_*.mp4`, `evidence_*.jpg`, `evidence_*.json` files
- All `crop_*.jpg` files in `data/evidence/crops/`
- In-memory event history and telemetry counters
- Tracker state — track IDs restart from `1`

---

## CLI Reference

```
python main.py <command> [options]

Commands:
  dashboard          Run orchestrator with live web dashboard
    --config PATH    Path to pipeline YAML config (default: configs/default_pipeline.yaml)
    --port INT       Dashboard port (default: 8501)
    --clean-db       Clean and reset DB before starting

  run                Run orchestrator in headless console mode
    --config PATH    Path to pipeline YAML config
    --clean-db       Clean and reset DB before starting

  clean-db           Clean Milvus DB, evidence files, and event logs
  benchmark          Benchmark all vision model stages individually
  generate-sample    Generate a synthetic factory test video
```

---

## Class Mappings

The custom RF-DETR checkpoint was trained on the following warehouse classes:

| Class ID | Class Name | Detected by Orchestrator |
|----------|------------|--------------------------|
| 0 | box | Filtered out |
| 1 | worker | Yes |
| 2 | forklift | Yes |
| 3 | pallet | Filtered out |
| 4 | safety helmet | Filtered out |

> Only `worker` and `forklift` are passed to downstream stages (tracker, pose estimator, rule engine). Other detected classes are silently discarded.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `rfdetr` | RF-DETR transformer detection model |
| `torch` / `torchvision` | PyTorch deep learning backend |
| `transformers` | ViTPose and CLIP model loading (Hugging Face) |
| `supervision` | ByteTrack tracking, annotation utilities |
| `pymilvus` | Milvus-Lite vector database |
| `opencv-python` | Video I/O and frame processing |
| `numpy` | Array operations |
| `pydantic` | Configuration schema validation |
| `pyyaml` | YAML config parsing |
| `scipy` | Distance calculations (proximity rules) |
| `pillow` | Image processing for CLIP crops |
| `rich` | Beautiful terminal output and tables |
| `requests` | HTTP client utilities |
| `tqdm` | Progress bars |
| `matplotlib` | Visualization utilities |

---

*Built for Project 8 — Industrial Vision Intelligence System*
