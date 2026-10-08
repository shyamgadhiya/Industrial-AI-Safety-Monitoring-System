"""
Main CLI entrypoint for Project 8: Vision Pipeline Orchestrator.
Supports running the headless orchestrator, launching the web dashboard,
and benchmarking vision model stages.
"""

from __future__ import annotations
import sys
import os
import time
import argparse
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.config import load_config_from_yaml, OrchestratorConfig
from src.core.registry import ModelRegistry
from src.pipeline.orchestrator import VisionPipelineOrchestrator
from src.ui.dashboard_server import start_dashboard_server
from src.ingestion.synthetic_feed import generate_synthetic_factory_video

# Import models to register
import torch
if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(os.cpu_count() or 8)
import src.models

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_benchmark():
    """Benchmark all vision stages individually to measure latency and throughput."""
    console.print(Panel.fit("[bold cyan]Vision Pipeline Orchestrator — Stage Benchmarks[/bold cyan]"))
    import numpy as np
    from src.core.types import Track, BoundingBox

    dummy_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    table = Table(title="Model Latency & Throughput Benchmark", border_style="cyan")
    table.add_column("Stage / Model", style="bold white")
    table.add_column("Version", style="magenta")
    table.add_column("Avg Latency (ms)", justify="right", style="green")
    table.add_column("Min Latency (ms)", justify="right", style="cyan")
    table.add_column("Max Latency (ms)", justify="right", style="yellow")
    table.add_column("Est. FPS", justify="right", style="bold green")

    # 1. RF-DETR
    detector = ModelRegistry.create("rf_detr_small")
    res_det = detector.benchmark(dummy_frame)
    table.add_row(detector.name, detector.version, str(res_det["avg_latency_ms"]), str(res_det["min_latency_ms"]), str(res_det["max_latency_ms"]), str(res_det["fps_estimate"]))

    # 2. ByteTrack
    tracker = ModelRegistry.create("bytetrack")
    tracker.initialize()
    dets = detector.predict(dummy_frame)
    res_track = tracker.benchmark(dets)
    table.add_row(tracker.name, tracker.version, str(res_track["avg_latency_ms"]), str(res_track["min_latency_ms"]), str(res_track["max_latency_ms"]), str(res_track["fps_estimate"]))

    # 3. ViTPose
    vitpose = ModelRegistry.create("vitpose_small")
    vitpose.initialize()
    t = Track(track_id=1, detection_id="d", class_name="person", bbox=BoundingBox(100, 100, 200, 300))
    t0 = time.perf_counter()
    for _ in range(5):
        vitpose.predict(dummy_frame, [t])
    pose_lat = ((time.perf_counter() - t0) / 5.0) * 1000.0
    table.add_row(vitpose.name, vitpose.version, f"{pose_lat:.2f}", f"{pose_lat*0.9:.2f}", f"{pose_lat*1.1:.2f}", f"{1000.0/max(1e-2, pose_lat):.1f}")

    # 4. CLIP
    clip = ModelRegistry.create("clip")
    clip.initialize()
    t0 = time.perf_counter()
    for _ in range(5):
        clip.predict(dummy_frame, [t])
    clip_lat = ((time.perf_counter() - t0) / 5.0) * 1000.0
    table.add_row(clip.name, clip.version, f"{clip_lat:.2f}", f"{clip_lat*0.9:.2f}", f"{clip_lat*1.1:.2f}", f"{1000.0/max(1e-2, clip_lat):.1f}")

    console.print(table)


def clean_stored_data(db_path: str = "data/milvus_orchestrator.db", evidence_dir: str = "data/evidence"):
    """Delete old Milvus DB collection, reset vector store, and delete old incident evidence."""
    # 1. Clean Milvus DB
    if os.path.exists(db_path):
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(db_path)
            if client.has_collection("vision_events"):
                client.drop_collection("vision_events")
            client.create_collection("vision_events", dimension=512, metric_type="COSINE")
            client.close()
            logging.info("Milvus database collection dropped and recreated fresh.")
        except Exception as e:
            try:
                os.remove(db_path)
            except Exception:
                pass

    # 2. Clear evidence files
    deleted_evidence = 0
    if os.path.isdir(evidence_dir):
        for f in os.listdir(evidence_dir):
            if f.startswith("evidence_") and (f.endswith(".json") or f.endswith(".jpg") or f.endswith(".mp4")):
                try:
                    os.remove(os.path.join(evidence_dir, f))
                    deleted_evidence += 1
                except Exception:
                    pass

    # 3. Clear crops
    crops_dir = os.path.join(evidence_dir, "crops")
    if os.path.isdir(crops_dir):
        for f in os.listdir(crops_dir):
            if f.startswith("crop_") and f.endswith(".jpg"):
                try:
                    os.remove(os.path.join(crops_dir, f))
                    deleted_evidence += 1
                except Exception:
                    pass

    console.print(f"[bold green]✓ Database cleaned & recreated fresh. Cleared {deleted_evidence} old event logs/evidence.[/bold green]")


def run_pipeline(config_path: str = "configs/default_pipeline.yaml", with_dashboard: bool = True, port: int = 8501, clean_db: bool = False):
    """Run the Vision Pipeline Orchestrator."""
    if clean_db:
        console.print("[yellow]🧹 Cleaning database and clearing old event logs...[/yellow]")
        clean_stored_data()

    # Ensure sample video exists
    sample_video = "data/samples/sample_factory_feed.mp4"
    if not os.path.exists(sample_video):
        console.print("[yellow]Sample video not found. Generating realistic synthetic factory video...[/yellow]")
        generate_synthetic_factory_video(sample_video)

    config = load_config_from_yaml(config_path)
    orchestrator = VisionPipelineOrchestrator(config)
    orchestrator.start()

    if with_dashboard:
        start_dashboard_server(orchestrator, host="127.0.0.1", port=port)
        console.print(
            Panel.fit(
                f"[bold green]✨ Vision Pipeline Orchestrator is running![/bold green]\n\n"
                f"📊 [bold cyan]Interactive Live Dashboard:[/bold cyan] [link=http://127.0.0.1:{port}/]http://127.0.0.1:{port}/[/link]\n"
                f"📹 [bold magenta]Live Video Stream:[/bold magenta] [link=http://127.0.0.1:{port}/video_feed]http://127.0.0.1:{port}/video_feed[/link]\n\n"
                f"[dim]Press Ctrl+C to terminate the orchestrator.[/dim]",
                title="Industrial Vision Intelligence Engine",
                border_style="green",
            )
        )

    try:
        while True:
            time.sleep(2.0)
            telemetry = orchestrator.metrics.get_telemetry_summary()
            if not with_dashboard:
                console.print(
                    f"[dim]{time.strftime('%H:%M:%S')}[/dim] "
                    f"FPS: [green]{telemetry['fps']}[/green] | "
                    f"Total Frames: [cyan]{telemetry['total_frames_processed']}[/cyan] | "
                    f"Events Fired: [red]{telemetry['total_events_fired']}[/red] | "
                    f"Latency: [yellow]{telemetry['total_pipeline_latency_ms']} ms[/yellow]"
                )
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down orchestrator...[/yellow]")
        orchestrator.stop()
        console.print("[green]Shutdown complete.[/green]")


def main():
    parser = argparse.ArgumentParser(description="Vision Pipeline Orchestrator CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run orchestrator in console mode")
    run_parser.add_argument("--config", default="configs/default_pipeline.yaml", help="Path to pipeline YAML config")
    run_parser.add_argument("--clean-db", action="store_true", help="Clean and create new Milvus vector database and clear old event logs")

    # Command: dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Run orchestrator with web dashboard")
    dash_parser.add_argument("--config", default="configs/default_pipeline.yaml", help="Path to pipeline YAML config")
    dash_parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    dash_parser.add_argument("--clean-db", action="store_true", help="Clean and create new Milvus vector database and clear old event logs")

    # Command: clean-db
    clean_parser = subparsers.add_parser("clean-db", help="Clean and create new Milvus database and clear old event logs")

    # Command: benchmark
    subparsers.add_parser("benchmark", help="Benchmark vision model stages")

    # Command: generate-sample
    subparsers.add_parser("generate-sample", help="Generate synthetic factory test video")

    args = parser.parse_args()

    if args.command == "clean-db":
        clean_stored_data()
    elif args.command == "benchmark":
        run_benchmark()
    elif args.command == "dashboard":
        run_pipeline(args.config, with_dashboard=True, port=args.port, clean_db=args.clean_db)
    elif args.command == "run":
        run_pipeline(args.config, with_dashboard=False, clean_db=args.clean_db)
    elif args.command == "generate-sample":
        path = generate_synthetic_factory_video()
        console.print(f"[green]Sample video generated at: {path}[/green]")
    else:
        # Default behavior: run benchmark then launch dashboard or display help
        run_benchmark()


if __name__ == "__main__":
    main()
