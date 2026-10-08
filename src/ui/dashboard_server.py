"""
Interactive Web Dashboard Server for Vision Pipeline Orchestrator.
Streams live MJPEG video with customizable overlays, camera switching,
real-time stage latency charts, event timelines, and Milvus semantic search.
"""

from __future__ import annotations
import os
import json
import time
import logging
import threading
from typing import Optional, Dict
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs, unquote
import cv2
import numpy as np

from src.core.types import FramePacket
from src.pipeline.orchestrator import VisionPipelineOrchestrator
from src.ui.renderer import FrameVisualizer

logger = logging.getLogger(__name__)


class DashboardState:
    """Thread-safe global dashboard state."""
    def __init__(self):
        self.camera_frames: Dict[str, bytes] = {}
        self.camera_packets: Dict[str, FramePacket] = {}
        self.latest_annotated_frame: Optional[bytes] = None
        self.active_camera_id: str = "cam_feed_01"
        self.latest_packet: Optional[FramePacket] = None
        self.show_detections: bool = True
        self.show_tracks: bool = True
        self.show_poses: bool = True
        self.show_masks: bool = True
        self.show_events: bool = True
        self.show_hud: bool = True
        self.lock = threading.Lock()


GLOBAL_STATE = DashboardState()
GLOBAL_VISUALIZER = FrameVisualizer()
GLOBAL_ORCHESTRATOR: Optional[VisionPipelineOrchestrator] = None


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vision Pipeline Orchestrator | Industrial Intelligence Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 29, 47, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --accent: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
    body { background-color: var(--bg); color: var(--text-main); min-height: 100vh; padding: 20px; overflow-x: hidden; }
    
    .navbar {
      display: flex; justify-content: space-between; align-items: center;
      background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--card-border);
      padding: 16px 24px; border-radius: 14px; margin-bottom: 20px;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon {
      width: 38px; height: 38px; background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
    }
    .brand-title { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.5px; }
    .brand-subtitle { font-size: 0.8rem; color: var(--accent); font-family: 'JetBrains Mono', monospace; }
    
    .cam-selector-box {
      display: flex; align-items: center; gap: 10px;
    }
    .cam-select {
      background: rgba(15, 23, 42, 0.9); color: #fff; border: 1px solid var(--primary);
      padding: 8px 14px; border-radius: 8px; font-weight: 600; font-size: 0.9rem; cursor: pointer; outline: none;
    }

    .status-badge {
      display: flex; align-items: center; gap: 8px; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;
      padding: 6px 14px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 20px; color: var(--accent);
    }
    .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }

    .grid-container {
      display: grid; grid-template-columns: 2.2fr 1fr; gap: 20px;
    }
    
    .video-card {
      background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--card-border);
      border-radius: 16px; overflow: hidden; display: flex; flex-direction: column;
    }
    .video-header {
      padding: 14px 20px; border-bottom: 1px solid var(--card-border);
      display: flex; justify-content: space-between; align-items: center;
    }
    .video-viewport {
      width: 100%; aspect-ratio: 16 / 9; background: #000; display: flex; align-items: center; justify-content: center;
    }
    .video-viewport img { width: 100%; height: 100%; object-fit: contain; }

    .controls-bar {
      padding: 14px 20px; background: rgba(15, 23, 42, 0.6); display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
    }
    .toggle-chip {
      display: flex; align-items: center; gap: 6px; font-size: 0.85rem; cursor: pointer; user-select: none;
      background: rgba(255, 255, 255, 0.05); padding: 6px 12px; border-radius: 8px; border: 1px solid var(--card-border);
      transition: all 0.2s ease;
    }
    .toggle-chip input { cursor: pointer; }
    .toggle-chip:hover { background: rgba(255, 255, 255, 0.1); }

    .metrics-col { display: flex; flex-direction: column; gap: 20px; }
    
    .card {
      background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--card-border);
      border-radius: 16px; padding: 20px;
    }
    .card-title {
      font-size: 1.05rem; font-weight: 600; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;
    }

    .kpi-row { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 16px; }
    .kpi-box {
      background: rgba(15, 23, 42, 0.8); border: 1px solid var(--card-border); border-radius: 12px; padding: 14px;
    }
    .kpi-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }
    .kpi-val { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }

    .latency-bar-group { display: flex; flex-direction: column; gap: 10px; }
    .lat-item { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; }
    .progress-track { height: 6px; background: rgba(255, 255, 255, 0.06); border-radius: 3px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 3px; background: var(--primary); transition: width 0.3s; }

    .timeline-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 10px; }
    .timeline-table th { text-align: left; padding: 8px; color: var(--text-muted); font-size: 0.75rem; border-bottom: 1px solid var(--card-border); }
    .timeline-table td { padding: 10px 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); vertical-align: middle; }
    .timeline-table th:last-child, .timeline-table td:last-child { text-align: right; }
    .sev-critical { color: var(--danger); font-weight: 600; }
    .sev-warning { color: var(--warning); font-weight: 600; }

    .view-evidence-btn {
      background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4);
      border-radius: 6px; padding: 4px 10px; font-size: 0.75rem; font-weight: 600; cursor: pointer;
      display: inline-flex; align-items: center; gap: 5px; transition: all 0.2s ease;
    }
    .view-evidence-btn:hover {
      background: rgba(59, 130, 246, 0.35); border-color: #3b82f6; color: #ffffff;
      transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }

    .search-box {
      display: flex; gap: 8px; margin-top: 12px;
    }
    .search-input {
      flex: 1; background: rgba(15, 23, 42, 0.8); border: 1px solid var(--card-border); border-radius: 8px;
      padding: 8px 12px; color: #fff; font-size: 0.85rem; outline: none;
    }
    .search-btn {
      background: var(--primary); color: #fff; border: none; border-radius: 8px; padding: 8px 14px;
      font-weight: 600; cursor: pointer; transition: background 0.2s;
    }
    .search-btn:hover { background: #2563eb; }
    .search-results-list { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; max-height: 240px; overflow-y: auto; }
    .search-item {
      background: rgba(15, 23, 42, 0.7); border: 1px solid var(--card-border); border-radius: 10px;
      padding: 8px 10px; display: flex; align-items: center; gap: 10px; font-size: 0.8rem;
      cursor: pointer; transition: all 0.2s ease;
    }
    .search-item:hover {
      background: rgba(30, 41, 59, 0.8); border-color: rgba(59, 130, 246, 0.4); transform: translateX(2px);
    }
    .search-thumb {
      width: 44px; height: 44px; border-radius: 8px; object-fit: cover;
      background: #000; border: 1px solid rgba(255, 255, 255, 0.1); flex-shrink: 0;
    }

    /* Modal Overlay & Dialog Styles */
    .modal-overlay {
      position: fixed; inset: 0; background: rgba(5, 8, 16, 0.85); backdrop-filter: blur(10px);
      z-index: 10000; display: none; align-items: center; justify-content: center; padding: 20px;
      animation: fadeIn 0.2s ease-out;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .modal-box {
      background: #111827; border: 1px solid rgba(255, 255, 255, 0.12);
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 30px rgba(59, 130, 246, 0.15);
      border-radius: 18px; width: 100%; max-width: 820px; max-height: 90vh;
      display: flex; flex-direction: column; overflow: hidden; animation: slideUp 0.25s ease-out;
    }
    @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    .modal-header {
      padding: 16px 22px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      display: flex; justify-content: space-between; align-items: center; background: rgba(17, 24, 39, 0.8);
    }
    .modal-title { font-size: 1.15rem; font-weight: 700; display: flex; align-items: center; gap: 10px; }
    .modal-close-btn {
      background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text-muted); width: 32px; height: 32px; border-radius: 8px;
      display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 1.1rem;
      transition: all 0.2s;
    }
    .modal-close-btn:hover { background: rgba(239, 68, 68, 0.2); color: #ef4444; border-color: rgba(239, 68, 68, 0.4); }
    .modal-tabs {
      display: flex; gap: 8px; padding: 12px 22px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(15, 23, 42, 0.6);
    }
    .modal-tab {
      padding: 8px 16px; border-radius: 8px 8px 0 0; font-size: 0.85rem; font-weight: 600; cursor: pointer;
      border: 1px solid transparent; border-bottom: none; color: var(--text-muted); background: transparent;
      transition: all 0.2s;
    }
    .modal-tab.active {
      color: #fff; background: #111827; border-color: rgba(255, 255, 255, 0.1); border-bottom: 2px solid var(--primary);
    }
    .modal-body {
      padding: 20px 22px; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 14px;
    }
    .media-container {
      width: 100%; min-height: 280px; max-height: 460px; background: #000; border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.08); display: flex; align-items: center; justify-content: center;
      overflow: hidden; position: relative;
    }
    .media-container img, .media-container video {
      max-width: 100%; max-height: 460px; object-fit: contain; width: 100%; height: auto;
    }
    .info-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;
    }
    .info-pill {
      background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 8px; padding: 8px 12px; font-size: 0.8rem;
    }
    .info-pill-label { color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }
    .info-pill-val { font-weight: 600; margin-top: 2px; }
    .modal-footer {
      padding: 14px 22px; border-top: 1px solid rgba(255, 255, 255, 0.08);
      display: flex; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.8);
    }
    .download-link {
      color: var(--primary); font-size: 0.85rem; text-decoration: none; display: flex; align-items: center; gap: 6px;
      font-family: 'JetBrains Mono', monospace;
    }
    .download-link:hover { text-decoration: underline; color: #60a5fa; }
  </style>
</head>
<body>

  <header class="navbar">
    <div class="brand">
      <div class="brand-icon">👁️</div>
      <div>
        <div class="brand-title">Vision Pipeline Orchestrator</div>
        <div class="brand-subtitle">Project 8 • Multi-Stream Industrial Intelligence Engine</div>
      </div>
    </div>
    
    <div class="cam-selector-box">
      <label style="font-size:0.85rem; color:var(--text-muted);">ACTIVE FEED:</label>
      <select id="cam-select" class="cam-select" onchange="switchCamera(this.value)">
        <option value="cam_feed_01" selected>Camera Feed 1 (Assembly Area - sample_feed1.mp4)</option>
        <option value="cam_feed_02">Camera Feed 2 (Logistics Floor - sample_feed2.mp4)</option>
      </select>
    </div>

    <div class="status-badge">
      <div class="pulse-dot"></div>
      <span id="active-pipeline-text">PIPELINE ACTIVE</span>
    </div>
  </header>

  <main class="grid-container">
    <section class="video-card">
      <div class="video-header">
        <h2 id="view-title" style="font-size:1.1rem; font-weight:600;">Feed: Camera Feed 1 - Assembly & Work Area</h2>
        <span style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; color:var(--accent);">REAL-TIME STREAM • EVERY FRAME</span>
      </div>
      <div class="video-viewport">
        <img id="stream-img" src="/video_feed?cam=cam_feed_01" alt="Vision Orchestrator Stream">
      </div>
      <div class="controls-bar">
        <label class="toggle-chip"><input type="checkbox" id="chk-det" checked onchange="updateToggles()"> RF-DETR Boxes</label>
        <label class="toggle-chip"><input type="checkbox" id="chk-track" checked onchange="updateToggles()"> ByteTrack IDs & Vel</label>
        <label class="toggle-chip"><input type="checkbox" id="chk-pose" checked onchange="updateToggles()"> ViTPose Skeletons</label>
        <label class="toggle-chip"><input type="checkbox" id="chk-event" checked onchange="updateToggles()"> Alert Banners</label>
        <label class="toggle-chip"><input type="checkbox" id="chk-hud" checked onchange="updateToggles()"> Telemetry HUD</label>
      </div>
    </section>

    <aside class="metrics-col">
      <div class="card">
        <div class="card-title">
          <span>System Telemetry</span>
          <span style="color:var(--accent); font-size:0.8rem; font-family:'JetBrains Mono', monospace;">REALTIME</span>
        </div>
        <div class="kpi-row">
          <div class="kpi-box">
            <div class="kpi-label">Effective FPS</div>
            <div class="kpi-val" id="kpi-fps">--</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Latency (ms)</div>
            <div class="kpi-val" id="kpi-latency">--</div>
          </div>
        </div>
        <div class="latency-bar-group">
          <div>
            <div class="lat-item"><span>RF-DETR Medium Detector</span><span id="lat-det">-- ms</span></div>
            <div class="progress-track"><div id="bar-det" class="progress-fill" style="width:0%; background:#3b82f6;"></div></div>
          </div>
          <div>
            <div class="lat-item"><span>ByteTrack Association</span><span id="lat-track">-- ms</span></div>
            <div class="progress-track"><div id="bar-track" class="progress-fill" style="width:0%; background:#10b981;"></div></div>
          </div>
          <div>
            <div class="lat-item"><span>ViTPose Posture</span><span id="lat-pose">-- ms</span></div>
            <div class="progress-track"><div id="bar-pose" class="progress-fill" style="width:0%; background:#f59e0b;"></div></div>
          </div>
          <div>
            <div class="lat-item"><span>CLIP & Milvus</span><span id="lat-clip">-- ms</span></div>
            <div class="progress-track"><div id="bar-clip" class="progress-fill" style="width:0%; background:#06b6d4;"></div></div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Milvus Semantic Search</span>
          <span style="color:var(--primary); font-size:0.75rem; font-family:'JetBrains Mono', monospace;">VECTOR DB</span>
        </div>
        <div class="search-box">
          <input type="text" id="semantic-search-input" class="search-input" placeholder="e.g. worker wearing safety vest...">
          <button class="search-btn" onclick="executeSemanticSearch()">Search</button>
        </div>
        <div id="search-results" class="search-results-list"></div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Safety Event Timeline</span>
          <div style="display:flex; align-items:center; gap:8px;">
            <span id="event-count-badge" style="background:rgba(239, 68, 68, 0.2); color:#ef4444; padding:2px 8px; border-radius:10px; font-size:0.75rem;">0 Events</span>
            <button id="btn-clean-db" onclick="cleanDatabaseAndEvents()" title="Clean and recreate Milvus vector database and clear old event logs" style="background:rgba(239, 68, 68, 0.15); border:1px solid rgba(239, 68, 68, 0.4); color:#fca5a5; font-size:0.75rem; padding:3px 10px; border-radius:8px; cursor:pointer; font-weight:600; display:flex; align-items:center; gap:4px; transition:all 0.2s;">
              🧹 Clean DB
            </button>
          </div>
        </div>
        <table class="timeline-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Incident</th>
              <th>Severity</th>
              <th>Tracks</th>
              <th style="text-align:right;">Evidence</th>
            </tr>
          </thead>
          <tbody id="timeline-body">
            <tr><td colspan="5" style="text-align:center; color:var(--text-muted);">Monitoring video feed for safety violations...</td></tr>
          </tbody>
        </table>
      </div>
    </aside>
  </main>

  <!-- High-Resolution Inspection & Evidence Media Modal -->
  <div id="modal-overlay" class="modal-overlay" onclick="handleOverlayClick(event)">
    <div class="modal-box" id="modal-box">
      <div class="modal-header">
        <div class="modal-title" id="modal-title">
          <span>Inspection Modal</span>
        </div>
        <button class="modal-close-btn" onclick="closeModal()">✕</button>
      </div>

      <div class="modal-tabs" id="modal-tabs">
        <button class="modal-tab active" id="tab-btn-snapshot" onclick="switchModalTab('snapshot')">📸 Snapshot Image</button>
        <button class="modal-tab" id="tab-btn-video" onclick="switchModalTab('video')">🎥 Recorded Video Clip</button>
        <button class="modal-tab" id="tab-btn-meta" onclick="switchModalTab('meta')">📋 Incident Metadata</button>
      </div>

      <div class="modal-body" id="modal-body"></div>

      <div class="modal-footer">
        <div id="modal-footer-left"></div>
        <button class="search-btn" style="padding:6px 18px;" onclick="closeModal()">Close</button>
      </div>
    </div>
  </div>

  <script>
    let currentCam = 'cam_feed_01';
    let currentModalData = null;
    let currentModalTab = 'snapshot';

    function switchCamera(camId) {
      currentCam = camId;
      document.getElementById('stream-img').src = '/video_feed?cam=' + camId + '&t=' + Date.now();
      const title = camId === 'cam_feed_01' 
        ? 'Feed: Camera Feed 1 - Assembly & Work Area (sample_feed1.mp4)'
        : 'Feed: Camera Feed 2 - Corridor & Logistics Dock (sample_feed2.mp4)';
      document.getElementById('view-title').innerText = title;

      fetch('/api/switch_camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: camId })
      });
    }

    async function updateTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) return;
        const data = await res.json();
        
        document.getElementById('kpi-fps').innerText = data.fps || '0.0';
        document.getElementById('kpi-latency').innerText = (data.total_pipeline_latency_ms || 0).toFixed(1);
        
        const lat = data.latencies_ms || {};
        updateBar('det', lat.rf_detr || 0, 1000);
        updateBar('track', lat.bytetrack || 0, 20);
        updateBar('pose', lat.vitpose || 0, 1000);
        updateBar('clip', lat.clip || 0, 500);
      } catch (err) {
        console.error(err);
      }
    }

    function updateBar(key, val, maxVal) {
      document.getElementById('lat-' + key).innerText = val.toFixed(1) + ' ms';
      const pct = Math.min(100, Math.max(5, (val / maxVal) * 100));
      document.getElementById('bar-' + key).style.width = pct + '%';
    }

    let cachedEventsJson = '';

    async function updateEvents() {
      try {
        const res = await fetch('/api/events');
        if (!res.ok) return;
        const events = await res.json();
        
        document.getElementById('event-count-badge').innerText = events.length + ' Events';
        const tbody = document.getElementById('timeline-body');
        if (events.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:20px;">No events recorded. Database is clean.</td></tr>';
          cachedEventsJson = '[]';
          window._currentEvents = [];
          return;
        }
        
        const newJson = JSON.stringify(events.map(e => e.event_id || e.timestamp));
        if (newJson === cachedEventsJson) return;
        cachedEventsJson = newJson;

        tbody.innerHTML = '';
        events.slice(0, 15).forEach(evt => {
          const row = document.createElement('tr');
          const timeStr = new Date(evt.timestamp * 1000).toLocaleTimeString();
          const sevClass = evt.severity === 'CRITICAL' ? 'sev-critical' : 'sev-warning';
          const trackStr = (evt.track_ids && evt.track_ids.length > 0)
            ? '#' + evt.track_ids.join(', #')
            : '--';

          row.innerHTML = `
            <td style="font-family:'JetBrains Mono', monospace; font-size:0.75rem;">${timeStr}</td>
            <td><strong>${evt.type.replace(/_/g, ' ')}</strong><br><small style="color:var(--text-muted);">${evt.description ? evt.description.substring(0, 42) + '...' : ''}</small></td>
            <td class="${sevClass}">${evt.severity}</td>
            <td style="font-family:'JetBrains Mono', monospace; font-size:0.8rem;">${trackStr}</td>
            <td style="text-align:right;">
              <button class="view-evidence-btn" onclick="openEvidenceModalById('${evt.event_id || ''}', ${evt.timestamp})">
                📹 View
              </button>
            </td>
          `;
          row.style.cursor = 'pointer';
          row.onclick = (e) => {
            if (e.target.tagName !== 'BUTTON') {
              openEvidenceModal(evt);
            }
          };
          tbody.appendChild(row);
        });

        window._currentEvents = events;
      } catch (err) {
        console.error(err);
      }
    }

    async function cleanDatabaseAndEvents() {
      if (!confirm("Are you sure you want to clean and recreate Milvus vector database, clear old event logs and evidence, and restart tracking from ID 1?")) {
        return;
      }
      const btn = document.getElementById('btn-clean-db');
      try {
        if (btn) btn.innerText = 'Cleaning...';
        const res = await fetch('/api/clean_db', { method: 'POST' });
        const data = await res.json();
        cachedEventsJson = '';
        window._currentEvents = [];
        document.getElementById('timeline-body').innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--accent); padding:20px;">✓ Database and events cleaned successfully.</td></tr>';
        document.getElementById('event-count-badge').innerText = '0 Events';
        if (btn) btn.innerText = '🧹 Clean DB';
        setTimeout(updateEvents, 1000);
      } catch (err) {
        alert('Error cleaning database: ' + err);
        if (btn) btn.innerText = '🧹 Clean DB';
      }
    }

    function openEvidenceModalById(eventId, timestamp) {
      if (window._currentEvents) {
        const found = window._currentEvents.find(e => (e.event_id && e.event_id === eventId) || Math.abs(e.timestamp - timestamp) < 0.001);
        if (found) {
          openEvidenceModal(found);
          return;
        }
      }
    }

    function openEvidenceModal(evt) {
      currentModalData = { type: 'evidence', evt: evt };
      const snapUri = evt.snapshot_uri || '';
      const clipUri = evt.evidence_uri || '';
      const snapFile = snapUri ? snapUri.split(/[\\\\/]/).pop() : '';
      const clipFile = clipUri ? clipUri.split(/[\\\\/]/).pop() : '';
      currentModalData.snapFile = snapFile;
      currentModalData.clipFile = clipFile;

      const titleEl = document.getElementById('modal-title');
      const sevClass = evt.severity === 'CRITICAL' ? 'sev-critical' : 'sev-warning';
      titleEl.innerHTML = `<span>🚨 Incident: ${evt.type.replace(/_/g, ' ')}</span> <span class="${sevClass}" style="font-size:0.75rem; padding:3px 8px; border-radius:12px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1);">${evt.severity}</span>`;

      document.getElementById('modal-tabs').style.display = 'flex';
      switchModalTab(clipFile ? 'video' : 'snapshot');
      document.getElementById('modal-overlay').style.display = 'flex';
    }

    function openCropModal(fname, rawCosine, matchPct) {
      currentModalData = { type: 'crop', fname, rawCosine, matchPct };
      const titleEl = document.getElementById('modal-title');
      titleEl.innerHTML = `<span>🔍 Milvus Vector Crop: ${fname}</span>`;
      document.getElementById('modal-tabs').style.display = 'none';

      const body = document.getElementById('modal-body');
      body.innerHTML = `
        <div class="media-container">
          <img src="/crops/${fname}" alt="${fname}" onerror="this.onerror=null; this.src='/evidence/crops/${fname}';">
        </div>
        <div class="info-grid">
          <div class="info-pill">
            <div class="info-pill-label">Semantic Match</div>
            <div class="info-pill-val" style="color:var(--accent); font-family:'JetBrains Mono';">${matchPct}% Match</div>
          </div>
          <div class="info-pill">
            <div class="info-pill-label">Cosine Similarity</div>
            <div class="info-pill-val" style="font-family:'JetBrains Mono';">${rawCosine}</div>
          </div>
          <div class="info-pill">
            <div class="info-pill-label">Vector Database</div>
            <div class="info-pill-val">Milvus Lite (Cosine Index)</div>
          </div>
          <div class="info-pill">
            <div class="info-pill-label">File Storage</div>
            <div class="info-pill-val" style="font-size:0.75rem; word-break:break-all;">data/evidence/crops/${fname}</div>
          </div>
        </div>
      `;

      document.getElementById('modal-footer-left').innerHTML = `
        <a class="download-link" href="/crops/${fname}" download="${fname}">
          ⬇️ Download Crop Image
        </a>
      `;
      document.getElementById('modal-overlay').style.display = 'flex';
    }

    function switchModalTab(tabName) {
      currentModalTab = tabName;
      ['snapshot', 'video', 'meta'].forEach(t => {
        const btn = document.getElementById('tab-btn-' + t);
        if (btn) btn.className = (t === tabName) ? 'modal-tab active' : 'modal-tab';
      });

      if (!currentModalData || currentModalData.type !== 'evidence') return;
      const evt = currentModalData.evt;
      const snapFile = currentModalData.snapFile;
      const clipFile = currentModalData.clipFile;
      const body = document.getElementById('modal-body');
      const footerLeft = document.getElementById('modal-footer-left');

      if (tabName === 'snapshot') {
        if (snapFile) {
          body.innerHTML = `
            <div class="media-container">
              <img src="/evidence/${snapFile}" alt="Incident Snapshot">
            </div>
            ${renderIncidentInfoPills(evt)}
          `;
          footerLeft.innerHTML = `<a class="download-link" href="/evidence/${snapFile}" download="${snapFile}">⬇️ Download High-Res Snapshot (JPEG)</a>`;
        } else {
          body.innerHTML = `<div style="text-align:center; padding:40px; color:var(--text-muted);">No snapshot image recorded for this incident.</div>`;
          footerLeft.innerHTML = '';
        }
      } else if (tabName === 'video') {
        if (clipFile) {
          body.innerHTML = `
            <div class="media-container">
              <video controls autoplay loop playsinline src="/evidence/${clipFile}">
                <source src="/evidence/${clipFile}" type="video/mp4">
                Your browser does not support HTML5 video playback.
              </video>
            </div>
            ${renderIncidentInfoPills(evt)}
          `;
          footerLeft.innerHTML = `<a class="download-link" href="/evidence/${clipFile}" download="${clipFile}">⬇️ Download Evidence Video (MP4)</a>`;
        } else {
          body.innerHTML = `<div style="text-align:center; padding:40px; color:var(--text-muted);">No video clip recorded for this incident.</div>`;
          footerLeft.innerHTML = '';
        }
      } else if (tabName === 'meta') {
        body.innerHTML = `
          <div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:4px;">Structured JSON Incident Event Record:</div>
          <pre style="background:rgba(15, 23, 42, 0.9); padding:16px; border-radius:10px; border:1px solid rgba(255,255,255,0.06); color:#10b981; font-family:'JetBrains Mono', monospace; font-size:0.8rem; overflow:auto; max-height:420px; line-height:1.5;">${escapeHtml(JSON.stringify(evt, null, 2))}</pre>
        `;
        footerLeft.innerHTML = '';
      }
    }

    function renderIncidentInfoPills(evt) {
      const timeStr = new Date(evt.timestamp * 1000).toLocaleString();
      const tracks = (evt.track_ids && evt.track_ids.length > 0) ? '#' + evt.track_ids.join(', #') : 'None';
      const desc = evt.description || 'No description provided.';
      return `
        <div style="font-size:0.85rem; color:#cbd5e1; background:rgba(15,23,42,0.6); padding:10px 14px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
          <strong>Description:</strong> ${desc}
        </div>
        <div class="info-grid">
          <div class="info-pill">
            <div class="info-pill-label">Incident Time</div>
            <div class="info-pill-val" style="font-family:'JetBrains Mono'; font-size:0.8rem;">${timeStr}</div>
          </div>
          <div class="info-pill">
            <div class="info-pill-label">Severity Level</div>
            <div class="info-pill-val" style="color:${evt.severity === 'CRITICAL' ? 'var(--danger)' : 'var(--warning)'}; font-weight:700;">${evt.severity}</div>
          </div>
          <div class="info-pill">
            <div class="info-pill-label">Involved Tracks</div>
            <div class="info-pill-val" style="font-family:'JetBrains Mono';">${tracks}</div>
          </div>
          <div class="info-pill">
            <div class="info-pill-label">Rule / Event ID</div>
            <div class="info-pill-val" style="font-size:0.75rem; font-family:'JetBrains Mono';">${(evt.event_id || '').substring(0, 16)}...</div>
          </div>
        </div>
      `;
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function closeModal() {
      const modal = document.getElementById('modal-overlay');
      modal.style.display = 'none';
      const video = modal.querySelector('video');
      if (video) {
        video.pause();
        video.src = '';
      }
      currentModalData = null;
    }

    function handleOverlayClick(event) {
      if (event.target.id === 'modal-overlay') {
        closeModal();
      }
    }

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal();
    });

    async function executeSemanticSearch() {
      const q = document.getElementById('semantic-search-input').value.trim();
      if (!q) return;
      const resContainer = document.getElementById('search-results');
      resContainer.innerHTML = '<span style="color:var(--text-muted); font-size:0.8rem;">Searching Milvus vector database...</span>';
      
      try {
        const res = await fetch('/api/search?q=' + encodeURIComponent(q));
        const matches = await res.json();
        if (matches.length === 0) {
          resContainer.innerHTML = '<span style="color:var(--text-muted); font-size:0.8rem;">No matches found in vector index.</span>';
          return;
        }
        resContainer.innerHTML = '';
        matches.forEach(m => {
          const item = document.createElement('div');
          item.className = 'search-item';
          const fname = m.file_path.split(/[\\\\/]/).pop();
          item.innerHTML = `
            <img src="/crops/${fname}" class="search-thumb" alt="${fname}" onerror="this.onerror=null; this.src='/evidence/crops/${fname}';">
            <div style="flex:1; min-width:0;">
              <div style="font-weight:600; font-size:0.85rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${fname}">${fname}</div>
              <div style="color:var(--text-muted); font-size:0.75rem; font-family:'JetBrains Mono';">Cosine: ${m.raw_cosine}</div>
            </div>
            <div style="display:flex; flex-direction:column; align-items:flex-end; gap:3px;">
              <span style="color:var(--accent); font-weight:700; font-family:'JetBrains Mono'; font-size:0.85rem;">${m.calibrated_match_pct}% Match</span>
              <span style="font-size:0.7rem; color:var(--primary); font-weight:600;">Inspect ↗</span>
            </div>
          `;
          item.onclick = () => openCropModal(fname, m.raw_cosine, m.calibrated_match_pct);
          resContainer.appendChild(item);
        });
      } catch (err) {
        resContainer.innerHTML = '<span style="color:var(--danger); font-size:0.8rem;">Search error: ' + err + '</span>';
      }
    }

    function updateToggles() {
      const payload = {
        show_detections: document.getElementById('chk-det') ? document.getElementById('chk-det').checked : false,
        show_tracks: document.getElementById('chk-track') ? document.getElementById('chk-track').checked : false,
        show_poses: document.getElementById('chk-pose') ? document.getElementById('chk-pose').checked : false,
        show_masks: false,
        show_events: document.getElementById('chk-event') ? document.getElementById('chk-event').checked : false,
        show_hud: document.getElementById('chk-hud') ? document.getElementById('chk-hud').checked : false,
      };
      fetch('/api/toggles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).catch(err => console.error('Error updating toggles:', err));
    }

    async function initToggles() {
      try {
        const res = await fetch('/api/toggles');
        if (res.ok) {
          const t = await res.json();
          if (document.getElementById('chk-det') && t.show_detections !== undefined) document.getElementById('chk-det').checked = !!t.show_detections;
          if (document.getElementById('chk-track') && t.show_tracks !== undefined) document.getElementById('chk-track').checked = !!t.show_tracks;
          if (document.getElementById('chk-pose') && t.show_poses !== undefined) document.getElementById('chk-pose').checked = !!t.show_poses;
          if (document.getElementById('chk-event') && t.show_events !== undefined) document.getElementById('chk-event').checked = !!t.show_events;
          if (document.getElementById('chk-hud') && t.show_hud !== undefined) document.getElementById('chk-hud').checked = !!t.show_hud;
        }
      } catch (e) {
        console.error('Error initializing toggles:', e);
      }
    }

    initToggles();
    setInterval(updateTelemetry, 1000);
    setInterval(updateEvents, 1500);
  </script>
</body>
</html>
"""


def get_persisted_events(limit: int = 30) -> list:
    """Load recent persisted JSON incident events from disk."""
    evidence_dir = "data/evidence"
    if not os.path.isdir(evidence_dir):
        return []

    events = []
    try:
        json_files = [
            os.path.join(evidence_dir, f)
            for f in os.listdir(evidence_dir)
            if f.startswith("evidence_") and f.endswith(".json")
        ]
        json_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for jf in json_files[:limit]:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    events.append(json.load(f))
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"Error loading persisted events: {e}")
    return events


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler for REST telemetry, Milvus query, media serving, and MJPEG stream."""

    def log_message(self, format, *args):
        pass

    def _serve_static_file(self, file_path: str):
        """Serve images, video clips, and JSON with range request support for smooth playback."""
        if not os.path.isfile(file_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")
            return

        lower_path = file_path.lower()
        if lower_path.endswith((".jpg", ".jpeg")):
            content_type = "image/jpeg"
        elif lower_path.endswith(".png"):
            content_type = "image/png"
        elif lower_path.endswith(".mp4"):
            content_type = "video/mp4"
        elif lower_path.endswith(".json"):
            content_type = "application/json"
        else:
            content_type = "application/octet-stream"

        file_size = os.path.getsize(file_path)
        range_header = self.headers.get("Range")

        # Handle partial range requests (required for HTML5 video seeking)
        if range_header and range_header.startswith("bytes="):
            try:
                range_val = range_header.split("=")[1].strip()
                parts = range_val.split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
                if end >= file_size:
                    end = file_size - 1
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    chunk_size = 64 * 1024
                    while remaining > 0:
                        read_amt = min(chunk_size, remaining)
                        chunk = f.read(read_amt)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return
            except Exception as e:
                logger.warning(f"Error handling Range request for {file_path}: {e}")

        # Full file response
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(file_path, "rb") as f:
                chunk_size = 64 * 1024
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if GLOBAL_ORCHESTRATOR:
                summary = GLOBAL_ORCHESTRATOR.metrics.get_telemetry_summary()
            else:
                summary = {"fps": 0.0, "latencies_ms": {}}
            self.wfile.write(json.dumps(summary).encode("utf-8"))

        elif path == "/api/toggles":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with GLOBAL_STATE.lock:
                toggles = {
                    "show_detections": GLOBAL_STATE.show_detections,
                    "show_tracks": GLOBAL_STATE.show_tracks,
                    "show_poses": GLOBAL_STATE.show_poses,
                    "show_masks": GLOBAL_STATE.show_masks,
                    "show_events": GLOBAL_STATE.show_events,
                    "show_hud": GLOBAL_STATE.show_hud,
                }
            self.wfile.write(json.dumps(toggles).encode("utf-8"))

        elif path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            combined_events = []
            seen_ids = set()

            # 1. In-memory events from current execution
            if GLOBAL_ORCHESTRATOR and GLOBAL_ORCHESTRATOR.rule_engine:
                for evt in GLOBAL_ORCHESTRATOR.rule_engine.event_history[-30:]:
                    ed = evt.to_dict()
                    combined_events.append(ed)
                    if ed.get("event_id"):
                        seen_ids.add(ed["event_id"])

            # 2. Persisted incident records from disk
            persisted = get_persisted_events(limit=30)
            for pe in persisted:
                eid = pe.get("event_id")
                if eid and eid not in seen_ids:
                    combined_events.append(pe)
                    seen_ids.add(eid)
                elif not eid:
                    combined_events.append(pe)

            # Sort most recent first
            combined_events.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            self.wfile.write(json.dumps(combined_events[:30]).encode("utf-8"))

        elif path == "/api/search":
            q_text = query.get("q", [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            results = []
            if GLOBAL_ORCHESTRATOR and GLOBAL_ORCHESTRATOR.clip:
                results = GLOBAL_ORCHESTRATOR.clip.query(q_text, top_k=6)
            self.wfile.write(json.dumps(results).encode("utf-8"))

        elif path.startswith("/evidence/"):
            rel = unquote(path[len("/evidence/"):].lstrip("/"))
            base = os.path.abspath("data/evidence")
            full_path = os.path.abspath(os.path.join(base, rel))
            if full_path.startswith(base) and os.path.isfile(full_path):
                self._serve_static_file(full_path)
            else:
                self.send_response(404)
                self.end_headers()

        elif path.startswith("/crops/"):
            rel = unquote(path[len("/crops/"):].lstrip("/"))
            base = os.path.abspath("data/evidence/crops")
            full_path = os.path.abspath(os.path.join(base, rel))
            if full_path.startswith(base) and os.path.isfile(full_path):
                self._serve_static_file(full_path)
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/video_feed":
            target_cam = query.get("cam", [GLOBAL_STATE.active_camera_id])[0]
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_frame_bytes = None
            while True:
                frame_bytes = None
                with GLOBAL_STATE.lock:
                    frame_bytes = GLOBAL_STATE.camera_frames.get(target_cam, GLOBAL_STATE.latest_annotated_frame)

                # If no processed frame yet (starting up), show raw camera frame
                if frame_bytes is None and GLOBAL_ORCHESTRATOR and target_cam in GLOBAL_ORCHESTRATOR.camera_contexts:
                    reader = GLOBAL_ORCHESTRATOR.camera_contexts[target_cam].get("reader")
                    if reader:
                        success, _, _, raw_bgr = reader.read_latest()
                        if success and raw_bgr is not None:
                            _, buffer = cv2.imencode(".jpg", raw_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                            frame_bytes = buffer.tobytes()

                if frame_bytes is not None and frame_bytes != last_frame_bytes:
                    last_frame_bytes = frame_bytes
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        break

                time.sleep(0.015)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/toggles":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            with GLOBAL_STATE.lock:
                if "show_detections" in data:
                    GLOBAL_STATE.show_detections = bool(data["show_detections"])
                if "show_tracks" in data:
                    GLOBAL_STATE.show_tracks = bool(data["show_tracks"])
                if "show_poses" in data:
                    GLOBAL_STATE.show_poses = bool(data["show_poses"])
                if "show_masks" in data:
                    GLOBAL_STATE.show_masks = bool(data["show_masks"])
                if "show_events" in data:
                    GLOBAL_STATE.show_events = bool(data["show_events"])
                if "show_hud" in data:
                    GLOBAL_STATE.show_hud = bool(data["show_hud"])

                # Immediately re-render cached frames with the new toggle settings
                for cam_id, pkt in GLOBAL_STATE.camera_packets.items():
                    if pkt is not None and pkt.image is not None:
                        rendered = GLOBAL_VISUALIZER.render(
                            pkt,
                            show_detections=GLOBAL_STATE.show_detections,
                            show_tracks=GLOBAL_STATE.show_tracks,
                            show_poses=GLOBAL_STATE.show_poses,
                            show_masks=GLOBAL_STATE.show_masks,
                            show_events=GLOBAL_STATE.show_events,
                            show_hud=GLOBAL_STATE.show_hud,
                        )
                        _, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        fb = buffer.tobytes()
                        GLOBAL_STATE.camera_frames[cam_id] = fb
                        if cam_id == GLOBAL_STATE.active_camera_id:
                            GLOBAL_STATE.latest_annotated_frame = fb

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        elif self.path == "/api/switch_camera":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            with GLOBAL_STATE.lock:
                GLOBAL_STATE.active_camera_id = data.get("camera_id", GLOBAL_STATE.active_camera_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        elif self.path == "/api/clean_db":
            deleted_count = 0
            if GLOBAL_ORCHESTRATOR and hasattr(GLOBAL_ORCHESTRATOR, "clean_db"):
                res = GLOBAL_ORCHESTRATOR.clean_db(clear_evidence=True)
                deleted_count = res.get("deleted_evidence_count", 0)
            else:
                # Standalone fallback if orchestrator not attached
                evidence_dir = "data/evidence"
                if os.path.isdir(evidence_dir):
                    for f in os.listdir(evidence_dir):
                        if f.startswith("evidence_") and (f.endswith(".json") or f.endswith(".jpg") or f.endswith(".mp4")):
                            try:
                                os.remove(os.path.join(evidence_dir, f))
                                deleted_count += 1
                            except Exception:
                                pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "deleted_count": deleted_count}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_dashboard_server(orchestrator: VisionPipelineOrchestrator, host: str = "127.0.0.1", port: int = 8501) -> HTTPServer:
    """Start dashboard HTTP server on background thread."""
    global GLOBAL_ORCHESTRATOR
    GLOBAL_ORCHESTRATOR = orchestrator

    def on_frame_processed(packet: FramePacket):
        with GLOBAL_STATE.lock:
            GLOBAL_STATE.latest_packet = packet
            GLOBAL_STATE.camera_packets[packet.camera_id] = packet
            rendered = GLOBAL_VISUALIZER.render(
                packet,
                show_detections=GLOBAL_STATE.show_detections,
                show_tracks=GLOBAL_STATE.show_tracks,
                show_poses=GLOBAL_STATE.show_poses,
                show_masks=GLOBAL_STATE.show_masks,
                show_events=GLOBAL_STATE.show_events,
                show_hud=GLOBAL_STATE.show_hud,
            )
            # Encode JPEG
            _, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_bytes = buffer.tobytes()
            GLOBAL_STATE.camera_frames[packet.camera_id] = frame_bytes
            if packet.camera_id == GLOBAL_STATE.active_camera_id:
                GLOBAL_STATE.latest_annotated_frame = frame_bytes

    orchestrator.add_packet_listener(on_frame_processed)

    server = ThreadedHTTPServer((host, port), DashboardHTTPHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    logger.info(f"Dashboard server running at http://{host}:{port}/")
    return server
