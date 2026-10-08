"""
CLIP Zero-Shot Semantic Retrieval & Verification Module with Milvus Vector Database.
Provides open-vocabulary safety compliance verification and semantic incident retrieval for hard cases.
Adheres to independent callable design principle.
"""

from __future__ import annotations
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src.core.types import Track
from src.core.registry import ModelRegistry
from src.models.base import BaseVisionModel
from src.models.hf_clip import HFCLIPModel
from src.storage.vectordb import VectorStore

logger = logging.getLogger(__name__)


@ModelRegistry.register("clip")
@ModelRegistry.register("zero_shot_clip")
class ZeroShotCLIPVerifier(BaseVisionModel):
    """
    CLIP (Contrastive Language-Image Pretraining) semantic verification engine.
    Uses HFCLIPModel for embedding generation and Milvus VectorStore for vector indexing & search.
    """

    def __init__(
        self,
        name: str = "clip-vit-b-32",
        version: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, version=version, device=device, options=options or {})
        self.conf_threshold = float(self.options.get("confidence_threshold", 0.30))
        self.default_queries = self.options.get(
            "safety_queries",
            [
                "worker wearing high visibility safety vest and hardhat",
                "worker not wearing safety vest",
                "worker fallen on factory floor",
                "forklift carrying load",
            ],
        )
        self.model_id = self.options.get("model_id", "openai/clip-vit-base-patch32")
        self.db_path = self.options.get("db_path", "data/milvus_orchestrator.db")
        self.crops_dir = self.options.get("crops_dir", "data/evidence/crops")
        
        self.clip_model: Optional[HFCLIPModel] = None
        self.vdb: Optional[VectorStore] = None
        self._track_cache: Dict[int, Tuple[float, Dict[str, float]]] = {}

    def initialize(self) -> None:
        logger.info(f"Initializing {self.name} ({self.version}) on '{self.device}'...")
        os.makedirs(self.crops_dir, exist_ok=True)
        
        # 1. Initialize CLIP Model
        self.clip_model = HFCLIPModel(model_id=self.model_id, device=self.device)
        
        # 2. Initialize Milvus Vector Database
        try:
            self.vdb = VectorStore(db_path=self.db_path, collection_name="vision_events")
            logger.info("Milvus VectorStore connected successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize Milvus VectorStore: {e}")
            self.vdb = None

        self._is_initialized = True
        logger.info("CLIP zero-shot verifier and vector database initialized.")

    def clean_database(self) -> None:
        """Clear cached track embeddings and clean/recreate Milvus vector database."""
        self._track_cache.clear()
        if self.vdb:
            self.vdb.clean_and_recreate()
        logger.info("[+] Cleaned CLIP track cache and Milvus vector collection.")

    @staticmethod
    def calibrate_score(raw_cosine: float) -> float:
        """Empirical min-max calibration (0.15 -> 0%, 0.38+ -> 100%)."""
        clamped = np.clip((raw_cosine - 0.15) / (0.38 - 0.15), 0.0, 1.0)
        return float(round(clamped * 100, 2))

    def predict(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        candidate_queries: Optional[List[str]] = None,
        auto_index: bool = True,
    ) -> Dict[int, Dict[str, float]]:
        """
        Verify safety attributes on tracked object crops using real CLIP cosine similarities.
        Optionally indexes crops into the Milvus vector database.
        Returns a mapping: {track_id: {query: similarity_score, ...}}
        """
        if not self._is_initialized:
            self.initialize()

        if frame is None or not tracks:
            return {}

        queries = candidate_queries or self.default_queries
        h, w = frame.shape[:2]
        results: Dict[int, Dict[str, float]] = {}
        crops_to_index: List[str] = []
        vectors_to_index: List[List[float]] = []

        now = time.time()
        for track in tracks:
            # Check track cache (memoize for 5.0 seconds)
            if track.track_id in self._track_cache:
                last_t, cached_scores = self._track_cache[track.track_id]
                if now - last_t < 5.0:
                    results[track.track_id] = cached_scores
                    continue

            x1, y1, x2, y2 = track.bbox.to_int_xyxy()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w < 15 or crop_h < 15:
                continue

            crop = frame[y1:y2, x1:x2]

            # 1. Compute CLIP similarities with queries and reuse single forward pass vector
            if self.clip_model:
                sim_scores, vec = self.clip_model.compute_similarity(crop, queries)
                results[track.track_id] = sim_scores
                self._track_cache[track.track_id] = (now, sim_scores)

                # 2. Save crop for Milvus VectorStore indexing
                if auto_index and self.vdb:
                    crop_filename = f"crop_track_{track.track_id}_{int(time.time() * 1000) % 100000}.jpg"
                    crop_path = os.path.join(self.crops_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    crops_to_index.append(crop_path)
                    vectors_to_index.append(vec)
            else:
                # Heuristic fallback if model not loaded
                results[track.track_id] = {q: 0.5 for q in queries}

        # Index new embeddings into Milvus
        if vectors_to_index and self.vdb:
            try:
                self.vdb.insert_batch(vectors_to_index, crops_to_index)
            except Exception as e:
                logger.error(f"Error inserting embeddings into Milvus: {e}")

        return results

    def query(self, text_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic natural-language query over indexed factory incident crops using Milvus VectorStore.
        Returns top matching crops with calibrated percentage scores.
        """
        if not self._is_initialized:
            self.initialize()

        if not text_query.strip() or not self.clip_model or not self.vdb:
            return []

        query_vec = self.clip_model.encode_text(text_query)
        matches = self.vdb.search(query_vec, top_k=top_k)

        formatted: List[Dict[str, Any]] = []
        for m in matches:
            raw_cosine = m.get("score", 0.0)
            calibrated = self.calibrate_score(raw_cosine)
            formatted.append({
                "id": m.get("id"),
                "file_path": m.get("file_path"),
                "raw_cosine": round(raw_cosine, 4),
                "calibrated_match_pct": calibrated,
            })

        return formatted
