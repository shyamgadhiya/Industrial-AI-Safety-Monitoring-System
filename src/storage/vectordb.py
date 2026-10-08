"""
Milvus Vector Store Module for Vision Pipeline Orchestrator.
Provides local embedded vector database storage (Milvus-Lite) for indexing and searching
incident visual embeddings and track crops.
"""

from __future__ import annotations
import os
import hashlib
import logging
from typing import List, Dict, Any, Set, Optional
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Embedded Milvus vector store for image/video clip embeddings.
    Allows similarity search, deduplication, and natural language semantic retrieval.
    """

    def __init__(
        self,
        db_path: str = "data/milvus_orchestrator.db",
        collection_name: str = "vision_events",
        dimension: int = 512,
        metric_type: str = "COSINE",
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type

        # Ensure database directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.client = MilvusClient(db_path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates collection if it doesn't already exist."""
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type=self.metric_type,
                auto_id=False,
            )
            logger.info(f"[+] Created Milvus collection '{self.collection_name}' (dim={self.dimension}, metric={self.metric_type}).")
        else:
            self.client.load_collection(self.collection_name)

    def clean_and_recreate(self) -> None:
        """Drop existing collection and create a fresh new collection in Milvus."""
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info(f"[+] Dropped old Milvus collection '{self.collection_name}'.")
        except Exception as e:
            logger.warning(f"Error dropping Milvus collection: {e}")
        self._ensure_collection()
        logger.info(f"[+] Cleaned and created fresh Milvus collection '{self.collection_name}'.")

    def get_indexed_paths(self) -> Set[str]:
        """Retrieves a set of all file paths already stored in the database."""
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=["file_path"],
                limit=16384,
            )
            return {os.path.abspath(item["file_path"]) for item in results if "file_path" in item}
        except Exception as e:
            logger.warning(f"Could not query indexed paths: {e}")
            return set()

    @staticmethod
    def generate_id_from_path(file_path: str) -> int:
        """Deterministic integer ID derived from file path hash to prevent collisions."""
        return int(hashlib.md5(os.path.abspath(file_path).encode("utf-8")).hexdigest()[:8], 16)

    def insert_batch(self, vectors: List[List[float]], file_paths: List[str]) -> None:
        """Inserts new image embeddings incrementally."""
        rows = []
        for vec, path in zip(vectors, file_paths):
            rows.append({
                "id": self.generate_id_from_path(path),
                "vector": vec,
                "file_path": os.path.abspath(path),
            })

        if rows:
            self.client.insert(collection_name=self.collection_name, data=rows)
            self.client.load_collection(self.collection_name)
            logger.info(f"[+] Inserted {len(rows)} embeddings into Milvus '{self.collection_name}'.")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search top-K nearest neighbors using cosine similarity."""
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["file_path"],
        )
        if not search_res or len(search_res[0]) == 0:
            return []

        formatted_results = []
        for hit in search_res[0]:
            entity = hit.get("entity", {})
            formatted_results.append({
                "id": hit.get("id"),
                "file_path": entity.get("file_path", ""),
                "score": float(hit.get("distance", 0.0)),
            })
        return formatted_results
