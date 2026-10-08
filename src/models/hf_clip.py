"""
Hugging Face CLIP Model Wrapper for Zero-Shot Safety Verification and Vector Embeddings.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Union
import numpy as np
import torch
from PIL import Image
import cv2

try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

logger = logging.getLogger(__name__)


class HFCLIPModel:
    """
    HuggingFace CLIP ViT model.
    Encodes text queries and image crops into normalized 512-dimensional vector embeddings.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = None):
        if device == "auto" or not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_id = model_id
        self.processor: CLIPProcessor = None
        self.model: CLIPModel = None
        self._text_cache: Dict[str, np.ndarray] = {}
        self.initialize()

    def initialize(self) -> None:
        if HAS_CLIP:
            try:
                logger.info(f"Loading CLIP model '{self.model_id}' on {self.device}...")
                self.processor = CLIPProcessor.from_pretrained(self.model_id)
                self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
                logger.info("CLIP model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load CLIP model from {self.model_id}: {e}")
                self.model = None

    def encode_text(self, text: str) -> List[float]:
        """Generate normalized 512-d vector for a text query with memoization."""
        if text in self._text_cache:
            return self._text_cache[text].tolist()

        if not self.model or not self.processor:
            return [0.0] * 512

        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            text_features = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        vec_np = text_features[0].cpu().numpy()
        self._text_cache[text] = vec_np
        return vec_np.tolist()

    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> List[float]:
        """Generate normalized 512-d vector for an image/crop."""
        if not self.model or not self.processor:
            return [0.0] * 512

        if isinstance(image, np.ndarray):
            # Convert BGR to RGB PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb = image
            pil_img = Image.fromarray(rgb)
        else:
            pil_img = image

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
            image_features = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features[0].cpu().numpy().tolist()

    def compute_similarity(self, image: Union[Image.Image, np.ndarray], queries: List[str]) -> Tuple[Dict[str, float], List[float]]:
        """Compute cosine similarity between image crop and multiple text queries, returning scores and image embedding."""
        img_vec_list = self.encode_image(image)
        if not self.model or not self.processor or not queries:
            return {q: 0.5 for q in queries}, img_vec_list

        img_vec = np.array(img_vec_list, dtype=np.float32)
        scores: Dict[str, float] = {}

        for q in queries:
            if q not in self._text_cache:
                self.encode_text(q)
            q_vec = self._text_cache[q]
            cosine = float(np.dot(img_vec, q_vec))
            scores[q] = round(cosine, 4)

        return scores, img_vec_list

    @staticmethod
    def calibrate_score(raw_cosine: float) -> float:
        """Empirical min-max calibration (0.15 -> 0%, 0.38+ -> 100%)."""
        clamped = np.clip((raw_cosine - 0.15) / (0.38 - 0.15), 0.0, 1.0)
        return float(round(clamped * 100, 2))
