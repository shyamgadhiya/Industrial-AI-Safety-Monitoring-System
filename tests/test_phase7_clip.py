"""Unit tests for Phase 7: CLIP Zero-Shot Verifier."""
import numpy as np
from src.core.types import BoundingBox, Track
from src.core.registry import ModelRegistry
from src.models.zero_shot_clip import ZeroShotCLIPVerifier


def test_clip_verifier():
    verifier = ModelRegistry.create("clip")
    verifier.initialize()

    dummy_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    track = Track(
        track_id=1,
        detection_id="det_clip_01",
        class_name="person",
        bbox=BoundingBox(100, 100, 200, 300),
        score=0.95
    )

    results = verifier.predict(
        dummy_frame,
        [track],
        candidate_queries=[
            "worker wearing high visibility safety vest and hardhat",
            "worker not wearing safety vest"
        ]
    )

    assert 1 in results
    scores = results[1]
    print("CLIP Query Similarity Scores for Track 1:", scores)
    assert len(scores) == 2

    prov = verifier.get_provenance()
    assert prov["model_name"] == "clip-vit-b-32"
    print("Phase 7 CLIP zero-shot verifier passed successfully!")


if __name__ == "__main__":
    test_clip_verifier()
