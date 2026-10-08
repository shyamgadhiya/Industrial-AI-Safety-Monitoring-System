"""Ingestion module initialization."""
from src.ingestion.reader import VideoStreamReader
from src.ingestion.sampler import FrameSampler

__all__ = ["VideoStreamReader", "FrameSampler"]
