"""
Abstract base class for all vision modules.
Ensures every model is independently callable and provides provenance tracking.
"""

from __future__ import annotations
import abc
import time
from typing import Dict, Any, Tuple


class BaseVisionModel(abc.ABC):
    """
    Standard interface for all Vision Models in the Orchestrator.
    Every model is independently callable, benchmarkable, and version-tracked.
    """

    def __init__(self, name: str, version: str, device: str = "cpu", options: Dict[str, Any] = None):
        self.name = name
        self.version = version
        self.device = device
        self.options = options or {}
        self._is_initialized = False

    @abc.abstractmethod
    def initialize(self) -> None:
        """Load weights, build computation graphs, allocate device memory."""
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Execute inference."""
        pass

    def get_provenance(self) -> Dict[str, str]:
        """Return model metadata for event auditing and compliance."""
        return {
            "model_name": self.name,
            "model_version": self.version,
            "device": self.device,
        }

    def benchmark(self, dummy_input: Any, warmup_runs: int = 2, benchmark_runs: int = 5) -> Dict[str, float]:
        """Benchmark model execution time on target device."""
        if not self._is_initialized:
            self.initialize()

        # Warmup
        for _ in range(warmup_runs):
            self.predict(dummy_input)

        latencies = []
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            self.predict(dummy_input)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        return {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "fps_estimate": round(1000.0 / (sum(latencies) / len(latencies)), 1),
        }
