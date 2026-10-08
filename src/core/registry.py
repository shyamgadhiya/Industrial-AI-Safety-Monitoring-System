"""
Central Model Registry and Factory for Vision Pipeline Orchestrator.
Manages model instances, instantiation, hardware allocation, and runtime swaps.
"""

from __future__ import annotations
import logging
from typing import Dict, Type, Optional, Any
from src.core.base_model import BaseVisionModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for models. Allows dynamic registration, lazy-loading,
    and runtime discovery of detectors, trackers, pose estimators, segmenters, and zero-shot models.
    """
    _registry: Dict[str, Type[BaseVisionModel]] = {}
    _active_instances: Dict[str, BaseVisionModel] = {}

    @classmethod
    def register(cls, stage_type: str):
        """Decorator to register a model class under a specific key."""
        def decorator(subclass: Type[BaseVisionModel]):
            cls._registry[stage_type.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, stage_type: str, device: str = "cpu", options: Optional[Dict[str, Any]] = None) -> BaseVisionModel:
        """Factory method to instantiate a registered model."""
        key = stage_type.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Model '{key}' not found in registry. Available models: {available}")
        
        model_cls = cls._registry[key]
        instance = model_cls(device=device, options=options or {})
        cls._active_instances[key] = instance
        return instance

    @classmethod
    def get_instance(cls, stage_type: str) -> Optional[BaseVisionModel]:
        return cls._active_instances.get(stage_type.lower())

    @classmethod
    def list_available(cls) -> Dict[str, str]:
        """Return dict of registered model names and classes."""
        return {k: v.__name__ for k, v in cls._registry.items()}

    @classmethod
    def clear_instances(cls) -> None:
        """Unload all active model instances to free memory."""
        cls._active_instances.clear()
