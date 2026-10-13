from __future__ import annotations

import threading
from typing import Dict, List, Optional

from backend.utils.config import settings
from backend.utils.logging import logger

_model_cache: Dict[str, object] = {}
_lock = threading.Lock()


def _resolve_device() -> Optional[str]:
    if settings.EMBEDDING_DEVICE:
        return settings.EMBEDDING_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class EmbeddingManager:
    """Loads and caches the sentence-embedding model.

    The model is small (~90MB) but loading it costs seconds, so it is resolved
    once and reused. A failed load falls back to the default model and, unlike
    the previous implementation, caches the fallback under the requested name
    too - otherwise every later request retried the broken name and re-entered
    the download path.
    """

    @staticmethod
    def _load(name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name, device=_resolve_device())

    @classmethod
    def get_model(cls, model_name: Optional[str] = None):
        name = model_name or settings.EMBEDDING_MODEL_NAME

        with _lock:
            if name in _model_cache:
                return _model_cache[name]

        logger.info(f"Loading embedding model: {name}")
        try:
            model = cls._load(name)
        except Exception as exc:
            logger.error(f"Could not load embedding model '{name}': {exc}")
            default = "sentence-transformers/all-MiniLM-L6-v2"
            with _lock:
                if default in _model_cache:
                    return _model_cache[default]
            model = cls._load(default)
            with _lock:
                _model_cache[default] = model
                # Cache under the broken name too so the retry cost is paid once.
                _model_cache[name] = model
            return model

        with _lock:
            _model_cache[name] = model
        logger.info(f"Embedding model ready: {name}")
        return model

    @classmethod
    def embed_query(cls, text: str, model_name: Optional[str] = None) -> List[float]:
        vector = cls.get_model(model_name).encode(text, convert_to_numpy=True)
        return vector.tolist()

    @classmethod
    def embed_documents(cls, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []
        vectors = cls.get_model(model_name).encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    @classmethod
    def preload(cls) -> None:
        """Warm the cache at startup so the first chat request is not slow."""
        cls.get_model()