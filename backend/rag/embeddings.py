from typing import List
from sentence_transformers import SentenceTransformer
from backend.utils.config import settings
from backend.utils.logging import logger

_model_cache = {}

class EmbeddingManager:
    @staticmethod
    def get_model(model_name: str = None) -> SentenceTransformer:
        if model_name is None:
            model_name = settings.EMBEDDING_MODEL_NAME
            
        if model_name not in _model_cache:
            logger.info(f"Loading embedding model: {model_name}...")
            # Load model (can specify device='cuda' if GPU is available)
            try:
                _model_cache[model_name] = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded embedding model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model {model_name}: {e}")
                # Fallback to default model if custom fails
                default_name = "sentence-transformers/all-MiniLM-L6-v2"
                if default_name not in _model_cache:
                    logger.info(f"Loading fallback default model: {default_name}...")
                    _model_cache[default_name] = SentenceTransformer(default_name)
                return _model_cache[default_name]
                
        return _model_cache[model_name]

    @classmethod
    def embed_query(cls, text: str, model_name: str = None) -> List[float]:
        model = cls.get_model(model_name)
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    @classmethod
    def embed_documents(cls, texts: List[str], model_name: str = None) -> List[List[float]]:
        if not texts:
            return []
        model = cls.get_model(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
