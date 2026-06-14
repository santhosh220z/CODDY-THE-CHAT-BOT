import os
import json
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import chromadb
import faiss
from backend.utils.config import settings
from backend.utils.logging import logger

class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(self, document_id: int, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add document chunks and their embeddings to the vector store."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the vector store for similar chunks. Returns list of chunk metadata and text."""
        pass

    @abstractmethod
    def delete_document_chunks(self, document_id: int):
        """Delete all chunks related to a document ID from the store."""
        pass

class ChromaVectorStore(BaseVectorStore):
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="coddy_chunks",
            metadata={"hnsw:space": "cosine"} # Use Cosine similarity
        )
        logger.info(f"Initialized ChromaDB vector store at: {settings.CHROMA_DB_PATH}")

    def add_chunks(self, document_id: int, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        ids = [str(c["chunk_index"]) + "_" + str(document_id) for c in chunks]
        documents = [c["text"] for c in chunks]
        
        # Flatten metadata dict for Chroma (Chroma accepts string, int, float, bool)
        metadatas = []
        for c in chunks:
            meta = c["metadata"].copy()
            meta["document_id"] = document_id
            metadatas.append(meta)
            
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Added {len(chunks)} chunks for document ID {document_id} to ChromaDB.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        if results and results["documents"] and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0]
            
            for i in range(len(docs)):
                formatted_results.append({
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "score": 1.0 - distances[i] # Chroma returns distance. Cosine distance -> cosine similarity
                })
        return formatted_results

    def delete_document_chunks(self, document_id: int):
        # Delete using metadata filter
        self.collection.delete(
            where={"document_id": document_id}
        )
        logger.info(f"Deleted chunks for document ID {document_id} from ChromaDB.")


class FAISSVectorStore(BaseVectorStore):
    def __init__(self):
        self.index_file = os.path.join(settings.FAISS_DB_PATH, "faiss_index.bin")
        self.meta_file = os.path.join(settings.FAISS_DB_PATH, "faiss_metadata.pkl")
        
        os.makedirs(settings.FAISS_DB_PATH, exist_ok=True)
        
        self.dimension = 384 # Default dimension for all-MiniLM-L6-v2. Adjusts dynamically if needed.
        self.index = None
        self.metadata_store: List[Dict[str, Any]] = [] # Index matches with row in FAISS index
        
        self._load_index()
        logger.info(f"Initialized FAISS vector store at: {settings.FAISS_DB_PATH}")

    def _load_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.meta_file, "rb") as f:
                    self.metadata_store = pickle.load(f)
                self.dimension = self.index.d
                logger.info(f"Loaded existing FAISS index with {len(self.metadata_store)} records.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Recreating...")
                self._create_empty_index()
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        # We use IndexFlatIP with normalized vectors for Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []
        self._save_index()

    def _save_index(self):
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.meta_file, "wb") as f:
                pickle.dump(self.metadata_store, f)
        except Exception as e:
            logger.error(f"Error saving FAISS store: {e}")

    def add_chunks(self, document_id: int, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        if not chunks:
            return
            
        emb_np = np.array(embeddings).astype("float32")
        # Ensure dimensionality matches
        if self.index is None or self.index.ntotal == 0:
            self.dimension = emb_np.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
            
        # Normalize vectors for Cosine Similarity (Inner Product of L2 normalized vectors)
        faiss.normalize_L2(emb_np)
        
        self.index.add(emb_np)
        
        # Add metadata mapping
        for i, c in enumerate(chunks):
            meta = c["metadata"].copy()
            meta["document_id"] = document_id
            self.metadata_store.append({
                "id": f"{c['chunk_index']}_{document_id}",
                "text": c["text"],
                "metadata": meta
            })
            
        self._save_index()
        logger.info(f"Added {len(chunks)} chunks for document ID {document_id} to FAISS.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
            
        q_np = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(q_np)
        
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_np, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            item = self.metadata_store[idx]
            results.append({
                "id": item["id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "score": float(score) # Cosine similarity score
            })
        return results

    def delete_document_chunks(self, document_id: int):
        if self.index is None or self.index.ntotal == 0:
            return
            
        # FAISS doesn't support easy dynamic deletion in IndexFlatIP without recreating,
        # so we filter out metadata and rebuild the index for consistency.
        new_metadata_store = []
        keep_indices = []
        
        # We need to extract all vectors we want to keep
        vectors_to_keep = []
        for idx, item in enumerate(self.metadata_store):
            if item["metadata"].get("document_id") != document_id:
                new_metadata_store.append(item)
                keep_indices.append(idx)
        
        if len(new_metadata_store) == 0:
            self._create_empty_index()
            return
            
        # Extract vectors using FAISS helper or rebuild from original embeddings (which requires storing them).
        # Since rebuilding from memory is easier, let's extract vectors from the old index.
        # IndexFlatIP allows extracting vectors by ID or reconstruct.
        reconstructed_vectors = []
        for idx in keep_indices:
            vec = self.index.reconstruct(idx)
            reconstructed_vectors.append(vec)
            
        # Create new index and write
        self.index = faiss.IndexFlatIP(self.dimension)
        if reconstructed_vectors:
            vectors_np = np.array(reconstructed_vectors).astype("float32")
            # Re-normalize just to be sure
            faiss.normalize_L2(vectors_np)
            self.index.add(vectors_np)
            
        self.metadata_store = new_metadata_store
        self._save_index()
        logger.info(f"Deleted chunks for document ID {document_id} from FAISS. Index rebuilt.")


# Factory mapping
_store_instance = None

def get_vector_store() -> BaseVectorStore:
    global _store_instance
    if _store_instance is not None:
        return _store_instance
        
    store_type = settings.VECTOR_STORE_TYPE.lower()
    if store_type == "chroma":
        _store_instance = ChromaVectorStore()
    elif store_type == "faiss":
        _store_instance = FAISSVectorStore()
    else:
        logger.warning(f"Unknown vector store type: '{store_type}'. Defaulting to ChromaDB.")
        _store_instance = ChromaVectorStore()
        
    return _store_instance
