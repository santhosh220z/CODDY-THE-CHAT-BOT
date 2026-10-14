from __future__ import annotations

import os
import pickle
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from backend.utils.config import settings
from backend.utils.logging import logger


class BaseVectorStore(ABC):
    """Vector index for document chunks.

    `search` takes a required `user_id`. It is deliberately not an optional
    filter dict: a filter with a default is a filter that eventually gets
    omitted, and omitting it here returns every user's content to anyone who
    asks. Passing the tenant explicitly means the unrestricted case has to be
    spelled out as `None`.
    """

    @abstractmethod
    def upsert_chunks(
        self, document_id: int, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        ...

    @abstractmethod
    def search(
        self, query_embedding: List[float], top_k: int, user_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Return at most `top_k` chunks owned by `user_id`, best first.

        `user_id=None` means unrestricted (administrators).
        """

    @abstractmethod
    def delete_document(self, document_id: int) -> None:
        ...


def _visible_to(metadata: Dict[str, Any], user_id: Optional[int]) -> bool:
    """Tenant predicate. Chunks missing user_id are never visible."""
    if user_id is None:
        return "user_id" in metadata
    return metadata.get("user_id") == user_id


class ChromaVectorStore(BaseVectorStore):
    COLLECTION = "coddy_chunks"
    COLLECTION_METADATA = {"hnsw:space": "cosine"}

    def __init__(self) -> None:
        import chromadb

        self._lock = threading.Lock()
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION, metadata=self.COLLECTION_METADATA
        )
        # get_or_create_collection ignores `metadata` for an existing
        # collection. A mismatched space silently produces wrong distances.
        existing = self.collection.metadata or {}
        drift = {
            k: (existing.get(k), v)
            for k, v in self.COLLECTION_METADATA.items()
            if existing.get(k) != v
        }
        if drift:
            logger.warning(
                f"Chroma collection '{self.COLLECTION}' metadata drift {drift} "
                f"(found, expected). Scores may be wrong. Delete "
                f"{settings.CHROMA_DB_PATH} and re-upload your documents."
            )
        logger.info(f"Chroma store ready at {settings.CHROMA_DB_PATH}")

    def upsert_chunks(
        self, document_id: int, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        if not chunks:
            return
        ids = [f"{c['chunk_index']}_{document_id}" for c in chunks]
        metadatas = []
        for chunk in chunks:
            meta = {k: v for k, v in chunk["metadata"].items() if v is not None}
            meta["document_id"] = int(document_id)
            metadatas.append(meta)

        with self._lock:
            # upsert, not add: `add` raises on duplicate ids, which broke
            # reindex whenever the preceding delete had not fully landed.
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=[c["text"] for c in chunks],
                metadatas=metadatas,
            )
        logger.info(f"Upserted {len(chunks)} chunks for document {document_id}")

    def search(
        self, query_embedding: List[float], top_k: int, user_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        where = None if user_id is None else {"user_id": int(user_id)}
        result = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )
        if not result or not result.get("documents") or not result["documents"][0]:
            return []

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result.get("distances", [[0.0] * len(docs)])[0]
        ids = result["ids"][0]

        # Re-check locally: never trust the backend to have honoured the filter.
        out = []
        for i in range(len(docs)):
            metadata = metas[i] or {}
            if _visible_to(metadata, user_id):
                out.append(
                    {
                        "id": ids[i],
                        "text": docs[i],
                        "metadata": metadata,
                        "score": 1.0 - dists[i],
                    }
                )
        return out

    def delete_document(self, document_id: int) -> None:
        with self._lock:
            self.collection.delete(where={"document_id": int(document_id)})
        logger.info(f"Removed vectors for document {document_id}")


class FaissVectorStore(BaseVectorStore):
    """Flat inner-product index over L2-normalised vectors (cosine).

    IndexFlatIP scans everything anyway, so filtering after the scan is cheap.
    """

    def __init__(self) -> None:
        import faiss

        os.makedirs(settings.FAISS_DB_PATH, exist_ok=True)
        self._index_file = os.path.join(settings.FAISS_DB_PATH, "index.faiss")
        self._meta_file = os.path.join(settings.FAISS_DB_PATH, "index_meta.pkl")
        # Reentrant: upsert_chunks calls delete_document while already holding
        # this lock, which a plain Lock would deadlock.
        self._lock = threading.RLock()
        self.dimension = 0
        self._index = None
        self._records: List[Dict[str, Any]] = []
        self._faiss = faiss
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._index_file) and os.path.exists(self._meta_file):
            try:
                with open(self._meta_file, "rb") as fh:
                    self._records = pickle.load(fh)
                self._index = self._faiss.read_index(self._index_file)
                self.dimension = self._index.d
                logger.info(f"FAISS store loaded: {len(self._records)} vectors")
                return
            except Exception as exc:
                logger.error(f"FAISS index unreadable, rebuilding empty: {exc}")
        self._records = []
        self._index = self._faiss.IndexFlatIP(0)
        self.dimension = 0

    def _persist(self) -> None:
        os.makedirs(settings.FAISS_DB_PATH, exist_ok=True)
        self._faiss.write_index(self._index, self._index_file)
        with open(self._meta_file, "wb") as fh:
            pickle.dump(self._records, fh)

    def upsert_chunks(
        self, document_id: int, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype="float32")
        self._faiss.normalize_L2(vectors)

        with self._lock:
            # Drop any prior copy of this document so ids stay unique.
            self.delete_document(document_id)

            if self._index.ntotal == 0:
                self.dimension = vectors.shape[1]
                self._index = self._faiss.IndexFlatIP(self.dimension)

            self._index.add(vectors)
            for i, chunk in enumerate(chunks):
                meta = {k: v for k, v in chunk["metadata"].items() if v is not None}
                meta["document_id"] = int(document_id)
                self._records.append(
                    {
                        "id": f"{chunk['chunk_index']}_{document_id}",
                        "text": chunk["text"],
                        "metadata": meta,
                    }
                )
            self._persist()
        logger.info(f"Upserted {len(chunks)} chunks for document {document_id}")

    def search(
        self, query_embedding: List[float], top_k: int, user_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            query = np.asarray([query_embedding], dtype="float32")
            self._faiss.normalize_L2(query)
            # Full scan, then filter, so we can still return top_k hits.
            scores, indices = self._index.search(query, self._index.ntotal)
            results: List[Dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._records):
                    continue
                record = self._records[idx]
                if _visible_to(record["metadata"], user_id):
                    results.append(
                        {
                            "id": record["id"],
                            "text": record["text"],
                            "metadata": record["metadata"],
                            "score": float(score),
                        }
                    )
                    if len(results) >= top_k:
                        break
            return results

    def delete_document(self, document_id: int) -> None:
        if self._index is None or self._index.ntotal == 0:
            return

        # Locked: this reassigns _index and _records and rewrites disk state,
        # so a concurrent search indexing _records would race.
        with self._lock:
            keep = [
                i for i, r in enumerate(self._records)
                if r["metadata"].get("document_id") != int(document_id)
            ]
            removed = len(self._records) - len(keep)
            if removed == 0:
                return

            if keep:
                vectors = np.vstack(
                    [self._index.reconstruct(i) for i in keep]
                ).astype("float32")
                self._faiss.normalize_L2(vectors)
                self._index = self._faiss.IndexFlatIP(vectors.shape[1])
                self._index.add(vectors)
                self._records = [self._records[i] for i in keep]
            else:
                self._records = []
                self._index = self._faiss.IndexFlatIP(self.dimension or 1)

            self._persist()
        logger.info(f"Removed {removed} vectors for document {document_id}")


_store: Optional[BaseVectorStore] = None
_store_lock = threading.Lock()


def get_vector_store() -> BaseVectorStore:
    """Process-wide singleton. Loading an index on every request is wasteful."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ChromaVectorStore() if settings.VECTOR_STORE_TYPE == "chroma" else FaissVectorStore()
    return _store


def reset_vector_store() -> None:
    """Test hook: drop the cached instance."""
    global _store
    with _store_lock:
        _store = None