from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.models import ChatMessage, Document, DocumentChunk, User
from backend.rag.embeddings import EmbeddingManager
from backend.rag.vector_store import get_vector_store
from backend.utils.config import settings
from backend.utils.logging import logger

# How many recent turns are replayed to the model.
HISTORY_MESSAGES = 10

# ---------------------------------------------------------------------------
# BM25 cache
#
# Rebuilding the index over every chunk on every question is O(total corpus
# tokens) per query. The corpus only changes when a document is added, removed
# or reindexed, so we build once per (revision, scope) and reuse it.
# ---------------------------------------------------------------------------
_cache: "OrderedDict[Tuple[int, int], Dict[str, Any]]" = OrderedDict()
_CACHE_MAX = 4
_revision = 0
_lock = threading.Lock()


def invalidate_retrieval_cache() -> None:
    """Call after any write that changes the searchable corpus."""
    global _revision
    with _lock:
        _revision += 1


def accessible_document_ids(db: Session, user: User) -> Optional[Set[int]]:
    """Which documents this user may retrieve from.

    None means unrestricted (admin). This is the single definition of
    visibility; the vector store and BM25 both derive from it so the two
    retrieval paths cannot drift apart and disagree about who can see what.
    """
    if user.role == "admin":
        return None
    rows = db.execute(select(Document.id).where(Document.owner_id == user.id)).all()
    return {row[0] for row in rows}


def _scope_filter(db: Session, user: User) -> Tuple[Optional[Set[int]], Optional[int]]:
    """(document id set, vector-store tenant) for this user.

    The tenant is None for admins, which the vector store treats as
    "unrestricted". Keeping it as a single Optional[int] means both backends
    receive the same signal instead of two differently-shaped filter dialects.
    """
    doc_ids = accessible_document_ids(db, user)
    return doc_ids, (None if doc_ids is None else user.id)


def _bm25_entry(db: Session, doc_ids: Optional[Set[int]], scope_key: int) -> Dict[str, Any]:
    """BM25 index over one visibility scope, built once per revision."""
    key = (_revision, scope_key)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    stmt = select(DocumentChunk)
    if doc_ids is not None:
        if not doc_ids:
            entry: Dict[str, Any] = {"index": None, "chunks": []}
            with _lock:
                _cache[key] = entry
            return entry
        stmt = stmt.where(DocumentChunk.document_id.in_(doc_ids))

    rows = db.execute(stmt).scalars().all()

    # One bulk fetch for names instead of a query per chunk.
    names = {
        row[0]: row[1]
        for row in db.execute(select(Document.id, Document.name)).all()
    }

    chunks: List[Dict[str, Any]] = []
    for row in rows:
        file_name = names.get(row.document_id, "unknown")
        try:
            metadata = json.loads(row.metadata_json or "{}")
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("file_name", file_name)
        metadata.setdefault("page_number", 1)

        chunks.append(
            {
                "id": f"{row.chunk_index}_{row.document_id}",
                "text": row.text,
                "metadata": metadata,
            }
        )

    index = None
    if chunks:
        try:
            corpus = [[t for t in c["text"].lower().split() if t] for c in chunks]
            index = BM25Okapi(corpus)
        except Exception as exc:
            logger.error(f"BM25 index could not be built: {exc}")

    entry = {"index": index, "chunks": chunks}
    with _lock:
        _cache[key] = entry
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return entry


def _min_max(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    return [(s - lo) / span for s in scores]


class RagPipeline:
    """Retrieval and prompt assembly."""

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    @classmethod
    def retrieve(cls, query: str, db: Session, user: User, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = limit or settings.RAG_TOP_K
        doc_ids, tenant = _scope_filter(db, user)

        # Over-fetch so a filtered backend can still return `limit` real hits.
        fetch = max(limit, limit * 3)

        vector_hits = get_vector_store().search(
            EmbeddingManager.embed_query(query), top_k=fetch, user_id=tenant
        )

        if not settings.HYBRID_SEARCH:
            return vector_hits[:limit]

        scope_key = 0 if doc_ids is None else user.id
        entry = _bm25_entry(db, doc_ids, scope_key)
        if entry["index"] is None:
            return vector_hits[:limit]

        scores = entry["index"].get_scores([t for t in query.lower().split() if t])
        bm25_hits = []
        for i, chunk in enumerate(entry["chunks"]):
            bm25_hits.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(scores[i]),
                }
            )
        bm25_hits.sort(key=lambda h: h["score"], reverse=True)
        bm25_hits = bm25_hits[:fetch]

        bm25_weight = settings.BM25_WEIGHT
        vector_weight = 1.0 - bm25_weight

        for hit, norm in zip(vector_hits, _min_max([h["score"] for h in vector_hits])):
            hit["norm"] = norm * vector_weight
        for hit, norm in zip(bm25_hits, _min_max([h["score"] for h in bm25_hits])):
            hit["norm"] = norm * bm25_weight

        # Weighted sum of min-max normalised scores. This is not Reciprocal
        # Rank Fusion; the README previously claimed it was.
        merged: Dict[str, Dict[str, Any]] = {}
        for hit in vector_hits:
            merged[hit["id"]] = {
                "text": hit["text"],
                "metadata": hit["metadata"],
                "score": hit["norm"],
            }
        for hit in bm25_hits:
            if hit["id"] in merged:
                merged[hit["id"]]["score"] += hit["norm"]
            else:
                merged[hit["id"]] = {
                    "text": hit["text"],
                    "metadata": hit["metadata"],
                    "score": hit["norm"],
                }

        ranked = sorted(
            ({"id": k, **v} for k, v in merged.items()),
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked[:limit]

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    @classmethod
    def recent_history(cls, db: Session, session_id: str) -> List[ChatMessage]:
        """The LAST N messages, oldest first.

        Ordering ascending and then taking N returns the oldest messages, so a
        long conversation loses the part that matters. The previous
        implementation did exactly that.
        """
        rows = (
            db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(HISTORY_MESSAGES)
            )
            .scalars()
            .all()
        )
        return list(reversed(rows))

    @staticmethod
    def build_prompt(
        query: str, history: Sequence[ChatMessage], chunks: Sequence[Dict[str, Any]]
    ) -> Tuple[str, str]:
        system = (
            "You are CODDY, a retrieval-augmented assistant. Answer strictly from "
            "the CONTEXT below.\n"
            "Rules:\n"
            "1. If the context does not contain the answer, reply exactly: "
            "\"I couldn't find that in your documents.\"\n"
            "2. Do not use outside knowledge, and do not guess.\n"
            "3. Keep answers concise and factual."
        )

        parts: List[str] = []
        if history:
            transcript = "\n".join(f"{m.sender}: {m.content}" for m in history)
            parts.append(f"CONVERSATION SO FAR:\n{transcript}")

        if chunks:
            blocks = "\n\n".join(
                f"[{i}] source={c['metadata'].get('file_name')} "
                f"page={c['metadata'].get('page_number')}\n{c['text']}"
                for i, c in enumerate(chunks, start=1)
            )
            parts.append(f"CONTEXT:\n{blocks}")
        else:
            parts.append("CONTEXT:\n(no documents matched this question)")

        parts.append(f"QUESTION:\n{query}")
        parts.append("ANSWER:")
        return "\n\n".join(parts), system

    @staticmethod
    def build_citations(chunks: Sequence[Dict[str, Any]], preview_chars: int = 220) -> List[Dict[str, Any]]:
        citations = []
        for position, chunk in enumerate(chunks, start=1):
            text = chunk["text"]
            citations.append(
                {
                    "position": position,
                    "file_name": chunk["metadata"].get("file_name", "unknown"),
                    "page_number": chunk["metadata"].get("page_number", 1),
                    "excerpt": text[:preview_chars] + ("..." if len(text) > preview_chars else ""),
                }
            )
        return citations

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    @classmethod
    def answer(cls, query: str, db: Session, user: User, llm) -> Dict[str, Any]:
        chunks = cls.retrieve(query, db, user)
        prompt, system = cls.build_prompt(query, cls.recent_history(db, _session_of(db, user)), chunks)
        return {
            "answer": llm.generate(prompt, system_prompt=system),
            "citations": cls.build_citations(chunks),
        }

    @classmethod
    def stream(
        cls, query: str, session_id: str, db: Session, user: User, llm
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield structured events. The caller owns SSE framing.

        The previous version emitted SSE strings that the endpoint immediately
        stripped and re-parsed, adding a JSON round-trip per token.
        """
        chunks = cls.retrieve(query, db, user)
        prompt, system = cls.build_prompt(query, cls.recent_history(db, session_id), chunks)

        try:
            for token in llm.generate_stream(prompt, system_prompt=system):
                yield {"type": "token", "content": token}
        except Exception as exc:
            logger.error(f"Generation failed: {exc}", exc_info=True)
            yield {"type": "error", "message": "The model failed to generate a response."}

        yield {"type": "citations", "citations": cls.build_citations(chunks)}


def _session_of(db: Session, user: User) -> str:
    """Most recent session for this user. Used by the non-streaming path."""
    from backend.models.models import ChatSession

    session = db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return session.id if session else ""