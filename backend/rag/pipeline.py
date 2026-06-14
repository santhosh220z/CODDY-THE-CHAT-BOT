import json
from typing import List, Dict, Any, Generator, Tuple
from sqlalchemy.orm import Session
from rank_bm25 import BM25Okapi

from backend.utils.config import settings
from backend.utils.logging import logger
from backend.models.models import DocumentChunk, ChatMessage, Document
from backend.rag.embeddings import EmbeddingManager
from backend.rag.vector_store import get_vector_store
from backend.services.llm import get_llm_provider

class RagPipeline:
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Simple lowercase word tokenizer
        return text.lower().split()

    @classmethod
    def hybrid_search(cls, query: str, db: Session, limit: int = 5) -> List[Dict[str, Any]]:
        vector_store = get_vector_store()
        query_embedding = EmbeddingManager.embed_query(query)
        
        # 1. Get vector search results (retrieve a bit more than limit for reranking/fusion)
        vector_results = vector_store.search(query_embedding, top_k=limit * 3)
        
        if not settings.HYBRID_SEARCH:
            return vector_results[:limit]
            
        # 2. Get all document chunks from database to build BM25 index
        # For performance in large systems, we can pre-filter or cache this. For this system, we query dynamically.
        db_chunks = db.query(DocumentChunk).all()
        if not db_chunks:
            return vector_results[:limit]
            
        corpus = [c.text for c in db_chunks]
        tokenized_corpus = [cls._tokenize(doc) for doc in corpus]
        
        try:
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = cls._tokenize(query)
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Pair each chunk with its BM25 score
            bm25_results = []
            for i, chunk in enumerate(db_chunks):
                # Retrieve document metadata to populate source info
                doc = db.query(Document).filter(Document.id == chunk.document_id).first()
                file_name = doc.name if doc else "Unknown"
                
                # Decode metadata if stored as JSON
                metadata = {}
                try:
                    metadata = json.loads(chunk.metadata_json)
                except Exception:
                    metadata = {
                        "file_name": file_name,
                        "page_number": 1,
                        "source": file_name
                    }
                    
                bm25_results.append({
                    "id": f"{chunk.chunk_index}_{chunk.document_id}",
                    "text": chunk.text,
                    "metadata": metadata,
                    "score": float(bm25_scores[i])
                })
                
            # Sort and take top bm25 hits
            bm25_results = sorted(bm25_results, key=lambda x: x["score"], reverse=True)[:limit * 3]
        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}")
            return vector_results[:limit]
            
        # 3. Hybrid fusion (weighted normalized sum)
        # Normalize vector scores
        v_scores = [r["score"] for r in vector_results]
        v_min, v_max = min(v_scores) if v_scores else 0, max(v_scores) if v_scores else 1
        v_range = (v_max - v_min) if (v_max - v_min) > 0 else 1
        
        for r in vector_results:
            r["norm_score"] = (r["score"] - v_min) / v_range
            
        # Normalize BM25 scores
        b_scores = [r["score"] for r in bm25_results]
        b_min, b_max = min(b_scores) if b_scores else 0, max(b_scores) if b_scores else 1
        b_range = (b_max - b_min) if (b_max - b_min) > 0 else 1
        
        for r in bm25_results:
            r["norm_score"] = (r["score"] - b_min) / b_range
            
        # Merge results by ID
        merged = {}
        bm25_weight = settings.BM25_WEIGHT
        vector_weight = 1.0 - bm25_weight
        
        # Populate from vector results
        for r in vector_results:
            merged[r["id"]] = {
                "text": r["text"],
                "metadata": r["metadata"],
                "score": r["norm_score"] * vector_weight
            }
            
        # Add or combine BM25 results
        for r in bm25_results:
            if r["id"] in merged:
                merged[r["id"]]["score"] += r["norm_score"] * bm25_weight
            else:
                merged[r["id"]] = {
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": r["norm_score"] * bm25_weight
                }
                
        # Sort merged dict items by score descending
        sorted_merged = sorted(
            [{"id": k, **v} for k, v in merged.items()],
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_merged[:limit]

    @classmethod
    def retrieve_relevant_chunks(cls, query: str, db: Session, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            return cls.hybrid_search(query, db, limit=limit)
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            return []

    @classmethod
    def query(cls, query: str, session_id: str, db: Session) -> Dict[str, Any]:
        """Non-streaming query method."""
        llm = get_llm_provider()
        
        # 1. Retrieve history (last 10 messages)
        history = db.query(ChatMessage)\
            .filter(ChatMessage.session_id == session_id)\
            .order_by(ChatMessage.created_at.asc())\
            .limit(10)\
            .all()
            
        # 2. Retrieve chunks
        chunks = cls.retrieve_relevant_chunks(query, db, limit=settings.RAG_TOP_K)
        
        # 3. Format inputs
        history_str = "\n".join([f"{m.sender.capitalize()}: {m.content}" for m in history])
        context_str = "\n\n".join([f"Source: {c['metadata'].get('file_name')} (Page {c['metadata'].get('page_number')})\nContent: {c['text']}" for c in chunks])
        
        # 4. Prompt construction
        system_prompt = (
            "You are CODDY – THE CHAT BOT.\n"
            "Answer only using the provided context.\n"
            "If the answer is not found in the context, respond with:\n"
            "\"I couldn't find that information in the knowledge base.\""
        )
        
        prompt = ""
        if history_str:
            prompt += f"Conversation History:\n{history_str}\n\n"
            
        prompt += f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nAnswer:"
        
        # 5. LLM Call
        response = llm.generate(prompt, system_prompt=system_prompt)
        
        # 6. Citations formatted
        citations = []
        for c in chunks:
            citations.append({
                "file_name": c["metadata"].get("file_name", "Unknown"),
                "page_number": c["metadata"].get("page_number", 1),
                "text_preview": c["text"][:150] + "..."
            })
            
        return {
            "answer": response,
            "citations": citations
        }

    @classmethod
    def query_stream(cls, query: str, session_id: str, db: Session) -> Generator[str, None, None]:
        """Streaming query yielding SSE formatted tokens or JSON chunks."""
        llm = get_llm_provider()
        
        # 1. Retrieve history (last 10 messages)
        history = db.query(ChatMessage)\
            .filter(ChatMessage.session_id == session_id)\
            .order_by(ChatMessage.created_at.asc())\
            .limit(10)\
            .all()
            
        # 2. Retrieve chunks
        chunks = cls.retrieve_relevant_chunks(query, db, limit=settings.RAG_TOP_K)
        
        # 3. Format context & history
        history_str = "\n".join([f"{m.sender.capitalize()}: {m.content}" for m in history])
        context_str = "\n\n".join([f"Source: {c['metadata'].get('file_name')} (Page {c['metadata'].get('page_number')})\nContent: {c['text']}" for c in chunks])
        
        # 4. Prompt construction
        system_prompt = (
            "You are CODDY – THE CHAT BOT.\n"
            "Answer only using the provided context.\n"
            "If the answer is not found in the context, respond with:\n"
            "\"I couldn't find that information in the knowledge base.\""
        )
        
        prompt = ""
        if history_str:
            prompt += f"Conversation History:\n{history_str}\n\n"
            
        prompt += f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nAnswer:"
        
        # 5. Extract citation structures
        citations = []
        for c in chunks:
            citations.append({
                "file_name": c["metadata"].get("file_name", "Unknown"),
                "page_number": c["metadata"].get("page_number", 1),
                "text_preview": c["text"]
            })
            
        # 6. Stream tokens
        # Standard Server-Sent Events pattern
        for token in llm.generate_stream(prompt, system_prompt=system_prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
            
        # 7. Finally, send citations metadata block at the end of the stream
        yield f"data: {json.dumps({'done': True, 'citations': citations})}\n\n"
