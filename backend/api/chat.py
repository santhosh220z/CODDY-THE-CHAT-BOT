import uuid
import json
import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database.connection import get_db, SessionLocal
from backend.models.models import ChatSession, ChatMessage, User
from backend.api.auth import get_current_user
from backend.rag.pipeline import RagPipeline
from backend.utils.logging import logger

router = APIRouter(prefix="/api/chat", tags=["Chat"])

# Pydantic Schemas
class QueryPayload(BaseModel):
    query: str
    session_id: str

class SessionCreate(BaseModel):
    title: Optional[str] = "New Conversation"

class SessionRename(BaseModel):
    title: str

@router.post("/session", status_code=status.HTTP_201_CREATED)
def create_session(
    payload: Optional[SessionCreate] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    title = payload.title if payload else "New Conversation"
    session_id = str(uuid.uuid4())
    
    new_session = ChatSession(
        id=session_id,
        user_id=current_user.id,
        title=title
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    logger.info(f"Created chat session: {session_id} for user: {current_user.email}")
    return {"session_id": session_id, "title": title}

@router.get("/sessions", response_model=List[dict])
def get_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(ChatSession)\
        .filter(ChatSession.user_id == current_user.id)\
        .order_by(ChatSession.created_at.desc())\
        .all()
        
    return [{
        "session_id": s.id,
        "title": s.title,
        "created_at": s.created_at.isoformat()
    } for s in sessions]

@router.put("/sessions/{session_id}")
def rename_session(
    session_id: str,
    payload: SessionRename,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found."
        )
    session.title = payload.title
    db.commit()
    return {"message": "Session renamed successfully", "session_id": session_id, "title": session.title}

@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found."
        )
    db.delete(session)
    db.commit()
    logger.info(f"Deleted chat session: {session_id}")
    return {"message": "Session deleted successfully."}

@router.delete("/sessions")
def clear_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(ChatSession).filter(ChatSession.user_id == current_user.id).delete()
    db.commit()
    logger.info(f"Cleared all chat sessions for user: {current_user.email}")
    return {"message": "All conversations cleared successfully."}

@router.get("/history/{session_id}", response_model=List[dict])
def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify session ownership
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found."
        )
        
    messages = db.query(ChatMessage)\
        .filter(ChatMessage.session_id == session_id)\
        .order_by(ChatMessage.created_at.asc())\
        .all()
        
    result = []
    for m in messages:
        sources = None
        if m.sources_json:
            try:
                sources = json.loads(m.sources_json)
            except Exception:
                pass
        result.append({
            "id": m.id,
            "sender": m.sender,
            "content": m.content,
            "sources": sources,
            "created_at": m.created_at.isoformat()
        })
    return result

@router.post("")
def chat(
    payload: QueryPayload,
    current_user: User = Depends(get_current_user)
):
    # Streaming endpoint returning EventSource protocol
    query = payload.query
    session_id = payload.session_id
    
    def event_stream_generator():
        # Open separate session for thread-safety in background generator
        gen_db = SessionLocal()
        try:
            # 1. Verify session exists and belongs to current user
            sess = gen_db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
            if not sess:
                yield f"data: {json.dumps({'error': 'Chat session not found or access denied.'})}\n\n"
                return
                
            # 2. Save User Message
            user_msg = ChatMessage(
                session_id=session_id,
                sender="user",
                content=query
            )
            gen_db.add(user_msg)
            gen_db.commit()
            
            # Update session title if it is still default "New Conversation" to match query snippet
            if sess.title == "New Conversation":
                sess.title = query[:30] + ("..." if len(query) > 30 else "")
                gen_db.commit()
                
            # 3. Process RAG Stream
            accumulated_answer = ""
            citations = []
            
            for chunk_sse in RagPipeline.query_stream(query, session_id, gen_db):
                # Clean prefix 'data: ' and suffix '\n\n' to extract json payload
                clean_payload = chunk_sse.strip().replace("data: ", "")
                if not clean_payload:
                    continue
                    
                try:
                    payload_dict = json.loads(clean_payload)
                    
                    if "token" in payload_dict:
                        accumulated_answer += payload_dict["token"]
                        
                    if "citations" in payload_dict:
                        citations = payload_dict["citations"]
                except Exception as parse_error:
                    logger.error(f"Error parsing SSE chunk: {parse_error}")
                    
                yield chunk_sse
                
            # 4. Save Assistant response with citations
            assistant_msg = ChatMessage(
                session_id=session_id,
                sender="assistant",
                content=accumulated_answer,
                sources_json=json.dumps(citations) if citations else None
            )
            gen_db.add(assistant_msg)
            gen_db.commit()
            
        except Exception as err:
            logger.error(f"Error in chat SSE stream: {err}")
            yield f"data: {json.dumps({'error': f'Generation error: {str(err)}'})}\n\n"
        finally:
            gen_db.close()
            
    return StreamingResponse(event_stream_generator(), media_type="text/event-stream")
