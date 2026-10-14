from __future__ import annotations

import json
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.database.connection import SessionLocal, get_db
from backend.models.models import ChatMessage, ChatSession, User
from backend.rag.pipeline import RagPipeline
from backend.services.llm import get_llm_provider
from backend.utils.logging import logger

router = APIRouter(prefix="/api/chat", tags=["chat"])


class SessionCreate(BaseModel):
    title: str = Field(default="New conversation", max_length=200)


class SessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str


def _owned_session(db: Session, session_id: str, user: User) -> ChatSession:
    """Fetch a session the caller owns.

    Returns 404 rather than 403 for someone else's session so the response
    does not confirm that the id exists.
    """
    session = db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    ).scalar_one_or_none()
    if session is None or (user.role != "admin" and session.user_id != user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return session


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(
    payload: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    session = ChatSession(id=str(uuid.uuid4()), user_id=current_user.id, title=payload.title)
    db.add(session)
    db.commit()
    return {"session_id": session.id, "title": session.title}


@router.get("/sessions")
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[dict]:
    stmt = select(ChatSession).where(ChatSession.user_id == current_user.id)
    sessions = db.execute(stmt.order_by(ChatSession.created_at.desc())).scalars().all()
    return [
        {"session_id": s.id, "title": s.title, "created_at": s.created_at.isoformat()}
        for s in sessions
    ]


@router.put("/sessions/{session_id}")
def rename_session(
    session_id: str,
    payload: SessionRename,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    session = _owned_session(db, session_id, current_user)
    session.title = payload.title
    db.commit()
    return {"session_id": session.id, "title": session.title}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    session = _owned_session(db, session_id, current_user)
    db.delete(session)
    db.commit()
    return {"deleted": session_id}


@router.get("/sessions/{session_id}/messages")
def session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[dict]:
    _owned_session(db, session_id, current_user)
    messages = db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at, ChatMessage.id)
    ).scalars().all()

    out = []
    for message in messages:
        try:
            sources = json.loads(message.sources_json) if message.sources_json else None
        except (TypeError, ValueError):
            sources = None
        out.append(
            {
                "id": message.id,
                "sender": message.sender,
                "content": message.content,
                "sources": sources,
                "created_at": message.created_at.isoformat(),
            }
        )
    return out


@router.post("")
def chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream an answer as Server-Sent Events.

    Events: {"token": str} | {"citations": [...]} | {"error": str}
    """
    # Fast path: reject an unknown or unowned session with a 404 before any
    # streaming begins. The generator re-checks against its own session, because
    # this one may be deleted between the two.
    _owned_session(db, payload.session_id, current_user)

    def sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    def stream():
        # The generator runs on a worker thread after the request handler has
        # returned, so it gets its own session. Ownership must be re-checked
        # against THIS session: checking on the request session and then writing
        # on another is a time-of-check/time-of-use gap, and a session deleted
        # in between would fail on insert with a confusing error.
        gen_db = SessionLocal()
        try:
            session = gen_db.get(ChatSession, payload.session_id)
            if session is None or (
                current_user.role != "admin" and session.user_id != current_user.id
            ):
                yield sse({"error": "Session not found."})
                return

            llm = get_llm_provider()

            gen_db.add(
                ChatMessage(
                    session_id=payload.session_id,
                    sender="user",
                    content=payload.query,
                )
            )
            gen_db.commit()

            # Give the conversation a useful name the first time it is used.
            if session.title == "New conversation":
                session.title = payload.query[:60]
                gen_db.commit()

            answer_parts: List[str] = []
            citations: List[dict] = []

            for event in RagPipeline.stream(
                payload.query, payload.session_id, gen_db, current_user, llm
            ):
                kind = event.get("type")
                if kind == "token":
                    answer_parts.append(event["content"])
                    yield sse({"token": event["content"]})
                elif kind == "citations":
                    citations = event["citations"]
                    yield sse({"citations": citations})
                elif kind == "error":
                    yield sse({"error": event["message"]})

            gen_db.add(
                ChatMessage(
                    session_id=payload.session_id,
                    sender="assistant",
                    content="".join(answer_parts),
                    sources_json=json.dumps(citations) if citations else None,
                )
            )
            gen_db.commit()

        except Exception as exc:
            logger.error("Chat stream failed", exc_info=True)
            yield sse({"error": "Something went wrong generating a response."})
        finally:
            gen_db.close()

    return _streaming_response(stream())


def _streaming_response(generator):
    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx and similar proxies from buffering the stream into a
            # single lump, which would break incremental rendering.
            "X-Accel-Buffering": "no",
        },
    )