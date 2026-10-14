"""Shared test fixtures.

The SQLite engine uses StaticPool so every session sees the same in-memory
database. Without it SQLite hands each thread its own empty database and the
tables silently vanish when FastAPI runs a handler on a worker thread.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.database.base import Base  # noqa: E402
from backend.database.connection import get_db  # noqa: E402
from backend.models.models import ChatMessage, ChatSession, Document, DocumentChunk, User  # noqa: E402

VALID_PASSWORD = "ValidPass123!"


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    """Isolated database and vector store per test."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("FAISS_DB_PATH", str(tmp_path / "faiss"))

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def session(engine):
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = factory()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(engine):
    """TestClient wired to the in-memory database."""
    from backend.main import create_app

    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Rate limiters are module-level singletons shared across tests.

    Without this, one test exhausting the registration quota makes later
    tests see 429 instead of the status they are asserting.
    """
    from backend.utils.rate_limit import login_limiter, register_limiter

    login_limiter._hits.clear()
    register_limiter._hits.clear()
    yield
    login_limiter._hits.clear()
    register_limiter._hits.clear()


@pytest.fixture()
def vector_store(engine):
    """A vector store bound to the test's temporary directory."""
    from backend.rag.vector_store import ChromaVectorStore

    return ChromaVectorStore()


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def make_user(db, email, role="user", password=VALID_PASSWORD):
    from backend.auth.security import hash_password

    user = User(email=email, hashed_password=hash_password(password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def make_admin(db, email="admin@example.com"):
    return make_user(db, email, role="admin")


def make_document(db, owner, name, chunks_text, file_type="txt"):
    import json

    document = Document(
        name=name,
        file_type=file_type,
        size_bytes=len(chunks_text),
        storage_path=f"/tmp/{name}",
        owner_id=owner.id,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    rows = []
    for index, text in enumerate(chunks_text, start=1):
        row = DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            text=text,
            metadata_json=json.dumps(
                {"file_name": name, "page_number": 1, "user_id": owner.id}
            ),
        )
        db.add(row)
        rows.append(row)
    db.commit()
    return document, rows


def make_session(db, user, title="Test", message_count=0):
    import datetime as dt
    import uuid

    session = ChatSession(id=str(uuid.uuid4()), user_id=user.id, title=title)
    db.add(session)
    db.commit()

    base = dt.datetime(2026, 1, 1, 12, 0, 0)
    for index in range(message_count):
        db.add(
            ChatMessage(
                session_id=session.id,
                sender="user" if index % 2 == 0 else "assistant",
                content=f"message {index}",
                created_at=base + dt.timedelta(minutes=index),
            )
        )
    db.commit()
    return session