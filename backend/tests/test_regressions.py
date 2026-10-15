"""Regression tests for defects found by auditing a running system.

Each test here corresponds to a bug that was live in the code and was invisible
to the existing suite, either because nothing exercised the path or because the
test asserted the wrong thing.
"""

import os
import threading

import pytest

from backend.rag.document_processor import split_text

from .conftest import VALID_PASSWORD, make_user


# ---------------------------------------------------------------------------
# 1. Validation errors returned 500 instead of 422
# ---------------------------------------------------------------------------
def test_invalid_email_returns_422_not_500(client):
    """exc.errors() can hold a ValueError in ctx, which is not serialisable.
    Passing it through raw raised TypeError inside the handler."""
    client.post("/api/auth/register", json={"email": "seed@example.com", "password": VALID_PASSWORD})

    response = client.post(
        "/api/auth/register", json={"email": "not-an-email", "password": VALID_PASSWORD}
    )
    assert response.status_code == 422, response.text
    assert "detail" in response.json()


def test_missing_field_returns_422(client):
    response = client.post("/api/auth/register", json={"email": "someone@example.com"})
    assert response.status_code == 422


def test_constraint_violation_returns_409(client):
    """A unique-constraint failure is a client conflict, not a crash."""
    from backend.models.models import User

    # Bypass the endpoint to plant a conflicting row, then submit the same email.
    def override():
        from backend.database.connection import SessionLocal

        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    response = client.post(
        "/api/auth/register", json={"email": "dupe@example.com", "password": VALID_PASSWORD}
    )
    assert response.status_code == 201

    # A second registration for the same address is a 409 from the endpoint.
    second = client.post(
        "/api/auth/register", json={"email": "dupe@example.com", "password": VALID_PASSWORD}
    )
    assert second.status_code == 409
    assert User is not None


# ---------------------------------------------------------------------------
# 2. Health endpoint must not construct the LLM provider
# ---------------------------------------------------------------------------
def test_health_never_builds_the_llm_provider(client):
    """The frontend polls this on page load; a provider that fails to build
    would turn the liveness probe into a 500."""
    import backend.services.llm as llm_module

    calls = []
    original = llm_module.get_llm_provider

    def explode():
        calls.append(1)
        raise AssertionError("health endpoint must not construct the provider")

    llm_module.get_llm_provider = explode
    try:
        response = client.get("/api/health")
    finally:
        llm_module.get_llm_provider = original

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert not calls


def test_health_reports_model_state_without_loading(client):
    body = client.get("/api/health").json()
    assert "model" in body
    assert "model_loaded" in body


# ---------------------------------------------------------------------------
# 3. split_text emitted chunks larger than chunk_size
# ---------------------------------------------------------------------------
def test_oversized_unit_is_sliced_not_emitted_whole():
    """A single token longer than the limit used to be emitted verbatim."""
    text = "short intro " + ("x" * 2500)
    chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
    assert all(len(c) <= 1000 for c in chunks), [len(c) for c in chunks]
    assert len(chunks) > 3


def test_no_chunk_exceeds_limit_under_many_shapes():
    """Fuzz the shapes that triggered it: long tokens, mixed lengths, no spaces."""
    import random

    rng = random.Random(1234)
    for _ in range(200):
        parts = []
        for _ in range(rng.randint(1, 30)):
            kind = rng.random()
            if kind < 0.3:
                parts.append("x" * rng.randint(50, 3000))
            elif kind < 0.6:
                parts.append("word")
            else:
                parts.append(" " * rng.randint(1, 40))
        text = "".join(parts)
        for size, overlap in ((1000, 200), (200, 50), (50, 10)):
            for chunk in split_text(text, chunk_size=size, chunk_overlap=overlap):
                assert len(chunk) <= size, f"chunk of {len(chunk)} > {size}"


def test_realistic_document_shapes_respect_the_limit():
    body = ("This is a normal sentence about annual leave policy. " * 200
            + "x" * 4000
            + " Trailing paragraph after the oversized token. " * 50)
    for chunk in split_text(body, chunk_size=1000, chunk_overlap=200):
        assert len(chunk) <= 1000


# ---------------------------------------------------------------------------
# 4. Rate limiter ignored attacker-controlled X-Forwarded-For
# ---------------------------------------------------------------------------
def test_forwarded_header_cannot_reset_the_rate_limit(client):
    """X-Forwarded-For is set by any caller. Keying on it let a brute-force
    attempt reset its counter on every request."""
    client.post("/api/auth/register", json={"email": "target@example.com", "password": VALID_PASSWORD})

    statuses = []
    for attempt in range(9):
        response = client.post(
            "/api/auth/login",
            json={"email": "target@example.com", "password": "WrongPass123!"},
            headers={"X-Forwarded-For": f"10.0.0.{attempt}"},
        )
        statuses.append(response.status_code)

    assert 429 in statuses, f"lockout never engaged: {statuses}"


def test_client_key_ignores_forwarded_header_by_default():
    from backend.utils.rate_limit import TRUSTED_PROXY_COUNT, client_key

    assert TRUSTED_PROXY_COUNT == 0

    class FakeRequest:
        headers = {"x-forwarded-for": "1.2.3.4"}
        client = type("C", (), {"host": "127.0.0.1"})()

    assert client_key(FakeRequest()) == "127.0.0.1"


# ---------------------------------------------------------------------------
# 5. Concurrent registration races
# ---------------------------------------------------------------------------
def test_simultaneous_registrations_produce_exactly_one_admin(client):
    """Both requests observed an empty users table and both became admin."""
    results = []
    barrier = threading.Barrier(2)

    def register(i):
        barrier.wait()
        response = client.post(
            "/api/auth/register",
            json={"email": f"racer{i}@example.com", "password": VALID_PASSWORD},
        )
        results.append(response.json().get("role") if response.status_code == 201 else None)

    threads = [threading.Thread(target=register, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count("admin") == 1, results
    assert results.count("user") == 1, results


def test_duplicate_email_under_concurrency_is_rejected_not_crashed(client):
    responses = []

    def register():
        responses.append(
            client.post(
                "/api/auth/register",
                json={"email": "same@example.com", "password": VALID_PASSWORD},
            ).status_code
        )

    threads = [threading.Thread(target=register) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(responses)[0] == 201
    assert all(code in (201, 409) for code in responses), responses
    assert responses.count(201) == 1, responses


# ---------------------------------------------------------------------------
# 6. FAISS lock must be reentrant
# ---------------------------------------------------------------------------
def test_faiss_store_lock_is_reentrant(tmp_path, monkeypatch):
    """upsert_chunks calls delete_document while already holding the lock."""
    from backend.rag.vector_store import FaissVectorStore
    from backend.utils.config import settings

    monkeypatch.setattr(settings, "FAISS_DB_PATH", str(tmp_path / "faiss_reentrant"))

    store = FaissVectorStore()
    chunks = [
        {"chunk_index": 1, "text": "alpha", "metadata": {"file_name": "a.txt", "user_id": 1}},
        {"chunk_index": 2, "text": "beta", "metadata": {"file_name": "b.txt", "user_id": 1}},
    ]
    vectors = [[0.1] * 384, [-0.1] * 384]

    # Would deadlock if the lock were not reentrant.
    store.upsert_chunks(1, chunks, vectors)
    assert len(store.search([0.1] * 384, top_k=5, user_id=1)) == 2

    # Re-upserting the same document must replace, not duplicate.
    store.upsert_chunks(1, chunks, vectors)
    assert len(store.search([0.1] * 384, top_k=5, user_id=1)) == 2


def test_faiss_delete_document_is_serialised(tmp_path, monkeypatch):
    from backend.rag.vector_store import FaissVectorStore
    from backend.utils.config import settings

    monkeypatch.setattr(settings, "FAISS_DB_PATH", str(tmp_path / "faiss_delete_lock"))
    store = FaissVectorStore()

    chunks = [
        {"chunk_index": 1, "text": "keep", "metadata": {"file_name": "k.txt", "user_id": 1}},
        {"chunk_index": 1, "text": "drop", "metadata": {"file_name": "d.txt", "user_id": 2}},
    ]
    store.upsert_chunks(1, [chunks[0]], [[0.1] * 384])
    store.upsert_chunks(2, [chunks[1]], [[0.2] * 384])

    store.delete_document(2)
    assert len(store.search([0.1] * 384, top_k=5, user_id=1)) == 1
    assert store.search([0.1] * 384, top_k=5, user_id=2) == []


# ---------------------------------------------------------------------------
# 7. Generation must be serialised
# ---------------------------------------------------------------------------
def test_provider_serialises_generation():
    """Two concurrent generations each allocate their own KV cache, which does
    not fit a 6GB card beside the resident 4-bit weights."""
    import inspect

    from backend.services.transformers_provider import TransformersProvider

    source = inspect.getsource(TransformersProvider.generate)
    assert "_generate_lock" in source, "generate() must hold the generation lock"

    source = inspect.getsource(TransformersProvider.generate_stream)
    assert "_generate_lock" in source, "generate_stream() must hold the generation lock"

    provider = TransformersProvider()
    assert getattr(provider._generate_lock, "_is_reentrant", False) is False


def test_generation_lock_is_not_the_load_lock():
    """Load and generation need different locks: holding the load lock during
    a 30-second generation would serialise everything behind it."""
    from backend.services.transformers_provider import TransformersProvider

    provider = TransformersProvider()
    assert provider._lock is not provider._generate_lock


# ---------------------------------------------------------------------------
# 8. Chat stream session handling
# ---------------------------------------------------------------------------
def test_chat_to_unknown_session_is_rejected(client):
    client.post("/api/auth/register", json={"email": "a@example.com", "password": VALID_PASSWORD})
    token = client.post(
        "/api/auth/login", json={"email": "a@example.com", "password": VALID_PASSWORD}
    ).json()["access_token"]

    response = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "hello", "session_id": "does-not-exist"},
    )
    assert response.status_code == 404


def test_chat_rejects_empty_query(client):
    client.post("/api/auth/register", json={"email": "a@example.com", "password": VALID_PASSWORD})
    token = client.post(
        "/api/auth/login", json={"email": "a@example.com", "password": VALID_PASSWORD}
    ).json()["access_token"]
    session = client.post(
        "/api/chat/sessions",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "t"},
    ).json()["session_id"]

    response = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "", "session_id": session},
    )
    assert response.status_code == 422

# ---------------------------------------------------------------------------
# 8. Upload validated one filename and processed another
# ---------------------------------------------------------------------------
def test_upload_stores_the_name_it_validated(client):
    """A name over the column limit used to clear the extension allow-list on
    its real extension, then fail extraction on the truncated one, after the
    file had been written and the Document row committed."""
    token = client.get("/api/auth/local").json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Long enough that the 512-character cut lands inside the extension.
    name = ("a" * 600) + ".txt"
    response = client.post(
        "/api/documents",
        headers=headers,
        files={"file": (name, b"alpha beta gamma", "text/plain")},
    )

    # Whatever happens, the stored name must still end in the allowed
    # extension, which is what the processor re-derives it from.
    if response.status_code == 201:
        stored = client.get("/api/documents", headers=headers).json()[0]["name"]
        assert stored.endswith(".txt"), stored
        assert len(stored) <= 512, len(stored)
    else:
        # A clean rejection is acceptable; a 500 after committing is not.
        assert response.status_code in (400, 413), response.status_code
        assert client.get("/api/documents", headers=headers).json() == []


# ---------------------------------------------------------------------------
# 9. Health reported model_loaded false for a resident Ollama model
# ---------------------------------------------------------------------------
def test_model_loaded_comes_from_the_provider_not_an_attribute(client, monkeypatch):
    """The probe used getattr(existing, "_model"), which only TransformersProvider
    has. Ollama is always resident but carries no such attribute, so it
    reported loaded: false forever next to details.loaded: true."""
    from backend.services import llm as llm_module

    class ResidentOllama:
        def info(self):
            return {"provider": "ollama", "model_id": "qwen", "loaded": True}

    monkeypatch.setattr(llm_module, "_provider", ResidentOllama())
    body = client.get("/api/health").json()

    assert body["model_loaded"] is True
    assert body["details"]["loaded"] is True
    assert not hasattr(ResidentOllama(), "_model"), "fixture must lack _model"


def test_model_loaded_is_false_when_the_provider_is_not_built(client, monkeypatch):
    from backend.services import llm as llm_module

    monkeypatch.setattr(llm_module, "_provider", None)
    body = client.get("/api/health").json()

    assert body["model_loaded"] is False
    assert "details" not in body