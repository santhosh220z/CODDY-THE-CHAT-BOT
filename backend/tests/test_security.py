"""Security regression tests.

Each test here corresponds to a vulnerability that existed in the previous
version of this project. They are written so a future change that reintroduces
the bug fails loudly rather than silently.
"""

import pytest

from backend.api.auth import UserRegister
from backend.auth.security import (
    PasswordPolicyError,
    create_access_token,
    decode_access_token,
    hash_password,
    validate_password_strength,
    verify_password,
)
from backend.rag.vector_store import _visible_to
from backend.utils.config import Settings
from backend.utils.validators import normalise_email

from .conftest import VALID_PASSWORD, make_user


# ---------------------------------------------------------------------------
# 1. Privilege escalation
# ---------------------------------------------------------------------------
def test_register_schema_has_no_role_field():
    """The body must not be able to carry a role at all."""
    parsed = UserRegister(email="attacker@example.com", password=VALID_PASSWORD, role="admin")
    assert "role" not in parsed.model_dump()
    assert not hasattr(parsed, "role")


def test_first_account_is_admin_and_the_rest_are_not(client):
    first = client.post(
        "/api/auth/register", json={"email": "first@example.com", "password": VALID_PASSWORD}
    )
    assert first.status_code == 201
    assert first.json()["role"] == "admin"

    second = client.post(
        "/api/auth/register", json={"email": "second@example.com", "password": VALID_PASSWORD}
    )
    assert second.status_code == 201
    assert second.json()["role"] == "user"


def test_role_in_request_body_is_ignored(client):
    # Bootstrap the first account so the attacker below is not the admin.
    client.post("/api/auth/register", json={"email": "owner@example.com", "password": VALID_PASSWORD})

    response = client.post(
        "/api/auth/register",
        json={"email": "attacker@example.com", "password": VALID_PASSWORD, "role": "admin"},
    )
    assert response.status_code == 201
    assert response.json()["role"] == "user"

    login = client.post(
        "/api/auth/login", json={"email": "attacker@example.com", "password": VALID_PASSWORD}
    )
    assert login.json()["role"] == "user"


def test_unknown_endpoint_is_404(client):
    """The admin stats endpoint was removed; it must not have crept back."""
    client.post("/api/auth/register", json={"email": "a@example.com", "password": VALID_PASSWORD})
    token = client.post(
        "/api/auth/login", json={"email": "a@example.com", "password": VALID_PASSWORD}
    ).json()["access_token"]

    response = client.get("/api/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 2. Cross-tenant data leakage
# ---------------------------------------------------------------------------
def test_tenant_predicate_never_matches_a_missing_user_id():
    """Chunks without ownership metadata must never be visible."""
    assert _visible_to({}, 1) is False
    assert _visible_to({"user_id": None}, 1) is False
    assert _visible_to({"user_id": 2}, 1) is False
    assert _visible_to({"user_id": 1}, 1) is True
    # Unrestricted still requires the field to exist.
    assert _visible_to({}, None) is False
    assert _visible_to({"user_id": 9}, None) is True


def test_vector_store_search_requires_a_user_id():
    """A filter with a default is a filter that eventually gets omitted."""
    import inspect

    from backend.rag.vector_store import BaseVectorStore

    for implementation in (BaseVectorStore.search,):
        signature = inspect.signature(implementation)
        parameter = signature.parameters["user_id"]
        assert parameter.default is inspect.Parameter.empty, (
            "user_id must be required so unscoped retrieval cannot happen by accident"
        )


def test_retrieval_never_returns_another_users_chunks(session, vector_store, monkeypatch):
    from backend.rag import pipeline as pipeline_module
    from backend.rag.embeddings import EmbeddingManager

    monkeypatch.setattr(pipeline_module, "get_vector_store", lambda: vector_store)

    alice = make_user(session, "alice@example.com")
    bob = make_user(session, "bob@example.com")

    alice_chunks = [{"chunk_index": 1, "text": "alice launch code ALPHA-111",
                     "metadata": {"file_name": "alice.txt", "page_number": 1, "user_id": alice.id}}]
    bob_chunks = [{"chunk_index": 1, "text": "bob launch code BRAVO-222",
                   "metadata": {"file_name": "bob.txt", "page_number": 1, "user_id": bob.id}}]

    document, _ = _store(session, vector_store, EmbeddingManager, alice, "alice.txt", alice_chunks)
    _store(session, vector_store, EmbeddingManager, bob, "bob.txt", bob_chunks)

    pipeline_module.invalidate_retrieval_cache()

    hits = pipeline_module.RagPipeline.retrieve("launch code", session, alice, limit=10)
    text = " ".join(hit["text"] for hit in hits)

    assert "ALPHA-111" in text
    assert "BRAVO-222" not in text, "leaked another user's document"


def _store(db, vector_store, embedding_manager, owner, name, chunks):
    import json

    from backend.models.models import Document, DocumentChunk

    document = Document(name=name, file_type="txt", size_bytes=10,
                        storage_path=f"/tmp/{name}", owner_id=owner.id)
    db.add(document)
    db.commit()
    db.refresh(document)

    for chunk in chunks:
        db.add(DocumentChunk(
            document_id=document.id,
            chunk_index=chunk["chunk_index"],
            text=chunk["text"],
            metadata_json=json.dumps(chunk["metadata"]),
        ))
    db.commit()

    embeddings = embedding_manager.embed_documents([c["text"] for c in chunks])
    vector_store.upsert_chunks(document.id, chunks, embeddings)
    return document, chunks


def test_user_cannot_read_another_users_session(client):
    client.post("/api/auth/register", json={"email": "alice@example.com", "password": VALID_PASSWORD})
    client.post("/api/auth/register", json={"email": "bob@example.com", "password": VALID_PASSWORD})

    alice_token = client.post(
        "/api/auth/login", json={"email": "alice@example.com", "password": VALID_PASSWORD}
    ).json()["access_token"]
    bob_token = client.post(
        "/api/auth/login", json={"email": "bob@example.com", "password": VALID_PASSWORD}
    ).json()["access_token"]

    alice_session = client.post(
        "/api/chat/sessions",
        headers={"Authorization": f"Bearer {alice_token}"},
        json={"title": "private"},
    ).json()["session_id"]

    bob_headers = {"Authorization": f"Bearer {bob_token}"}
    assert client.get(f"/api/chat/sessions/{alice_session}/messages", headers=bob_headers).status_code == 404
    assert client.delete(f"/api/chat/sessions/{alice_session}", headers=bob_headers).status_code == 404
    assert client.put(
        f"/api/chat/sessions/{alice_session}", headers=bob_headers, json={"title": "hijack"}
    ).status_code == 404

    sessions = client.get("/api/chat/sessions", headers=bob_headers).json()
    assert all(item["session_id"] != alice_session for item in sessions)


def test_user_cannot_delete_another_users_document(client):
    client.post("/api/auth/register", json={"email": "alice@example.com", "password": VALID_PASSWORD})
    client.post("/api/auth/register", json={"email": "bob@example.com", "password": VALID_PASSWORD})

    def token_for(email):
        return client.post("/api/auth/login", json={"email": email, "password": VALID_PASSWORD}).json()["access_token"]

    alice = {"Authorization": f"Bearer {token_for('alice@example.com')}"}
    bob = {"Authorization": f"Bearer {token_for('bob@example.com')}"}

    created = client.post(
        "/api/documents", headers=alice,
        files={"file": ("secret.txt", b"alice private content", "text/plain")},
    ).json()

    assert client.delete(f"/api/documents/{created['id']}", headers=bob).status_code == 404
    assert client.get("/api/documents", headers=bob).json() == []


# ---------------------------------------------------------------------------
# 3. Cross-site scripting
# ---------------------------------------------------------------------------
def test_frontend_vendors_and_uses_a_sanitiser():
    """Assistant text can quote uploaded documents, so it is untrusted."""
    import pathlib

    frontend = pathlib.Path(__file__).resolve().parents[2] / "frontend"
    app_js = (frontend / "js" / "app.js").read_text(encoding="utf-8")
    index_html = (frontend / "index.html").read_text(encoding="utf-8")

    assert "DOMPurify.sanitize" in app_js, "model output must be sanitized"
    assert "textContent" in app_js, "user-controlled values must use textContent"
    assert (frontend / "js" / "vendor" / "purify.min.js").exists(), "sanitiser must be vendored"
    assert (frontend / "js" / "vendor" / "marked.min.js").exists()
    # Must not be loaded from a third-party origin.
    assert "cdn.jsdelivr" not in index_html
    assert "cdn.tailwindcss" not in index_html


def test_sse_reader_carries_a_buffer_between_reads():
    """Frames can straddle network reads; without a buffer tokens are lost."""
    import pathlib

    app_js = (
        pathlib.Path(__file__).resolve().parents[2] / "frontend" / "js" / "app.js"
    ).read_text(encoding="utf-8")

    assert "frames.pop()" in app_js, "SSE remainder must be carried forward"
    assert "decoder.decode(value, { stream: true })" in app_js


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
def test_upload_rejects_content_that_contradicts_its_extension(client):
    client.post("/api/auth/register", json={"email": "a@example.com", "password": VALID_PASSWORD})
    token = client.post(
        "/api/auth/login", json={"email": "a@example.com", "password": VALID_PASSWORD}
    ).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    fake = client.post(
        "/api/documents", headers=headers,
        files={"file": ("innocent.pdf", b"this is definitely not a pdf", "application/pdf")},
    )
    assert fake.status_code == 400
    assert "contents" in fake.json()["detail"].lower()

    binary = client.post(
        "/api/documents", headers=headers,
        files={"file": ("notes.txt", b"has\x00nul\x00bytes", "text/plain")},
    )
    assert binary.status_code == 400


def test_upload_rejects_unsupported_extension(client):
    client.post("/api/auth/register", json={"email": "a@example.com", "password": VALID_PASSWORD})
    token = client.post(
        "/api/auth/login", json={"email": "a@example.com", "password": VALID_PASSWORD}
    ).json()["access_token"]

    response = client.post(
        "/api/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("payload.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_stored_filename_cannot_escape_the_upload_directory():
    from backend.api.documents import _validate_extension

    # Traversal-shaped names are replaced by a generated uuid, but the
    # extension check is the only place the client's string is inspected.
    assert _validate_extension("../../etc/passwd.txt") == ".txt"
    assert _validate_extension("report.pdf") == ".pdf"


# ---------------------------------------------------------------------------
# Passwords, tokens, configuration
# ---------------------------------------------------------------------------
def test_password_rejects_bcrypt_overlong_input():
    """bcrypt truncates past 72 bytes, which would make two passwords equal."""
    with pytest.raises(PasswordPolicyError):
        validate_password_strength("a" * 73)


def test_password_rejects_short_input():
    with pytest.raises(PasswordPolicyError):
        validate_password_strength("short")


def test_password_hash_and_verify():
    hashed = hash_password(VALID_PASSWORD)
    assert hashed != VALID_PASSWORD
    assert verify_password(VALID_PASSWORD, hashed)
    assert not verify_password("wrong", hashed)


def test_verify_password_survives_a_corrupt_hash():
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_jwt_round_trip_and_rejection():
    token = create_access_token(subject="a@example.com", role="user")
    claims = decode_access_token(token)
    assert claims["sub"] == "a@example.com"
    assert decode_access_token("garbage") is None


def test_expired_token_is_rejected():
    import datetime as dt

    token = create_access_token(subject="a@example.com", role="user",
                                expires_delta=dt.timedelta(seconds=-10))
    assert decode_access_token(token) is None


def test_weak_signing_key_is_refused():
    from backend.utils.config import INSECURE_DEFAULT_SECRET

    with pytest.raises(ValueError):
        Settings(_env_file=None, SECRET_KEY=INSECURE_DEFAULT_SECRET)
    with pytest.raises(ValueError):
        Settings(_env_file=None, SECRET_KEY="tooshort")


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Settings(_env_file=None, CHUNK_SIZE=500, CHUNK_OVERLAP=500)


def test_email_validation_does_not_do_a_dns_lookup():
    """pydantic's EmailStr queries DNS and rejects unresolvable domains."""
    assert normalise_email("  User@Example.COM  ") == "User@example.com"
    # A domain that cannot resolve must still be accepted.
    assert normalise_email("dev@internal.invalid-tld") == "dev@internal.invalid-tld"
    for bad in ("not-an-email", "@example.com", "user@", "no-at-sign"):
        with pytest.raises(ValueError):
            normalise_email(bad)


def test_login_is_rate_limited(client):
    for _ in range(5):
        client.post(
            "/api/auth/login", json={"email": "brute@example.com", "password": "WrongPass123!"}
        )
    locked = client.post(
        "/api/auth/login", json={"email": "brute@example.com", "password": "WrongPass123!"}
    )
    assert locked.status_code == 429


def test_authentication_is_required(client):
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/chat/sessions").status_code == 401
    assert client.post("/api/chat", json={"query": "x", "session_id": "y"}).status_code == 401


# ---------------------------------------------------------------------------
# 11. The local session
# ---------------------------------------------------------------------------
def test_local_session_provisions_an_account_and_issues_a_usable_token(client, session):
    """The UI has no sign-in step, so the server provisions its own account.

    The token must be a real one that the normal auth dependency accepts, not
    a bypass: every other guard in the app has to keep working unchanged.
    """
    first = client.get("/api/auth/local")
    assert first.status_code == 200
    body = first.json()
    assert body["token_type"] == "bearer"
    assert body["email"] == "local@coddy.home"

    profile = client.get("/api/auth/profile", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert profile.status_code == 200
    assert profile.json()["email"] == "local@coddy.home"

    # And it owns documents like any other account, so tenant scoping applies.
    assert client.get(
        "/api/documents", headers={"Authorization": f"Bearer {body['access_token']}"}
    ).status_code == 200


def test_local_session_is_idempotent(client, session):
    """Repeat boots must reuse the one account, never provision a second."""
    from backend.api.auth import LOCAL_EMAIL
    from backend.models.models import User

    first = client.get("/api/auth/local").json()
    second = client.get("/api/auth/local").json()

    assert first["email"] == second["email"] == LOCAL_EMAIL
    rows = session.query(User).filter(User.email == LOCAL_EMAIL).all()
    assert len(rows) == 1


def test_local_account_cannot_be_signed_into(client):
    """The provisioned account owns documents but has no usable password.

    Its stored hash is of a secret generated at provision time and discarded,
    so the credential path cannot reach it at all.
    """
    assert client.get("/api/auth/local").status_code == 200
    assert client.post(
        "/api/auth/login",
        json={"email": "local@coddy.home", "password": VALID_PASSWORD},
    ).status_code == 401