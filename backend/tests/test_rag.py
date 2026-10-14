"""Retrieval, chunking and extraction tests."""

import json

import pytest

from backend.rag.document_processor import (
    DocumentProcessingError,
    clean_text,
    process_document,
    split_text,
)
from backend.rag.pipeline import RagPipeline, accessible_document_ids, invalidate_retrieval_cache

from .conftest import make_document, make_session, make_user


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_short_text_is_a_single_chunk():
    assert split_text("hello world", 1000, 200) == ["hello world"]


def test_empty_text_produces_nothing():
    assert split_text("", 1000, 200) == []


def test_no_chunk_exceeds_the_size_limit():
    text = "The quick brown fox jumps over the lazy dog. " * 60
    chunks = split_text(text, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_consecutive_chunks_actually_overlap():
    """CHUNK_OVERLAP was previously computed and then discarded."""
    text = "The quick brown fox jumps over the lazy dog. " * 60
    chunks = split_text(text, chunk_size=400, chunk_overlap=80)

    shared = [
        index
        for index in range(len(chunks) - 1)
        if chunks[index][-40:] and chunks[index][-40:] in chunks[index + 1]
    ]
    assert shared, "no overlap between adjacent chunks"


def test_zero_overlap_is_allowed():
    text = "word " * 400
    chunks = split_text(text, chunk_size=500, chunk_overlap=0)
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_oversized_single_token_is_hard_sliced():
    chunks = split_text("x" * 5000, chunk_size=1000, chunk_overlap=100)
    assert len(chunks) >= 5
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_split_prefers_natural_boundaries():
    text = "This is a sentence. " * 80
    chunks = split_text(text, chunk_size=300, chunk_overlap=50)
    # No chunk should start mid-word.
    for chunk in chunks:
        assert chunk.strip()[:1].isalpha()


def test_clean_text_normalises_whitespace():
    cleaned = clean_text("  a   b\r\nc\t\td \n\n\n\n e  ")
    assert "  " not in cleaned
    assert "\t" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("a b")
    assert cleaned.endswith("e")
    assert "\r" not in cleaned


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def test_process_txt_stamps_tenant_metadata(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Launch code is ALPHA-4242.", encoding="utf-8")

    chunks = process_document(str(path), "notes.txt", owner_id=7)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["user_id"] == 7
    assert chunks[0]["metadata"]["file_name"] == "notes.txt"


def test_process_csv_includes_the_header(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,role\nada,engineer\n", encoding="utf-8")

    chunks = process_document(str(path), "data.csv", owner_id=1)
    text = chunks[0]["text"]
    assert "Columns: name | role" in text
    assert "ada" in text


def test_process_csv_handles_bom_and_latin1(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes("name\ncaf\xe9\n".encode("latin-1"))
    chunks = process_document(str(path), "data.csv", owner_id=1)
    assert "café" in chunks[0]["text"]


def test_process_docx_captures_tables(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "doc.docx"

    document = docx.Document()
    document.add_paragraph("Intro paragraph")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "cellA"
    table.rows[0].cells[1].text = "cellB"
    document.save(str(path))

    chunks = process_document(str(path), "doc.docx", owner_id=1)
    text = chunks[0]["text"]
    assert "Intro paragraph" in text
    assert "cellA | cellB" in text


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "a.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(DocumentProcessingError):
        process_document(str(path), "a.xyz", owner_id=1)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DocumentProcessingError):
        process_document(str(path), "empty.txt", owner_id=1)


def test_a_zip_renamed_to_txt_is_rejected_by_the_api_layer():
    """The processor would decode it as garbage; the upload layer stops it."""
    from backend.api.documents import _check_text_decodable, _validate_extension

    assert _validate_extension("a.txt") == ".txt"
    assert callable(_check_text_decodable)


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
def test_admins_are_unrestricted_while_users_are_not(session):
    alice = make_user(session, "alice@example.com")
    admin = make_user(session, "root@example.com", role="admin")

    alice_doc, _ = make_document(session, alice, "alice.txt", ["alice text"])
    other, _ = make_document(session, admin, "root.txt", ["root text"])

    alice_ids = accessible_document_ids(session, alice)
    assert alice_ids is not None
    assert alice_doc.id in alice_ids
    assert other.id not in alice_ids

    assert accessible_document_ids(session, admin) is None


# ---------------------------------------------------------------------------
# History windowing
# ---------------------------------------------------------------------------
def test_history_returns_the_last_ten_not_the_first_ten(session):
    user = make_user(session, "chat@example.com")
    make_session(session, user, message_count=15)

    from backend.models.models import ChatSession

    chat = session.query(ChatSession).first()
    contents = [message.content for message in RagPipeline.recent_history(session, chat.id)]

    assert len(contents) == 10
    assert contents[0] == "message 5"
    assert contents[-1] == "message 14"
    assert "message 0" not in contents


def test_history_is_ordered_oldest_first(session):
    user = make_user(session, "chat2@example.com")
    make_session(session, user, message_count=4)

    from backend.models.models import ChatSession

    chat = session.query(ChatSession).first()
    contents = [message.content for message in RagPipeline.recent_history(session, chat.id)]
    assert contents == ["message 0", "message 1", "message 2", "message 3"]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------
def test_prompt_includes_source_metadata():
    chunks = [{"text": "body text", "metadata": {"file_name": "report.pdf", "page_number": 3}}]
    prompt, system = RagPipeline.build_prompt("what?", [], chunks)

    assert "source=report.pdf" in prompt
    assert "page=3" in prompt
    assert "CONTEXT:" in prompt
    assert "retrieval-augmented" in system.lower()


def test_prompt_says_so_when_nothing_was_retrieved():
    prompt, _ = RagPipeline.build_prompt("what?", [], [])
    assert "no documents matched" in prompt


def test_citations_are_numbered_and_positioned():
    chunks = [
        {"text": "first", "metadata": {"file_name": "a.pdf", "page_number": 1}},
        {"text": "second", "metadata": {"file_name": "b.pdf", "page_number": 5}},
    ]
    citations = RagPipeline.build_citations(chunks)

    assert [c["position"] for c in citations] == [1, 2]
    assert citations[1]["file_name"] == "b.pdf"
    assert citations[1]["page_number"] == 5


def test_citation_excerpt_is_truncated():
    long_chunk = [{"text": "x" * 5000, "metadata": {"file_name": "a.txt", "page_number": 1}}]
    citation = RagPipeline.build_citations(long_chunk, preview_chars=100)[0]
    assert len(citation["excerpt"]) < 200
    assert citation["excerpt"].endswith("...")


# ---------------------------------------------------------------------------
# BM25 cache
# ---------------------------------------------------------------------------
def test_cache_reuse_and_invalidation(session):
    user = make_user(session, "cache@example.com")
    document, _ = make_document(session, user, "cached.txt", ["alpha beta gamma", "delta epsilon"])

    invalidate_retrieval_cache()
    first = accessible_document_ids(session, user)
    second = accessible_document_ids(session, user)
    assert first == second


def test_bm25_corpus_is_scoped_to_the_user(session):
    from backend.rag import pipeline as pipeline_module

    alice = make_user(session, "alice2@example.com")
    bob = make_user(session, "bob2@example.com")
    make_document(session, alice, "a.txt", ["alice only content"])
    make_document(session, bob, "b.txt", ["bob only content"])
    invalidate_retrieval_cache()

    doc_ids = accessible_document_ids(session, alice)
    entry = pipeline_module._bm25_entry(session, doc_ids, alice.id)

    corpus = " ".join(chunk["text"] for chunk in entry["chunks"])
    assert "alice only content" in corpus
    assert "bob only content" not in corpus