import os
import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database.connection import Base
from backend.models.models import User, Document, DocumentChunk, ChatSession, ChatMessage
from backend.auth.security import get_password_hash, verify_password, create_access_token, decode_access_token
from backend.rag.document_processor import DocumentProcessor
from backend.rag.vector_store import FAISSVectorStore, get_vector_store
from backend.utils.config import settings

# Setup clean in-memory test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

# -------------------------------------------------------------
# SECURITY & AUTHENTICATION TESTS
# -------------------------------------------------------------
def test_password_hashing():
    pwd = "MySecretPassword123!"
    hashed = get_password_hash(pwd)
    
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("wrong_password", hashed) is False

def test_jwt_tokens():
    payload = {"sub": "test@coddy.ai", "role": "admin"}
    token = create_access_token(payload)
    
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "test@coddy.ai"
    assert decoded["role"] == "admin"

def test_expired_or_invalid_jwt():
    decoded = decode_access_token("invalid_token_string")
    assert decoded is None

# -------------------------------------------------------------
# DOCUMENT PROCESSOR & CHUNKING TESTS
# -------------------------------------------------------------
def test_clean_text():
    raw_text = "  Hello   World!\r\nThis is a \t test of whitespace \n\n\n\n normalization. "
    cleaned = DocumentProcessor.clean_text(raw_text)
    
    assert "  " not in cleaned
    assert "\t" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Hello")
    assert cleaned.endswith("normalization.")

def test_split_text_recursive():
    # Construct a large text block of about 1500 chars to test splitting logic
    word = "word "
    large_text = word * 300 # 300 * 5 = 1500 chars
    
    chunks = DocumentProcessor.split_text_recursive(large_text, chunk_size=1000, chunk_overlap=200)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1000
        # Check that chunk overlap logic doesn't crash
        assert len(chunk) > 100

def test_process_txt_document(tmp_path):
    # Create a temporary txt file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello Coddy RAG! This is a simple text file used for validating our extraction pipeline.")
    
    chunks = DocumentProcessor.process_document(
        file_path=str(test_file),
        file_name="test.txt",
        uploaded_by_email="uploader@coddy.ai"
    )
    
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello Coddy RAG! This is a simple text file used for validating our extraction pipeline."
    assert chunks[0]["metadata"]["file_name"] == "test.txt"
    assert chunks[0]["metadata"]["uploaded_by"] == "uploader@coddy.ai"

# -------------------------------------------------------------
# VECTOR DATABASE TESTS
# -------------------------------------------------------------
def test_faiss_vector_store(tmp_path):
    # Override settings for FAISS paths to point to a temporary test folder
    settings.FAISS_DB_PATH = str(tmp_path / "faiss_test")
    
    store = FAISSVectorStore()
    
    chunks = [
        {"chunk_index": 1, "text": "This is a document segment about machine learning.", "metadata": {"file_name": "ml.txt", "page_number": 1}},
        {"chunk_index": 2, "text": "This is a document segment about cooking recipes.", "metadata": {"file_name": "food.txt", "page_number": 1}}
    ]
    # Simple mock embeddings (dimension 384)
    emb1 = [0.1] * 384
    emb2 = [-0.1] * 384
    embeddings = [emb1, emb2]
    
    # Test Add Chunks
    store.add_chunks(document_id=1, chunks=chunks, embeddings=embeddings)
    assert len(store.metadata_store) == 2
    
    # Test Search
    query_emb = [0.09] * 384 # close to emb1
    results = store.search(query_emb, top_k=2)
    
    assert len(results) == 2
    # First item should be ml.txt because query vector matches emb1 closer
    assert results[0]["metadata"]["file_name"] == "ml.txt"
    
    # Test Delete Chunks
    store.delete_document_chunks(document_id=1)
    assert len(store.metadata_store) == 0
