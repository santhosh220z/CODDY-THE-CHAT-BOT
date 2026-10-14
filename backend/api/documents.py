from __future__ import annotations

import json
import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.database.connection import get_db
from backend.models.models import Document, DocumentChunk, User
from backend.rag.document_processor import DocumentProcessingError, process_document
from backend.rag.embeddings import EmbeddingManager
from backend.rag.pipeline import invalidate_retrieval_cache
from backend.rag.vector_store import get_vector_store
from backend.utils.config import settings
from backend.utils.logging import logger

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Magic bytes per extension. An extension check alone accepts any file renamed
# to .pdf, which then goes straight into a parser.
MAGIC = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04", b"PK\x05\x06"],  # zip/OOXML container
}
# Text formats have no reliable signature; they are validated by decoding.
_TEXTUAL = {".txt", ".md", ".markdown", ".csv"}


def _validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: "
            f"{', '.join(sorted(settings.allowed_extensions))}.",
        )
    return ext


def _check_magic(ext: str, header: bytes) -> None:
    expected = MAGIC.get(ext)
    if expected and not any(header.startswith(sig) for sig in expected):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File contents do not match the file extension.",
        )


def _can_access(user: User, document: Document) -> bool:
    return user.role == "admin" or document.owner_id == user.id


async def _save_upload(file: UploadFile) -> tuple[str, int]:
    """Stream to disk, enforcing the size cap as we go.

    The file is never named by the client, so a crafted name cannot escape the
    upload directory.
    """
    ext = _validate_extension(file.filename or "upload")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    stored_name = f"{uuid.uuid4().hex}{ext}"
    destination = os.path.abspath(os.path.join(settings.UPLOAD_DIR, stored_name))

    # Belt and braces: even though the name is generated, refuse anything that
    # does not land inside UPLOAD_DIR.
    if os.path.commonpath([destination, os.path.abspath(settings.UPLOAD_DIR)]) != os.path.abspath(
        settings.UPLOAD_DIR
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upload path.")

    written = 0
    try:
        with open(destination, "wb") as out:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                if written > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the "
                        f"{settings.MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
                    )
                out.write(block)

            if written == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty."
                )

            if ext not in _TEXTUAL:
                _check_magic(ext, _header_of(destination))
    except HTTPException:
        os.remove(destination) if os.path.exists(destination) else None
        raise
    except Exception as exc:
        if os.path.exists(destination):
            os.remove(destination)
        logger.error(f"Failed to store upload: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not store the upload."
        )

    if ext in _TEXTUAL:
        try:
            _check_text_decodable(destination)
        except HTTPException:
            os.remove(destination)
            raise

    return destination, written


def _header_of(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(8)


def _check_text_decodable(path: str) -> None:
    """Reject binary blobs masquerading as .txt/.csv/.md."""
    with open(path, "rb") as fh:
        sample = fh.read(4096)
    if b"\x00" in sample:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File appears to be binary, not text.",
        )


class DocumentSummary(BaseModel):
    id: int
    name: str
    file_type: str
    size_bytes: int
    created_at: str


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    path, size = await _save_upload(file)
    ext = os.path.splitext(path)[1].lower()

    document = Document(
        name=(file.filename or f"document{ext}")[:512],
        file_type=ext.lstrip("."),
        size_bytes=size,
        storage_path=path,
        owner_id=current_user.id,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        chunks = process_document(
            path=path,
            file_name=document.name,
            owner_id=document.owner_id,
        )
    except DocumentProcessingError as exc:
        _cleanup(document, db)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        _cleanup(document, db)
        logger.error(f"Processing failed for {document.name}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process the document.",
        ) from exc

    try:
        for chunk in chunks:
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"],
                    metadata_json=json.dumps(chunk["metadata"]),
                )
            )
        db.commit()

        embeddings = EmbeddingManager.embed_documents([c["text"] for c in chunks])
        get_vector_store().upsert_chunks(document.id, chunks, embeddings)
    except Exception as exc:
        db.rollback()
        _cleanup(document, db)
        logger.error(f"Indexing failed for {document.name}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not index the document.",
        ) from exc

    invalidate_retrieval_cache()
    logger.info(f"{current_user.email} uploaded {document.name} ({len(chunks)} chunks)")
    return {
        "id": document.id,
        "name": document.name,
        "chunks": len(chunks),
    }


def _cleanup(document: Document, db: Session) -> None:
    """Remove vectors, the file on disk, and the row. Never raises.

    Every step is guarded independently: a failure part-way must not leave the
    caller with an unhandled exception after some of the teardown has already
    happened, and must not leave the row behind with the vectors already gone.
    """
    try:
        get_vector_store().delete_document(document.id)
    except Exception:
        logger.warning("Vector cleanup failed for document %s", document.id, exc_info=True)

    try:
        if os.path.exists(document.storage_path):
            os.remove(document.storage_path)
    except OSError:
        logger.warning("Could not remove %s", document.storage_path)

    try:
        db.delete(document)
        db.commit()
    except Exception:
        db.rollback()
        logger.error(
            "Could not delete document row %s; its vectors and file are already "
            "gone. Delete this row manually to avoid an orphan.",
            document.id,
            exc_info=True,
        )


@router.get("", response_model=List[DocumentSummary])
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[DocumentSummary]:
    stmt = select(Document)
    if current_user.role != "admin":
        stmt = stmt.where(Document.owner_id == current_user.id)
    documents = db.execute(stmt.order_by(Document.created_at.desc())).scalars().all()
    return [
        DocumentSummary(
            id=d.id,
            name=d.name,
            file_type=d.file_type,
            size_bytes=d.size_bytes,
            created_at=d.created_at.isoformat(),
        )
        for d in documents
    ]


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if not _can_access(current_user, document):
        # 404 rather than 403 so the response does not confirm the document
        # exists to someone who may not read it.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    _cleanup(document, db)
    invalidate_retrieval_cache()
    logger.info(f"Deleted document {document_id}")
    return {"deleted": document_id}


@router.post("/{document_id}/reindex")
def reindex_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    document = db.get(Document, document_id)
    if document is None or not _can_access(current_user, document):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if not os.path.exists(document.storage_path):
        raise HTTPException(
            status_code=status.HTTP_410_GONE, detail="The stored file is missing."
        )

    try:
        # owner_id, not current_user.id: an admin reindexing someone else's
        # document must preserve the original owner, or the chunks would become
        # visible to the admin and invisible to their owner.
        chunks = process_document(
            path=document.storage_path,
            file_name=document.name,
            owner_id=document.owner_id,
        )

        # Embed before touching the database. The previous order committed the
        # chunk delete first, so a failure here left the document with zero
        # chunks in SQL while stale vectors remained in the index - the two
        # retrieval paths then disagreed permanently, and db.rollback() could
        # not undo an already-committed delete.
        embeddings = EmbeddingManager.embed_documents([c["text"] for c in chunks])
        get_vector_store().upsert_chunks(document.id, chunks, embeddings)

        db.execute(
            DocumentChunk.__table__.delete().where(DocumentChunk.document_id == document.id)
        )
        for chunk in chunks:
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"],
                    metadata_json=json.dumps(chunk["metadata"]),
                )
            )
        db.commit()
    except DocumentProcessingError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        logger.error(f"Reindex failed for document {document_id}: {exc}")
        # The vector store now holds the new chunks. Put it back in step with
        # the rows that survived the rollback.
        try:
            get_vector_store().delete_document(document.id)
        except Exception:
            logger.warning("Could not clear vectors after a failed reindex.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Reindexing failed."
        ) from exc

    invalidate_retrieval_cache()
    return {"id": document.id, "chunks": len(chunks)}