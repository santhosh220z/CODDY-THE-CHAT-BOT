import os
import uuid
import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.models.models import Document, DocumentChunk, User
from backend.api.auth import get_current_user
from backend.rag.document_processor import DocumentProcessor
from backend.rag.embeddings import EmbeddingManager
from backend.rag.vector_store import get_vector_store
from backend.utils.config import settings
from backend.utils.logging import logger

router = APIRouter(prefix="/api/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".md", ".markdown"}
MAX_FILE_SIZE = 15 * 1024 * 1024 # 15MB

def validate_file(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: PDF, DOCX, TXT, CSV, MD."
        )
    # Check size if possible (note: file.size is available in newer starlette/fastapi,
    # fallback to seek check if needed)
    try:
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
        if size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File exceeds maximum size of 15MB."
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.warning(f"Could not determine file size during validation: {e}")

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    validate_file(file)
    
    # Generate unique filename on disk to prevent collisions
    file_id = str(uuid.uuid4())
    original_name = file.filename
    ext = os.path.splitext(original_name)[1].lower()
    stored_filename = f"{file_id}{ext}"
    storage_path = os.path.join(settings.UPLOAD_DIR, stored_filename)
    
    # Save file to upload directory
    try:
        with open(storage_path, "wb") as f:
            content = await file.read()
            f.write(content)
            file_size = len(content)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file to disk."
        )
        
    try:
        # 1. Save document record to DB
        new_doc = Document(
            name=original_name,
            type=ext.lstrip('.'),
            size=file_size,
            storage_path=storage_path,
            uploaded_by=current_user.id
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        
        # 2. Extract and Process document text into chunks
        processed_chunks = DocumentProcessor.process_document(
            file_path=storage_path,
            file_name=original_name,
            uploaded_by_email=current_user.email
        )
        
        if not processed_chunks:
            # Cleanup if no text could be extracted
            db.delete(new_doc)
            db.commit()
            if os.path.exists(storage_path):
                os.remove(storage_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract readable text content from the document."
            )
            
        # 3. Add chunks to database
        db_chunks = []
        chunk_texts = []
        for chunk in processed_chunks:
            db_chunk = DocumentChunk(
                document_id=new_doc.id,
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
                metadata_json=json.dumps(chunk["metadata"])
            )
            db_chunks.append(db_chunk)
            chunk_texts.append(chunk["text"])
            
        db.add_all(db_chunks)
        db.commit()
        
        # 4. Generate embeddings for the chunks
        embeddings = EmbeddingManager.embed_documents(chunk_texts)
        
        # 5. Insert embeddings into Vector Store
        vector_store = get_vector_store()
        # Chroma/FAISS needs lists of dicts
        vector_store_chunks = []
        for i, chunk in enumerate(processed_chunks):
            # Pass model schema
            vector_store_chunks.append({
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })
            
        vector_store.add_chunks(
            document_id=new_doc.id,
            chunks=vector_store_chunks,
            embeddings=embeddings
        )
        
        logger.info(f"Successfully uploaded and indexed document: {original_name}")
        return {
            "message": "Document uploaded and indexed successfully",
            "document_id": new_doc.id,
            "filename": original_name,
            "chunks_count": len(processed_chunks)
        }
        
    except Exception as e:
        logger.error(f"Error during document indexing: {e}")
        # Rollback database changes
        db.rollback()
        # Clean up files
        if os.path.exists(storage_path):
            try:
                os.remove(storage_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while indexing document: {str(e)}"
        )

@router.get("", response_model=List[dict])
def get_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Non-admins can only see their own documents. Admins can see all.
    if current_user.role == "admin":
        docs = db.query(Document).all()
    else:
        docs = db.query(Document).filter(Document.uploaded_by == current_user.id).all()
        
    result = []
    for d in docs:
        result.append({
            "id": d.id,
            "name": d.name,
            "type": d.type,
            "size": d.size,
            "created_at": d.created_at.isoformat(),
            "uploaded_by": d.uploaded_by
        })
    return result

@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Retrieve document
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    # Check permissions
    if current_user.role != "admin" and doc.uploaded_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this document."
        )
        
    try:
        # Delete from Vector Store
        vector_store = get_vector_store()
        vector_store.delete_document_chunks(document_id)
        
        # Delete file from disk
        if os.path.exists(doc.storage_path):
            os.remove(doc.storage_path)
            
        # Delete from database (cascades automatically to DocumentChunk table due to relationship cascade)
        db.delete(doc)
        db.commit()
        
        logger.info(f"Deleted document: {doc.name}")
        return {"message": "Document deleted successfully."}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {e}"
        )

@router.post("/{document_id}/reindex")
def reindex_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Reindexing: delete vector representation, re-extract and re-embed.
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    # Permission check
    if current_user.role != "admin" and doc.uploaded_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to modify this document."
        )
        
    try:
        # 1. Clean vector store
        vector_store = get_vector_store()
        vector_store.delete_document_chunks(document_id)
        
        # 2. Delete existing chunk rows in DB
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        db.commit()
        
        # 3. Re-process document
        processed_chunks = DocumentProcessor.process_document(
            file_path=doc.storage_path,
            file_name=doc.name,
            uploaded_by_email=current_user.email
        )
        
        # 4. Insert chunks to DB
        db_chunks = []
        chunk_texts = []
        for chunk in processed_chunks:
            db_chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
                metadata_json=json.dumps(chunk["metadata"])
            )
            db_chunks.append(db_chunk)
            chunk_texts.append(chunk["text"])
            
        db.add_all(db_chunks)
        db.commit()
        
        # 5. Embed
        embeddings = EmbeddingManager.embed_documents(chunk_texts)
        
        # 6. Insert to Vector DB
        vector_store_chunks = []
        for i, chunk in enumerate(processed_chunks):
            vector_store_chunks.append({
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })
            
        vector_store.add_chunks(
            document_id=doc.id,
            chunks=vector_store_chunks,
            embeddings=embeddings
        )
        
        logger.info(f"Re-indexed document: {doc.name}")
        return {"message": "Document re-indexed successfully", "chunks_count": len(processed_chunks)}
    except Exception as e:
        db.rollback()
        logger.error(f"Re-indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Re-indexing failed: {e}"
        )
