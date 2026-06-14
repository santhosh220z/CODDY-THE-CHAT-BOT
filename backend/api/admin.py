import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.models.models import User, Document, ChatSession, ChatMessage
from backend.api.auth import get_admin_user
from backend.utils.config import settings
from backend.utils.logging import logger

router = APIRouter(prefix="/api/admin", tags=["Admin"])

def get_directory_size(path: str) -> int:
    total_size = 0
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

@router.get("/stats")
def get_admin_stats(
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    try:
        # DB Stats
        total_users = db.query(User).count()
        total_docs = db.query(Document).count()
        total_sessions = db.query(ChatSession).count()
        total_messages = db.query(ChatMessage).count()
        
        # Storage Stats
        uploads_size = get_directory_size(settings.UPLOAD_DIR)
        
        vector_store_path = settings.CHROMA_DB_PATH if settings.VECTOR_STORE_TYPE == "chroma" else settings.FAISS_DB_PATH
        vector_store_size = get_directory_size(vector_store_path)
        
        # Total storage in bytes
        total_storage_bytes = uploads_size + vector_store_size
        
        # Convert bytes to human readable format (MB)
        uploads_size_mb = round(uploads_size / (1024 * 1024), 2)
        vector_store_size_mb = round(vector_store_size / (1024 * 1024), 2)
        total_storage_mb = round(total_storage_bytes / (1024 * 1024), 2)
        
        # Active Config Details
        active_llm = f"{settings.LLM_PROVIDER.upper()} - {settings.OLLAMA_MODEL if settings.LLM_PROVIDER == 'ollama' else settings.HF_MODEL}"
        active_embedding = settings.EMBEDDING_MODEL_NAME
        vector_store_type = settings.VECTOR_STORE_TYPE.upper()
        
        # System Health (Basic validation)
        db_healthy = True
        try:
            db.execute("SELECT 1")
        except Exception:
            db_healthy = False
            
        system_health = "Healthy" if db_healthy else "Unhealthy"
        
        return {
            "stats": {
                "total_users": total_users,
                "total_documents": total_docs,
                "chat_sessions": total_sessions,
                "total_messages": total_messages
            },
            "storage": {
                "uploads_mb": uploads_size_mb,
                "vector_db_mb": vector_store_size_mb,
                "total_used_mb": total_storage_mb
            },
            "configuration": {
                "active_llm": active_llm,
                "active_embedding": active_embedding,
                "vector_store": vector_store_type
            },
            "system_health": system_health
        }
    except Exception as e:
        logger.error(f"Failed to fetch admin statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stats: {e}"
        )
