import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # JWT & Authentication
    SECRET_KEY: str = Field(default="super-secret-key-change-in-production-123456")
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)

    # Database Configuration
    DATABASE_URL: str = Field(default="sqlite:///./coddy.db")

    # Vector Database Configuration
    VECTOR_STORE_TYPE: str = Field(default="chroma") # chroma or faiss
    CHROMA_DB_PATH: str = Field(default="./vector_store/chroma")
    FAISS_DB_PATH: str = Field(default="./vector_store/faiss")

    # Document Chunking Configuration
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    # Embedding System Configuration
    EMBEDDING_MODEL_NAME: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Retrieval Configuration
    RAG_TOP_K: int = Field(default=5)
    HYBRID_SEARCH: bool = Field(default=True)
    BM25_WEIGHT: float = Field(default=0.3)

    # LLM Service Configuration
    LLM_PROVIDER: str = Field(default="ollama") # ollama or huggingface
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="deepseek-r1:1.5b")

    # Hugging Face Settings
    HF_API_KEY: str = Field(default="")
    HF_MODEL: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct")

    # General Directories
    UPLOAD_DIR: str = Field(default="./uploads")
    LOG_LEVEL: str = Field(default="INFO")
    PORT: int = Field(default=8000)
    HOST: str = Field(default="0.0.0.0")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create a global instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.CHROMA_DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(settings.FAISS_DB_PATH), exist_ok=True)
