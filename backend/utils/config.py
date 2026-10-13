import os
from typing import Dict, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Published in the repository history and in .env.example of earlier versions.
# Anyone who knows it can mint a valid JWT for any account, so it is refused.
INSECURE_DEFAULT_SECRET = "super-secret-key-change-in-production-123456"


class Settings(BaseSettings):
    # --- Security ---------------------------------------------------------
    SECRET_KEY: str = Field(default=INSECURE_DEFAULT_SECRET)
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)
    MIN_PASSWORD_LENGTH: int = Field(default=8)
    # bcrypt ignores everything past 72 bytes. Longer input is rejected rather
    # than silently truncated, which would make two different passphrases
    # interchangeable.
    MAX_PASSWORD_BYTES: int = Field(default=72)

    # --- Database ---------------------------------------------------------
    DATABASE_URL: str = Field(default="sqlite:///./coddy.db")

    # --- Vector store -----------------------------------------------------
    VECTOR_STORE_TYPE: str = Field(default="chroma")  # chroma | faiss
    CHROMA_DB_PATH: str = Field(default="./vector_store/chroma")
    FAISS_DB_PATH: str = Field(default="./vector_store/faiss")

    # --- Chunking ---------------------------------------------------------
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    # --- Embeddings -------------------------------------------------------
    EMBEDDING_MODEL_NAME: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = Field(default="")  # blank = let torch decide

    # --- Retrieval --------------------------------------------------------
    RAG_TOP_K: int = Field(default=5)
    # Hybrid fusion uses a weighted sum of min-max normalised scores, not
    # Reciprocal Rank Fusion. Named accordingly.
    HYBRID_SEARCH: bool = Field(default=True)
    BM25_WEIGHT: float = Field(default=0.4)

    # --- Local model ------------------------------------------------------
    # transformers | ollama
    LLM_PROVIDER: str = Field(default="transformers")

    # transformers provider
    # Qwen3-4B-Instruct-2507 is the non-thinking build: no <think> blocks,
    # faster time-to-first-token, cleaner citations. 4-bit nf4 is required -
    # bf16 is ~8GB and does not fit a 6GB card.
    HF_MODEL_ID: str = Field(default="Qwen/Qwen3-4B-Instruct-2507")
    LOAD_IN_4BIT: bool = Field(default=True)
    BNB_4BIT_COMPUTE_DTYPE: str = Field(default="bfloat16")
    BNB_4BIT_QUANT_TYPE: str = Field(default="nf4")
    MAX_NEW_TOKENS: int = Field(default=1024)
    MODEL_TEMPERATURE: float = Field(default=0.3)
    MODEL_TOP_P: float = Field(default=0.9)
    MODEL_MAX_CONTEXT: int = Field(default=8192)

    # ollama provider (optional)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="qwen3:4b")

    # --- Uploads ----------------------------------------------------------
    MAX_UPLOAD_BYTES: int = Field(default=15 * 1024 * 1024)
    UPLOAD_DIR: str = Field(default="./uploads")

    # --- Server -----------------------------------------------------------
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="INFO")
    # Comma-separated origins. Empty means same-origin only, which is the
    # default posture for a locally-served app.
    CORS_ORIGINS: str = Field(default="")

    @field_validator("SECRET_KEY")
    @classmethod
    def _reject_weak_secret(cls, v: str) -> str:
        if v == INSECURE_DEFAULT_SECRET:
            raise ValueError(
                "SECRET_KEY is still the published example value, so anyone could "
                "forge a valid token. Generate a real one with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters.")
        return v

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def _validate_overlap(cls, v: int, info) -> int:
        size = info.data.get("CHUNK_SIZE")
        if size is not None and v >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if v < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")
        return v

    @field_validator("VECTOR_STORE_TYPE")
    @classmethod
    def _validate_store(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"chroma", "faiss"}:
            raise ValueError("VECTOR_STORE_TYPE must be 'chroma' or 'faiss'.")
        return v

    @field_validator("BNB_4BIT_COMPUTE_DTYPE")
    @classmethod
    def _validate_dtype(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"float16", "bfloat16", "float32"}:
            raise ValueError("BNB_4BIT_COMPUTE_DTYPE must be float16, bfloat16 or float32.")
        return v

    @property
    def cors_origin_list(self) -> List[str]:
        if not self.CORS_ORIGINS.strip():
            return []
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_extensions(self) -> Dict[str, None]:
        return {e: None for e in (".pdf", ".docx", ".txt", ".csv", ".md", ".markdown")}

    def ensure_directories(self) -> None:
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.CHROMA_DB_PATH) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.FAISS_DB_PATH) or ".", exist_ok=True)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
settings.ensure_directories()