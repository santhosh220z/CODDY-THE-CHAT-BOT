from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.utils.config import settings
from backend.utils.logging import logger

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

# SQLite objects are not thread-safe and are bound to the creating thread.
# The app opens sessions from both request handlers and SSE stream generators,
# which run on a worker thread.
connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    future=True,
)


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_connection, connection_record):
    """SQLite ignores foreign keys unless asked, per connection."""
    if not _is_sqlite:
        return
    cursor = dbapi_connection.cursor()
    # ON DELETE CASCADE is declared in the models and is required for
    # removing a document to remove its chunks.
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables. Import models first so they register on Base.metadata."""
    from backend.database.base import Base
    import backend.models.models  # noqa: F401

    logger.info("Creating database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ready.")


def get_db() -> Iterator[Session]:
    """FastAPI dependency. One session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()