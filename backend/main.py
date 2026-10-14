from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api import auth, chat, documents
from backend.database.connection import init_db
from backend.services.llm import warm_up_in_background
from backend.utils.config import settings
from backend.utils.logging import logger

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_db()
        logger.info("-" * 62)
        logger.info(f"CODDY ready  |  llm={settings.LLM_PROVIDER}:{settings.HF_MODEL_ID}")
        logger.info(f"            |  store={settings.VECTOR_STORE_TYPE}  top_k={settings.RAG_TOP_K}")
        logger.info("-" * 62)

        if settings.PREWARM_MODEL:
            # Load the model in the background so the server is usable
            # immediately instead of the first request paying ~20s.
            warm_up_in_background()
        yield
        logger.info("Shutting down.")

    app = FastAPI(
        title="CODDY",
        description="Local retrieval-augmented chatbot. Inference runs on this machine.",
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = settings.cors_origin_list
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )
        logger.info(f"CORS enabled for: {origins}")
    else:
        # Same-origin only. The previous configuration used a wildcard origin
        # alongside credentials, which lets any site drive the API.
        logger.info("CORS disabled - serving same-origin only.")

    # --- Error handling: never leak internals -------------------------------
    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        # exc.errors() can carry a ValueError instance in ctx, which is not
        # JSON serialisable. Passing it through raw raised a TypeError inside
        # the handler, so a typo'd email returned 500 instead of 422.
        return JSONResponse(
            # Literal rather than the status constant: starlette renamed it across
# versions and the literal is stable.
    status_code=422,
            content=jsonable_encoder({"detail": exc.errors()}),
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_exception(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(IntegrityError)
    async def on_integrity_error(request: Request, exc: IntegrityError):
        """A unique-constraint violation is a client conflict, not a crash."""
        logger.warning(f"Integrity error on {request.url.path}: {exc.orig}")
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "That value is already taken."},
        )

    @app.exception_handler(Exception)
    async def on_unexpected_error(request: Request, exc: Exception):
        # Log the detail server-side, return something generic to the caller.
        logger.error(f"Unhandled error on {request.url.path}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    # --- Routes -------------------------------------------------------------
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.get("/api/health", tags=["meta"])
    def health() -> Dict[str, Any]:
        """Liveness probe.

        Deliberately does not construct the LLM provider: the frontend polls
        this on page load, and a provider that fails to build would turn the
        liveness probe into a 500.
        """
        payload: Dict[str, Any] = {
            "status": "ok",
            "llm": settings.LLM_PROVIDER,
            "model": settings.HF_MODEL_ID if settings.LLM_PROVIDER != "ollama" else settings.OLLAMA_MODEL,
            "model_loaded": False,
            "vector_store": settings.VECTOR_STORE_TYPE,
        }

        # Report residency only if a provider already exists; do not create one.
        try:
            from backend.services import llm as llm_module

            existing = llm_module._provider
            if existing is not None and hasattr(existing, "info"):
                payload["model_loaded"] = getattr(existing, "_model", None) is not None
                payload["details"] = existing.info()
        except Exception:
            logger.debug("Provider introspection failed", exc_info=True)

        return payload

    # --- Frontend -----------------------------------------------------------
    if os.path.isdir(FRONTEND_DIR):
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    else:
        logger.warning(f"Frontend directory missing at {FRONTEND_DIR}")

    return app


app = create_app()