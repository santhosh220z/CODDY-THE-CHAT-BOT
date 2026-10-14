from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api import auth, chat, documents
from backend.database.connection import init_db
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
        yield

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
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_exception(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

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
        from backend.services.llm import get_llm_provider

        payload: Dict[str, Any] = {
            "status": "ok",
            "llm": settings.LLM_PROVIDER,
            "vector_store": settings.VECTOR_STORE_TYPE,
        }
        if hasattr(get_llm_provider(), "info"):
            payload["details"] = get_llm_provider().info()
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