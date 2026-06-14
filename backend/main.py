import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.utils.config import settings
from backend.utils.logging import logger
from backend.database.connection import init_db
from backend.api import auth, documents, chat, admin

# Lifespan manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up CODDY RAG Chatbot Server...")
    # Initialize DB schemas on startup
    init_db()
    yield
    logger.info("Shutting down CODDY RAG Chatbot Server...")

app = FastAPI(
    title="CODDY – THE CHAT BOT",
    description="Production-ready RAG Chatbot powered by local LLMs (Ollama) and Vector Search",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers for clean APIs
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error for request {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Input validation failed."}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Register API Routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(admin.router)

# Mount frontend files statically
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_path):
    # Mount css, js, pages folders
    app.mount("/css", StaticFiles(directory=os.path.join(frontend_path, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(frontend_path, "js")), name="js")
    
    # Expose main landing page at root URL
    @app.get("/", response_class=HTMLResponse)
    def read_root():
        index_file = os.path.join(frontend_path, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>Welcome to CODDY – THE CHAT BOT Backend</h1><p>Frontend assets not found on server.</p>"
else:
    logger.warning(f"Frontend directory not found at: {frontend_path}")
    @app.get("/", response_class=HTMLResponse)
    def read_root():
        return "<h1>CODDY – THE CHAT BOT API Server</h1><p>API is active. Frontend folder was not found.</p>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
