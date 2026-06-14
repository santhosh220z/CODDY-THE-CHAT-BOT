# Use slim python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential, etc. needed for compiling some python extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python packages
# Use cache mount if supported, disable pip warnings
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and frontend source files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env.example .env

# Create persistent storage directories
RUN mkdir -p uploads vector_store

# Expose server port
EXPOSE 8000

# Start FastAPI application
CMD ["python", "backend/main.py"]
