FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/usage /app/data/feedback

# Environment variables (override in docker-compose or at runtime)
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
ENV QDRANT_PATH=/app/data/qdrant_local
ENV PARENT_CHUNKS_DIR=/app/data/chunks

# Expose backend port
EXPOSE 5001

# Default command (can be overridden)
CMD ["python", "-m", "src.backend.bot"]

