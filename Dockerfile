# AI-MEM Server Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ ./src/

# Create data directory
RUN mkdir -p /app/data

# Environment variables
ENV MEM_HOST=0.0.0.0
ENV MEM_PORT=8000
ENV MEM_DATA_DIR=/app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "ai_mem", "--host", "0.0.0.0", "--port", "8000"]
