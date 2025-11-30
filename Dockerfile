# NIS TOOLKIT SUIT v4.0.0 - Production Dockerfile
# Multi-stage build for optimized container size and security

FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="NIS Protocol Team"
LABEL version="4.0.0"
LABEL description="NIS TOOLKIT SUIT - Universal AI Development Framework"

# Security: Create non-root user
RUN groupadd -r nis && useradd -r -g nis nis

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY constraints.txt .
COPY nis-core-toolkit/requirements.txt ./nis-core-toolkit/

# Upgrade pip and install Python dependencies in one optimized layer
RUN pip install --upgrade --no-cache-dir pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt -c constraints.txt && \
    pip cache purge

#=============================================================================
# Development Stage
#=============================================================================
FROM base AS development

# Install development dependencies in one optimized layer
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter \
    ipython && \
    pip cache purge

# Copy source code
COPY . .

# Change ownership to nis user
RUN chown -R nis:nis /app

# Switch to non-root user
USER nis

# Expose ports
EXPOSE 8000 8080 9090

# Development entrypoint
CMD ["python", "-m", "examples.nis_v321_migration_demo"]

#=============================================================================
# Production Stage
#=============================================================================
FROM base AS production

# Copy only necessary files
COPY nis-core-toolkit/ ./nis-core-toolkit/
COPY nis-agent-toolkit/ ./nis-agent-toolkit/
COPY nis-integrity-toolkit/ ./nis-integrity-toolkit/
COPY examples/ ./examples/
COPY VERSION .
COPY UPGRADE_TO_V321.md .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R nis:nis /app

# Switch to non-root user
USER nis

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose ports
EXPOSE 8000

# Production entrypoint
CMD ["python", "-m", "nis-core-toolkit.cli.main"]

#=============================================================================
# Edge Computing Stage
#=============================================================================
FROM python:3.11-alpine AS edge

# Metadata for edge
LABEL variant="edge"
LABEL size="minimal"

# Install minimal dependencies and create user in one layer
RUN apk add --no-cache gcc musl-dev && \
    addgroup -S nis && adduser -S nis -G nis

WORKDIR /app

# Copy and install minimal requirements for edge deployment
COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check \
    $(grep -E "(fastapi|pydantic|numpy)" requirements.txt) && \
    pip cache purge

# Copy minimal core
COPY nis-core-toolkit/src/core/ ./core/
COPY examples/nis_v321_edge_deployment.py ./

# Set user
USER nis

# Edge-optimized entrypoint
CMD ["python", "nis_v321_edge_deployment.py"]
