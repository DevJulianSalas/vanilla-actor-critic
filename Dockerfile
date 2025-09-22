FROM python:3.9-slim


#install dependencies for clang
# Install necessary packages for Clang and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    clang \
    llvm \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy requirements.txt to a temporary location first for better layer caching
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/ ./src

RUN chown -R appuser:appgroup /app

USER appuser