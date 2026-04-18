FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python-is-python3 \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fix SSL issues globally
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Copy requirements
COPY requirements.txt .

# Install Python packages (as root)
RUN python -m pip install --upgrade pip certifi && \
    python -m pip install -r requirements.txt

# Create vscode user AFTER installs
RUN useradd -ms /bin/bash vscode

# Set workspace
WORKDIR /workspace

# Switch to vscode user
USER vscode