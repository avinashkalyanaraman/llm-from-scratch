FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=60

# System utilities (vim included)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    vim \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install uv
RUN python -m pip install -U pip uv

WORKDIR /workspace

# Copy requirements first to leverage Docker layer caching
COPY runpod_requirements.txt /workspace/runpod_requirements.txt

# Install Python dependencies into system Python
RUN uv pip install --system -r /workspace/runpod_requirements.txt

# Default shell
CMD ["bash"]
