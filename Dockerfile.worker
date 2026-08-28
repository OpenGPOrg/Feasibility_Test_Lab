# ═══════════════════════════════════════════════════════════════════════════════
# Distributed Inference — Worker Node Docker Image
#
# Build:  docker build -f Dockerfile.worker -t dist-worker .
# Run:    docker run -p 9900:9900 -p 9899:9899/udp dist-worker
#
# With persistent component cache (recommended):
#   docker run -p 9900:9900 -p 9899:9899/udp \
#              -v dist-worker-cache:/root/.dist_inference_cache \
#              dist-worker
#
# Custom port:
#   docker run -p 9901:9901 -p 9899:9899/udp dist-worker --port 9901
#
# Custom broadcast address (for VPN):
#   docker run -p 9900:9900 -p 9899:9899/udp --network host \
#              dist-worker --broadcast 10.8.0.255
#
# For VPN (use host networking so UDP broadcast works):
#   docker run --network host dist-worker
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim

LABEL maintainer="LoadTime Distributed Inference"
LABEL description="Worker node for distributed LLM inference"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# CPU-only PyTorch to keep image small (~700MB vs ~5GB with CUDA)
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --default-timeout=1000 --retries 10 \
        transformers \
        psutil \
        accelerate \
        bitsandbytes

# Create cache directory for component persistence
RUN mkdir -p /root/.dist_inference_cache

WORKDIR /app

# Copy only the worker script
COPY worker_node.py .

# Expose TCP (inference) and UDP (discovery) ports
EXPOSE 9900/tcp
EXPOSE 9899/udp

# Health check — verify the TCP port is listening
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost',9900)); s.close()" || exit 1

ENTRYPOINT ["python", "worker_node.py"]
CMD []
