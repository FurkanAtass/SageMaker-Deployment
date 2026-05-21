FROM nvidia/cuda:12.4.1-base-ubuntu22.04

WORKDIR /app

COPY pyproject.toml .python-version uv.lock ./
RUN apt update && \
    apt install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv sync --frozen --no-cache && \
    # Clean up apt cache and remove package lists to minimize final image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8080
RUN printf '#!/bin/sh\nexec uvicorn src.services.cloth_detection.app:app --host 0.0.0.0 --port 8080\n' \
    > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

