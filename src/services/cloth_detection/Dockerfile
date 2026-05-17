FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04

COPY pyproject.toml .python-version ./
# TODO did not created the image, need to check the error logs
RUN apt update && \
    apt install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv sync && \
    # Clean up apt cache and remove package lists to minimize final image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY src ./src

EXPOSE 8080
CMD ["./run.sh"]    