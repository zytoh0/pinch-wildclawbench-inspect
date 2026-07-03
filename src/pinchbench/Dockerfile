FROM node:22-bookworm@sha256:c601a46abb4d2ab80a9dc3da208d50d1122642d53f17a101926ace71e5a9bf1c

LABEL org.opencontainers.image.source="https://github.com/zytoh0/pinch-wildclawbench-inspect"
LABEL org.opencontainers.image.description="Inspect AI PinchBench wrapper runner"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    jq \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g openclaw@2026.6.10

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    fabric==3.2.3 \
    paramiko==5.0.0 \
    pyyaml==6.0.3

COPY src/_openai_compatible_proxy.py /usr/local/bin/pinchbench_model_proxy.py
COPY src/pinchbench/openclaw_compat.py /usr/local/bin/pinchbench_openclaw_compat.py

WORKDIR /benchmark/pinchbench_skill
ENV PYTHONUNBUFFERED=1
ENTRYPOINT []
CMD ["python3", "scripts/benchmark.py", "--help"]
