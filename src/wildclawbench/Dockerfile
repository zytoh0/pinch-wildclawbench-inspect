FROM node:22-bookworm@sha256:c601a46abb4d2ab80a9dc3da208d50d1122642d53f17a101926ace71e5a9bf1c

LABEL org.opencontainers.image.source="https://github.com/zytoh0/pinch-wildclawbench-inspect"
LABEL org.opencontainers.image.description="Inspect AI WildClawBench wrapper runner"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    jq \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g openclaw@2026.6.10

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    beautifulsoup4==4.15.0 \
    lxml==6.1.1 \
    pypdf==6.14.2 \
    python-dotenv==1.2.2 \
    pyyaml==6.0.3 \
    requests==2.34.2

WORKDIR /tmp_workspace
ENV PYTHONUNBUFFERED=1
ENTRYPOINT []
CMD ["tail", "-f", "/dev/null"]
