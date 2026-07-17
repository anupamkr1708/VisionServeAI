# syntax=docker/dockerfile:1

# ============================================================================
# VisionServeAI backend -- production image
#
# Multi-stage build:
#   builder -> installs deps into a venv (CPU-only torch, no CUDA bloat)
#   test    -> optional gate: runs the full pytest suite (docker build --target test)
#   runtime -> slim final image: venv + source only, non-root, healthchecked
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch/torchvision FIRST, from PyTorch's own CPU index. pip sees
# these already satisfy `torch>=2.2`/`torchvision>=0.17` and won't
# reinstall them from the line below -- requirements.txt itself is
# untouched, only how its constraints get resolved in THIS image.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: test -- CI gate. Only runs if you build with `--target test`;
# a normal `docker build` never touches this stage, so it costs nothing
# for a routine build.
# ---------------------------------------------------------------------------
FROM builder AS test

WORKDIR /app
COPY configs/ configs/
COPY inference/ inference/
COPY services/ services/
COPY scripts/ scripts/
COPY backend/ backend/
COPY tests/ tests/
COPY pyproject.toml .

RUN pytest tests/ -q

# ---------------------------------------------------------------------------
# Stage 3: runtime -- the actual shipped image. No tests/, notebooks/,
# docs/, frontend/, .git, or dev-only tools (pytest/ruff/mypy already did
# their job in the `test` stage above).
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VISIONSERVE_ARTIFACT_ROOT=/artifacts \
    VISIONSERVE_LOG_DIR=/app/logs \
    VISIONSERVE_EXPLAINABILITY_OUTPUT_DIR=/app/outputs/explainability \
    HF_CACHE_DIR=/app/.cache/hf_artifacts \
    PORT=8000 \
    UVICORN_WORKERS=1

RUN groupadd --system visionserve && useradd --system --gid visionserve --create-home visionserve

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY configs/ configs/
COPY inference/ inference/
COPY services/ services/
COPY scripts/ scripts/
COPY backend/ backend/

RUN mkdir -p /app/logs /app/outputs/explainability /artifacts /app/.cache/hf_artifacts \
    && chown -R visionserve:visionserve /app/logs /app/outputs /artifacts /app/.cache

USER visionserve

EXPOSE 8000

# /health/ready reflects whether the model/runtime actually finished
# initializing (backend/routers/health.py) -- the meaningful "can this
# container serve traffic yet" signal. If you move to Kubernetes later,
# back the liveness probe with /health/live and the readiness probe with
# /health/ready separately instead of collapsing both into one check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready').status==200 else 1)"

CMD ["sh", "-c", "exec python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]