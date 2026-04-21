# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the mm-cxr-diag inference service.
#
# The base image ships a working PyTorch + CUDA toolchain, so we avoid the
# slow/fragile pip-install-torch-in-CI path. The builder stage compiles a
# self-contained venv under /opt/venv; the runtime stage copies only that
# venv + the source tree, dropping ~all build artifacts.

ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# ─── builder ──────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy only metadata first so dependency resolution is cache-friendly.
COPY pyproject.toml README.md ./
COPY mm_cxr_diag/ ./mm_cxr_diag/
COPY configs/ ./configs/

# Use a dedicated venv so runtime copies are deterministic.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Torch is preinstalled in the base; pin it into the venv by symlink so our
# install step doesn't re-resolve a possibly mismatched CUDA build.
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-deps "/build" && \
    python -m pip install ".[serve]"

# ─── runtime ──────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MM_CXR_LOG_LEVEL=info

# Create a non-root user. UID 10001 is a common "app" range.
RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --home /home/app --create-home app

WORKDIR /home/app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app configs/ ./configs/

USER app

EXPOSE 8000

# Mount checkpoints into /ckpts at runtime and set env vars — e.g.:
#   docker run -v $(pwd)/checkpoints:/ckpts \
#     -e MM_CXR_STAGE1_CKPT=/ckpts/stage1.pth \
#     -e MM_CXR_STAGE2_CKPT=/ckpts/stage2.pth \
#     -p 8000:8000 mm-cxr-diag:latest
#
# We do not bake checkpoints into the image — that keeps the image generic
# and lets users swap models without a rebuild.

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; \
u=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3); \
sys.exit(0 if u.status==200 else 1)" || exit 1

ENTRYPOINT ["mm-cxr-diag", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
