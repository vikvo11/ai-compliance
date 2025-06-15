######################################################################
# --------------- Stage 1 — build all wheels once ------------------ #
######################################################################
# If you always build for a single architecture, leave the
# --platform flag; otherwise pass it at build time, e.g.:
#   DOCKER_BUILDKIT=1 docker build --platform=linux/amd64 -t ui .
FROM --platform=linux/amd64 python:3.11-slim AS build

WORKDIR /wheels

# Copy requirements first to maximise cache hits
COPY requirements.txt .

# Build wheels for every dependency; nothing is installed yet
# We add `uv` CLI here because the runtime stage will use it
RUN pip wheel --no-cache-dir -r requirements.txt \
 && pip wheel --no-cache-dir uv


######################################################################
# -------------------- Stage 2 — runtime image --------------------- #
######################################################################
FROM --platform=linux/amd64 python:3.11-slim

# Disable .pyc creation and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /ui

# Install curl (used for health checks) while still root
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the pre-built wheels from stage 1 and install them
COPY --from=build /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels

# Copy application source code after dependencies (better layer cache)
COPY . .

EXPOSE 8001

# Lightweight liveness probe (defined in app.py)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD \
  curl -fsS http://localhost:8001/health || exit 1

# Stand-alone entry point: run the Flask UI via uv
CMD ["uv", "run", "app.py", "--port", "8001", "--host", "0.0.0.0"]
