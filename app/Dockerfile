######################################################################
# ----------------- Stage 1 — build all wheels once ---------------- #
######################################################################
# If you always build for a single architecture, leave the
# --platform flag; otherwise pass it at build time:
#   DOCKER_BUILDKIT=1 docker build --platform=linux/amd64 -t myapp .
FROM --platform=linux/amd64 python:3.11-slim AS build

WORKDIR /wheels

# Copy only the requirements first to maximise cache hits
COPY requirements.txt .

# Build wheels for every dependency; nothing is installed yet
RUN pip wheel --no-cache-dir -r requirements.txt


######################################################################
# ----------------- Stage 2 — runtime image ------------------------ #
######################################################################
FROM --platform=linux/amd64 python:3.11-slim

# Disable .pyc creation and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install curl (used for HTTP requests or health checks)
RUN apt-get update && apt-get install -y curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the pre-built wheels from stage 1
COPY --from=build /wheels /tmp/wheels

# Install ONLY .whl files and then delete the temp directory
RUN pip install --no-cache-dir /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels

# Copy application source code after dependencies (better layer cache)
COPY . .

# Create runtime data directory
RUN mkdir -p /app/data

EXPOSE 5005

# Development/stand-alone entry point
CMD ["python", "app.py"]