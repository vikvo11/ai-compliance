######################################################################
# ----------------- Stage 1 — build all wheels once ---------------- #
######################################################################
# If you always build for a single architecture, leave the
# --platform flag; otherwise pass it at build time:
#   DOCKER_BUILDKIT=1 docker build --platform=linux/amd64 -t myapp .
FROM --platform=linux/amd64 python:3.11-slim AS build

WORKDIR /wheels

# Copy only the requirements first to maximise cache-hits
COPY requirements.txt .

# Build wheels for every dependency; nothing is installed yet
# English comments as requested
RUN pip wheel --no-cache-dir -r requirements.txt

######################################################################
# ----------------- Stage 2 — runtime image ------------------------ #
######################################################################
FROM --platform=linux/amd64 python:3.11-slim

# Disable .pyc creation and output buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

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

# Production entry point (swap for `CMD ["python", "app.py"]` if desired)
CMD ["gunicorn", "--bind", "0.0.0.0:5005", "app:app"]
