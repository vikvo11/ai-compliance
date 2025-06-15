######################################################################
# Stage 1 – build wheels for dependencies (including uv)             #
######################################################################
FROM --platform=linux/amd64 python:3.11-slim AS build

WORKDIR /wheels
COPY requirements.txt .

# Build wheels for project deps and for the uv CLI
RUN pip wheel --no-cache-dir -r requirements.txt && \
    pip wheel --no-cache-dir uv

######################################################################
# Stage 2 – lightweight runtime image                                #
######################################################################
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /mcp

# Curl is handy for health checks / debug
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the pre-built wheels (includes uv) and remove temp folder
COPY --from=build /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy application source after dependencies for better layer caching
COPY . .

EXPOSE 8000

# Simple health check: endpoint must return 200
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD \
  curl -fsS http://localhost:8000/mcp/ || exit 1

# Run the MCP server with uv
CMD ["uv", "run", "server.py", "--port", "8000", "--host", "0.0.0.0"]