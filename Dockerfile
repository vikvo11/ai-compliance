# ---------- Stage 1: build wheels ----------
FROM --platform=linux/amd64 python:3.11-slim AS build

WORKDIR /wheels
COPY requirements.txt .

# Build wheels (only once)
RUN pip wheel --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install pre-built wheels (no tmp bloat)
COPY --from=build /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/* && rm -rf /tmp/wheels

COPY . .
RUN mkdir -p /app/data
EXPOSE 5005

# Prefer gunicorn in production; else switch CMD
CMD ["gunicorn", "--bind", "0.0.0.0:5005", "app:app"]
