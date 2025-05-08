# ---------- Stage 1: build wheels ----------
FROM python:3.11-slim AS build

WORKDIR /wheels
COPY requirements.txt .

# Build wheels once; cache them as .whl files
RUN pip wheel --no-binary=:none: -r requirements.txt

# ---------- Stage 2: final runtime image ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the already-built wheels (no cache, no compile)
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . .
RUN mkdir -p /app/data
EXPOSE 5005
CMD ["python", "app.py"]
