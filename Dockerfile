FROM python:3.11-slim

# Reduce size by avoiding cache and unnecessary locales
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install essential build tools only when needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Create persistent volume directory
RUN mkdir -p /app/data

EXPOSE 5005

CMD ["python", "app.py"]
