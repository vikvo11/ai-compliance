FROM python:3.11-slim

# Reduce size by avoiding cache and unnecessary locales
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


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
