FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Ensure data dir exists and pre-create database
RUN mkdir -p data && python setup_db.py

EXPOSE 5005

CMD ["python", "app.py"]