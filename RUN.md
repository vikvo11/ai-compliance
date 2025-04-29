# 🚀 Running AI Compliance Flask App with Docker

Build the Docker image:
```bash
docker build -t ai-compliance .
```

Run the container:
```bash
docker run -p 5000:5000 ai-compliance
```

Then open your browser and go to:
```
http://localhost:5000
```

This will serve the Flask app on port 5000 using Gunicorn.
