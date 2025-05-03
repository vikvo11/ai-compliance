#!/bin/bash

# Step 1: Pull latest changes from Git
echo "Ensuring local ./data directory exists with write permissions..."
mkdir -p ./data
chmod 777 ./data
echo "Pulling latest changes from Git..."
git pull

# Step 2: Build Docker image
echo "Building Docker image..."
docker build -t ai-compliance .

# Step 3: Remove old container if it exists
echo "Removing old container if it exists..."
docker rm -f ai-compliance-container 2>/dev/null

# Step 4: Run new container
echo "Running new Docker container..."
docker run --name ai-compliance-container -p 5005:5005 -v $(pwd)/data:/app/data ai-compliance
