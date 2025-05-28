#!/usr/bin/env sh
# restart.sh – rebuild and restart the Compose project

set -e  # exit immediately if a command fails
set -u  # treat unset variables as errors

# PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# echo "▶ Stopping any running containers..."
# docker compose -f "$PROJECT_DIR/compose.yaml" down --remove-orphans

# echo "▶ Rebuilding images and starting containers..."
# docker compose -f "$PROJECT_DIR/compose.yaml" up --build -d

# echo "✔ Done. Use 'docker compose logs -f' to follow logs."

docker compose -f "docker-compose.yaml" down --remove-orphans

docker compose -f "docker-compose.yaml" up --build -d
