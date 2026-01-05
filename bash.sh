#!/bin/bash

echo "Starting deployment"

# Check if Docker exists
command -v docker >/dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Docker is not installed"
  exit 1
fi

echo "Docker is installed"

# Start containers
echo "Starting containers"
docker compose up -d --build

# Wait a bit
sleep 5

# Check if app is running
curl http://localhost:8000/ >/dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "App is not running"
  echo "Rolling back..."
  docker compose down
  exit 1
fi

echo "Deployment successful"
