#!/bin/bash
set -e

echo "🚀 Deploying Echo Brain..."

# Run tests
make test

# Build Docker image
make docker-build

# Tag and push
docker tag tower-echo-brain:latest tower-echo-brain:$(git rev-parse --short HEAD)

# Deploy
docker-compose up -d echo-brain

echo "✅ Deployment complete!"
