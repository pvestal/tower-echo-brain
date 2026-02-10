# Echo Brain Environment Variables

## Required Variables

### Database Configuration

```bash
# PostgreSQL connection
DATABASE_URL=postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost:5432/echo_brain
DB_PASSWORD=RP78eIrW7cI2jYvL5akt1yurE
DB_USER=patrick
DB_NAME=echo_brain
DB_HOST=localhost
DB_PORT=5432

# Connection pool settings
DB_POOL_MIN=2
DB_POOL_MAX=10
```

### Vector Database

```bash
# Qdrant configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=echo_memory
QDRANT_API_KEY=  # Optional, for production
```

### Service Configuration

```bash
# Echo Brain API
ECHO_BRAIN_PORT=8309
ECHO_BRAIN_HOST=0.0.0.0
API_BASE_PATH=/api/echo

# Service identification
SERVICE_NAME=echo-brain
SERVICE_VERSION=0.4.0
```

### Embedding Configuration

```bash
# Ollama embedding service
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
EMBEDDING_BATCH_SIZE=100

# Optional: OpenAI fallback
OPENAI_API_KEY=  # If provided, uses OpenAI embeddings
```

### Agent Models

```bash
# Ollama models for agents
CODING_AGENT_MODEL=deepseek-coder-v2:16b
REASONING_AGENT_MODEL=deepseek-r1:8b
NARRATION_AGENT_MODEL=gemma2:9b

# Default query model
DEFAULT_LLM_MODEL=mistral:7b
```

## Optional Variables

### Logging & Monitoring

```bash
# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO
LOG_FORMAT=json  # or 'text'
LOG_FILE=/var/log/echo-brain/app.log

# Metrics
ENABLE_METRICS=true
METRICS_PORT=9090
```

### Performance Tuning

```bash
# Request handling
MAX_CONCURRENT_REQUESTS=50
REQUEST_TIMEOUT_SECONDS=60
LLM_TIMEOUT_SECONDS=120

# Memory settings
MAX_CONTEXT_LENGTH=8192
MAX_SEARCH_RESULTS=20
CACHE_TTL_SECONDS=3600
```

### Security

```bash
# API authentication (future)
ENABLE_AUTH=false
API_KEY_HEADER=X-API-Key

# CORS settings
CORS_ORIGINS=http://localhost,http://192.168.50.135
CORS_ALLOW_CREDENTIALS=true
```

### HashiCorp Vault Integration

```bash
# Vault for secrets management
VAULT_ADDR=http://127.0.0.1:8200
VAULT_TOKEN=  # Or use ~/.vault-token file
VAULT_MOUNT_POINT=secret
VAULT_PATH_PREFIX=echo-brain/
```

### Development & Debug

```bash
# Development mode
ENVIRONMENT=development  # or 'production'
DEBUG=false
RELOAD=false  # Auto-reload on code changes

# Testing
TEST_MODE=false
TEST_DATABASE_URL=postgresql://patrick:password@localhost:5432/echo_brain_test
```

### Ingestion Pipeline

```bash
# Automated ingestion
ENABLE_AUTO_INGESTION=true
INGESTION_INTERVAL_HOURS=6
INGESTION_SOURCE_PATH=/home/patrick/.claude/projects/

# Batch processing
INGESTION_BATCH_SIZE=100
INGESTION_MAX_WORKERS=4
```

### Frontend Configuration

```bash
# Vue.js frontend
FRONTEND_PORT=8080
FRONTEND_BASE_PATH=/echo-brain
API_PROXY_PATH=/api
VITE_API_BASE_URL=http://localhost:8309
```

### Network & Deployment

```bash
# Network settings
BIND_ADDRESS=0.0.0.0
ALLOWED_HOSTS=localhost,192.168.50.135
PROXY_HEADERS=true  # Behind nginx

# Health checks
HEALTH_CHECK_PATH=/health
READINESS_PATH=/api/ready
LIVENESS_PATH=/api/alive
```

### Resource Limits

```bash
# Memory management
MAX_MEMORY_MB=2048
VECTOR_CACHE_SIZE_MB=512

# GPU settings (if applicable)
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.8
```

## Loading Order

1. **System environment** - OS-level variables
2. **`.env` file** - Project root directory
3. **Vault secrets** - If VAULT_TOKEN is set
4. **Default values** - Hardcoded in application

## Example `.env` File

```bash
# Core configuration
DATABASE_URL=postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost:5432/echo_brain
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
ECHO_BRAIN_PORT=8309

# Models
EMBEDDING_MODEL=nomic-embed-text
CODING_AGENT_MODEL=deepseek-coder-v2:16b
REASONING_AGENT_MODEL=deepseek-r1:8b
NARRATION_AGENT_MODEL=gemma2:9b

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

## Docker Environment

When running in Docker, pass environment variables via:

```bash
# Docker run
docker run -d \
  -e DATABASE_URL="postgresql://..." \
  -e QDRANT_URL="http://qdrant:6333" \
  -p 8309:8309 \
  echo-brain:latest

# Docker Compose
version: '3.8'
services:
  echo-brain:
    image: echo-brain:latest
    environment:
      - DATABASE_URL=postgresql://patrick:password@db:5432/echo_brain
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_URL=http://ollama:11434
    env_file:
      - .env
```

## Systemd Service

For systemd services, set environment in the unit file:

```ini
[Service]
Environment="DATABASE_URL=postgresql://..."
Environment="ECHO_BRAIN_PORT=8309"
EnvironmentFile=/etc/echo-brain/environment
```

## Validation Script

Check all required variables are set:

```python
#!/usr/bin/env python3
import os
import sys

required = [
    "DATABASE_URL",
    "QDRANT_URL",
    "OLLAMA_URL",
    "EMBEDDING_MODEL"
]

missing = []
for var in required:
    if not os.getenv(var):
        missing.append(var)

if missing:
    print(f"L Missing required variables: {', '.join(missing)}")
    sys.exit(1)
else:
    print(" All required environment variables are set")
```

## Security Notes

1. **Never commit `.env` files** to version control
2. **Use Vault** for production secrets
3. **Rotate API keys** regularly
4. **Limit database permissions** to minimum required
5. **Use read-only replicas** where possible