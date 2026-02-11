#!/bin/bash
# Tower Echo Brain startup script with Vault integration

# Source Vault token from environment (set in systemd unit or .env)
export VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
# VAULT_TOKEN should be set via systemd Environment or sourced from a secure file

# Database credentials from environment
export DB_PASSWORD="${DB_PASSWORD:?DB_PASSWORD must be set}"
export PGPASSWORD="$DB_PASSWORD"
export DATABASE_URL="postgresql://patrick:${DB_PASSWORD}@localhost:5432/echo_brain"

# Echo Brain configuration
export ECHO_BRAIN_PORT=8309
export QDRANT_URL="http://localhost:6333"
export OLLAMA_URL="http://localhost:11434"
export EMBEDDING_MODEL="nomic-embed-text"

# Activate virtual environment and start the service
cd /opt/tower-echo-brain
source venv/bin/activate
exec python3 -m uvicorn src.main:app --host 0.0.0.0 --port $ECHO_BRAIN_PORT