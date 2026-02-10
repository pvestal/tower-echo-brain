#!/bin/bash
# Tower Echo Brain startup script with Vault integration

# Source Vault token
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="hvs.yI6XmqUjZABfcJTxJHzTaruB"

# Set database password from environment
export PGPASSWORD="RP78eIrW7cI2jYvL5akt1yurE"
export DATABASE_URL="postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost:5432/echo_brain"
export DB_PASSWORD="RP78eIrW7cI2jYvL5akt1yurE"

# Echo Brain configuration
export ECHO_BRAIN_PORT=8309
export QDRANT_URL="http://localhost:6333"
export OLLAMA_URL="http://localhost:11434"
export EMBEDDING_MODEL="nomic-embed-text"

# Activate virtual environment and start the service
cd /opt/tower-echo-brain
source venv/bin/activate
exec python3 -m uvicorn src.main:app --host 0.0.0.0 --port $ECHO_BRAIN_PORT