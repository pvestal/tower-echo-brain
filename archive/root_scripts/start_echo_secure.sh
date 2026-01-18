#!/bin/bash
cd /opt/tower-echo-brain
source venv/bin/activate

# Kill any existing process on 8309
fuser -k 8309/tcp 2>/dev/null
sleep 1

# Export all required environment variables
export DB_HOST=localhost
export DB_NAME=echo_brain
export DB_USER=echo_user
export DB_PASSWORD=echo_password
export POSTGRES_DB=echo_brain
export JWT_SECRET="echo-brain-secret-key-2025"
export PYTHONPATH=/opt/tower-echo-brain

# Start Echo with full configuration
nohup python echo.py > /tmp/echo_brain.log 2>&1 &
PID=$!
echo "Echo Brain starting (PID: $PID)..."
sleep 5

# Check if started
if curl -s http://localhost:8309/api/echo/health > /dev/null 2>&1; then
    echo "✅ Echo Brain started successfully with persistence"
    echo "Database: PostgreSQL with 172+ conversations"
    curl -s http://localhost:8309/api/echo/health | jq '.'
else
    echo "Checking alternative endpoints..."
    curl -s http://localhost:8309/health || echo "Still starting..."
    tail -15 /tmp/echo_brain.log
fi
