#!/bin/bash
cd /opt/tower-echo-brain
source venv/bin/activate

# Kill any existing process on 8309
fuser -k 8309/tcp 2>/dev/null
sleep 1

# Export database configuration
export DB_HOST=localhost
export DB_NAME=echo_brain
export DB_USER=echo_user
export DB_PASSWORD=echo_password
export POSTGRES_DB=echo_brain

# Start Echo with persistence
nohup python echo.py > /tmp/echo_brain.log 2>&1 &
echo "Echo Brain starting on port 8309..."
sleep 3

# Check if started
if curl -s http://localhost:8309/api/echo/health > /dev/null 2>&1; then
    echo "✅ Echo Brain started successfully"
else
    echo "⚠️ Echo Brain may not have started. Checking logs:"
    tail -20 /tmp/echo_brain.log
fi
