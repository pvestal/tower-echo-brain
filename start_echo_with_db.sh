#!/bin/bash
cd /opt/tower-echo-brain
source venv/bin/activate

# Database configuration
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=echo_brain
export PGUSER=echo_user
export PGPASSWORD=echo_password

# Alternative format
export DATABASE_URL="postgresql://echo_user:echo_password@localhost:5432/echo_brain"
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=echo_brain
export DB_USER=echo_user
export DB_PASSWORD=echo_password

# JWT and other configs
export JWT_SECRET="echo-brain-secret-2025"
export PYTHONPATH=/opt/tower-echo-brain

echo "Starting Echo Brain with database persistence..."
python echo.py
