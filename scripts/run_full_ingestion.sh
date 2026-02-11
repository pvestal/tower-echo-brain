#!/bin/bash
# Echo Brain Full Ingestion Script
# Runs all ingestion tasks with proper error handling and logging

set -e  # Exit on error
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/opt/tower-echo-brain/venv"
LOG_FILE="/tmp/echo_brain_ingestion_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Echo Brain ingestion at $(date)" | tee -a "$LOG_FILE"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Set environment variables
export PYTHONPATH="/opt/tower-echo-brain:$PYTHONPATH"

# Fetch database credentials from Vault
source "$SCRIPT_DIR/vault_get_db_password.sh"

# Function to run script with error handling
run_ingestion() {
    local script_name=$1
    local description=$2

    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Running: $description" | tee -a "$LOG_FILE"
    echo "Script: $script_name" | tee -a "$LOG_FILE"
    echo "Started: $(date)" | tee -a "$LOG_FILE"

    if python "$SCRIPT_DIR/$script_name" >> "$LOG_FILE" 2>&1; then
        echo "✅ Success: $description" | tee -a "$LOG_FILE"
    else
        echo "❌ Failed: $description (continuing anyway)" | tee -a "$LOG_FILE"
    fi

    echo "Completed: $(date)" | tee -a "$LOG_FILE"
}

# Main ingestion tasks
echo "=== STARTING INGESTION PIPELINE ===" | tee -a "$LOG_FILE"

# 1. Ingest Claude conversations (most important)
run_ingestion "ingest_conversations.py" "Claude Conversations"

# 2. Ingest source code changes
run_ingestion "ingest_source_code.py" "Source Code"

# 3. Ingest database schemas
run_ingestion "ingest_schemas.py" "Database Schemas"

# Summary
echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "=== INGESTION COMPLETE ===" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Check results
echo "Checking ingestion results..." | tee -a "$LOG_FILE"
python -c "
import psycopg2
import os
from datetime import datetime, timedelta

conn = psycopg2.connect(
    host='localhost',
    database='echo_brain',
    user='patrick',
    password=os.environ.get('DB_PASSWORD')
)

cur = conn.cursor()
cur.execute('''
    SELECT
        'conversations' as table_name,
        COUNT(*) as total,
        COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as last_hour
    FROM conversations
    UNION ALL
    SELECT
        'claude_conversations',
        COUNT(*),
        COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END)
    FROM claude_conversations
''')

print('\nIngestion Results:')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]} total, {row[2]} new in last hour')

cur.close()
conn.close()
" | tee -a "$LOG_FILE"

exit 0