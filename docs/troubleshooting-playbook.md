# Echo Brain Troubleshooting Playbook & Operational Runbooks

## Overview

This comprehensive troubleshooting guide provides systematic approaches to diagnosing and resolving issues with the Echo Brain Advanced AI Orchestrator. It covers common problems, diagnostic procedures, and operational maintenance tasks.

## Table of Contents

1. [Quick Diagnostic Commands](#quick-diagnostic-commands)
2. [Common Issues & Solutions](#common-issues--solutions)
3. [Service Health Diagnostics](#service-health-diagnostics)
4. [Model Management Issues](#model-management-issues)
5. [Board of Directors Problems](#board-of-directors-problems)
6. [WebSocket & Streaming Issues](#websocket--streaming-issues)
7. [Database & Storage Problems](#database--storage-problems)
8. [Performance Issues](#performance-issues)
9. [Security & Authentication Problems](#security--authentication-problems)
10. [Operational Runbooks](#operational-runbooks)
11. [Emergency Procedures](#emergency-procedures)
12. [Monitoring & Alerting](#monitoring--alerting)

---

## Quick Diagnostic Commands

### Health Check Commands

```bash
# Quick system health check
curl -k https://192.168.50.135/api/echo/health

# Direct service health (bypassing nginx)
curl http://192.168.50.135:8309/api/echo/health

# Check all Tower services status
curl -k https://192.168.50.135/api/echo/tower/status

# Test specific service connectivity
curl -X POST http://192.168.50.135:8309/api/echo/test/comfyui \
  -H "Content-Type: application/json" \
  -d '{"target": "comfyui", "test_type": "universal"}'
```

### Service Status Commands

```bash
# Check Echo Brain systemd service
sudo systemctl status tower-echo-brain

# View service logs
sudo journalctl -u tower-echo-brain -f --lines=50

# Check Ollama service
sudo systemctl status ollama
curl http://localhost:11434/api/tags

# Check PostgreSQL connection
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "SELECT 1;"

# Check Redis connection
redis-cli -h 192.168.50.135 ping
```

### Model Management Diagnostics

```bash
# List installed models
curl http://localhost:11434/api/tags

# Check model storage usage
du -sh /usr/share/ollama/.ollama/models/

# Test model inference
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "Test query", "stream": false}'
```

---

## Common Issues & Solutions

### Issue 1: Service Won't Start

**Symptoms:**
- `sudo systemctl start tower-echo-brain` fails
- Port 8309 not listening
- Service immediately exits

**Diagnostic Steps:**
```bash
# Check service status and logs
sudo systemctl status tower-echo-brain
sudo journalctl -u tower-echo-brain -n 50

# Check port availability
netstat -tlnp | grep :8309
lsof -i :8309

# Verify Python environment
source /opt/tower-echo-brain/venv/bin/activate
python -c "import fastapi, uvicorn, psycopg2; print('Dependencies OK')"

# Check configuration files
ls -la /opt/tower-echo-brain/.env
cat /opt/tower-echo-brain/.env
```

**Solutions:**

1. **Port Already in Use:**
   ```bash
   # Kill process using port 8309
   sudo kill $(lsof -t -i:8309)
   # Or restart the service
   sudo systemctl restart tower-echo-brain
   ```

2. **Missing Dependencies:**
   ```bash
   cd /opt/tower-echo-brain
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Database Connection Issues:**
   ```bash
   # Test PostgreSQL connection
   psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "\dt"

   # Check .env file for correct database credentials
   grep -E "DB_|DATABASE_" /opt/tower-echo-brain/.env
   ```

4. **Ollama Not Available:**
   ```bash
   # Start Ollama service
   sudo systemctl start ollama
   sudo systemctl enable ollama

   # Verify Ollama is responding
   curl http://localhost:11434/api/tags
   ```

### Issue 2: AI Queries Timing Out

**Symptoms:**
- `/api/echo/query` requests timeout
- Model responses take too long
- 504 Gateway Timeout errors

**Diagnostic Steps:**
```bash
# Check Ollama service status
sudo systemctl status ollama
curl http://localhost:11434/api/tags

# Test direct model inference
time curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "Hello", "stream": false}'

# Check system resources
htop
nvidia-smi  # If using GPU
df -h  # Check disk space

# Monitor Echo Brain logs during query
sudo journalctl -u tower-echo-brain -f &
curl -X POST http://localhost:8309/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "intelligence_level": "basic"}'
```

**Solutions:**

1. **Model Not Loaded:**
   ```bash
   # Check if model is installed
   curl http://localhost:11434/api/tags

   # Pull missing model
   curl http://localhost:11434/api/pull \
     -H "Content-Type: application/json" \
     -d '{"name": "llama3.1:8b"}'
   ```

2. **Insufficient Resources:**
   ```bash
   # Check memory usage
   free -h

   # Check GPU memory (if applicable)
   nvidia-smi

   # Restart Ollama to clear memory
   sudo systemctl restart ollama
   ```

3. **Timeout Configuration:**
   ```bash
   # Increase nginx timeout (if using proxy)
   sudo nano /etc/nginx/sites-available/tower.conf
   # Add: proxy_read_timeout 300s;
   sudo nginx -t && sudo systemctl reload nginx
   ```

### Issue 3: Authentication Failures

**Symptoms:**
- 401 Unauthorized errors
- JWT token validation fails
- Board endpoints require authentication

**Diagnostic Steps:**
```bash
# Test auth service
curl http://192.168.50.135:8088/api/auth/health

# Check JWT token validity
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://192.168.50.135:8088/api/auth/verify

# Test Echo Brain auth integration
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://192.168.50.135:8309/api/echo/board/status
```

**Solutions:**

1. **Auth Service Down:**
   ```bash
   sudo systemctl start tower-auth
   sudo systemctl status tower-auth
   ```

2. **Invalid/Expired Token:**
   ```bash
   # Get new token
   curl -X POST http://192.168.50.135:8088/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "your_username", "password": "your_password"}'
   ```

3. **Redis Session Issues:**
   ```bash
   # Check Redis connection
   redis-cli -h 192.168.50.135 ping

   # Clear Redis sessions if needed
   redis-cli -h 192.168.50.135 flushdb
   ```

### Issue 4: WebSocket Connections Failing

**Symptoms:**
- WebSocket connections timeout
- Real-time updates not working
- Board WebSocket authentication fails

**Diagnostic Steps:**
```bash
# Test WebSocket endpoint
wscat -c ws://192.168.50.135:8309/api/echo/stream

# Test with authentication
wscat -c "ws://192.168.50.135:8309/api/echo/board/ws?token=YOUR_JWT_TOKEN"

# Check nginx WebSocket configuration
sudo nginx -t
grep -A 5 -B 5 "upgrade" /etc/nginx/sites-available/tower.conf
```

**Solutions:**

1. **Nginx WebSocket Configuration:**
   ```nginx
   # Add to location block
   proxy_http_version 1.1;
   proxy_set_header Upgrade $http_upgrade;
   proxy_set_header Connection "upgrade";
   proxy_read_timeout 86400;
   ```

2. **Firewall Issues:**
   ```bash
   # Check firewall rules
   sudo ufw status
   sudo ufw allow 8309/tcp
   ```

3. **Connection Limits:**
   ```bash
   # Check current connections
   netstat -an | grep :8309 | grep ESTABLISHED | wc -l

   # Monitor connection manager logs
   sudo journalctl -u tower-echo-brain -f | grep -i websocket
   ```

---

## Service Health Diagnostics

### Comprehensive Health Check Script

```bash
#!/bin/bash
# /opt/tower-echo-brain/scripts/health-check.sh

echo "=== Echo Brain Health Check ==="
echo "Timestamp: $(date)"
echo ""

# 1. Service Status
echo "1. Service Status:"
sudo systemctl is-active tower-echo-brain
echo "   PID: $(pgrep -f "echo.py" || echo "Not running")"
echo ""

# 2. Port Listening
echo "2. Port Status:"
if netstat -tlnp | grep -q ":8309"; then
    echo "   ✓ Port 8309 is listening"
else
    echo "   ✗ Port 8309 is NOT listening"
fi
echo ""

# 3. Dependencies
echo "3. Dependencies:"

# PostgreSQL
if psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "SELECT 1;" >/dev/null 2>&1; then
    echo "   ✓ PostgreSQL connection OK"
else
    echo "   ✗ PostgreSQL connection FAILED"
fi

# Ollama
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "   ✓ Ollama service OK"
else
    echo "   ✗ Ollama service FAILED"
fi

# Redis
if redis-cli -h 192.168.50.135 ping >/dev/null 2>&1; then
    echo "   ✓ Redis connection OK"
else
    echo "   ✗ Redis connection FAILED"
fi

# Auth Service
if curl -s http://192.168.50.135:8088/api/auth/health >/dev/null 2>&1; then
    echo "   ✓ Auth service OK"
else
    echo "   ✗ Auth service FAILED"
fi
echo ""

# 4. API Health Check
echo "4. API Health:"
if curl -s http://localhost:8309/api/echo/health >/dev/null 2>&1; then
    echo "   ✓ Echo Brain API responding"
    # Get detailed health info
    curl -s http://localhost:8309/api/echo/health | jq -r '.status'
else
    echo "   ✗ Echo Brain API NOT responding"
fi
echo ""

# 5. Resource Usage
echo "5. Resource Usage:"
echo "   Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "   Disk: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "   GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print $1"MB/"$2"MB"}')"
fi
echo ""

# 6. Recent Errors
echo "6. Recent Errors (last 10):"
sudo journalctl -u tower-echo-brain --since "1 hour ago" --no-pager | grep -i error | tail -10 || echo "   No recent errors"
```

### Automated Health Monitoring

```bash
#!/bin/bash
# /opt/tower-echo-brain/scripts/health-monitor.sh
# Run this script via cron for continuous monitoring

LOG_FILE="/opt/tower-echo-brain/logs/health-monitor.log"
ALERT_EMAIL="admin@tower.local"

# Function to send alert
send_alert() {
    local message="$1"
    echo "$(date): ALERT - $message" >> "$LOG_FILE"

    # Send email alert (if configured)
    if command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "Echo Brain Alert" "$ALERT_EMAIL"
    fi

    # Send to Echo Brain voice notification
    curl -s -X POST http://localhost:8309/api/echo/voice/notify \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$message\", \"priority\": \"urgent\"}" || true
}

# Check service status
if ! systemctl is-active --quiet tower-echo-brain; then
    send_alert "Echo Brain service is not running"
    exit 1
fi

# Check API response
if ! curl -s -f http://localhost:8309/api/echo/health >/dev/null 2>&1; then
    send_alert "Echo Brain API is not responding"
    exit 1
fi

# Check Ollama
if ! curl -s -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    send_alert "Ollama service is not responding"
    exit 1
fi

# Check database
if ! psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "SELECT 1;" >/dev/null 2>&1; then
    send_alert "PostgreSQL database is not accessible"
    exit 1
fi

# Log healthy status
echo "$(date): All systems healthy" >> "$LOG_FILE"
```

---

## Model Management Issues

### Model Download/Installation Problems

**Issue: Model Download Fails**

```bash
# Check available disk space
df -h /usr/share/ollama/.ollama/models/

# Check network connectivity
curl -I https://ollama.ai

# Check Ollama logs
sudo journalctl -u ollama -f

# Manual model download
ollama pull llama3.1:8b

# Check download progress
curl http://localhost:11434/api/ps
```

**Issue: Model Removal Blocked by Board**

```bash
# Check board decision status
curl -X GET http://localhost:8309/api/echo/board/decisions/TASK_ID

# Override board decision (admin only)
curl -X POST http://localhost:8309/api/echo/board/feedback/TASK_ID \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "feedback_type": "approve",
    "feedback_content": "Emergency removal required",
    "override_recommendation": "approve",
    "reasoning": "Critical disk space issue"
  }'
```

### Model Performance Issues

```bash
# Check model response times
time curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "Hello", "stream": false}'

# Monitor GPU usage during inference
nvidia-smi -l 1

# Check for model corruption
ollama list
ollama rm corrupted-model
ollama pull model-name
```

---

## Board of Directors Problems

### Board Decision System Issues

**Issue: Board Not Processing Tasks**

```bash
# Check board system status
curl http://localhost:8309/api/echo/board/status

# Check director registry
curl http://localhost:8309/api/echo/board/directors

# Review board logs
sudo journalctl -u tower-echo-brain -f | grep -i board

# Test board submission
curl -X POST http://localhost:8309/api/echo/board/task \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Test board functionality",
    "priority": "normal"
  }'
```

**Issue: Director Evaluation Failures**

```bash
# Check director implementation
python3 -c "
from directors.director_registry import DirectorRegistry
registry = DirectorRegistry()
print('Available directors:', list(registry.directors.keys()))
"

# Test individual director
curl -X POST http://localhost:8309/api/echo/board/test-director \
  -H "Content-Type: application/json" \
  -d '{"director_id": "security_director", "test_task": "sample task"}'
```

### Board Database Issues

```bash
# Check board tables
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
\dt board_*
SELECT COUNT(*) FROM board_decisions;
SELECT COUNT(*) FROM board_director_evaluations;
"

# Clean up stale board data
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
DELETE FROM board_decisions WHERE created_at < NOW() - INTERVAL '30 days';
"
```

---

## WebSocket & Streaming Issues

### WebSocket Connection Problems

```bash
# Test WebSocket connectivity
npm install -g wscat
wscat -c ws://localhost:8309/api/echo/stream

# Check connection limits
netstat -an | grep :8309 | grep ESTABLISHED | wc -l

# Monitor WebSocket logs
sudo journalctl -u tower-echo-brain -f | grep -i websocket
```

### Streaming Performance Issues

```bash
# Check brain activity generation
curl http://localhost:8309/api/echo/brain

# Monitor streaming performance
curl -N http://localhost:8309/api/echo/stream | head -20

# Check event broadcasting
python3 << 'EOF'
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8309/api/echo/stream"
    try:
        async with websockets.connect(uri) as websocket:
            for i in range(5):
                message = await websocket.recv()
                data = json.loads(message)
                print(f"Received: {data.get('timestamp', 'no timestamp')}")
    except Exception as e:
        print(f"WebSocket error: {e}")

asyncio.run(test_websocket())
EOF
```

---

## Database & Storage Problems

### PostgreSQL Connection Issues

```bash
# Test database connection
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "\l"

# Check connection limits
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
SELECT count(*) as active_connections,
       max_conn,
       max_conn-count(*) as remaining_connections
FROM pg_stat_activity,
     (SELECT setting::int as max_conn FROM pg_settings WHERE name='max_connections') AS mc
GROUP BY max_conn;"

# Check table sizes
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
SELECT schemaname,tablename,attname,n_distinct,correlation
FROM pg_stats
WHERE tablename LIKE 'echo_%'
ORDER BY tablename, attname;"
```

### Storage Space Issues

```bash
# Check disk usage
df -h
du -sh /usr/share/ollama/.ollama/models/*
du -sh /opt/tower-echo-brain/logs/*

# Clean up old logs
find /opt/tower-echo-brain/logs -name "*.log" -mtime +7 -delete

# Rotate large log files
sudo logrotate /etc/logrotate.d/tower-echo-brain

# Archive old conversation data
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
DELETE FROM echo_unified_interactions
WHERE created_at < NOW() - INTERVAL '90 days';"
```

---

## Performance Issues

### High CPU/Memory Usage

```bash
# Monitor system resources
htop
iotop
sudo iftop

# Check Echo Brain process
ps aux | grep echo.py
pmap -d $(pgrep -f echo.py)

# Profile memory usage
python3 -c "
import psutil
import os
process = psutil.Process($(pgrep -f echo.py))
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'CPU: {process.cpu_percent()}%')
"
```

### Slow Query Performance

```bash
# Enable PostgreSQL query logging
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_duration = 'on';
SELECT pg_reload_conf();
"

# Check slow queries
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY mean_exec_time DESC
LIMIT 10;"

# Analyze table statistics
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
ANALYZE;
SELECT tablename, n_tup_ins, n_tup_upd, n_tup_del, n_live_tup, n_dead_tup
FROM pg_stat_user_tables
WHERE tablename LIKE 'echo_%';"
```

### Model Inference Optimization

```bash
# Check Ollama model loading
curl http://localhost:11434/api/ps

# Optimize model caching
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "", "keep_alive": "24h"}'

# Monitor GPU memory
watch -n 1 nvidia-smi

# Check model quantization options
ollama list
```

---

## Security & Authentication Problems

### JWT Token Issues

```bash
# Decode JWT token (requires jq)
echo "YOUR_JWT_TOKEN" | cut -d. -f2 | base64 -d | jq .

# Test token validation
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://192.168.50.135:8088/api/auth/verify

# Check token expiration
python3 << 'EOF'
import jwt
import datetime

token = "YOUR_JWT_TOKEN"
decoded = jwt.decode(token, options={"verify_signature": False})
exp_time = datetime.datetime.fromtimestamp(decoded['exp'])
print(f"Token expires: {exp_time}")
print(f"Time until expiry: {exp_time - datetime.datetime.now()}")
EOF
```

### Permission Issues

```bash
# Check user permissions
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://192.168.50.135:8088/api/auth/user/permissions

# Test specific endpoint permissions
curl -H "Authorization: Bearer YOUR_TOKEN" \
  -X POST http://localhost:8309/api/echo/board/task \
  -H "Content-Type: application/json" \
  -d '{"task_description": "test", "priority": "low"}'
```

---

## Operational Runbooks

### Daily Operations Checklist

```bash
#!/bin/bash
# Daily Echo Brain Operations Checklist

echo "=== Daily Echo Brain Operations - $(date) ==="

# 1. Service Health Check
echo "1. Checking service health..."
bash /opt/tower-echo-brain/scripts/health-check.sh

# 2. Model Status Review
echo "2. Checking model status..."
curl -s http://localhost:11434/api/tags | jq -r '.models[] | "\(.name) - \(.size)"'

# 3. Board Activity Review
echo "3. Checking board activity..."
curl -s http://localhost:8309/api/echo/board/status | jq .

# 4. Resource Usage Check
echo "4. Checking resource usage..."
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"

# 5. Error Log Review
echo "5. Checking for errors..."
sudo journalctl -u tower-echo-brain --since "24 hours ago" --no-pager | grep -i error | wc -l

# 6. Performance Metrics
echo "6. Performance metrics..."
curl -s http://localhost:8309/api/echo/stats | jq -r '.statistics[] | "\(.model): \(.usage_count) queries"'

echo "=== Daily check complete ==="
```

### Weekly Maintenance Tasks

```bash
#!/bin/bash
# Weekly Echo Brain Maintenance

echo "=== Weekly Echo Brain Maintenance - $(date) ==="

# 1. Database Maintenance
echo "1. Database maintenance..."
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
VACUUM ANALYZE echo_unified_interactions;
VACUUM ANALYZE board_decisions;
VACUUM ANALYZE board_director_evaluations;
"

# 2. Log Rotation
echo "2. Log rotation..."
sudo logrotate -f /etc/logrotate.d/tower-echo-brain

# 3. Clean Old Data
echo "3. Cleaning old data..."
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
DELETE FROM echo_unified_interactions WHERE created_at < NOW() - INTERVAL '30 days';
DELETE FROM board_decisions WHERE created_at < NOW() - INTERVAL '60 days' AND status = 'completed';
"

# 4. Model Cleanup
echo "4. Model cleanup..."
# Remove unused models (implement based on usage statistics)
curl -s http://localhost:8309/api/echo/stats | jq -r '
.statistics[] | select(.usage_count == 0) | .model
' | while read model; do
    echo "Consider removing unused model: $model"
done

# 5. Security Audit
echo "5. Security audit..."
# Check for failed authentication attempts
sudo journalctl -u tower-echo-brain --since "7 days ago" --no-pager | grep -i "401\|403" | wc -l

# 6. Performance Report
echo "6. Performance report..."
psql -h 192.168.50.135 -U patrick -d tower_consolidated -c "
SELECT
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as queries,
    AVG(processing_time) as avg_time
FROM echo_unified_interactions
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date;
"

echo "=== Weekly maintenance complete ==="
```

### Model Update Procedure

```bash
#!/bin/bash
# Model Update Procedure

MODEL_NAME="$1"
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

echo "=== Model Update Procedure for $MODEL_NAME ==="

# 1. Check current model status
echo "1. Checking current model status..."
curl -s http://localhost:11434/api/tags | jq -r ".models[] | select(.name | contains(\"$MODEL_NAME\"))"

# 2. Submit board approval for large models
if [[ "$MODEL_NAME" =~ (70b|32b) ]]; then
    echo "2. Submitting board approval for large model..."
    TASK_ID=$(curl -s -X POST http://localhost:8309/api/echo/board/task \
        -H "Authorization: Bearer $JWT_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"task_description\": \"Update model $MODEL_NAME\",
            \"priority\": \"normal\",
            \"context\": {\"model_name\": \"$MODEL_NAME\", \"operation\": \"update\"}
        }" | jq -r '.task_id')

    echo "Board task submitted: $TASK_ID"
    echo "Waiting for board approval..."

    # Wait for approval (timeout after 5 minutes)
    for i in {1..30}; do
        STATUS=$(curl -s http://localhost:8309/api/echo/board/decisions/$TASK_ID | jq -r '.current_status')
        if [ "$STATUS" = "completed" ]; then
            echo "Board approval received"
            break
        fi
        echo "Waiting... ($i/30)"
        sleep 10
    done
fi

# 3. Perform model update
echo "3. Performing model update..."
curl -X POST http://localhost:8309/api/echo/models/manage \
    -H "Content-Type: application/json" \
    -d "{
        \"action\": \"update\",
        \"model\": \"$MODEL_NAME\",
        \"reason\": \"Scheduled maintenance update\"
    }"

# 4. Verify update
echo "4. Verifying update..."
sleep 30  # Wait for update to process
curl -s http://localhost:11434/api/tags | jq -r ".models[] | select(.name | contains(\"$MODEL_NAME\"))"

# 5. Test updated model
echo "5. Testing updated model..."
curl -X POST http://localhost:8309/api/echo/query \
    -H "Content-Type: application/json" \
    -d "{
        \"query\": \"Test query for model validation\",
        \"intelligence_level\": \"auto\"
    }" | jq -r '.model_used'

echo "=== Model update procedure complete ==="
```

---

## Emergency Procedures

### Service Recovery Procedure

```bash
#!/bin/bash
# Emergency Echo Brain Recovery

echo "=== EMERGENCY ECHO BRAIN RECOVERY ==="
echo "Timestamp: $(date)"

# 1. Stop all related services
echo "1. Stopping services..."
sudo systemctl stop tower-echo-brain
sudo systemctl stop ollama

# 2. Check for hanging processes
echo "2. Checking for hanging processes..."
pkill -f "echo.py"
pkill -f "ollama"

# 3. Clear temporary files
echo "3. Clearing temporary files..."
rm -f /opt/tower-echo-brain/logs/echo-daemon.log
rm -f /tmp/ollama-*

# 4. Check disk space
echo "4. Checking disk space..."
df -h
if [ $(df / | tail -1 | awk '{print $5}' | sed 's/%//') -gt 90 ]; then
    echo "WARNING: Disk space critical!"
    # Emergency cleanup
    find /opt/tower-echo-brain/logs -name "*.log" -mtime +1 -delete
fi

# 5. Start services in order
echo "5. Starting services..."
sudo systemctl start ollama
sleep 10

# Verify Ollama is responding
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama started successfully"
else
    echo "ERROR: Ollama failed to start"
    exit 1
fi

sudo systemctl start tower-echo-brain
sleep 15

# 6. Verify recovery
echo "6. Verifying recovery..."
if curl -s http://localhost:8309/api/echo/health >/dev/null 2>&1; then
    echo "✓ Echo Brain recovered successfully"

    # Send recovery notification
    curl -X POST http://localhost:8309/api/echo/voice/notify \
        -H "Content-Type: application/json" \
        -d '{"message": "Echo Brain emergency recovery completed", "priority": "normal"}'
else
    echo "✗ Echo Brain recovery failed"
    exit 1
fi

echo "=== Emergency recovery complete ==="
```

### Disaster Recovery

```bash
#!/bin/bash
# Disaster Recovery Procedure

echo "=== DISASTER RECOVERY PROCEDURE ==="

# 1. Assessment
echo "1. System assessment..."
systemctl status tower-echo-brain ollama postgresql redis

# 2. Database backup
echo "2. Creating emergency database backup..."
pg_dump -h 192.168.50.135 -U patrick tower_consolidated > \
    "/tmp/emergency_backup_$(date +%Y%m%d_%H%M%S).sql"

# 3. Configuration backup
echo "3. Backing up configuration..."
tar -czf "/tmp/echo_config_backup_$(date +%Y%m%d_%H%M%S).tar.gz" \
    /opt/tower-echo-brain/.env \
    /opt/tower-echo-brain/requirements.txt \
    /etc/systemd/system/tower-echo-brain.service

# 4. Model inventory
echo "4. Documenting model inventory..."
curl -s http://localhost:11434/api/tags > "/tmp/model_inventory_$(date +%Y%m%d_%H%M%S).json"

# 5. Reset to known good state
echo "5. Resetting to known good state..."
cd /opt/tower-echo-brain
git status
git stash
git checkout main
git pull origin main

# 6. Rebuild environment
echo "6. Rebuilding environment..."
source venv/bin/activate
pip install -r requirements.txt

# 7. Restart services
echo "7. Restarting services..."
sudo systemctl restart tower-echo-brain

echo "=== Disaster recovery complete ==="
echo "Backup files created in /tmp/"
ls -la /tmp/*backup* /tmp/*inventory*
```

---

## Monitoring & Alerting

### Prometheus Metrics Export

```bash
# Create metrics endpoint for Prometheus
cat > /opt/tower-echo-brain/metrics_exporter.py << 'EOF'
#!/usr/bin/env python3
"""Prometheus metrics exporter for Echo Brain"""

import time
import psycopg2
import requests
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# Metrics definitions
echo_requests_total = Counter('echo_requests_total', 'Total Echo Brain requests', ['method', 'endpoint'])
echo_processing_time = Histogram('echo_processing_time_seconds', 'Request processing time')
echo_model_usage = Gauge('echo_model_usage_total', 'Model usage count', ['model'])
echo_board_decisions = Counter('echo_board_decisions_total', 'Board decisions', ['status'])
echo_service_health = Gauge('echo_service_health', 'Service health status', ['service'])

def collect_metrics():
    """Collect metrics from Echo Brain"""
    try:
        # Database metrics
        conn = psycopg2.connect(
            host="192.168.50.135",
            database="tower_consolidated",
            user="patrick"
        )
        cursor = conn.cursor()

        # Model usage statistics
        cursor.execute("""
            SELECT model_used, COUNT(*)
            FROM echo_unified_interactions
            WHERE created_at > NOW() - INTERVAL '1 hour'
            GROUP BY model_used
        """)
        for model, count in cursor.fetchall():
            echo_model_usage.labels(model=model).set(count)

        # Board decisions
        cursor.execute("""
            SELECT status, COUNT(*)
            FROM board_decisions
            WHERE created_at > NOW() - INTERVAL '1 hour'
            GROUP BY status
        """)
        for status, count in cursor.fetchall():
            echo_board_decisions.labels(status=status).inc(count)

        conn.close()

        # Service health checks
        services = ['ollama', 'postgres', 'redis', 'auth']
        for service in services:
            health = check_service_health(service)
            echo_service_health.labels(service=service).set(1 if health else 0)

    except Exception as e:
        print(f"Metrics collection error: {e}")

def check_service_health(service):
    """Check health of individual services"""
    try:
        if service == 'ollama':
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            return response.status_code == 200
        elif service == 'postgres':
            conn = psycopg2.connect(
                host="192.168.50.135",
                database="tower_consolidated",
                user="patrick",
                connect_timeout=5
            )
            conn.close()
            return True
        # Add other service checks...
        return True
    except:
        return False

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(8310)

    while True:
        collect_metrics()
        time.sleep(60)  # Collect metrics every minute
EOF

# Create systemd service for metrics exporter
sudo tee /etc/systemd/system/echo-brain-metrics.service > /dev/null << 'EOF'
[Unit]
Description=Echo Brain Prometheus Metrics Exporter
After=network.target

[Service]
Type=simple
User=patrick
WorkingDirectory=/opt/tower-echo-brain
ExecStart=/opt/tower-echo-brain/venv/bin/python /opt/tower-echo-brain/metrics_exporter.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable echo-brain-metrics
sudo systemctl start echo-brain-metrics
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Echo Brain Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(echo_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Model Usage",
        "type": "pie",
        "targets": [
          {
            "expr": "echo_model_usage_total",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "echo_service_health",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, echo_processing_time_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# /etc/prometheus/rules/echo-brain.yml
groups:
  - name: echo-brain
    rules:
      - alert: EchoBrainDown
        expr: echo_service_health{service="echo-brain"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Echo Brain service is down"

      - alert: HighErrorRate
        expr: rate(echo_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: SlowProcessing
        expr: histogram_quantile(0.95, echo_processing_time_seconds_bucket) > 30
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow query processing detected"

      - alert: DiskSpaceHigh
        expr: (100 - (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100) > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Disk space usage high"
```

This comprehensive troubleshooting playbook provides systematic approaches to diagnosing and resolving Echo Brain issues, along with operational procedures for maintaining system health and performance.