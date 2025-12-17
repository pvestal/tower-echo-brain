# Echo Brain Learning Pipeline Deployment Guide

## Quick Start (Production)

```bash
# 1. Fix database schema
sudo -u postgres psql -d echo_brain -f /opt/tower-echo-brain/services/learning_pipeline/scripts/fix_database_schema.sql

# 2. Install dependencies
cd /opt/tower-echo-brain/services/learning_pipeline
python3 -m pip install -r requirements.txt

# 3. Setup systemd timer (replaces broken cron)
chmod +x scripts/create_systemd_timer.sh
sudo ./scripts/create_systemd_timer.sh

# 4. Verify installation
/opt/tower-echo-brain/scripts/check_learning_pipeline_health.sh

# 5. Run manually to test
sudo systemctl start echo-learning-pipeline.service
```

## Detailed Deployment Steps

### Phase 1: Database Setup

1. **Connect to PostgreSQL as patrick user**:
```bash
psql -h localhost -U patrick -d echo_brain
```

2. **Execute schema fix**:
```sql
\i /opt/tower-echo-brain/services/learning_pipeline/scripts/fix_database_schema.sql
```

3. **Verify tables created**:
```sql
\dt learning_*
\dt pipeline_*
SELECT * FROM get_pipeline_stats();
```

### Phase 2: Environment Setup

1. **Set environment variables**:
```bash
export ECHO_BRAIN_DB_PASSWORD="***REMOVED***"
export PYTHONPATH="/opt/tower-echo-brain/services/learning_pipeline/src"
```

2. **Install Python dependencies**:
```bash
cd /opt/tower-echo-brain/services/learning_pipeline
python3 -m pip install -r requirements.txt
```

3. **Create log directory**:
```bash
sudo mkdir -p /opt/tower-echo-brain/logs
sudo chown patrick:patrick /opt/tower-echo-brain/logs
```

### Phase 3: Service Configuration

1. **Remove broken cron job**:
```bash
# Check current cron jobs
crontab -l | grep echo

# Remove broken entry
(crontab -l 2>/dev/null | grep -v "echo_learning_pipeline.py" || true) | crontab -
```

2. **Install systemd timer**:
```bash
cd /opt/tower-echo-brain/services/learning_pipeline
sudo chmod +x scripts/create_systemd_timer.sh
sudo ./scripts/create_systemd_timer.sh
```

3. **Verify timer installation**:
```bash
systemctl status echo-learning-pipeline.timer
systemctl list-timers echo-learning-pipeline.timer
```

### Phase 4: Initial Testing

1. **Dependency check**:
```bash
cd /opt/tower-echo-brain/services/learning_pipeline
python3 scripts/run_pipeline.py --check-deps
```

2. **Health check**:
```bash
python3 scripts/run_pipeline.py --health-check
```

3. **Dry run test**:
```bash
python3 scripts/run_pipeline.py --dry-run
```

4. **Manual execution**:
```bash
/opt/tower-echo-brain/scripts/run_learning_pipeline_manual.sh
```

### Phase 5: Monitoring Setup

1. **Check logs**:
```bash
tail -f /opt/tower-echo-brain/logs/learning_pipeline.log
```

2. **Monitor systemd journal**:
```bash
sudo journalctl -u echo-learning-pipeline.service -f
```

3. **Check database growth**:
```sql
SELECT * FROM get_pipeline_stats();
SELECT COUNT(*) FROM learning_conversations;
SELECT COUNT(*) FROM learning_items;
```

## Configuration

### Database Connection Fix

The main issue was services trying to connect to `***REMOVED***` instead of `localhost`. The production configuration now correctly uses:

```yaml
database:
  host: "localhost"  # Fixed: was ***REMOVED***
  port: 5432
  name: "echo_brain"
  user: "patrick"
```

### Cron Job Replacement

**Old broken cron job**:
```
0 */6 * * * cd /home/patrick/Tower && python3 echo_learning_pipeline.py >> /tmp/echo_learning.log 2>&1
```

**New systemd timer**:
- **Service**: `echo-learning-pipeline.service`
- **Timer**: `echo-learning-pipeline.timer`
- **Schedule**: Every 6 hours at :30 minutes (00:30, 06:30, 12:30, 18:30)
- **Randomization**: Up to 15 minutes random delay to avoid conflicts
- **Logging**: Centralized to `/opt/tower-echo-brain/logs/learning_pipeline.log`

### Vector Database Integration

**Qdrant Collections**:
- `claude_conversations`: Main conversation embeddings
- `echo_memories`: Echo Brain memory storage

**Current Status**: 1,780 stale vectors that will be updated by new pipeline

## Troubleshooting

### Common Issues

1. **Database connection timeout**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql
# Check if echo_brain database exists
psql -h localhost -U patrick -l | grep echo_brain
```

2. **Qdrant not accessible**:
```bash
# Check Qdrant service
curl http://localhost:6333/collections
# Start if needed
sudo systemctl start qdrant
```

3. **Permission issues**:
```bash
# Fix file permissions
sudo chown -R patrick:patrick /opt/tower-echo-brain/services/learning_pipeline
sudo chmod +x /opt/tower-echo-brain/services/learning_pipeline/scripts/*.py
```

4. **Missing dependencies**:
```bash
# Reinstall requirements
cd /opt/tower-echo-brain/services/learning_pipeline
python3 -m pip install -r requirements.txt --upgrade
```

### Debugging Commands

```bash
# Check timer status
systemctl status echo-learning-pipeline.timer

# Check service logs
sudo journalctl -u echo-learning-pipeline.service --since "1 hour ago"

# Manual execution with debug logging
python3 scripts/run_pipeline.py --log-level DEBUG --dry-run

# Check database stats
psql -h localhost -U patrick -d echo_brain -c "SELECT * FROM get_pipeline_stats();"

# Check vector database
curl http://localhost:6333/collections/claude_conversations
```

### Performance Monitoring

**Database queries for monitoring**:
```sql
-- Recent pipeline runs
SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 10;

-- Processing performance
SELECT * FROM pipeline_performance ORDER BY day DESC LIMIT 7;

-- Recent learning items
SELECT COUNT(*), item_type FROM recent_learning_items
GROUP BY item_type ORDER BY COUNT(*) DESC;

-- Failed conversations
SELECT * FROM learning_conversations
WHERE processing_status = 'failed'
ORDER BY processed_at DESC;
```

**System resource monitoring**:
```bash
# Check disk space for logs
df -h /opt/tower-echo-brain/logs

# Check memory usage during processing
ps aux | grep run_pipeline.py

# Monitor Qdrant memory usage
docker stats qdrant  # if running in Docker
```

## Maintenance

### Regular Tasks

1. **Weekly**: Check pipeline performance and error rates
2. **Monthly**: Review and clean old logs
3. **Quarterly**: Analyze learning item quality and update processing logic

### Cleanup Commands

```bash
# Clean old pipeline data (90 days)
psql -h localhost -U patrick -d echo_brain -c "SELECT cleanup_old_pipeline_data(90);"

# Rotate logs manually
sudo logrotate /etc/logrotate.d/echo-brain-pipeline

# Check and prune Docker containers (if using Qdrant via Docker)
docker system prune -f
```

## Success Metrics

### Performance Targets (From Architecture)

- **Processing Speed**: >1000 conversations/hour
- **Accuracy**: >95% successful processing rate
- **Availability**: >99.9% uptime
- **Latency**: <30 seconds end-to-end processing

### Monitoring Queries

```sql
-- Check if targets are being met
SELECT
    AVG(conversations_processed/EXTRACT(EPOCH FROM (completed_at - started_at)) * 3600) as conversations_per_hour,
    AVG(CASE WHEN status = 'completed' THEN 1.0 ELSE 0.0 END) as success_rate,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds
FROM pipeline_runs
WHERE started_at >= NOW() - INTERVAL '7 days';
```

## Integration with Existing Echo Brain

The learning pipeline integrates with:

1. **Echo Brain Database** (`echo_brain`): Stores processed conversations and learning items
2. **Qdrant Vector Database**: Stores embeddings for semantic search
3. **Echo Brain API** (Port 8309): Will consume learning data for enhanced responses
4. **Tower Dashboard**: Can display pipeline metrics and status

The pipeline runs independently but feeds data into the broader Echo Brain ecosystem for enhanced AI capabilities.