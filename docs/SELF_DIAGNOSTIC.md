# Echo Brain Self-Diagnostic System

## Overview

The Echo Brain self-diagnostic system provides comprehensive health monitoring, automatic issue detection, and self-improvement capabilities. It continuously monitors system services, data quality, resource usage, and API health to ensure optimal performance.

## Components

### 1. Core Diagnostic Module
**Location:** `/opt/tower-echo-brain/src/core/self_diagnostic.py`

The main diagnostic engine that orchestrates all health checks and generates comprehensive reports.

#### Categories Monitored:
- **System Services** - All Tower services and critical ports
- **Knowledge Sources** - PostgreSQL facts, Qdrant vectors, conversations
- **Unified Layer** - Integration between all data sources
- **MCP Integration** - Model Context Protocol service methods
- **API Endpoints** - Critical endpoint availability and response times
- **Ingestion Pipeline** - Timer status and data freshness
- **Codebase Quality** - Code analysis, TODOs, and issues
- **Data Quality** - Duplicate detection, confidence levels, stale data
- **Usage Patterns** - Query patterns and system utilization
- **Resource Usage** - CPU, memory, disk, and process monitoring

### 2. Contract Monitor
**Location:** `/opt/tower-echo-brain/src/monitoring/contract_monitor.py`

Validates API contracts between frontend and backend components.

- Runs every 5 minutes via systemd timer
- Tests 15 critical API endpoints
- Tracks response times and data structures
- Stores results in PostgreSQL for trend analysis
- Detects breaking changes and API drift

### 3. Status Script
**Location:** `/opt/tower-echo-brain/scripts/status-verbose.sh`

Bash script for quick command-line status checks.

- Color-coded output for easy reading
- Real-time service and port status
- Database connectivity checks
- Ingestion pipeline monitoring
- MCP functionality testing

## Running Diagnostics

### Command Line

```bash
# Quick status check with verbose output
echo-brain-status

# Watch status continuously (updates every 5 seconds)
watch -n 5 -c echo-brain-status

# Run full Python diagnostic
cd /opt/tower-echo-brain
venv/bin/python -c "
import asyncio
from src.core.self_diagnostic import run_diagnostic
import json

async def test():
    result = await run_diagnostic()
    print(json.dumps(result, indent=2, default=str))

asyncio.run(test())
"
```

### API Endpoints

```bash
# Basic diagnostic test
curl http://localhost:8309/api/echo/diagnostics/test

# Contract monitor status
curl http://localhost:8309/api/echo/diagnostics/contracts

# Contract history
curl http://localhost:8309/api/echo/diagnostics/contracts/history

# Trigger manual contract test
curl -X POST http://localhost:8309/api/echo/diagnostics/contracts/run

# Get open issues only
curl http://localhost:8309/api/echo/diagnostics/contracts/issues

# Detailed health check
curl http://localhost:8309/api/echo/health/detailed
```

## Health Score Calculation

The system calculates an overall health score (0-100%) based on:

- **Critical Issues:** 0 points (immediate attention required)
- **Warning Issues:** 60 points (should be addressed soon)
- **Healthy Status:** 100 points (everything working properly)
- **Info Status:** 100 points (informational, not affecting health)

### Health Thresholds:
- **90-100%:** FULLY OPERATIONAL - All systems functioning optimally
- **70-89%:** DEGRADED - Some non-critical issues present
- **Below 70%:** CRITICAL - Immediate attention required

## Automatic Monitoring

### Systemd Timers

The system includes several automated monitoring tasks:

```bash
# Check active timers
systemctl list-timers | grep echo

# Key timers:
- echo-brain-ingest.timer      # Ingestion pipeline (every 6 hours)
- contract-monitor.timer        # API contract validation (every 5 minutes)
- echo-brain-maintenance.timer  # System maintenance (as configured)
```

### Worker Scheduler

Background workers run on various schedules:
- File watcher (10 min)
- Log monitor (15 min)
- Fact extraction (30 min)
- Reasoning worker (30 min)
- Conversation watcher (60 min)
- Self test runner (60 min)
- Improvement engine (2 hours)
- Knowledge graph builder (daily)

## Diagnostic Results Format

### JSON Output Structure

```json
{
  "timestamp": "2026-02-11T01:23:35.659049",
  "elapsed_seconds": 3.78,
  "health_score": 85.5,
  "summary": {
    "healthy": 10,
    "warning": 3,
    "critical": 1,
    "info": 2
  },
  "results": [
    {
      "category": "services",
      "name": "tower-echo-brain",
      "status": "healthy",
      "message": "Service active and port 8309 listening",
      "details": {
        "active": true,
        "port": 8309,
        "port_open": true
      },
      "recommendation": null
    }
  ],
  "recommendations": [
    "🔴 [critical] Fix PostgreSQL connection",
    "🟡 [warning] Run fact extraction on recent conversations"
  ],
  "quick_stats": {
    "facts_total": 2558,
    "vectors_total": 61932,
    "conversations_total": 171,
    "codebase_lines": 58350
  }
}
```

## Security Considerations

### Password Management

The diagnostic system reads database passwords securely:
1. First checks environment variable `DB_PASSWORD`
2. Falls back to reading from systemd service configuration
3. Never hardcodes passwords in source code

Configuration location: `/etc/systemd/system/tower-echo-brain.service.d/database.conf`

## Logging

All diagnostic results are logged with appropriate severity levels:
- **ERROR:** Critical issues requiring immediate attention
- **WARNING:** Non-critical issues that should be addressed
- **INFO:** Normal operation and statistics

Logs can be viewed with:
```bash
sudo journalctl -u tower-echo-brain -f
```

## Troubleshooting

### Common Issues and Solutions

1. **Port Detection False Negatives**
   - Script uses both `lsof` and `netstat`/`ss` for reliability
   - Some processes may not show in `lsof` but are actually running

2. **Database Connection Failed**
   - Ensure `DB_PASSWORD` environment variable is set
   - Check PostgreSQL is running: `systemctl status postgresql`
   - Verify user permissions in PostgreSQL

3. **MCP Empty Responses**
   - This is often a JSON parsing issue in the status script
   - The MCP service itself is likely working correctly
   - Use direct API calls to verify functionality

4. **High Resource Usage**
   - Check for runaway processes with the diagnostic tool
   - Consider restarting memory-intensive services
   - Clean up old logs in `/tmp/` (automated cleanup implemented)

## Development

### Adding New Diagnostic Checks

To add a new diagnostic check:

1. Add a new method in `/opt/tower-echo-brain/src/core/self_diagnostic.py`:
```python
async def _diagnose_new_feature(self):
    """Check new feature health"""
    # Perform checks
    self.results.append(DiagnosticResult(
        category="new_feature",
        name="Feature Name",
        status="healthy|warning|critical|info",
        message="Status message",
        details={"key": "value"},
        recommendation="Fix suggestion if needed"
    ))
```

2. Call it from `run_full_diagnostic()` method

3. Update documentation with new check details

### Testing

Run the test suite:
```bash
cd /opt/tower-echo-brain
venv/bin/python -m pytest tests/test_self_diagnostic.py -v
```

## Maintenance

### Regular Tasks

- **Weekly:** Review diagnostic trends for degradation patterns
- **Monthly:** Archive old diagnostic logs and contract monitor results
- **Quarterly:** Update diagnostic thresholds based on system growth

### Cleanup

Old test files and logs are automatically cleaned:
- Test files archived to `/opt/tower-echo-brain/tests/tmp_archive/`
- Large log files (>1GB) should be rotated or removed
- Temporary files in `/tmp/` are monitored and cleaned

## Integration with Other Systems

The self-diagnostic system integrates with:

- **Contract Monitor** - API health validation
- **Worker Scheduler** - Background task monitoring
- **MCP Service** - Claude integration health
- **Ingestion Pipeline** - Data freshness monitoring
- **Autonomous System** - Goal and task tracking

## Future Enhancements

Planned improvements for the diagnostic system:

1. **Predictive Analytics** - Detect issues before they occur
2. **Auto-Remediation** - Automatically fix common issues
3. **Grafana Dashboard** - Visual monitoring interface
4. **Alerting System** - Proactive notifications for critical issues
5. **Performance Baselines** - Track performance over time
6. **Distributed Tracing** - End-to-end request tracking