# Echo Brain Repair Plan

## Current State (Reality)
- **225 total endpoints discovered** in codebase
- **57 working (25.3%)** - Actually respond with 200
- **144 not mounted (64%)** - Code exists but not wired up
- **8 server errors (3.6%)** - Database/dependency failures
- **3 connection timeouts (1.3%)** - Infinite loops or deadlocks

## Problem Analysis

### 1. Marketing Bullshit in Naming
Files with promotional terms that need renaming:
- `ENHANCED_COMPREHENSIVE_TEST.py` → `test_endpoints.py`
- `ENHANCED_TEST_RESULTS.json` → `test_results.json`
- `unified_context_manager.py` → `context_manager.py`
- `enhanced_memory.py` → `memory.py`
- `comprehensive_indexer.py` → `indexer.py`
- "Unified API" → "API"
- "Enhanced" anything → Remove adjective
- "Comprehensive" → Remove
- "Consolidated" → Remove

### 2. Unmounted Routers (144 endpoints = 404)
Routers defined but never included in main.py:
- `/api/delegation/*` - delegation_routes.py exists, not mounted
- `/api/notifications/*` - notifications_api.py exists, not mounted
- `/api/db/*` - db_metrics.py exists, not mounted
- `/api/git/*` - git_operations.py exists, not mounted
- `/api/vault/*` - vault.py exists, not mounted
- `/api/knowledge/*` - knowledge.py exists, not mounted
- `/api/tasks/*` - tasks.py exists, not mounted
- `/api/resilience/*` - resilience_status.py exists, not mounted
- `/api/calendar/*` - google_calendar_api.py exists, not mounted
- `/api/preferences/*` - preferences.py exists, not mounted
- `/api/repair/*` - repair_api.py exists, not mounted
- `/settings` - echo_settings_api.py exists, not mounted
- `/integrations/*` - integrations.py exists, not mounted
- And 100+ more...

### 3. Database Errors (8 endpoints = 500)
Failing with actual errors:
- `/api/echo/code` - Missing model/schema
- `/api/context` - 'ResilientQdrantMemory' has no attribute 'search_similar'
- `/api/echo/codebase/search` - column "signature" does not exist
- `/api/echo/codebase/file/*` - Database query error
- `/api/autonomous/kill` - Missing implementation
- `/api/autonomous/audit` - Table doesn't exist
- `/git/commit` - Git operation failure
- `/api/agents/status` - Agent initialization error

### 4. Timeout Issues (3 endpoints)
- `/api/chat/simple` - Timeout after 10 seconds
- `/api/echo/agents/reasoning` - Timeout (model not loaded?)
- `/test/echo-brain` - Timeout

## Repair Actions

### Phase 1: Remove Marketing Bullshit (Immediate)
```bash
# Rename files
mv ENHANCED_COMPREHENSIVE_TEST.py test_endpoints.py
mv ENHANCED_TEST_RESULTS.json test_results.json
mv COMPREHENSIVE_TEST.py old_test.py

# Update imports and references
grep -r "ENHANCED\|COMPREHENSIVE\|UNIFIED\|CONSOLIDATED" --include="*.py" | grep -v ".git"
```

### Phase 2: Mount Missing Routers (1 hour)
Edit src/main.py to include all routers:
```python
# Add all missing routers
from .api import (
    delegation_routes,
    notifications_api,
    db_metrics,
    git_operations,
    vault,
    knowledge,
    tasks,
    resilience_status,
    google_calendar_api,
    preferences,
    repair_api,
    echo_settings_api,
    integrations,
    # ... etc
)

app.include_router(delegation_routes.router, prefix="/api/delegation")
app.include_router(notifications_api.router, prefix="/api/notifications")
# ... etc for all 144 missing endpoints
```

### Phase 3: Fix Database Errors (2 hours)
1. Fix Qdrant memory search:
   - Update `ResilientQdrantMemory` to have `search_similar` method
   - Or change calls to use correct method name

2. Fix codebase search:
   - Add missing "signature" column to database
   - Or remove from SQL query

3. Fix autonomous tables:
   - Create missing audit_log table
   - Add proper schemas

### Phase 4: Create Health Monitor (30 mins)
```python
# health_monitor.py
import asyncio
from datetime import datetime

class HealthMonitor:
    def __init__(self):
        self.endpoint_status = {}

    async def check_endpoint(self, method, path):
        # Test endpoint and record status
        pass

    async def continuous_monitor(self):
        while True:
            for endpoint in ALL_ENDPOINTS:
                status = await self.check_endpoint(endpoint)
                self.endpoint_status[endpoint] = {
                    'status': status,
                    'timestamp': datetime.now(),
                    'response_time': response_time
                }
            await asyncio.sleep(300)  # Check every 5 mins

    def get_health_report(self):
        working = sum(1 for e in self.endpoint_status.values() if e['status'] == 200)
        total = len(self.endpoint_status)
        return {
            'total': total,
            'working': working,
            'percentage': (working/total)*100,
            'failing': [k for k,v in self.endpoint_status.items() if v['status'] >= 500]
        }
```

### Phase 5: Automated Testing (30 mins)
```python
# test_runner.py
@app.on_event("startup")
async def startup_tests():
    """Run tests on startup and log results"""
    results = await test_all_endpoints()
    if results['percentage'] < 80:
        logger.error(f"Only {results['percentage']}% endpoints working!")

# Add to crontab
*/30 * * * * cd /opt/tower-echo-brain && python3 test_endpoints.py >> /var/log/echo-brain-health.log 2>&1
```

### Phase 6: Reality Documentation
Create `ACTUAL_STATUS.md`:
```markdown
# Echo Brain Actual Status

## What Works
- Basic chat endpoints (57 endpoints)
- Model management
- Some agent operations

## What Doesn't Work
- 64% of endpoints not connected
- Database queries failing
- Git integration broken
- Most "advanced" features non-functional

## In Development
- [List actual work in progress]

## Not Implemented
- [List claimed features that don't exist]
```

## Success Metrics
- [ ] All marketing terms removed from codebase
- [ ] 90%+ endpoints returning non-404 responses
- [ ] Zero 500 errors
- [ ] Automated health monitoring running
- [ ] Reality documented, not aspirations

## Timeline
- Hour 1: Remove marketing terms, mount routers
- Hour 2: Fix database errors
- Hour 3: Setup monitoring and automation
- Ongoing: Maintain reality-based documentation