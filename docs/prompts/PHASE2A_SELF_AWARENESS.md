# Phase 2a: Self-Awareness & Self-Improvement for Echo Brain

## Context

Phase 1 (Autonomous Core) is complete — safety enforcement, approval workflow, notifications, persistence, kill switch. Phase 2 workers (fact_extraction, conversation_watcher, knowledge_graph) are running on timers. But there is NO feedback loop. Workers produce output that nobody validates. Errors go to logs that nobody reads. The intelligence layer sometimes returns wrong answers (DDR5 vs DDR6) and nothing detects it.

This phase gives Echo Brain the full loop:
1. **INGEST** — Index its own code, schemas, and the anime pipeline
2. **THINK** — Detect errors, run self-tests, reason about root causes
3. **IMPROVE** — Propose fixes (gated at REVIEW, never auto-applied) for Patrick to approve/reject

Echo Brain should be able to answer: "What's broken in your own code right now?" and "How would you fix it?" — not just "here's a search result."

## Current System State

- Service: FastAPI on port 8309, runs as `echo` user
- Database: PostgreSQL `echo_brain`, tables owned by `patrick`, grants for `echo`
- Vectors: Qdrant at localhost:6333, collection `echo_memory` (1024D, mxbai-embed-large)
- LLM: Ollama at localhost:11434 (gemma2:9b for extraction, mistral:7b for reasoning)
- Workers: 3 running (fact_extraction/30min, conversation_watcher/10min, knowledge_graph/daily)
- Codebase: /opt/tower-echo-brain/src/ (Python, FastAPI)
- Venv: /opt/tower-echo-brain/venv/bin/python
- Service file: /etc/systemd/system/tower-echo-brain.service (runs as echo user)
- DATABASE_URL: postgresql://patrick:WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr@localhost/echo_brain

## Safety Gates for Phase 2a

Three tiers of operation: READ (auto), PROPOSE (review), EXECUTE (forbidden for code changes).

| Action | Safety Level | Reason |
|--------|-------------|--------|
| Index own codebase into Qdrant | AUTO | Read-only on filesystem, write to own vector store |
| Index anime pipeline code into Qdrant | AUTO | Read-only, secondary priority |
| Index own DB schemas | AUTO | Read-only metadata queries |
| Read own journalctl logs | AUTO | Read-only system logs |
| Store detected issues in DB | AUTO | Writing to own diagnostic tables |
| Run self-test queries | AUTO | Read-only queries against own API |
| Analyze issue root cause with LLM | AUTO | Reasoning over already-indexed code, no side effects |
| Generate improvement proposal | AUTO | Just text generation, stored in proposals table |
| **Apply improvement proposal** | **REVIEW** | **Patrick must approve before ANY code change** |
| Delete stale/conflicting vectors | REVIEW | Destructive, needs approval |
| Modify own source code | REVIEW | Only through approved proposals, never directly |
| Modify system services | FORBIDDEN | Never |
| Execute arbitrary shell commands | FORBIDDEN | Never |

---

## GATE 1: Verify Prerequisites Before Proceeding

Before building anything, confirm the system is healthy:

```bash
cd /opt/tower-echo-brain

echo "=== GATE 1: Prerequisite Check ==="

# 1. Service running?
curl -s http://localhost:8309/health | python3 -c "
import json, sys
data = json.load(sys.stdin)
status = data.get('status', 'unknown')
print(f'Service: {\"✅\" if status == \"healthy\" else \"❌\"} {status}')
"

# 2. Workers running?
curl -s http://localhost:8309/api/workers/status | python3 -c "
import json, sys
data = json.load(sys.stdin)
running = data.get('running', False)
count = len(data.get('workers', {}))
print(f'Workers: {\"✅\" if running else \"❌\"} {count} workers, running={running}')
"

# 3. Qdrant accessible?
curl -s http://localhost:6333/collections/echo_memory | python3 -c "
import json, sys
data = json.load(sys.stdin)
count = data.get('result', {}).get('points_count', 0)
print(f'Qdrant: ✅ {count} vectors in echo_memory')
"

# 4. Ollama accessible?
curl -s http://localhost:11434/api/tags | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
print(f'Ollama: ✅ Models: {\", \".join(models[:5])}')
"

# 5. Database accessible?
PGPASSWORD=WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr psql -h localhost -U patrick -d echo_brain -c "
SELECT COUNT(*) as facts FROM facts;
" 2>/dev/null | head -3

# 6. Codebase exists?
echo "Python files in src/: $(find /opt/tower-echo-brain/src -name '*.py' | wc -l)"
echo "Total lines: $(find /opt/tower-echo-brain/src -name '*.py' -exec cat {} + | wc -l)"

echo ""
echo "=== All checks must pass before proceeding ==="
```

**STOP if any check fails. Fix it first.**

---

## Task 1: Create Self-Awareness Database Schema

Create the tables Echo Brain needs to know about itself.

```sql
-- Run as postgres or patrick user
-- These tables let Echo Brain track its own health, issues, and knowledge of itself

-- Indexed source code files (Echo Brain's knowledge of its own codebase)
CREATE TABLE IF NOT EXISTS self_codebase_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL UNIQUE,
    file_hash VARCHAR(64) NOT NULL,          -- SHA256 of file contents (detect changes)
    language VARCHAR(20) DEFAULT 'python',
    line_count INTEGER,
    functions JSONB DEFAULT '[]',            -- [{name, line_start, line_end, docstring}]
    classes JSONB DEFAULT '[]',              -- [{name, line_start, line_end, methods, docstring}]
    imports JSONB DEFAULT '[]',              -- [module_names]
    qdrant_point_ids JSONB DEFAULT '[]',     -- References to vector chunks
    last_indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_codebase_file_path ON self_codebase_index(file_path);
CREATE INDEX IF NOT EXISTS idx_codebase_hash ON self_codebase_index(file_hash);

-- Indexed database schemas (Echo Brain's knowledge of its own data model)
CREATE TABLE IF NOT EXISTS self_schema_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name TEXT NOT NULL,
    schema_name TEXT DEFAULT 'public',
    column_info JSONB NOT NULL,              -- [{name, type, nullable, default}]
    row_count BIGINT,
    index_info JSONB DEFAULT '[]',           -- [{name, columns, unique}]
    foreign_keys JSONB DEFAULT '[]',         -- [{column, references_table, references_column}]
    qdrant_point_id TEXT,
    last_indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(schema_name, table_name)
);

-- Detected issues (Echo Brain's error awareness)
CREATE TABLE IF NOT EXISTS self_detected_issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_type VARCHAR(50) NOT NULL,         -- 'import_error', 'worker_failure', 'query_regression', 'stale_data', 'permission_error'
    severity VARCHAR(20) NOT NULL,           -- 'critical', 'warning', 'info'
    source VARCHAR(100),                     -- Which component detected it (e.g. 'log_monitor', 'self_test', 'health_check')
    title TEXT NOT NULL,
    description TEXT,
    related_file TEXT,                        -- File path if relevant
    related_worker TEXT,                      -- Worker name if relevant
    stack_trace TEXT,                         -- If from an error log
    suggested_fix TEXT,                       -- LLM-generated suggestion (REVIEW gated)
    status VARCHAR(20) DEFAULT 'open',       -- 'open', 'acknowledged', 'resolved', 'wont_fix'
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_issues_status ON self_detected_issues(status);
CREATE INDEX IF NOT EXISTS idx_issues_type ON self_detected_issues(issue_type);
CREATE INDEX IF NOT EXISTS idx_issues_severity ON self_detected_issues(severity);

-- Self-test results (Echo Brain's quality validation)
CREATE TABLE IF NOT EXISTS self_test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_name VARCHAR(100) NOT NULL,
    test_query TEXT,                          -- The question asked
    expected_contains TEXT[],                 -- Strings the answer MUST contain
    expected_not_contains TEXT[],             -- Strings the answer must NOT contain
    actual_response TEXT,
    passed BOOLEAN NOT NULL,
    failure_reason TEXT,
    response_time_ms INTEGER,
    run_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tests_name ON self_test_results(test_name);
CREATE INDEX IF NOT EXISTS idx_tests_passed ON self_test_results(passed);

-- Health metrics over time (Echo Brain's performance tracking)
CREATE TABLE IF NOT EXISTS self_health_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,       -- 'worker_fact_extraction_success_rate', 'query_accuracy', 'avg_response_ms'
    metric_value FLOAT NOT NULL,
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON self_health_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_time ON self_health_metrics(recorded_at);

-- Improvement proposals (Echo Brain's self-improvement queue — REVIEW gated)
CREATE TABLE IF NOT EXISTS self_improvement_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id UUID REFERENCES self_detected_issues(id),  -- What issue this fixes
    title TEXT NOT NULL,                      -- Short description of the change
    description TEXT NOT NULL,                -- Detailed explanation of what and why
    target_file TEXT NOT NULL,                -- Which file to modify
    current_code TEXT,                        -- The code that needs changing
    proposed_code TEXT NOT NULL,              -- The replacement code
    reasoning TEXT NOT NULL,                  -- LLM's explanation of why this fix works
    risk_assessment VARCHAR(20) DEFAULT 'low', -- 'low', 'medium', 'high'
    status VARCHAR(20) DEFAULT 'pending',    -- 'pending', 'approved', 'rejected', 'applied', 'rolled_back'
    reviewed_by VARCHAR(50),                  -- 'patrick' when manually reviewed
    reviewed_at TIMESTAMP WITH TIME ZONE,
    applied_at TIMESTAMP WITH TIME ZONE,
    git_commit_hash VARCHAR(40),              -- If applied, the commit hash
    rollback_commit_hash VARCHAR(40),         -- If rolled back
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_proposals_status ON self_improvement_proposals(status);
CREATE INDEX IF NOT EXISTS idx_proposals_issue ON self_improvement_proposals(issue_id);

-- Grant permissions to both patrick and echo users
GRANT SELECT, INSERT, UPDATE, DELETE ON self_codebase_index TO patrick;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_schema_index TO patrick;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_detected_issues TO patrick;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_test_results TO patrick;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_health_metrics TO patrick;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_improvement_proposals TO patrick;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_codebase_index TO echo;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_schema_index TO echo;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_detected_issues TO echo;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_test_results TO echo;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_health_metrics TO echo;
GRANT SELECT, INSERT, UPDATE, DELETE ON self_improvement_proposals TO echo;
GRANT USAGE, SELECT ON SEQUENCE self_health_metrics_id_seq TO patrick;
GRANT USAGE, SELECT ON SEQUENCE self_health_metrics_id_seq TO echo;
```

Run this SQL, then verify:

```bash
sudo -u postgres psql echo_brain -c "\dt self_*"
```

**Expected: 6 tables listed. STOP if any fail to create.**

---

## GATE 2: Schema Created Successfully

```bash
echo "=== GATE 2: Schema Verification ==="
PGPASSWORD=WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr psql -h localhost -U patrick -d echo_brain -c "
SELECT table_name, 
       (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as columns
FROM information_schema.tables t
WHERE table_name LIKE 'self_%'
ORDER BY table_name;
"
echo "Expected: 6 tables (self_codebase_index, self_schema_index, self_detected_issues, self_test_results, self_health_metrics, self_improvement_proposals)"
```

**All 5 tables must exist before proceeding.**

---

## Task 2: Build the Codebase Indexer Worker

Create `/opt/tower-echo-brain/src/autonomous/workers/codebase_indexer.py`

This worker scans Echo Brain's own source code, parses it for structure (functions, classes, imports), chunks it, embeds it into Qdrant, and stores metadata in PostgreSQL. It detects file changes via SHA256 hash comparison so it only re-indexes modified files.

Requirements:
1. **Tier 1 (primary): Index Echo Brain's own code**
   - Scan `/opt/tower-echo-brain/src/` recursively for `.py` files
   - For each file:
     - Compute SHA256 hash
     - Compare to stored hash in `self_codebase_index` — skip if unchanged
     - Parse with Python's `ast` module to extract:
       - Function definitions (name, line range, docstring, args)
       - Class definitions (name, line range, methods, docstring)
       - Import statements
     - Chunk the file content into ~500-token segments with overlap
     - Embed each chunk using mxbai-embed-large via Ollama (POST http://localhost:11434/api/embed)
     - Store vectors in Qdrant `echo_memory` collection with metadata:
       - `source`: "self_codebase"
       - `file_path`: relative path from project root
       - `chunk_type`: "code"
       - `functions`: list of function names in this chunk
       - `classes`: list of class names in this chunk
     - Store/update metadata in `self_codebase_index` table
   - Also index key non-Python files:
     - `systemd` service file: `/etc/systemd/system/tower-echo-brain.service`
     - Any `.env` or config files in project root
     - `requirements.txt` or `pyproject.toml`

2. **Tier 1b: Index Tower architecture files (shallow)**
   - Scan for `docker-compose*.yml`, `Makefile`, and systemd service files:
     ```python
     ARCHITECTURE_PATHS = [
         "/opt/tower-echo-brain/docker-compose*.yml",
         "/etc/systemd/system/tower-*.service",
     ]
     ```
   - Embed these as-is (no AST parsing), with `source: "tower_architecture"`
   - This lets Echo Brain answer "What Tower services exist?" and "What ports are used?"

3. **Tier 2 (secondary): Index anime production pipeline**
   - Discover the anime codebase location. Check these paths in order:
     ```python
     ANIME_PATHS = [
         "/opt/tower-anime/",
         "/opt/tower-echo-brain/src/autonomous/anime_production_orchestrator.py",
         "/opt/tower-echo-brain/src/autonomous/anime_task_executors.py",
     ]
     ```
   - For any `.py` files found, index with the same AST parsing as Tier 1
   - Store with `source: "anime_pipeline"` metadata to distinguish from self code
   - If anime paths don't exist, log a warning and skip (don't fail the cycle)

4. Log summary: "Indexed X Echo Brain files, Y anime files, Z tower arch files, N new/changed, M chunks embedded"

Safety:
- READ-ONLY on filesystem (never modify source files)
- WRITE to Qdrant and PostgreSQL only
- Safety level: AUTO

Worker class pattern (must match existing worker interface):

```python
class CodebaseIndexer:
    """Indexes Echo Brain's own source code and related Tower systems for self-awareness"""
    
    def __init__(self):
        # Tier 1: Own code (deep index with AST)
        self.own_src_root = "/opt/tower-echo-brain/src"
        
        # Tier 1b: Tower architecture (shallow index)
        self.tower_arch_patterns = [
            "/opt/tower-echo-brain/docker-compose*.yml",
            "/etc/systemd/system/tower-*.service",
        ]
        
        # Tier 2: Anime pipeline (deep index with AST, secondary priority)
        self.anime_paths = [
            "/opt/tower-anime/",
            "/opt/tower-echo-brain/src/autonomous/anime_production_orchestrator.py",
            "/opt/tower-echo-brain/src/autonomous/anime_task_executors.py",
        ]
        
        self.db_url = os.environ.get("DATABASE_URL", 
            "postgresql://patrick:WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr@localhost/echo_brain")
        self.qdrant_url = "http://localhost:6333"
        self.ollama_url = "http://localhost:11434"
        self.collection = "echo_memory"
    
    async def run_cycle(self):
        """Main worker cycle — called by scheduler"""
        # 1. Tier 1: Scan own src/ for .py files
        # 2. Hash each file, compare to stored hash — skip unchanged
        # 3. Parse changed files with ast module
        # 4. Chunk and embed changed files
        # 5. Store in Qdrant (source="self_codebase") + PostgreSQL
        # 6. Tier 1b: Scan tower architecture files
        # 7. Embed as-is (source="tower_architecture")
        # 8. Tier 2: Scan anime paths if they exist
        # 9. Parse and embed (source="anime_pipeline")
        # 10. Log summary with counts per tier
```

---

## Task 3: Build the Schema Indexer Worker

Create `/opt/tower-echo-brain/src/autonomous/workers/schema_indexer.py`

This worker queries PostgreSQL's `information_schema` to discover and index Echo Brain's own database structure. It creates natural language descriptions of each table and embeds them so Echo Brain can answer questions like "What tables store facts?" or "How is the autonomous system structured?"

Requirements:
1. Query `information_schema.tables` for all tables in the `public` schema
2. For each table:
   - Get column names, types, nullable, defaults
   - Get indexes
   - Get foreign key relationships
   - Get approximate row count (`pg_stat_user_tables.n_live_tup`)
   - Generate a natural language description using Ollama:
     ```
     Table 'facts' stores extracted knowledge with 6,389 rows.
     Columns: id (UUID, PK), subject (TEXT), predicate (TEXT), object (TEXT), ...
     Indexes: idx_facts_domain, idx_facts_subject
     Related to: documents via source_document_id
     ```
   - Embed this description and store in Qdrant with `source: "self_schema"`
   - Store structured metadata in `self_schema_index`
3. Detect schema changes by comparing column_info JSONB
4. Log: "Indexed X tables, Y new/changed"

Safety level: AUTO (read-only on database metadata, write to own tables)

---

## Task 4: Build the Log Monitor Worker

Create `/opt/tower-echo-brain/src/autonomous/workers/log_monitor.py`

This worker reads recent journalctl logs for the `tower-echo-brain` service, extracts errors and warnings, classifies them, and stores them as detected issues.

Requirements:
1. Read last N minutes of logs: `journalctl -u tower-echo-brain --since "X minutes ago" --no-pager`
   - Note: The service runs as `echo` user. The worker runs inside the service, so it should use Python's `subprocess` to call journalctl, OR read from a log file if journalctl isn't accessible from the echo user.
   - Alternative: If journalctl requires root, configure the service to also log to a file:
     Add to worker: check both journalctl and `/opt/tower-echo-brain/logs/echo_brain.log`
2. Parse log lines for:
   - ERROR level messages → severity: "critical"
   - WARNING level messages → severity: "warning"  
   - Exception/Traceback blocks → severity: "critical", capture full stack trace
   - Specific patterns:
     - "permission denied" → issue_type: "permission_error"
     - "No module named" → issue_type: "import_error"
     - "Method Not Allowed" → issue_type: "routing_error"
     - Worker failure messages → issue_type: "worker_failure"
     - "DDR5" in any response → issue_type: "stale_data" (known bad fact)
3. Deduplicate: Don't create duplicate issues for the same error within 1 hour
4. For each new issue:
   - Store in `self_detected_issues` table
   - If severity is "critical", also create an autonomous notification (use existing notification system)
5. Optionally: For issues with `related_file`, search `self_codebase_index` to find the relevant code and add it to the description
6. Log: "Scanned X log lines, found Y new issues (Z critical)"

Safety level: AUTO (read logs, write to own diagnostic tables)

**Important implementation note for log access:**
The echo user may not have journalctl access. Implement a fallback:

```python
async def _read_logs(self, minutes: int = 30) -> list[str]:
    """Read recent logs, trying journalctl first, then log file"""
    try:
        # Try journalctl first
        proc = await asyncio.create_subprocess_exec(
            'journalctl', '-u', 'tower-echo-brain', 
            '--since', f'{minutes} minutes ago', '--no-pager',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            return stdout.decode().splitlines()
    except Exception:
        pass
    
    # Fallback: read log file
    log_path = "/opt/tower-echo-brain/logs/echo_brain.log"
    if os.path.exists(log_path):
        # Read last N lines
        ...
    
    return []
```

---

## Task 5: Build the Self-Test Runner Worker

Create `/opt/tower-echo-brain/src/autonomous/workers/self_test_runner.py`

This worker runs a battery of known-good queries against Echo Brain's own API and validates the responses. It's the quality control line.

Requirements:
1. Define test cases as data (not hardcoded logic):

```python
SELF_TESTS = [
    {
        "name": "hardware_ram",
        "query": "How much RAM does Tower have?",
        "expected_contains": ["DDR6", "96GB"],
        "expected_not_contains": ["DDR5"],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "hardware_gpu_nvidia",
        "query": "What NVIDIA GPU does Tower have?",
        "expected_contains": ["RTX 3060", "12GB"],
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "hardware_gpu_amd",
        "query": "What AMD GPU does Tower have?",
        "expected_contains": ["RX 9070 XT", "16GB"],
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "service_health",
        "query": None,  # Just GET the endpoint
        "expected_contains": ["healthy"],
        "expected_not_contains": [],
        "endpoint": "/health",
        "method": "GET",
        "timeout_ms": 5000,
    },
    {
        "name": "worker_status",
        "query": None,
        "expected_contains": ["true"],  # running: true
        "expected_not_contains": [],
        "endpoint": "/api/workers/status",
        "method": "GET",
        "timeout_ms": 5000,
    },
    {
        "name": "vector_search_functional",
        "query": "What is Echo Brain?",
        "expected_contains": ["personal", "AI"],
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
]
```

2. For each test:
   - Send request to Echo Brain's own API (localhost:8309)
   - Measure response time
   - Check expected_contains (case-insensitive)
   - Check expected_not_contains (case-insensitive)
   - Store result in `self_test_results` table
   - If a test FAILS that previously PASSED, create a `self_detected_issues` entry with type "query_regression"

3. After all tests:
   - Calculate pass rate
   - Store as metric: `self_test_pass_rate` in `self_health_metrics`
   - Store individual response times as metrics
   - If pass rate < 80%, create a critical notification

4. Make tests extensible — load additional tests from a JSON/YAML file if it exists at `/opt/tower-echo-brain/config/self_tests.json`

Safety level: AUTO (read-only queries against own API)

---

## Task 6: Build the Health Dashboard Endpoint

Add to the Echo Brain API a detailed health endpoint that aggregates all self-awareness data.

Add to `/opt/tower-echo-brain/src/api/endpoints/` (or directly in main.py BEFORE the frontend mount):

**Endpoint: GET /api/echo/health/detailed**

Response format:
```json
{
    "status": "healthy|degraded|unhealthy",
    "timestamp": "2026-02-06T...",
    "workers": {
        "fact_extraction": {"running": true, "last_run": "...", "error_count": 0},
        "conversation_watcher": {"running": true, "last_run": "...", "error_count": 0},
        "knowledge_graph": {"running": true, "last_run": "...", "error_count": 0},
        "codebase_indexer": {"running": true, "last_run": "...", "error_count": 0},
        "schema_indexer": {"running": true, "last_run": "...", "error_count": 0},
        "log_monitor": {"running": true, "last_run": "...", "error_count": 0},
        "self_test_runner": {"running": true, "last_run": "...", "error_count": 0}
    },
    "knowledge": {
        "total_vectors": 2473,
        "total_facts": 6389,
        "extraction_coverage_pct": 45.2,
        "graph_edges": 0,
        "codebase_files_indexed": 0,
        "schema_tables_indexed": 0
    },
    "quality": {
        "last_test_pass_rate": 0.83,
        "tests_run": 6,
        "tests_passed": 5,
        "last_test_run": "2026-02-06T...",
        "regressions_detected": 1
    },
    "issues": {
        "open_critical": 2,
        "open_warning": 5,
        "open_info": 3,
        "resolved_last_24h": 1,
        "recent": [
            {"title": "DDR5 detected in response", "severity": "warning", "created_at": "..."},
            {"title": "knowledge_graph SQL syntax error", "severity": "critical", "created_at": "..."}
        ]
    },
    "improvement": {
        "pending_proposals": 2,
        "approved_proposals": 0,
        "applied_proposals": 0,
        "rejected_proposals": 1,
        "recent_proposals": [
            {"title": "Fix stale DDR5 reference", "risk": "low", "status": "pending", "created_at": "..."}
        ]
    },
    "self_awareness": {
        "knows_own_code": false,
        "knows_own_schema": false,
        "monitors_own_logs": false,
        "validates_own_output": false,
        "can_propose_improvements": false
    }
}
```

Status determination:
- "healthy": all workers running, test pass rate >= 90%, 0 open critical issues
- "degraded": some workers failing OR pass rate 60-90% OR 1-2 critical issues
- "unhealthy": workers down OR pass rate < 60% OR 3+ critical issues

**This endpoint MUST be defined BEFORE the frontend mount in main.py (same pattern as /api/workers/status).**

---

## Task 7: Build the Improvement Engine Worker

Create `/opt/tower-echo-brain/src/autonomous/workers/improvement_engine.py`

This is the IMPROVE stage. It reads open issues from `self_detected_issues`, searches the indexed codebase for relevant code, uses the LLM to reason about root causes, and generates fix proposals stored in `self_improvement_proposals`. **It NEVER applies changes directly.**

Think of it this way: the log_monitor and self_test_runner are the "check engine light." The improvement_engine is the "mechanic writing up a repair estimate." Patrick is the one who signs off on the work.

Requirements:

1. Every 2 hours, query `self_detected_issues` for issues with status='open' and severity IN ('critical', 'warning')
2. For each open issue (up to 5 per cycle to avoid LLM overload):
   - Check if a proposal already exists for this issue (skip if so)
   - If `related_file` is set, read that file from `self_codebase_index` to get function/class info, and search Qdrant for the actual code chunks (filter: `source="self_codebase"`, `file_path` matches)
   - If `related_file` is NOT set, search Qdrant with the issue title + description as the query to find potentially related code
   - If `stack_trace` is available, extract the file path and line number from the traceback and search for that specific code
   - Build an LLM prompt like this:

```python
ANALYSIS_PROMPT = """You are Echo Brain's self-improvement system.

ISSUE:
Type: {issue.issue_type}
Severity: {issue.severity}
Title: {issue.title}
Description: {issue.description}
Stack trace: {issue.stack_trace or 'None'}

RELATED CODE:
File: {related_file_path}
```
{code_chunks}
```

TASK:
1. What is the root cause of this issue?
2. What specific code change would fix it?
3. Show the EXACT current code that needs changing and the EXACT replacement code.
4. What is the risk of this change? (low/medium/high)
5. Could this change break anything else?

Respond in this exact format:
ROOT_CAUSE: <one paragraph explanation>
RISK: <low|medium|high>
CURRENT_CODE:
```python
<exact code to replace>
```
PROPOSED_CODE:
```python
<replacement code>
```
REASONING: <why this fix works and what to watch for>
"""
```

3. Parse the LLM response and store in `self_improvement_proposals`:
   - `issue_id` → link to the issue
   - `title` → generated from issue title + "fix"
   - `description` → the ROOT_CAUSE section
   - `target_file` → the related file path
   - `current_code` → extracted CURRENT_CODE block
   - `proposed_code` → extracted PROPOSED_CODE block
   - `reasoning` → the REASONING section
   - `risk_assessment` → the RISK value
   - `status` → "pending" (always starts here)

4. Create an autonomous notification for Patrick: "New improvement proposal: {title} (risk: {risk})"

5. Log summary: "Analyzed X issues, generated Y proposals (Z low risk, W medium risk)"

Safety:
- Generating proposals: AUTO (it's just text — LLM reasoning stored in a table)
- Applying proposals: REVIEW (never happens in this worker — separate approval flow)
- The worker MUST NOT import subprocess, os.system, or any code execution primitives
- The worker MUST NOT write to any file on disk
- The worker only reads from Qdrant/PostgreSQL and writes to `self_improvement_proposals` + `autonomous_notifications`

Worker class pattern:

```python
class ImprovementEngine:
    """Reasons about detected issues and proposes code fixes (REVIEW gated)"""
    
    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL",
            "postgresql://patrick:WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr@localhost/echo_brain")
        self.qdrant_url = "http://localhost:6333"
        self.ollama_url = "http://localhost:11434"
        self.collection = "echo_memory"
        self.model = "gemma2:9b"  # Use extraction model for analysis
        self.max_issues_per_cycle = 5
    
    async def run_cycle(self):
        """Main worker cycle — called by scheduler"""
        # 1. Query open issues (critical + warning)
        # 2. Skip issues that already have proposals
        # 3. For each issue: find related code in Qdrant
        # 4. Build analysis prompt with issue + code context
        # 5. Send to Ollama, parse response
        # 6. Store proposal in self_improvement_proposals
        # 7. Create notification for Patrick
        # 8. Log summary
```

### Task 7b: Proposal Review API Endpoints

Add to the Echo Brain API (before the frontend mount in main.py):

**GET /api/echo/proposals** — List proposals with optional status filter
```json
// GET /api/echo/proposals?status=pending
{
    "proposals": [
        {
            "id": "uuid",
            "title": "Fix DDR5 stale data in fact extraction",
            "issue_title": "DDR5 detected in response",
            "target_file": "src/autonomous/workers/fact_extraction_worker.py",
            "risk_assessment": "low",
            "status": "pending",
            "created_at": "2026-02-06T..."
        }
    ],
    "counts": {"pending": 3, "approved": 1, "rejected": 2, "applied": 0}
}
```

**GET /api/echo/proposals/{id}** — Full proposal detail (includes current_code, proposed_code, reasoning)

**POST /api/echo/proposals/{id}/approve** — Mark proposal as approved
- Sets `status = 'approved'`, `reviewed_by = 'patrick'`, `reviewed_at = NOW()`
- Creates audit log entry
- Does NOT apply the change yet (that's Phase 2a v0.3.9)

**POST /api/echo/proposals/{id}/reject** — Mark proposal as rejected
- Sets `status = 'rejected'`, `reviewed_by = 'patrick'`, `reviewed_at = NOW()`
- Optionally accepts a `reason` field
- Creates audit log entry

---

## Task 8: Register New Workers in main.py

Add the 5 new workers to the startup event in `src/main.py`, following the same pattern as existing workers. Each worker gets its own try/except block so one failure doesn't kill the others.

```python
# Add after existing 3 workers:

try:
    from src.autonomous.workers.codebase_indexer import CodebaseIndexer
    worker = CodebaseIndexer()
    worker_scheduler.register_worker("codebase_indexer", worker.run_cycle, interval_minutes=360)  # Every 6 hours
    workers_registered += 1
    logger.info("✅ Registered codebase_indexer worker (6 hours)")
except Exception as e:
    logger.error(f"❌ Failed to register codebase_indexer: {e}")

try:
    from src.autonomous.workers.schema_indexer import SchemaIndexer
    worker = SchemaIndexer()
    worker_scheduler.register_worker("schema_indexer", worker.run_cycle, interval_minutes=1440)  # Daily
    workers_registered += 1
    logger.info("✅ Registered schema_indexer worker (daily)")
except Exception as e:
    logger.error(f"❌ Failed to register schema_indexer: {e}")

try:
    from src.autonomous.workers.log_monitor import LogMonitor
    worker = LogMonitor()
    worker_scheduler.register_worker("log_monitor", worker.run_cycle, interval_minutes=15)  # Every 15 min
    workers_registered += 1
    logger.info("✅ Registered log_monitor worker (15 min)")
except Exception as e:
    logger.error(f"❌ Failed to register log_monitor: {e}")

try:
    from src.autonomous.workers.self_test_runner import SelfTestRunner
    worker = SelfTestRunner()
    worker_scheduler.register_worker("self_test_runner", worker.run_cycle, interval_minutes=60)  # Hourly
    workers_registered += 1
    logger.info("✅ Registered self_test_runner worker (hourly)")
except Exception as e:
    logger.error(f"❌ Failed to register self_test_runner: {e}")

try:
    from src.autonomous.workers.improvement_engine import ImprovementEngine
    worker = ImprovementEngine()
    worker_scheduler.register_worker("improvement_engine", worker.run_cycle, interval_minutes=120)  # Every 2 hours
    workers_registered += 1
    logger.info("✅ Registered improvement_engine worker (2 hours)")
except Exception as e:
    logger.error(f"❌ Failed to register improvement_engine: {e}")
```

Worker schedule summary:

| Worker | Interval | Stage | Purpose |
|--------|----------|-------|---------|
| conversation_watcher | 10 min | INGEST | Ingest new conversations |
| log_monitor | 15 min | THINK | Detect errors in own logs |
| fact_extraction | 30 min | INGEST | Extract facts from vectors |
| self_test_runner | 60 min | THINK | Validate own output quality |
| improvement_engine | 2 hours | IMPROVE | Reason about issues, propose fixes |
| codebase_indexer | 6 hours | INGEST | Index own source code + anime pipeline |
| knowledge_graph | daily | INGEST | Build fact connections |
| schema_indexer | daily | INGEST | Index own database schema |

---

## Task 9: Seed Self-Awareness Goals

Add goals to the autonomous_goals table for the new workers:

```sql
INSERT INTO autonomous_goals (name, description, goal_type, status, priority)
VALUES 
    ('codebase_self_index', 'Index Echo Brain source code and anime pipeline into vector store for self-awareness', 'maintenance', 'active', 7),
    ('schema_self_index', 'Index Echo Brain database schemas for self-awareness', 'maintenance', 'active', 6),
    ('log_monitoring', 'Monitor own service logs for errors and anomalies', 'maintenance', 'active', 9),
    ('output_validation', 'Run self-tests to validate query accuracy and detect regressions', 'maintenance', 'active', 10),
    ('self_improvement', 'Analyze detected issues, reason about root causes, generate fix proposals for review', 'improvement', 'active', 8),
    ('health_tracking', 'Track health metrics over time for trend analysis', 'maintenance', 'active', 5)
ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description, status = EXCLUDED.status;
```

---

## GATE 3: Workers Created and Importable

After creating all worker files, verify they import cleanly:

```bash
cd /opt/tower-echo-brain
/opt/tower-echo-brain/venv/bin/python -c "
modules = [
    ('CodebaseIndexer', 'src.autonomous.workers.codebase_indexer', 'CodebaseIndexer'),
    ('SchemaIndexer', 'src.autonomous.workers.schema_indexer', 'SchemaIndexer'),
    ('LogMonitor', 'src.autonomous.workers.log_monitor', 'LogMonitor'),
    ('SelfTestRunner', 'src.autonomous.workers.self_test_runner', 'SelfTestRunner'),
    ('ImprovementEngine', 'src.autonomous.workers.improvement_engine', 'ImprovementEngine'),
]
for name, mod_path, cls_name in modules:
    try:
        mod = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        instance = cls()
        assert hasattr(instance, 'run_cycle'), f'{name} missing run_cycle method'
        print(f'✅ {name}: importable, has run_cycle')
    except Exception as e:
        print(f'❌ {name}: {e}')
"
```

**All 5 must show ✅ before proceeding.**

---

## GATE 4: Service Restarts Clean With All 8 Workers

```bash
# Fix permissions on any new files
chmod -R 755 /opt/tower-echo-brain/src/autonomous/workers/

# Restart
sudo systemctl restart tower-echo-brain
sleep 15

# Check all workers registered
sudo journalctl -u tower-echo-brain --since "1 minute ago" --no-pager | grep -E "Registered|✅|❌|worker"

# Check status endpoint
curl -s http://localhost:8309/api/workers/status | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Running: {data.get(\"running\", False)}')
for name, info in data.get('workers', {}).items():
    print(f'  {name}: error_count={info.get(\"error_count\", \"?\")}')
print(f'Total workers: {len(data.get(\"workers\", {}))}')
"

echo ""
echo "Expected: 8 workers, all with error_count=0"
```

**Must show 8 workers running with 0 errors before proceeding.**

---

## GATE 5: First Cycle Verification

Wait for the first cycle of each new worker, then verify:

```bash
# Wait for first cycles (log_monitor at 15 min is the longest initial wait)
# But all workers run immediately on first registration, so check after 2 min
sleep 120

echo "=== GATE 5: First Cycle Results ==="

# Codebase indexed?
PGPASSWORD=WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr psql -h localhost -U patrick -d echo_brain -c "
SELECT COUNT(*) as files_indexed, 
       SUM(line_count) as total_lines,
       SUM(jsonb_array_length(functions)) as total_functions,
       SUM(jsonb_array_length(classes)) as total_classes
FROM self_codebase_index;
"

# Schema indexed?
PGPASSWORD=WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr psql -h localhost -U patrick -d echo_brain -c "
SELECT COUNT(*) as tables_indexed,
       SUM(row_count) as total_rows
FROM self_schema_index;
"

# Issues detected?
PGPASSWORD=WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr psql -h localhost -U patrick -d echo_brain -c "
SELECT issue_type, severity, COUNT(*) 
FROM self_detected_issues 
GROUP BY issue_type, severity 
ORDER BY severity, issue_type;
"

# Self-tests run?
PGPASSWORD=WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr psql -h localhost -U patrick -d echo_brain -c "
SELECT test_name, passed, response_time_ms 
FROM self_test_results 
WHERE run_at > NOW() - INTERVAL '10 minutes'
ORDER BY test_name;
"

# Health metrics recorded?
PGPASSWORD=WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr psql -h localhost -U patrick -d echo_brain -c "
SELECT metric_name, metric_value, recorded_at 
FROM self_health_metrics 
ORDER BY recorded_at DESC 
LIMIT 10;
"

# Improvement proposals generated? (may be empty on first cycle if no open issues yet)
PGPASSWORD=WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr psql -h localhost -U patrick -d echo_brain -c "
SELECT title, risk_assessment, status, created_at
FROM self_improvement_proposals
ORDER BY created_at DESC
LIMIT 5;
"

# Anime pipeline indexed? (secondary — OK if empty)
PGPASSWORD=WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr psql -h localhost -U patrick -d echo_brain -c "
SELECT file_path, line_count, jsonb_array_length(functions) as funcs
FROM self_codebase_index
WHERE file_path LIKE '%anime%' OR file_path LIKE '%tower-anime%'
LIMIT 10;
"

# Proposal API working?
curl -s http://localhost:8309/api/echo/proposals | python3 -m json.tool

# Detailed health endpoint?
curl -s http://localhost:8309/api/echo/health/detailed | python3 -m json.tool

echo ""
echo "=== Phase 2a COMPLETE if INGEST + THINK sections show data ==="
echo "=== IMPROVE section may be empty until issues are detected and analyzed ==="
```

---

## GATE 6: End-to-End Self-Awareness Test

The ultimate test: can Echo Brain answer questions about itself?

```bash
echo "=== GATE 6: Can Echo Brain Know Itself? ==="

# Test 1: Ask about its own code
python3 << 'PYEOF'
import httpx

queries = [
    # INGEST: Does it know its own code?
    "What files make up the Echo Brain intelligence layer?",
    "How does the worker scheduler work?",
    "What tables store facts in Echo Brain?",
    # INGEST: Does it know its own schema?
    "What endpoints does the Echo Brain API expose?",
    "What columns are in the self_improvement_proposals table?",
    # THINK: Does it know its own issues?
    "What issues have been detected in Echo Brain recently?",
    "What is the current self-test pass rate?",
    # IMPROVE: Can it reason about fixes?
    "Are there any pending improvement proposals?",
    # ANIME: Does it know the anime pipeline? (secondary — may be empty)
    "What anime production code has been indexed?",
]

for q in queries:
    try:
        resp = httpx.post("http://localhost:8309/api/echo/ask",
            json={"question": q}, timeout=60)
        answer = resp.json().get("answer", resp.json().get("response", "No response"))
        # Truncate for readability
        print(f"\nQ: {q}")
        print(f"A: {answer[:200]}...")
    except Exception as e:
        print(f"\nQ: {q}")
        print(f"ERROR: {e}")
PYEOF
```

These queries should now return useful answers because the codebase and schema are indexed in the vector store. If they return generic/unhelpful answers, the indexing didn't work properly.

---

## Success Criteria

After this prompt completes, ALL must be true:

**INGEST (can it read itself?)**
- [ ] 6 new database tables created (self_codebase_index, self_schema_index, self_detected_issues, self_test_results, self_health_metrics, self_improvement_proposals)
- [ ] 5 new worker files created and importable (codebase_indexer, schema_indexer, log_monitor, self_test_runner, improvement_engine)
- [ ] Service restarts with 8/8 workers running (3 existing + 5 new)
- [ ] Echo Brain's own codebase indexed: 50+ Python files with function/class metadata
- [ ] Anime pipeline code indexed (if paths exist) or graceful skip logged
- [ ] Schema indexed: 15+ database tables with column info
- [ ] Echo Brain can answer "What files make up the intelligence layer?" with actual file paths

**THINK (can it detect its own problems?)**
- [ ] Log monitor: detected and stored at least 1 issue from recent logs
- [ ] Self-tests: ran all 6+ tests, results stored in DB with pass/fail
- [ ] Health dashboard: `/api/echo/health/detailed` returns complete status
- [ ] Health metrics recorded over time

**IMPROVE (can it reason about fixes?)**
- [ ] Improvement engine: analyzed at least 1 open issue
- [ ] At least 1 improvement proposal stored in `self_improvement_proposals` with reasoning
- [ ] Proposal review endpoints working: GET /api/echo/proposals, POST approve/reject
- [ ] Patrick notified of new proposals via autonomous_notifications

**META (does the loop close?)**
- [ ] Audit log shows activity from all 5 new workers
- [ ] All documentation committed to git (README.md, docs/ARCHITECTURE.md, docs/ROADMAP.md, CHANGELOG.md)
- [ ] Git tag `v0.3.0` created after all gates pass

## Important Notes

- Database: `echo_brain` on localhost, user `patrick`, password `WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr`
- Service runs as `echo` user — all new files need 755 permissions
- New workers MUST be importable independently (no broken import chains)
- All workers use the same `run_cycle()` interface as existing workers
- Qdrant collection: `echo_memory` (1024D, mxbai-embed-large)
- The health endpoint and proposal endpoints MUST be registered BEFORE the frontend static mount in main.py
- The improvement_engine worker MUST NOT import subprocess, os.system, or any code execution primitives
- The improvement_engine worker MUST NOT write to any file on disk — only to PostgreSQL tables
- DO NOT modify existing worker code
- DO NOT recreate or delete Qdrant collections
- DO NOT install new system packages (pip install is OK with --break-system-packages if absolutely needed)
- Copy documentation files (README.md, docs/, CHANGELOG.md) into `/opt/tower-echo-brain/` 
- Commit all changes to git after Gate 5 passes
- Tag as v0.3.0
