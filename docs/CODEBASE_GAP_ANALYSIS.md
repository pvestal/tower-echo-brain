# Echo Brain Codebase Gap Analysis

Generated: 2026-02-11
Branch: feature/semantic-memory-architecture
Audited by: 3 parallel agents (source code, tests/scripts, wiring/integration)

---

## Executive Summary

| Category | Critical | High | Medium | Low/Info |
|----------|----------|------|--------|----------|
| Syntax Errors | 9 | - | - | - |
| Missing Modules | 4 | - | - | - |
| Router Wiring | 1 | 1 | 2 | - |
| Unmounted Routers | - | 19 | - | - |
| Database Schema | - | 16+ tables | - | - |
| Credentials | - | - | 5+ locations | - |
| Dependencies | - | 1 | 1 | - |
| Tests/Config | - | 1 | 3 | 6 |
| Hardcoded URLs | - | - | 121 files | - |

**Total: 14 critical, 38+ high, 130+ medium, 6+ info**

---

## Part 1: Source Code Gaps

### CRITICAL: Syntax Errors (9 files)

All in `src/modules/generation/video/` -- broken string interpolation (missing `f` prefix):

| File | Line | Pattern |
|------|------|---------|
| `fix_video_generation_permissions.py` | 15 | `"/home/{os.getenv(...)}/Videos/"` -- not an f-string |
| `video_generator_multi.py` | 134 | `"/home/{os.getenv(...)}/ComfyUI/output/..."` -- not an f-string |
| `svd_video_fixed.py` | 107 | `"/home/{os.getenv(...)}/ComfyUI/output/..."` -- not an f-string |
| `fix_video_now.py` | 126 | Unterminated string literal |
| `video_generation_system.py` | 21 | `Path("/home/{os.getenv(...)}/Videos")` -- not an f-string |
| `video_generation_fixed.py` | 19 | `Path("/home/{os.getenv(...)}/Videos")` -- not an f-string |
| `svd_video_generator.py` | 151 | `"/home/{os.getenv(...)}/ComfyUI/output/..."` -- not an f-string |
| `autonomous_loop_v2.py` | 273 | Nested quote mismatch in sed command |
| `tower_monitor.py` | 57 | Nested quote mismatch |

**Impact:** These files produce literal strings like `/home/{os.getenv('TOWER_USER', 'patrick')}/Videos` instead of resolving paths. All video generation paths break silently.

### CRITICAL: Missing Module Dependencies (4 modules)

| Module | Importing File(s) | Issue |
|--------|-------------------|-------|
| `src.model_config` | `echo_main_router.py:14`, `resilient_model_manager.py`, `autonomous_solo_generator.py` | Module does not exist. `get_model` is defined in `src.api.models_manager` |
| `src.agents.coding_agent` | `agents.py:15,20` | File doesn't exist. Only `src/agents/autonomous/coding_agent.py` exists (different path) |
| `src.agents.reasoning_agent` | `agents.py:16,21` | Does not exist anywhere in codebase |
| `src.agents.narration_agent` | `agents.py:17,22` | Does not exist anywhere in codebase |

**Also missing:**
- `src/interfaces/` directory -- imported by `src/core/container.py` lines 16-28
- `src/mocks/` directory -- imported by `src/core/container.py` lines 31-37

**Impact:** Agent router fails to load (caught by try/except, degrades gracefully). Main router import of `model_config` would fail but is also caught.

### WARNING: Deprecated Files Still Present

| File | Issue |
|------|-------|
| `src/api/model_manager.py` | Line 1: "DEPRECATED: Use unified_router.py instead" -- contains only Protocol stubs |

### INFO: TODO Comments in Production Code

| File | Line | Comment |
|------|------|---------|
| `src/main.py` | 466 | `# TODO: Add embedding generation here` |
| `src/integrations/moltbook/client.py` | 45, 105 | API implementation TODOs |
| `src/modules/video_generator_fixed.py` | 100 | SVD interpolation TODO |
| `src/core/container.py` | 256-260 | Production implementations not yet created |

---

## Part 2: Router & Wiring Gaps

### CRITICAL: Duplicate Reasoning Router Mount

**File:** `src/main.py`
- **Lines 177-178:** `from src.api.endpoints.reasoning_router import router as reasoning_router` mounted at `/api/echo`
- **Lines 197-198:** `from src.api.endpoints.echo_reasoning_router import router as reasoning_router` mounted at `/api/echo`

The second mount shadows the first. Variable name `reasoning_router` is reused. Endpoints from the first file become inaccessible.

### HIGH: 19 Unmounted Routers

These routers are fully implemented but **never included in main.py**:

| Router File | Expected Prefix | Endpoints |
|-------------|----------------|-----------|
| `src/api/google_calendar_api.py` | `/api/calendar` | Calendar integration |
| `src/api/models_manager.py` | `/api/models` | Model management |
| `src/api/solutions.py` | (inline) | Past solutions lookup |
| `src/api/takeout_stub.py` | (inline) | Google Takeout ingestion |
| `src/api/preferences.py` | (none) | User preferences |
| `src/api/google_data.py` | `/google` | Google data import |
| `src/api/home_assistant_api.py` | `/api/home` | Home Assistant |
| `src/api/integrations.py` | (none) | Integration management |
| `src/api/claude_bridge.py` | `/api/echo` | Claude integration |
| `src/api/delegation_routes.py` | `/delegate` | Task delegation |
| `src/api/notifications_api.py` | `/api/notifications` | Notification management |
| `src/api/git_operations.py` | `/git` | Git operations |
| `src/api/vault.py` | `/api/vault` | Secret vault |
| `src/api/repair_api.py` | `/api/repair` | Self-repair |
| `src/api/codebase.py` | (inline) | Codebase operations |
| `src/api/anime.py` | `/api/echo/anime` | Anime pipeline |
| `src/api/knowledge.py` | `/api/knowledge` | Knowledge management |
| `src/api/lora_training.py` | `/api/lora` | LoRA training |
| `src/api/resilience_status.py` | `/api/resilience` | Resilience monitoring |

**Decision needed:** Mount these or delete them. Currently they're dead code.

### MEDIUM: Duplicate Endpoint Definitions

Same endpoints defined in multiple routers (last-mounted wins, others are shadowed):

| Endpoint | Defined In |
|----------|-----------|
| `/api/echo/memory/status` | `echo_main_router.py` + `memory_router.py` |
| `/api/echo/memory/search` | `echo_main_router.py` + `memory_router.py` |
| `/api/echo/memory/ingest` | `echo_main_router.py` + `memory_router.py` |
| `/api/echo/health` | `echo_main_router.py` + multiple others |

---

## Part 3: Database Schema Gaps

### Potentially Missing Tables

Code references these tables but they may not exist in the database:

| Table | Referenced In | Purpose |
|-------|-------------|---------|
| `contract_monitor_results` | main.py:208 | Contract test results |
| `contract_monitor_issues` | main.py:686 | Open contract issues |
| `contract_monitor_snapshots` | main.py:685 | API snapshots |
| `self_codebase_index` | main.py:578 | Own code index |
| `self_schema_index` | main.py:581 | Own DB schema index |
| `self_test_results` | main.py:604 | Self-test results |
| `self_detected_issues` | main.py:619 | Detected issues |
| `self_improvement_proposals` | main.py:773 | Improvement proposals |
| `vector_content` | main.py:406 | Vector content cache |
| `ingestion_tracking` | main.py:486 | Ingestion progress |
| `past_solutions` | solutions.py:36 | Past solutions |
| `domain_category_stats` | main.py:1025 | Domain stats |
| `domain_ingestion_log` | main.py:1028 | Domain ingestion log |
| `knowledge_facts` | main.py:1067 | Knowledge triples |
| `knowledge_connections` | main.py:1103 | Fact relationships |
| `autonomous_audit_log` | Has schema.sql | Audit logging |

**Impact:** Endpoints querying missing tables return 500 errors. Many of these tables ARE created at startup (auto-create in main.py), but this should be verified.

---

## Part 4: Tests, Scripts & Configuration Gaps

### HIGH: Incomplete requirements.txt

**File:** `/opt/tower-echo-brain/requirements.txt`

Current contents (minimal, unversioned):
```
fastapi
uvicorn[standard]
httpx
asyncpg
pydantic
python-multipart
psutil
```

**Missing:** qdrant-client, requests, python-dotenv, pyyaml, whisper, piper-tts, torch, and ~20 other runtime dependencies. A bare `pip install -r requirements.txt` will not produce a working service.

### MEDIUM: Contract Tests Reference Wrong API Version

**File:** `contract-tests/consumer/tests/echo-brain.consumer.spec.ts`

Pact consumer contracts define expectations for `/api/v1/health` and `/api/v1/query`, but the provider implements `/api/echo/*`. Contract verification will fail on next run.

### MEDIUM: Test References Non-Existent Routes

**File:** `tests/test_router_inclusion.py`
- Checks for moltbook routes that don't exist in main.py

### MEDIUM: Frontend Version Mismatch

**File:** `frontend/package.json`
- Version: `"0.0.0"` -- should be `"0.5.0"` to match backend

### INFO: Plaintext Password in .env.example

**File:** `config/.env.example` line 10 -- still contains actual database password

### INFO: GPU Configuration Conflict

**File:** `/etc/systemd/system/tower-echo-brain.service.d/gpu.conf`
- Sets both `CUDA_VISIBLE_DEVICES=0` (NVIDIA) and `HIP_VISIBLE_DEVICES=1` (AMD)
- Unclear which GPU is actually used for inference

### INFO: Systemd Hard Dependency on vault.service

**File:** `/etc/systemd/system/tower-echo-brain.service`
- `Requires=vault.service` -- service won't start if vault.service doesn't exist

### INFO: 6 of 8 CI/CD Workflows Disabled

Only `main-ci.yml` and `release.yml` are active. All comprehensive test workflows are `.disabled`.

### INFO: Archive Artifacts in Root

- `echo-brain-contract-tests.tar.gz` (13.5 KB)
- `tower-deploy.tar.gz` (20.9 KB)

---

## Part 5: Security & Credentials

### Hardcoded Credentials in Source Code (11 files)

These files contain the database password as env var defaults (functional, not just docs):

| File | Line | Pattern |
|------|------|---------|
| `src/main.py` | 60 | `os.getenv('DB_PASSWORD', '<password>')` |
| `src/context_assembly/retriever.py` | 33 | Old password as default |
| `src/autonomous/workers/improvement_engine.py` | 24 | Old password hardcoded |
| `tests/echo_brain_smoke_test.py` | 350 | Password in test config |
| `scripts/status-verbose.sh` | - | Password in connection string |
| `scripts/test_single_file_ingestion.py` | - | Password in connection string |
| `tower-deploy/deploy.sh` | - | Password in deploy script |
| `tower-deploy/scripts/generation_monitor.py` | - | Password hardcoded |
| `tower-deploy/scripts/generation_output_validator.py` | - | Password hardcoded |
| `tower-deploy/sql/schema_alignment.sql` | - | Password in SQL |
| `start_with_vault.sh` | - | Password reference |

**Note:** Doc files were scrubbed in the previous commit. Source code files retain passwords as functional env var defaults.

### Hardcoded Service URLs (121 files)

Common patterns that should be centralized:
- `http://localhost:6333` -- Qdrant (~30 files)
- `http://localhost:11434` -- Ollama (~40 files)
- `http://localhost:8188` -- ComfyUI (~20 files)
- `http://localhost:8309` -- Echo Brain API (~15 files)
- `postgresql://patrick:...@localhost` -- DB connection (~8 files)

---

## Part 6: What's Working Correctly

| Component | Status |
|-----------|--------|
| All 12 workers registered and running | OK |
| MCP tools (search_memory, get_facts, store_fact) aligned | OK |
| Nginx WebSocket proxy for voice | OK |
| External service URLs consistent across files | OK |
| HTTP request middleware configured | OK |
| Voice service (STT + TTS + WS) | OK |
| Frontend build (TypeScript, no errors) | OK |
| Pytest smoke tests (26 pass) | OK |
| Vitest contract tests (11 pass) | OK |

---

## Action Items (Priority Order)

### Critical (Fix Now)
1. Fix 9 syntax errors in `src/modules/generation/video/` -- add `f` prefix to string interpolation
2. Create `src/model_config.py` with `get_model()` or fix import in `echo_main_router.py`
3. Fix or remove missing agent imports (`coding_agent`, `reasoning_agent`, `narration_agent`)
4. Remove duplicate reasoning router mount in `main.py` (lines 175-183 or 195-203)

### High (Fix Before Next Deploy)
5. Mount needed routers or delete unused router files (19 unmounted)
6. Fix `requirements.txt` with complete, versioned dependencies
7. Update Pact consumer contracts from `/api/v1/*` to `/api/echo/*`
8. Verify all referenced database tables exist (16+ tables)
9. Create `src/interfaces/` and `src/mocks/` or remove imports from `container.py`

### Medium (Address Soon)
10. Consolidate duplicate endpoint definitions across routers
11. Centralize hardcoded service URLs to environment variables or config
12. Scrub password from `config/.env.example`
13. Fix frontend `package.json` version to `0.5.0`
14. Re-enable comprehensive CI/CD workflows
15. Remove `vault.service` hard dependency from systemd unit

### Low (Housekeeping)
16. Remove deprecated `src/api/model_manager.py`
17. Clean up archive tarballs from project root
18. Resolve GPU configuration conflict in systemd overrides
19. Address remaining TODO comments or mark as deferred
