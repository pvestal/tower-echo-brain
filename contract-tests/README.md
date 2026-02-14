# Echo Brain Contract Tests

Contract testing suite for Echo Brain's 203 API routes. Includes consumer-driven Pact tests (frontend/backend sync) and comprehensive all-routes contract verification.

## Test Types

### 1. Consumer/Provider (Pact)
Verifies the Vue.js frontend and FastAPI backend agree on API shapes.

```
consumer/  → Vue/TS tests that generate contract.json
contracts/ → Generated Pact contract files
provider/  → Python tests that verify FastAPI satisfies contracts
```

### 2. All-Routes Contract Test
Hits every route from the OpenAPI spec with proper payloads, verifying no 500s.

```
provider/test_all_routes.py  → Tests all 203 routes against running service
```

## Quick Start

```bash
# Run all-routes test (most common)
./scripts/run_contract_tests.sh routes

# Run consumer + provider Pact tests
./scripts/run_contract_tests.sh both

# Run everything
./scripts/run_contract_tests.sh all

# Run all-routes directly with JSON output
cd provider && python3 test_all_routes.py --json
```

## All-Routes Test Details

The `test_all_routes.py` script:
1. Fetches the OpenAPI spec from the running service
2. Tests every GET/POST/PUT/PATCH route with appropriate payloads
3. Reports PASS (< 500), FAIL (>= 500 or timeout), SKIP (dangerous/slow)
4. Exits 0 if all testable routes pass, 1 if any fail

**Skipped by design:**
- DELETE methods (destructive)
- Autonomous lifecycle endpoints (kill/stop/start/pause/resume)
- Known slow endpoints (> 15s: deep diagnostics, batch photo analysis)

**Expected external-dependency failures** (503 when not configured):
- Home Assistant routes (need HA server + token)
- Apple Music playlists (need developer + user tokens)
- Narration agent (not yet implemented)

### Adding Payloads for New POST Routes

Edit `POST_PAYLOADS` in `test_all_routes.py`:

```python
POST_PAYLOADS["/api/new/endpoint"] = {"field": "value"}
```

Set to `None` to explicitly skip a route:
```python
POST_PAYLOADS["/api/slow/endpoint"] = None  # skip
```

## Architecture

```
┌─────────────────┐     contract.json     ┌─────────────────┐
│   Vue Frontend   │ ──── generates ────> │   Pact Broker    │
│   (Consumer)     │                       │   (or local)     │
└─────────────────┘                       └────────┬─────────┘
                                                   │
                                          verifies against
                                                   │
                                          ┌────────v─────────┐
                                          │  FastAPI Backend  │
                                          │   (Provider)      │
                                          └──────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  test_all_routes.py                                          │
│  Fetches /openapi.json → tests ALL 203 routes → report      │
└──────────────────────────────────────────────────────────────┘
```

## CI Integration

See [INTEGRATION.md](./INTEGRATION.md) for wiring into GitHub Actions and git hooks.

## Current Coverage (2026-02-14)

| Metric | Count |
|--------|-------|
| Total routes | 203 |
| Passing | 178+ |
| Skipped (by design) | 13 |
| Failing (external deps) | 12 |
