# TASK: Deploy SearXNG + Build Web Search Integration for Echo Brain

## CONTEXT

Echo Brain is Patrick's self-hosted AI assistant at `/opt/tower-echo-brain/`. It currently has NO web search capability — this is the single biggest gap vs Perplexity. We're fixing that now.

**This is a BUILD task, not analysis.** You are deploying real code and services.

## SYSTEM FACTS (VERIFIED)

```
Echo Brain:     /opt/tower-echo-brain/
API:            FastAPI on port 8309
Service:        tower-echo-brain.service (runs as user 'echo', group 'patrick')
Startup:        /opt/tower-echo-brain/start_with_vault.sh
Config:         .env at /opt/tower-echo-brain/.env
Database:       PostgreSQL (echo_brain), user=patrick
Vector DB:      Qdrant at localhost:6333, collection=echo_memory (768D, nomic-embed-text)
Embeddings:     nomic-embed-text via Ollama at localhost:11434
Python:         venv at /opt/tower-echo-brain/venv/
Frontend:       Vue3 at /opt/tower-echo-brain/frontend/
MCP Server:     /opt/tower-echo-brain/mcp_server/stdio_server.py
Agents:         5 agents, managed via agent registry with hot-reload
Vault:          http://127.0.0.1:8200
Docker:         Already running (Wyze Bridge, possibly others)
```

## PHASE 1: DEPLOY SEARXNG (Docker)

### Step 1: Create directory structure

```bash
mkdir -p /opt/tower-echo-brain/docker
mkdir -p /opt/tower-echo-brain/config/searxng
```

### Step 2: Create SearXNG settings

Create `/opt/tower-echo-brain/config/searxng/settings.yml`:

```yaml
use_default_settings: true

server:
  secret_key: "$(openssl rand -hex 32)"
  limiter: false
  image_proxy: false

search:
  safe_search: 0
  autocomplete: ""
  default_lang: "en"
  formats:
    - html
    - json

engines:
  - name: google
    engine: google
    shortcut: g
    weight: 1.5
  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    weight: 1.0
  - name: bing
    engine: bing
    shortcut: b
    weight: 1.0
  - name: wikipedia
    engine: wikipedia
    shortcut: wp
    weight: 1.2
  - name: github
    engine: github
    shortcut: gh
    weight: 0.8
    categories: [it]
  - name: stackoverflow
    engine: stackoverflow
    shortcut: so
    weight: 1.0
    categories: [it]
  - name: arxiv
    engine: arxiv
    shortcut: ar
    weight: 0.8
    categories: [science]

outgoing:
  request_timeout: 5.0
  max_request_timeout: 10.0
  useragent_suffix: "EchoBrain/0.7"
```

**IMPORTANT:** Generate a real secret key with `openssl rand -hex 32` and substitute it into the file.

### Step 3: Create Docker Compose

Create `/opt/tower-echo-brain/docker/docker-compose.searxng.yml`:

```yaml
version: "3.8"

services:
  searxng:
    image: searxng/searxng:latest
    container_name: tower-searxng
    ports:
      - "127.0.0.1:8888:8080"
    volumes:
      - /opt/tower-echo-brain/config/searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Step 4: Start SearXNG

```bash
cd /opt/tower-echo-brain/docker
docker compose -f docker-compose.searxng.yml up -d
sleep 10
```

### Step 5: VERIFY SearXNG is working

```bash
# Health check
curl -s http://localhost:8888/healthz
echo ""

# JSON API test
curl -s "http://localhost:8888/search?q=python+fastapi&format=json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('results', [])
print(f'SearXNG returned {len(results)} results')
for r in results[:3]:
    print(f'  [{r.get(\"engine\",\"?\")}] {r.get(\"title\",\"\")}')
    print(f'    {r.get(\"url\",\"\")}')
"
```

**DO NOT PROCEED** unless SearXNG returns real search results.

---

## PHASE 2: BUILD SEARCH SERVICE

### Step 1: Understand existing code structure

```bash
# Find where services live
ls /opt/tower-echo-brain/src/services/
# Find existing imports and patterns
head -50 /opt/tower-echo-brain/src/services/*.py 2>/dev/null | head -100
# Check if httpx is already installed
/opt/tower-echo-brain/venv/bin/python -c "import httpx; print(f'httpx {httpx.__version__}')"
# Check for existing circuit breaker implementation
grep -rn "CircuitBreaker\|circuit_breaker\|breaker" /opt/tower-echo-brain/src/ --include="*.py" | head -10
```

### Step 2: Create the SearchService

Create `/opt/tower-echo-brain/src/services/search_service.py`

Use the implementation from `/opt/tower-echo-brain/docs/web-search-implementation-spec.md` as the base (the analysis prompt already created this file). Key requirements:

- Async with httpx (already installed)
- SearXNG primary at `http://localhost:8888`
- Brave Search API fallback (key from Vault if available, otherwise skip)
- 15-minute in-memory cache (dict with TTL)
- Circuit breaker on SearXNG (use existing pattern from codebase if one exists, otherwise simple implementation)
- `search()` method: returns SearchResponse with list of SearchResult
- `search_and_fetch()` method: search + fetch top-N pages + strip HTML + chunk
- HTML stripping: regex approach for MVP (no new dependency needed)
- Chunking: ~1500 words per chunk with 200-word overlap

**CRITICAL:** Follow the existing code patterns in `src/services/`. Match the import style, logging approach, and error handling patterns already in use. Do NOT introduce new patterns that conflict with the codebase.

### Step 3: Install any missing dependencies

```bash
# httpx should already be installed, but verify:
/opt/tower-echo-brain/venv/bin/pip install httpx --quiet
```

---

## PHASE 3: API ENDPOINTS

### Step 1: Find where routers are registered

```bash
# Find router registration pattern
grep -rn "include_router\|APIRouter" /opt/tower-echo-brain/src/api/ --include="*.py" | head -20
grep -rn "include_router" /opt/tower-echo-brain/src/main.py | head -10
```

### Step 2: Create search router

Create the router file following the pattern you found above. It needs:

- `POST /api/echo/search/web` — Full web search with options
- `GET /api/echo/search/web?q=...&n=10` — Quick search via GET

Both endpoints should return:
```json
{
  "query": "string",
  "results": [{"title": "...", "url": "...", "snippet": "...", "source_engine": "...", "position": 1}],
  "total_results": 42,
  "search_time_ms": 350.5,
  "source": "searxng",
  "cached": false
}
```

### Step 3: Register the router in main.py

Add the `include_router()` call following the existing pattern.

---

## PHASE 4: MCP TOOL

### Step 1: Find the MCP server

```bash
cat /opt/tower-echo-brain/mcp_server/stdio_server.py | head -80
grep -n "@mcp.tool\|def.*tool" /opt/tower-echo-brain/mcp_server/stdio_server.py | head -20
```

### Step 2: Add web_search MCP tool

Following the existing MCP tool pattern, add a `web_search` tool that:
1. Calls the Echo Brain `/api/echo/search/web` endpoint
2. Formats results as readable text for Claude Code
3. Includes title, URL, snippet for each result
4. Shows metadata (source engine, search time, total results)

---

## PHASE 5: INTEGRATE WITH ASK PIPELINE

### Step 1: Understand the current ask flow

```bash
# Find the /ask endpoint
grep -rn "def.*ask\|/ask\|/api/echo/ask" /opt/tower-echo-brain/src/ --include="*.py" | head -20

# Find the classifier
find /opt/tower-echo-brain/src -name "classifier*" -o -name "intent*" | head -5
ls /opt/tower-echo-brain/src/context_assembly/ 2>/dev/null

# Find the retriever/compiler
find /opt/tower-echo-brain/src -name "retriever*" -o -name "compiler*" -o -name "context*" | head -10

# Trace the ask flow
grep -n "async def" /opt/tower-echo-brain/src/api/endpoints/*.py | grep -i "ask\|query\|chat"
```

### Step 2: Add web search intent detection

In the classifier/intent detection code, add recognition for queries that need web search:
- Temporal signals: "latest", "current", "recent", "today", "2025", "2026"
- Information signals: "news", "weather", "price", "stock", "release", "update"
- Question signals that hit low-confidence memory results: "what is", "who is", "how to"
- Explicit: "search", "look up", "find out", "google"

### Step 3: Add web results to context compilation

In the retriever/compiler, add web search as a parallel retrieval source alongside vector search. The compiled context should include citations:
- `[W1]`, `[W2]` for web sources
- `[M1]`, `[M2]` for memory sources  
- `[F1]`, `[F2]` for facts

The LLM system prompt should instruct it to cite sources using these markers.

---

## PHASE 6: UPDATE .env AND RESTART

```bash
# Add web search config to .env
cat >> /opt/tower-echo-brain/.env << 'EOF'

# Web Search Configuration
SEARXNG_URL=http://localhost:8888
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MAX_RESULTS=10
WEB_SEARCH_CACHE_TTL_MINUTES=15
WEB_SEARCH_FETCH_TIMEOUT=15
EOF

# Restart Echo Brain
sudo systemctl restart tower-echo-brain
sleep 5

# Verify it's up
curl -s http://localhost:8309/health | python3 -m json.tool
```

---

## PHASE 7: VERIFY EVERYTHING

Run ALL of these. Every single one must pass.

```bash
echo "=== TEST 1: SearXNG Direct ==="
curl -s "http://localhost:8888/search?q=python+fastapi&format=json" | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(f'PASS: {len(d[\"results\"])} results') if len(d.get('results',[])) > 0 else print('FAIL: no results')
"

echo ""
echo "=== TEST 2: Echo Brain Search Endpoint ==="
curl -s -X POST http://localhost:8309/api/echo/search/web \
  -H "Content-Type: application/json" \
  -d '{"query": "python fastapi tutorial 2026"}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
r = d.get('results', [])
print(f'PASS: {len(r)} results, source={d.get(\"source\",\"?\")}') if len(r) > 0 else print('FAIL: no results')
"

echo ""
echo "=== TEST 3: Echo Brain GET endpoint ==="
curl -s "http://localhost:8309/api/echo/search/web?q=test&n=5" | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(f'PASS: {len(d.get(\"results\",[]))} results') if d.get('results') else print('FAIL')
"

echo ""
echo "=== TEST 4: Cache Test ==="
START=$(date +%s%N)
curl -s "http://localhost:8309/api/echo/search/web?q=cache+test+query&n=3" > /dev/null
MID=$(date +%s%N)
curl -s "http://localhost:8309/api/echo/search/web?q=cache+test+query&n=3" | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(f'PASS: cached={d.get(\"cached\",False)}') if d.get('cached') else print('WARN: cache miss (may be timing)')
"
END=$(date +%s%N)
echo "  First call: $(( ($MID - $START) / 1000000 ))ms, Second call: $(( ($END - $MID) / 1000000 ))ms"

echo ""
echo "=== TEST 5: Ask with Web Search ==="
curl -s -X POST http://localhost:8309/api/echo/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the latest version of Python released in 2025?"}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
resp = d.get('response', '')
print(f'PASS: Response length={len(resp)}') if len(resp) > 50 else print(f'FAIL: Response too short: {resp[:100]}')
print(f'  Preview: {resp[:200]}...')
"

echo ""
echo "=== TEST 6: MCP Tool (if accessible via API) ==="
curl -s -X POST http://localhost:8309/mcp \
  -H "Content-Type: application/json" \
  -d '{"method":"tools/call","params":{"name":"web_search","arguments":{"query":"hello world"}}}' 2>/dev/null | python3 -c "
import sys, json
try:
    d=json.load(sys.stdin)
    print('PASS: MCP tool responded') if d else print('FAIL')
except: print('SKIP: MCP endpoint not available via HTTP (normal if stdio-only)')
"

echo ""
echo "=== TEST 7: Service Health ==="
curl -s http://localhost:8309/health | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(f'Echo Brain: {d.get(\"status\",\"unknown\")}')
"
curl -s http://localhost:8888/healthz > /dev/null && echo "SearXNG: healthy" || echo "SearXNG: DOWN"
curl -s http://localhost:6333/collections > /dev/null && echo "Qdrant: healthy" || echo "Qdrant: DOWN"
curl -s http://localhost:11434/api/tags > /dev/null && echo "Ollama: healthy" || echo "Ollama: DOWN"

echo ""
echo "=== SUMMARY ==="
echo "If Tests 1-5 pass, web search is fully operational."
echo "Test 6 may skip if MCP is stdio-only — that's fine."
```

---

## CONSTRAINTS

- Do NOT change the embedding model. It's nomic-embed-text 768D. Leave it alone.
- Do NOT restructure existing code. Add to it.
- Do NOT install large new dependencies. httpx is already there.
- Match existing code style and patterns in the codebase.
- SearXNG must only bind to 127.0.0.1 (no external exposure).
- If you need to modify an existing file, make the MINIMUM change necessary.
- Log every new component with loguru (already used in the codebase).

## DONE WHEN

- [ ] SearXNG Docker container running at localhost:8888
- [ ] `/api/echo/search/web` returns web results
- [ ] `web_search` MCP tool registered and functional
- [ ] `/api/echo/ask` uses web search for current-events queries
- [ ] All 7 verification tests pass
- [ ] `.env` updated with web search config
- [ ] `tower-echo-brain.service` restarted and healthy
