# Echo Brain Upgrade Prompts — Execution Guide

**Created:** 2026-02-27
**Purpose:** Close the Perplexity gap and extend Echo Brain's unique advantages

---

## Prompt Sequence

Run these in order. Each prompt is self-contained but builds on the previous.

| # | Prompt | What It Does | Effort | Depends On |
|---|--------|-------------|--------|------------|
| **01** | `01-WEB-SEARCH-DEPLOY.md` | Deploy SearXNG + build SearchService + API endpoints + MCP tool + integrate with /ask pipeline | 8-12h | Nothing |
| **02** | `02-DOCUMENT-INGESTION.md` | PDF/DOCX/XLSX/CSV/code parser + semantic chunking + Qdrant storage + upload API + MCP tool | 10-14h | Nothing (can run parallel with 01) |
| **03** | `03-DEEP-RESEARCH-ENGINE.md` | Multi-step research: decompose → search web+memory → evaluate → synthesize cited reports | 20-30h | 01 (web search) |
| **04** | `04-SEARXNG-HARDENING.md` | Systemd auto-start + health monitoring integration | 1-2h | 01 |
| **05** | `05-PERSONAL-DATA-INDEXING.md` | Batch document ingester + Google Takeout parser + priority doc ingestion | 8-12h | 02 (doc ingestion) |

---

## How to Run

Each prompt is a markdown file designed to be fed directly to Claude Code:

```bash
cd /opt/tower-echo-brain

# Option A: Pipe directly
cat ~/prompts/01-WEB-SEARCH-DEPLOY.md | claude

# Option B: Reference as context
claude "Read and execute the instructions in ~/prompts/01-WEB-SEARCH-DEPLOY.md"

# Option C: Copy-paste the content into a Claude Code session
```

---

## Execution Strategy

### Week 1-2: Foundation (Prompts 01 + 02 in parallel)
- **Day 1-2:** Run Prompt 01 (web search). Verify all 7 tests pass.
- **Day 2-3:** Run Prompt 02 (document ingestion). Verify all 7 tests pass.
- **Day 3:** Run Prompt 04 (SearXNG hardening). Quick follow-up.
- **Day 3-4:** Run Prompt 05 (personal data indexing). Ingest your first batch of real documents.

### Week 3-5: Intelligence (Prompt 03)
- **Day 5-10:** Run Prompt 03 (deep research engine). This is the biggest build.
- Test with real questions: "Should I upgrade my RV batteries to X or Y?"
- Verify it combines web + memory results with citations.

### After Tier 1: What's Next
These prompts cover Tier 1 from the upgrade roadmap. After completion, the next priorities are:
1. **Home Assistant Bridge** — Smart home awareness
2. **Proactive Monitoring** — Financial anomaly detection, security alerts
3. **Memory Intelligence Extraction** — Turn raw vectors into structured insights

---

## Verification Checklist

After all prompts are executed:

```bash
# Full system health
curl -s http://localhost:8309/health | python3 -m json.tool

# Web search works
curl -s "http://localhost:8309/api/echo/search/web?q=test" | python3 -c "import sys,json; print(f'{len(json.load(sys.stdin)[\"results\"])} results')"

# Document ingestion works
curl -s http://localhost:8309/api/echo/documents | python3 -c "import sys,json; print(f'{len(json.load(sys.stdin))} documents ingested')"

# Deep research works
curl -s -X POST http://localhost:8309/api/echo/research \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the best practices for LiFePO4 battery management in RVs?","depth":"quick"}' | python3 -m json.tool

# MCP tools registered
curl -s http://localhost:8309/mcp -d '{"method":"tools/list"}' 2>/dev/null | python3 -c "
import sys,json
try:
    tools = json.load(sys.stdin).get('result',{}).get('tools',[])
    names = [t['name'] for t in tools]
    for expected in ['web_search','ingest_document','deep_research']:
        print(f'  {expected}: {\"✓\" if expected in names else \"✗\"}')
except: print('MCP check via HTTP not available (stdio mode)')
"

# Vector count grew
curl -s http://localhost:6333/collections/echo_memory | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'Total vectors: {d[\"result\"][\"vectors_count\"]}')
"
```

---

## Troubleshooting

**SearXNG won't start:**
```bash
docker logs tower-searxng --tail 50
# Common: port 8888 already in use
ss -tlnp | grep 8888
```

**Echo Brain won't restart after changes:**
```bash
sudo journalctl -u tower-echo-brain --since "5 minutes ago" --no-pager
# Common: import error in new code
/opt/tower-echo-brain/venv/bin/python -c "from src.services.search_service import SearchService; print('OK')"
```

**Embedding failures during document ingestion:**
```bash
# Check Ollama is up
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; [print(m['name']) for m in json.load(sys.stdin)['models']]"
# Check nomic-embed-text is available
curl -s http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":"test"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Dims: {len(d[\"embeddings\"][0])}')"
```

**Research engine times out:**
```bash
# Check all dependencies are healthy
curl -s http://localhost:8888/healthz && echo "SearXNG: OK" || echo "SearXNG: DOWN"
curl -s http://localhost:6333/collections && echo "Qdrant: OK" || echo "Qdrant: DOWN"  
curl -s http://localhost:11434/api/tags && echo "Ollama: OK" || echo "Ollama: DOWN"
```
