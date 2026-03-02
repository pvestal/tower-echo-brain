# Implementation Spec: Web Search Integration

**Priority:** P0 — Highest Impact Tier 1
**Estimated Effort:** 8-12 hours
**Author:** Gap Analysis, 2026-02-27

---

## Overview

Add web search capability to Echo Brain via a self-hosted SearXNG instance, with Brave Search API as a fallback. This is the single highest-impact upgrade because it:

1. Closes the largest gap with Perplexity (real-time web information)
2. Unblocks Deep Research, Finance Enhancement, and Proactive Monitoring
3. Requires zero ongoing cost (SearXNG is free, self-hosted)
4. Integrates cleanly with existing architecture (new service + endpoints + MCP tool)

---

## Architecture

```
┌─────────────────┐
│  User Query      │
│  "What's the     │
│   latest on X?"  │
└────────┬────────┘
         │
         v
┌─────────────────┐     ┌──────────────────┐
│  Echo Brain      │────>│  Domain Classifier│
│  /api/echo/ask   │     │  (classifier.py)  │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         │  needs_web_search=True│
         v                       v
┌─────────────────┐     ┌──────────────────┐
│  SearchService   │────>│  SearXNG          │
│  (new)           │     │  localhost:8888   │
└────────┬────────┘     └──────────────────┘
         │                       │
         │  fallback             v
         │              ┌──────────────────┐
         └─────────────>│  Brave Search API │
                        │  (if SearXNG down)│
                        └──────────────────┘
         │
         v
┌─────────────────┐
│  ContentFetcher  │  Fetch top-N page content
│  (enhanced       │  Clean HTML → text
│   web_fetch)     │  Chunk into passages
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Context Compiler│  Merge web results with
│  (compiler.py)   │  memory results + facts
└────────┬────────┘
         │
         v
┌─────────────────┐
│  LLM Response    │  Cite sources: [1] web,
│  with Citations  │  [M1] memory, [F1] fact
└─────────────────┘
```

---

## Component 1: SearXNG Deployment

### Docker Compose

File: `/opt/tower-echo-brain/docker/docker-compose.searxng.yml`

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
      - SEARXNG_SECRET=generate-random-secret-here
    restart: unless-stopped
    networks:
      - tower
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  tower:
    external: true
```

### SearXNG Configuration

File: `/opt/tower-echo-brain/config/searxng/settings.yml`

```yaml
use_default_settings: true

server:
  secret_key: "generate-random-secret-here"
  limiter: false  # Local only, no rate limiting needed
  image_proxy: false

search:
  safe_search: 0
  autocomplete: ""
  default_lang: "en"
  formats:
    - html
    - json  # Critical: enables JSON API

engines:
  # Prioritize reliable engines
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

  # Disable noisy/slow engines
  - name: brave
    disabled: true  # Using Brave API as direct fallback instead

  - name: yahoo
    disabled: true

outgoing:
  request_timeout: 5.0
  max_request_timeout: 10.0
  useragent_suffix: "EchoBrain/0.6"
```

### Systemd Service

File: `/etc/systemd/system/tower-searxng.service`

```ini
[Unit]
Description=Tower SearXNG Search Engine
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/tower-echo-brain/docker
ExecStart=/usr/bin/docker compose -f docker-compose.searxng.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.searxng.yml down
User=patrick
Group=patrick

[Install]
WantedBy=multi-user.target
```

---

## Component 2: SearchService

### File: `src/services/search_service.py`

```python
"""Web search service with SearXNG primary and Brave API fallback."""

import asyncio
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import httpx
from loguru import logger

from src.resilience.service_breakers import CircuitBreaker


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source_engine: str
    position: int
    fetched_content: Optional[str] = None
    content_chunks: list[str] = field(default_factory=list)


@dataclass
class WebDocument:
    """A fetched and chunked web document."""
    url: str
    title: str
    content: str
    chunks: list[str]
    fetched_at: datetime
    char_count: int = 0


@dataclass
class SearchResponse:
    """Complete search response with results and metadata."""
    query: str
    results: list[SearchResult]
    total_results: int
    search_time_ms: float
    source: str  # "searxng" or "brave"
    cached: bool = False


class SearchService:
    """Web search via SearXNG with Brave Search API fallback."""

    CACHE_TTL = timedelta(minutes=15)
    MAX_FETCH_CONCURRENT = 5
    MAX_CONTENT_CHARS = 15_000  # Per page
    CHUNK_SIZE = 1500  # Tokens (roughly)
    CHUNK_OVERLAP = 200

    def __init__(
        self,
        searxng_url: str = "http://localhost:8888",
        brave_api_key: Optional[str] = None,
    ):
        self.searxng_url = searxng_url
        self.brave_api_key = brave_api_key
        self._cache: dict[str, tuple[SearchResponse, datetime]] = {}
        self._searxng_breaker = CircuitBreaker(
            name="searxng",
            failure_threshold=3,
            success_threshold=2,
            timeout=60.0,
        )

    async def search(
        self,
        query: str,
        num_results: int = 10,
        categories: list[str] | None = None,
        time_range: str | None = None,
    ) -> SearchResponse:
        """Search the web. Tries SearXNG first, falls back to Brave."""
        cache_key = self._cache_key(query, num_results, categories)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Try SearXNG first
        if self._searxng_breaker.is_closed:
            try:
                response = await self._search_searxng(
                    query, num_results, categories, time_range
                )
                self._searxng_breaker.record_success()
                self._set_cached(cache_key, response)
                return response
            except Exception as e:
                logger.warning(f"SearXNG search failed: {e}")
                self._searxng_breaker.record_failure()

        # Fallback to Brave
        if self.brave_api_key:
            try:
                response = await self._search_brave(query, num_results)
                self._set_cached(cache_key, response)
                return response
            except Exception as e:
                logger.error(f"Brave search also failed: {e}")
                raise

        raise RuntimeError("All search backends unavailable")

    async def search_and_fetch(
        self,
        query: str,
        max_pages: int = 5,
        max_chars_per_page: int = 15_000,
    ) -> list[WebDocument]:
        """Search, fetch top pages, clean and chunk content."""
        search_response = await self.search(query, num_results=max_pages * 2)

        # Fetch pages concurrently with semaphore
        sem = asyncio.Semaphore(self.MAX_FETCH_CONCURRENT)
        tasks = []
        for result in search_response.results[:max_pages]:
            tasks.append(self._fetch_and_chunk(result.url, result.title, sem))

        documents = await asyncio.gather(*tasks, return_exceptions=True)
        return [doc for doc in documents if isinstance(doc, WebDocument)]

    async def _search_searxng(
        self,
        query: str,
        num_results: int,
        categories: list[str] | None,
        time_range: str | None,
    ) -> SearchResponse:
        """Search via SearXNG JSON API."""
        import time
        start = time.monotonic()

        params = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        if categories:
            params["categories"] = ",".join(categories)
        if time_range:
            params["time_range"] = time_range  # day, week, month, year

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.searxng_url}/search", params=params
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for i, r in enumerate(data.get("results", [])[:num_results]):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                source_engine=r.get("engine", "unknown"),
                position=i + 1,
            ))

        elapsed = (time.monotonic() - start) * 1000
        return SearchResponse(
            query=query,
            results=results,
            total_results=data.get("number_of_results", len(results)),
            search_time_ms=elapsed,
            source="searxng",
        )

    async def _search_brave(
        self, query: str, num_results: int
    ) -> SearchResponse:
        """Search via Brave Search API (fallback)."""
        import time
        start = time.monotonic()

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": num_results},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.brave_api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for i, r in enumerate(data.get("web", {}).get("results", [])[:num_results]):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                source_engine="brave",
                position=i + 1,
            ))

        elapsed = (time.monotonic() - start) * 1000
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time_ms=elapsed,
            source="brave",
        )

    async def _fetch_and_chunk(
        self, url: str, title: str, sem: asyncio.Semaphore
    ) -> WebDocument:
        """Fetch a URL, strip HTML, chunk for embedding."""
        async with sem:
            async with httpx.AsyncClient(
                timeout=15.0, follow_redirects=True,
                headers={"User-Agent": "EchoBrain/0.6 (research assistant)"}
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text

        # Strip HTML to text
        content = self._html_to_text(html)
        content = content[:self.MAX_CONTENT_CHARS]

        # Chunk
        chunks = self._chunk_text(content)

        return WebDocument(
            url=url,
            title=title,
            content=content,
            chunks=chunks,
            fetched_at=datetime.utcnow(),
            char_count=len(content),
        )

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Convert HTML to clean text. Uses simple regex approach."""
        import re
        # Remove script and style blocks
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode entities
        import html as html_module
        text = html_module.unescape(text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def _cache_key(self, query, num_results, categories) -> str:
        raw = f"{query}:{num_results}:{categories}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> SearchResponse | None:
        if key in self._cache:
            response, cached_at = self._cache[key]
            if datetime.utcnow() - cached_at < self.CACHE_TTL:
                response.cached = True
                return response
            del self._cache[key]
        return None

    def _set_cached(self, key: str, response: SearchResponse):
        self._cache[key] = (response, datetime.utcnow())
        # Prune old entries
        now = datetime.utcnow()
        expired = [k for k, (_, t) in self._cache.items() if now - t > self.CACHE_TTL]
        for k in expired:
            del self._cache[k]
```

---

## Component 3: API Endpoints

### File: New routes in `src/api/endpoints/search_router.py`

```python
"""Web search API endpoints."""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/echo/search", tags=["search"])


class WebSearchRequest(BaseModel):
    query: str
    num_results: int = 10
    categories: list[str] | None = None
    time_range: str | None = None  # day, week, month, year
    fetch_content: bool = False  # Also fetch and chunk page content
    max_pages: int = 5


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source_engine: str
    position: int


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResult]
    total_results: int
    search_time_ms: float
    source: str
    cached: bool


@router.post("/web", response_model=WebSearchResponse)
async def web_search(request: WebSearchRequest):
    """Search the web via SearXNG/Brave."""
    search_service = get_search_service()  # Dependency injection
    try:
        response = await search_service.search(
            query=request.query,
            num_results=request.num_results,
            categories=request.categories,
            time_range=request.time_range,
        )
        return WebSearchResponse(
            query=response.query,
            results=[WebSearchResult(
                title=r.title, url=r.url, snippet=r.snippet,
                source_engine=r.source_engine, position=r.position
            ) for r in response.results],
            total_results=response.total_results,
            search_time_ms=response.search_time_ms,
            source=response.source,
            cached=response.cached,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search unavailable: {e}")


@router.get("/web")
async def web_search_get(
    q: str = Query(..., description="Search query"),
    n: int = Query(10, description="Number of results"),
):
    """Quick web search via GET."""
    return await web_search(WebSearchRequest(query=q, num_results=n))
```

---

## Component 4: MCP Tool

### Addition to `mcp_server/stdio_server.py`

```python
@mcp.tool()
async def web_search(
    query: str,
    num_results: int = 10,
    categories: str = "",
    fetch_content: bool = False,
) -> str:
    """Search the web using SearXNG. Returns titles, URLs, and snippets.

    Args:
        query: Search query
        num_results: Max results (default 10)
        categories: Comma-separated categories (general, science, it, news)
        fetch_content: Also fetch and chunk page content (slower)
    """
    cats = [c.strip() for c in categories.split(",") if c.strip()] or None
    payload = {
        "query": query,
        "num_results": num_results,
        "categories": cats,
        "fetch_content": fetch_content,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ECHO_BRAIN_URL}/api/echo/search/web",
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()

    # Format for Claude
    lines = [f"Web search results for: {query}\n"]
    for r in data["results"]:
        lines.append(f"[{r['position']}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['snippet']}")
        lines.append("")

    lines.append(f"\nSource: {data['source']} | {data['search_time_ms']:.0f}ms | {data['total_results']} total")
    return "\n".join(lines)
```

---

## Component 5: Integration with Ask Pipeline

### Modifications to `src/context_assembly/classifier.py`

Add web search intent detection:

```python
# In classify_query():
WEB_SEARCH_SIGNALS = [
    r"\b(latest|current|recent|today|now|2026|2025)\b",
    r"\b(news|weather|price|stock|score|release)\b",
    r"\b(what is|who is|where is|when did|how to)\b",
    r"\b(search|look up|find out|google)\b",
]

def needs_web_search(query: str, memory_results: list) -> bool:
    """Determine if query needs web search in addition to memory."""
    # Explicit search request
    if any(re.search(p, query, re.IGNORECASE) for p in WEB_SEARCH_SIGNALS[:1]):
        return True

    # Memory results are insufficient (low confidence or few results)
    if not memory_results or all(r.score < 0.6 for r in memory_results):
        # Check if it's a factual question
        if any(re.search(p, query, re.IGNORECASE) for p in WEB_SEARCH_SIGNALS[2:]):
            return True

    return False
```

### Modifications to `src/context_assembly/retriever.py`

Add web search as parallel retrieval source:

```python
# In ParallelRetriever.retrieve():
async def retrieve(self, query: str, domain: str) -> RetrievalResult:
    # Existing: parallel vector + text search
    tasks = [
        self._vector_search(query),
        self._text_search(query),
    ]

    # New: conditionally add web search
    if self.classifier.needs_web_search(query, preview_results=None):
        tasks.append(self._web_search(query))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    # ... merge and rank as before, but include web results with source attribution
```

### Modifications to `src/context_assembly/compiler.py`

Add citation tracking:

```python
# In compile_context():
def compile_context(self, facts, memory_results, web_results=None):
    context_parts = []
    citations = []

    # Facts (highest priority, no citation needed)
    for fact in facts:
        context_parts.append(f"Known fact: {fact.subject} {fact.predicate} {fact.object}")

    # Memory results
    for i, mem in enumerate(memory_results):
        context_parts.append(f"[M{i+1}] From memory: {mem.text}")
        citations.append({"id": f"M{i+1}", "type": "memory", "source": mem.source})

    # Web results (new)
    if web_results:
        for i, web in enumerate(web_results):
            context_parts.append(f"[W{i+1}] From web ({web.title}): {web.snippet}")
            citations.append({"id": f"W{i+1}", "type": "web",
                            "url": web.url, "title": web.title})

    return CompiledContext(
        text="\n\n".join(context_parts),
        citations=citations,
        token_count=self._count_tokens("\n\n".join(context_parts))
    )
```

---

## Component 6: Autonomous Health Monitoring

### Addition to `config/autonomous_services.yaml`

```yaml
services:
  # ... existing services ...
  searxng:
    name: "SearXNG Search"
    health_endpoint: "http://localhost:8888/healthz"
    restart_command: "docker restart tower-searxng"
    check_interval: 60
    max_retries: 3
    circuit_breaker:
      failure_threshold: 3
      timeout: 120
```

---

## Database Schema Changes

**None required.** Web search results are transient (cached in memory for 15 minutes) and injected into the existing context pipeline. If we want to persist web search results for future retrieval, we can store them as regular vectors in `echo_memory` with `type: "web_content"` — the existing schema supports this via the payload field.

Optional future enhancement:
```sql
-- Track search history for analytics (not required for MVP)
CREATE TABLE IF NOT EXISTS web_search_log (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    source VARCHAR(20) NOT NULL,  -- searxng, brave
    result_count INTEGER,
    search_time_ms FLOAT,
    cached BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## New Dependencies

```
# Add to requirements.txt
# None new — httpx (already installed) handles all HTTP
# SearXNG runs in Docker, no Python deps needed

# Optional: beautifulsoup4 for better HTML parsing
# (but regex approach works for MVP)
```

SearXNG Docker image: ~150MB
Memory overhead: ~100MB RAM for SearXNG container
Network: localhost only (no external ports exposed)

---

## Configuration

### Environment Variables (add to .env)

```bash
# Web Search
SEARXNG_URL=http://localhost:8888
BRAVE_SEARCH_API_KEY=  # Optional fallback, store in Vault
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MAX_RESULTS=10
WEB_SEARCH_FETCH_TIMEOUT=15
WEB_SEARCH_CACHE_TTL_MINUTES=15
```

### Vault Secret (if using Brave fallback)

```bash
vault kv put secret/echo-brain/brave-search api_key=YOUR_KEY_HERE
```

---

## Test Plan

### Unit Tests

```python
# tests/test_search_service.py

class TestSearchService:
    async def test_searxng_search_returns_results(self):
        """SearXNG returns valid search results."""
        service = SearchService()
        results = await service.search("python fastapi tutorial")
        assert len(results.results) > 0
        assert results.source == "searxng"
        assert all(r.url.startswith("http") for r in results.results)

    async def test_brave_fallback_on_searxng_failure(self):
        """Falls back to Brave when SearXNG is down."""
        service = SearchService(
            searxng_url="http://localhost:99999",  # Invalid
            brave_api_key="test-key"
        )
        # Mock Brave response
        results = await service.search("test query")
        assert results.source == "brave"

    async def test_cache_returns_cached_results(self):
        """Second identical query returns cached results."""
        service = SearchService()
        r1 = await service.search("test query")
        r2 = await service.search("test query")
        assert r2.cached is True
        assert r1.results == r2.results

    async def test_fetch_and_chunk_produces_documents(self):
        """search_and_fetch returns chunked web documents."""
        service = SearchService()
        docs = await service.search_and_fetch("python tutorial", max_pages=2)
        assert len(docs) > 0
        assert all(len(d.chunks) > 0 for d in docs)

    async def test_html_to_text_strips_tags(self):
        """HTML conversion produces clean text."""
        html = "<html><script>evil()</script><p>Hello <b>world</b></p></html>"
        text = SearchService._html_to_text(html)
        assert "Hello world" in text
        assert "<" not in text
        assert "evil" not in text

    async def test_circuit_breaker_opens_on_failures(self):
        """Circuit breaker opens after repeated SearXNG failures."""
        service = SearchService(searxng_url="http://localhost:99999")
        for _ in range(4):
            try:
                await service.search("test")
            except:
                pass
        assert not service._searxng_breaker.is_closed
```

### Integration Tests

```python
# tests/integration/test_web_search_integration.py

class TestWebSearchIntegration:
    async def test_ask_with_web_search(self):
        """/api/echo/ask uses web search for current-events queries."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8309/api/echo/ask",
                json={"question": "What's the latest Python release?"}
            )
            data = resp.json()
            assert resp.status_code == 200
            # Response should contain web-sourced information
            assert "python" in data["response"].lower()

    async def test_mcp_web_search_tool(self):
        """MCP web_search tool returns formatted results."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8309/mcp",
                json={
                    "method": "tools/call",
                    "params": {
                        "name": "web_search",
                        "arguments": {"query": "FastAPI latest version"}
                    }
                }
            )
            data = resp.json()
            assert "results" in str(data) or "FastAPI" in str(data)

    async def test_searxng_health(self):
        """SearXNG container is reachable."""
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8888/healthz")
            assert resp.status_code == 200
```

### Smoke Tests

```bash
# Quick verification after deployment
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'SearXNG: {len(d[\"results\"])} results')"
curl -s -X POST http://localhost:8309/api/echo/search/web -H "Content-Type: application/json" -d '{"query":"test"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Echo Brain search: {len(d[\"results\"])} results')"
curl -s -X POST http://localhost:8309/mcp -H "Content-Type: application/json" -d '{"method":"tools/call","params":{"name":"web_search","arguments":{"query":"hello world"}}}' | python3 -c "import sys,json; print('MCP tool: OK' if 'result' in json.load(sys.stdin) else 'FAIL')"
```

---

## Estimated Token/Compute Costs

### Per Query (with web search)

| Component | Tokens | Time | Cost |
|---|---|---|---|
| Query classification | ~200 | 0.5s | Free (local Ollama) |
| SearXNG search | 0 | 0.3-1s | Free (self-hosted) |
| Page fetching (5 pages) | 0 | 1-3s | Free (bandwidth only) |
| Content chunking | 0 | 0.1s | Free (CPU) |
| Context compilation | ~3,000-5,000 | 0.1s | Free (local) |
| LLM response generation | ~2,000-4,000 | 3-8s | Free (local Ollama) |
| **Total per query** | **~5,000-9,000** | **5-12s** | **$0.00** |

### If using Claude API instead of Ollama (optional, for complex research)

| Component | Tokens | Cost |
|---|---|---|
| Input (context + query) | ~8,000 | $0.024 (Haiku) / $0.24 (Sonnet) |
| Output (response) | ~2,000 | $0.01 (Haiku) / $0.06 (Sonnet) |
| **Total per query** | **~10,000** | **$0.034 (Haiku) / $0.30 (Sonnet)** |

### Infrastructure Cost

| Resource | Cost |
|---|---|
| SearXNG Docker container | ~100MB RAM, negligible CPU |
| Brave Search API (fallback) | Free for 2,000 queries/month, $3/1K after |
| Disk for SearXNG image | ~150MB |
| **Monthly total** | **$0.00** (unless Brave fallback exceeds free tier) |

---

## Deployment Steps

```bash
# 1. Create SearXNG config directory
mkdir -p /opt/tower-echo-brain/config/searxng
mkdir -p /opt/tower-echo-brain/docker

# 2. Write SearXNG settings.yml (from Component 1 above)

# 3. Write docker-compose.searxng.yml (from Component 1 above)

# 4. Start SearXNG
cd /opt/tower-echo-brain/docker
docker compose -f docker-compose.searxng.yml up -d

# 5. Verify SearXNG is running
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -m json.tool | head -20

# 6. Add SearchService to Echo Brain codebase
# (copy src/services/search_service.py from this spec)

# 7. Add search router
# (copy src/api/endpoints/search_router.py from this spec)

# 8. Register router in main.py
# In src/main.py, add:
#   from src.api.endpoints.search_router import router as search_router
#   app.include_router(search_router)

# 9. Add web_search MCP tool
# (add to mcp_server/stdio_server.py from Component 4)

# 10. Update .env
echo "SEARXNG_URL=http://localhost:8888" >> /opt/tower-echo-brain/.env
echo "WEB_SEARCH_ENABLED=true" >> /opt/tower-echo-brain/.env

# 11. Restart Echo Brain
sudo systemctl restart tower-echo-brain

# 12. Run smoke tests
# (from Test Plan above)

# 13. (Optional) Create systemd service for SearXNG auto-start
sudo cp tower-searxng.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tower-searxng
```

---

## Rollback Plan

If web search causes issues:

```bash
# Disable web search (no code changes needed)
echo "WEB_SEARCH_ENABLED=false" >> /opt/tower-echo-brain/.env
sudo systemctl restart tower-echo-brain

# Or stop SearXNG entirely
docker stop tower-searxng

# Echo Brain continues working normally — web search is additive, not replacing anything
```

---

## Success Criteria

1. `curl "http://localhost:8888/search?q=python+fastapi&format=json"` returns 10+ results
2. `POST /api/echo/search/web` returns results with titles, URLs, and snippets
3. MCP `web_search` tool works from Claude Code
4. `/api/echo/ask` automatically includes web results for current-events queries
5. Responses cite web sources with `[W1]`, `[W2]` notation
6. Circuit breaker correctly fails over to Brave when SearXNG is down
7. Cache returns results within 1ms for repeated queries within 15-minute window
8. No regression in existing memory-only queries (web search is additive)
