"""Web search service with SearXNG primary and Brave API fallback."""

import asyncio
import hashlib
import html as html_module
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import httpx

from src.resilience.service_breakers import ServiceCircuitBreaker

logger = logging.getLogger(__name__)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8890")
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"


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
    MAX_CONTENT_CHARS = 15_000
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200

    def __init__(
        self,
        searxng_url: Optional[str] = None,
        brave_api_key: Optional[str] = None,
    ):
        self.searxng_url = searxng_url or SEARXNG_URL
        self.brave_api_key = brave_api_key
        self._cache: dict[str, tuple[SearchResponse, datetime]] = {}
        self._searxng_breaker = ServiceCircuitBreaker(
            name="searxng",
            failure_threshold=3,
            recovery_timeout_seconds=60.0,
            half_open_max_attempts=2,
        )

    async def search(
        self,
        query: str,
        num_results: int = 10,
        categories: Optional[list[str]] = None,
        time_range: Optional[str] = None,
    ) -> SearchResponse:
        """Search the web. Tries SearXNG first, falls back to Brave."""
        if not WEB_SEARCH_ENABLED:
            raise RuntimeError("Web search is disabled")

        cache_key = self._cache_key(query, num_results, categories)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Try SearXNG first
        if self._searxng_breaker.state != "open":
            try:
                response = await self._search_searxng(
                    query, num_results, categories, time_range
                )
                self._searxng_breaker._on_success()
                self._set_cached(cache_key, response)
                return response
            except Exception as e:
                logger.warning(f"SearXNG search failed: {e}")
                self._searxng_breaker._on_failure()

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
    ) -> list[WebDocument]:
        """Search, fetch top pages, clean and chunk content."""
        search_response = await self.search(query, num_results=max_pages * 2)

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
        categories: Optional[list[str]],
        time_range: Optional[str],
    ) -> SearchResponse:
        """Search via SearXNG JSON API."""
        start = time.monotonic()

        params: dict = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        if categories:
            params["categories"] = ",".join(categories)
        if time_range:
            params["time_range"] = time_range

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.searxng_url}/search", params=params)
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
                raw_html = resp.text

        content = self._html_to_text(raw_html)
        content = content[:self.MAX_CONTENT_CHARS]
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
    def _html_to_text(raw_html: str) -> str:
        """Convert HTML to clean text."""
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html_module.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
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

    def _get_cached(self, key: str) -> Optional[SearchResponse]:
        if key in self._cache:
            response, cached_at = self._cache[key]
            if datetime.utcnow() - cached_at < self.CACHE_TTL:
                response.cached = True
                return response
            del self._cache[key]
        return None

    def _set_cached(self, key: str, response: SearchResponse):
        self._cache[key] = (response, datetime.utcnow())
        now = datetime.utcnow()
        expired = [k for k, (_, t) in self._cache.items() if now - t > self.CACHE_TTL]
        for k in expired:
            del self._cache[k]


# Singleton instance
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get or create the singleton SearchService."""
    global _search_service
    if _search_service is None:
        brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
        _search_service = SearchService(brave_api_key=brave_key)
    return _search_service
