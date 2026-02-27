"""Web search API endpoints — SearXNG + Brave fallback."""

import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["web-search"])


class WebSearchRequest(BaseModel):
    query: str
    num_results: int = 10
    categories: Optional[list[str]] = None
    time_range: Optional[str] = None
    fetch_content: bool = False
    max_pages: int = 5


class WebSearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str
    source_engine: str
    position: int


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResultItem]
    total_results: int
    search_time_ms: float
    source: str
    cached: bool


class WebDocumentItem(BaseModel):
    url: str
    title: str
    chunks: list[str]
    char_count: int


class SearchAndFetchResponse(BaseModel):
    query: str
    documents: list[WebDocumentItem]
    total_documents: int


@router.post("/web", response_model=WebSearchResponse)
async def web_search(request: WebSearchRequest):
    """Search the web via SearXNG (self-hosted) with Brave API fallback."""
    from src.services.search_service import get_search_service

    service = get_search_service()
    try:
        if request.fetch_content:
            # Also fetch and chunk page content
            docs = await service.search_and_fetch(
                query=request.query,
                max_pages=request.max_pages,
            )
            # Still return the search results format
            response = await service.search(
                query=request.query,
                num_results=request.num_results,
                categories=request.categories,
                time_range=request.time_range,
            )
        else:
            response = await service.search(
                query=request.query,
                num_results=request.num_results,
                categories=request.categories,
                time_range=request.time_range,
            )

        return WebSearchResponse(
            query=response.query,
            results=[WebSearchResultItem(
                title=r.title, url=r.url, snippet=r.snippet,
                source_engine=r.source_engine, position=r.position,
            ) for r in response.results],
            total_results=response.total_results,
            search_time_ms=response.search_time_ms,
            source=response.source,
            cached=response.cached,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        raise HTTPException(status_code=503, detail=f"Search unavailable: {e}")


@router.get("/web")
async def web_search_get(
    q: str = Query(..., description="Search query"),
    n: int = Query(10, ge=1, le=50, description="Number of results"),
    categories: Optional[str] = Query(None, description="Comma-separated categories"),
    time_range: Optional[str] = Query(None, description="day, week, month, year"),
):
    """Quick web search via GET."""
    cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else None
    return await web_search(WebSearchRequest(
        query=q, num_results=n, categories=cats, time_range=time_range,
    ))


@router.post("/web/fetch", response_model=SearchAndFetchResponse)
async def search_and_fetch(request: WebSearchRequest):
    """Search the web and fetch+chunk top page content for RAG."""
    from src.services.search_service import get_search_service

    service = get_search_service()
    try:
        docs = await service.search_and_fetch(
            query=request.query,
            max_pages=request.max_pages,
        )
        return SearchAndFetchResponse(
            query=request.query,
            documents=[WebDocumentItem(
                url=d.url, title=d.title,
                chunks=d.chunks, char_count=d.char_count,
            ) for d in docs],
            total_documents=len(docs),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Search and fetch failed: {e}")
        raise HTTPException(status_code=503, detail=f"Search unavailable: {e}")
