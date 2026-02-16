"""
Knowledge Graph API Endpoints
Exposes graph traversal, path finding, and statistics.
"""
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/graph", tags=["graph"])


class PathRequest(BaseModel):
    from_entity: str
    to_entity: str


@router.get("/related/{entity}")
async def get_related(
    entity: str,
    depth: int = Query(default=2, ge=1, le=3),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Get entities related to the given entity via graph traversal."""
    from src.core.graph_engine import get_graph_engine

    engine = get_graph_engine()
    await engine._ensure_loaded()
    related = engine.get_related(entity, depth=depth, max_results=limit)
    return {
        "entity": entity,
        "depth": depth,
        "results": related,
        "count": len(related),
    }


@router.post("/path")
async def find_path(request: PathRequest):
    """Find shortest path between two entities."""
    from src.core.graph_engine import get_graph_engine

    engine = get_graph_engine()
    await engine._ensure_loaded()
    path = engine.find_path(request.from_entity, request.to_entity)
    return {
        "from": request.from_entity,
        "to": request.to_entity,
        "path": path,
        "hops": len(path),
    }


@router.get("/neighborhood/{entity}")
async def get_neighborhood(
    entity: str,
    hops: int = Query(default=2, ge=1, le=3),
):
    """Get ego subgraph stats around an entity."""
    from src.core.graph_engine import get_graph_engine

    engine = get_graph_engine()
    await engine._ensure_loaded()
    return engine.get_neighborhood(entity, hops=hops)


@router.get("/stats")
async def graph_stats():
    """Graph-level statistics: nodes, edges, components, density."""
    from src.core.graph_engine import get_graph_engine

    engine = get_graph_engine()
    await engine._ensure_loaded()
    return engine.get_stats()


@router.post("/refresh")
async def refresh_graph():
    """Trigger an incremental (or full if overdue) graph refresh."""
    from src.core.graph_engine import get_graph_engine

    engine = get_graph_engine()
    await engine.refresh()
    return {"status": "refreshed", **engine.get_stats()}
