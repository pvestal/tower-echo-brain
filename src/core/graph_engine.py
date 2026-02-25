"""
Knowledge Graph Engine — NetworkX-based graph traversal over Echo Brain facts.
Lazy-loaded: graph is built on first query, then cached.
Incremental refresh: only new facts are added; full rebuild every 24h.
"""
import logging
import asyncpg
import networkx as nx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("echo.core.graph_engine")

_instance: Optional["GraphEngine"] = None


def get_graph_engine() -> "GraphEngine":
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = GraphEngine()
    return _instance


class GraphEngine:
    """NetworkX-backed knowledge graph over Echo Brain facts + graph_edges."""

    def __init__(self):
        self._graph: Optional[nx.DiGraph] = None
        self._db_url: Optional[str] = None
        self._last_refresh: Optional[datetime] = None
        self._last_full_rebuild: Optional[datetime] = None
        self._full_rebuild_interval = timedelta(hours=24)

    def initialize(self, db_url: str):
        """Store db_url for lazy loading. Does NOT load graph yet."""
        self._db_url = db_url
        logger.info("GraphEngine initialized (lazy — will load on first query)")

    async def _ensure_loaded(self):
        """Load graph on first access or if stale."""
        if self._graph is None:
            await self._full_load()
        elif (
            self._last_full_rebuild
            and datetime.now(timezone.utc) - self._last_full_rebuild > self._full_rebuild_interval
        ):
            await self._full_load()
        elif self._last_refresh:
            await self._incremental_load()

    async def _full_load(self):
        """Load all facts and edges into a fresh graph."""
        if not self._db_url:
            logger.warning("GraphEngine: no db_url configured, skipping load")
            return

        g = nx.DiGraph()
        try:
            conn = await asyncpg.connect(self._db_url)
            try:
                # Load facts as nodes + edges
                rows = await conn.fetch("""
                    SELECT subject, predicate, object, confidence
                    FROM facts
                    WHERE subject IS NOT NULL AND predicate IS NOT NULL AND object IS NOT NULL
                """)
                for row in rows:
                    subj = row["subject"].lower().strip()
                    obj = row["object"].lower().strip()
                    pred = row["predicate"].strip()
                    conf = float(row["confidence"])

                    g.add_node(subj)
                    g.add_node(obj)
                    # Use (subj, obj, pred) to allow multiple edge types
                    if g.has_edge(subj, obj) and g[subj][obj].get("predicate") == pred:
                        # Update confidence if this edge already exists
                        g[subj][obj]["confidence"] = max(g[subj][obj]["confidence"], conf)
                    else:
                        g.add_edge(subj, obj, predicate=pred, confidence=conf)

                # Load explicit graph_edges if table exists
                try:
                    edges = await conn.fetch("""
                        SELECT from_entity, to_entity, relation_type, confidence
                        FROM graph_edges
                    """)
                    for edge in edges:
                        fe = edge["from_entity"].lower().strip()
                        te = edge["to_entity"].lower().strip()
                        g.add_node(fe)
                        g.add_node(te)
                        g.add_edge(
                            fe, te,
                            predicate=edge["relation_type"],
                            confidence=float(edge.get("confidence", 0.5)),
                        )
                except Exception:
                    pass  # Table may not exist

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"GraphEngine full load failed: {e}")
            if g.number_of_nodes() == 0:
                return

        self._graph = g
        self._last_refresh = datetime.now(timezone.utc)
        self._last_full_rebuild = datetime.now(timezone.utc)
        logger.info(f"GraphEngine loaded: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    async def _incremental_load(self):
        """Add only facts created since last refresh."""
        if not self._db_url or not self._graph or not self._last_refresh:
            return

        try:
            conn = await asyncpg.connect(self._db_url)
            try:
                rows = await conn.fetch("""
                    SELECT subject, predicate, object, confidence
                    FROM facts
                    WHERE created_at > $1
                      AND subject IS NOT NULL AND predicate IS NOT NULL AND object IS NOT NULL
                """, self._last_refresh.replace(tzinfo=None))

                added = 0
                for row in rows:
                    subj = row["subject"].lower().strip()
                    obj = row["object"].lower().strip()
                    pred = row["predicate"].strip()
                    conf = float(row["confidence"])

                    self._graph.add_node(subj)
                    self._graph.add_node(obj)
                    self._graph.add_edge(subj, obj, predicate=pred, confidence=conf)
                    added += 1

                if added:
                    logger.info(f"GraphEngine incremental: added {added} edges")

            finally:
                await conn.close()

        except Exception as e:
            logger.debug(f"GraphEngine incremental load failed: {e}")

        self._last_refresh = datetime.now(timezone.utc)

    async def refresh(self):
        """Force incremental refresh, or full rebuild if overdue."""
        if (
            self._last_full_rebuild
            and datetime.now(timezone.utc) - self._last_full_rebuild > self._full_rebuild_interval
        ):
            await self._full_load()
        else:
            await self._incremental_load()

    def get_related(self, entity: str, depth: int = 2, max_results: int = 50) -> List[Dict]:
        """BFS traversal: find entities related within N hops."""
        if not self._graph:
            return []

        entity_lower = entity.lower().strip()
        if entity_lower not in self._graph:
            # Try partial match
            candidates = [n for n in self._graph.nodes if entity_lower in n]
            if not candidates:
                return []
            entity_lower = candidates[0]

        results = []
        visited = {entity_lower}
        queue = [(entity_lower, 0)]

        while queue and len(results) < max_results:
            node, d = queue.pop(0)
            if d >= depth:
                continue

            # Outgoing edges
            for _, neighbor, data in self._graph.edges(node, data=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    results.append({
                        "from": node,
                        "to": neighbor,
                        "predicate": data.get("predicate", "related_to"),
                        "confidence": data.get("confidence", 0.5),
                        "depth": d + 1,
                    })
                    queue.append((neighbor, d + 1))

            # Incoming edges
            for predecessor, _, data in self._graph.in_edges(node, data=True):
                if predecessor not in visited:
                    visited.add(predecessor)
                    results.append({
                        "from": predecessor,
                        "to": node,
                        "predicate": data.get("predicate", "related_to"),
                        "confidence": data.get("confidence", 0.5),
                        "depth": d + 1,
                    })
                    queue.append((predecessor, d + 1))

        return results[:max_results]

    def find_path(self, entity_a: str, entity_b: str) -> List[Dict]:
        """Find shortest path between two entities."""
        if not self._graph:
            return []

        a = entity_a.lower().strip()
        b = entity_b.lower().strip()

        # Allow partial matching
        if a not in self._graph:
            candidates = [n for n in self._graph.nodes if a in n]
            a = candidates[0] if candidates else a
        if b not in self._graph:
            candidates = [n for n in self._graph.nodes if b in n]
            b = candidates[0] if candidates else b

        try:
            # Use undirected view for path finding
            path = nx.shortest_path(self._graph.to_undirected(), a, b)
            result = []
            for i in range(len(path) - 1):
                edge_data = self._graph.get_edge_data(path[i], path[i + 1]) or {}
                if not edge_data:
                    edge_data = self._graph.get_edge_data(path[i + 1], path[i]) or {}
                result.append({
                    "from": path[i],
                    "to": path[i + 1],
                    "predicate": edge_data.get("predicate", "connected_to"),
                    "confidence": edge_data.get("confidence", 0.5),
                })
            return result
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_neighborhood(self, entity: str, hops: int = 2) -> Dict:
        """Get ego subgraph stats around an entity."""
        if not self._graph:
            return {"entity": entity, "found": False}

        entity_lower = entity.lower().strip()
        if entity_lower not in self._graph:
            candidates = [n for n in self._graph.nodes if entity_lower in n]
            if not candidates:
                return {"entity": entity, "found": False}
            entity_lower = candidates[0]

        try:
            ego = nx.ego_graph(self._graph, entity_lower, radius=hops, undirected=True)
            return {
                "entity": entity_lower,
                "found": True,
                "nodes": list(ego.nodes)[:100],
                "node_count": ego.number_of_nodes(),
                "edge_count": ego.number_of_edges(),
                "hops": hops,
            }
        except Exception:
            return {"entity": entity, "found": False}

    def get_entity_importance(self, entity: str) -> float:
        """Return a 0-1 importance score based on degree centrality.

        Hub entities (many connections) score higher. Returns 0 if entity
        is not in the graph or graph is not loaded.
        """
        if not self._graph:
            return 0.0
        entity_lower = entity.lower().strip()
        if entity_lower not in self._graph:
            # Try partial match
            candidates = [n for n in self._graph.nodes if entity_lower in n]
            if not candidates:
                return 0.0
            entity_lower = candidates[0]
        # Degree centrality: fraction of nodes this entity connects to
        try:
            centrality = nx.degree_centrality(self._graph)
            return centrality.get(entity_lower, 0.0)
        except Exception:
            return 0.0

    def get_connected_entities(self, entity: str, max_hops: int = 1) -> List[Dict]:
        """Lightweight wrapper: get directly connected entities with predicates.

        Returns list of {"entity": str, "predicate": str, "direction": "out"|"in"}.
        """
        if not self._graph:
            return []
        entity_lower = entity.lower().strip()
        if entity_lower not in self._graph:
            candidates = [n for n in self._graph.nodes if entity_lower in n]
            if not candidates:
                return []
            entity_lower = candidates[0]

        results = []
        visited = {entity_lower}
        queue = [(entity_lower, 0)]

        while queue:
            node, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            for _, neighbor, data in self._graph.edges(node, data=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    results.append({
                        "entity": neighbor,
                        "predicate": data.get("predicate", "related_to"),
                        "direction": "out",
                    })
                    if depth + 1 < max_hops:
                        queue.append((neighbor, depth + 1))
            for predecessor, _, data in self._graph.in_edges(node, data=True):
                if predecessor not in visited:
                    visited.add(predecessor)
                    results.append({
                        "entity": predecessor,
                        "predicate": data.get("predicate", "related_to"),
                        "direction": "in",
                    })
                    if depth + 1 < max_hops:
                        queue.append((predecessor, depth + 1))
        return results

    def get_stats(self) -> Dict:
        """Graph-level statistics."""
        if not self._graph:
            return {
                "loaded": False,
                "nodes": 0,
                "edges": 0,
                "components": 0,
                "density": 0,
            }

        undirected = self._graph.to_undirected()
        return {
            "loaded": True,
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "components": nx.number_connected_components(undirected),
            "density": round(nx.density(self._graph), 6),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "last_full_rebuild": self._last_full_rebuild.isoformat() if self._last_full_rebuild else None,
        }
