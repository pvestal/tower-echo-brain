"""
Deep Research Engine for Echo Brain.

Takes complex questions, decomposes into sub-questions, searches web + memory + facts
in parallel using the full intelligence stack (ParallelRetriever, graph engine,
domain classification), discovers connections between entities, evaluates gaps
intelligently, and synthesizes a cited report. Stores findings back into Echo Brain
for future reference.
"""
import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — depth-dependent, not magic numbers
# ---------------------------------------------------------------------------

DEPTH_CONFIG = {
    "quick":    {"max_iterations": 1, "llm_timeout": 60,  "search_timeout": 60,  "total_timeout": 120, "web_results": 5, "memory_results": 5, "fact_results": 5},
    "standard": {"max_iterations": 2, "llm_timeout": 90,  "search_timeout": 90,  "total_timeout": 240, "web_results": 8, "memory_results": 8, "fact_results": 8},
    "deep":     {"max_iterations": 3, "llm_timeout": 120, "search_timeout": 120, "total_timeout": 420, "web_results": 10, "memory_results": 10, "fact_results": 10},
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SubQuestion:
    query: str
    purpose: str
    domain: str = ""  # populated by domain classifier


@dataclass
class Source:
    ref: str          # e.g. W1, M3, F2, G1
    source_type: str  # web, memory, fact, graph
    title: str
    snippet: str
    url: str = ""
    score: float = 0.0


@dataclass
class FindingSet:
    sub_question: str
    web_results: list[dict] = field(default_factory=list)
    memory_results: list[dict] = field(default_factory=list)
    fact_results: list[dict] = field(default_factory=list)
    graph_results: list[dict] = field(default_factory=list)
    retriever_confidence: float = 0.0
    domain: str = ""


@dataclass
class Connection:
    entity_a: str
    entity_b: str
    relationship: str
    path: list[str] = field(default_factory=list)
    strength: float = 0.0


@dataclass
class Evaluation:
    sufficient: bool
    confidence: float
    gaps: list[str] = field(default_factory=list)
    follow_up_queries: list[SubQuestion] = field(default_factory=list)
    connections_found: list[Connection] = field(default_factory=list)


@dataclass
class ResearchReport:
    answer: str
    sources: list[Source] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    entities_discovered: list[str] = field(default_factory=list)
    iterations: int = 0
    total_sources_consulted: int = 0


class JobStatus(str, Enum):
    pending = "pending"
    decomposing = "decomposing"
    searching = "searching"
    connecting = "connecting"
    evaluating = "evaluating"
    synthesizing = "synthesizing"
    storing = "storing"
    complete = "complete"
    failed = "failed"


@dataclass
class ResearchJob:
    id: str
    question: str
    depth: str
    status: str = JobStatus.pending
    progress: dict = field(default_factory=dict)
    report: Optional[ResearchReport] = None
    error_message: Optional[str] = None
    iterations: int = 0
    sources_consulted: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_time_ms: float = 0.0
    _follow_ups: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ResearchEngine:
    def __init__(self):
        self._jobs: dict[str, ResearchJob] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._table_ready = False

    # -- public API ----------------------------------------------------------

    def start_research(self, question: str, depth: str = "standard") -> ResearchJob:
        job_id = str(uuid.uuid4())
        job = ResearchJob(
            id=job_id,
            question=question,
            depth=depth,
            started_at=datetime.utcnow(),
        )
        self._jobs[job_id] = job
        self._queues[job_id] = asyncio.Queue()
        asyncio.create_task(self._run_research(job))
        return job

    async def get_job(self, job_id: str) -> Optional[ResearchJob]:
        if job_id in self._jobs:
            return self._jobs[job_id]
        try:
            await self._ensure_table()
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute("SET search_path TO public")
                row = await conn.fetchrow(
                    "SELECT * FROM research_jobs WHERE id = $1", uuid.UUID(job_id)
                )
                if row:
                    return self._row_to_job(row)
        except Exception as e:
            logger.error(f"DB lookup failed for job {job_id}: {e}")
        return None

    async def get_history(self, limit: int = 20) -> list[dict]:
        try:
            await self._ensure_table()
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute("SET search_path TO public")
                rows = await conn.fetch(
                    """SELECT id, question, depth, status, iterations,
                              sources_consulted, started_at, completed_at, total_time_ms
                       FROM research_jobs ORDER BY started_at DESC LIMIT $1""",
                    limit,
                )
                return [
                    {
                        "id": str(r["id"]),
                        "question": r["question"],
                        "depth": r["depth"],
                        "status": r["status"],
                        "iterations": r["iterations"],
                        "sources_consulted": r["sources_consulted"],
                        "started_at": r["started_at"].isoformat() if r["started_at"] else None,
                        "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
                        "total_time_ms": r["total_time_ms"],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"get_history error: {e}")
            return []

    async def stream_progress(self, job_id: str):
        q = self._queues.get(job_id)
        if not q:
            yield {"event": "error", "data": {"message": "Job not found"}}
            return
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=15)
                yield event
                if event.get("event") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield {"event": "keepalive", "data": {}}

    # -- main pipeline -------------------------------------------------------

    async def _run_research(self, job: ResearchJob):
        t0 = time.time()
        cfg = DEPTH_CONFIG.get(job.depth, DEPTH_CONFIG["standard"])

        try:
            # 1. DECOMPOSE — break question into domain-classified sub-questions
            job.status = JobStatus.decomposing
            self._emit(job.id, "decomposing", {"status": "Breaking question into sub-questions"})
            sub_questions = await asyncio.wait_for(
                self._decompose(job.question), timeout=cfg["llm_timeout"]
            )
            self._emit(job.id, "decomposing", {
                "sub_questions": [{"query": sq.query, "purpose": sq.purpose, "domain": sq.domain}
                                  for sq in sub_questions]
            })

            all_findings: list[FindingSet] = []
            all_sources: list[Source] = []
            all_entities: set[str] = set()
            all_connections: list[Connection] = []

            for iteration in range(cfg["max_iterations"]):
                job.iterations = iteration + 1

                # 2. SEARCH — parallel across web + retriever + graph per sub-question
                job.status = JobStatus.searching
                queries_to_search = sub_questions if iteration == 0 else job._follow_ups
                if not queries_to_search:
                    break

                for sq in queries_to_search:
                    self._emit(job.id, "searching", {
                        "sub_question": sq.query, "iteration": iteration + 1,
                    })

                findings = await asyncio.wait_for(
                    self._search_all(queries_to_search, cfg), timeout=cfg["search_timeout"]
                )
                all_findings.extend(findings)

                for f in findings:
                    self._emit(job.id, "searching", {
                        "sub_question": f.sub_question, "iteration": iteration + 1,
                        "web": len(f.web_results), "memory": len(f.memory_results),
                        "facts": len(f.fact_results), "graph": len(f.graph_results),
                        "confidence": round(f.retriever_confidence, 2), "domain": f.domain,
                    })

                # Build source list and extract entities
                all_sources = self._build_sources(all_findings)
                job.sources_consulted = len(all_sources)
                new_entities = self._extract_entities(all_findings)
                all_entities.update(new_entities)

                # 3. CONNECT — use graph engine to find relationships between entities
                if new_entities and len(new_entities) >= 2:
                    job.status = JobStatus.connecting
                    self._emit(job.id, "connecting", {
                        "entities": list(new_entities)[:20],
                        "iteration": iteration + 1,
                    })
                    connections = await self._discover_connections(new_entities)
                    all_connections.extend(connections)
                    if connections:
                        self._emit(job.id, "connecting", {
                            "found": len(connections),
                            "connections": [
                                {"a": c.entity_a, "b": c.entity_b, "rel": c.relationship}
                                for c in connections[:10]
                            ],
                        })

                # 4. EVALUATE — intelligent gap analysis with connection awareness
                if iteration < cfg["max_iterations"] - 1:
                    job.status = JobStatus.evaluating
                    evaluation = await asyncio.wait_for(
                        self._evaluate(job.question, all_findings, all_connections,
                                       all_entities, iteration + 1),
                        timeout=cfg["llm_timeout"]
                    )
                    self._emit(job.id, "evaluating", {
                        "iteration": iteration + 1,
                        "sufficient": evaluation.sufficient,
                        "confidence": evaluation.confidence,
                        "gaps": evaluation.gaps,
                    })

                    if evaluation.sufficient:
                        break

                    if evaluation.follow_up_queries:
                        job._follow_ups = evaluation.follow_up_queries
                        self._emit(job.id, "follow_up", {
                            "queries": [{"query": sq.query, "purpose": sq.purpose}
                                        for sq in evaluation.follow_up_queries]
                        })
                    else:
                        break

            # 5. SYNTHESIZE — reasoning-informed report with connections
            job.status = JobStatus.synthesizing
            self._emit(job.id, "synthesizing", {
                "sources_count": len(all_sources),
                "connections_count": len(all_connections),
                "entities_count": len(all_entities),
            })
            report = await asyncio.wait_for(
                self._synthesize(job.question, all_findings, all_sources,
                                 all_connections, all_entities),
                timeout=cfg["llm_timeout"]
            )
            report.iterations = job.iterations
            report.total_sources_consulted = job.sources_consulted
            report.sub_questions = [sq.query for sq in sub_questions]
            report.entities_discovered = list(all_entities)[:50]
            report.connections = all_connections

            job.report = report
            job.status = JobStatus.complete
            job.completed_at = datetime.utcnow()
            job.total_time_ms = (time.time() - t0) * 1000

            # 6. STORE — persist findings back into Echo Brain for future reference
            job.status = JobStatus.storing
            self._emit(job.id, "storing", {"status": "Saving findings to Echo Brain"})
            await self._store_findings(job)
            job.status = JobStatus.complete

            self._emit(job.id, "complete", {
                "report": self._report_to_dict(report),
            })

        except asyncio.TimeoutError:
            elapsed = (time.time() - t0) * 1000
            job.status = JobStatus.failed
            job.error_message = f"Research timed out after {elapsed/1000:.0f}s (limit: {cfg['total_timeout']}s)"
            job.completed_at = datetime.utcnow()
            job.total_time_ms = elapsed
            self._emit(job.id, "error", {"message": job.error_message})
        except Exception as e:
            logger.exception(f"Research failed for job {job.id}")
            job.status = JobStatus.failed
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            job.total_time_ms = (time.time() - t0) * 1000
            self._emit(job.id, "error", {"message": str(e)})

        try:
            await self._persist_job(job)
        except Exception as e:
            logger.error(f"Failed to persist job {job.id}: {e}")

    # -- decompose -----------------------------------------------------------

    async def _decompose(self, question: str) -> list[SubQuestion]:
        from src.services.llm_service import get_llm_service
        llm = get_llm_service()

        prompt = f"""You are a research strategist. Break this question into 3-6 focused sub-questions.

For each sub-question, think about:
- What DIFFERENT angle does this explore?
- What type of source would answer it best? (web for current info, memory for personal history, facts for known relationships)
- What entities or names should we search for specifically?

Question: {question}

Return ONLY a JSON array:
[
  {{"query": "specific focused search query", "purpose": "what gap this fills"}}
]

Make queries specific and searchable — use names, identifiers, exact terms. Not vague."""

        resp = await llm.generate(
            prompt=prompt,
            model="deepseek-r1:8b",
            temperature=0.3,
            max_tokens=1024,
        )
        sub_qs = self._parse_sub_questions(resp.content)

        # Classify each sub-question's domain using the domain classifier
        try:
            from src.context_assembly.classifier import DomainClassifier
            classifier = DomainClassifier()
            for sq in sub_qs:
                result = classifier.classify(sq.query)
                sq.domain = result.get("domain", "GENERAL") if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.warning(f"Domain classification failed: {e}")

        return sub_qs

    def _parse_sub_questions(self, text: str) -> list[SubQuestion]:
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        for attempt in [cleaned, None]:
            source = attempt if attempt else cleaned
            try:
                data = json.loads(source)
                if isinstance(data, list):
                    return [SubQuestion(query=item["query"], purpose=item.get("purpose", ""))
                            for item in data if isinstance(item, dict) and "query" in item]
            except (json.JSONDecodeError, TypeError):
                pass
            # Regex fallback
            match = re.search(r'\[.*\]', source, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    if isinstance(data, list):
                        return [SubQuestion(query=item["query"], purpose=item.get("purpose", ""))
                                for item in data if isinstance(item, dict) and "query" in item]
                except (json.JSONDecodeError, TypeError):
                    pass
            break

        logger.warning("Failed to parse sub-questions, using original question")
        return [SubQuestion(query=cleaned[:200], purpose="original question")]

    # -- search (uses full intelligence stack) --------------------------------

    async def _search_all(self, sub_questions: list[SubQuestion], cfg: dict) -> list[FindingSet]:
        tasks = [self._search_one(sq, cfg) for sq in sub_questions]
        return await asyncio.gather(*tasks)

    async def _search_one(self, sq: SubQuestion, cfg: dict) -> FindingSet:
        finding = FindingSet(sub_question=sq.query, domain=sq.domain)

        web_task = self._search_web(sq.query, cfg["web_results"])
        retriever_task = self._search_retriever(sq.query, cfg["memory_results"])
        facts_task = self._search_facts(sq.query, cfg["fact_results"])

        results = await asyncio.gather(web_task, retriever_task, facts_task, return_exceptions=True)

        if not isinstance(results[0], Exception):
            finding.web_results = results[0]
        else:
            logger.warning(f"Web search failed for '{sq.query}': {results[0]}")

        if not isinstance(results[1], Exception):
            ret_data = results[1]
            finding.memory_results = ret_data.get("memories", [])
            finding.fact_results.extend(ret_data.get("facts", []))
            finding.graph_results = ret_data.get("graph", [])
            finding.retriever_confidence = ret_data.get("confidence", 0.0)
            finding.domain = ret_data.get("domain", sq.domain)
        else:
            logger.warning(f"Retriever failed for '{sq.query}': {results[1]}")

        if not isinstance(results[2], Exception):
            finding.fact_results.extend(results[2])
        else:
            logger.warning(f"Facts search failed for '{sq.query}': {results[2]}")

        return finding

    async def _search_web(self, query: str, limit: int = 5) -> list[dict]:
        try:
            from src.services.search_service import get_search_service
            svc = get_search_service()
            resp = await svc.search(query=query, num_results=limit)
            return [
                {"title": r.title, "url": r.url, "snippet": r.snippet}
                for r in resp.results[:limit]
            ]
        except Exception as e:
            logger.warning(f"Web search error: {e}")
            return []

    async def _search_retriever(self, query: str, limit: int = 5) -> dict:
        """Use ParallelRetriever — gets domain filtering, graph enrichment,
        confidence gating, time-decay boost automatically."""
        try:
            from src.context_assembly.retriever import ParallelRetriever
            retriever = ParallelRetriever()
            result = await retriever.retrieve(query, max_results=limit)

            memories = []
            facts = []
            graph = []

            for src in result.get("sources", []):
                entry = {
                    "content": src.get("text", src.get("content", ""))[:300],
                    "score": src.get("relevance_score", src.get("score", 0)),
                    "source_type": src.get("source_type", "unknown"),
                    "source": src.get("collection", src.get("source", "")),
                }
                st = entry["source_type"]
                if st in ("qdrant_vectors", "memory", "vector"):
                    memories.append(entry)
                elif st in ("facts_table", "fact", "core"):
                    facts.append(entry)
                elif st in ("graph_edge", "graph"):
                    graph.append(entry)
                else:
                    memories.append(entry)

            return {
                "memories": memories,
                "facts": facts,
                "graph": graph,
                "confidence": result.get("confidence", 0.0),
                "domain": result.get("domain", ""),
            }
        except Exception as e:
            logger.warning(f"ParallelRetriever failed, falling back to UnifiedKnowledge: {e}")
            # Fallback to direct search
            return await self._search_retriever_fallback(query, limit)

    async def _search_retriever_fallback(self, query: str, limit: int = 5) -> dict:
        """Fallback: use UnifiedKnowledgeLayer directly."""
        try:
            from src.core.unified_knowledge import get_unified_knowledge
            uk = get_unified_knowledge()
            vectors = await uk.search_vectors(query, limit=limit)
            return {
                "memories": [
                    {"content": r.content, "score": r.confidence,
                     "source_type": "memory", "source": r.metadata.get("source", "")}
                    for r in vectors
                ],
                "facts": [],
                "graph": [],
                "confidence": max((r.confidence for r in vectors), default=0.0),
                "domain": "",
            }
        except Exception as e:
            logger.warning(f"UnifiedKnowledge fallback also failed: {e}")
            return {"memories": [], "facts": [], "graph": [], "confidence": 0.0, "domain": ""}

    async def _search_facts(self, query: str, limit: int = 5) -> list[dict]:
        try:
            from src.core.unified_knowledge import get_unified_knowledge
            uk = get_unified_knowledge()
            results = await uk.search_facts(query, limit=limit)
            return [
                {"content": r.content, "confidence": r.confidence, "type": r.source_type}
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Facts search error: {e}")
            return []

    # -- connect (graph-based relationship discovery) -------------------------

    async def _discover_connections(self, entities: set[str]) -> list[Connection]:
        """Use the graph engine to find relationships between discovered entities."""
        connections = []
        try:
            from src.core.graph_engine import get_graph_engine
            graph = get_graph_engine()
            await graph._ensure_loaded()

            entity_list = list(entities)[:20]  # cap to prevent explosion

            # Find paths between entity pairs
            for i, entity_a in enumerate(entity_list):
                for entity_b in entity_list[i+1:]:
                    try:
                        path = graph.find_path(entity_a, entity_b)
                        if path and len(path) <= 4:  # only short meaningful paths
                            connections.append(Connection(
                                entity_a=entity_a,
                                entity_b=entity_b,
                                relationship=" → ".join(
                                    p.get("predicate", p.get("relationship", "related"))
                                    for p in path if isinstance(p, dict)
                                ) or "connected",
                                path=[str(p) for p in path],
                                strength=1.0 / len(path),  # shorter = stronger
                            ))
                    except Exception:
                        pass

                # Also get neighborhood for each entity (1-hop connections)
                try:
                    related = graph.get_related(entity_a, depth=1, max_results=5)
                    for r in (related or []):
                        if isinstance(r, dict):
                            target = r.get("entity", r.get("target", r.get("object", "")))
                            rel = r.get("predicate", r.get("relationship", "related to"))
                            if target and target not in entities:
                                connections.append(Connection(
                                    entity_a=entity_a,
                                    entity_b=str(target),
                                    relationship=str(rel),
                                    strength=float(r.get("weight", r.get("score", 0.5))),
                                ))
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Graph connection discovery failed: {e}")

        # Deduplicate
        seen = set()
        unique = []
        for c in connections:
            key = frozenset([c.entity_a, c.entity_b, c.relationship])
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique[:30]  # cap connections

    def _extract_entities(self, findings: list[FindingSet]) -> set[str]:
        """Extract named entities from findings for graph exploration."""
        entities = set()
        for f in findings:
            for r in f.web_results:
                title = r.get("title", "")
                # Extract capitalized names/phrases from titles
                for match in re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', title):
                    entities.add(match)
                # Extract domain names
                url = r.get("url", "")
                if url:
                    domain = re.search(r'https?://(?:www\.)?([^/]+)', url)
                    if domain:
                        entities.add(domain.group(1).split('.')[0])

            for r in f.memory_results:
                content = r.get("content", "")
                # Extract emails, names, services
                for email in re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', content):
                    entities.add(email)
                for name in re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content):
                    entities.add(name)

            for r in f.fact_results:
                content = r.get("content", "")
                for name in re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content):
                    entities.add(name)

        # Remove noise — very short or very common
        noise = {"The", "This", "That", "What", "When", "Where", "Which", "Web Results",
                 "Memory Results", "Fact Results", "Sub Question", "Search Query"}
        return {e for e in entities if len(e) > 3 and e not in noise}

    # -- evaluate (intelligent gap analysis) ----------------------------------

    async def _evaluate(self, question: str, findings: list[FindingSet],
                        connections: list[Connection], entities: set[str],
                        iteration: int) -> Evaluation:
        from src.services.llm_service import get_llm_service
        llm = get_llm_service()

        findings_summary = self._summarize_findings(findings)
        connections_text = "\n".join(
            f"  - {c.entity_a} → {c.relationship} → {c.entity_b} (strength: {c.strength:.2f})"
            for c in connections[:15]
        ) or "  (none found)"
        entities_text = ", ".join(list(entities)[:30]) or "(none)"

        # Calculate evidence distribution
        web_count = sum(len(f.web_results) for f in findings)
        mem_count = sum(len(f.memory_results) for f in findings)
        fact_count = sum(len(f.fact_results) for f in findings)
        avg_confidence = sum(f.retriever_confidence for f in findings) / max(len(findings), 1)

        prompt = f"""You are evaluating research evidence. Be critical and specific.

QUESTION: {question}

ITERATION: {iteration}

EVIDENCE SUMMARY:
{findings_summary}

CONNECTIONS DISCOVERED:
{connections_text}

ENTITIES FOUND: {entities_text}

EVIDENCE STATS: {web_count} web, {mem_count} memory, {fact_count} facts, avg confidence {avg_confidence:.2f}

Think about:
1. What SPECIFIC information is still missing to fully answer the question?
2. Are there entities we found but haven't explored the CONNECTIONS between?
3. Are there contradictions that need resolving?
4. What would a human researcher look for next?

Return ONLY a JSON object:
{{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "gaps": ["specific missing piece 1", "specific missing piece 2"],
  "follow_up_queries": [
    {{"query": "exact search query targeting a gap", "purpose": "what specific gap this fills"}}
  ]
}}

If generating follow-ups: make them DIFFERENT from what was already searched. Target the gaps.
Return ONLY the JSON object."""

        resp = await llm.generate(
            prompt=prompt,
            model="deepseek-r1:8b",
            temperature=0.2,
            max_tokens=1024,
        )

        return self._parse_evaluation(resp.content)

    def _parse_evaluation(self, text: str) -> Evaluation:
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        data = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return Evaluation(sufficient=True, confidence=0.5)

        follow_ups = []
        for fq in data.get("follow_up_queries", []):
            if isinstance(fq, dict) and "query" in fq:
                follow_ups.append(SubQuestion(query=fq["query"], purpose=fq.get("purpose", "")))

        return Evaluation(
            sufficient=bool(data.get("sufficient", True)),
            confidence=float(data.get("confidence", 0.5)),
            gaps=data.get("gaps", []),
            follow_up_queries=follow_ups,
        )

    # -- synthesize (connection-aware) ----------------------------------------

    async def _synthesize(self, question: str, findings: list[FindingSet],
                          sources: list[Source], connections: list[Connection],
                          entities: set[str]) -> ResearchReport:
        from src.services.llm_service import get_llm_service
        llm = get_llm_service()

        findings_text = self._format_findings_for_llm(findings, max_chars=6000)

        source_index = "\n".join(
            f"[{s.ref}] ({s.source_type}) {s.title}: {s.snippet[:120]}"
            for s in sources[:60]  # cap source index
        )

        connections_text = ""
        if connections:
            connections_text = "\n\nKNOWN CONNECTIONS:\n" + "\n".join(
                f"- {c.entity_a} → {c.relationship} → {c.entity_b}"
                for c in connections[:20]
            )

        prompt = f"""You are a research analyst synthesizing findings into a comprehensive report.

QUESTION: {question}

EVIDENCE:
{findings_text}

SOURCE INDEX:
{source_index}
{connections_text}

INSTRUCTIONS:
- Write a clear, structured report answering the question
- Cite sources inline: [W1] for web, [M1] for memory, [F1] for facts, [G1] for graph
- Group related findings into coherent sections
- Highlight CONNECTIONS between entities — what patterns emerge?
- Note contradictions or unreliable sources explicitly
- Distinguish between what the evidence SHOWS vs what you're INFERRING
- If evidence is insufficient for part of the question, say so
- End with a "Key Connections" section summarizing the relationship map"""

        resp = await llm.generate(
            prompt=prompt,
            model="mistral:7b",
            temperature=0.4,
            max_tokens=4096,
        )

        return ResearchReport(answer=resp.content, sources=sources)

    # -- store findings back into Echo Brain ----------------------------------

    async def _store_findings(self, job: ResearchJob):
        """Store research results back into Echo Brain for future retrieval."""
        if not job.report:
            return

        try:
            from src.integrations.mcp_service import mcp_service

            # Store the research report as a memory
            summary = (
                f"Deep Research on: {job.question}\n"
                f"Depth: {job.depth}, Iterations: {job.iterations}, "
                f"Sources: {job.sources_consulted}\n"
                f"Answer: {job.report.answer[:500]}"
            )
            await mcp_service.store_memory(summary, type_="research_report")

            # Store key findings as facts
            if job.report.connections:
                for conn in job.report.connections[:10]:
                    try:
                        await mcp_service.store_fact(
                            subject=conn.entity_a,
                            predicate=conn.relationship or "related to",
                            object_=conn.entity_b,
                            confidence=conn.strength,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to store connection as fact: {e}")

            logger.info(f"Stored research findings for job {job.id}")
        except Exception as e:
            logger.warning(f"Failed to store findings in Echo Brain: {e}")

    # -- helpers -------------------------------------------------------------

    def _build_sources(self, findings: list[FindingSet]) -> list[Source]:
        sources = []
        web_idx, mem_idx, fact_idx, graph_idx = 1, 1, 1, 1
        seen_urls = set()
        seen_content = set()

        for f in findings:
            for r in f.web_results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append(Source(
                        ref=f"W{web_idx}", source_type="web",
                        title=r.get("title", ""), snippet=r.get("snippet", ""),
                        url=url,
                    ))
                    web_idx += 1

            for r in f.memory_results:
                content = r.get("content", "")[:80]
                if content and content not in seen_content:
                    seen_content.add(content)
                    sources.append(Source(
                        ref=f"M{mem_idx}", source_type="memory",
                        title=f"Memory: {content[:50]}",
                        snippet=r.get("content", ""),
                        score=r.get("score", 0),
                    ))
                    mem_idx += 1

            for r in f.fact_results:
                content = r.get("content", "")[:80]
                if content and content not in seen_content:
                    seen_content.add(content)
                    sources.append(Source(
                        ref=f"F{fact_idx}", source_type="fact",
                        title=f"Fact: {content[:50]}",
                        snippet=r.get("content", ""),
                        score=r.get("confidence", 0),
                    ))
                    fact_idx += 1

            for r in f.graph_results:
                content = r.get("content", "")[:80]
                if content and content not in seen_content:
                    seen_content.add(content)
                    sources.append(Source(
                        ref=f"G{graph_idx}", source_type="graph",
                        title=f"Graph: {content[:50]}",
                        snippet=r.get("content", ""),
                        score=r.get("score", 0),
                    ))
                    graph_idx += 1

        return sources

    def _format_findings_for_llm(self, findings: list[FindingSet], max_chars: int = 0) -> str:
        parts = []
        for f in findings:
            parts.append(f"\n--- Sub-question: {f.sub_question} (domain: {f.domain}, confidence: {f.retriever_confidence:.2f}) ---")
            if f.web_results:
                parts.append("Web:")
                for r in f.web_results:
                    parts.append(f"  - {r.get('title', '')} | {r.get('snippet', '')[:200]}")
            if f.memory_results:
                parts.append("Memory:")
                for r in f.memory_results:
                    parts.append(f"  - [{r.get('score', 0):.2f}] {r.get('content', '')[:200]}")
            if f.fact_results:
                parts.append("Facts:")
                for r in f.fact_results:
                    parts.append(f"  - [{r.get('confidence', 0):.1f}] {r.get('content', '')[:200]}")
            if f.graph_results:
                parts.append("Graph connections:")
                for r in f.graph_results:
                    parts.append(f"  - {r.get('content', '')[:200]}")
        text = "\n".join(parts)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text

    def _summarize_findings(self, findings: list[FindingSet]) -> str:
        """Compact summary for evaluation — focuses on what was found, not raw data."""
        parts = []
        for f in findings:
            web_titles = [r.get("title", "")[:60] for r in f.web_results[:3]]
            mem_snippets = [r.get("content", "")[:80] for r in f.memory_results[:3]]
            fact_snippets = [r.get("content", "")[:80] for r in f.fact_results[:3]]

            parts.append(f"Q: {f.sub_question}")
            if web_titles:
                parts.append(f"  Web found: {'; '.join(web_titles)}")
            if mem_snippets:
                parts.append(f"  Memory found: {'; '.join(mem_snippets)}")
            if fact_snippets:
                parts.append(f"  Facts found: {'; '.join(fact_snippets)}")
            if not web_titles and not mem_snippets and not fact_snippets:
                parts.append("  (no results)")
        return "\n".join(parts)

    def _report_to_dict(self, report: ResearchReport) -> dict:
        return {
            "answer": report.answer,
            "sources": [
                {"ref": s.ref, "type": s.source_type, "title": s.title,
                 "snippet": s.snippet, "url": s.url}
                for s in report.sources
            ],
            "connections": [
                {"entity_a": c.entity_a, "entity_b": c.entity_b,
                 "relationship": c.relationship, "strength": c.strength}
                for c in report.connections
            ],
            "entities_discovered": report.entities_discovered,
            "sub_questions": report.sub_questions,
            "iterations": report.iterations,
            "total_sources_consulted": report.total_sources_consulted,
        }

    def _emit(self, job_id: str, event_type: str, data: dict):
        q = self._queues.get(job_id)
        if q:
            try:
                q.put_nowait({"event": event_type, "data": data})
            except asyncio.QueueFull:
                pass
        job = self._jobs.get(job_id)
        if job:
            job.progress = {"event": event_type, **data}

    # -- database ------------------------------------------------------------

    _pool = None

    async def _get_pool(self):
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", 5432)),
                user=os.getenv("DB_USER", "patrick"),
                password=os.getenv("DB_PASSWORD", ""),
                database=os.getenv("DB_NAME", "echo_brain"),
                min_size=1,
                max_size=5,
            )
        return self._pool

    async def _ensure_table(self):
        if self._table_ready:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SET search_path TO public")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_jobs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    question TEXT NOT NULL,
                    depth VARCHAR(20) NOT NULL DEFAULT 'standard',
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    progress JSONB DEFAULT '{}',
                    report JSONB,
                    error_message TEXT,
                    iterations INTEGER DEFAULT 0,
                    sources_consulted INTEGER DEFAULT 0,
                    started_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ,
                    total_time_ms FLOAT DEFAULT 0
                )
            """)
        self._table_ready = True

    async def _persist_job(self, job: ResearchJob):
        await self._ensure_table()
        pool = await self._get_pool()

        report_json = None
        if job.report:
            report_json = json.dumps(self._report_to_dict(job.report))

        async with pool.acquire() as conn:
            await conn.execute("SET search_path TO public")
            await conn.execute("""
                INSERT INTO research_jobs (id, question, depth, status, progress,
                    report, error_message, iterations, sources_consulted,
                    started_at, completed_at, total_time_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    progress = EXCLUDED.progress,
                    report = EXCLUDED.report,
                    error_message = EXCLUDED.error_message,
                    iterations = EXCLUDED.iterations,
                    sources_consulted = EXCLUDED.sources_consulted,
                    completed_at = EXCLUDED.completed_at,
                    total_time_ms = EXCLUDED.total_time_ms
            """,
                uuid.UUID(job.id), job.question, job.depth, job.status,
                json.dumps(job.progress), report_json, job.error_message,
                job.iterations, job.sources_consulted,
                job.started_at, job.completed_at, job.total_time_ms,
            )

    def _row_to_job(self, row) -> ResearchJob:
        report = None
        if row["report"]:
            rd = row["report"] if isinstance(row["report"], dict) else json.loads(row["report"])
            report = ResearchReport(
                answer=rd.get("answer", ""),
                sources=[
                    Source(ref=s["ref"], source_type=s["type"], title=s["title"],
                           snippet=s["snippet"], url=s.get("url", ""), score=s.get("score", 0))
                    for s in rd.get("sources", [])
                ],
                connections=[
                    Connection(entity_a=c["entity_a"], entity_b=c["entity_b"],
                               relationship=c["relationship"], strength=c.get("strength", 0))
                    for c in rd.get("connections", [])
                ],
                entities_discovered=rd.get("entities_discovered", []),
                sub_questions=rd.get("sub_questions", []),
                iterations=rd.get("iterations", 0),
                total_sources_consulted=rd.get("total_sources_consulted", 0),
            )
        return ResearchJob(
            id=str(row["id"]),
            question=row["question"],
            depth=row["depth"],
            status=row["status"],
            progress=row["progress"] if isinstance(row["progress"], dict) else {},
            report=report,
            error_message=row["error_message"],
            iterations=row["iterations"],
            sources_consulted=row["sources_consulted"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            total_time_ms=row["total_time_ms"] or 0,
        )


# Singleton
_engine: Optional[ResearchEngine] = None


def get_research_engine() -> ResearchEngine:
    global _engine
    if _engine is None:
        _engine = ResearchEngine()
    return _engine
