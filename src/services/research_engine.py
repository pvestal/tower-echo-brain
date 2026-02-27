"""
Deep Research Engine for Echo Brain.

Takes complex questions, decomposes into sub-questions, searches web + memory + facts
in parallel, evaluates sufficiency, and synthesizes a cited report.
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
# Data models
# ---------------------------------------------------------------------------

class Depth(str, Enum):
    quick = "quick"
    standard = "standard"
    deep = "deep"

MAX_ITERATIONS = {"quick": 1, "standard": 2, "deep": 3}


@dataclass
class SubQuestion:
    query: str
    purpose: str


@dataclass
class Source:
    ref: str          # e.g. W1, M3, F2
    source_type: str  # web, memory, fact
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


@dataclass
class Evaluation:
    sufficient: bool
    confidence: float
    gaps: list[str] = field(default_factory=list)
    follow_up_queries: list[SubQuestion] = field(default_factory=list)


@dataclass
class ResearchReport:
    answer: str
    sources: list[Source] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    iterations: int = 0
    total_sources_consulted: int = 0


class JobStatus(str, Enum):
    pending = "pending"
    decomposing = "decomposing"
    searching = "searching"
    evaluating = "evaluating"
    synthesizing = "synthesizing"
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
        # fallback to DB
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
        try:
            # 1. Decompose
            job.status = JobStatus.decomposing
            self._emit(job.id, "decomposing", {"status": "Breaking question into sub-questions"})
            sub_questions = await asyncio.wait_for(
                self._decompose(job.question), timeout=60
            )
            self._emit(job.id, "decomposing", {
                "sub_questions": [{"query": sq.query, "purpose": sq.purpose} for sq in sub_questions]
            })

            all_findings: list[FindingSet] = []
            all_sources: list[Source] = []
            max_iter = MAX_ITERATIONS.get(job.depth, 2)

            for iteration in range(max_iter):
                job.iterations = iteration + 1

                # 2. Search in parallel per sub-question
                job.status = JobStatus.searching
                queries_to_search = sub_questions if iteration == 0 else [
                    sq for sq in (getattr(job, '_follow_ups', []) or [])
                ]
                if not queries_to_search:
                    break

                for sq in queries_to_search:
                    self._emit(job.id, "searching", {
                        "sub_question": sq.query,
                        "iteration": iteration + 1,
                    })

                findings = await asyncio.wait_for(
                    self._search_all(queries_to_search), timeout=90
                )
                all_findings.extend(findings)

                # Emit search counts per sub-question
                for f in findings:
                    self._emit(job.id, "searching", {
                        "sub_question": f.sub_question,
                        "iteration": iteration + 1,
                        "web": len(f.web_results),
                        "memory": len(f.memory_results),
                        "facts": len(f.fact_results),
                    })

                # Build source list
                all_sources = self._build_sources(all_findings)
                job.sources_consulted = len(all_sources)

                # 3. Evaluate sufficiency (skip on last allowed iteration)
                if iteration < max_iter - 1:
                    job.status = JobStatus.evaluating
                    evaluation = await asyncio.wait_for(
                        self._evaluate_sufficiency(job.question, all_findings), timeout=60
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

            # 4. Synthesize
            job.status = JobStatus.synthesizing
            self._emit(job.id, "synthesizing", {"sources_count": len(all_sources)})
            report = await asyncio.wait_for(
                self._synthesize(job.question, all_findings, all_sources), timeout=120
            )
            report.iterations = job.iterations
            report.total_sources_consulted = job.sources_consulted
            report.sub_questions = [sq.query for sq in sub_questions]

            job.report = report
            job.status = JobStatus.complete
            job.completed_at = datetime.utcnow()
            job.total_time_ms = (time.time() - t0) * 1000

            self._emit(job.id, "complete", {
                "report": {
                    "answer": report.answer,
                    "sources": [
                        {"ref": s.ref, "type": s.source_type, "title": s.title,
                         "snippet": s.snippet, "url": s.url}
                        for s in report.sources
                    ],
                    "sub_questions": report.sub_questions,
                    "iterations": report.iterations,
                    "total_sources_consulted": report.total_sources_consulted,
                }
            })

        except asyncio.TimeoutError:
            job.status = JobStatus.failed
            job.error_message = "Research timed out"
            job.completed_at = datetime.utcnow()
            job.total_time_ms = (time.time() - t0) * 1000
            self._emit(job.id, "error", {"message": "Research timed out"})
        except Exception as e:
            logger.exception(f"Research failed for job {job.id}")
            job.status = JobStatus.failed
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            job.total_time_ms = (time.time() - t0) * 1000
            self._emit(job.id, "error", {"message": str(e)})

        # Persist to DB
        try:
            await self._persist_job(job)
        except Exception as e:
            logger.error(f"Failed to persist job {job.id}: {e}")

    # -- decompose -----------------------------------------------------------

    async def _decompose(self, question: str) -> list[SubQuestion]:
        from src.services.llm_service import get_llm_service
        llm = get_llm_service()

        prompt = f"""Break this research question into 3-6 focused sub-questions for searching.
Return ONLY a JSON array of objects with "query" and "purpose" fields.

Question: {question}

Example format:
[
  {{"query": "specific search query", "purpose": "what this answers"}},
  {{"query": "another search query", "purpose": "what this answers"}}
]

Return ONLY the JSON array, no other text."""

        resp = await llm.generate(
            prompt=prompt,
            model="deepseek-r1:8b",
            temperature=0.3,
            max_tokens=1024,
        )
        return self._parse_sub_questions(resp.content)

    def _parse_sub_questions(self, text: str) -> list[SubQuestion]:
        # Strip <think>...</think> tags from deepseek-r1
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Try direct JSON parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [SubQuestion(query=item["query"], purpose=item.get("purpose", ""))
                        for item in data if "query" in item]
        except json.JSONDecodeError:
            pass

        # Regex fallback: find JSON array
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [SubQuestion(query=item["query"], purpose=item.get("purpose", ""))
                            for item in data if "query" in item]
            except json.JSONDecodeError:
                pass

        # Last resort: use original question as single sub-question
        logger.warning(f"Failed to parse sub-questions, using original question")
        return [SubQuestion(query=text[:200] if len(text) > 200 else text,
                            purpose="original question")]

    # -- search --------------------------------------------------------------

    async def _search_all(self, sub_questions: list[SubQuestion]) -> list[FindingSet]:
        tasks = [self._search_one(sq) for sq in sub_questions]
        return await asyncio.gather(*tasks)

    async def _search_one(self, sq: SubQuestion) -> FindingSet:
        finding = FindingSet(sub_question=sq.query)

        web_task = self._search_web(sq.query)
        memory_task = self._search_memory(sq.query)
        facts_task = self._search_facts(sq.query)

        results = await asyncio.gather(web_task, memory_task, facts_task, return_exceptions=True)

        if not isinstance(results[0], Exception):
            finding.web_results = results[0]
        else:
            logger.warning(f"Web search failed for '{sq.query}': {results[0]}")

        if not isinstance(results[1], Exception):
            finding.memory_results = results[1]
        else:
            logger.warning(f"Memory search failed for '{sq.query}': {results[1]}")

        if not isinstance(results[2], Exception):
            finding.fact_results = results[2]
        else:
            logger.warning(f"Facts search failed for '{sq.query}': {results[2]}")

        return finding

    async def _search_web(self, query: str) -> list[dict]:
        try:
            from src.services.search_service import get_search_service
            svc = get_search_service()
            resp = await svc.search(query=query, num_results=5)
            return [
                {"title": r.title, "url": r.url, "snippet": r.snippet}
                for r in resp.results[:5]
            ]
        except Exception as e:
            logger.warning(f"Web search error: {e}")
            return []

    async def _search_memory(self, query: str) -> list[dict]:
        try:
            from src.core.unified_knowledge import get_unified_knowledge
            uk = get_unified_knowledge()
            results = await uk.search_vectors(query, limit=5)
            return [
                {"content": r.content, "score": r.confidence, "source": r.metadata.get("source", "")}
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Memory search error: {e}")
            return []

    async def _search_facts(self, query: str) -> list[dict]:
        try:
            from src.core.unified_knowledge import get_unified_knowledge
            uk = get_unified_knowledge()
            results = await uk.search_facts(query, limit=5)
            return [
                {"content": r.content, "confidence": r.confidence, "type": r.source_type}
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Facts search error: {e}")
            return []

    # -- evaluate ------------------------------------------------------------

    async def _evaluate_sufficiency(self, question: str, findings: list[FindingSet]) -> Evaluation:
        from src.services.llm_service import get_llm_service
        llm = get_llm_service()

        findings_text = self._format_findings_for_llm(findings)

        prompt = f"""Evaluate whether the evidence below is sufficient to answer the question.

Question: {question}

Evidence gathered:
{findings_text}

Return ONLY a JSON object:
{{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "gaps": ["list of missing information"],
  "follow_up_queries": [{{"query": "...", "purpose": "..."}}]
}}

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

    # -- synthesize ----------------------------------------------------------

    async def _synthesize(self, question: str, findings: list[FindingSet],
                          sources: list[Source]) -> ResearchReport:
        from src.services.llm_service import get_llm_service
        llm = get_llm_service()

        findings_text = self._format_findings_for_llm(findings)

        source_index = "\n".join(
            f"[{s.ref}] ({s.source_type}) {s.title}: {s.snippet[:100]}"
            for s in sources
        )

        prompt = f"""You are a research assistant. Synthesize a comprehensive answer from the evidence below.

Question: {question}

Evidence:
{findings_text}

Source Index:
{source_index}

Instructions:
- Write a clear, well-structured answer
- Cite sources inline using their reference tags like [W1], [M1], [F1]
- If sources conflict, note the disagreement
- Be direct and factual
- If evidence is insufficient, say so honestly"""

        resp = await llm.generate(
            prompt=prompt,
            model="mistral:7b",
            temperature=0.4,
            max_tokens=4096,
        )

        return ResearchReport(answer=resp.content, sources=sources)

    # -- helpers -------------------------------------------------------------

    def _build_sources(self, findings: list[FindingSet]) -> list[Source]:
        sources = []
        web_idx, mem_idx, fact_idx = 1, 1, 1
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

        return sources

    def _format_findings_for_llm(self, findings: list[FindingSet]) -> str:
        parts = []
        for f in findings:
            parts.append(f"\n--- Sub-question: {f.sub_question} ---")
            if f.web_results:
                parts.append("Web results:")
                for r in f.web_results:
                    parts.append(f"  - {r.get('title', '')} | {r.get('snippet', '')[:200]}")
            if f.memory_results:
                parts.append("Memory results:")
                for r in f.memory_results:
                    parts.append(f"  - [{r.get('score', 0):.2f}] {r.get('content', '')[:200]}")
            if f.fact_results:
                parts.append("Facts:")
                for r in f.fact_results:
                    parts.append(f"  - [{r.get('confidence', 0):.1f}] {r.get('content', '')[:200]}")
        return "\n".join(parts)

    def _emit(self, job_id: str, event_type: str, data: dict):
        q = self._queues.get(job_id)
        if q:
            try:
                q.put_nowait({"event": event_type, "data": data})
            except asyncio.QueueFull:
                pass
        # Also update job progress
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
            report_json = json.dumps({
                "answer": job.report.answer,
                "sources": [
                    {"ref": s.ref, "type": s.source_type, "title": s.title,
                     "snippet": s.snippet, "url": s.url, "score": s.score}
                    for s in job.report.sources
                ],
                "sub_questions": job.report.sub_questions,
                "iterations": job.report.iterations,
                "total_sources_consulted": job.report.total_sources_consulted,
            })

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
