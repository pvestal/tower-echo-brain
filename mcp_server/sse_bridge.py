"""
Echo Brain MCP Server — SSE transport, runs in-process with main.py.

Eliminates the stdio proxy layer by calling service methods directly.
Mounted inside the FastAPI app at /mcp/sse.

Usage in ~/.claude.json:
  "echo-brain": {"type": "sse", "url": "http://localhost:8309/mcp/sse"}
"""

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("echo_brain.mcp_sse")

mcp = FastMCP(
    "echo-brain",
    instructions=(
        "Echo Brain memory system — search memories, get/store facts, "
        "explore knowledge graph, manage Ollama models, search photos/videos. "
        "Telegram bot listener accepts messages and answers questions via reasoning engine. "
        "Web search via SearXNG, deep research engine (decompose→search→synthesize with citations), "
        "and credit monitoring bridge."
    ),
)

# ── Service references (set by init_sse_bridge() during app startup) ──

_mcp_service = None
_app_ref = None  # Reference to FastAPI app for accessing other services


def init_sse_bridge(mcp_service, app=None):
    """Called during main.py startup to inject the live service instance."""
    global _mcp_service, _app_ref
    _mcp_service = mcp_service
    _app_ref = app
    logger.info("SSE bridge initialized with mcp_service")


def get_sse_app(mount_path: str = "/mcp/sse"):
    """Return the Starlette SSE app for mounting in FastAPI."""
    return mcp.sse_app(mount_path)


# ── Core memory tools ────────────────────────────────────────────────


@mcp.tool()
async def search_memory(
    query: str, limit: int = 5, after: str = "", before: str = ""
) -> str:
    """Search Echo Brain memories using semantic similarity. Returns relevant memories with confidence scores.

    Args:
        query: Search query
        limit: Max results (default 5)
        after: Only return memories after this ISO datetime (e.g. 2026-02-01T00:00:00)
        before: Only return memories before this ISO datetime
    """
    try:
        raw = await _mcp_service.search_memory(query, limit, after=after, before=before)
    except Exception as e:
        logger.error(f"search_memory failed: {e}")
        return f"Search error: {e}"
    if not raw:
        return "No memories found."
    # search_memory can return a list of dicts OR a dict with {"results": [...]}
    if isinstance(raw, dict):
        results = raw.get("results", [])
    else:
        results = raw
    if not results:
        return "No memories found."
    lines = []
    for r in results:
        if isinstance(r, str):
            lines.append(r)
            continue
        score = r.get("score", 0)
        content = r.get("content", "")[:500]
        source = r.get("source", "")
        lines.append(f"[{score:.2f}] ({source}) {content}")
    return "\n---\n".join(lines)


@mcp.tool()
async def get_facts(topic: str) -> str:
    """Get structured facts from Echo Brain about a topic."""
    facts = await _mcp_service.get_facts(topic)
    if not facts:
        return f"No facts found for '{topic}'."
    lines = []
    for f in facts:
        content = f.get("content", "")
        conf = f.get("confidence", 0)
        lines.append(f"[{conf:.1f}] {content}")
    return "\n".join(lines) if lines else f"No facts found for '{topic}'."


@mcp.tool()
async def store_fact(subject: str, predicate: str, object: str) -> str:
    """Store a new fact in Echo Brain as a subject-predicate-object triple."""
    if not subject or not predicate or not object:
        return json.dumps({"fact_id": "", "stored": False, "error": "subject, predicate, and object are all required"})
    result = await _mcp_service.store_fact(subject, predicate, object)
    if isinstance(result, dict):
        return json.dumps(result)
    if result:
        return json.dumps({"fact_id": result, "stored": True})
    return json.dumps({"fact_id": "", "stored": False, "error": "Embedding or storage failed"})


@mcp.tool()
async def store_memory(content: str, type: str = "memory") -> str:
    """Store free-form text memory in Echo Brain (not a structured triple).

    Args:
        content: The text content to store
        type: Memory type (default: memory)
    """
    if not content:
        return json.dumps({"memory_id": "", "stored": False, "error": "content is required"})
    result = await _mcp_service.store_memory(content, type_=type)
    if isinstance(result, dict):
        return json.dumps(result)
    if result:
        return json.dumps({"memory_id": result, "stored": True})
    return json.dumps({"memory_id": "", "stored": False, "error": "Embedding or storage failed"})


@mcp.tool()
async def explore_graph(entity: str, depth: int = 2) -> str:
    """Explore the knowledge graph: find related entities, paths, and neighborhood stats."""
    depth = min(depth, 3)
    try:
        from src.core.graph_engine import get_graph_engine
        engine = get_graph_engine()
        await engine._ensure_loaded()
        related = engine.get_related(entity, depth=depth, max_results=50)
        neighborhood = engine.get_neighborhood(entity, hops=depth)
        stats = engine.get_stats()
        return json.dumps({
            "entity": entity,
            "related": related,
            "neighborhood": neighborhood,
            "graph_stats": stats,
        }, default=str)
    except Exception as e:
        return json.dumps({"error": f"Graph not available: {e}"})


# ── Ollama management ────────────────────────────────────────────────


@mcp.tool()
async def manage_ollama(action: str, model: str = "") -> str:
    """Manage Ollama models. Actions: list, running, pull, delete, refresh, show."""
    import httpx
    OLLAMA_URL = "http://localhost:11434"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if action == "list":
                resp = await client.get(f"{OLLAMA_URL}/api/tags")
                data = resp.json()
                models = data.get("models", [])
                lines = [f"Ollama models ({len(models)}):"]
                for m in models:
                    size_gb = m.get("size", 0) / 1e9
                    lines.append(f"  {m['name']} ({size_gb:.1f}GB)")
                return "\n".join(lines)
            elif action == "running":
                resp = await client.get(f"{OLLAMA_URL}/api/ps")
                data = resp.json()
                models = data.get("models", [])
                if not models:
                    return "No models currently loaded."
                lines = ["Running models:"]
                for m in models:
                    vram = m.get("size_vram", 0) / 1e9
                    lines.append(f"  {m['name']} (VRAM: {vram:.1f}GB)")
                return "\n".join(lines)
            elif action == "show" and model:
                resp = await client.post(f"{OLLAMA_URL}/api/show", json={"name": model})
                data = resp.json()
                return json.dumps(data, indent=2, default=str)[:3000]
            elif action == "pull" and model:
                resp = await client.post(f"{OLLAMA_URL}/api/pull", json={"name": model}, timeout=600)
                return f"Pulled {model}: {resp.status_code}"
            elif action == "delete" and model:
                resp = await client.delete(f"{OLLAMA_URL}/api/delete", json={"name": model})
                return f"Deleted {model}: {resp.status_code}"
            else:
                return f"Unknown action '{action}' or missing model parameter"
    except Exception as e:
        return f"Ollama error: {e}"


# ── Photo/video search ───────────────────────────────────────────────


@mcp.tool()
async def search_photos(
    query: str,
    media_type: str = "",
    year: str = "",
    category: str = "",
    person: str = "",
    limit: int = 10,
) -> str:
    """Search personal photos and videos using semantic similarity.
    Returns matching media with descriptions, categories, people, and scores.

    Args:
        query: Natural language search query (e.g. "beach sunset", "family dinner", "hiking")
        media_type: Filter by type: "photo" or "video" (empty = both)
        year: Filter by year (e.g. "2024")
        category: Filter by category (e.g. "travel", "family", "food")
        person: Filter by person name (from face detection clusters)
        limit: Max results (default 10)
    """
    try:
        from src.services.photo_dedup_service import PhotoDedupService
        svc = PhotoDedupService()
        results = await svc.search_media(
            query=query,
            media_type=media_type or None,
            year=year or None,
            category=category or None,
            person=person or None,
            limit=limit,
        )
    except Exception as e:
        return f"Photo search error: {e}"

    if not results:
        return f"No photos/videos found for '{query}'."

    lines = []
    for r in results:
        score = r.get("score", 0)
        media = r.get("media_type", "photo")
        filename = r.get("filename", "")
        desc = r.get("description", "")[:300]
        people = r.get("people", [])
        year_val = r.get("year", "")
        path = r.get("file_path", "")

        parts = [f"[{score:.2f}] [{media}] {filename}"]
        if year_val:
            parts.append(f"({year_val})")
        if people:
            parts.append(f"People: {', '.join(people)}")
        parts.append(f"\n  {desc}")
        parts.append(f"\n  Path: {path}")
        lines.append(" ".join(parts))

    return "\n---\n".join(lines)


# ── Notifications & reminders ────────────────────────────────────────


@mcp.tool()
async def send_notification(
    message: str,
    title: str = "",
    channels: list[str] = [],
    priority: str = "normal",
) -> str:
    """Send a notification via Telegram, ntfy, or email.

    Args:
        message: The message to send
        title: Optional title
        channels: Channels to send to (default: ["telegram"]). Options: telegram, ntfy, email
        priority: Notification priority: low, normal, high, urgent (default: normal)
    """
    data = await _mcp_service.send_notification(
        message=message,
        title=title or None,
        channels=channels or None,
        priority=priority,
    )
    if data.get("sent"):
        ch_list = ", ".join(data.get("channels", {}).keys())
        return f"Notification sent via {ch_list}: {data.get('message', '')}"
    return f"Failed to send notification: {data.get('error', 'unknown error')}"


@mcp.tool()
async def check_services() -> str:
    """Check health status of all Tower services (postgres, ollama, qdrant, mcp, comfyui)."""
    data = await _mcp_service.check_services()
    if "error" in data:
        return f"Error: {data['error']}"

    lines = [f"Overall: {data.get('overall_status', '?')} (uptime: {data.get('uptime_seconds', 0)}s)"]
    for svc in data.get("services", []):
        status_icon = "[OK]" if svc["status"] == "healthy" else "[DOWN]"
        line = f"  {status_icon} {svc['name']}: {svc['latency_ms']}ms"
        if svc.get("error"):
            line += f" - {svc['error']}"
        lines.append(line)

    res = data.get("resources", {})
    lines.append(f"  CPU: {res.get('cpu_percent', 0)}% | RAM: {res.get('memory_percent', 0)}% | Disk: {res.get('disk_percent', 0)}%")
    return "\n".join(lines)


@mcp.tool()
async def schedule_reminder(
    message: str, remind_at: str, title: str = "", channel: str = "telegram"
) -> str:
    """Schedule a reminder notification for a future time.

    Args:
        message: The reminder message
        remind_at: ISO 8601 datetime (e.g. 2026-02-24T15:30:00)
        title: Optional title
        channel: Channel to send reminder to: telegram, ntfy, email (default: telegram)
    """
    data = await _mcp_service.schedule_reminder(
        message=message,
        remind_at=remind_at,
        title=title or None,
        channel=channel,
    )
    if data.get("scheduled"):
        return f"Reminder scheduled for {data['remind_at']} via {data['channel']} (ID: {data['reminder_id']})"
    return f"Failed to schedule reminder: {data.get('error', 'unknown error')}"


# ── Anime generation ─────────────────────────────────────────────────


@mcp.tool()
async def trigger_generation(
    character_slug: str, count: int = 1, prompt_override: str = ""
) -> str:
    """Trigger image generation for a character via Anime Studio.

    Args:
        character_slug: Character slug (e.g. mario, goblin_slayer)
        count: Number of images to generate (1-5, default: 1)
        prompt_override: Optional prompt override
    """
    data = await _mcp_service.trigger_generation(
        character_slug=character_slug,
        count=count,
        prompt_override=prompt_override or None,
    )
    if data.get("errors"):
        return f"Errors: {'; '.join(data['errors'])}"
    generated = data.get("generated", 0)
    if generated == 0:
        return "No images generated."
    ids = [r.get("prompt_id", "?") for r in data.get("results", [])]
    return f"Triggered {generated} generation(s) for {character_slug}\nPrompt IDs: {', '.join(ids)}"


# ── Web search & fetch ───────────────────────────────────────────────


@mcp.tool()
async def web_search(
    query: str,
    num_results: int = 10,
    categories: str = "",
    time_range: str = "",
) -> str:
    """Search the web using self-hosted SearXNG. Returns titles, URLs, and snippets.

    Args:
        query: Search query
        num_results: Max results (default 10)
        categories: Comma-separated categories (general, science, it, news)
        time_range: Filter by time: day, week, month, year (empty = all time)
    """
    try:
        from src.services.search_service import get_search_service
        search_svc = get_search_service()
        cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else None
        search_resp = await search_svc.search(
            query=query,
            num_results=num_results,
            categories=cats,
            time_range=time_range or None,
        )
    except Exception as e:
        return f"Search error: {e}"

    lines = [f"Web search results for: {query}\n"]
    for r in search_resp.results:
        lines.append(f"[{r.position}] {r.title}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    {r.snippet}")
        lines.append("")

    lines.append(
        f"\nSource: {search_resp.source} | {search_resp.search_time_ms:.0f}ms | "
        f"{search_resp.total_results} total"
    )
    if search_resp.cached:
        lines.append("(cached)")
    return "\n".join(lines)


@mcp.tool()
async def web_fetch(url: str, max_length: int = 5000) -> str:
    """Fetch a URL and return its text content (HTML tags stripped).

    Args:
        url: The URL to fetch
        max_length: Max characters to return (default: 5000)
    """
    data = await _mcp_service.web_fetch(url=url, max_length=max_length)
    if "error" in data:
        return f"Error fetching {url}: {data['error']}"

    content = data.get("content", "")
    ct = data.get("content_type", "unknown")
    length = data.get("length", 0)
    trunc = " (truncated)" if data.get("truncated") else ""

    return f"URL: {data.get('url', url)}\nType: {ct}\nLength: {length}{trunc}\n\n{content}"


# ── Telegram ─────────────────────────────────────────────────────────


@mcp.tool()
async def telegram_bot_status() -> str:
    """Get Telegram bot listener status (running, offset, configured)."""
    data = await _mcp_service.telegram_bot_status()
    if data.get("running"):
        return (
            f"Telegram bot: RUNNING\n"
            f"  Offset: {data.get('offset', 0)}\n"
            f"  Admin chat: {data.get('admin_chat_id', '?')}\n"
            f"  Client configured: {data.get('client_configured', False)}"
        )
    error = data.get("error", "")
    return f"Telegram bot: STOPPED{' — ' + error if error else ''}"


# ── Deep research ────────────────────────────────────────────────────


@mcp.tool()
async def deep_research(question: str, depth: str = "standard") -> str:
    """Run deep research on a complex question. Decomposes into sub-questions,
    searches web + memory + facts in parallel, evaluates sufficiency, and
    synthesizes a cited report.

    Args:
        question: The research question to investigate
        depth: Research depth: quick (1 iteration), standard (up to 2), deep (up to 3)
    """
    import asyncio

    if not question:
        return "Error: question is required"

    try:
        from src.services.research_engine import get_research_engine
        engine = get_research_engine()
        job = engine.start_research(question, depth)

        for _ in range(60):
            await asyncio.sleep(2)
            current = await engine.get_job(job.id)
            if current and current.status in ("complete", "failed"):
                break

        current = await engine.get_job(job.id)
        if not current:
            return "Error: research job lost"
        if current.status == "failed":
            return f"Research failed: {current.error_message or 'unknown error'}"
        if current.status != "complete":
            return f"Research timed out (status: {current.status}). Job ID: {job.id}"

        if not current.report:
            return f"Research completed but no report generated. Job ID: {job.id}"

        lines = [f"Deep Research Report\n{'=' * 40}\n"]
        lines.append(f"Question: {question}")
        lines.append(
            f"Depth: {depth} | Iterations: {current.report.iterations} | "
            f"Sources: {current.report.total_sources_consulted} | "
            f"Time: {current.total_time_ms:.0f}ms\n"
        )
        lines.append(current.report.answer)

        if current.report.sources:
            lines.append(f"\n{'=' * 40}\nSources ({len(current.report.sources)}):\n")
            for s in current.report.sources:
                line = f"[{s.ref}] ({s.source_type}) {s.title}"
                if s.url:
                    line += f"\n    {s.url}"
                lines.append(line)

        if current.report.sub_questions:
            lines.append("\nSub-questions explored:")
            for sq in current.report.sub_questions:
                lines.append(f"  - {sq}")

        return "\n".join(lines)
    except Exception as e:
        return f"Research error: {e}"


# ── Session summary ──────────────────────────────────────────────────


@mcp.tool()
async def session_summary(
    summary: str,
    topics: list[str] = [],
    project: str = "",
    decisions: list[str] = [],
) -> str:
    """Store a session summary at end of a Claude Code session. Captures what was done,
    decisions made, and topics covered as a high-quality memory.

    Args:
        summary: Summary of what was accomplished in this session
        topics: Key topics covered (e.g. ['echo-brain', 'MCP', 'SSE transport'])
        project: Project name or path
        decisions: Key decisions or preferences expressed (e.g. ['use SSE over stdio'])
    """
    from datetime import datetime

    if not summary:
        return json.dumps({"stored": False, "error": "summary is required"})

    parts = [f"SESSION SUMMARY: {summary}"]
    if project:
        parts.append(f"Project: {project}")
    if topics:
        parts.append(f"Topics: {', '.join(topics)}")
    if decisions:
        parts.append(f"Decisions: {'; '.join(decisions)}")
    parts.append(f"Date: {datetime.now().isoformat()}")
    content = "\n".join(parts)

    result = await _mcp_service.store_memory(
        content,
        type_="session_summary",
        metadata={
            "source": "claude_code_session",
            "topics": topics,
            "project": project,
            "decisions": decisions,
        },
    )
    if isinstance(result, dict):
        return json.dumps(result)
    if result:
        # Store decisions as individual facts
        for decision in decisions:
            await _mcp_service.store_fact("Patrick", "decided", decision)
        return json.dumps({"stored": True, "memory_id": result, "facts_stored": len(decisions)})
    return json.dumps({"stored": False, "error": "Failed to store session summary"})


# ── Credit monitoring (external service proxy) ───────────────────────

CREDIT_MONITOR_URL = "http://localhost:8400"


@mcp.tool()
async def credit_dashboard() -> str:
    """Get credit monitoring dashboard: accounts, alerts, scores, and Treasury rates."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{CREDIT_MONITOR_URL}/api/internal/dashboard")
            if resp.status_code != 200:
                return f"Credit monitor unavailable: {resp.status_code}"
            data = resp.json()

        lines = ["Credit Monitor Dashboard\n"]
        for acc in data.get("accounts", [])[:10]:
            bal = acc.get("current_balance", "?")
            lines.append(f"  {acc['name']} ({acc.get('type','')}/{acc.get('subtype','')}) — ${bal}")

        for s in data.get("creditScores", []):
            lines.append(f"  {s['source']}: {s['credit_score']} ({s.get('score_model','')}) — {s['created_at']}")

        for a in data.get("alerts", [])[:5]:
            lines.append(f"  [{a.get('severity','?')}] {a.get('title','')}: {a.get('description','')[:80]}")

        rates = data.get("treasuryRates")
        if rates:
            avg = rates.get("averageRate", {})
            lines.append(f"\nTreasury Rates: trend={rates.get('trend','?')}, avg={avg.get('current','?')}%")

        return "\n".join(lines) if len(lines) > 1 else "Credit monitor returned no data"
    except Exception as e:
        return f"Credit monitor error: {e}"


@mcp.tool()
async def credit_alerts(severity: str = "") -> str:
    """Get credit and financial alerts. Filter by severity: critical, high, medium, low.

    Args:
        severity: Filter by severity level (empty = all)
    """
    import httpx
    try:
        params = {}
        if severity:
            params["severity"] = severity
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{CREDIT_MONITOR_URL}/api/internal/alerts", params=params)
            if resp.status_code != 200:
                return f"Failed to get alerts: {resp.status_code}"
            alerts = resp.json()

        if not alerts:
            return f"No alerts{' with severity ' + severity if severity else ''}"

        lines = [f"Credit Alerts ({len(alerts)}):\n"]
        for a in alerts:
            lines.append(f"[{a.get('severity','?')}] {a.get('alert_type','')}: {a.get('title','')}")
            desc = a.get("description", "")
            if desc:
                lines.append(f"  {desc[:120]}")
            lines.append(f"  Created: {a.get('created_at','')}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Credit alerts error: {e}"


@mcp.tool()
async def treasury_rates() -> str:
    """Get current US Treasury interest rates and trends."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{CREDIT_MONITOR_URL}/api/internal/treasury-rates")
            if resp.status_code != 200:
                return f"Failed to get Treasury rates: {resp.status_code}"
            data = resp.json()

        if "error" in data:
            return f"Treasury rates unavailable: {data['error']}"

        lines = ["US Treasury Interest Rates\n"]
        avg = data.get("averageRate", {})
        lines.append(
            f"Average Rate: {avg.get('current', '?')}% "
            f"(previous: {avg.get('previous', '?')}%, change: {avg.get('change', '?')}%)"
        )
        lines.append(f"Trend: {data.get('trend', '?')}")

        for s in data.get("marketableSecurities", [])[:5]:
            lines.append(f"  {s.get('security_desc', '?')}: {s.get('avg_interest_rate_amt', '?')}%")

        return "\n".join(lines)
    except Exception as e:
        return f"Treasury rates error: {e}"
