#!/usr/bin/env python3
"""
Echo Brain MCP Server — stdio transport for Claude Code.

Bridges the MCP stdio protocol to Echo Brain's HTTP API on port 8309.
Configured in ~/.claude.json as:
  "command": "/opt/tower-echo-brain/venv/bin/python",
  "args": ["/opt/tower-echo-brain/mcp_server/stdio_server.py"]
"""

import httpx
from mcp.server.fastmcp import FastMCP

ECHO_BRAIN_URL = "http://localhost:8309"

mcp = FastMCP("echo-brain", instructions="Echo Brain memory system — search memories, get/store facts, explore knowledge graph, manage Ollama models, search photos/videos. Telegram bot listener accepts messages and answers questions via reasoning engine. Web search via SearXNG, deep research engine (decompose→search→synthesize with citations), and credit monitoring bridge.")


async def _call_echo_brain(tool_name: str, arguments: dict) -> dict:
    """Forward a tool call to the Echo Brain HTTP MCP endpoint."""
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{ECHO_BRAIN_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def search_memory(query: str, limit: int = 5, after: str = "", before: str = "") -> str:
    """Search Echo Brain memories using semantic similarity. Returns relevant memories with confidence scores.

    Args:
        query: Search query
        limit: Max results (default 5)
        after: Only return memories after this ISO datetime (e.g. 2026-02-01T00:00:00)
        before: Only return memories before this ISO datetime
    """
    args = {"query": query, "limit": limit}
    if after:
        args["after"] = after
    if before:
        args["before"] = before
    data = await _call_echo_brain("search_memory", args)
    results = data.get("results", data) if isinstance(data, dict) else data
    if not results:
        return "No memories found."
    lines = []
    if isinstance(results, list):
        for r in results:
            score = r.get("score", 0)
            content = r.get("content", "")[:500]
            source = r.get("source", "")
            lines.append(f"[{score:.2f}] ({source}) {content}")
    elif isinstance(results, dict) and "results" in results:
        for r in results["results"]:
            score = r.get("score", 0)
            content = r.get("content", "")[:500]
            source = r.get("source", "")
            lines.append(f"[{score:.2f}] ({source}) {content}")
    else:
        return str(results)
    return "\n---\n".join(lines)


@mcp.tool()
async def get_facts(topic: str) -> str:
    """Get structured facts from Echo Brain about a topic."""
    data = await _call_echo_brain("get_facts", {"topic": topic})
    if not data:
        return f"No facts found for '{topic}'."
    if isinstance(data, list):
        lines = []
        for f in data:
            content = f.get("content", "")
            conf = f.get("confidence", 0)
            lines.append(f"[{conf:.1f}] {content}")
        return "\n".join(lines) if lines else f"No facts found for '{topic}'."
    return str(data)


@mcp.tool()
async def store_fact(subject: str, predicate: str, object: str) -> str:
    """Store a new fact in Echo Brain as a subject-predicate-object triple."""
    data = await _call_echo_brain("store_fact", {
        "subject": subject,
        "predicate": predicate,
        "object": object,
    })
    return str(data)


@mcp.tool()
async def store_memory(content: str, type: str = "memory") -> str:
    """Store free-form text memory in Echo Brain (not a structured triple).

    Args:
        content: The text content to store
        type: Memory type (default: memory)
    """
    data = await _call_echo_brain("store_memory", {"content": content, "type": type})
    return str(data)


@mcp.tool()
async def explore_graph(entity: str, depth: int = 2) -> str:
    """Explore the knowledge graph: find related entities, paths, and neighborhood stats."""
    data = await _call_echo_brain("explore_graph", {"entity": entity, "depth": depth})
    return str(data)


@mcp.tool()
async def manage_ollama(action: str, model: str = "") -> str:
    """Manage Ollama models. Actions: list, running, pull, delete, refresh, show."""
    args = {"action": action}
    if model:
        args["model"] = model
    data = await _call_echo_brain("manage_ollama", args)
    return str(data)


@mcp.tool()
async def search_photos(query: str, media_type: str = "", year: str = "",
                        category: str = "", person: str = "", limit: int = 10) -> str:
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
    # Call the photos search API directly (not MCP, since this is a direct HTTP endpoint)
    search_body = {"query": query, "limit": limit}
    if media_type:
        search_body["media_type"] = media_type
    if year:
        search_body["year"] = year
    if category:
        search_body["category"] = category
    if person:
        search_body["person"] = person

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{ECHO_BRAIN_URL}/api/photos/search",
            json=search_body,
        )
        if resp.status_code != 200:
            return f"Search failed: {resp.status_code}"

        data = resp.json()
        results = data.get("results", [])

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


@mcp.tool()
async def send_notification(message: str, title: str = "", channels: list[str] = [],
                            priority: str = "normal") -> str:
    """Send a notification via Telegram, ntfy, or email.

    Args:
        message: The message to send
        title: Optional title
        channels: Channels to send to (default: ["telegram"]). Options: telegram, ntfy, email
        priority: Notification priority: low, normal, high, urgent (default: normal)
    """
    args: dict = {"message": message, "priority": priority}
    if title:
        args["title"] = title
    if channels:
        args["channels"] = channels
    data = await _call_echo_brain("send_notification", args)
    if data.get("sent"):
        ch_list = ", ".join(data.get("channels", {}).keys())
        return f"Notification sent via {ch_list}: {data.get('message', '')}"
    return f"Failed to send notification: {data.get('error', 'unknown error')}"


@mcp.tool()
async def check_services() -> str:
    """Check health status of all Tower services (postgres, ollama, qdrant, mcp, comfyui)."""
    data = await _call_echo_brain("check_services", {})
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
async def schedule_reminder(message: str, remind_at: str, title: str = "",
                            channel: str = "telegram") -> str:
    """Schedule a reminder notification for a future time.

    Args:
        message: The reminder message
        remind_at: ISO 8601 datetime (e.g. 2026-02-24T15:30:00)
        title: Optional title
        channel: Channel to send reminder to: telegram, ntfy, email (default: telegram)
    """
    args: dict = {"message": message, "remind_at": remind_at, "channel": channel}
    if title:
        args["title"] = title
    data = await _call_echo_brain("schedule_reminder", args)
    if data.get("scheduled"):
        return f"Reminder scheduled for {data['remind_at']} via {data['channel']} (ID: {data['reminder_id']})"
    return f"Failed to schedule reminder: {data.get('error', 'unknown error')}"


@mcp.tool()
async def trigger_generation(character_slug: str, count: int = 1,
                             prompt_override: str = "") -> str:
    """Trigger image generation for a character via Anime Studio.

    Args:
        character_slug: Character slug (e.g. mario, goblin_slayer)
        count: Number of images to generate (1-5, default: 1)
        prompt_override: Optional prompt override
    """
    args: dict = {"character_slug": character_slug, "count": count}
    if prompt_override:
        args["prompt_override"] = prompt_override
    data = await _call_echo_brain("trigger_generation", args)

    if data.get("errors"):
        return f"Errors: {'; '.join(data['errors'])}"

    generated = data.get("generated", 0)
    if generated == 0:
        return "No images generated."

    ids = [r.get("prompt_id", "?") for r in data.get("results", [])]
    return f"Triggered {generated} generation(s) for {character_slug}\nPrompt IDs: {', '.join(ids)}"


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
    cats = [c.strip() for c in categories.split(",") if c.strip()] or None
    payload: dict = {
        "query": query,
        "num_results": num_results,
    }
    if cats:
        payload["categories"] = cats
    if time_range:
        payload["time_range"] = time_range

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{ECHO_BRAIN_URL}/api/echo/search/web",
            json=payload,
        )
        if resp.status_code != 200:
            return f"Search failed: {resp.status_code} - {resp.text[:200]}"
        data = resp.json()

    lines = [f"Web search results for: {query}\n"]
    for r in data.get("results", []):
        lines.append(f"[{r['position']}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['snippet']}")
        lines.append("")

    lines.append(f"\nSource: {data.get('source', '?')} | {data.get('search_time_ms', 0):.0f}ms | {data.get('total_results', 0)} total")
    if data.get("cached"):
        lines.append("(cached)")
    return "\n".join(lines)


@mcp.tool()
async def web_fetch(url: str, max_length: int = 5000) -> str:
    """Fetch a URL and return its text content (HTML tags stripped).

    Args:
        url: The URL to fetch
        max_length: Max characters to return (default: 5000)
    """
    data = await _call_echo_brain("web_fetch", {"url": url, "max_length": max_length})
    if "error" in data:
        return f"Error fetching {url}: {data['error']}"

    content = data.get("content", "")
    ct = data.get("content_type", "unknown")
    length = data.get("length", 0)
    trunc = " (truncated)" if data.get("truncated") else ""

    return f"URL: {data.get('url', url)}\nType: {ct}\nLength: {length}{trunc}\n\n{content}"


@mcp.tool()
async def telegram_bot_status() -> str:
    """Get Telegram bot listener status (running, offset, configured)."""
    data = await _call_echo_brain("telegram_bot_status", {})
    if data.get("running"):
        return (
            f"Telegram bot: RUNNING\n"
            f"  Offset: {data.get('offset', 0)}\n"
            f"  Admin chat: {data.get('admin_chat_id', '?')}\n"
            f"  Client configured: {data.get('client_configured', False)}"
        )
    error = data.get("error", "")
    return f"Telegram bot: STOPPED{' — ' + error if error else ''}"


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

    # Start the research job
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{ECHO_BRAIN_URL}/api/echo/research",
            json={"question": question, "depth": depth},
        )
        if resp.status_code != 200:
            return f"Failed to start research: {resp.status_code} - {resp.text[:200]}"
        job_data = resp.json()
        job_id = job_data["job_id"]

    # Poll until complete (up to 120s)
    for _ in range(60):
        await asyncio.sleep(2)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{ECHO_BRAIN_URL}/api/echo/research/{job_id}")
            if resp.status_code != 200:
                continue
            data = resp.json()
            status = data.get("status", "")
            if status in ("complete", "failed"):
                break

    if status == "failed":
        return f"Research failed: {data.get('error_message', 'unknown error')}"

    if status != "complete":
        return f"Research timed out (status: {status}). Job ID: {job_id}"

    # Format the report
    report = data.get("report", {})
    lines = [f"Deep Research Report\n{'=' * 40}\n"]
    lines.append(f"Question: {question}")
    lines.append(f"Depth: {depth} | Iterations: {data.get('iterations', 0)} | "
                 f"Sources: {data.get('sources_consulted', 0)} | "
                 f"Time: {data.get('total_time_ms', 0):.0f}ms\n")

    lines.append(report.get("answer", "No answer generated."))

    sources = report.get("sources", [])
    if sources:
        lines.append(f"\n{'=' * 40}\nSources ({len(sources)}):\n")
        for s in sources:
            ref = s.get("ref", "?")
            stype = s.get("type", "?")
            title = s.get("title", "")
            url = s.get("url", "")
            line = f"[{ref}] ({stype}) {title}"
            if url:
                line += f"\n    {url}"
            lines.append(line)

    sub_qs = report.get("sub_questions", [])
    if sub_qs:
        lines.append(f"\nSub-questions explored:")
        for sq in sub_qs:
            lines.append(f"  - {sq}")

    return "\n".join(lines)


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
    data = await _call_echo_brain("session_summary", {
        "summary": summary,
        "topics": topics,
        "project": project,
        "decisions": decisions,
    })
    if data.get("stored"):
        facts = data.get("facts_stored", 0)
        return f"Session summary stored (memory_id: {data.get('memory_id', '?')}, {facts} decisions saved as facts)"
    return f"Failed to store summary: {data.get('error', 'unknown error')}"


CREDIT_MONITOR_URL = "http://localhost:8400"


@mcp.tool()
async def credit_dashboard() -> str:
    """Get credit monitoring dashboard: accounts, alerts, scores, and Treasury rates."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{CREDIT_MONITOR_URL}/api/internal/dashboard")
            if resp.status_code != 200:
                return f"Credit monitor unavailable: {resp.status_code}"
            data = resp.json()

        lines = ["Credit Monitor Dashboard\n"]

        # Accounts
        accounts = data.get("accounts", [])
        if accounts:
            lines.append(f"Accounts ({len(accounts)}):")
            for acc in accounts[:10]:
                bal = acc.get("current_balance", "?")
                lines.append(f"  {acc['name']} ({acc.get('type','')}/{acc.get('subtype','')}) — ${bal}")

        # Credit scores
        scores = data.get("creditScores", [])
        if scores:
            lines.append(f"\nCredit Scores:")
            for s in scores:
                lines.append(f"  {s['source']}: {s['credit_score']} ({s.get('score_model','')}) — {s['created_at']}")

        # Alerts
        alerts = data.get("alerts", [])
        if alerts:
            lines.append(f"\nAlerts ({len(alerts)}):")
            for a in alerts[:5]:
                lines.append(f"  [{a.get('severity','?')}] {a.get('title','')}: {a.get('description','')[:80]}")

        # Treasury rates
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
        lines.append(f"Average Rate: {avg.get('current', '?')}% (previous: {avg.get('previous', '?')}%, change: {avg.get('change', '?')}%)")
        lines.append(f"Trend: {data.get('trend', '?')}")

        marketable = data.get("marketableSecurities", [])
        if marketable:
            lines.append(f"\nMarketable Securities ({len(marketable)} records)")
            for s in marketable[:5]:
                lines.append(f"  {s.get('security_desc', '?')}: {s.get('avg_interest_rate_amt', '?')}%")

        return "\n".join(lines)
    except Exception as e:
        return f"Treasury rates error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
