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

mcp = FastMCP("echo-brain", instructions="Echo Brain memory system — search memories, get/store facts, explore knowledge graph, manage Ollama models, search photos/videos. Telegram bot listener accepts messages and answers questions via reasoning engine.")


async def _call_echo_brain(tool_name: str, arguments: dict) -> dict:
    """Forward a tool call to the Echo Brain HTTP MCP endpoint."""
    async with httpx.AsyncClient(timeout=60) as client:
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
