"""
Telegram Bot Listener for Echo Brain.

Standalone async polling loop that reads incoming Telegram messages,
routes commands, and dispatches natural language questions to the
ReasoningEngine for LLM-powered answers.
"""

import asyncio
import logging
import re
from datetime import datetime

from src.integrations.telegram_client import TelegramClient, CHAT_ID

logger = logging.getLogger("echo-brain.telegram-bot")


class TelegramBot:
    """Telegram bot listener with long-polling and reasoning integration."""

    def __init__(self):
        self._client = TelegramClient()
        self._running = False
        self._task: asyncio.Task | None = None
        self._offset = 0
        self._admin_chat_id = int(CHAT_ID)

    async def start(self):
        """Initialize the client and start the polling loop."""
        ok = await self._client.initialize()
        if not ok:
            logger.warning("Telegram client not configured — bot listener disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Telegram bot listener started")

    async def stop(self):
        """Stop the polling loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Telegram bot listener stopped")

    def get_status(self) -> dict:
        """Return current bot status for MCP introspection."""
        return {
            "running": self._running,
            "offset": self._offset,
            "admin_chat_id": str(self._admin_chat_id),
            "client_configured": self._client.is_configured,
        }

    # ── Polling loop ──────────────────────────────────────────────────

    async def _poll_loop(self):
        """Long-poll Telegram for updates, dispatch each one."""
        while self._running:
            try:
                updates = await self._client.get_updates(
                    offset=self._offset, timeout=30
                )
                for update in updates:
                    self._offset = update["update_id"] + 1
                    try:
                        await self._handle_update(update)
                    except Exception as e:
                        logger.error(f"Error handling update {update.get('update_id')}: {e}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Poll loop error: {e}")
                await asyncio.sleep(5)

    # ── Update routing ────────────────────────────────────────────────

    async def _handle_update(self, update: dict):
        """Route an incoming update to the right handler."""
        message = update.get("message")
        if not message:
            return

        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()

        logger.info(f"Incoming message from chat_id={chat_id}: {text[:80]!r}")

        # Security gate — only respond to admin
        if chat_id != self._admin_chat_id:
            logger.warning(f"Blocked message from non-admin chat_id={chat_id} (expected {self._admin_chat_id})")
            return

        if not text:
            return

        if text.startswith("/"):
            await self._handle_command(text, chat_id)
        else:
            await self._handle_question(text, chat_id)

    # ── Helpers ───────────────────────────────────────────────────────

    async def _reply(self, chat_id: int, text: str):
        """Send a plain-text reply, truncating to Telegram's 4096 limit."""
        if len(text) > 4000:
            text = text[:4000] + "\n... (truncated)"
        await self._client.send_message(text, parse_mode="", chat_id=str(chat_id))

    async def _keep_typing(self, chat_id: int, event: asyncio.Event):
        """Send 'typing' action every 4s until event is set."""
        chat_id_str = str(chat_id)
        while not event.is_set():
            await self._client.send_chat_action("typing", chat_id=chat_id_str)
            try:
                await asyncio.wait_for(event.wait(), timeout=4)
            except asyncio.TimeoutError:
                pass

    # ── Command dispatch ──────────────────────────────────────────────

    COMMANDS = {
        "/start": "help", "/help": "help",
        "/status": "status",
        "/briefing": "briefing",
        "/remind": "remind",
        "/search": "search",
        "/facts": "facts",
        "/remember": "remember",
        "/generate": "generate",
        "/ollama": "ollama",
        "/photos": "photos",
        "/fetch": "fetch",
    }

    async def _handle_command(self, text: str, chat_id: int):
        """Dispatch slash commands."""
        parts = text.split(None, 1)
        cmd = parts[0].lower().split("@")[0]  # strip @botname suffix
        arg = parts[1].strip() if len(parts) > 1 else ""

        handler_name = self.COMMANDS.get(cmd)
        if handler_name:
            handler = getattr(self, f"_cmd_{handler_name}")
            await handler(arg, chat_id)
        else:
            await self._reply(chat_id, f"Unknown command: {cmd}\nUse /help for available commands.")

    # ── Commands ──────────────────────────────────────────────────────

    async def _cmd_help(self, arg: str, chat_id: int):
        msg = (
            "Echo Brain Bot\n\n"
            "Commands:\n"
            "/briefing - Send daily briefing now\n"
            "/status - Tower service health\n"
            "/search <query> - Search 61K+ memory vectors\n"
            "/facts <topic> - Get structured facts\n"
            "/remember <text> - Store a new memory\n"
            "/remind HH:MM message - Schedule a reminder\n"
            "/generate <character> [count] - Generate images\n"
            "/ollama [list|running] - Ollama model status\n"
            "/photos <query> - Search photo library\n"
            "/fetch <url> - Fetch and summarize a URL\n"
            "/help - This message\n\n"
            "Or just type a question."
        )
        await self._reply(chat_id, msg)

    async def _cmd_briefing(self, arg: str, chat_id: int):
        """Force-send the daily briefing right now."""
        done_event = asyncio.Event()
        typing_task = asyncio.create_task(self._keep_typing(chat_id, done_event))
        try:
            from src.autonomous.workers.daily_briefing_worker import DailyBriefingWorker
            worker = DailyBriefingWorker()
            # Bypass time check — build and send directly
            sections = []
            events = await worker._get_calendar_events()
            sections.append(worker._fmt_calendar(events))
            balances, transactions = await worker._get_finance()
            sections.append(worker._fmt_finance(balances, transactions))
            inbox_unread = await worker._get_inbox_unread()
            sections.append(worker._fmt_email(inbox_unread))
            reminders = await worker._get_pending_reminders()
            sections.append(worker._fmt_reminders(reminders))

            now = datetime.now()
            briefing = f"Daily Briefing — {now.strftime('%A, %B %d %H:%M')}\n\n" + "\n\n".join(s for s in sections if s)

            done_event.set()
            await typing_task
            await self._reply(chat_id, briefing)
        except Exception as e:
            done_event.set()
            await typing_task
            await self._reply(chat_id, f"Briefing error: {e}")

    async def _cmd_status(self, arg: str, chat_id: int):
        await self._client.send_chat_action("typing", chat_id=str(chat_id))
        try:
            from src.integrations.mcp_service import mcp_service
            data = await mcp_service.check_services()
            lines = [f"Overall: {data.get('overall_status', '?')}"]
            for svc in data.get("services", []):
                icon = "OK" if svc["status"] == "healthy" else "DOWN"
                line = f"  [{icon}] {svc['name']}: {svc['latency_ms']}ms"
                if svc.get("error"):
                    line += f" - {svc['error']}"
                lines.append(line)
            res = data.get("resources", {})
            lines.append(
                f"  CPU: {res.get('cpu_percent', 0)}%  "
                f"RAM: {res.get('memory_percent', 0)}%  "
                f"Disk: {res.get('disk_percent', 0)}%"
            )
            await self._reply(chat_id, "\n".join(lines))
        except Exception as e:
            await self._reply(chat_id, f"Status check failed: {e}")

    async def _cmd_remind(self, arg: str, chat_id: int):
        match = re.match(r"(\d{1,2}):(\d{2})\s+(.+)", arg)
        if not match:
            await self._reply(chat_id, "Usage: /remind HH:MM your message here")
            return

        hour, minute, message = int(match.group(1)), int(match.group(2)), match.group(3)
        now = datetime.now()
        remind_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if remind_dt <= now:
            from datetime import timedelta
            remind_dt += timedelta(days=1)

        try:
            from src.integrations.mcp_service import mcp_service
            result = await mcp_service.schedule_reminder(
                message=message,
                remind_at=remind_dt.isoformat(),
                title="Telegram Reminder",
                channel="telegram",
            )
            if result.get("scheduled"):
                await self._reply(chat_id, f"Reminder set for {remind_dt.strftime('%Y-%m-%d %H:%M')}:\n{message}")
            else:
                await self._reply(chat_id, f"Failed to schedule: {result.get('error', 'unknown')}")
        except Exception as e:
            await self._reply(chat_id, f"Reminder error: {e}")

    async def _cmd_search(self, arg: str, chat_id: int):
        if not arg:
            await self._reply(chat_id, "Usage: /search <query>")
            return
        await self._client.send_chat_action("typing", chat_id=str(chat_id))
        try:
            from src.integrations.mcp_service import mcp_service
            data = await mcp_service.search_memory(arg, limit=5)
            results = data.get("results", data) if isinstance(data, dict) else data
            if not results:
                await self._reply(chat_id, f"No results for: {arg}")
                return
            items = results if isinstance(results, list) else results.get("results", [])
            lines = []
            for r in items[:5]:
                score = r.get("score", 0)
                content = r.get("content", "")[:200]
                source = r.get("source", "")
                lines.append(f"[{score:.2f}] ({source})\n{content}")
            await self._reply(chat_id, "\n---\n".join(lines) or "No readable results.")
        except Exception as e:
            await self._reply(chat_id, f"Search error: {e}")

    async def _cmd_facts(self, arg: str, chat_id: int):
        if not arg:
            await self._reply(chat_id, "Usage: /facts <topic>")
            return
        await self._client.send_chat_action("typing", chat_id=str(chat_id))
        try:
            from src.integrations.mcp_service import mcp_service
            data = await mcp_service.get_facts(arg, limit=10)
            if not data:
                await self._reply(chat_id, f"No facts found for: {arg}")
                return
            items = data if isinstance(data, list) else []
            lines = []
            for f in items[:10]:
                content = f.get("content", "")[:200]
                conf = f.get("confidence", 0)
                lines.append(f"[{conf:.1f}] {content}")
            await self._reply(chat_id, "\n".join(lines) or "No facts found.")
        except Exception as e:
            await self._reply(chat_id, f"Facts error: {e}")

    async def _cmd_remember(self, arg: str, chat_id: int):
        if not arg:
            await self._reply(chat_id, "Usage: /remember <text to store>")
            return
        try:
            from src.integrations.mcp_service import mcp_service
            point_id = await mcp_service.store_memory(arg, type_="telegram")
            if point_id:
                await self._reply(chat_id, f"Stored: {arg[:100]}...")
            else:
                await self._reply(chat_id, "Failed to store memory.")
        except Exception as e:
            await self._reply(chat_id, f"Store error: {e}")

    async def _cmd_generate(self, arg: str, chat_id: int):
        if not arg:
            await self._reply(chat_id, "Usage: /generate <character_slug> [count]")
            return
        parts = arg.split(None, 1)
        slug = parts[0]
        count = 1
        if len(parts) > 1:
            try:
                count = max(1, min(int(parts[1]), 5))
            except ValueError:
                pass
        await self._client.send_chat_action("typing", chat_id=str(chat_id))
        try:
            from src.integrations.mcp_service import mcp_service
            data = await mcp_service.trigger_generation(character_slug=slug, count=count)
            if data.get("errors"):
                await self._reply(chat_id, f"Errors: {'; '.join(data['errors'])}")
            elif data.get("generated", 0) > 0:
                ids = [r.get("prompt_id", "?") for r in data.get("results", [])]
                await self._reply(chat_id, f"Triggered {data['generated']} generation(s) for {slug}\nIDs: {', '.join(str(i) for i in ids)}")
            else:
                await self._reply(chat_id, "No images generated.")
        except Exception as e:
            await self._reply(chat_id, f"Generation error: {e}")

    async def _cmd_ollama(self, arg: str, chat_id: int):
        action = arg.strip().lower() if arg else "list"
        if action not in ("list", "running"):
            action = "list"
        await self._client.send_chat_action("typing", chat_id=str(chat_id))
        try:
            import httpx
            ollama_url = "http://localhost:11434"
            if action == "list":
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"{ollama_url}/api/tags")
                    data = resp.json()
                lines = []
                for m in data.get("models", []):
                    size = round(m.get("size", 0) / 1e9, 1)
                    lines.append(f"  {m['name']} ({size}GB)")
                await self._reply(chat_id, f"Ollama models ({len(lines)}):\n" + "\n".join(lines) if lines else "No models found.")
            else:  # running
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"{ollama_url}/api/ps")
                    data = resp.json()
                running = data.get("models", [])
                if not running:
                    await self._reply(chat_id, "No models currently loaded.")
                else:
                    lines = []
                    for m in running:
                        vram = round(m.get("size_vram", 0) / 1e9, 1)
                        lines.append(f"  {m.get('name')} (VRAM: {vram}GB)")
                    await self._reply(chat_id, f"Running models:\n" + "\n".join(lines))
        except Exception as e:
            await self._reply(chat_id, f"Ollama error: {e}")

    async def _cmd_photos(self, arg: str, chat_id: int):
        if not arg:
            await self._reply(chat_id, "Usage: /photos <query>")
            return
        await self._client.send_chat_action("typing", chat_id=str(chat_id))
        try:
            from src.services.photo_dedup_service import PhotoDedupService
            svc = PhotoDedupService()
            results = await svc.search_media(query=arg, limit=5)
            if not results:
                await self._reply(chat_id, f"No photos/videos found for: {arg}")
                return
            lines = []
            for r in results[:5]:
                score = r.get("score", 0)
                media = r.get("media_type", "photo")
                fname = r.get("filename", "")
                desc = r.get("description", "")[:150]
                lines.append(f"[{score:.2f}] [{media}] {fname}\n  {desc}")
            await self._reply(chat_id, "\n---\n".join(lines))
        except Exception as e:
            await self._reply(chat_id, f"Photo search error: {e}")

    async def _cmd_fetch(self, arg: str, chat_id: int):
        if not arg:
            await self._reply(chat_id, "Usage: /fetch <url>")
            return
        url = arg.split()[0]
        await self._client.send_chat_action("typing", chat_id=str(chat_id))
        try:
            from src.integrations.mcp_service import mcp_service
            data = await mcp_service.web_fetch(url=url, max_length=3000)
            if "error" in data:
                await self._reply(chat_id, f"Fetch error: {data['error']}")
            else:
                content = data.get("content", "")[:3000]
                ct = data.get("content_type", "?")
                trunc = " (truncated)" if data.get("truncated") else ""
                await self._reply(chat_id, f"URL: {data.get('url', url)}\nType: {ct}{trunc}\n\n{content}")
        except Exception as e:
            await self._reply(chat_id, f"Fetch error: {e}")

    # ── Natural language questions ────────────────────────────────────

    async def _handle_question(self, text: str, chat_id: int):
        """Send the question through ReasoningEngine and reply."""
        session_id = f"telegram_{chat_id}"

        # Start typing indicator immediately
        done_event = asyncio.Event()
        typing_task = asyncio.create_task(self._keep_typing(chat_id, done_event))

        try:
            from src.intelligence.reasoner import get_reasoning_engine
            from src.services.conversation_service import get_conversation_service

            conv = get_conversation_service()
            reasoner = get_reasoning_engine()

            # Save user turn
            await conv.store_turn(session_id, "user", text)

            # Process through reasoning engine
            result = await reasoner.process(
                query=text,
                allow_actions=False,
                session_id=session_id,
            )

            response_text = result.response if hasattr(result, "response") else str(result)

            # Save assistant turn
            await conv.store_turn(session_id, "assistant", response_text)

            # Stop typing before sending reply
            done_event.set()
            await typing_task

            await self._reply(chat_id, response_text)

        except Exception as e:
            done_event.set()
            await typing_task
            logger.error(f"Question handling failed: {e}")
            await self._reply(chat_id, f"Sorry, I hit an error processing that: {e}")
