"""
Daily Briefing Worker — sends a morning Telegram briefing at a configured time.

Collects: weather, calendar events + next upcoming event, financial balances,
recent transactions, email (unread count + top senders/subjects),
pending reminders. Sends a single digest message.

Registered at 1-minute interval; fires once per day at BRIEFING_HOUR:BRIEFING_MINUTE
in the configured LOCAL_TZ (default: America/Los_Angeles).
"""

import logging
import os
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger("echo-brain.daily-briefing")

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
BRIEFING_HOUR = 6   # 6:30 AM Pacific (PST or PDT, handled automatically)
BRIEFING_MINUTE = 30


class DailyBriefingWorker:

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable required")
        self._last_briefing_date: date | None = None

    _MARKER_FILE = "/tmp/echo-brain-briefing-last-date"

    async def _already_sent_today(self, today: date) -> bool:
        """Check file marker to see if briefing was already sent today (survives restarts)."""
        try:
            from pathlib import Path
            marker = Path(self._MARKER_FILE)
            if marker.exists():
                stored = marker.read_text().strip()
                return stored == today.isoformat()
        except Exception as e:
            logger.warning(f"[DailyBriefing] Dedup check failed: {e}")
        return False

    def _mark_sent(self, today: date):
        """Write today's date to marker file."""
        from pathlib import Path
        Path(self._MARKER_FILE).write_text(today.isoformat())

    async def run_cycle(self):
        now_local = datetime.now(LOCAL_TZ)

        # Fire within a 5-minute window (covers scheduler drift / missed ticks)
        if now_local.hour != BRIEFING_HOUR or not (BRIEFING_MINUTE <= now_local.minute < BRIEFING_MINUTE + 5):
            return
        if self._last_briefing_date == now_local.date():
            return
        if await self._already_sent_today(now_local.date()):
            self._last_briefing_date = now_local.date()
            return

        logger.info("[DailyBriefing] Building morning briefing...")
        self._last_briefing_date = now_local.date()
        now = now_local  # use local time for formatting

        try:
            sections = []

            # ── Weather ──────────────────────────────────────────────
            try:
                weather = await self._get_weather()
                if weather:
                    sections.append(weather)
            except Exception as e:
                logger.warning(f"[DailyBriefing] Weather section failed: {e}")
                sections.append("Weather: unavailable")

            # ── Calendar (today + next upcoming) ─────────────────────
            try:
                events = await self._get_calendar_events()
                next_event = await self._get_next_event()
                sections.append(self._fmt_calendar(events, next_event))
            except Exception as e:
                logger.warning(f"[DailyBriefing] Calendar section failed: {e}")
                sections.append("Calendar: unavailable")

            # ── Finance (reads from local DB only, no Plaid API calls) ──
            try:
                balances, transactions = await self._get_finance()
                sections.append(self._fmt_finance(balances, transactions))
            except Exception as e:
                logger.warning(f"[DailyBriefing] Finance section failed: {e}")
                sections.append("Finance: unavailable")

            # ── Email ─────────────────────────────────────────────────
            try:
                email_section = await self._get_email_summary()
                sections.append(email_section)
            except Exception as e:
                logger.warning(f"[DailyBriefing] Email section failed: {e}")
                sections.append("Email: unavailable")

            # ── Reminders ─────────────────────────────────────────────
            try:
                reminders = await self._get_pending_reminders()
                sections.append(self._fmt_reminders(reminders))
            except Exception as e:
                logger.warning(f"[DailyBriefing] Reminders section failed: {e}")

            # ── Assemble & send ───────────────────────────────────────
            briefing = f"Good morning — {now.strftime('%A, %B %d')}\n\n" + "\n\n".join(s for s in sections if s)

            await self._send(briefing)
            self._mark_sent(now_local.date())
            logger.info("[DailyBriefing] Briefing sent")

        except Exception as e:
            logger.error(f"[DailyBriefing] Failed to build/send briefing: {e}")

    # ── Data collectors ───────────────────────────────────────────

    async def _get_weather(self) -> str | None:
        """Fetch weather from Open-Meteo (free, no API key) for Vista, CA."""
        # Vista, CA: 33.20, -117.24
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=33.20&longitude=-117.24"
            "&current=temperature_2m,apparent_temperature,weather_code"
            "&daily=temperature_2m_max,temperature_2m_min"
            "&temperature_unit=fahrenheit"
            "&timezone=America/Los_Angeles"
            "&forecast_days=1"
        )
        WMO = {
            0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
            55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
            81: "Moderate showers", 82: "Heavy showers", 95: "Thunderstorm",
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                cur = data.get("current", {})
                daily = data.get("daily", {})
                temp = cur.get("temperature_2m")
                feels = cur.get("apparent_temperature")
                code = cur.get("weather_code", 0)
                hi = daily.get("temperature_2m_max", [None])[0]
                lo = daily.get("temperature_2m_min", [None])[0]
                desc = WMO.get(code, "")
                parts = []
                if desc:
                    parts.append(desc)
                if temp is not None:
                    parts.append(f"{temp:.0f}F")
                if feels is not None and feels != temp:
                    parts.append(f"feels {feels:.0f}F")
                if hi is not None and lo is not None:
                    parts.append(f"H:{hi:.0f} L:{lo:.0f}")
                return f"Weather (Vista, CA): {' | '.join(parts)}"
        except Exception as e:
            logger.warning(f"[DailyBriefing] Weather unavailable: {e}")
        return None

    async def _get_next_event(self) -> str | None:
        """Get the next upcoming calendar event as a formatted string."""
        try:
            from src.integrations.google_calendar import get_calendar_bridge
            bridge = await get_calendar_bridge()
            if not bridge:
                return None
            events = await bridge.get_upcoming_events(hours_ahead=168, max_results=1)
            if not events:
                return None
            ev = events[0]
            summary = ev.get("summary", "No title")
            start_raw = ev.get("start", "")
            if isinstance(start_raw, dict):
                dt_str = start_raw.get("dateTime", start_raw.get("date", ""))
            else:
                dt_str = str(start_raw)
            if "T" in dt_str:
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    local_dt = dt.astimezone(LOCAL_TZ)
                    today = datetime.now(LOCAL_TZ).date()
                    if local_dt.date() == today:
                        when = f"today at {local_dt.strftime('%H:%M')}"
                    elif local_dt.date() == today + timedelta(days=1):
                        when = f"tomorrow at {local_dt.strftime('%H:%M')}"
                    else:
                        when = local_dt.strftime("%a %b %d at %H:%M")
                except ValueError:
                    when = dt_str
            else:
                when = dt_str
            return f"{summary} — {when}"
        except Exception as e:
            logger.warning(f"[DailyBriefing] Next event unavailable: {e}")
        return None

    async def _get_calendar_events(self) -> list[dict]:
        try:
            from src.integrations.google_calendar import get_calendar_bridge
            bridge = await get_calendar_bridge()
            if bridge:
                return await bridge.get_events_for_date(datetime.now(LOCAL_TZ))
        except Exception as e:
            logger.warning(f"[DailyBriefing] Calendar unavailable: {e}")
        return []

    async def _get_finance(self) -> tuple[list[dict], list[dict]]:
        """Get financial data from tower-auth DB endpoints (no live Plaid API calls)."""
        balances_list: list[dict] = []
        transactions_list: list[dict] = []
        import httpx

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get("http://127.0.0.1:8088/api/auth/plaid/db/accounts")
                if resp.status_code == 200:
                    balances_list = resp.json().get("accounts", [])
        except Exception as e:
            logger.warning(f"[DailyBriefing] DB accounts unavailable: {e}")

        try:
            yesterday = (datetime.now(LOCAL_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
            today = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "http://127.0.0.1:8088/api/auth/plaid/db/transactions",
                    params={"start_date": yesterday, "end_date": today, "limit": 50},
                )
                if resp.status_code == 200:
                    transactions_list = resp.json().get("transactions", [])
        except Exception as e:
            logger.warning(f"[DailyBriefing] DB transactions unavailable: {e}")

        return balances_list, transactions_list

    async def _get_email_summary(self) -> str:
        """Get unread count + top recent sender/subject lines."""
        try:
            from src.integrations.tower_auth_bridge import tower_auth
            service = await tower_auth.get_gmail_service()
            if not service:
                return "Email: unavailable"

            # Exact unread count
            label = service.users().labels().get(userId="me", id="INBOX").execute()
            unread = label.get("messagesUnread", 0)

            lines = [f"Email ({unread:,} unread)"]

            # Top recent unread messages — skip Google security noise
            skip_patterns = ["security alert", "google account was recovered"]
            msgs = service.users().messages().list(
                userId="me", q="is:unread in:inbox", maxResults=15
            ).execute()
            shown = 0
            for m in msgs.get("messages", []):
                if shown >= 5:
                    break
                try:
                    msg = service.users().messages().get(
                        userId="me", id=m["id"], format="metadata",
                        metadataHeaders=["Subject", "From"],
                    ).execute()
                    headers = {
                        h["name"]: h["value"]
                        for h in msg.get("payload", {}).get("headers", [])
                    }
                    sender = headers.get("From", "?")
                    if "<" in sender:
                        sender = sender.split("<")[0].strip().strip('"')
                    subject = headers.get("Subject", "(no subject)")
                    # Skip noisy automated emails
                    subj_lower = subject.lower().strip()
                    if any(p in subj_lower for p in skip_patterns):
                        continue
                    lines.append(f"  {sender}")
                    lines.append(f"    {subject[:70]}")
                    shown += 1
                except Exception:
                    continue
            if shown == 0:
                lines.append("  (no notable unread)")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"[DailyBriefing] Gmail unavailable: {e}")
            return "Email: unavailable"

    async def _get_pending_reminders(self) -> list[dict]:
        try:
            import asyncpg
            conn = await asyncpg.connect(self.db_url, timeout=5)
            try:
                rows = await conn.fetch("""
                    SELECT title, body, scheduled_for
                    FROM notifications
                    WHERE status = 'pending'
                      AND category = 'reminder'
                      AND scheduled_for IS NOT NULL
                      AND scheduled_for BETWEEN NOW() AND NOW() + INTERVAL '24 hours'
                    ORDER BY scheduled_for ASC
                    LIMIT 10
                """)
                return [dict(r) for r in rows]
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"[DailyBriefing] Reminders unavailable: {e}")
            return []

    # ── Formatters ────────────────────────────────────────────────

    def _fmt_calendar(self, events: list[dict], next_event: str | None = None) -> str:
        if not events and not next_event:
            return "Calendar: No events today"
        if not events:
            lines = ["Calendar: No events today"]
            if next_event:
                lines.append(f"  Next: {next_event}")
            return "\n".join(lines)
        lines = [f"Calendar ({len(events)} events)"]
        for ev in events[:8]:
            summary = ev.get("summary", "No title")
            start_raw = ev.get("start", "")
            # Parse time portion
            time_str = ""
            if isinstance(start_raw, dict):
                # All-day events have 'date', timed events have 'dateTime'
                dt_str = start_raw.get("dateTime", start_raw.get("date", ""))
            else:
                dt_str = str(start_raw)
            if "T" in dt_str:
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    time_str = dt.astimezone(LOCAL_TZ).strftime("%H:%M")
                except ValueError:
                    time_str = dt_str
            else:
                time_str = "all day"
            location = ev.get("location", "")
            loc_str = f" @ {location}" if location else ""
            lines.append(f"  {time_str} — {summary}{loc_str}")
        if len(events) > 8:
            lines.append(f"  ...and {len(events) - 8} more")
        if next_event:
            lines.append(f"  Next: {next_event}")
        return "\n".join(lines)

    def _fmt_finance(self, balances: list[dict], transactions: list[dict]) -> str:
        if not balances and not transactions:
            return "Finance: unavailable"

        def _bal(v) -> float:
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        # Group accounts by institution
        from collections import defaultdict
        by_inst: dict[str, list[dict]] = defaultdict(list)
        for acct in balances:
            inst = acct.get("institution_name", "Other")
            by_inst[inst].append(acct)

        lines = ["Finance"]
        total_cash = 0.0
        total_credit_owed = 0.0
        total_credit_avail = 0.0
        total_loans = 0.0
        total_insurance = 0.0

        # Deduplicate account names within each institution
        for inst, accounts in sorted(by_inst.items()):
            lines.append(f"  {inst}")
            # Count name occurrences for disambiguation
            name_counts: dict[str, int] = defaultdict(int)
            for a in accounts:
                name_counts[a.get("name", "Account")] += 1
            name_seen: dict[str, int] = defaultdict(int)

            for acct in accounts:
                name = acct.get("name", acct.get("official_name", "Account"))
                current = _bal(acct.get("current_balance"))
                avail = acct.get("available_balance")
                limit = acct.get("credit_limit")
                acct_type = acct.get("type", "")
                subtype = acct.get("subtype", "")

                # Disambiguate duplicate names
                if name_counts[name] > 1:
                    name_seen[name] += 1
                    label = f"{name} #{name_seen[name]}"
                else:
                    label = name

                if acct_type == "credit" or "credit card" in subtype:
                    limit_val = _bal(limit) if limit else 0
                    avail_val = _bal(avail) if avail else 0
                    lines.append(
                        f"    {label}: ${current:,.2f} owed"
                        + (f" | ${avail_val:,.2f} avail" if avail else "")
                        + (f" | ${limit_val:,.0f} limit" if limit else "")
                    )
                    total_credit_owed += current
                    total_credit_avail += avail_val
                elif acct_type == "loan":
                    loan_type = subtype.replace("_", " ").title() if subtype else "Loan"
                    lines.append(f"    {label} ({loan_type}): ${current:,.2f} owed")
                    total_loans += current
                elif "insurance" in subtype or acct_type == "investment":
                    if current > 0:
                        lines.append(f"    {label}: ${current:,.2f}")
                        total_insurance += current
                    # Skip zero-balance policies (e.g. term life)
                elif current > 0:
                    lines.append(f"    {label}: ${current:,.2f}")
                    total_cash += current

        # Totals
        lines.append("  ──")
        lines.append(f"  Cash/Savings: ${total_cash:,.2f}")
        if total_credit_owed > 0:
            lines.append(f"  Credit cards: ${total_credit_owed:,.2f} owed (${total_credit_avail:,.2f} avail)")
        if total_loans > 0:
            lines.append(f"  Loans: ${total_loans:,.2f} owed")
        if total_insurance > 0:
            lines.append(f"  Insurance CV: ${total_insurance:,.2f}")
        net = total_cash + total_insurance - total_credit_owed - total_loans
        lines.append(f"  Net: ${net:,.2f}")

        # Recent transactions
        if transactions:
            lines.append(f"  Recent: {len(transactions)} transactions")
            for txn in transactions[:5]:
                amt = _bal(txn.get("amount", 0))
                merchant = txn.get("merchant_name") or txn.get("name", "?")
                lines.append(f"    ${abs(amt):,.2f} — {merchant}")
            if len(transactions) > 5:
                lines.append(f"    ...and {len(transactions) - 5} more")

        return "\n".join(lines)

    def _fmt_reminders(self, reminders: list[dict]) -> str:
        if not reminders:
            return ""
        lines = [f"Reminders today ({len(reminders)})"]
        for r in reminders:
            sched = r.get("scheduled_for")
            if sched:
                if sched.tzinfo is None:
                    sched = sched.replace(tzinfo=ZoneInfo("UTC"))
                time_str = sched.astimezone(LOCAL_TZ).strftime("%H:%M")
            else:
                time_str = "?"
            body = r.get("body", r.get("title", ""))[:80]
            lines.append(f"  {time_str} — {body}")
        return "\n".join(lines)

    # ── Send ──────────────────────────────────────────────────────

    async def _send(self, message: str):
        from src.services.notification_service import (
            get_notification_service,
            NotificationType,
            NotificationChannel,
        )
        service = await get_notification_service()
        if service:
            await service.send_notification(
                message=message,
                title="Daily Briefing",
                notification_type=NotificationType.INFO,
                channels=[NotificationChannel.TELEGRAM],
            )
        else:
            logger.error("[DailyBriefing] Notification service unavailable")
