# Credit Monitoring Integration with Echo Brain

**Date:** 2026-02-27
**Status:** Planning — code-complete system at `/opt/family-credit-monitor/` needs deployment + Echo Brain bridge

---

## Current State

### What Exists (Code-Complete, Not Deployed)

**Location:** `/opt/family-credit-monitor/`
**Stack:** Node.js/Express + Vue 3 + PostgreSQL + Redis
**Port:** 8400 (configured, not running)

| Component | Status |
|---|---|
| Database (`family_credit_monitor`) | Created, 17 tables with data schema |
| Redis | Running on port 6379 |
| Backend code | Written, TypeScript, **NOT compiled** (`dist/` missing) |
| Frontend code | Written, Vue 3 + Vite, **NOT built** (`dist/` missing) |
| Nginx config | **NOT configured** for `/credit-monitor` |
| Systemd service file | Exists but has **wrong path** (`/home/patrick/` vs `/opt/`) |
| node_modules | Installed (pnpm) |

### Services Inside Family Credit Monitor

1. **Plaid Connector** — Account linking, balance sync, transaction ingestion
   - Uses same Plaid credentials as tower-auth
   - 4-hour sync cycle

2. **Credit Monitor** — Bureau integration (Experian, Equifax, TransUnion)
   - Soft/hard credit pull management
   - **API keys NOT yet configured** — needs `EXPERIAN_API_KEY`, `EQUIFAX_API_KEY`, `TRANSUNION_API_KEY`

3. **Public Data Aggregator** — CFPB complaints, HMDA lending data, Treasury rates
   - FREE APIs, no keys needed
   - 6-hour refresh cycle with Redis cache (6-24hr TTL)

4. **ML Analyzer** — TensorFlow.js fraud detection, risk prediction
   - Transaction pattern analysis, merchant risk scoring, geographic anomalies
   - Needs training data (cold start)

5. **Alert Service** — Multi-tier notification pipeline
   - Critical (immediate): fraud, data leaks
   - High (1hr): major credit changes
   - Medium (daily): optimization opportunities
   - Low (weekly): general insights
   - Channels: in-app, email, SMS, WebSocket

6. **Report Generator** — Comprehensive financial reports
7. **Scheduler** — Bull queues + node-cron for background jobs

### What Already Works in Tower (via tower-auth + Plaid)

- 5 linked institutions: Capital One, SCU, Northwestern Mutual, USAA, Chase
- Real-time transaction syncing
- Balance history (90-day snapshots)
- Spending analysis
- Net worth calculation
- Daily briefing at 6:30 AM (includes top 3 transactions)
- Dashboard at `/financial` with 6 tabs

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Echo Brain (port 8309)                │
│                                                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ MCP Tools     │  │ Proactive     │  │ Daily        │ │
│  │ credit_score  │  │ Monitor       │  │ Briefing     │ │
│  │ credit_alerts │  │ Worker        │  │ (enhanced)   │ │
│  │ credit_report │  │ (5 min cycle) │  │              │ │
│  └──────┬───────┘  └──────┬────────┘  └──────┬───────┘ │
│         │                  │                   │         │
│         └──────────────────┼───────────────────┘         │
│                            │                              │
│                   ┌────────▼────────┐                    │
│                   │ Credit Bridge   │                    │
│                   │ Service (new)   │                    │
│                   └────────┬────────┘                    │
└────────────────────────────┼─────────────────────────────┘
                             │ HTTP
                    ┌────────▼────────┐
                    │ Family Credit    │
                    │ Monitor          │
                    │ (port 8400)      │
                    │                  │
                    │ ├─ Plaid         │
                    │ ├─ Credit Bureau │
                    │ ├─ Public Data   │
                    │ ├─ ML Analysis   │
                    │ └─ Alerts        │
                    └─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼───┐   ┌─────▼────┐  ┌─────▼────┐
         │ Plaid   │   │ CFPB     │  │ Treasury │
         │ API     │   │ API      │  │ API      │
         └────────┘   └──────────┘  └──────────┘
```

---

## Deployment Steps (Phase 1: Get It Running)

### Step 1: Fix systemd service file

```ini
# /etc/systemd/system/family-credit-monitor.service
[Unit]
Description=Family Credit Monitor
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=patrick
WorkingDirectory=/opt/family-credit-monitor/backend
ExecStart=/usr/bin/node dist/server.js
Restart=always
RestartSec=10
Environment=NODE_ENV=production
Environment=PORT=8400

[Install]
WantedBy=multi-user.target
```

### Step 2: Compile backend

```bash
cd /opt/family-credit-monitor/backend
npx tsc  # or npm run build
```

### Step 3: Build frontend

```bash
cd /opt/family-credit-monitor/frontend
npm run build  # Creates dist/
```

### Step 4: Configure nginx

Add to `/etc/nginx/sites-enabled/tower-https`:
```nginx
location /credit-monitor/ {
    proxy_pass http://localhost:8400/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
}
```

### Step 5: Start service

```bash
sudo systemctl daemon-reload
sudo systemctl enable family-credit-monitor
sudo systemctl start family-credit-monitor
```

### Step 6: Verify

```bash
curl -s http://localhost:8400/health
```

---

## Echo Brain Integration (Phase 2: Bridge Service)

### New File: `src/services/credit_service.py`

```python
"""Bridge to Family Credit Monitor for credit intelligence."""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

CREDIT_MONITOR_URL = "http://localhost:8400"


class CreditService:
    """Proxy to Family Credit Monitor API."""

    def __init__(self, base_url: str = CREDIT_MONITOR_URL):
        self.base_url = base_url

    async def get_dashboard(self) -> dict:
        """Get aggregated credit dashboard data."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/dashboard")
            resp.raise_for_status()
            return resp.json()

    async def get_credit_scores(self) -> dict:
        """Get latest credit scores from all bureaus."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/credit/scores")
            resp.raise_for_status()
            return resp.json()

    async def get_alerts(self, severity: Optional[str] = None) -> list:
        """Get credit alerts, optionally filtered by severity."""
        params = {}
        if severity:
            params["severity"] = severity
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{self.base_url}/api/alerts", params=params
            )
            resp.raise_for_status()
            return resp.json()

    async def get_public_data(self) -> dict:
        """Get public data insights (CFPB, HMDA, Treasury rates)."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/public-data")
            resp.raise_for_status()
            return resp.json()

    async def get_report(self, report_type: str = "summary") -> dict:
        """Generate a credit report."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.base_url}/api/reports/{report_type}"
            )
            resp.raise_for_status()
            return resp.json()

    async def health(self) -> bool:
        """Check if credit monitor is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
```

### New MCP Tools

Add to both `mcp_server/stdio_server.py` and `main.py` MCP handler:

- `credit_dashboard()` — Get credit overview (scores, alerts, accounts)
- `credit_alerts(severity)` — Get credit/fraud alerts
- `treasury_rates()` — Get current Treasury rates and trends

### Enhance Daily Briefing

In `src/autonomous/workers/daily_briefing_worker.py`:

```python
# Add credit section to morning briefing
async def _get_credit_section(self) -> str:
    """Fetch credit highlights for daily briefing."""
    try:
        credit_svc = CreditService()
        alerts = await credit_svc.get_alerts(severity="high")
        public_data = await credit_svc.get_public_data()

        lines = ["Credit Monitor:"]
        if alerts:
            lines.append(f"  {len(alerts)} alert(s) requiring attention")
            for a in alerts[:3]:
                lines.append(f"  - {a['title']}: {a['description'][:80]}")

        # Treasury rates
        rates = public_data.get("treasury_rates", {})
        if rates:
            lines.append(f"  Treasury: 10Y={rates.get('10y', '?')}%, 30Y={rates.get('30y', '?')}%")

        return "\n".join(lines)
    except Exception as e:
        return f"Credit Monitor: unavailable ({e})"
```

### New Autonomous Worker: Credit Proactive Monitor

```python
# src/autonomous/workers/credit_monitor_worker.py
class CreditMonitorWorker(BaseWorker):
    """Proactive credit monitoring with anomaly detection."""
    interval = timedelta(hours=1)

    async def execute(self):
        credit_svc = CreditService()

        # Check for new alerts
        alerts = await credit_svc.get_alerts(severity="critical")
        for alert in alerts:
            if not await self._already_notified(alert["id"]):
                await self.notification_service.send(
                    message=f"CREDIT ALERT: {alert['title']}\n{alert['description']}",
                    channels=["telegram", "ntfy"],
                    priority="high",
                )
                await self._mark_notified(alert["id"])

        # Store credit facts for Echo Brain knowledge
        dashboard = await credit_svc.get_dashboard()
        if dashboard.get("credit_score"):
            await self.facts.upsert(
                subject="patrick",
                predicate="credit_score",
                object=str(dashboard["credit_score"]),
                confidence=0.95,
                source="credit_monitor"
            )
```

### Add to Autonomous Health Monitoring

Already done — add to `config/autonomous_services.yaml`:
```yaml
family-credit-monitor:
  enabled: true
  port: 8400
  health_endpoint: "/health"
  restart_command: "sudo systemctl restart family-credit-monitor"
  max_retries: 3
  backoff_base: 30
  critical: false
```

---

## Integration with Existing Financial Stack

```
Existing Stack:
├── tower-auth (Plaid) → account linking, transactions, balances
├── tower-dashboard → /financial UI (6 tabs)
├── tower-echo-brain → /api/finance/* proxy, daily briefing
└── tower-crypto-trader → ML trading recommendations

+ Family Credit Monitor adds:
├── Credit bureau scores (when API keys configured)
├── Public data intelligence (CFPB, HMDA, Treasury — FREE)
├── ML fraud detection
├── Multi-tier alert pipeline
├── Credit optimization recommendations
└── Family financial coordination
```

**Key insight:** The public data features (CFPB complaints, HMDA lending patterns, Treasury rates) are **immediately available with zero API keys** and provide significant value — they can inform mortgage decisions, identify problematic lenders, and track interest rate trends.

---

## Priority

**Where this fits in the roadmap:** Tier 2 (Beat Perplexity), specifically enhancing Proactive Monitoring (2.3).

Perplexity has Finance tools (stock quotes, earnings hub) but **cannot** do:
- Personal credit score tracking
- Fraud detection on your accounts
- CFPB complaint monitoring for your lenders
- Proactive alerts when something looks wrong
- Family financial coordination

**Recommended order within the credit monitoring integration:**
1. Deploy the service (2-3 hours) — compile, nginx, systemd
2. Echo Brain bridge service (2-3 hours) — proxy endpoints
3. MCP tools (1-2 hours) — credit_dashboard, credit_alerts, treasury_rates
4. Daily briefing enhancement (1 hour)
5. Proactive monitoring worker (2-3 hours)
6. Credit bureau API keys (when ready, independent of above)

**Total effort:** 8-12 hours
