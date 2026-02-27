"""Bridge to Family Credit Monitor for credit intelligence."""

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

CREDIT_MONITOR_URL = "http://localhost:8400"


class CreditService:
    """Proxy to Family Credit Monitor internal API."""

    def __init__(self, base_url: str = CREDIT_MONITOR_URL):
        self.base_url = base_url

    async def get_dashboard(self) -> Dict[str, Any]:
        """Get aggregated credit dashboard data."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/internal/dashboard")
            resp.raise_for_status()
            return resp.json()

    async def get_credit_scores(self) -> List[Dict[str, Any]]:
        """Get latest credit scores from all bureaus."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/internal/credit/scores")
            resp.raise_for_status()
            return resp.json()

    async def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get credit alerts, optionally filtered by severity."""
        params: Dict[str, str] = {}
        if severity:
            params["severity"] = severity
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{self.base_url}/api/internal/alerts", params=params
            )
            resp.raise_for_status()
            return resp.json()

    async def get_treasury_rates(self) -> Dict[str, Any]:
        """Get current Treasury interest rates and trends."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self.base_url}/api/internal/treasury-rates")
            resp.raise_for_status()
            return resp.json()

    async def get_report(self, report_type: str = "treasury") -> Dict[str, Any]:
        """Get a financial report."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.base_url}/api/internal/reports/{report_type}"
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


# Singleton
_credit_service: Optional[CreditService] = None


def get_credit_service() -> CreditService:
    global _credit_service
    if _credit_service is None:
        _credit_service = CreditService()
    return _credit_service
