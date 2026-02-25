"""Circuit breaker monitoring, event log, Prometheus metrics."""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .service_breakers import ServiceBreakerManager

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerEvent:
    service_name: str
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class ServiceHealthMetrics:
    service_name: str
    status: str
    failure_rate: float
    avg_latency_ms: float
    uptime_percentage: float
    last_state_change: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "status": self.status,
            "failure_rate": self.failure_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "uptime_percentage": self.uptime_percentage,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
        }


class CircuitBreakerMonitor:
    """Observability layer over ServiceBreakerManager."""

    def __init__(self, service_manager: ServiceBreakerManager):
        self.service_manager = service_manager
        self.monitoring: bool = False
        self.events: List[CircuitBreakerEvent] = []
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.alert_callbacks: List[Callable] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._previous_states: Dict[str, str] = {}

    # ── dashboard ────────────────────────────────────────────────────

    def get_system_health_dashboard(self) -> Dict[str, Any]:
        summary = self.service_manager.get_health_summary()
        services = {}
        for name, breaker in self.service_manager.get_all_breakers().items():
            metrics = breaker.get_metrics()
            services[name] = {
                **metrics,
                "health": self.get_service_health_metrics(name),
            }
            if services[name]["health"]:
                services[name]["health"] = services[name]["health"].to_dict()
        return {
            "summary": summary,
            "services": services,
            "monitoring_active": self.monitoring,
            "total_events": len(self.events),
        }

    def get_service_health_metrics(self, service_name: str) -> Optional[ServiceHealthMetrics]:
        breaker = self.service_manager.get_breaker(service_name)
        if not breaker:
            return None
        total = breaker.total_calls or 1
        failure_rate = breaker.failure_count / total
        status = {"closed": "healthy", "half_open": "degraded", "open": "down"}.get(breaker.state, "unknown")
        uptime = 1.0 - failure_rate
        return ServiceHealthMetrics(
            service_name=service_name,
            status=status,
            failure_rate=round(failure_rate, 4),
            avg_latency_ms=0.0,
            uptime_percentage=round(uptime * 100, 2),
            last_state_change=breaker.opened_at,
        )

    # ── Prometheus ───────────────────────────────────────────────────

    def export_metrics_prometheus(self) -> str:
        lines = [
            "# HELP echo_circuit_breaker_state Circuit breaker state (0=closed, 1=half_open, 2=open)",
            "# TYPE echo_circuit_breaker_state gauge",
        ]
        state_map = {"closed": 0, "half_open": 1, "open": 2}
        for name, breaker in self.service_manager.get_all_breakers().items():
            val = state_map.get(breaker.state, -1)
            lines.append(f'echo_circuit_breaker_state{{service="{name}"}} {val}')

        lines += [
            "# HELP echo_circuit_breaker_failures_total Total failure count",
            "# TYPE echo_circuit_breaker_failures_total counter",
        ]
        for name, breaker in self.service_manager.get_all_breakers().items():
            lines.append(f'echo_circuit_breaker_failures_total{{service="{name}"}} {breaker.failure_count}')

        lines += [
            "# HELP echo_circuit_breaker_calls_total Total call count",
            "# TYPE echo_circuit_breaker_calls_total counter",
        ]
        for name, breaker in self.service_manager.get_all_breakers().items():
            lines.append(f'echo_circuit_breaker_calls_total{{service="{name}"}} {breaker.total_calls}')

        return "\n".join(lines) + "\n"

    # ── monitoring loop ──────────────────────────────────────────────

    async def start_monitoring(self, interval: float = 30.0):
        self.monitoring = True
        logger.info(f"Circuit breaker monitoring started (interval={interval}s)")
        try:
            while self.monitoring:
                self._poll_state_changes()
                await asyncio.sleep(interval)
        finally:
            self.monitoring = False

    async def stop_monitoring(self):
        self.monitoring = False

    def _poll_state_changes(self):
        for name, breaker in self.service_manager.get_all_breakers().items():
            prev = self._previous_states.get(name)
            curr = breaker.state
            if prev is not None and prev != curr:
                event = CircuitBreakerEvent(
                    service_name=name,
                    event_type=f"{prev}_to_{curr}",
                    details=breaker.get_metrics(),
                )
                self.events.append(event)
                self._fire_alert(name, prev, curr)
            self._previous_states[name] = curr

    def _fire_alert(self, service: str, old: str, new: str):
        level = "critical" if new == "open" else "warning" if new == "half_open" else "info"
        alert = {
            "level": level,
            "service": service,
            "message": f"Circuit breaker {service}: {old} → {new}",
        }
        for cb in self.alert_callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: Callable):
        self.alert_callbacks.append(callback)


# singleton
_monitor_instance: Optional[CircuitBreakerMonitor] = None


def get_circuit_breaker_monitor(service_manager: ServiceBreakerManager) -> CircuitBreakerMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = CircuitBreakerMonitor(service_manager)
    return _monitor_instance
