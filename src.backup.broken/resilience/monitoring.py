#!/usr/bin/env python3
"""
Circuit breaker monitoring and metrics collection
Provides real-time monitoring and alerting for circuit breaker states
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .service_breakers import ServiceBreakerManager

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerEvent:
    """Circuit breaker state change event"""
    service_name: str
    old_state: str
    new_state: str
    timestamp: datetime
    duration_in_previous_state: float
    failure_count: int
    success_count: int
    total_requests: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ServiceHealthMetrics:
    """Health metrics for a service"""
    service_name: str
    state: str
    uptime_percentage: float
    average_response_time: float
    total_requests: int
    success_rate: float
    failure_rate: float
    consecutive_failures: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    circuit_opens_today: int
    time_in_open_state_today: float  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None
        }


class CircuitBreakerMonitor:
    """
    Monitors circuit breaker states and provides metrics/alerting
    """

    def __init__(self, service_manager: ServiceBreakerManager, alert_threshold: int = 3):
        self.service_manager = service_manager
        self.alert_threshold = alert_threshold

        # Event storage
        self.events: deque = deque(maxlen=1000)  # Last 1000 events
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # Last 100 metrics per service

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Monitoring state
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Daily statistics (reset at midnight)
        self._daily_stats_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self._daily_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'circuit_opens': 0,
            'total_downtime': 0.0,
            'requests': 0,
            'failures': 0
        })

        # Setup state change listeners
        self._setup_state_listeners()

        logger.info("Circuit breaker monitor initialized")

    def _setup_state_listeners(self):
        """Setup listeners for circuit breaker state changes"""
        for service_name, breaker in self.service_manager.get_all_breakers().items():
            breaker.add_state_listener(self._on_state_change)

    def _on_state_change(self, service_name: str, old_state: CircuitBreakerState, new_state: CircuitBreakerState):
        """Handle circuit breaker state changes"""
        breaker = self.service_manager.get_breaker(service_name)
        if not breaker:
            return

        metrics = breaker.get_metrics()

        # Create event
        event = CircuitBreakerEvent(
            service_name=service_name,
            old_state=old_state.value,
            new_state=new_state.value,
            timestamp=datetime.utcnow(),
            duration_in_previous_state=metrics.get('state_duration', 0),
            failure_count=metrics['metrics']['failed_requests'],
            success_count=metrics['metrics']['successful_requests'],
            total_requests=metrics['metrics']['total_requests']
        )

        # Store event
        self.events.append(event)

        # Update daily statistics
        self._update_daily_stats(service_name, old_state, new_state, event.duration_in_previous_state)

        # Check for alerts
        self._check_alerts(service_name, new_state, event)

        logger.info(f"Circuit breaker state change: {service_name} {old_state.value} -> {new_state.value}")

    def _update_daily_stats(self, service_name: str, old_state: CircuitBreakerState,
                           new_state: CircuitBreakerState, duration: float):
        """Update daily statistics"""
        # Reset daily stats if it's a new day
        now = datetime.utcnow()
        if now.date() > self._daily_stats_reset_time.date():
            self._daily_stats.clear()
            self._daily_stats_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)

        stats = self._daily_stats[service_name]

        if new_state == CircuitBreakerState.OPEN:
            stats['circuit_opens'] += 1

        if old_state == CircuitBreakerState.OPEN:
            stats['total_downtime'] += duration

    def _check_alerts(self, service_name: str, new_state: CircuitBreakerState, event: CircuitBreakerEvent):
        """Check if alerts should be triggered"""
        alerts = []

        # Critical: Circuit breaker opened
        if new_state == CircuitBreakerState.OPEN:
            alerts.append({
                'level': 'critical',
                'service': service_name,
                'message': f"Service {service_name} is DOWN - circuit breaker opened",
                'event': event.to_dict(),
                'recommended_actions': [
                    f"Check {service_name} service health",
                    f"Review {service_name} logs for errors",
                    f"Verify {service_name} dependencies",
                    "Consider manual service restart if needed"
                ]
            })

        # Warning: Circuit breaker in half-open (testing recovery)
        elif new_state == CircuitBreakerState.HALF_OPEN:
            alerts.append({
                'level': 'warning',
                'service': service_name,
                'message': f"Service {service_name} is testing recovery - circuit breaker half-open",
                'event': event.to_dict(),
                'recommended_actions': [
                    f"Monitor {service_name} recovery attempts",
                    "Avoid high load during recovery testing"
                ]
            })

        # Info: Circuit breaker closed (recovered)
        elif new_state == CircuitBreakerState.CLOSED:
            alerts.append({
                'level': 'info',
                'service': service_name,
                'message': f"Service {service_name} has RECOVERED - circuit breaker closed",
                'event': event.to_dict(),
                'recommended_actions': []
            })

        # Check for repeated failures
        recent_opens = len([e for e in self.events
                           if e.service_name == service_name
                           and e.new_state == 'open'
                           and e.timestamp > datetime.utcnow() - timedelta(hours=1)])

        if recent_opens >= self.alert_threshold:
            alerts.append({
                'level': 'critical',
                'service': service_name,
                'message': f"Service {service_name} has failed {recent_opens} times in the last hour",
                'event': event.to_dict(),
                'recommended_actions': [
                    "Investigate underlying service issues",
                    "Check service configuration",
                    "Review service dependencies",
                    "Consider service replacement or upgrade"
                ]
            })

        # Send alerts
        for alert in alerts:
            self._send_alert(alert)

    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
        logger.info(f"Added alert callback: {callback.__name__}")

    async def start_monitoring(self, interval: float = 30.0):
        """Start continuous monitoring"""
        if self.monitoring:
            logger.warning("Monitoring already running")
            return

        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Started circuit breaker monitoring (interval: {interval}s)")

    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped circuit breaker monitoring")

    async def _monitor_loop(self, interval: float):
        """Continuous monitoring loop"""
        try:
            while self.monitoring:
                await self._collect_metrics()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Monitor loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}")

    async def _collect_metrics(self):
        """Collect current metrics from all circuit breakers"""
        timestamp = datetime.utcnow()

        for service_name, breaker in self.service_manager.get_all_breakers().items():
            metrics = breaker.get_metrics()

            # Store metrics in history
            metrics_entry = {
                'timestamp': timestamp,
                'service': service_name,
                **metrics['metrics'],
                'state': metrics['state']
            }

            self.metrics_history[service_name].append(metrics_entry)

    def get_service_health_metrics(self, service_name: str) -> Optional[ServiceHealthMetrics]:
        """Get health metrics for specific service"""
        breaker = self.service_manager.get_breaker(service_name)
        if not breaker:
            return None

        metrics = breaker.get_metrics()
        service_events = [e for e in self.events if e.service_name == service_name]

        # Calculate uptime percentage for last 24 hours
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)

        downtime_events = [e for e in service_events
                          if e.new_state == 'open' and e.timestamp > last_24h]

        total_downtime = sum(e.duration_in_previous_state for e in downtime_events
                           if e.old_state == 'open')

        uptime_percentage = ((24 * 3600 - total_downtime) / (24 * 3600)) * 100

        # Circuit opens today
        today_opens = len([e for e in service_events
                          if e.new_state == 'open'
                          and e.timestamp.date() == now.date()])

        return ServiceHealthMetrics(
            service_name=service_name,
            state=metrics['state'],
            uptime_percentage=round(uptime_percentage, 2),
            average_response_time=metrics['metrics']['average_response_time'],
            total_requests=metrics['metrics']['total_requests'],
            success_rate=metrics['metrics']['success_rate'],
            failure_rate=metrics['metrics']['failure_rate'],
            consecutive_failures=metrics['metrics']['consecutive_failures'],
            last_failure_time=datetime.fromisoformat(metrics['metrics']['last_failure'])
            if metrics['metrics']['last_failure'] else None,
            last_success_time=datetime.fromisoformat(metrics['metrics']['last_success'])
            if metrics['metrics']['last_success'] else None,
            circuit_opens_today=today_opens,
            time_in_open_state_today=total_downtime
        )

    def get_system_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system health dashboard"""
        services_health = {}
        overall_metrics = {
            'total_services': 0,
            'healthy_services': 0,
            'degraded_services': 0,
            'failed_services': 0,
            'average_uptime': 0.0
        }

        uptimes = []

        for service_name in self.service_manager.get_all_breakers().keys():
            health = self.get_service_health_metrics(service_name)
            if health:
                services_health[service_name] = health.to_dict()
                overall_metrics['total_services'] += 1
                uptimes.append(health.uptime_percentage)

                if health.state == 'closed' and health.uptime_percentage >= 99.0:
                    overall_metrics['healthy_services'] += 1
                elif health.state == 'half_open' or health.uptime_percentage >= 95.0:
                    overall_metrics['degraded_services'] += 1
                else:
                    overall_metrics['failed_services'] += 1

        overall_metrics['average_uptime'] = sum(uptimes) / len(uptimes) if uptimes else 0.0

        # Recent events (last hour)
        recent_events = [e.to_dict() for e in self.events
                        if e.timestamp > datetime.utcnow() - timedelta(hours=1)]

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_health': {
                'status': self._get_overall_health_status(overall_metrics),
                **overall_metrics
            },
            'services': services_health,
            'recent_events': recent_events[-10:],  # Last 10 events
            'daily_statistics': dict(self._daily_stats)
        }

    def _get_overall_health_status(self, metrics: Dict[str, Any]) -> str:
        """Determine overall health status"""
        if metrics['failed_services'] > 0:
            if metrics['failed_services'] >= metrics['total_services'] // 2:
                return 'critical'
            else:
                return 'degraded'
        elif metrics['degraded_services'] > 0:
            return 'degraded'
        else:
            return 'healthy'

    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = [
            '# HELP circuit_breaker_state Current state of circuit breaker (0=closed, 1=half_open, 2=open)',
            '# TYPE circuit_breaker_state gauge',
            '# HELP circuit_breaker_requests_total Total number of requests',
            '# TYPE circuit_breaker_requests_total counter',
            '# HELP circuit_breaker_failures_total Total number of failures',
            '# TYPE circuit_breaker_failures_total counter',
            '# HELP circuit_breaker_response_time_avg Average response time in seconds',
            '# TYPE circuit_breaker_response_time_avg gauge'
        ]

        state_map = {'closed': 0, 'half_open': 1, 'open': 2}

        for service_name, breaker in self.service_manager.get_all_breakers().items():
            metrics = breaker.get_metrics()
            state_value = state_map.get(metrics['state'], 2)

            lines.extend([
                f'circuit_breaker_state{{service="{service_name}"}} {state_value}',
                f'circuit_breaker_requests_total{{service="{service_name}"}} {metrics["metrics"]["total_requests"]}',
                f'circuit_breaker_failures_total{{service="{service_name}"}} {metrics["metrics"]["failed_requests"]}',
                f'circuit_breaker_response_time_avg{{service="{service_name}"}} {metrics["metrics"]["average_response_time"]}'
            ])

        return '\n'.join(lines)

    def clear_history(self, service_name: Optional[str] = None):
        """Clear monitoring history"""
        if service_name:
            if service_name in self.metrics_history:
                self.metrics_history[service_name].clear()
            self.events = deque([e for e in self.events if e.service_name != service_name], maxlen=1000)
            logger.info(f"Cleared monitoring history for {service_name}")
        else:
            self.events.clear()
            self.metrics_history.clear()
            self._daily_stats.clear()
            logger.info("Cleared all monitoring history")


# Global monitor instance
_circuit_breaker_monitor = None


def get_circuit_breaker_monitor(service_manager: Optional[ServiceBreakerManager] = None) -> CircuitBreakerMonitor:
    """Get global circuit breaker monitor instance"""
    global _circuit_breaker_monitor
    if _circuit_breaker_monitor is None:
        if service_manager is None:
            from .service_breakers import get_service_breaker_manager
            service_manager = get_service_breaker_manager()
        _circuit_breaker_monitor = CircuitBreakerMonitor(service_manager)
    return _circuit_breaker_monitor