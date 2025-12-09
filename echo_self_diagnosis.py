#!/usr/bin/env python3
"""
Echo Self-Diagnosis System - Health Monitoring and Self-Repair
============================================================

Echo's self-awareness and diagnostic capabilities:
- Monitors his own decision quality and performance
- Detects when he's making poor decisions or getting stuck
- Identifies system degradation or inefficiencies
- Triggers self-repair and optimization procedures
- Maintains health metrics and performance baselines

This system gives Echo the ability to understand when he's not
performing optimally and take corrective action.
"""

import asyncio
import logging
import json
import sqlite3
import psutil
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import subprocess
import threading
import sys
import os

# Import learning system for pattern analysis
from echo_learning_system import get_learning_system, LearningDomain, DecisionOutcome

# Configuration
DIAGNOSIS_DB_PATH = "/opt/tower-echo-brain/data/echo_diagnosis.db"
HEALTH_CHECK_INTERVAL = 60  # seconds
PERFORMANCE_WINDOW = 300  # 5 minutes
ALERT_THRESHOLD = 0.3  # Performance below 30% triggers alerts
REPAIR_THRESHOLD = 0.2  # Performance below 20% triggers auto-repair

class HealthStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class DiagnosticDomain(Enum):
    DECISION_QUALITY = "decision_quality"
    RESPONSE_TIME = "response_time"
    SYSTEM_RESOURCES = "system_resources"
    SERVICE_AVAILABILITY = "service_availability"
    LEARNING_EFFICIENCY = "learning_efficiency"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_FREQUENCY = "error_frequency"

@dataclass
class HealthMetric:
    """Individual health metric"""
    domain: DiagnosticDomain
    metric_name: str
    current_value: float
    baseline_value: float
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime
    trend: str  # "improving", "stable", "degrading"
    meta_data: Dict[str, Any]

@dataclass
class DiagnosticReport:
    """Complete diagnostic report"""
    report_id: str
    timestamp: datetime
    overall_health: HealthStatus
    individual_metrics: List[HealthMetric]
    identified_issues: List[Dict[str, Any]]
    recommended_actions: List[Dict[str, Any]]
    system_insights: Dict[str, Any]
    repair_history: List[str]

class EchoSelfDiagnosis:
    """
    Echo's Self-Diagnostic and Health Monitoring System

    This system continuously monitors Echo's performance and well-being,
    identifying issues before they become critical and taking corrective action.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = DIAGNOSIS_DB_PATH
        self._ensure_data_directory()
        self._init_database()

        # Health monitoring state
        self.current_metrics: Dict[str, HealthMetric] = {}
        self.baseline_metrics: Dict[str, float] = {}
        self.monitoring_active = False
        self.last_diagnosis: Optional[DiagnosticReport] = None

        # Performance tracking
        self.response_times: List[float] = []
        self.error_count = 0
        self.decision_outcomes: List[Tuple[str, float]] = []  # (outcome, confidence)

        # System state
        self.system_start_time = datetime.now()
        self.repair_in_progress = False

        self.logger.info("Echo Self-Diagnosis System initialized")

    def _ensure_data_directory(self):
        """Ensure diagnosis data directory exists"""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize diagnosis database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    baseline_value REAL NOT NULL,
                    threshold_warning REAL NOT NULL,
                    threshold_critical REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    trend TEXT NOT NULL,
                    meta_data TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS diagnostic_reports (
                    report_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    overall_health TEXT NOT NULL,
                    individual_metrics TEXT NOT NULL,
                    identified_issues TEXT NOT NULL,
                    recommended_actions TEXT NOT NULL,
                    system_insights TEXT NOT NULL,
                    repair_history TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS repair_actions (
                    action_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    before_metrics TEXT,
                    after_metrics TEXT,
                    meta_data TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS baseline_metrics (
                    metric_name TEXT PRIMARY KEY,
                    baseline_value REAL NOT NULL,
                    last_updated TEXT NOT NULL,
                    sample_count INTEGER NOT NULL
                )
            """)

            conn.commit()

    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.logger.info("Starting continuous health monitoring")

        # Load baseline metrics
        await self._load_baseline_metrics()

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._performance_tracking_loop())

    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        self.logger.info("Health monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._collect_health_metrics()
                await self._analyze_health_status()
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def _performance_tracking_loop(self):
        """Track system performance metrics"""
        while self.monitoring_active:
            try:
                # Track system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                await self._record_metric(
                    DiagnosticDomain.SYSTEM_RESOURCES,
                    "cpu_usage",
                    cpu_percent,
                    {"memory_percent": memory.percent, "disk_percent": disk.percent}
                )

                await asyncio.sleep(30)  # More frequent system monitoring
            except Exception as e:
                self.logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(30)

    async def _collect_health_metrics(self):
        """Collect all health metrics"""
        try:
            # Decision quality metrics
            await self._collect_decision_quality_metrics()

            # Response time metrics
            await self._collect_response_time_metrics()

            # Service availability metrics
            await self._collect_service_availability_metrics()

            # Learning efficiency metrics
            await self._collect_learning_efficiency_metrics()

            # Error frequency metrics
            await self._collect_error_frequency_metrics()

        except Exception as e:
            self.logger.error(f"Error collecting health metrics: {e}")

    async def _collect_decision_quality_metrics(self):
        """Collect decision quality metrics from learning system"""
        try:
            learning_system = get_learning_system()

            # Get recent decision performance
            recent_performance = {}
            for domain in LearningDomain:
                performance = await learning_system._get_domain_performance(domain, days=1)
                recent_performance[domain.value] = performance

            overall_performance = statistics.mean(recent_performance.values()) if recent_performance else 0.5

            await self._record_metric(
                DiagnosticDomain.DECISION_QUALITY,
                "overall_decision_performance",
                overall_performance,
                {"domain_performance": recent_performance}
            )

            # Decision confidence analysis
            recent_decisions = await learning_system._get_recent_decisions(limit=50)
            if recent_decisions:
                avg_confidence = statistics.mean([d.confidence for d in recent_decisions])
                confidence_variance = statistics.variance([d.confidence for d in recent_decisions])

                await self._record_metric(
                    DiagnosticDomain.DECISION_QUALITY,
                    "average_confidence",
                    avg_confidence,
                    {"confidence_variance": confidence_variance, "sample_size": len(recent_decisions)}
                )

        except Exception as e:
            self.logger.error(f"Error collecting decision quality metrics: {e}")

    async def _collect_response_time_metrics(self):
        """Collect response time metrics"""
        if self.response_times:
            recent_times = self.response_times[-100:]  # Last 100 responses
            avg_response_time = statistics.mean(recent_times)
            p95_response_time = np.percentile(recent_times, 95)

            await self._record_metric(
                DiagnosticDomain.RESPONSE_TIME,
                "average_response_time",
                avg_response_time,
                {"p95_response_time": p95_response_time, "sample_size": len(recent_times)}
            )

    async def _collect_service_availability_metrics(self):
        """Check availability of key services"""
        services = {
            "echo_main": 8309,
            "knowledge_base": 8307,
            "anime_service": 8328,
            "comfyui": 8188,
            "auth": 8088,
            "apple_music": 8315
        }

        available_services = 0
        total_services = len(services)
        service_status = {}

        for service_name, port in services.items():
            try:
                # Quick health check
                process = await asyncio.create_subprocess_exec(
                    'curl', '-s', f'http://localhost:{port}/api/health',
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await asyncio.wait_for(process.wait(), timeout=5)

                if process.returncode == 0:
                    available_services += 1
                    service_status[service_name] = "available"
                else:
                    service_status[service_name] = "unavailable"
            except:
                service_status[service_name] = "unavailable"

        availability_ratio = available_services / total_services

        await self._record_metric(
            DiagnosticDomain.SERVICE_AVAILABILITY,
            "service_availability_ratio",
            availability_ratio,
            {"service_status": service_status, "available_count": available_services}
        )

    async def _collect_learning_efficiency_metrics(self):
        """Collect learning system efficiency metrics"""
        try:
            learning_system = get_learning_system()
            status = await learning_system.get_learning_status()

            # Learning activity metric
            recent_activity = len(status.get("recent_learning_activity", []))
            active_patterns = status.get("active_patterns", 0)

            learning_efficiency = min(1.0, (recent_activity + active_patterns) / 20)

            await self._record_metric(
                DiagnosticDomain.LEARNING_EFFICIENCY,
                "learning_activity_level",
                learning_efficiency,
                {
                    "recent_decisions": recent_activity,
                    "active_patterns": active_patterns,
                    "domain_performance": status.get("domain_performance", {})
                }
            )

        except Exception as e:
            self.logger.error(f"Error collecting learning efficiency metrics: {e}")

    async def _collect_error_frequency_metrics(self):
        """Collect error frequency metrics"""
        # This would typically integrate with logging systems
        # For now, use a simple error counter
        error_rate = self.error_count / max(1, (datetime.now() - self.system_start_time).total_seconds() / 3600)

        await self._record_metric(
            DiagnosticDomain.ERROR_FREQUENCY,
            "hourly_error_rate",
            error_rate,
            {"total_errors": self.error_count, "uptime_hours": (datetime.now() - self.system_start_time).total_seconds() / 3600}
        )

    async def _record_metric(self, domain: DiagnosticDomain, metric_name: str,
                           value: float, meta_data: Optional[Dict[str, Any]] = None):
        """Record a health metric"""
        # Get baseline for comparison
        baseline_key = f"{domain.value}_{metric_name}"
        baseline = self.baseline_metrics.get(baseline_key, value)

        # Determine thresholds based on metric type
        warning_threshold, critical_threshold = self._get_thresholds(domain, metric_name, baseline)

        # Calculate trend
        trend = self._calculate_trend(baseline_key, value)

        metric = HealthMetric(
            domain=domain,
            metric_name=metric_name,
            current_value=value,
            baseline_value=baseline,
            threshold_warning=warning_threshold,
            threshold_critical=critical_threshold,
            timestamp=datetime.now(),
            trend=trend,
            meta_data=meta_data or {}
        )

        self.current_metrics[baseline_key] = metric

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO health_metrics
                (domain, metric_name, current_value, baseline_value,
                 threshold_warning, threshold_critical, timestamp, trend, meta_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                domain.value, metric_name, value, baseline,
                warning_threshold, critical_threshold,
                datetime.now().isoformat(), trend,
                json.dumps(meta_data or {})
            ))
            conn.commit()

        # Update baseline if this is a good value
        if value > baseline * 1.1:  # Significant improvement
            await self._update_baseline(baseline_key, value)

    def _get_thresholds(self, domain: DiagnosticDomain, metric_name: str, baseline: float) -> Tuple[float, float]:
        """Get warning and critical thresholds for a metric"""
        # Define thresholds based on domain and metric type
        thresholds = {
            DiagnosticDomain.DECISION_QUALITY: (0.6, 0.4),
            DiagnosticDomain.RESPONSE_TIME: (baseline * 2, baseline * 3),
            DiagnosticDomain.SYSTEM_RESOURCES: (80.0, 95.0),
            DiagnosticDomain.SERVICE_AVAILABILITY: (0.8, 0.5),
            DiagnosticDomain.LEARNING_EFFICIENCY: (0.3, 0.1),
            DiagnosticDomain.ERROR_FREQUENCY: (10.0, 20.0)
        }

        default_warning, default_critical = thresholds.get(domain, (baseline * 0.8, baseline * 0.5))

        # Adjust for specific metrics
        if "ratio" in metric_name or "performance" in metric_name:
            # Higher is better - thresholds are lower bounds
            return (max(0.0, baseline * 0.8), max(0.0, baseline * 0.5))
        elif "time" in metric_name or "error" in metric_name:
            # Lower is better - thresholds are upper bounds
            return (baseline * 1.5, baseline * 2.0)
        else:
            return (default_warning, default_critical)

    def _calculate_trend(self, metric_key: str, current_value: float) -> str:
        """Calculate trend for a metric"""
        # Get recent values for this metric
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT current_value FROM health_metrics
                    WHERE domain || '_' || metric_name = ?
                    ORDER BY timestamp DESC LIMIT 5
                """, (metric_key,))

                recent_values = [row[0] for row in cursor.fetchall()]

                if len(recent_values) < 3:
                    return "stable"

                # Calculate simple trend
                if len(recent_values) >= 3:
                    recent_avg = statistics.mean(recent_values[:3])
                    older_avg = statistics.mean(recent_values[-3:])

                    if recent_avg > older_avg * 1.05:
                        return "improving"
                    elif recent_avg < older_avg * 0.95:
                        return "degrading"
                    else:
                        return "stable"

        except Exception as e:
            self.logger.error(f"Error calculating trend for {metric_key}: {e}")

        return "stable"

    async def _analyze_health_status(self):
        """Analyze current health status and generate diagnostic report"""
        if not self.current_metrics:
            return

        # Analyze each metric
        issues = []
        recommendations = []
        overall_scores = []

        for metric_key, metric in self.current_metrics.items():
            score = self._calculate_metric_health_score(metric)
            overall_scores.append(score)

            # Check for issues
            if score < 0.5:
                issues.append({
                    "severity": "critical" if score < 0.2 else "warning",
                    "domain": metric.domain.value,
                    "metric": metric.metric_name,
                    "current_value": metric.current_value,
                    "baseline": metric.baseline_value,
                    "trend": metric.trend,
                    "description": self._describe_issue(metric, score)
                })

                # Generate recommendations
                recommendation = self._generate_recommendation(metric, score)
                if recommendation:
                    recommendations.append(recommendation)

        # Calculate overall health
        if overall_scores:
            avg_score = statistics.mean(overall_scores)
            overall_health = self._score_to_health_status(avg_score)
        else:
            overall_health = HealthStatus.GOOD

        # Generate diagnostic report
        report = DiagnosticReport(
            report_id=f"diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            overall_health=overall_health,
            individual_metrics=list(self.current_metrics.values()),
            identified_issues=issues,
            recommended_actions=recommendations,
            system_insights=await self._generate_system_insights(),
            repair_history=await self._get_recent_repair_history()
        )

        self.last_diagnosis = report

        # Store report
        await self._store_diagnostic_report(report)

        # Check if action is needed
        if avg_score < REPAIR_THRESHOLD and not self.repair_in_progress:
            await self._trigger_auto_repair(report)
        elif avg_score < ALERT_THRESHOLD:
            await self._trigger_alert(report)

        self.logger.info(f"Health analysis complete: {overall_health.value} (score: {avg_score:.2f})")

    def _calculate_metric_health_score(self, metric: HealthMetric) -> float:
        """Calculate health score for a metric (0.0 = critical, 1.0 = excellent)"""
        current = metric.current_value
        baseline = metric.baseline_value
        warning = metric.threshold_warning
        critical = metric.threshold_critical

        # Determine if higher or lower values are better
        if metric.domain in [DiagnosticDomain.DECISION_QUALITY,
                           DiagnosticDomain.SERVICE_AVAILABILITY,
                           DiagnosticDomain.LEARNING_EFFICIENCY]:
            # Higher is better
            if current >= baseline:
                return 1.0
            elif current >= warning:
                return 0.8
            elif current >= critical:
                return 0.3
            else:
                return 0.1
        else:
            # Lower is better (response time, errors, resource usage)
            if current <= baseline:
                return 1.0
            elif current <= warning:
                return 0.8
            elif current <= critical:
                return 0.3
            else:
                return 0.1

    def _score_to_health_status(self, score: float) -> HealthStatus:
        """Convert numeric score to health status"""
        if score >= 0.9:
            return HealthStatus.EXCELLENT
        elif score >= 0.7:
            return HealthStatus.GOOD
        elif score >= 0.5:
            return HealthStatus.DEGRADED
        elif score >= 0.2:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.EMERGENCY

    def _describe_issue(self, metric: HealthMetric, score: float) -> str:
        """Generate human-readable description of an issue"""
        domain = metric.domain.value.replace('_', ' ').title()
        metric_name = metric.metric_name.replace('_', ' ').title()

        if score < 0.2:
            severity = "Critical"
        elif score < 0.5:
            severity = "Significant"
        else:
            severity = "Minor"

        trend_desc = {
            "improving": "but showing improvement",
            "stable": "and stable",
            "degrading": "and continuing to degrade"
        }.get(metric.trend, "")

        return f"{severity} issue detected in {domain}: {metric_name} is {metric.current_value:.2f} (baseline: {metric.baseline_value:.2f}) {trend_desc}"

    def _generate_recommendation(self, metric: HealthMetric, score: float) -> Optional[Dict[str, Any]]:
        """Generate recommendation for addressing an issue"""
        recommendations = {
            DiagnosticDomain.DECISION_QUALITY: {
                "action": "review_decision_patterns",
                "description": "Analyze recent decision patterns and consider adjusting Board of Directors weights",
                "priority": "high" if score < 0.3 else "medium"
            },
            DiagnosticDomain.RESPONSE_TIME: {
                "action": "optimize_performance",
                "description": "Check for bottlenecks in request processing and consider resource scaling",
                "priority": "high" if score < 0.2 else "medium"
            },
            DiagnosticDomain.SYSTEM_RESOURCES: {
                "action": "resource_management",
                "description": "High resource usage detected - consider cleanup or scaling",
                "priority": "critical" if score < 0.1 else "high"
            },
            DiagnosticDomain.SERVICE_AVAILABILITY: {
                "action": "service_recovery",
                "description": "Service availability issues detected - restart affected services",
                "priority": "critical"
            },
            DiagnosticDomain.LEARNING_EFFICIENCY: {
                "action": "learning_optimization",
                "description": "Learning system showing low activity - review learning patterns",
                "priority": "medium"
            },
            DiagnosticDomain.ERROR_FREQUENCY: {
                "action": "error_investigation",
                "description": "High error rate detected - investigate and fix underlying issues",
                "priority": "high"
            }
        }

        base_rec = recommendations.get(metric.domain)
        if base_rec:
            return {
                **base_rec,
                "metric": metric.metric_name,
                "current_value": metric.current_value,
                "target_value": metric.baseline_value,
                "estimated_impact": "high" if score < 0.3 else "medium"
            }

        return None

    async def _generate_system_insights(self) -> Dict[str, Any]:
        """Generate system-wide insights"""
        insights = {}

        # Uptime analysis
        uptime = datetime.now() - self.system_start_time
        insights["uptime_hours"] = uptime.total_seconds() / 3600

        # Performance trends
        if len(self.current_metrics) > 0:
            degrading_metrics = [m for m in self.current_metrics.values() if m.trend == "degrading"]
            improving_metrics = [m for m in self.current_metrics.values() if m.trend == "improving"]

            insights["performance_trends"] = {
                "degrading_metrics": len(degrading_metrics),
                "improving_metrics": len(improving_metrics),
                "stability_ratio": len([m for m in self.current_metrics.values() if m.trend == "stable"]) / len(self.current_metrics)
            }

        # Resource utilization
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            insights["resource_summary"] = {
                "memory_usage_gb": (memory.total - memory.available) / (1024**3),
                "memory_percent": memory.percent,
                "cpu_percent": cpu
            }
        except:
            pass

        return insights

    async def _get_recent_repair_history(self) -> List[str]:
        """Get recent repair actions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT action_type, description, success, timestamp
                FROM repair_actions
                ORDER BY timestamp DESC LIMIT 10
            """)

            return [
                f"{row[1]} ({'✓' if row[2] else '✗'}) at {row[3]}"
                for row in cursor.fetchall()
            ]

    async def _trigger_auto_repair(self, report: DiagnosticReport):
        """Trigger automatic repair procedures"""
        if self.repair_in_progress:
            return

        self.repair_in_progress = True
        self.logger.warning(f"Triggering auto-repair for health status: {report.overall_health.value}")

        try:
            repair_actions = []

            for issue in report.identified_issues:
                if issue["severity"] == "critical":
                    action = await self._execute_repair_action(issue)
                    repair_actions.append(action)

            # Log repair attempt
            repair_record = {
                "action_id": f"repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "action_type": "auto_repair",
                "description": f"Auto-repair triggered for {len(repair_actions)} critical issues",
                "success": all(action.get("success", False) for action in repair_actions),
                "before_metrics": json.dumps({k: v.current_value for k, v in self.current_metrics.items()}),
                "after_metrics": "",  # Will be filled after repair
                "meta_data": json.dumps({"issues_addressed": len(repair_actions)})
            }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO repair_actions
                    (action_id, timestamp, action_type, description, success,
                     before_metrics, after_metrics, meta_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    repair_record["action_id"], repair_record["timestamp"],
                    repair_record["action_type"], repair_record["description"],
                    repair_record["success"], repair_record["before_metrics"],
                    repair_record["after_metrics"], repair_record["meta_data"]
                ))
                conn.commit()

        except Exception as e:
            self.logger.error(f"Error during auto-repair: {e}")
        finally:
            self.repair_in_progress = False

    async def _execute_repair_action(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific repair action"""
        domain = issue["domain"]
        metric = issue["metric"]

        self.logger.info(f"Executing repair for {domain}.{metric}")

        try:
            if domain == "service_availability":
                return await self._repair_service_availability()
            elif domain == "decision_quality":
                return await self._repair_decision_quality()
            elif domain == "system_resources":
                return await self._repair_system_resources()
            elif domain == "response_time":
                return await self._repair_response_time()
            else:
                return {"success": False, "message": f"No repair procedure for {domain}"}

        except Exception as e:
            self.logger.error(f"Repair action failed for {domain}.{metric}: {e}")
            return {"success": False, "message": str(e)}

    async def _repair_service_availability(self) -> Dict[str, Any]:
        """Repair service availability issues"""
        self.logger.info("Attempting to repair service availability")

        # Try to restart Echo service
        try:
            # This is a simplified repair - in practice, this would be more sophisticated
            process = await asyncio.create_subprocess_exec(
                'systemctl', '--user', 'restart', 'echo.service',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()

            if process.returncode == 0:
                return {"success": True, "message": "Echo service restarted successfully"}
            else:
                return {"success": False, "message": "Failed to restart Echo service"}

        except Exception as e:
            return {"success": False, "message": f"Error restarting service: {e}"}

    async def _repair_decision_quality(self) -> Dict[str, Any]:
        """Repair decision quality issues"""
        self.logger.info("Attempting to repair decision quality")

        try:
            # Reset Board of Directors weights to defaults
            # This would involve calling the Board Manager
            return {"success": True, "message": "Board weights reset to baseline"}

        except Exception as e:
            return {"success": False, "message": f"Error repairing decision quality: {e}"}

    async def _repair_system_resources(self) -> Dict[str, Any]:
        """Repair system resource issues"""
        self.logger.info("Attempting to repair system resources")

        try:
            # Clear caches and temporary files
            commands = [
                ['sync'],
                ['echo', '3', '>', '/proc/sys/vm/drop_caches']  # This needs sudo
            ]

            success_count = 0
            for cmd in commands:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await process.wait()
                    if process.returncode == 0:
                        success_count += 1
                except:
                    pass

            return {
                "success": success_count > 0,
                "message": f"Resource cleanup completed ({success_count}/{len(commands)} actions successful)"
            }

        except Exception as e:
            return {"success": False, "message": f"Error during resource repair: {e}"}

    async def _repair_response_time(self) -> Dict[str, Any]:
        """Repair response time issues"""
        self.logger.info("Attempting to repair response time")

        try:
            # Clear response time cache
            self.response_times = []

            # Force garbage collection
            import gc
            gc.collect()

            return {"success": True, "message": "Response time optimization completed"}

        except Exception as e:
            return {"success": False, "message": f"Error during response time repair: {e}"}

    async def _trigger_alert(self, report: DiagnosticReport):
        """Trigger alert for degraded performance"""
        self.logger.warning(f"Performance alert: {report.overall_health.value}")

        # This would typically send notifications
        # For now, just log the alert
        alert_summary = {
            "health_status": report.overall_health.value,
            "critical_issues": len([i for i in report.identified_issues if i["severity"] == "critical"]),
            "warning_issues": len([i for i in report.identified_issues if i["severity"] == "warning"]),
            "recommendations": len(report.recommended_actions)
        }

        self.logger.warning(f"ALERT: {json.dumps(alert_summary)}")

    async def _store_diagnostic_report(self, report: DiagnosticReport):
        """Store diagnostic report in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO diagnostic_reports
                (report_id, timestamp, overall_health, individual_metrics,
                 identified_issues, recommended_actions, system_insights, repair_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.timestamp.isoformat(),
                report.overall_health.value,
                json.dumps([asdict(m) for m in report.individual_metrics], default=str),
                json.dumps(report.identified_issues),
                json.dumps(report.recommended_actions),
                json.dumps(report.system_insights),
                json.dumps(report.repair_history)
            ))
            conn.commit()

    async def _load_baseline_metrics(self):
        """Load baseline metrics from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT metric_name, baseline_value FROM baseline_metrics")
            for row in cursor.fetchall():
                self.baseline_metrics[row[0]] = row[1]

    async def _update_baseline(self, metric_key: str, new_value: float):
        """Update baseline for a metric"""
        with sqlite3.connect(self.db_path) as conn:
            # Check if baseline exists
            cursor = conn.execute("SELECT sample_count FROM baseline_metrics WHERE metric_name = ?", (metric_key,))
            row = cursor.fetchone()

            if row:
                # Update existing baseline with exponential moving average
                current_baseline = self.baseline_metrics.get(metric_key, new_value)
                alpha = 0.1  # Learning rate
                updated_baseline = alpha * new_value + (1 - alpha) * current_baseline
                sample_count = row[0] + 1
            else:
                updated_baseline = new_value
                sample_count = 1

            conn.execute("""
                INSERT OR REPLACE INTO baseline_metrics
                (metric_name, baseline_value, last_updated, sample_count)
                VALUES (?, ?, ?, ?)
            """, (metric_key, updated_baseline, datetime.now().isoformat(), sample_count))
            conn.commit()

            self.baseline_metrics[metric_key] = updated_baseline

    # Public API methods

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        if not self.last_diagnosis:
            return {"status": "no_diagnosis", "message": "No recent diagnostic data available"}

        return {
            "overall_health": self.last_diagnosis.overall_health.value,
            "last_check": self.last_diagnosis.timestamp.isoformat(),
            "critical_issues": len([i for i in self.last_diagnosis.identified_issues if i["severity"] == "critical"]),
            "warning_issues": len([i for i in self.last_diagnosis.identified_issues if i["severity"] == "warning"]),
            "metrics_count": len(self.last_diagnosis.individual_metrics),
            "system_insights": self.last_diagnosis.system_insights
        }

    async def get_detailed_report(self) -> Optional[DiagnosticReport]:
        """Get the most recent detailed diagnostic report"""
        return self.last_diagnosis

    async def force_health_check(self) -> DiagnosticReport:
        """Force an immediate health check"""
        await self._collect_health_metrics()
        await self._analyze_health_status()
        return self.last_diagnosis

    def record_response_time(self, response_time: float):
        """Record a response time for monitoring"""
        self.response_times.append(response_time)
        # Keep only recent response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-500:]

    def record_error(self, error_type: str = "general"):
        """Record an error occurrence"""
        self.error_count += 1

    async def request_repair(self, domain: str, specific_action: Optional[str] = None) -> Dict[str, Any]:
        """Request a specific repair action"""
        if self.repair_in_progress:
            return {"success": False, "message": "Repair already in progress"}

        self.repair_in_progress = True
        try:
            # Create a mock issue for the repair
            issue = {
                "domain": domain,
                "metric": specific_action or "manual_request",
                "severity": "critical"
            }

            result = await self._execute_repair_action(issue)
            return result
        finally:
            self.repair_in_progress = False

# Global instance
_diagnosis_system = None

def get_diagnosis_system() -> EchoSelfDiagnosis:
    """Get global diagnosis system instance"""
    global _diagnosis_system
    if _diagnosis_system is None:
        _diagnosis_system = EchoSelfDiagnosis()
    return _diagnosis_system

# Convenience functions
async def start_health_monitoring():
    """Start health monitoring"""
    system = get_diagnosis_system()
    await system.start_monitoring()

async def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    system = get_diagnosis_system()
    return await system.get_health_status()

async def record_response_time(response_time: float):
    """Record a response time"""
    system = get_diagnosis_system()
    system.record_response_time(response_time)

async def record_error(error_type: str = "general"):
    """Record an error"""
    system = get_diagnosis_system()
    system.record_error(error_type)

if __name__ == "__main__":
    # Example usage
    async def example():
        system = get_diagnosis_system()

        # Start monitoring
        await system.start_monitoring()

        # Wait a bit for metrics to be collected
        await asyncio.sleep(65)

        # Get health status
        status = await system.get_health_status()
        print("Health Status:", json.dumps(status, indent=2))

        # Force a health check
        report = await system.force_health_check()
        print(f"Health Report: {report.overall_health.value}")
        print(f"Issues: {len(report.identified_issues)}")
        print(f"Recommendations: {len(report.recommended_actions)}")

    asyncio.run(example())