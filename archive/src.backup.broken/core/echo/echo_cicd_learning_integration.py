#!/usr/bin/env python3
"""
Echo CI/CD Learning Integration
Integrates CI/CD pipeline results with Echo's learning system
Provides feedback loops for continuous improvement
"""

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import asyncio
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np


@dataclass
class CICDMetrics:
    """CI/CD pipeline metrics for learning"""

    pipeline_id: str
    app_name: str
    success: bool
    duration_seconds: float
    test_coverage: float
    security_score: int
    stages_completed: int
    total_stages: int
    failure_stage: Optional[str]
    git_commit: str
    trigger_type: str
    target_environments: List[str]
    timestamp: datetime


@dataclass
class LearningPattern:
    """Identified pattern from CI/CD data"""

    pattern_type: str
    confidence: float
    description: str
    recommendations: List[str]
    affected_metrics: List[str]
    evidence: Dict[str, Any]


class CICDAnalyzer:
    """Analyzes CI/CD patterns for learning"""

    def __init__(self):
        self.min_samples = 10
        self.confidence_threshold = 0.7

    def analyze_success_patterns(
        self, metrics: List[CICDMetrics]
    ) -> List[LearningPattern]:
        """Analyze patterns in successful vs failed pipelines"""
        patterns = []

        if len(metrics) < self.min_samples:
            return patterns

        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]

        if not successful or not failed:
            return patterns

        # Analyze test coverage impact
        success_coverage = np.mean([m.test_coverage for m in successful])
        failed_coverage = np.mean([m.test_coverage for m in failed])

        if success_coverage > failed_coverage + 10:  # 10% difference threshold
            patterns.append(
                LearningPattern(
                    pattern_type="coverage_correlation",
                    confidence=min(
                        0.9, (success_coverage - failed_coverage) / 50),
                    description=f"Higher test coverage correlates with success ({success_coverage:.1f}% vs {failed_coverage:.1f}%)",
                    recommendations=[
                        "Increase minimum test coverage requirement",
                        "Generate more comprehensive tests for low-coverage modules",
                        "Focus test generation on critical code paths",
                    ],
                    affected_metrics=["test_coverage", "success_rate"],
                    evidence={
                        "success_coverage": success_coverage,
                        "failed_coverage": failed_coverage,
                        "sample_size": len(metrics),
                    },
                )
            )

        # Analyze security score impact
        success_security = np.mean([m.security_score for m in successful])
        failed_security = np.mean([m.security_score for m in failed])

        if success_security > failed_security + 15:  # 15 point difference
            patterns.append(
                LearningPattern(
                    pattern_type="security_correlation",
                    confidence=min(
                        0.9, (success_security - failed_security) / 50),
                    description=f"Higher security scores correlate with success ({success_security:.0f} vs {failed_security:.0f})",
                    recommendations=[
                        "Implement stricter security scanning thresholds",
                        "Add automated security fixes where possible",
                        "Review security patterns in failed deployments",
                    ],
                    affected_metrics=["security_score", "success_rate"],
                    evidence={
                        "success_security": success_security,
                        "failed_security": failed_security,
                    },
                )
            )

        # Analyze failure stages
        failure_stages = [m.failure_stage for m in failed if m.failure_stage]
        if failure_stages:
            stage_counts = {}
            for stage in failure_stages:
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

            most_common_stage = max(stage_counts, key=stage_counts.get)
            failure_rate = stage_counts[most_common_stage] / len(failed)

            if failure_rate > 0.3:  # 30% of failures in same stage
                patterns.append(
                    LearningPattern(
                        pattern_type="failure_hotspot",
                        confidence=failure_rate,
                        description=f"Most failures occur in {most_common_stage} stage ({failure_rate:.1%})",
                        recommendations=[
                            f"Improve reliability of {most_common_stage} stage",
                            f"Add better error handling in {most_common_stage}",
                            f"Consider splitting {most_common_stage} into smaller steps",
                        ],
                        affected_metrics=["success_rate", "failure_stage"],
                        evidence={
                            "stage_failures": stage_counts,
                            "most_common": most_common_stage,
                            "failure_rate": failure_rate,
                        },
                    )
                )

        return patterns

    def analyze_performance_patterns(
        self, metrics: List[CICDMetrics]
    ) -> List[LearningPattern]:
        """Analyze performance patterns"""
        patterns = []

        if len(metrics) < self.min_samples:
            return patterns

        durations = [m.duration_seconds for m in metrics]
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)

        # Find slow pipelines
        slow_threshold = mean_duration + 2 * std_duration
        slow_pipelines = [
            m for m in metrics if m.duration_seconds > slow_threshold]

        if len(slow_pipelines) > len(metrics) * 0.1:  # More than 10% are slow
            # Analyze what makes pipelines slow
            slow_apps = [m.app_name for m in slow_pipelines]
            app_counts = {}
            for app in slow_apps:
                app_counts[app] = app_counts.get(app, 0) + 1

            if app_counts:
                slowest_app = max(app_counts, key=app_counts.get)
                patterns.append(
                    LearningPattern(
                        pattern_type="performance_bottleneck",
                        confidence=0.8,
                        description=f"Application {slowest_app} consistently has slow pipelines",
                        recommendations=[
                            f"Optimize build process for {slowest_app}",
                            f"Consider parallel execution for {slowest_app} tests",
                            f"Review dependencies and build steps for {slowest_app}",
                        ],
                        affected_metrics=["duration_seconds"],
                        evidence={
                            "slow_threshold": slow_threshold,
                            "slow_apps": app_counts,
                            "mean_duration": mean_duration,
                        },
                    )
                )

        return patterns

    def analyze_temporal_patterns(
        self, metrics: List[CICDMetrics]
    ) -> List[LearningPattern]:
        """Analyze patterns over time"""
        patterns = []

        if len(metrics) < self.min_samples * 2:
            return patterns

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Split into recent and older
        split_point = len(sorted_metrics) // 2
        older_metrics = sorted_metrics[:split_point]
        recent_metrics = sorted_metrics[split_point:]

        # Compare success rates
        older_success_rate = sum(1 for m in older_metrics if m.success) / len(
            older_metrics
        )
        recent_success_rate = sum(1 for m in recent_metrics if m.success) / len(
            recent_metrics
        )

        improvement = recent_success_rate - older_success_rate

        if abs(improvement) > 0.1:  # 10% change
            if improvement > 0:
                patterns.append(
                    LearningPattern(
                        pattern_type="improving_trend",
                        confidence=min(0.9, improvement * 2),
                        description=f"Success rate improving over time ({older_success_rate:.1%} → {recent_success_rate:.1%})",
                        recommendations=[
                            "Continue current improvement practices",
                            "Document what changes led to improvement",
                            "Apply successful patterns to other applications",
                        ],
                        affected_metrics=["success_rate"],
                        evidence={
                            "older_rate": older_success_rate,
                            "recent_rate": recent_success_rate,
                            "improvement": improvement,
                        },
                    )
                )
            else:
                patterns.append(
                    LearningPattern(
                        pattern_type="degrading_trend",
                        confidence=min(0.9, abs(improvement) * 2),
                        description=f"Success rate degrading over time ({older_success_rate:.1%} → {recent_success_rate:.1%})",
                        recommendations=[
                            "Investigate recent changes causing degradation",
                            "Review and update testing strategies",
                            "Consider rolling back recent infrastructure changes",
                        ],
                        affected_metrics=["success_rate"],
                        evidence={
                            "older_rate": older_success_rate,
                            "recent_rate": recent_success_rate,
                            "degradation": improvement,
                        },
                    )
                )

        return patterns


class CICDLearningSystem:
    """Main learning system for CI/CD pipeline optimization"""

    def __init__(self):
        self.analyzer = CICDAnalyzer()
        self.db_path = "/opt/tower-echo-brain/data/cicd_learning.db"
        self._init_database()

    def _init_database(self):
        """Initialize learning database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_id TEXT UNIQUE,
                    app_name TEXT,
                    success BOOLEAN,
                    duration_seconds REAL,
                    test_coverage REAL,
                    security_score INTEGER,
                    stages_completed INTEGER,
                    total_stages INTEGER,
                    failure_stage TEXT,
                    git_commit TEXT,
                    trigger_type TEXT,
                    target_environments TEXT,
                    timestamp TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    confidence REAL,
                    description TEXT,
                    recommendations TEXT,
                    affected_metrics TEXT,
                    evidence TEXT,
                    identified_at TEXT,
                    status TEXT DEFAULT 'active',
                    applied_at TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id INTEGER,
                    action_type TEXT,
                    description TEXT,
                    parameters TEXT,
                    applied_at TEXT,
                    success BOOLEAN,
                    impact_metrics TEXT,
                    FOREIGN KEY (pattern_id) REFERENCES learning_patterns (id)
                )
            """
            )

    async def record_pipeline_metrics(self, metrics: CICDMetrics):
        """Record pipeline metrics for learning"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pipeline_metrics 
                (pipeline_id, app_name, success, duration_seconds, test_coverage, 
                 security_score, stages_completed, total_stages, failure_stage, 
                 git_commit, trigger_type, target_environments, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.pipeline_id,
                    metrics.app_name,
                    metrics.success,
                    metrics.duration_seconds,
                    metrics.test_coverage,
                    metrics.security_score,
                    metrics.stages_completed,
                    metrics.total_stages,
                    metrics.failure_stage,
                    metrics.git_commit,
                    metrics.trigger_type,
                    json.dumps(metrics.target_environments),
                    metrics.timestamp.isoformat(),
                ),
            )

    async def analyze_and_learn(self) -> List[LearningPattern]:
        """Analyze recent data and identify learning patterns"""
        # Get recent metrics (last 30 days)
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM pipeline_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """,
                (cutoff,),
            )

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

        if not rows:
            return []

        # Convert to metrics objects
        metrics = []
        for row in rows:
            data = dict(zip(columns, row))
            metrics.append(
                CICDMetrics(
                    pipeline_id=data["pipeline_id"],
                    app_name=data["app_name"],
                    success=bool(data["success"]),
                    duration_seconds=data["duration_seconds"],
                    test_coverage=data["test_coverage"],
                    security_score=data["security_score"],
                    stages_completed=data["stages_completed"],
                    total_stages=data["total_stages"],
                    failure_stage=data["failure_stage"],
                    git_commit=data["git_commit"],
                    trigger_type=data["trigger_type"],
                    target_environments=json.loads(
                        data["target_environments"]),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                )
            )

        # Analyze patterns
        patterns = []
        patterns.extend(self.analyzer.analyze_success_patterns(metrics))
        patterns.extend(self.analyzer.analyze_performance_patterns(metrics))
        patterns.extend(self.analyzer.analyze_temporal_patterns(metrics))

        # Save new patterns
        for pattern in patterns:
            await self._save_pattern(pattern)

        return patterns

    async def _save_pattern(self, pattern: LearningPattern):
        """Save learning pattern to database"""
        with sqlite3.connect(self.db_path) as conn:
            # Check if similar pattern already exists
            cursor = conn.execute(
                """
                SELECT id FROM learning_patterns 
                WHERE pattern_type = ? AND status = 'active'
                ORDER BY identified_at DESC LIMIT 1
            """,
                (pattern.pattern_type,),
            )

            existing = cursor.fetchone()
            if existing and pattern.confidence < 0.9:
                # Don't save low-confidence duplicates
                return

            conn.execute(
                """
                INSERT INTO learning_patterns 
                (pattern_type, confidence, description, recommendations, 
                 affected_metrics, evidence, identified_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pattern.pattern_type,
                    pattern.confidence,
                    pattern.description,
                    json.dumps(pattern.recommendations),
                    json.dumps(pattern.affected_metrics),
                    json.dumps(pattern.evidence, default=str),
                    datetime.now().isoformat(),
                ),
            )

    async def get_active_patterns(self) -> List[LearningPattern]:
        """Get currently active learning patterns"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM learning_patterns 
                WHERE status = 'active' 
                ORDER BY confidence DESC, identified_at DESC
            """
            )

            columns = [description[0] for description in cursor.description]
            patterns = []

            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                patterns.append(
                    LearningPattern(
                        pattern_type=data["pattern_type"],
                        confidence=data["confidence"],
                        description=data["description"],
                        recommendations=json.loads(data["recommendations"]),
                        affected_metrics=json.loads(data["affected_metrics"]),
                        evidence=json.loads(data["evidence"]),
                    )
                )

            return patterns

    async def apply_optimization(
        self, pattern_id: int, action_type: str, parameters: Dict[str, Any]
    ) -> bool:
        """Apply an optimization based on a learning pattern"""
        try:
            # Record the action
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO optimization_actions 
                    (pattern_id, action_type, description, parameters, applied_at, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern_id,
                        action_type,
                        f"Applied {action_type} optimization",
                        json.dumps(parameters),
                        datetime.now().isoformat(),
                        True,  # Will be updated based on actual result
                    ),
                )

            # Apply the optimization (implement specific actions)
            if action_type == "increase_coverage_threshold":
                await self._update_coverage_threshold(
                    parameters.get("new_threshold", 85)
                )
            elif action_type == "improve_security_scanning":
                await self._enhance_security_scanning(parameters)
            elif action_type == "optimize_build_process":
                await self._optimize_build_process(parameters)

            return True

        except Exception as e:
            print(f"Error applying optimization: {e}")
            return False

    async def _update_coverage_threshold(self, new_threshold: float):
        """Update test coverage threshold"""
        # Update configuration in test generator
        config_path = "/opt/tower-echo-brain/config/test_config.json"
        config = {}

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        config["coverage_threshold"] = new_threshold

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    async def _enhance_security_scanning(self, parameters: Dict[str, Any]):
        """Enhance security scanning based on learning"""
        # Update security configuration
        config_path = "/opt/tower-echo-brain/config/security_config.json"
        config = {}

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        config.update(parameters)

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    async def _optimize_build_process(self, parameters: Dict[str, Any]):
        """Optimize build process based on learning"""
        # Update build configuration
        config_path = "/opt/tower-echo-brain/config/build_config.json"
        config = {}

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        config.update(parameters)

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights"""
        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_pipelines,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_seconds) as avg_duration,
                    AVG(test_coverage) as avg_coverage,
                    AVG(security_score) as avg_security
                FROM pipeline_metrics
                WHERE timestamp > datetime('now', '-30 days')
            """
            )

            stats = cursor.fetchone()

            # Recent trends
            cursor = conn.execute(
                """
                SELECT DATE(timestamp) as date, 
                       AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as daily_success_rate
                FROM pipeline_metrics
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            """
            )

            trends = cursor.fetchall()

            # Active patterns count
            cursor = conn.execute(
                """
                SELECT pattern_type, COUNT(*) as count
                FROM learning_patterns
                WHERE status = 'active'
                GROUP BY pattern_type
            """
            )

            pattern_counts = dict(cursor.fetchall())

        return {
            "total_pipelines": stats[0] if stats else 0,
            "success_rate": stats[1] if stats else 0,
            "avg_duration_minutes": (stats[2] / 60) if stats and stats[2] else 0,
            "avg_coverage": stats[3] if stats else 0,
            "avg_security_score": stats[4] if stats else 0,
            "daily_trends": [
                {"date": row[0], "success_rate": row[1]} for row in trends
            ],
            "active_patterns": pattern_counts,
            "learning_status": "active" if pattern_counts else "insufficient_data",
        }


# Integration endpoint for pipeline feedback

app = FastAPI(title="Echo CI/CD Learning Integration", version="1.0.0")
learning_system = CICDLearningSystem()


class PipelineFeedback(BaseModel):
    pipeline_id: str
    app_name: str
    success: bool
    duration_seconds: float
    test_coverage: float = 0.0
    security_score: int = 0
    stages_completed: int = 0
    total_stages: int = 0
    failure_stage: Optional[str] = None
    git_commit: str
    trigger_type: str = "unknown"
    target_environments: List[str] = []


@app.post("/api/learning/pipeline-feedback")
async def record_pipeline_feedback(feedback: PipelineFeedback):
    """Record pipeline feedback for learning"""
    try:
        metrics = CICDMetrics(
            pipeline_id=feedback.pipeline_id,
            app_name=feedback.app_name,
            success=feedback.success,
            duration_seconds=feedback.duration_seconds,
            test_coverage=feedback.test_coverage,
            security_score=feedback.security_score,
            stages_completed=feedback.stages_completed,
            total_stages=feedback.total_stages,
            failure_stage=feedback.failure_stage,
            git_commit=feedback.git_commit,
            trigger_type=feedback.trigger_type,
            target_environments=feedback.target_environments,
            timestamp=datetime.now(),
        )

        await learning_system.record_pipeline_metrics(metrics)

        # Trigger learning analysis periodically
        if hash(feedback.pipeline_id) % 10 == 0:  # 10% of the time
            patterns = await learning_system.analyze_and_learn()
            return {
                "status": "recorded",
                "new_patterns": len(patterns),
                "learning_triggered": True,
            }

        return {"status": "recorded", "learning_triggered": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/learning/patterns")
async def get_learning_patterns():
    """Get active learning patterns"""
    patterns = await learning_system.get_active_patterns()
    return {"patterns": [asdict(p) for p in patterns], "count": len(patterns)}


@app.get("/api/learning/insights")
async def get_learning_insights():
    """Get comprehensive learning insights"""
    return await learning_system.get_learning_insights()


@app.post("/api/learning/analyze")
async def trigger_analysis():
    """Manually trigger learning analysis"""
    patterns = await learning_system.analyze_and_learn()
    return {"new_patterns": len(patterns), "patterns": [asdict(p) for p in patterns]}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "echo_cicd_learning_integration",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8344)
