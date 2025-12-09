#!/usr/bin/env python3
"""
Echo Outcome Tracker - Decision Results and Impact Analysis
=========================================================

Echo's comprehensive outcome tracking system:
- Tracks results of all decisions and actions over time
- Correlates decisions with their real-world outcomes
- Measures impact and effectiveness of different approaches
- Provides feedback loops for continuous improvement
- Generates insights on decision-making patterns and success factors

This system enables Echo to learn from experience and improve
decision-making through systematic outcome analysis.
"""

import asyncio
import logging
import json
import sqlite3
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import uuid

# Import other learning systems
from echo_learning_system import get_learning_system, LearningDomain, DecisionOutcome
from echo_board_manager import get_board_manager
from echo_task_decomposer import get_task_decomposer

# Configuration
OUTCOME_TRACKER_DB_PATH = "/opt/tower-echo-brain/data/echo_outcome_tracker.db"
CORRELATION_WINDOW_HOURS = 24
MIN_SAMPLES_FOR_CORRELATION = 10
IMPACT_MEASUREMENT_WINDOW_DAYS = 7

class OutcomeType(Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

class ImpactLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    USER_SATISFACTION = "user_satisfaction"
    SYSTEM_HEALTH = "system_health"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    COMPLETION_TIME = "completion_time"

@dataclass
class OutcomeRecord:
    """Record of a decision outcome with detailed metrics"""
    outcome_id: str
    decision_id: str
    decision_context: Dict[str, Any]
    decision_maker: str  # "echo", "board", "user", etc.
    outcome_type: OutcomeType
    timestamp: datetime
    observed_results: Dict[str, Any]
    quantitative_metrics: Dict[str, float]
    qualitative_assessment: str
    impact_level: ImpactLevel
    success_indicators: List[str]
    failure_indicators: List[str]
    unexpected_results: List[str]
    correlation_data: Dict[str, Any]
    learning_insights: List[str]

@dataclass
class ImpactAnalysis:
    """Analysis of decision impact over time"""
    analysis_id: str
    decision_period: Tuple[datetime, datetime]
    decisions_analyzed: int
    overall_success_rate: float
    average_impact_level: float
    key_success_factors: List[str]
    common_failure_patterns: List[str]
    performance_trends: Dict[str, List[float]]
    correlation_insights: List[str]
    recommendations: List[str]
    confidence_level: float

@dataclass
class DecisionPattern:
    """Pattern of decision-making and outcomes"""
    pattern_id: str
    pattern_type: str
    decision_characteristics: Dict[str, Any]
    typical_outcomes: Dict[str, Any]
    success_probability: float
    average_impact: float
    optimal_conditions: List[str]
    risk_factors: List[str]
    improvement_opportunities: List[str]
    sample_size: int
    last_validated: datetime

@dataclass
class PerformanceMetric:
    """Individual performance metric tracking"""
    metric_id: str
    metric_name: str
    metric_type: MetricType
    current_value: float
    baseline_value: float
    trend_direction: str  # "improving", "stable", "degrading"
    measurement_history: List[Tuple[datetime, float]]
    correlation_factors: Dict[str, float]
    impact_on_outcomes: float

class EchoOutcomeTracker:
    """
    Echo's Comprehensive Outcome Tracking System

    This system provides Echo with the ability to learn from experience
    by systematically tracking decision outcomes and their impacts.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = OUTCOME_TRACKER_DB_PATH
        self._ensure_data_directory()
        self._init_database()

        # Tracking state
        self.active_outcomes: Dict[str, OutcomeRecord] = {}
        self.decision_patterns: Dict[str, DecisionPattern] = {}
        self.performance_metrics: Dict[str, PerformanceMetric] = {}

        # Analysis cache
        self.recent_analyses: Dict[str, ImpactAnalysis] = {}
        self.correlation_cache: Dict[str, Any] = {}

        # Integration with other systems
        self.learning_system = get_learning_system()

        self.logger.info("Echo Outcome Tracker initialized")

    def _ensure_data_directory(self):
        """Ensure outcome tracker data directory exists"""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize outcome tracker database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outcome_records (
                    outcome_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    decision_context TEXT NOT NULL,
                    decision_maker TEXT NOT NULL,
                    outcome_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    observed_results TEXT NOT NULL,
                    quantitative_metrics TEXT,
                    qualitative_assessment TEXT,
                    impact_level TEXT NOT NULL,
                    success_indicators TEXT,
                    failure_indicators TEXT,
                    unexpected_results TEXT,
                    correlation_data TEXT,
                    learning_insights TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS impact_analyses (
                    analysis_id TEXT PRIMARY KEY,
                    decision_period_start TEXT NOT NULL,
                    decision_period_end TEXT NOT NULL,
                    decisions_analyzed INTEGER NOT NULL,
                    overall_success_rate REAL NOT NULL,
                    average_impact_level REAL NOT NULL,
                    key_success_factors TEXT,
                    common_failure_patterns TEXT,
                    performance_trends TEXT,
                    correlation_insights TEXT,
                    recommendations TEXT,
                    confidence_level REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    decision_characteristics TEXT NOT NULL,
                    typical_outcomes TEXT NOT NULL,
                    success_probability REAL NOT NULL,
                    average_impact REAL NOT NULL,
                    optimal_conditions TEXT,
                    risk_factors TEXT,
                    improvement_opportunities TEXT,
                    sample_size INTEGER NOT NULL,
                    last_validated TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    metric_name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    baseline_value REAL NOT NULL,
                    trend_direction TEXT NOT NULL,
                    measurement_history TEXT,
                    correlation_factors TEXT,
                    impact_on_outcomes REAL NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS outcome_correlations (
                    correlation_id TEXT PRIMARY KEY,
                    factor_name TEXT NOT NULL,
                    outcome_metric TEXT NOT NULL,
                    correlation_strength REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    discovery_date TEXT NOT NULL,
                    validation_count INTEGER DEFAULT 0
                )
            """)

            conn.commit()

    async def record_outcome(self,
                           decision_id: str,
                           decision_context: Dict[str, Any],
                           decision_maker: str,
                           observed_results: Dict[str, Any],
                           outcome_type: OutcomeType = OutcomeType.IMMEDIATE,
                           delay_hours: float = 0) -> str:
        """
        Record an outcome for a decision

        Args:
            decision_id: ID of the original decision
            decision_context: Context in which the decision was made
            decision_maker: Who/what made the decision
            observed_results: The actual results observed
            outcome_type: Type of outcome (immediate, short-term, etc.)
            delay_hours: Hours to wait before recording (for delayed outcomes)

        Returns:
            outcome_id: Unique identifier for this outcome record
        """
        outcome_id = f"outcome_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if delay_hours > 0:
            # Schedule delayed outcome recording
            await asyncio.sleep(delay_hours * 3600)

        self.logger.info(f"Recording outcome: {outcome_id} for decision {decision_id}")

        # Analyze the observed results
        analysis = await self._analyze_outcome_results(observed_results, decision_context)

        # Create outcome record
        outcome_record = OutcomeRecord(
            outcome_id=outcome_id,
            decision_id=decision_id,
            decision_context=decision_context,
            decision_maker=decision_maker,
            outcome_type=outcome_type,
            timestamp=datetime.now(),
            observed_results=observed_results,
            quantitative_metrics=analysis["quantitative_metrics"],
            qualitative_assessment=analysis["qualitative_assessment"],
            impact_level=analysis["impact_level"],
            success_indicators=analysis["success_indicators"],
            failure_indicators=analysis["failure_indicators"],
            unexpected_results=analysis["unexpected_results"],
            correlation_data={},  # Will be populated by correlation analysis
            learning_insights=[]  # Will be populated by learning analysis
        )

        # Store the outcome
        self.active_outcomes[outcome_id] = outcome_record
        await self._store_outcome_record(outcome_record)

        # Trigger analysis and learning
        await self._analyze_correlations(outcome_record)
        await self._extract_learning_insights(outcome_record)
        await self._update_performance_metrics(outcome_record)

        # Update decision patterns
        await self._update_decision_patterns(outcome_record)

        # Notify learning system
        await self._notify_learning_systems(outcome_record)

        self.logger.info(f"Outcome recorded and analyzed: {outcome_id}")
        return outcome_id

    async def _analyze_outcome_results(self,
                                     observed_results: Dict[str, Any],
                                     decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze observed results to extract insights"""
        analysis = {
            "quantitative_metrics": {},
            "qualitative_assessment": "",
            "impact_level": ImpactLevel.MEDIUM,
            "success_indicators": [],
            "failure_indicators": [],
            "unexpected_results": []
        }

        # Extract quantitative metrics
        for key, value in observed_results.items():
            if isinstance(value, (int, float)):
                analysis["quantitative_metrics"][key] = float(value)

        # Determine impact level
        analysis["impact_level"] = self._calculate_impact_level(
            observed_results, decision_context
        )

        # Identify success indicators
        analysis["success_indicators"] = self._identify_success_indicators(
            observed_results, decision_context
        )

        # Identify failure indicators
        analysis["failure_indicators"] = self._identify_failure_indicators(
            observed_results, decision_context
        )

        # Identify unexpected results
        analysis["unexpected_results"] = self._identify_unexpected_results(
            observed_results, decision_context
        )

        # Generate qualitative assessment
        analysis["qualitative_assessment"] = self._generate_qualitative_assessment(
            analysis, observed_results, decision_context
        )

        return analysis

    def _calculate_impact_level(self,
                              observed_results: Dict[str, Any],
                              decision_context: Dict[str, Any]) -> ImpactLevel:
        """Calculate the impact level of the outcome"""
        impact_factors = []

        # Check for performance impact
        if "performance_improvement" in observed_results:
            perf_impact = observed_results["performance_improvement"]
            if perf_impact > 0.5:
                impact_factors.append(3)  # High impact
            elif perf_impact > 0.2:
                impact_factors.append(2)  # Medium impact
            else:
                impact_factors.append(1)  # Low impact

        # Check for error rates
        if "error_rate" in observed_results:
            error_rate = observed_results["error_rate"]
            if error_rate > 0.1:
                impact_factors.append(3)  # High negative impact
            elif error_rate > 0.05:
                impact_factors.append(2)  # Medium negative impact
            else:
                impact_factors.append(1)  # Low impact

        # Check for user satisfaction
        if "user_satisfaction" in observed_results:
            satisfaction = observed_results["user_satisfaction"]
            if satisfaction > 0.8:
                impact_factors.append(2)  # Medium positive impact
            elif satisfaction < 0.4:
                impact_factors.append(3)  # High negative impact
            else:
                impact_factors.append(1)  # Low impact

        # Check decision context for scope indicators
        if decision_context.get("scope") == "system_wide":
            impact_factors.append(3)  # System-wide changes have high impact
        elif decision_context.get("scope") == "critical":
            impact_factors.append(3)  # Critical decisions have high impact

        # Calculate average impact
        if impact_factors:
            avg_impact = statistics.mean(impact_factors)
            if avg_impact >= 2.5:
                return ImpactLevel.HIGH
            elif avg_impact >= 2.0:
                return ImpactLevel.MEDIUM
            elif avg_impact >= 1.5:
                return ImpactLevel.LOW
            else:
                return ImpactLevel.MINIMAL
        else:
            return ImpactLevel.MEDIUM  # Default

    def _identify_success_indicators(self,
                                   observed_results: Dict[str, Any],
                                   decision_context: Dict[str, Any]) -> List[str]:
        """Identify indicators of success in the outcome"""
        indicators = []

        # Performance improvements
        if observed_results.get("performance_improvement", 0) > 0.1:
            indicators.append("performance_improvement_achieved")

        # Error rate reductions
        if observed_results.get("error_rate", 1.0) < 0.05:
            indicators.append("low_error_rate_maintained")

        # User satisfaction
        if observed_results.get("user_satisfaction", 0) > 0.7:
            indicators.append("high_user_satisfaction")

        # Task completion
        if observed_results.get("task_completed", False):
            indicators.append("task_successfully_completed")

        # Time efficiency
        if observed_results.get("completion_time", float('inf')) < decision_context.get("estimated_time", float('inf')):
            indicators.append("completed_ahead_of_schedule")

        # Resource efficiency
        if observed_results.get("resource_usage", 1.0) < 0.8:
            indicators.append("efficient_resource_usage")

        return indicators

    def _identify_failure_indicators(self,
                                   observed_results: Dict[str, Any],
                                   decision_context: Dict[str, Any]) -> List[str]:
        """Identify indicators of failure in the outcome"""
        indicators = []

        # High error rates
        if observed_results.get("error_rate", 0) > 0.1:
            indicators.append("high_error_rate_detected")

        # Poor performance
        if observed_results.get("performance_improvement", 0) < -0.1:
            indicators.append("performance_degradation")

        # Low user satisfaction
        if observed_results.get("user_satisfaction", 1.0) < 0.4:
            indicators.append("low_user_satisfaction")

        # Task failure
        if observed_results.get("task_failed", False):
            indicators.append("task_execution_failed")

        # Timeout or delays
        if observed_results.get("completion_time", 0) > decision_context.get("deadline", float('inf')):
            indicators.append("deadline_exceeded")

        # Resource overuse
        if observed_results.get("resource_usage", 0) > 1.2:
            indicators.append("excessive_resource_usage")

        # System instability
        if observed_results.get("system_instability", False):
            indicators.append("system_instability_caused")

        return indicators

    def _identify_unexpected_results(self,
                                   observed_results: Dict[str, Any],
                                   decision_context: Dict[str, Any]) -> List[str]:
        """Identify unexpected results that weren't anticipated"""
        unexpected = []

        # Compare with expected results if available
        expected_results = decision_context.get("expected_results", {})

        for key, actual_value in observed_results.items():
            if key in expected_results:
                expected_value = expected_results[key]
                if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
                    deviation = abs(actual_value - expected_value) / max(abs(expected_value), 1)
                    if deviation > 0.5:  # 50% deviation threshold
                        unexpected.append(f"unexpected_{key}_deviation")

        # Check for emergent behaviors
        emergent_keywords = ["unexpected", "surprise", "emergent", "unintended"]
        for key, value in observed_results.items():
            if isinstance(value, str) and any(keyword in value.lower() for keyword in emergent_keywords):
                unexpected.append(f"emergent_behavior_in_{key}")

        return unexpected

    def _generate_qualitative_assessment(self,
                                       analysis: Dict[str, Any],
                                       observed_results: Dict[str, Any],
                                       decision_context: Dict[str, Any]) -> str:
        """Generate qualitative assessment of the outcome"""
        success_count = len(analysis["success_indicators"])
        failure_count = len(analysis["failure_indicators"])
        unexpected_count = len(analysis["unexpected_results"])

        if success_count > failure_count and unexpected_count == 0:
            assessment = "Outcome largely successful with expected results"
        elif success_count > failure_count and unexpected_count > 0:
            assessment = "Outcome successful but with some unexpected elements"
        elif success_count == failure_count:
            assessment = "Outcome mixed with both positive and negative elements"
        elif failure_count > success_count and unexpected_count == 0:
            assessment = "Outcome largely unsuccessful but predictable"
        else:
            assessment = "Outcome unsuccessful with unexpected complications"

        # Add impact level context
        impact_level = analysis["impact_level"]
        assessment += f" (Impact level: {impact_level.value})"

        return assessment

    async def _analyze_correlations(self, outcome_record: OutcomeRecord):
        """Analyze correlations between decision factors and outcomes"""
        decision_context = outcome_record.decision_context
        quantitative_metrics = outcome_record.quantitative_metrics

        correlations = {}

        # Analyze correlations between context factors and outcome metrics
        for context_key, context_value in decision_context.items():
            if isinstance(context_value, (int, float)):
                for metric_key, metric_value in quantitative_metrics.items():
                    correlation = self._calculate_simple_correlation(
                        context_value, metric_value
                    )
                    if abs(correlation) > 0.3:  # Minimum correlation threshold
                        correlations[f"{context_key}_vs_{metric_key}"] = correlation

        # Store correlations
        outcome_record.correlation_data = correlations

        # Update global correlation tracking
        await self._update_correlation_database(correlations, outcome_record)

    def _calculate_simple_correlation(self, factor_value: float, outcome_value: float) -> float:
        """Calculate simple correlation between two values"""
        # This is a simplified correlation - in practice, you'd use historical data
        # For now, return a normalized relationship
        if factor_value == 0:
            return 0.0

        ratio = outcome_value / factor_value
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, (ratio - 1.0)))

    async def _update_correlation_database(self,
                                         correlations: Dict[str, float],
                                         outcome_record: OutcomeRecord):
        """Update correlation database with new findings"""
        for correlation_key, correlation_value in correlations.items():
            correlation_id = f"corr_{correlation_key}_{datetime.now().strftime('%Y%m%d')}"

            # Check if correlation already exists
            existing_correlation = await self._get_existing_correlation(correlation_key)

            if existing_correlation:
                # Update existing correlation
                updated_strength = (existing_correlation["strength"] + correlation_value) / 2
                updated_confidence = min(1.0, existing_correlation["confidence"] + 0.1)
                await self._store_correlation_update(
                    existing_correlation["id"], updated_strength, updated_confidence
                )
            else:
                # Create new correlation
                await self._store_new_correlation(
                    correlation_id, correlation_key, correlation_value, outcome_record
                )

    async def _get_existing_correlation(self, correlation_key: str) -> Optional[Dict[str, Any]]:
        """Get existing correlation from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT correlation_id, correlation_strength, confidence_level, sample_size
                FROM outcome_correlations
                WHERE factor_name = ?
                ORDER BY discovery_date DESC LIMIT 1
            """, (correlation_key,))

            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "strength": row[1],
                    "confidence": row[2],
                    "sample_size": row[3]
                }
        return None

    async def _store_correlation_update(self,
                                      correlation_id: str,
                                      new_strength: float,
                                      new_confidence: float):
        """Update existing correlation in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE outcome_correlations
                SET correlation_strength = ?, confidence_level = ?, validation_count = validation_count + 1
                WHERE correlation_id = ?
            """, (new_strength, new_confidence, correlation_id))
            conn.commit()

    async def _store_new_correlation(self,
                                   correlation_id: str,
                                   correlation_key: str,
                                   correlation_value: float,
                                   outcome_record: OutcomeRecord):
        """Store new correlation in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO outcome_correlations
                (correlation_id, factor_name, outcome_metric, correlation_strength,
                 confidence_level, sample_size, discovery_date, validation_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                correlation_id, correlation_key, "general_outcome", correlation_value,
                0.5, 1, datetime.now().isoformat(), 1
            ))
            conn.commit()

    async def _extract_learning_insights(self, outcome_record: OutcomeRecord):
        """Extract learning insights from the outcome"""
        insights = []

        # Analyze success patterns
        if len(outcome_record.success_indicators) > len(outcome_record.failure_indicators):
            insights.append(f"Success pattern: {', '.join(outcome_record.success_indicators)}")

        # Analyze failure patterns
        if outcome_record.failure_indicators:
            insights.append(f"Failure pattern identified: {', '.join(outcome_record.failure_indicators)}")

        # Analyze unexpected results
        if outcome_record.unexpected_results:
            insights.append(f"Unexpected behaviors: {', '.join(outcome_record.unexpected_results)}")

        # Decision maker effectiveness
        insights.append(f"Decision maker '{outcome_record.decision_maker}' effectiveness: {outcome_record.impact_level.value} impact")

        # Timing insights
        if outcome_record.outcome_type == OutcomeType.IMMEDIATE:
            insights.append("Immediate feedback available - enables rapid learning")
        elif outcome_record.outcome_type == OutcomeType.LONG_TERM:
            insights.append("Long-term impact observed - validates decision quality")

        outcome_record.learning_insights = insights

    async def _update_performance_metrics(self, outcome_record: OutcomeRecord):
        """Update performance metrics based on outcome"""
        for metric_name, metric_value in outcome_record.quantitative_metrics.items():
            metric_id = f"metric_{metric_name}"

            if metric_id in self.performance_metrics:
                metric = self.performance_metrics[metric_id]
                # Update metric with new value
                metric.measurement_history.append((outcome_record.timestamp, metric_value))
                metric.current_value = metric_value
                metric.trend_direction = self._calculate_trend(metric.measurement_history)
            else:
                # Create new metric
                metric = PerformanceMetric(
                    metric_id=metric_id,
                    metric_name=metric_name,
                    metric_type=self._classify_metric_type(metric_name),
                    current_value=metric_value,
                    baseline_value=metric_value,
                    trend_direction="stable",
                    measurement_history=[(outcome_record.timestamp, metric_value)],
                    correlation_factors={},
                    impact_on_outcomes=0.5  # Default impact
                )
                self.performance_metrics[metric_id] = metric

            await self._store_performance_metric(metric)

    def _classify_metric_type(self, metric_name: str) -> MetricType:
        """Classify the type of metric based on its name"""
        metric_name_lower = metric_name.lower()

        type_mapping = {
            "performance": MetricType.PERFORMANCE,
            "efficiency": MetricType.EFFICIENCY,
            "accuracy": MetricType.ACCURACY,
            "satisfaction": MetricType.USER_SATISFACTION,
            "health": MetricType.SYSTEM_HEALTH,
            "resource": MetricType.RESOURCE_USAGE,
            "error": MetricType.ERROR_RATE,
            "time": MetricType.COMPLETION_TIME
        }

        for keyword, metric_type in type_mapping.items():
            if keyword in metric_name_lower:
                return metric_type

        return MetricType.PERFORMANCE  # Default

    def _calculate_trend(self, measurement_history: List[Tuple[datetime, float]]) -> str:
        """Calculate trend direction from measurement history"""
        if len(measurement_history) < 3:
            return "stable"

        # Get recent values
        recent_values = [value for _, value in measurement_history[-5:]]

        # Calculate simple trend
        if len(recent_values) >= 3:
            early_avg = statistics.mean(recent_values[:2])
            late_avg = statistics.mean(recent_values[-2:])

            if late_avg > early_avg * 1.05:
                return "improving"
            elif late_avg < early_avg * 0.95:
                return "degrading"

        return "stable"

    async def _update_decision_patterns(self, outcome_record: OutcomeRecord):
        """Update decision patterns based on outcome"""
        decision_characteristics = self._extract_decision_characteristics(outcome_record)
        pattern_key = self._generate_pattern_key(decision_characteristics)

        if pattern_key in self.decision_patterns:
            # Update existing pattern
            pattern = self.decision_patterns[pattern_key]
            pattern.sample_size += 1

            # Update success probability
            is_success = len(outcome_record.success_indicators) > len(outcome_record.failure_indicators)
            alpha = 0.1  # Learning rate
            pattern.success_probability = (
                alpha * (1.0 if is_success else 0.0) +
                (1 - alpha) * pattern.success_probability
            )

            # Update average impact
            impact_value = self._impact_level_to_numeric(outcome_record.impact_level)
            pattern.average_impact = (
                alpha * impact_value +
                (1 - alpha) * pattern.average_impact
            )

            pattern.last_validated = datetime.now()

        else:
            # Create new pattern
            is_success = len(outcome_record.success_indicators) > len(outcome_record.failure_indicators)
            pattern = DecisionPattern(
                pattern_id=f"pattern_{pattern_key}_{datetime.now().strftime('%Y%m%d')}",
                pattern_type=outcome_record.decision_maker,
                decision_characteristics=decision_characteristics,
                typical_outcomes={
                    "success_indicators": outcome_record.success_indicators,
                    "failure_indicators": outcome_record.failure_indicators,
                    "impact_level": outcome_record.impact_level.value
                },
                success_probability=1.0 if is_success else 0.0,
                average_impact=self._impact_level_to_numeric(outcome_record.impact_level),
                optimal_conditions=[],
                risk_factors=[],
                improvement_opportunities=[],
                sample_size=1,
                last_validated=datetime.now()
            )
            self.decision_patterns[pattern_key] = pattern

        await self._store_decision_pattern(pattern)

    def _extract_decision_characteristics(self, outcome_record: OutcomeRecord) -> Dict[str, Any]:
        """Extract key characteristics of the decision"""
        context = outcome_record.decision_context
        characteristics = {}

        # Extract key context features
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                characteristics[key] = value

        # Add outcome-derived characteristics
        characteristics["decision_maker"] = outcome_record.decision_maker
        characteristics["outcome_type"] = outcome_record.outcome_type.value
        characteristics["has_success_indicators"] = len(outcome_record.success_indicators) > 0
        characteristics["has_failure_indicators"] = len(outcome_record.failure_indicators) > 0

        return characteristics

    def _generate_pattern_key(self, characteristics: Dict[str, Any]) -> str:
        """Generate a key for pattern matching"""
        # Create a hash-like key from key characteristics
        key_parts = []

        # Sort characteristics for consistent key generation
        for key, value in sorted(characteristics.items()):
            if isinstance(value, bool):
                key_parts.append(f"{key}:{value}")
            elif isinstance(value, str):
                key_parts.append(f"{key}:{value[:10]}")  # Truncate strings
            elif isinstance(value, (int, float)):
                # Bucket numeric values
                if value > 1:
                    key_parts.append(f"{key}:high")
                elif value > 0.5:
                    key_parts.append(f"{key}:medium")
                else:
                    key_parts.append(f"{key}:low")

        return "_".join(key_parts[:5])  # Limit key length

    def _impact_level_to_numeric(self, impact_level: ImpactLevel) -> float:
        """Convert impact level to numeric value"""
        mapping = {
            ImpactLevel.MINIMAL: 0.1,
            ImpactLevel.LOW: 0.3,
            ImpactLevel.MEDIUM: 0.5,
            ImpactLevel.HIGH: 0.8,
            ImpactLevel.CRITICAL: 1.0
        }
        return mapping.get(impact_level, 0.5)

    async def _notify_learning_systems(self, outcome_record: OutcomeRecord):
        """Notify other learning systems about the outcome"""
        try:
            # Convert to DecisionOutcome for learning system
            if len(outcome_record.success_indicators) > len(outcome_record.failure_indicators):
                learning_outcome = DecisionOutcome.SUCCESS
            elif len(outcome_record.failure_indicators) > 0:
                learning_outcome = DecisionOutcome.FAILURE
            else:
                learning_outcome = DecisionOutcome.PARTIAL

            # Notify learning system if we have a decision ID that maps to it
            if hasattr(self.learning_system, 'record_outcome'):
                await self.learning_system.record_outcome(
                    outcome_record.decision_id,
                    learning_outcome,
                    outcome_record.quantitative_metrics,
                    outcome_record.qualitative_assessment
                )

        except Exception as e:
            self.logger.error(f"Error notifying learning systems: {e}")

    async def analyze_impact_over_period(self,
                                       start_date: datetime,
                                       end_date: datetime,
                                       decision_maker: Optional[str] = None) -> ImpactAnalysis:
        """Analyze decision impact over a specific period"""
        analysis_id = f"analysis_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"

        self.logger.info(f"Analyzing impact from {start_date} to {end_date}")

        # Get outcomes in period
        outcomes = await self._get_outcomes_in_period(start_date, end_date, decision_maker)

        if not outcomes:
            return ImpactAnalysis(
                analysis_id=analysis_id,
                decision_period=(start_date, end_date),
                decisions_analyzed=0,
                overall_success_rate=0.0,
                average_impact_level=0.0,
                key_success_factors=[],
                common_failure_patterns=[],
                performance_trends={},
                correlation_insights=[],
                recommendations=[],
                confidence_level=0.0
            )

        # Calculate overall metrics
        success_count = sum(
            1 for outcome in outcomes
            if len(outcome.success_indicators) > len(outcome.failure_indicators)
        )
        overall_success_rate = success_count / len(outcomes)

        impact_values = [self._impact_level_to_numeric(outcome.impact_level) for outcome in outcomes]
        average_impact_level = statistics.mean(impact_values)

        # Analyze success factors
        key_success_factors = self._identify_key_success_factors(outcomes)

        # Analyze failure patterns
        common_failure_patterns = self._identify_failure_patterns(outcomes)

        # Analyze performance trends
        performance_trends = self._analyze_performance_trends(outcomes)

        # Generate correlation insights
        correlation_insights = await self._generate_correlation_insights(outcomes)

        # Generate recommendations
        recommendations = self._generate_impact_recommendations(
            outcomes, overall_success_rate, average_impact_level
        )

        # Calculate confidence level
        confidence_level = min(1.0, len(outcomes) / 50)  # Higher confidence with more data

        analysis = ImpactAnalysis(
            analysis_id=analysis_id,
            decision_period=(start_date, end_date),
            decisions_analyzed=len(outcomes),
            overall_success_rate=overall_success_rate,
            average_impact_level=average_impact_level,
            key_success_factors=key_success_factors,
            common_failure_patterns=common_failure_patterns,
            performance_trends=performance_trends,
            correlation_insights=correlation_insights,
            recommendations=recommendations,
            confidence_level=confidence_level
        )

        self.recent_analyses[analysis_id] = analysis
        await self._store_impact_analysis(analysis)

        return analysis

    async def _get_outcomes_in_period(self,
                                    start_date: datetime,
                                    end_date: datetime,
                                    decision_maker: Optional[str] = None) -> List[OutcomeRecord]:
        """Get outcome records within a time period"""
        outcomes = []

        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM outcome_records
                WHERE timestamp BETWEEN ? AND ?
            """
            params = [start_date.isoformat(), end_date.isoformat()]

            if decision_maker:
                query += " AND decision_maker = ?"
                params.append(decision_maker)

            query += " ORDER BY timestamp"

            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                outcome = self._row_to_outcome_record(row)
                outcomes.append(outcome)

        return outcomes

    def _identify_key_success_factors(self, outcomes: List[OutcomeRecord]) -> List[str]:
        """Identify key factors that correlate with success"""
        success_factor_counts = {}

        for outcome in outcomes:
            is_success = len(outcome.success_indicators) > len(outcome.failure_indicators)
            if is_success:
                for indicator in outcome.success_indicators:
                    success_factor_counts[indicator] = success_factor_counts.get(indicator, 0) + 1

        # Sort by frequency and return top factors
        sorted_factors = sorted(success_factor_counts.items(), key=lambda x: x[1], reverse=True)
        return [factor for factor, count in sorted_factors[:5] if count >= 2]

    def _identify_failure_patterns(self, outcomes: List[OutcomeRecord]) -> List[str]:
        """Identify common failure patterns"""
        failure_pattern_counts = {}

        for outcome in outcomes:
            is_failure = len(outcome.failure_indicators) > len(outcome.success_indicators)
            if is_failure:
                for indicator in outcome.failure_indicators:
                    failure_pattern_counts[indicator] = failure_pattern_counts.get(indicator, 0) + 1

        # Sort by frequency and return top patterns
        sorted_patterns = sorted(failure_pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return [pattern for pattern, count in sorted_patterns[:5] if count >= 2]

    def _analyze_performance_trends(self, outcomes: List[OutcomeRecord]) -> Dict[str, List[float]]:
        """Analyze performance trends over the period"""
        trends = {}

        # Group outcomes by time windows
        time_windows = self._create_time_windows(outcomes)

        for window_name, window_outcomes in time_windows.items():
            # Calculate metrics for each window
            window_metrics = {}
            for outcome in window_outcomes:
                for metric_name, metric_value in outcome.quantitative_metrics.items():
                    if metric_name not in window_metrics:
                        window_metrics[metric_name] = []
                    window_metrics[metric_name].append(metric_value)

            # Average metrics for this window
            for metric_name, values in window_metrics.items():
                if metric_name not in trends:
                    trends[metric_name] = []
                trends[metric_name].append(statistics.mean(values))

        return trends

    def _create_time_windows(self, outcomes: List[OutcomeRecord]) -> Dict[str, List[OutcomeRecord]]:
        """Create time windows for trend analysis"""
        if not outcomes:
            return {}

        # Sort outcomes by timestamp
        sorted_outcomes = sorted(outcomes, key=lambda x: x.timestamp)

        # Create daily windows
        windows = {}
        current_date = None
        current_window = []

        for outcome in sorted_outcomes:
            outcome_date = outcome.timestamp.date()

            if current_date != outcome_date:
                if current_window:
                    windows[f"day_{current_date}"] = current_window
                current_date = outcome_date
                current_window = []

            current_window.append(outcome)

        # Add final window
        if current_window:
            windows[f"day_{current_date}"] = current_window

        return windows

    async def _generate_correlation_insights(self, outcomes: List[OutcomeRecord]) -> List[str]:
        """Generate insights about correlations"""
        insights = []

        # Analyze correlations across all outcomes
        all_correlations = {}
        for outcome in outcomes:
            for corr_key, corr_value in outcome.correlation_data.items():
                if corr_key not in all_correlations:
                    all_correlations[corr_key] = []
                all_correlations[corr_key].append(corr_value)

        # Find strong correlations
        for corr_key, values in all_correlations.items():
            if len(values) >= 3:  # Minimum sample size
                avg_correlation = statistics.mean(values)
                if abs(avg_correlation) > 0.5:
                    direction = "positive" if avg_correlation > 0 else "negative"
                    insights.append(f"Strong {direction} correlation found: {corr_key}")

        return insights

    def _generate_impact_recommendations(self,
                                       outcomes: List[OutcomeRecord],
                                       success_rate: float,
                                       average_impact: float) -> List[str]:
        """Generate recommendations based on impact analysis"""
        recommendations = []

        if success_rate < 0.6:
            recommendations.append("Overall success rate is low - review decision-making processes")

        if average_impact < 0.4:
            recommendations.append("Average impact is low - consider focusing on higher-impact decisions")

        # Analyze decision makers
        decision_maker_performance = {}
        for outcome in outcomes:
            maker = outcome.decision_maker
            is_success = len(outcome.success_indicators) > len(outcome.failure_indicators)

            if maker not in decision_maker_performance:
                decision_maker_performance[maker] = {"successes": 0, "total": 0}

            decision_maker_performance[maker]["total"] += 1
            if is_success:
                decision_maker_performance[maker]["successes"] += 1

        # Recommend based on decision maker performance
        for maker, performance in decision_maker_performance.items():
            if performance["total"] >= 3:  # Minimum sample size
                success_rate = performance["successes"] / performance["total"]
                if success_rate < 0.5:
                    recommendations.append(f"Consider reviewing {maker} decision-making effectiveness")
                elif success_rate > 0.8:
                    recommendations.append(f"{maker} shows strong decision-making performance")

        # Pattern-based recommendations
        failure_indicators = []
        for outcome in outcomes:
            failure_indicators.extend(outcome.failure_indicators)

        if failure_indicators:
            most_common_failure = max(set(failure_indicators), key=failure_indicators.count)
            recommendations.append(f"Address recurring issue: {most_common_failure}")

        return recommendations

    # Storage methods

    async def _store_outcome_record(self, outcome_record: OutcomeRecord):
        """Store outcome record in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO outcome_records
                (outcome_id, decision_id, decision_context, decision_maker, outcome_type,
                 timestamp, observed_results, quantitative_metrics, qualitative_assessment,
                 impact_level, success_indicators, failure_indicators, unexpected_results,
                 correlation_data, learning_insights)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome_record.outcome_id, outcome_record.decision_id,
                json.dumps(outcome_record.decision_context),
                outcome_record.decision_maker, outcome_record.outcome_type.value,
                outcome_record.timestamp.isoformat(),
                json.dumps(outcome_record.observed_results),
                json.dumps(outcome_record.quantitative_metrics),
                outcome_record.qualitative_assessment, outcome_record.impact_level.value,
                json.dumps(outcome_record.success_indicators),
                json.dumps(outcome_record.failure_indicators),
                json.dumps(outcome_record.unexpected_results),
                json.dumps(outcome_record.correlation_data),
                json.dumps(outcome_record.learning_insights)
            ))
            conn.commit()

    async def _store_impact_analysis(self, analysis: ImpactAnalysis):
        """Store impact analysis in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO impact_analyses
                (analysis_id, decision_period_start, decision_period_end, decisions_analyzed,
                 overall_success_rate, average_impact_level, key_success_factors,
                 common_failure_patterns, performance_trends, correlation_insights,
                 recommendations, confidence_level, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.analysis_id, analysis.decision_period[0].isoformat(),
                analysis.decision_period[1].isoformat(), analysis.decisions_analyzed,
                analysis.overall_success_rate, analysis.average_impact_level,
                json.dumps(analysis.key_success_factors),
                json.dumps(analysis.common_failure_patterns),
                json.dumps(analysis.performance_trends),
                json.dumps(analysis.correlation_insights),
                json.dumps(analysis.recommendations), analysis.confidence_level,
                datetime.now().isoformat()
            ))
            conn.commit()

    async def _store_decision_pattern(self, pattern: DecisionPattern):
        """Store decision pattern in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO decision_patterns
                (pattern_id, pattern_type, decision_characteristics, typical_outcomes,
                 success_probability, average_impact, optimal_conditions, risk_factors,
                 improvement_opportunities, sample_size, last_validated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id, pattern.pattern_type,
                json.dumps(pattern.decision_characteristics),
                json.dumps(pattern.typical_outcomes), pattern.success_probability,
                pattern.average_impact, json.dumps(pattern.optimal_conditions),
                json.dumps(pattern.risk_factors),
                json.dumps(pattern.improvement_opportunities),
                pattern.sample_size, pattern.last_validated.isoformat()
            ))
            conn.commit()

    async def _store_performance_metric(self, metric: PerformanceMetric):
        """Store performance metric in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO performance_metrics
                (metric_id, metric_name, metric_type, current_value, baseline_value,
                 trend_direction, measurement_history, correlation_factors,
                 impact_on_outcomes, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_id, metric.metric_name, metric.metric_type.value,
                metric.current_value, metric.baseline_value, metric.trend_direction,
                json.dumps([(ts.isoformat(), val) for ts, val in metric.measurement_history]),
                json.dumps(metric.correlation_factors), metric.impact_on_outcomes,
                datetime.now().isoformat()
            ))
            conn.commit()

    def _row_to_outcome_record(self, row) -> OutcomeRecord:
        """Convert database row to OutcomeRecord"""
        return OutcomeRecord(
            outcome_id=row[0],
            decision_id=row[1],
            decision_context=json.loads(row[2]),
            decision_maker=row[3],
            outcome_type=OutcomeType(row[4]),
            timestamp=datetime.fromisoformat(row[5]),
            observed_results=json.loads(row[6]),
            quantitative_metrics=json.loads(row[7]) if row[7] else {},
            qualitative_assessment=row[8] or "",
            impact_level=ImpactLevel(row[9]),
            success_indicators=json.loads(row[10]) if row[10] else [],
            failure_indicators=json.loads(row[11]) if row[11] else [],
            unexpected_results=json.loads(row[12]) if row[12] else [],
            correlation_data=json.loads(row[13]) if row[13] else {},
            learning_insights=json.loads(row[14]) if row[14] else []
        )

    # Public API methods

    async def get_recent_outcomes(self, days: int = 7, decision_maker: Optional[str] = None) -> List[OutcomeRecord]:
        """Get recent outcome records"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return await self._get_outcomes_in_period(start_date, end_date, decision_maker)

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all tracked metrics"""
        summary = {
            "total_outcomes_tracked": len(self.active_outcomes),
            "performance_metrics": len(self.performance_metrics),
            "decision_patterns": len(self.decision_patterns),
            "recent_analyses": len(self.recent_analyses)
        }

        # Add performance metric summary
        if self.performance_metrics:
            improving_metrics = sum(1 for m in self.performance_metrics.values()
                                  if m.trend_direction == "improving")
            degrading_metrics = sum(1 for m in self.performance_metrics.values()
                                  if m.trend_direction == "degrading")

            summary["metric_trends"] = {
                "improving": improving_metrics,
                "stable": len(self.performance_metrics) - improving_metrics - degrading_metrics,
                "degrading": degrading_metrics
            }

        return summary

    async def get_decision_pattern_insights(self) -> List[Dict[str, Any]]:
        """Get insights from decision patterns"""
        insights = []

        for pattern in self.decision_patterns.values():
            if pattern.sample_size >= 5:  # Minimum sample size for reliable insights
                insight = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "success_probability": pattern.success_probability,
                    "average_impact": pattern.average_impact,
                    "sample_size": pattern.sample_size,
                    "key_characteristics": list(pattern.decision_characteristics.keys())[:3],
                    "recommendation": "reliable" if pattern.success_probability > 0.7 else "needs_improvement"
                }
                insights.append(insight)

        return sorted(insights, key=lambda x: x["success_probability"], reverse=True)

# Global instance
_outcome_tracker = None

def get_outcome_tracker() -> EchoOutcomeTracker:
    """Get global outcome tracker instance"""
    global _outcome_tracker
    if _outcome_tracker is None:
        _outcome_tracker = EchoOutcomeTracker()
    return _outcome_tracker

# Convenience functions
async def record_outcome(decision_id: str,
                        decision_context: Dict[str, Any],
                        decision_maker: str,
                        observed_results: Dict[str, Any],
                        outcome_type: OutcomeType = OutcomeType.IMMEDIATE,
                        delay_hours: float = 0) -> str:
    """Convenience function to record an outcome"""
    tracker = get_outcome_tracker()
    return await tracker.record_outcome(
        decision_id, decision_context, decision_maker,
        observed_results, outcome_type, delay_hours
    )

async def analyze_impact_over_period(start_date: datetime,
                                   end_date: datetime,
                                   decision_maker: Optional[str] = None) -> ImpactAnalysis:
    """Convenience function to analyze impact over a period"""
    tracker = get_outcome_tracker()
    return await tracker.analyze_impact_over_period(start_date, end_date, decision_maker)

async def get_performance_summary() -> Dict[str, Any]:
    """Convenience function to get performance summary"""
    tracker = get_outcome_tracker()
    return await tracker.get_performance_summary()

if __name__ == "__main__":
    # Example usage
    async def example():
        tracker = get_outcome_tracker()

        # Record an outcome
        outcome_id = await tracker.record_outcome(
            decision_id="decision_123",
            decision_context={
                "request_type": "performance_optimization",
                "complexity": 0.7,
                "expected_improvement": 0.2
            },
            decision_maker="echo",
            observed_results={
                "performance_improvement": 0.25,
                "error_rate": 0.02,
                "user_satisfaction": 0.85,
                "completion_time": 45.0,
                "resource_usage": 0.75
            },
            outcome_type=OutcomeType.SHORT_TERM
        )

        print(f"Outcome recorded: {outcome_id}")

        # Analyze impact over the last week
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        analysis = await tracker.analyze_impact_over_period(start_date, end_date, "echo")
        print("Impact Analysis:", json.dumps(asdict(analysis), indent=2, default=str))

        # Get performance summary
        summary = await tracker.get_performance_summary()
        print("Performance Summary:", json.dumps(summary, indent=2))

    asyncio.run(example())