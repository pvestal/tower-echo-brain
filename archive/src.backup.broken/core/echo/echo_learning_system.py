#!/usr/bin/env python3
"""
Echo Learning System - Core Intelligence and Feedback Loop
========================================================

Echo's central learning intelligence that:
- Learns from every decision outcome
- Manages meta-cognition and self-improvement
- Orchestrates all other learning components
- Tracks patterns in successful/failed decisions
- Develops Echo's decision-making capabilities over time

This is Echo's "consciousness" - the meta-layer that observes and improves
Echo's own thinking processes.
"""

import asyncio
import logging
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import hashlib
from pathlib import Path

# Configuration
LEARNING_DB_PATH = "/opt/tower-echo-brain/data/echo_learning.db"
CONFIDENCE_THRESHOLD = 0.7
LEARNING_RATE = 0.1
MIN_SAMPLES_FOR_PATTERN = 5

class DecisionOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"

class LearningDomain(Enum):
    BOARD_MANAGEMENT = "board_management"
    TASK_DECOMPOSITION = "task_decomposition"
    SELF_DIAGNOSIS = "self_diagnosis"
    GENERAL_REASONING = "general_reasoning"
    USER_INTERACTION = "user_interaction"
    SYSTEM_INTEGRATION = "system_integration"

@dataclass
class DecisionRecord:
    """Complete record of a decision and its outcome"""
    decision_id: str
    timestamp: datetime
    domain: LearningDomain
    context: Dict[str, Any]
    decision_factors: Dict[str, float]
    decision_made: str
    confidence: float
    outcome: DecisionOutcome
    outcome_metrics: Dict[str, float]
    learning_applied: List[str]
    meta_data: Dict[str, Any]

@dataclass
class LearningPattern:
    """Discovered pattern from decision analysis"""
    pattern_id: str
    domain: LearningDomain
    pattern_type: str
    conditions: Dict[str, Any]
    success_rate: float
    confidence: float
    sample_count: int
    last_validated: datetime
    meta_insights: Dict[str, Any]

class EchoLearningSystem:
    """
    Echo's Core Learning Intelligence

    This system implements Echo's meta-cognitive capabilities:
    - Pattern recognition across all decision domains
    - Self-improvement through outcome analysis
    - Dynamic strategy adjustment
    - Knowledge synthesis and insight generation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = LEARNING_DB_PATH
        self._ensure_data_directory()
        self._init_database()

        # Learning state
        self.active_patterns: Dict[str, LearningPattern] = {}
        self.decision_history: List[DecisionRecord] = []
        self.learning_metrics: Dict[str, float] = {}

        # Meta-learning parameters
        self.adaptation_rate = LEARNING_RATE
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        self.logger.info("Echo Learning System initialized")

    def _ensure_data_directory(self):
        """Ensure learning data directory exists"""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize learning database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_records (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    context TEXT NOT NULL,
                    decision_factors TEXT NOT NULL,
                    decision_made TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    outcome TEXT,
                    outcome_metrics TEXT,
                    learning_applied TEXT,
                    meta_data TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    confidence REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    last_validated TEXT NOT NULL,
                    meta_insights TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    metric_name TEXT PRIMARY KEY,
                    metric_value REAL NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)

            conn.commit()

    async def record_decision(self,
                            domain: LearningDomain,
                            context: Dict[str, Any],
                            decision_factors: Dict[str, float],
                            decision_made: str,
                            confidence: float,
                            meta_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a decision for later outcome tracking

        Args:
            domain: Learning domain for this decision
            context: Contextual information about the decision
            decision_factors: Weighted factors that influenced the decision
            decision_made: The actual decision made
            confidence: Confidence level in the decision
            meta_data: Additional metadata

        Returns:
            decision_id: Unique identifier for tracking this decision
        """
        decision_id = self._generate_decision_id(domain, context, decision_made)

        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=datetime.now(),
            domain=domain,
            context=context,
            decision_factors=decision_factors,
            decision_made=decision_made,
            confidence=confidence,
            outcome=DecisionOutcome.UNKNOWN,
            outcome_metrics={},
            learning_applied=self._get_applied_patterns(domain, context),
            meta_data=meta_data or {}
        )

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO decision_records
                (decision_id, timestamp, domain, context, decision_factors,
                 decision_made, confidence, outcome, outcome_metrics,
                 learning_applied, meta_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.decision_id,
                record.timestamp.isoformat(),
                record.domain.value,
                json.dumps(record.context),
                json.dumps(record.decision_factors),
                record.decision_made,
                record.confidence,
                record.outcome.value,
                json.dumps(record.outcome_metrics),
                json.dumps(record.learning_applied),
                json.dumps(record.meta_data)
            ))
            conn.commit()

        self.decision_history.append(record)
        self.logger.info(f"Recorded decision {decision_id} in domain {domain.value}")

        return decision_id

    async def record_outcome(self,
                           decision_id: str,
                           outcome: DecisionOutcome,
                           outcome_metrics: Dict[str, float],
                           feedback: Optional[str] = None):
        """
        Record the outcome of a previous decision and trigger learning

        Args:
            decision_id: ID of the decision to update
            outcome: The observed outcome
            outcome_metrics: Quantifiable metrics about the outcome
            feedback: Optional human feedback
        """
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE decision_records
                SET outcome = ?, outcome_metrics = ?
                WHERE decision_id = ?
            """, (outcome.value, json.dumps(outcome_metrics), decision_id))
            conn.commit()

        # Update in-memory record
        for record in self.decision_history:
            if record.decision_id == decision_id:
                record.outcome = outcome
                record.outcome_metrics = outcome_metrics
                break

        self.logger.info(f"Recorded outcome {outcome.value} for decision {decision_id}")

        # Trigger learning from this outcome
        await self._learn_from_outcome(decision_id, outcome, outcome_metrics, feedback)

    async def _learn_from_outcome(self,
                                decision_id: str,
                                outcome: DecisionOutcome,
                                outcome_metrics: Dict[str, float],
                                feedback: Optional[str] = None):
        """
        Learn from a decision outcome and update patterns
        """
        record = self._get_decision_record(decision_id)
        if not record:
            self.logger.error(f"Decision record {decision_id} not found")
            return

        # Analyze this outcome in context of similar decisions
        similar_decisions = self._find_similar_decisions(record)

        # Update or create patterns
        await self._update_patterns(record, similar_decisions)

        # Update meta-learning metrics
        await self._update_learning_metrics(record)

        # If this was a failure, trigger deeper analysis
        if outcome == DecisionOutcome.FAILURE:
            await self._analyze_failure(record, feedback)

        self.logger.info(f"Learning completed for decision {decision_id}")

    def _get_decision_record(self, decision_id: str) -> Optional[DecisionRecord]:
        """Retrieve a decision record by ID"""
        for record in self.decision_history:
            if record.decision_id == decision_id:
                return record

        # Try to load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM decision_records WHERE decision_id = ?
            """, (decision_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_decision_record(row)

        return None

    def _find_similar_decisions(self, record: DecisionRecord) -> List[DecisionRecord]:
        """Find decisions similar to the given record"""
        similar = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM decision_records
                WHERE domain = ? AND outcome != 'unknown'
                ORDER BY timestamp DESC LIMIT 100
            """, (record.domain.value,))

            for row in cursor.fetchall():
                other_record = self._row_to_decision_record(row)
                if self._calculate_similarity(record, other_record) > 0.5:
                    similar.append(other_record)

        return similar

    def _calculate_similarity(self, record1: DecisionRecord, record2: DecisionRecord) -> float:
        """Calculate similarity between two decision records"""
        if record1.domain != record2.domain:
            return 0.0

        # Compare decision factors
        factors1 = record1.decision_factors
        factors2 = record2.decision_factors

        common_factors = set(factors1.keys()) & set(factors2.keys())
        if not common_factors:
            return 0.0

        similarity_sum = 0.0
        for factor in common_factors:
            diff = abs(factors1[factor] - factors2[factor])
            similarity_sum += 1.0 - diff  # Assuming factors are normalized 0-1

        return similarity_sum / len(common_factors)

    async def _update_patterns(self, record: DecisionRecord, similar_decisions: List[DecisionRecord]):
        """Update learning patterns based on new outcome"""
        domain = record.domain
        outcome_success = record.outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL]

        # Group similar decisions by their characteristics
        pattern_groups = self._group_decisions_by_patterns(similar_decisions + [record])

        for pattern_key, decisions in pattern_groups.items():
            if len(decisions) < MIN_SAMPLES_FOR_PATTERN:
                continue

            success_count = sum(1 for d in decisions
                              if d.outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL])
            success_rate = success_count / len(decisions)

            # Create or update pattern
            pattern_id = f"{domain.value}_{pattern_key}"
            pattern = LearningPattern(
                pattern_id=pattern_id,
                domain=domain,
                pattern_type="decision_outcome",
                conditions=self._extract_pattern_conditions(decisions),
                success_rate=success_rate,
                confidence=min(1.0, len(decisions) / 20.0),  # Confidence grows with samples
                sample_count=len(decisions),
                last_validated=datetime.now(),
                meta_insights=self._generate_meta_insights(decisions)
            )

            self.active_patterns[pattern_id] = pattern
            await self._store_pattern(pattern)

    def _group_decisions_by_patterns(self, decisions: List[DecisionRecord]) -> Dict[str, List[DecisionRecord]]:
        """Group decisions by their pattern characteristics"""
        groups = {}

        for decision in decisions:
            # Create pattern key based on decision characteristics
            key_factors = []

            # Include top decision factors
            sorted_factors = sorted(decision.decision_factors.items(),
                                  key=lambda x: x[1], reverse=True)[:3]
            for factor, weight in sorted_factors:
                if weight > 0.5:
                    key_factors.append(f"{factor}:high")
                elif weight > 0.2:
                    key_factors.append(f"{factor}:med")

            pattern_key = "_".join(sorted(key_factors))
            if pattern_key not in groups:
                groups[pattern_key] = []
            groups[pattern_key].append(decision)

        return groups

    def _extract_pattern_conditions(self, decisions: List[DecisionRecord]) -> Dict[str, Any]:
        """Extract common conditions from a group of decisions"""
        if not decisions:
            return {}

        # Analyze decision factors
        all_factors = {}
        for decision in decisions:
            for factor, weight in decision.decision_factors.items():
                if factor not in all_factors:
                    all_factors[factor] = []
                all_factors[factor].append(weight)

        # Calculate average and consistency for each factor
        conditions = {}
        for factor, weights in all_factors.items():
            if len(weights) > len(decisions) * 0.5:  # Factor appears in majority of decisions
                avg_weight = statistics.mean(weights)
                consistency = 1.0 - statistics.stdev(weights) if len(weights) > 1 else 1.0
                conditions[factor] = {
                    "average_weight": avg_weight,
                    "consistency": consistency,
                    "frequency": len(weights) / len(decisions)
                }

        return conditions

    def _generate_meta_insights(self, decisions: List[DecisionRecord]) -> Dict[str, Any]:
        """Generate meta-insights from a pattern group"""
        insights = {}

        if not decisions:
            return insights

        # Confidence patterns
        confidences = [d.confidence for d in decisions]
        insights["confidence_analysis"] = {
            "avg_confidence": statistics.mean(confidences),
            "confidence_variance": statistics.variance(confidences) if len(confidences) > 1 else 0,
            "high_confidence_success_rate": self._calculate_conditional_success_rate(
                decisions, lambda d: d.confidence > 0.8
            )
        }

        # Temporal patterns
        timestamps = [d.timestamp for d in decisions]
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds()
                         for i in range(1, len(timestamps))]
            insights["temporal_patterns"] = {
                "avg_time_between_decisions": statistics.mean(time_diffs),
                "decision_frequency": len(decisions) / max(1, (max(timestamps) - min(timestamps)).days)
            }

        return insights

    def _calculate_conditional_success_rate(self, decisions: List[DecisionRecord], condition) -> float:
        """Calculate success rate for decisions meeting a condition"""
        filtered = [d for d in decisions if condition(d)]
        if not filtered:
            return 0.0

        successes = sum(1 for d in filtered
                       if d.outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL])
        return successes / len(filtered)

    async def _analyze_failure(self, record: DecisionRecord, feedback: Optional[str] = None):
        """Analyze a failure to identify improvement opportunities"""
        self.logger.warning(f"Analyzing failure in decision {record.decision_id}")

        # Find what went wrong
        failure_analysis = {
            "decision_factors": record.decision_factors,
            "confidence": record.confidence,
            "outcome_metrics": record.outcome_metrics,
            "context": record.context,
            "human_feedback": feedback
        }

        # Look for patterns in similar failures
        similar_failures = [d for d in self._find_similar_decisions(record)
                           if d.outcome == DecisionOutcome.FAILURE]

        if len(similar_failures) >= 3:
            # We have a pattern of failures - need intervention
            await self._trigger_intervention(record.domain, similar_failures, failure_analysis)

        # Store failure analysis
        await self._store_failure_analysis(record.decision_id, failure_analysis)

    async def _trigger_intervention(self, domain: LearningDomain,
                                  failures: List[DecisionRecord],
                                  analysis: Dict[str, Any]):
        """Trigger intervention for repeated failures in a domain"""
        self.logger.error(f"Intervention needed in domain {domain.value} - {len(failures)} recent failures")

        # This would trigger Echo's self-diagnosis system
        # For now, log the intervention need
        intervention_data = {
            "domain": domain.value,
            "failure_count": len(failures),
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

        # In a real implementation, this would notify other Echo systems
        self.logger.critical(f"INTERVENTION REQUIRED: {json.dumps(intervention_data, indent=2)}")

    async def get_recommendations(self, domain: LearningDomain,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get learning-based recommendations for a decision context

        Args:
            domain: The domain of the decision
            context: Current decision context

        Returns:
            Recommendations including patterns, confidence adjustments, and warnings
        """
        recommendations = {
            "applicable_patterns": [],
            "confidence_adjustment": 0.0,
            "warnings": [],
            "suggestions": [],
            "meta_advice": ""
        }

        # Find applicable patterns
        for pattern in self.active_patterns.values():
            if pattern.domain == domain and self._pattern_applies(pattern, context):
                recommendations["applicable_patterns"].append({
                    "pattern_id": pattern.pattern_id,
                    "success_rate": pattern.success_rate,
                    "confidence": pattern.confidence,
                    "conditions": pattern.conditions,
                    "insights": pattern.meta_insights
                })

        # Generate confidence adjustment
        if recommendations["applicable_patterns"]:
            avg_success_rate = statistics.mean([p["success_rate"]
                                              for p in recommendations["applicable_patterns"]])
            recommendations["confidence_adjustment"] = (avg_success_rate - 0.5) * 0.2

        # Generate warnings and suggestions
        await self._generate_contextual_advice(domain, context, recommendations)

        return recommendations

    def _pattern_applies(self, pattern: LearningPattern, context: Dict[str, Any]) -> bool:
        """Check if a pattern applies to the current context"""
        # This is a simplified implementation
        # In practice, this would involve more sophisticated pattern matching
        return pattern.confidence > self.confidence_threshold

    async def _generate_contextual_advice(self, domain: LearningDomain,
                                        context: Dict[str, Any],
                                        recommendations: Dict[str, Any]):
        """Generate contextual advice based on learning history"""
        # Recent failure analysis
        recent_failures = await self._get_recent_failures(domain)
        if len(recent_failures) > 2:
            recommendations["warnings"].append(
                f"High failure rate in {domain.value} recently - consider conservative approach"
            )

        # Pattern-based suggestions
        if recommendations["applicable_patterns"]:
            best_pattern = max(recommendations["applicable_patterns"],
                             key=lambda p: p["success_rate"] * p["confidence"])
            recommendations["suggestions"].append(
                f"Consider pattern {best_pattern['pattern_id']} with {best_pattern['success_rate']:.1%} success rate"
            )

        # Meta-advice based on overall learning
        overall_performance = await self._get_domain_performance(domain)
        if overall_performance < 0.6:
            recommendations["meta_advice"] = f"Performance in {domain.value} below optimal - recommend escalation or consultation"
        elif overall_performance > 0.85:
            recommendations["meta_advice"] = f"Strong performance in {domain.value} - high confidence recommended"

    async def _get_recent_failures(self, domain: LearningDomain, days: int = 7) -> List[DecisionRecord]:
        """Get recent failures in a domain"""
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM decision_records
                WHERE domain = ? AND outcome = 'failure' AND timestamp > ?
                ORDER BY timestamp DESC
            """, (domain.value, cutoff.isoformat()))

            return [self._row_to_decision_record(row) for row in cursor.fetchall()]

    async def _get_domain_performance(self, domain: LearningDomain, days: int = 30) -> float:
        """Get overall performance metrics for a domain"""
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT outcome FROM decision_records
                WHERE domain = ? AND outcome != 'unknown' AND timestamp > ?
            """, (domain.value, cutoff.isoformat()))

            outcomes = [row[0] for row in cursor.fetchall()]
            if not outcomes:
                return 0.5  # Neutral performance if no data

            successes = sum(1 for outcome in outcomes
                           if outcome in ['success', 'partial'])
            return successes / len(outcomes)

    def _generate_decision_id(self, domain: LearningDomain,
                            context: Dict[str, Any],
                            decision: str) -> str:
        """Generate unique decision ID"""
        data = f"{domain.value}_{json.dumps(context, sort_keys=True)}_{decision}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _get_applied_patterns(self, domain: LearningDomain, context: Dict[str, Any]) -> List[str]:
        """Get list of patterns that apply to this context"""
        applied = []
        for pattern in self.active_patterns.values():
            if pattern.domain == domain and self._pattern_applies(pattern, context):
                applied.append(pattern.pattern_id)
        return applied

    async def _store_pattern(self, pattern: LearningPattern):
        """Store pattern in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO learning_patterns
                (pattern_id, domain, pattern_type, conditions, success_rate,
                 confidence, sample_count, last_validated, meta_insights)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.domain.value,
                pattern.pattern_type,
                json.dumps(pattern.conditions),
                pattern.success_rate,
                pattern.confidence,
                pattern.sample_count,
                pattern.last_validated.isoformat(),
                json.dumps(pattern.meta_insights)
            ))
            conn.commit()

    async def _store_failure_analysis(self, decision_id: str, analysis: Dict[str, Any]):
        """Store failure analysis for future reference"""
        # This could be stored in a separate table or as part of meta_data
        pass

    async def _update_learning_metrics(self, record: DecisionRecord):
        """Update overall learning metrics"""
        domain = record.domain.value
        outcome_value = 1.0 if record.outcome == DecisionOutcome.SUCCESS else 0.0

        # Update rolling performance metrics
        metric_name = f"performance_{domain}"
        current_metric = await self._get_metric(metric_name)

        # Exponential moving average
        alpha = 0.1  # Learning rate
        new_metric = alpha * outcome_value + (1 - alpha) * current_metric

        await self._store_metric(metric_name, new_metric)

    async def _get_metric(self, metric_name: str) -> float:
        """Get a learning metric value"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_value FROM learning_metrics WHERE metric_name = ?
            """, (metric_name,))
            row = cursor.fetchone()
            return row[0] if row else 0.5  # Default to neutral

    async def _store_metric(self, metric_name: str, value: float):
        """Store a learning metric"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO learning_metrics
                (metric_name, metric_value, last_updated)
                VALUES (?, ?, ?)
            """, (metric_name, value, datetime.now().isoformat()))
            conn.commit()

    def _row_to_decision_record(self, row) -> DecisionRecord:
        """Convert database row to DecisionRecord"""
        return DecisionRecord(
            decision_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            domain=LearningDomain(row[2]),
            context=json.loads(row[3]),
            decision_factors=json.loads(row[4]),
            decision_made=row[5],
            confidence=row[6],
            outcome=DecisionOutcome(row[7]) if row[7] else DecisionOutcome.UNKNOWN,
            outcome_metrics=json.loads(row[8]) if row[8] else {},
            learning_applied=json.loads(row[9]) if row[9] else [],
            meta_data=json.loads(row[10]) if row[10] else {}
        )

    async def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning system status"""
        status = {
            "system_status": "operational",
            "active_patterns": len(self.active_patterns),
            "decision_history_size": len(self.decision_history),
            "domain_performance": {},
            "recent_learning_activity": [],
            "system_insights": []
        }

        # Get performance by domain
        for domain in LearningDomain:
            performance = await self._get_domain_performance(domain)
            status["domain_performance"][domain.value] = performance

        # Recent activity
        recent_decisions = await self._get_recent_decisions(limit=10)
        status["recent_learning_activity"] = [
            {
                "decision_id": d.decision_id,
                "domain": d.domain.value,
                "outcome": d.outcome.value,
                "confidence": d.confidence,
                "timestamp": d.timestamp.isoformat()
            }
            for d in recent_decisions
        ]

        # System insights
        total_decisions = len(self.decision_history)
        if total_decisions > 0:
            avg_confidence = statistics.mean([d.confidence for d in self.decision_history])
            success_rate = len([d for d in self.decision_history
                              if d.outcome == DecisionOutcome.SUCCESS]) / total_decisions

            status["system_insights"] = [
                f"Overall success rate: {success_rate:.1%}",
                f"Average confidence: {avg_confidence:.2f}",
                f"Active learning patterns: {len(self.active_patterns)}",
                f"Total decisions analyzed: {total_decisions}"
            ]

        return status

    async def _get_recent_decisions(self, limit: int = 10) -> List[DecisionRecord]:
        """Get recent decisions from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM decision_records
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))

            return [self._row_to_decision_record(row) for row in cursor.fetchall()]

# Global instance
_learning_system = None

def get_learning_system() -> EchoLearningSystem:
    """Get global learning system instance"""
    global _learning_system
    if _learning_system is None:
        _learning_system = EchoLearningSystem()
    return _learning_system

# Convenience functions for other systems
async def record_decision(domain: LearningDomain,
                         context: Dict[str, Any],
                         decision_factors: Dict[str, float],
                         decision_made: str,
                         confidence: float,
                         meta_data: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to record a decision"""
    system = get_learning_system()
    return await system.record_decision(domain, context, decision_factors,
                                       decision_made, confidence, meta_data)

async def record_outcome(decision_id: str,
                        outcome: DecisionOutcome,
                        outcome_metrics: Dict[str, float],
                        feedback: Optional[str] = None):
    """Convenience function to record an outcome"""
    system = get_learning_system()
    await system.record_outcome(decision_id, outcome, outcome_metrics, feedback)

async def get_recommendations(domain: LearningDomain,
                            context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to get recommendations"""
    system = get_learning_system()
    return await system.get_recommendations(domain, context)

if __name__ == "__main__":
    # Example usage
    async def example():
        system = get_learning_system()

        # Record a decision
        decision_id = await system.record_decision(
            domain=LearningDomain.BOARD_MANAGEMENT,
            context={"user_request": "analyze system performance", "complexity": "medium"},
            decision_factors={"confidence": 0.8, "expertise_match": 0.9, "resource_availability": 0.7},
            decision_made="escalate_to_performance_director",
            confidence=0.85
        )

        # Later, record the outcome
        await system.record_outcome(
            decision_id=decision_id,
            outcome=DecisionOutcome.SUCCESS,
            outcome_metrics={"task_completion_time": 30.5, "user_satisfaction": 0.95},
            feedback="Task completed efficiently"
        )

        # Get recommendations for similar future decisions
        recommendations = await system.get_recommendations(
            domain=LearningDomain.BOARD_MANAGEMENT,
            context={"user_request": "analyze system performance", "complexity": "high"}
        )

        print("Recommendations:", json.dumps(recommendations, indent=2))

        # Get system status
        status = await system.get_learning_status()
        print("Learning Status:", json.dumps(status, indent=2))

    asyncio.run(example())