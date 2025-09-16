"""
Director Registry for Echo Brain Board of Directors System

This module provides the central registry for managing all directors
in the Echo Brain system. It handles director registration, task routing,
consensus building, and performance tracking across all directors.

The registry acts as the coordination layer that:
- Maintains a roster of all available directors
- Routes tasks to appropriate directors based on expertise
- Builds consensus from multiple director evaluations
- Tracks director performance and reliability
- Provides unified interfaces for the Echo Brain system

Author: Echo Brain Board of Directors System
Created: 2025-09-16
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import threading

from .base_director import DirectorBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectorRegistry:
    """
    Central registry and coordination system for all Echo Brain directors.

    The registry manages the board of directors, routes tasks to appropriate
    experts, builds consensus from multiple opinions, and maintains performance
    metrics for the entire board.
    """

    def __init__(self, consensus_threshold: float = 0.6, max_directors_per_task: int = 5):
        """
        Initialize the director registry.

        Args:
            consensus_threshold (float): Minimum agreement level for consensus (0.0-1.0)
            max_directors_per_task (int): Maximum number of directors to consult per task
        """
        self.directors: Dict[str, DirectorBase] = {}
        self.expertise_map: Dict[str, List[str]] = defaultdict(list)
        self.consensus_threshold = consensus_threshold
        self.max_directors_per_task = max_directors_per_task

        # Performance tracking
        self.board_metrics = {
            "total_evaluations": 0,
            "consensus_achieved": 0,
            "average_response_time": 0.0,
            "director_agreement_matrix": defaultdict(lambda: defaultdict(int))
        }

        # Task history for learning
        self.task_history: List[Dict[str, Any]] = []
        self.evaluation_cache: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self._lock = threading.Lock()

        logger.info(f"Initialized DirectorRegistry with consensus_threshold={consensus_threshold}")

    def register_director(self, director: DirectorBase) -> bool:
        """
        Register a new director with the board.

        Args:
            director (DirectorBase): Director instance to register

        Returns:
            bool: True if registration successful, False if already exists
        """
        with self._lock:
            if director.name in self.directors:
                logger.warning(f"Director {director.name} already registered")
                return False

            self.directors[director.name] = director

            # Update expertise mapping
            expertise_keywords = self._extract_expertise_keywords(director.expertise)
            for keyword in expertise_keywords:
                self.expertise_map[keyword.lower()].append(director.name)

            logger.info(f"Registered director: {director.name} with expertise: {director.expertise}")
            return True

    def unregister_director(self, director_name: str) -> bool:
        """
        Remove a director from the board.

        Args:
            director_name (str): Name of director to remove

        Returns:
            bool: True if removal successful, False if not found
        """
        with self._lock:
            if director_name not in self.directors:
                logger.warning(f"Director {director_name} not found for removal")
                return False

            director = self.directors.pop(director_name)

            # Update expertise mapping
            expertise_keywords = self._extract_expertise_keywords(director.expertise)
            for keyword in expertise_keywords:
                if director_name in self.expertise_map[keyword.lower()]:
                    self.expertise_map[keyword.lower()].remove(director_name)
                    if not self.expertise_map[keyword.lower()]:
                        del self.expertise_map[keyword.lower()]

            logger.info(f"Unregistered director: {director_name}")
            return True

    def get_available_directors(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered directors.

        Returns:
            List[Dict[str, Any]]: List of director information
        """
        return [
            {
                "name": director.name,
                "expertise": director.expertise,
                "version": director.version,
                "performance": director.get_performance_summary()
            }
            for director in self.directors.values()
        ]

    def find_relevant_directors(self, task: Dict[str, Any],
                              context: Dict[str, Any] = None) -> List[str]:
        """
        Find directors most relevant to a given task.

        Args:
            task (Dict[str, Any]): Task information
            context (Dict[str, Any]): Additional context

        Returns:
            List[str]: List of relevant director names, ordered by relevance
        """
        task_description = task.get("description", "").lower()
        task_type = task.get("type", "").lower()
        task_keywords = set(task_description.split() + task_type.split())

        # Score directors based on keyword overlap
        director_scores = {}

        for keyword in task_keywords:
            for expertise_keyword, director_names in self.expertise_map.items():
                if keyword in expertise_keyword or expertise_keyword in keyword:
                    for director_name in director_names:
                        director_scores[director_name] = director_scores.get(director_name, 0) + 1

        # Add performance-based scoring
        for director_name, director in self.directors.items():
            performance = director.get_performance_summary()
            base_score = director_scores.get(director_name, 0)

            # Boost score based on success rate and confidence
            success_rate = performance.get("recommendation_success_rate", 0.5)
            avg_confidence = performance.get("average_confidence", 0.5)
            performance_boost = (success_rate + avg_confidence) / 2

            director_scores[director_name] = base_score + performance_boost

        # Sort by score and limit to max directors per task
        relevant_directors = sorted(
            director_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [name for name, score in relevant_directors[:self.max_directors_per_task]]

    def evaluate_task(self, task: Dict[str, Any],
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate a task using relevant directors and build consensus.

        Args:
            task (Dict[str, Any]): Task to evaluate
            context (Dict[str, Any]): Additional context

        Returns:
            Dict[str, Any]: Consolidated evaluation result with consensus
        """
        start_time = datetime.now()
        context = context or {}

        # Check cache first
        task_hash = self._generate_task_hash(task, context)
        if task_hash in self.evaluation_cache:
            cached_result = self.evaluation_cache[task_hash]
            logger.info(f"Returning cached evaluation for task: {task.get('description', 'Unknown')[:50]}...")
            return cached_result

        # Find relevant directors
        relevant_directors = self.find_relevant_directors(task, context)

        if not relevant_directors:
            logger.warning("No relevant directors found for task")
            return {
                "status": "error",
                "message": "No relevant directors available for this task",
                "task": task,
                "timestamp": datetime.now().isoformat()
            }

        # Collect evaluations from relevant directors
        evaluations = {}
        errors = {}

        for director_name in relevant_directors:
            try:
                director = self.directors[director_name]
                evaluation = director.evaluate(task, context)
                evaluations[director_name] = evaluation

                # Update director metrics
                director.update_metrics(evaluation)

            except Exception as e:
                logger.error(f"Error evaluating task with {director_name}: {str(e)}")
                errors[director_name] = str(e)

        # Build consensus from evaluations
        consensus_result = self._build_consensus(evaluations, task, context)

        # Add metadata
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()

        consensus_result.update({
            "evaluation_metadata": {
                "directors_consulted": list(evaluations.keys()),
                "directors_with_errors": list(errors.keys()),
                "response_time_seconds": response_time,
                "consensus_achieved": consensus_result.get("consensus_achieved", False),
                "task_hash": task_hash,
                "timestamp": end_time.isoformat()
            }
        })

        # Update board metrics
        self._update_board_metrics(evaluations, response_time, consensus_result)

        # Cache result
        self.evaluation_cache[task_hash] = consensus_result

        # Store in task history
        self.task_history.append({
            "task": task,
            "context": context,
            "result": consensus_result,
            "timestamp": end_time.isoformat()
        })

        return consensus_result

    def _build_consensus(self, evaluations: Dict[str, Dict[str, Any]],
                        task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build consensus from multiple director evaluations.

        Args:
            evaluations (Dict[str, Dict[str, Any]]): Evaluations from different directors
            task (Dict[str, Any]): Original task
            context (Dict[str, Any]): Context information

        Returns:
            Dict[str, Any]: Consensus evaluation result
        """
        if not evaluations:
            return {
                "status": "error",
                "message": "No evaluations available for consensus",
                "consensus_achieved": False
            }

        # Extract key metrics
        assessments = []
        confidences = []
        all_recommendations = []
        all_risk_factors = []
        reasoning_parts = []

        for director_name, evaluation in evaluations.items():
            assessments.append(evaluation.get("assessment", ""))
            confidences.append(evaluation.get("confidence", 0.5))
            all_recommendations.extend(evaluation.get("recommendations", []))
            all_risk_factors.extend(evaluation.get("risk_factors", []))

            reasoning_parts.append(f"\n{director_name}: {evaluation.get('reasoning', 'No reasoning provided')}")

        # Calculate consensus metrics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        confidence_variance = self._calculate_variance(confidences)
        consensus_achieved = confidence_variance < (1 - self.consensus_threshold)

        # Consolidate recommendations (remove duplicates, prioritize)
        consolidated_recommendations = self._consolidate_recommendations(all_recommendations)
        consolidated_risks = self._consolidate_risk_factors(all_risk_factors)

        # Generate consensus assessment
        consensus_assessment = self._generate_consensus_assessment(assessments, avg_confidence)

        return {
            "status": "success",
            "consensus_achieved": consensus_achieved,
            "assessment": consensus_assessment,
            "confidence": round(avg_confidence, 3),
            "confidence_variance": round(confidence_variance, 3),
            "recommendations": consolidated_recommendations,
            "risk_factors": consolidated_risks,
            "reasoning": f"Consensus built from {len(evaluations)} directors:" + "".join(reasoning_parts),
            "individual_evaluations": evaluations,
            "director_count": len(evaluations),
            "task": task
        }

    def _consolidate_recommendations(self, recommendations: List[str]) -> List[Dict[str, Any]]:
        """
        Consolidate and prioritize recommendations from multiple directors.

        Args:
            recommendations (List[str]): All recommendations from directors

        Returns:
            List[Dict[str, Any]]: Consolidated and prioritized recommendations
        """
        # Count recommendation frequency
        rec_counts = defaultdict(int)
        for rec in recommendations:
            rec_key = rec.lower().strip()
            rec_counts[rec_key] += 1

        # Sort by frequency and create structured recommendations
        consolidated = []
        for rec_text, count in sorted(rec_counts.items(), key=lambda x: x[1], reverse=True):
            priority = "high" if count >= len(self.directors) * 0.7 else "medium" if count >= 2 else "low"

            consolidated.append({
                "description": rec_text,
                "priority": priority,
                "director_agreement": count,
                "confidence": min(1.0, count / len(self.directors))
            })

        return consolidated[:10]  # Limit to top 10

    def _consolidate_risk_factors(self, risk_factors: List[str]) -> List[Dict[str, Any]]:
        """
        Consolidate risk factors from multiple directors.

        Args:
            risk_factors (List[str]): All risk factors from directors

        Returns:
            List[Dict[str, Any]]: Consolidated risk factors
        """
        # Similar to recommendations but focused on risks
        risk_counts = defaultdict(int)
        for risk in risk_factors:
            risk_key = risk.lower().strip()
            risk_counts[risk_key] += 1

        consolidated = []
        for risk_text, count in sorted(risk_counts.items(), key=lambda x: x[1], reverse=True):
            severity = "critical" if count >= len(self.directors) * 0.7 else "high" if count >= 2 else "medium"

            consolidated.append({
                "description": risk_text,
                "severity": severity,
                "director_agreement": count,
                "likelihood": min(1.0, count / len(self.directors))
            })

        return consolidated[:8]  # Limit to top 8

    def _generate_consensus_assessment(self, assessments: List[str],
                                     avg_confidence: float) -> str:
        """
        Generate a consensus assessment from multiple director assessments.

        Args:
            assessments (List[str]): Individual assessments
            avg_confidence (float): Average confidence score

        Returns:
            str: Consensus assessment
        """
        if avg_confidence >= 0.8:
            consensus_tone = "The board strongly agrees that"
        elif avg_confidence >= 0.6:
            consensus_tone = "The board generally agrees that"
        else:
            consensus_tone = "The board has mixed opinions, but leans toward"

        # Simple assessment classification
        positive_words = ["good", "excellent", "strong", "recommended", "feasible", "solid"]
        negative_words = ["poor", "weak", "risky", "problematic", "difficult", "challenging"]

        assessment_text = " ".join(assessments).lower()
        positive_score = sum(1 for word in positive_words if word in assessment_text)
        negative_score = sum(1 for word in negative_words if word in assessment_text)

        if positive_score > negative_score:
            sentiment = "this is a well-structured and feasible approach"
        elif negative_score > positive_score:
            sentiment = "this approach has significant challenges that need attention"
        else:
            sentiment = "this approach has both strengths and areas for improvement"

        return f"{consensus_tone} {sentiment}. Average confidence: {avg_confidence:.1%}"

    def _extract_expertise_keywords(self, expertise: str) -> List[str]:
        """Extract keywords from expertise description for matching."""
        # Simple keyword extraction - can be enhanced with NLP
        common_words = {"and", "or", "the", "in", "of", "to", "for", "with", "on", "at", "by"}
        words = expertise.lower().replace(",", " ").split()
        return [word for word in words if word not in common_words and len(word) > 2]

    def _generate_task_hash(self, task: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a hash for task caching."""
        task_str = json.dumps(task, sort_keys=True)
        context_str = json.dumps(context, sort_keys=True)
        combined = f"{task_str}:{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _update_board_metrics(self, evaluations: Dict[str, Dict[str, Any]],
                             response_time: float, consensus_result: Dict[str, Any]):
        """Update board-level performance metrics."""
        with self._lock:
            self.board_metrics["total_evaluations"] += 1

            if consensus_result.get("consensus_achieved", False):
                self.board_metrics["consensus_achieved"] += 1

            # Update rolling average response time
            current_avg = self.board_metrics["average_response_time"]
            count = self.board_metrics["total_evaluations"]

            self.board_metrics["average_response_time"] = (
                (current_avg * (count - 1) + response_time) / count
            )

            # Update director agreement matrix
            director_names = list(evaluations.keys())
            for i, dir1 in enumerate(director_names):
                for dir2 in director_names[i+1:]:
                    conf1 = evaluations[dir1].get("confidence", 0.5)
                    conf2 = evaluations[dir2].get("confidence", 0.5)

                    # Consider agreement if confidence scores are within 0.2
                    if abs(conf1 - conf2) <= 0.2:
                        self.board_metrics["director_agreement_matrix"][dir1][dir2] += 1
                        self.board_metrics["director_agreement_matrix"][dir2][dir1] += 1

    def get_board_performance(self) -> Dict[str, Any]:
        """
        Get overall board performance metrics.

        Returns:
            Dict[str, Any]: Board performance summary
        """
        consensus_rate = 0.0
        if self.board_metrics["total_evaluations"] > 0:
            consensus_rate = (
                self.board_metrics["consensus_achieved"] /
                self.board_metrics["total_evaluations"]
            )

        return {
            "board_summary": {
                "total_directors": len(self.directors),
                "total_evaluations": self.board_metrics["total_evaluations"],
                "consensus_rate": round(consensus_rate, 3),
                "average_response_time": round(self.board_metrics["average_response_time"], 3),
                "expertise_areas": len(self.expertise_map)
            },
            "individual_directors": [
                director.get_performance_summary()
                for director in self.directors.values()
            ],
            "director_agreement_matrix": dict(self.board_metrics["director_agreement_matrix"])
        }

    def clear_cache(self) -> int:
        """
        Clear the evaluation cache.

        Returns:
            int: Number of cached evaluations cleared
        """
        with self._lock:
            count = len(self.evaluation_cache)
            self.evaluation_cache.clear()
            logger.info(f"Cleared {count} cached evaluations")
            return count

    def prune_task_history(self, days_to_keep: int = 30) -> int:
        """
        Remove old task history entries.

        Args:
            days_to_keep (int): Number of days of history to retain

        Returns:
            int: Number of entries removed
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with self._lock:
            initial_count = len(self.task_history)

            self.task_history = [
                entry for entry in self.task_history
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
            ]

            removed_count = initial_count - len(self.task_history)
            logger.info(f"Pruned {removed_count} task history entries older than {days_to_keep} days")
            return removed_count

    def __len__(self) -> int:
        """Return the number of registered directors."""
        return len(self.directors)

    def __str__(self) -> str:
        """String representation of the registry."""
        return f"DirectorRegistry({len(self.directors)} directors, {len(self.expertise_map)} expertise areas)"

    def __repr__(self) -> str:
        """Detailed string representation of the registry."""
        return (
            f"DirectorRegistry(directors={len(self.directors)}, "
            f"evaluations={self.board_metrics['total_evaluations']}, "
            f"consensus_rate={self.board_metrics['consensus_achieved'] / max(1, self.board_metrics['total_evaluations']):.2f})"
        )