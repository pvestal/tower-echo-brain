"""
Base Director Framework for Echo Brain Board of Directors System

This module provides the abstract base class for all specialized directors
in the Echo Brain system. Each director acts as a domain expert that can
evaluate tasks, provide reasoning, and suggest improvements.

Author: Echo Brain Board of Directors System
Created: 2025-09-16
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectorBase(ABC):
    """
    Abstract base class for all Echo Brain directors.

    Each director represents a specialized domain expert that can:
    - Evaluate tasks and provide domain-specific insights
    - Load and maintain domain knowledge bases
    - Generate detailed reasoning for decisions
    - Calculate confidence scores for recommendations
    - Suggest improvements and optimizations
    """

    def __init__(self, name: str, expertise: str, version: str = "1.0.0"):
        """
        Initialize a director with basic attributes.

        Args:
            name (str): The director's name/identifier
            expertise (str): Domain of expertise description
            version (str): Version of the director implementation
        """
        self.name = name
        self.expertise = expertise
        self.version = version
        self.created_at = datetime.now()
        self.evaluation_history = []

        # Load domain-specific knowledge
        self.knowledge_base = self.load_knowledge()

        # Initialize performance metrics
        self.metrics = {
            "evaluations_count": 0,
            "average_confidence": 0.0,
            "successful_recommendations": 0,
            "total_recommendations": 0
        }

        logger.info(f"Initialized {self.name} director with expertise in {self.expertise}")

    @abstractmethod
    def evaluate(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a task from this director's domain perspective.

        Args:
            task (Dict[str, Any]): Task information including type, description, requirements
            context (Dict[str, Any]): Additional context including user info, system state

        Returns:
            Dict[str, Any]: Evaluation result containing:
                - assessment: Overall assessment of the task
                - confidence: Confidence score (0.0 to 1.0)
                - reasoning: Detailed reasoning for the evaluation
                - recommendations: List of specific recommendations
                - risk_factors: Identified risks or concerns
                - estimated_effort: Effort estimation if applicable
        """
        raise NotImplementedError("Subclasses must implement the evaluate method")

    def load_knowledge(self) -> Dict[str, List[str]]:
        """
        Load domain-specific knowledge base.

        This method should be overridden by subclasses to load specialized
        knowledge relevant to their domain of expertise.

        Returns:
            Dict[str, List[str]]: Knowledge base structure containing:
                - best_practices: List of recommended approaches
                - anti_patterns: List of patterns to avoid
                - risk_factors: List of potential risks to consider
                - optimization_strategies: List of optimization techniques
        """
        return {
            "best_practices": [
                "Follow established patterns and conventions",
                "Maintain clear documentation and comments",
                "Implement proper error handling and logging",
                "Ensure scalability and maintainability"
            ],
            "anti_patterns": [
                "Avoid over-engineering solutions",
                "Don't ignore error handling",
                "Avoid tight coupling between components",
                "Don't sacrifice readability for cleverness"
            ],
            "risk_factors": [
                "Undefined requirements or scope creep",
                "Insufficient testing coverage",
                "Performance bottlenecks",
                "Security vulnerabilities"
            ],
            "optimization_strategies": [
                "Implement caching where appropriate",
                "Use efficient algorithms and data structures",
                "Minimize resource usage and dependencies",
                "Profile and monitor performance"
            ]
        }

    def generate_reasoning(self, assessment: str, factors: List[str],
                          context: Dict[str, Any]) -> str:
        """
        Generate detailed reasoning for an evaluation or decision.

        Args:
            assessment (str): The main assessment or conclusion
            factors (List[str]): Key factors that influenced the decision
            context (Dict[str, Any]): Additional context information

        Returns:
            str: Detailed reasoning explanation
        """
        reasoning_parts = [
            f"Assessment: {assessment}",
            "",
            "Key factors considered:"
        ]

        for i, factor in enumerate(factors, 1):
            reasoning_parts.append(f"{i}. {factor}")

        if context.get("constraints"):
            reasoning_parts.extend([
                "",
                "Constraints taken into account:",
                *[f"- {constraint}" for constraint in context.get("constraints", [])]
            ])

        if context.get("assumptions"):
            reasoning_parts.extend([
                "",
                "Operating assumptions:",
                *[f"- {assumption}" for assumption in context.get("assumptions", [])]
            ])

        reasoning_parts.extend([
            "",
            f"This evaluation is based on my expertise in {self.expertise} and ",
            f"consideration of {len(self.knowledge_base.get('best_practices', []))} best practices ",
            f"and {len(self.knowledge_base.get('risk_factors', []))} potential risk factors."
        ])

        return "\n".join(reasoning_parts)

    def calculate_confidence(self, factors: Dict[str, float]) -> float:
        """
        Calculate confidence score based on various factors.

        Args:
            factors (Dict[str, float]): Dictionary of factors and their weights
                Example: {
                    "requirements_clarity": 0.9,
                    "technical_feasibility": 0.8,
                    "resource_availability": 0.7,
                    "risk_level": 0.6
                }

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not factors:
            return 0.5  # Default neutral confidence

        # Apply weights and normalize
        weighted_sum = sum(score * (1.0 / len(factors)) for score in factors.values())

        # Ensure result is within valid range
        confidence = max(0.0, min(1.0, weighted_sum))

        logger.debug(f"{self.name} calculated confidence: {confidence:.3f} from factors: {factors}")

        return confidence

    def suggest_improvements(self, task: Dict[str, Any],
                           evaluation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest specific improvements based on evaluation results.

        Args:
            task (Dict[str, Any]): Original task information
            evaluation_result (Dict[str, Any]): Result from evaluate() method

        Returns:
            List[Dict[str, Any]]: List of improvement suggestions, each containing:
                - category: Type of improvement (e.g., "performance", "security", "maintainability")
                - priority: Priority level ("high", "medium", "low")
                - description: Detailed description of the improvement
                - implementation_notes: How to implement the improvement
                - expected_impact: Expected benefit of the improvement
        """
        improvements = []

        # Check against best practices
        for practice in self.knowledge_base.get("best_practices", []):
            if self._should_suggest_improvement(practice, task, evaluation_result):
                improvements.append({
                    "category": "best_practice",
                    "priority": "medium",
                    "description": f"Consider implementing: {practice}",
                    "implementation_notes": f"Review current approach against this best practice",
                    "expected_impact": "Improved code quality and maintainability"
                })

        # Check for anti-patterns
        for anti_pattern in self.knowledge_base.get("anti_patterns", []):
            if self._detect_anti_pattern(anti_pattern, task, evaluation_result):
                improvements.append({
                    "category": "anti_pattern_avoidance",
                    "priority": "high",
                    "description": f"Avoid: {anti_pattern}",
                    "implementation_notes": "Review and refactor affected areas",
                    "expected_impact": "Reduced technical debt and improved reliability"
                })

        # Add optimization suggestions based on confidence level
        confidence = evaluation_result.get("confidence", 0.5)
        if confidence < 0.7:
            improvements.append({
                "category": "confidence_improvement",
                "priority": "high",
                "description": "Low confidence detected - consider additional validation",
                "implementation_notes": "Gather more requirements, conduct feasibility analysis, or prototype key components",
                "expected_impact": "Increased success probability and reduced risk"
            })

        return improvements

    def _should_suggest_improvement(self, practice: str, task: Dict[str, Any],
                                  evaluation_result: Dict[str, Any]) -> bool:
        """
        Determine if a specific best practice should be suggested.

        This is a simple heuristic that can be overridden by subclasses.
        """
        # Suggest improvement if confidence is moderate and no specific mention of the practice
        confidence = evaluation_result.get("confidence", 0.5)
        task_description = task.get("description", "").lower()

        return confidence < 0.8 and practice.lower() not in task_description

    def _detect_anti_pattern(self, anti_pattern: str, task: Dict[str, Any],
                           evaluation_result: Dict[str, Any]) -> bool:
        """
        Detect if an anti-pattern might be present.

        This is a simple heuristic that can be overridden by subclasses.
        """
        # Simple keyword-based detection (subclasses should implement more sophisticated logic)
        task_description = task.get("description", "").lower()

        # Example: detect over-engineering
        if "over-engineering" in anti_pattern.lower():
            return "complex" in task_description and "simple" not in task_description

        return False

    def update_metrics(self, evaluation_result: Dict[str, Any], success: bool = None):
        """
        Update director performance metrics.

        Args:
            evaluation_result (Dict[str, Any]): Result from an evaluation
            success (bool): Whether the recommendation was successful (if known)
        """
        self.metrics["evaluations_count"] += 1

        confidence = evaluation_result.get("confidence", 0.5)
        current_avg = self.metrics["average_confidence"]
        count = self.metrics["evaluations_count"]

        # Update rolling average confidence
        self.metrics["average_confidence"] = (
            (current_avg * (count - 1) + confidence) / count
        )

        # Update recommendation tracking
        if "recommendations" in evaluation_result:
            self.metrics["total_recommendations"] += len(evaluation_result["recommendations"])

        if success is not None and success:
            self.metrics["successful_recommendations"] += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this director's performance metrics.

        Returns:
            Dict[str, Any]: Performance summary
        """
        success_rate = 0.0
        if self.metrics["total_recommendations"] > 0:
            success_rate = (
                self.metrics["successful_recommendations"] /
                self.metrics["total_recommendations"]
            )

        return {
            "director_name": self.name,
            "expertise": self.expertise,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "evaluations_performed": self.metrics["evaluations_count"],
            "average_confidence": round(self.metrics["average_confidence"], 3),
            "recommendation_success_rate": round(success_rate, 3),
            "knowledge_base_size": {
                category: len(items)
                for category, items in self.knowledge_base.items()
            }
        }

    def export_knowledge_base(self) -> str:
        """
        Export the knowledge base as a JSON string for persistence or sharing.

        Returns:
            str: JSON representation of the knowledge base
        """
        export_data = {
            "director_info": {
                "name": self.name,
                "expertise": self.expertise,
                "version": self.version,
                "exported_at": datetime.now().isoformat()
            },
            "knowledge_base": self.knowledge_base,
            "metrics": self.metrics
        }

        return json.dumps(export_data, indent=2)

    def __str__(self) -> str:
        """String representation of the director."""
        return f"Director({self.name}, expertise='{self.expertise}', version={self.version})"

    def __repr__(self) -> str:
        """Detailed string representation of the director."""
        return (
            f"DirectorBase(name='{self.name}', expertise='{self.expertise}', "
            f"version='{self.version}', evaluations={self.metrics['evaluations_count']})"
        )