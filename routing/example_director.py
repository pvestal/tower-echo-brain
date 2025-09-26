"""
Example Director Implementation

This module provides a simple example of how to implement a specialized director
that inherits from DirectorBase. This example shows best practices for:
- Implementing the abstract evaluate method
- Overriding load_knowledge for domain-specific expertise
- Using the base class helper methods effectively

This can be used as a template for creating new specialized directors.

Author: Echo Brain Board of Directors System
Created: 2025-09-16
"""

import logging
import sys
import os
from typing import Dict, List, Any

# Handle imports for both module usage and standalone execution
try:
    from .base_director import DirectorBase
except ImportError:
    # Add parent directory to path for standalone execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from routing.base_director import DirectorBase

logger = logging.getLogger(__name__)


class ExampleDirector(DirectorBase):
    """
    Example director implementation for demonstrating the framework.

    This director specializes in code quality and best practices evaluation.
    It can be used as a template for creating other specialized directors.
    """

    def __init__(self):
        """Initialize the example director."""
        super().__init__(
            name="ExampleDirector",
            expertise="Code quality, best practices, and software architecture",
            version="1.0.0"
        )

    def evaluate(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a task from the code quality perspective.

        Args:
            task (Dict[str, Any]): Task information
            context (Dict[str, Any]): Additional context

        Returns:
            Dict[str, Any]: Evaluation result
        """
        try:
            task_type = task.get("type", "unknown")
            task_description = task.get("description", "")

            # Analyze task characteristics
            analysis_factors = self._analyze_task(task, context)

            # Calculate confidence based on analysis
            confidence = self.calculate_confidence(analysis_factors)

            # Generate assessment
            assessment = self._generate_assessment(task, analysis_factors, confidence)

            # Create recommendations
            recommendations = self._create_recommendations(task, analysis_factors)

            # Identify risks
            risk_factors = self._identify_risks(task, analysis_factors)

            # Generate reasoning
            reasoning_factors = [
                f"Task type '{task_type}' matches my expertise in code quality",
                f"Analysis confidence: {confidence:.2f}",
                f"Identified {len(recommendations)} recommendations",
                f"Found {len(risk_factors)} potential risk factors"
            ]

            reasoning = self.generate_reasoning(
                assessment, reasoning_factors, context
            )

            # Estimate effort (example logic)
            estimated_effort = self._estimate_effort(task, analysis_factors)

            return {
                "assessment": assessment,
                "confidence": confidence,
                "reasoning": reasoning,
                "recommendations": recommendations,
                "risk_factors": risk_factors,
                "estimated_effort": estimated_effort,
                "analysis_factors": analysis_factors
            }

        except Exception as e:
            logger.error(f"Error evaluating task with {self.name}: {str(e)}")
            return {
                "assessment": "Error occurred during evaluation",
                "confidence": 0.0,
                "reasoning": f"Evaluation failed due to: {str(e)}",
                "recommendations": [],
                "risk_factors": ["Evaluation system error"],
                "estimated_effort": "unknown"
            }

    def load_knowledge(self) -> Dict[str, List[str]]:
        """
        Load code quality specific knowledge base.

        Returns:
            Dict[str, List[str]]: Domain-specific knowledge
        """
        return {
            "best_practices": [
                "Write clean, readable, and maintainable code",
                "Follow consistent naming conventions",
                "Implement comprehensive error handling",
                "Write unit tests for all critical functionality",
                "Use version control effectively with meaningful commits",
                "Document code thoroughly with comments and docstrings",
                "Follow SOLID principles in object-oriented design",
                "Implement proper logging and monitoring",
                "Use configuration management for environment-specific settings",
                "Perform regular code reviews and refactoring"
            ],
            "anti_patterns": [
                "Writing overly complex or clever code",
                "Ignoring error conditions and edge cases",
                "Creating tight coupling between components",
                "Duplicating code instead of creating reusable functions",
                "Using magic numbers and hardcoded values",
                "Skipping validation of input data",
                "Creating overly deep inheritance hierarchies",
                "Mixing business logic with presentation logic",
                "Using global variables excessively",
                "Writing functions that do too many things"
            ],
            "risk_factors": [
                "Insufficient or missing unit test coverage",
                "Lack of input validation and sanitization",
                "Poor error handling and recovery mechanisms",
                "Unclear or missing requirements specification",
                "Tight coupling between system components",
                "Inadequate documentation and code comments",
                "Missing or insufficient logging and monitoring",
                "Hardcoded configuration values",
                "Security vulnerabilities in code implementation",
                "Performance bottlenecks and scalability issues"
            ],
            "optimization_strategies": [
                "Implement efficient algorithms and data structures",
                "Use caching strategies for frequently accessed data",
                "Optimize database queries and indexing",
                "Minimize network calls and data transfer",
                "Profile code to identify performance bottlenecks",
                "Use lazy loading for expensive operations",
                "Implement connection pooling for database access",
                "Compress data and responses where appropriate",
                "Use asynchronous processing for long-running tasks",
                "Implement proper memory management and garbage collection"
            ]
        }

    def _analyze_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze task characteristics and return scoring factors.

        Args:
            task (Dict[str, Any]): Task information
            context (Dict[str, Any]): Additional context

        Returns:
            Dict[str, float]: Analysis factors with scores (0.0-1.0)
        """
        task_description = task.get("description", "").lower()
        task_type = task.get("type", "").lower()

        factors = {}

        # Requirements clarity
        clarity_keywords = ["clear", "specific", "detailed", "documented"]
        unclear_keywords = ["maybe", "might", "unclear", "not sure"]

        clarity_score = 0.5  # Default
        if any(keyword in task_description for keyword in clarity_keywords):
            clarity_score += 0.3
        if any(keyword in task_description for keyword in unclear_keywords):
            clarity_score -= 0.3

        factors["requirements_clarity"] = max(0.0, min(1.0, clarity_score))

        # Technical complexity
        complexity_keywords = ["complex", "advanced", "sophisticated", "enterprise"]
        simple_keywords = ["simple", "basic", "straightforward", "easy"]

        complexity_score = 0.5  # Default
        if any(keyword in task_description for keyword in complexity_keywords):
            complexity_score += 0.4
        if any(keyword in task_description for keyword in simple_keywords):
            complexity_score -= 0.2

        factors["technical_complexity"] = max(0.0, min(1.0, complexity_score))

        # Code quality relevance
        code_keywords = ["code", "implementation", "development", "programming", "software"]
        relevance_score = 0.3  # Default low relevance

        if any(keyword in task_description or keyword in task_type for keyword in code_keywords):
            relevance_score = 0.9

        factors["code_quality_relevance"] = relevance_score

        # Resource availability (based on context)
        resources = context.get("resources", {})
        time_available = resources.get("time", "unknown")
        team_size = resources.get("team_size", 1)

        resource_score = 0.5  # Default
        if isinstance(team_size, int) and team_size > 3:
            resource_score += 0.2
        if time_available == "unlimited" or "plenty" in str(time_available):
            resource_score += 0.2
        elif time_available == "tight" or "urgent" in str(time_available):
            resource_score -= 0.3

        factors["resource_availability"] = max(0.0, min(1.0, resource_score))

        return factors

    def _generate_assessment(self, task: Dict[str, Any],
                           factors: Dict[str, float], confidence: float) -> str:
        """Generate overall assessment based on analysis."""
        task_type = task.get("type", "task")

        if confidence >= 0.8:
            confidence_phrase = "highly confident"
        elif confidence >= 0.6:
            confidence_phrase = "moderately confident"
        else:
            confidence_phrase = "cautiously optimistic"

        code_relevance = factors.get("code_quality_relevance", 0.0)
        if code_relevance >= 0.8:
            relevance_phrase = "directly aligned with code quality best practices"
        elif code_relevance >= 0.5:
            relevance_phrase = "partially related to code quality concerns"
        else:
            relevance_phrase = "tangentially related to code quality"

        return f"I am {confidence_phrase} about this {task_type}, which is {relevance_phrase}."

    def _create_recommendations(self, task: Dict[str, Any],
                              factors: Dict[str, float]) -> List[str]:
        """Create specific recommendations based on analysis."""
        recommendations = []

        # Requirements clarity recommendations
        if factors.get("requirements_clarity", 0.5) < 0.6:
            recommendations.append("Clarify requirements and create detailed specifications")
            recommendations.append("Define acceptance criteria and success metrics")

        # Technical complexity recommendations
        if factors.get("technical_complexity", 0.5) > 0.7:
            recommendations.append("Break down complex tasks into smaller, manageable components")
            recommendations.append("Create prototypes to validate technical approaches")
            recommendations.append("Implement comprehensive testing strategy")

        # Code quality specific recommendations
        if factors.get("code_quality_relevance", 0.0) > 0.5:
            recommendations.extend([
                "Implement code review process with peer reviews",
                "Set up automated testing and continuous integration",
                "Establish coding standards and style guidelines",
                "Use static analysis tools for code quality checking"
            ])

        # Resource-based recommendations
        if factors.get("resource_availability", 0.5) < 0.5:
            recommendations.append("Consider prioritizing features and implementing MVP approach")
            recommendations.append("Identify and mitigate resource constraints early")

        return recommendations

    def _identify_risks(self, task: Dict[str, Any],
                       factors: Dict[str, float]) -> List[str]:
        """Identify potential risks based on analysis."""
        risks = []

        # Low clarity risks
        if factors.get("requirements_clarity", 0.5) < 0.5:
            risks.extend([
                "Scope creep due to unclear requirements",
                "Frequent changes and rework cycles"
            ])

        # High complexity risks
        if factors.get("technical_complexity", 0.5) > 0.7:
            risks.extend([
                "Technical implementation challenges",
                "Extended development timeline",
                "Integration difficulties"
            ])

        # Resource constraint risks
        if factors.get("resource_availability", 0.5) < 0.4:
            risks.extend([
                "Insufficient time or resources to complete properly",
                "Quality compromises due to rushed implementation"
            ])

        # Code quality specific risks
        if factors.get("code_quality_relevance", 0.0) > 0.5:
            risks.extend([
                "Technical debt accumulation",
                "Maintenance and scalability issues",
                "Security vulnerabilities if not properly reviewed"
            ])

        return risks

    def _estimate_effort(self, task: Dict[str, Any],
                        factors: Dict[str, float]) -> str:
        """Estimate effort required for the task."""
        complexity = factors.get("technical_complexity", 0.5)
        clarity = factors.get("requirements_clarity", 0.5)
        relevance = factors.get("code_quality_relevance", 0.0)

        # Base effort calculation
        base_effort = (complexity + (1 - clarity) + relevance) / 3

        if base_effort <= 0.3:
            return "Low (1-3 days)"
        elif base_effort <= 0.6:
            return "Medium (3-7 days)"
        elif base_effort <= 0.8:
            return "High (1-2 weeks)"
        else:
            return "Very High (2+ weeks)"


# Example usage and testing function
def demonstrate_director():
    """
    Demonstrate the Example Director functionality.
    This can be run to test the director implementation.
    """
    print("=== Echo Brain Director Framework Demo ===\n")

    # Create director instance
    director = ExampleDirector()
    print(f"Created director: {director}")
    print(f"Expertise: {director.expertise}\n")

    # Example task
    task = {
        "type": "code_review",
        "description": "Review the authentication system implementation for security and code quality issues",
        "priority": "high"
    }

    context = {
        "user": "developer",
        "resources": {
            "time": "moderate",
            "team_size": 3
        },
        "constraints": ["Must maintain backward compatibility"]
    }

    print("Evaluating task:")
    print(f"Task: {task}")
    print(f"Context: {context}\n")

    # Evaluate task
    result = director.evaluate(task, context)

    print("Evaluation Results:")
    print(f"Assessment: {result['assessment']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Estimated Effort: {result['estimated_effort']}")
    print(f"\nRecommendations ({len(result['recommendations'])}):")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i}. {rec}")

    print(f"\nRisk Factors ({len(result['risk_factors'])}):")
    for i, risk in enumerate(result['risk_factors'], 1):
        print(f"  {i}. {risk}")

    print(f"\nReasoning:\n{result['reasoning']}")

    # Get performance summary
    performance = director.get_performance_summary()
    print(f"\nDirector Performance:")
    print(f"Evaluations: {performance['evaluations_performed']}")
    print(f"Average Confidence: {performance['average_confidence']}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demonstrate_director()