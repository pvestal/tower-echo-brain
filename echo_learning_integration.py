#!/usr/bin/env python3
"""
Echo Learning Integration - Unified Learning System Integration
=============================================================

Integration layer that connects all Echo learning components:
- Coordinates between learning system, self-diagnosis, board manager, task decomposer, and outcome tracker
- Provides unified API for Echo's learning capabilities
- Manages data flow between learning components
- Orchestrates learning workflows and decision loops
- Provides centralized learning insights and recommendations

This is the main integration point for Echo's comprehensive learning system.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from echo_board_manager import (BoardDecisionType, consult_board,
                                get_board_manager, get_board_status,
                                record_board_outcome)
# Import all learning components
from echo_learning_system import (DecisionOutcome, LearningDomain,
                                  get_learning_system)
from echo_learning_system import \
    get_recommendations as learning_get_recommendations
from echo_learning_system import record_decision as learning_record_decision
from echo_learning_system import record_outcome as learning_record_outcome
from echo_outcome_tracker import (ImpactLevel, OutcomeType,
                                  analyze_impact_over_period,
                                  get_outcome_tracker, get_performance_summary)
from echo_outcome_tracker import record_outcome as tracker_record_outcome
from echo_self_diagnosis import (get_diagnosis_system, get_health_status,
                                 record_error, record_response_time,
                                 start_health_monitoring)
from echo_task_decomposer import (TaskPriority, decompose_request,
                                  execute_workflow, get_task_decomposer,
                                  get_workflow_status)


@dataclass
class LearningSystemStatus:
    """Overall status of Echo's learning systems"""

    learning_core_status: str
    self_diagnosis_status: str
    board_manager_status: str
    task_decomposer_status: str
    outcome_tracker_status: str
    overall_health: str
    active_learning_processes: int
    recent_insights: List[str]
    system_recommendations: List[str]


@dataclass
class EchoDecisionRequest:
    """Comprehensive decision request through Echo's learning system"""

    request_id: str
    original_request: str
    context: Dict[str, Any]
    priority: str
    complexity_level: float
    requires_board_input: bool
    requires_task_decomposition: bool
    expected_outcome_tracking: bool


@dataclass
class EchoDecisionResponse:
    """Comprehensive decision response from Echo's learning system"""

    request_id: str
    decision_made: str
    confidence: float
    decision_path: List[str]
    board_consultation_used: bool
    task_workflow_created: Optional[str]
    learning_insights: List[str]
    monitoring_recommendations: List[str]
    expected_outcomes: Dict[str, Any]


class EchoLearningIntegration:
    """
    Unified Learning System Integration for Echo

    This class orchestrates all of Echo's learning capabilities,
    providing a single interface for intelligent decision-making,
    learning, and self-improvement.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize all learning components
        self.learning_system = get_learning_system()
        self.diagnosis_system = get_diagnosis_system()
        self.board_manager = get_board_manager()
        self.task_decomposer = get_task_decomposer()
        self.outcome_tracker = get_outcome_tracker()

        # Integration state
        self.active_decisions: Dict[str, EchoDecisionRequest] = {}
        self.decision_outcomes: Dict[str, Dict[str, Any]] = {}
        self.learning_insights_cache: List[str] = []

        # Performance tracking
        self.decision_start_times: Dict[str, datetime] = {}

        self.logger.info("Echo Learning Integration initialized")

    async def initialize_learning_systems(self):
        """Initialize all learning systems"""
        try:
            self.logger.info("Initializing Echo learning systems...")

            # Initialize board manager
            await self.board_manager.initialize_board()

            # Start health monitoring
            await start_health_monitoring()

            # Initialize other components as needed
            # (Most components initialize automatically)

            self.logger.info("Echo learning systems initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize learning systems: {e}")
            return False

    async def make_intelligent_decision(
        self,
        request: str,
        context: Dict[str, Any],
        priority: str = "medium",
        user_context: Optional[Dict[str, Any]] = None,
    ) -> EchoDecisionResponse:
        """
        Make an intelligent decision using Echo's full learning capabilities

        Args:
            request: The decision request or task
            context: Context about the request
            priority: Priority level (low, medium, high, critical)
            user_context: Additional user-specific context

        Returns:
            Comprehensive decision response with learning insights
        """
        request_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request) % 10000}"
        start_time = datetime.now()
        self.decision_start_times[request_id] = start_time

        self.logger.info(f"Making intelligent decision: {request_id}")

        # Analyze request complexity and requirements
        analysis = await self._analyze_decision_requirements(request, context, priority)

        # Create decision request
        decision_request = EchoDecisionRequest(
            request_id=request_id,
            original_request=request,
            context=context,
            priority=priority,
            complexity_level=analysis["complexity"],
            requires_board_input=analysis["requires_board"],
            requires_task_decomposition=analysis["requires_decomposition"],
            expected_outcome_tracking=analysis["requires_tracking"],
        )

        self.active_decisions[request_id] = decision_request

        try:
            # Get learning system recommendations
            learning_recs = await learning_get_recommendations(
                LearningDomain.GENERAL_REASONING,
                {
                    "request": request,
                    "context": context,
                    "priority": priority,
                    "complexity": analysis["complexity"],
                },
            )

            decision_path = ["learning_analysis"]
            decision_confidence = 0.7
            board_used = False
            workflow_id = None
            insights = []
            monitoring_recs = []

            # Consult board if needed
            if decision_request.requires_board_input:
                board_result = await self._consult_board_for_decision(
                    request, context, analysis
                )
                decision_confidence = max(
                    decision_confidence, board_result["confidence"]
                )
                decision_path.append("board_consultation")
                board_used = True
                insights.extend(board_result.get("insights", []))

            # Decompose into tasks if needed
            if decision_request.requires_task_decomposition:
                workflow_id = await self._decompose_decision_into_tasks(
                    request, context, analysis, priority
                )
                decision_path.append("task_decomposition")
                insights.append(
                    f"Created workflow {workflow_id} for task execution")

            # Generate decision
            final_decision = await self._synthesize_final_decision(
                request, context, learning_recs, board_used, workflow_id
            )

            # Record decision for learning
            learning_decision_id = await learning_record_decision(
                domain=LearningDomain.GENERAL_REASONING,
                context={
                    "request": request,
                    "context": context,
                    "analysis": analysis,
                    "decision_path": decision_path,
                },
                decision_factors={
                    "complexity": analysis["complexity"],
                    "board_input": 1.0 if board_used else 0.0,
                    "task_decomposition": 1.0 if workflow_id else 0.0,
                    "learning_confidence": learning_recs.get(
                        "confidence_adjustment", 0.0
                    ),
                },
                decision_made=final_decision,
                confidence=decision_confidence,
            )

            # Generate monitoring recommendations
            monitoring_recs = await self._generate_monitoring_recommendations(
                decision_request, final_decision, analysis
            )

            # Track response time
            response_time = (datetime.now() - start_time).total_seconds()
            await record_response_time(response_time)

            response = EchoDecisionResponse(
                request_id=request_id,
                decision_made=final_decision,
                confidence=decision_confidence,
                decision_path=decision_path,
                board_consultation_used=board_used,
                task_workflow_created=workflow_id,
                learning_insights=insights,
                monitoring_recommendations=monitoring_recs,
                expected_outcomes=analysis.get("expected_outcomes", {}),
            )

            # Schedule outcome tracking if needed
            if decision_request.expected_outcome_tracking:
                await self._schedule_outcome_tracking(request_id, response, analysis)

            self.logger.info(
                f"Decision completed: {request_id} (confidence: {decision_confidence:.2f})"
            )
            return response

        except Exception as e:
            self.logger.error(f"Decision making failed: {e}")
            await record_error("decision_making")

            # Return fallback response
            return EchoDecisionResponse(
                request_id=request_id,
                decision_made=f"Error processing request: {e}",
                confidence=0.1,
                decision_path=["error"],
                board_consultation_used=False,
                task_workflow_created=None,
                learning_insights=[f"Decision failed: {e}"],
                monitoring_recommendations=["Monitor for system issues"],
                expected_outcomes={},
            )

    async def _analyze_decision_requirements(
        self, request: str, context: Dict[str, Any], priority: str
    ) -> Dict[str, Any]:
        """Analyze what decision-making components are needed"""
        analysis = {
            "complexity": 0.5,
            "requires_board": False,
            "requires_decomposition": False,
            "requires_tracking": True,
            "expected_outcomes": {},
            "risk_factors": [],
        }

        # Analyze complexity
        complexity_indicators = [
            len(request.split()) / 50,  # Length factor
            len(context) / 10,  # Context richness
            int("complex" in request.lower()) * 0.3,
            int("system" in request.lower()) * 0.2,
            int("integrate" in request.lower()) * 0.4,
            {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}.get(
                priority, 0.5
            ),
        ]

        analysis["complexity"] = min(1.0, sum(complexity_indicators))

        # Determine if board consultation is needed
        board_keywords = ["critical", "security",
                          "ethics", "complex", "strategic"]
        analysis["requires_board"] = (
            analysis["complexity"] > 0.7
            or priority in ["high", "critical"]
            or any(keyword in request.lower() for keyword in board_keywords)
        )

        # Determine if task decomposition is needed
        decomposition_keywords = [
            "implement",
            "create",
            "build",
            "develop",
            "comprehensive",
        ]
        analysis["requires_decomposition"] = (
            analysis["complexity"] > 0.6
            or len(request.split()) > 20
            or any(keyword in request.lower() for keyword in decomposition_keywords)
        )

        # Identify risk factors
        risk_keywords = ["security", "critical", "system-wide", "irreversible"]
        analysis["risk_factors"] = [
            keyword for keyword in risk_keywords if keyword in request.lower()
        ]

        # Set expected outcomes based on request type
        if "performance" in request.lower():
            analysis["expected_outcomes"]["performance_improvement"] = 0.2
        if "error" in request.lower():
            analysis["expected_outcomes"]["error_reduction"] = 0.5
        if "user" in request.lower():
            analysis["expected_outcomes"]["user_satisfaction"] = 0.8

        return analysis

    async def _consult_board_for_decision(
        self, request: str, context: Dict[str, Any], analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Consult board for complex decisions"""
        try:
            # Determine board decision type
            if analysis["complexity"] > 0.8:
                decision_type = BoardDecisionType.FULL_ANALYSIS
            elif len(analysis["risk_factors"]) > 0:
                decision_type = BoardDecisionType.EXPERT_REVIEW
            else:
                decision_type = BoardDecisionType.CONSULTATION

            # Consult board
            board_result = await consult_board(
                request=request,
                context={
                    **context,
                    "complexity_analysis": analysis,
                    "complexity_indicators": [
                        "high_complexity",
                        "expert_review_needed",
                    ],
                },
                decision_type=decision_type,
                domain="general",
                required_confidence=0.7,
            )

            return {
                "decision": board_result.get("recommendation", "No recommendation"),
                "confidence": board_result.get("confidence", 0.5),
                "insights": board_result.get("individual_insights", []),
                "consensus": board_result.get("consensus_reached", False),
            }

        except Exception as e:
            self.logger.error(f"Board consultation failed: {e}")
            return {
                "decision": f"Board consultation failed: {e}",
                "confidence": 0.3,
                "insights": [],
                "consensus": False,
            }

    async def _decompose_decision_into_tasks(
        self,
        request: str,
        context: Dict[str, Any],
        analysis: Dict[str, Any],
        priority: str,
    ) -> Optional[str]:
        """Decompose decision into executable tasks"""
        try:
            # Map priority to TaskPriority enum
            priority_mapping = {
                "low": TaskPriority.LOW,
                "medium": TaskPriority.MEDIUM,
                "high": TaskPriority.HIGH,
                "critical": TaskPriority.CRITICAL,
            }

            task_priority = priority_mapping.get(priority, TaskPriority.MEDIUM)

            # Decompose request
            workflow_id = await decompose_request(
                request=request,
                user_context={
                    **context,
                    "analysis": analysis,
                    "complexity_indicators": [
                        "decomposition_needed",
                        "multi_step_process",
                    ],
                },
                priority=task_priority,
            )

            return workflow_id

        except Exception as e:
            self.logger.error(f"Task decomposition failed: {e}")
            return None

    async def _synthesize_final_decision(
        self,
        request: str,
        context: Dict[str, Any],
        learning_recs: Dict[str, Any],
        board_used: bool,
        workflow_id: Optional[str],
    ) -> str:
        """Synthesize final decision from all inputs"""
        decision_parts = []

        # Start with base decision
        decision_parts.append(f"For the request '{request}', Echo recommends:")

        # Add learning insights
        if learning_recs.get("suggestions"):
            decision_parts.append(
                f"Based on learning patterns: {learning_recs['suggestions'][0]}"
            )

        # Add board input if used
        if board_used:
            decision_parts.append(
                "Board consultation has been incorporated into this decision"
            )

        # Add task workflow if created
        if workflow_id:
            decision_parts.append(
                f"Implementation will proceed via workflow {workflow_id}"
            )

        # Add confidence adjustments
        confidence_adj = learning_recs.get("confidence_adjustment", 0.0)
        if confidence_adj > 0.1:
            decision_parts.append(
                "High confidence based on successful pattern history")
        elif confidence_adj < -0.1:
            decision_parts.append(
                "Proceeding with caution due to historical concerns")

        return " ".join(decision_parts)

    async def _generate_monitoring_recommendations(
        self,
        decision_request: EchoDecisionRequest,
        final_decision: str,
        analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations for monitoring the decision outcome"""
        recommendations = []

        # Based on complexity
        if analysis["complexity"] > 0.7:
            recommendations.append("Monitor closely due to high complexity")

        # Based on risk factors
        if analysis["risk_factors"]:
            recommendations.append(
                f"Watch for risks: {', '.join(analysis['risk_factors'])}"
            )

        # Based on decision type
        if decision_request.requires_task_decomposition:
            recommendations.append(
                "Track task workflow progress and completion")

        if decision_request.requires_board_input:
            recommendations.append(
                "Validate board recommendations are being followed")

        # Based on expected outcomes
        if analysis.get("expected_outcomes"):
            recommendations.append("Track expected outcome metrics")

        # General monitoring
        recommendations.append(
            "Monitor system performance during implementation")

        return recommendations

    async def _schedule_outcome_tracking(
        self, request_id: str, response: EchoDecisionResponse, analysis: Dict[str, Any]
    ):
        """Schedule outcome tracking for the decision"""
        try:
            # Schedule immediate outcome tracking
            asyncio.create_task(
                self._track_immediate_outcome(request_id, response, analysis)
            )

            # Schedule delayed outcome tracking if appropriate
            if analysis["complexity"] > 0.6:
                asyncio.create_task(
                    self._track_delayed_outcome(
                        request_id, response, analysis, hours=24
                    )
                )

        except Exception as e:
            self.logger.error(f"Failed to schedule outcome tracking: {e}")

    async def _track_immediate_outcome(
        self, request_id: str, response: EchoDecisionResponse, analysis: Dict[str, Any]
    ):
        """Track immediate outcome of a decision"""
        await asyncio.sleep(5)  # Wait a bit for immediate effects

        try:
            # Simulate immediate outcome observation
            immediate_results = {
                "decision_executed": True,
                "immediate_errors": 0,
                "system_stability": 1.0,
                "response_generated": True,
            }

            await tracker_record_outcome(
                decision_id=request_id,
                decision_context={
                    "original_request": response.request_id,
                    "decision_made": response.decision_made,
                    "confidence": response.confidence,
                    "complexity": analysis["complexity"],
                },
                decision_maker="echo_learning_system",
                observed_results=immediate_results,
                outcome_type=OutcomeType.IMMEDIATE,
            )

            self.logger.info(f"Immediate outcome tracked for {request_id}")

        except Exception as e:
            self.logger.error(f"Failed to track immediate outcome: {e}")

    async def _track_delayed_outcome(
        self,
        request_id: str,
        response: EchoDecisionResponse,
        analysis: Dict[str, Any],
        hours: int = 24,
    ):
        """Track delayed outcome of a decision"""
        await asyncio.sleep(hours * 3600)  # Wait for specified hours

        try:
            # Get system status for delayed outcome assessment
            health_status = await get_health_status()
            board_status = await get_board_status()
            performance_summary = await get_performance_summary()

            delayed_results = {
                "system_health": health_status.get("overall_health", "unknown"),
                "board_performance": board_status.get("average_performance", 0.5),
                "performance_metrics": performance_summary.get("metric_trends", {}),
                "decision_effectiveness": 0.8,  # Would be calculated based on actual metrics
            }

            await tracker_record_outcome(
                decision_id=request_id,
                decision_context={
                    "original_request": response.request_id,
                    "decision_made": response.decision_made,
                    "hours_elapsed": hours,
                },
                decision_maker="echo_learning_system",
                observed_results=delayed_results,
                outcome_type=OutcomeType.MEDIUM_TERM,
            )

            self.logger.info(
                f"Delayed outcome tracked for {request_id} after {hours} hours"
            )

        except Exception as e:
            self.logger.error(f"Failed to track delayed outcome: {e}")

    async def record_decision_outcome(
        self,
        request_id: str,
        outcome: str,
        metrics: Dict[str, float],
        user_feedback: Optional[str] = None,
    ):
        """Record the actual outcome of a decision"""
        if request_id not in self.active_decisions:
            self.logger.warning(
                f"Decision {request_id} not found for outcome recording"
            )
            return

        decision_request = self.active_decisions[request_id]

        # Convert outcome to DecisionOutcome enum
        if "success" in outcome.lower() or "completed" in outcome.lower():
            learning_outcome = DecisionOutcome.SUCCESS
        elif "failed" in outcome.lower() or "error" in outcome.lower():
            learning_outcome = DecisionOutcome.FAILURE
        else:
            learning_outcome = DecisionOutcome.PARTIAL

        # Record in learning system
        await learning_record_outcome(
            decision_id=request_id,
            outcome=learning_outcome,
            outcome_metrics=metrics,
            feedback=user_feedback,
        )

        # Record in outcome tracker
        await tracker_record_outcome(
            decision_id=request_id,
            decision_context=decision_request.context,
            decision_maker="echo_learning_system",
            observed_results={
                "outcome_description": outcome,
                "user_feedback": user_feedback,
                **metrics,
            },
        )

        # Update decision outcomes
        self.decision_outcomes[request_id] = {
            "outcome": outcome,
            "metrics": metrics,
            "user_feedback": user_feedback,
            "timestamp": datetime.now(),
        }

        self.logger.info(
            f"Decision outcome recorded for {request_id}: {outcome}")

    async def get_learning_system_status(self) -> LearningSystemStatus:
        """Get comprehensive status of all learning systems"""
        try:
            # Get individual system statuses
            health_status = await get_health_status()
            board_status = await get_board_status()
            performance_summary = await get_performance_summary()

            # Compile overall status
            status = LearningSystemStatus(
                learning_core_status="operational",
                self_diagnosis_status=(
                    health_status.get("system_insights", ["unknown"])[0]
                    if health_status.get("system_insights")
                    else "unknown"
                ),
                board_manager_status=board_status.get(
                    "system_health", "unknown"),
                task_decomposer_status="operational",
                outcome_tracker_status="operational",
                overall_health="good",
                active_learning_processes=len(self.active_decisions),
                # Last 5 insights
                recent_insights=self.learning_insights_cache[-5:],
                system_recommendations=await self._generate_system_recommendations(),
            )

            # Determine overall health
            system_healths = [
                status.learning_core_status,
                status.self_diagnosis_status,
                status.board_manager_status,
                status.task_decomposer_status,
                status.outcome_tracker_status,
            ]

            if "critical" in system_healths:
                status.overall_health = "critical"
            elif "degraded" in system_healths:
                status.overall_health = "degraded"
            elif all(h in ["operational", "good", "excellent"] for h in system_healths):
                status.overall_health = "excellent"
            else:
                status.overall_health = "good"

            return status

        except Exception as e:
            self.logger.error(f"Failed to get learning system status: {e}")
            return LearningSystemStatus(
                learning_core_status="error",
                self_diagnosis_status="error",
                board_manager_status="error",
                task_decomposer_status="error",
                outcome_tracker_status="error",
                overall_health="critical",
                active_learning_processes=0,
                recent_insights=[f"Error getting status: {e}"],
                system_recommendations=["Investigate system errors"],
            )

    async def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations based on current state"""
        recommendations = []

        try:
            # Get recent performance data
            health_status = await get_health_status()
            board_status = await get_board_status()

            # Health-based recommendations
            if health_status.get("critical_issues", 0) > 0:
                recommendations.append(
                    "Address critical health issues immediately")

            # Board performance recommendations
            avg_board_performance = board_status.get(
                "average_performance", 0.5)
            if avg_board_performance < 0.6:
                recommendations.append(
                    "Review and optimize Board of Directors performance"
                )

            # Decision-making recommendations
            if len(self.active_decisions) > 10:
                recommendations.append(
                    "High decision load - consider prioritization")

            # Learning system recommendations
            recent_errors = len(
                [
                    d
                    for d in self.decision_outcomes.values()
                    if "error" in d.get("outcome", "").lower()
                ]
            )
            if recent_errors > 3:
                recommendations.append(
                    "High error rate detected - review decision processes"
                )

            # Default recommendations if none generated
            if not recommendations:
                recommendations.extend(
                    [
                        "System operating normally",
                        "Continue monitoring performance",
                        "Regular learning pattern review recommended",
                    ]
                )

        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")

        return recommendations

    async def get_recent_learning_insights(
        self, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent learning insights from all systems"""
        insights = []

        try:
            # Get insights from learning system
            learning_status = await self.learning_system.get_learning_status()
            insights.extend(
                [
                    {
                        "source": "learning_system",
                        "insight": insight,
                        "timestamp": datetime.now(),
                    }
                    for insight in learning_status.get("system_insights", [])
                ]
            )

            # Get insights from outcome tracker
            recent_outcomes = await self.outcome_tracker.get_recent_outcomes(days=1)
            for outcome in recent_outcomes[-5:]:  # Last 5 outcomes
                insights.extend(
                    [
                        {
                            "source": "outcome_tracker",
                            "insight": insight,
                            "timestamp": outcome.timestamp,
                        }
                        for insight in outcome.learning_insights
                    ]
                )

            # Get insights from board manager
            board_insights = await self.board_manager.get_director_performance(
                "quality_director"
            )
            if board_insights:
                insights.extend(
                    [
                        {
                            "source": "board_manager",
                            "insight": rec,
                            "timestamp": datetime.now(),
                        }
                        for rec in board_insights.recommendations
                    ]
                )

            # Sort by timestamp
            insights.sort(key=lambda x: x["timestamp"], reverse=True)

            return insights[:20]  # Return most recent 20 insights

        except Exception as e:
            self.logger.error(f"Failed to get learning insights: {e}")
            return [
                {
                    "source": "error",
                    "insight": f"Error getting insights: {e}",
                    "timestamp": datetime.now(),
                }
            ]

    async def optimize_learning_systems(self) -> Dict[str, Any]:
        """Perform optimization across all learning systems"""
        optimization_results = {
            "optimizations_performed": [],
            "improvements_detected": [],
            "recommendations": [],
            "overall_impact": 0.0,
        }

        try:
            self.logger.info("Starting learning system optimization")

            # Optimize board composition
            try:
                optimized_composition = (
                    await self.board_manager.optimize_board_composition(
                        BoardDecisionType.CONSULTATION, "general"
                    )
                )
                optimization_results["optimizations_performed"].append(
                    "board_composition_optimized"
                )
                optimization_results["improvements_detected"].append(
                    f"Board quality improved to {optimized_composition.expected_quality:.2f}"
                )
            except Exception as e:
                self.logger.error(f"Board optimization failed: {e}")

            # Analyze recent impact
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                impact_analysis = await analyze_impact_over_period(start_date, end_date)

                if impact_analysis.overall_success_rate > 0.8:
                    optimization_results["improvements_detected"].append(
                        "High success rate maintained"
                    )
                else:
                    optimization_results["recommendations"].append(
                        "Review decision-making processes"
                    )

                optimization_results["overall_impact"] = (
                    impact_analysis.overall_success_rate
                )

            except Exception as e:
                self.logger.error(f"Impact analysis failed: {e}")

            # Generate optimization recommendations
            optimization_results["recommendations"].extend(
                [
                    "Continue monitoring system performance",
                    "Regular learning pattern validation",
                    "Board director performance review",
                ]
            )

            self.logger.info("Learning system optimization completed")
            return optimization_results

        except Exception as e:
            self.logger.error(f"Learning system optimization failed: {e}")
            return {
                "optimizations_performed": [],
                "improvements_detected": [],
                "recommendations": [f"Optimization failed: {e}"],
                "overall_impact": 0.0,
            }


# Global instance
_learning_integration = None


def get_learning_integration() -> EchoLearningIntegration:
    """Get global learning integration instance"""
    global _learning_integration
    if _learning_integration is None:
        _learning_integration = EchoLearningIntegration()
    return _learning_integration


# Convenience functions for main Echo system
async def make_intelligent_decision(
    request: str,
    context: Dict[str, Any],
    priority: str = "medium",
    user_context: Optional[Dict[str, Any]] = None,
) -> EchoDecisionResponse:
    """Convenience function for intelligent decision making"""
    integration = get_learning_integration()
    return await integration.make_intelligent_decision(
        request, context, priority, user_context
    )


async def record_decision_outcome(
    request_id: str,
    outcome: str,
    metrics: Dict[str, float],
    user_feedback: Optional[str] = None,
):
    """Convenience function for recording decision outcomes"""
    integration = get_learning_integration()
    await integration.record_decision_outcome(
        request_id, outcome, metrics, user_feedback
    )


async def get_learning_system_status() -> LearningSystemStatus:
    """Convenience function for getting learning system status"""
    integration = get_learning_integration()
    return await integration.get_learning_system_status()


async def initialize_echo_learning():
    """Initialize all Echo learning systems"""
    integration = get_learning_integration()
    return await integration.initialize_learning_systems()


if __name__ == "__main__":
    # Example usage
    async def example():
        # Initialize learning systems
        success = await initialize_echo_learning()
        print(f"Learning systems initialized: {success}")

        # Make an intelligent decision
        response = await make_intelligent_decision(
            request="Optimize the video generation pipeline for better performance",
            context={
                "current_performance": 0.6,
                "user_complaints": 3,
                "system_load": 0.8,
                "available_resources": True,
            },
            priority="high",
        )

        print("Decision Response:")
        print(f"  Decision: {response.decision_made}")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Decision Path: {' -> '.join(response.decision_path)}")
        print(f"  Board Used: {response.board_consultation_used}")
        print(f"  Workflow Created: {response.task_workflow_created}")

        # Later, record the outcome
        await record_decision_outcome(
            response.request_id,
            "Successfully optimized pipeline with 25% performance improvement",
            {
                "performance_improvement": 0.25,
                "user_satisfaction": 0.9,
                "implementation_time": 2.5,
            },
            "Optimization was successful and user-friendly",
        )

        # Get system status
        status = await get_learning_system_status()
        print(f"\nLearning System Status: {status.overall_health}")
        print(f"Active Processes: {status.active_learning_processes}")
        print(f"Recent Insights: {len(status.recent_insights)}")

    asyncio.run(example())
