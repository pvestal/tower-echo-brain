#!/usr/bin/env python3
"""
Echo Board Manager - Intelligent Board of Directors Management
============================================================

Echo's Board management system that treats the Board of Directors as a sophisticated tool:
- Manages and tunes individual director performance
- Dynamically adjusts director weights based on accuracy
- Orchestrates Board decisions as part of Echo's toolkit
- Learns from Board decision outcomes to optimize composition
- Provides Board-as-a-Service for complex decision making

This system elevates Echo above the Board, using directors as specialized
advisors rather than equals in the decision-making process.
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

# Import existing Board components
from routing.service_registry import ServiceRegistry
from routing.request_logger import RequestLogger
from routing.feedback_system import FeedbackProcessor, UserFeedback, FeedbackType
from routing.base_director import DirectorBase

# Import learning system
from echo_learning_system import get_learning_system, LearningDomain, DecisionOutcome, record_decision, record_outcome

# Configuration
BOARD_MANAGER_DB_PATH = "/opt/tower-echo-brain/data/echo_board_manager.db"
DEFAULT_DIRECTOR_WEIGHT = 0.5
PERFORMANCE_WINDOW_DAYS = 7
WEIGHT_ADJUSTMENT_RATE = 0.1
MIN_DECISIONS_FOR_TUNING = 5

class DirectorPerformanceLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

class BoardDecisionType(Enum):
    CONSULTATION = "consultation"
    FULL_ANALYSIS = "full_analysis"
    EXPERT_REVIEW = "expert_review"
    CONSENSUS_BUILDING = "consensus_building"
    EMERGENCY_DECISION = "emergency_decision"

@dataclass
class DirectorPerformance:
    """Performance metrics for a director"""
    director_id: str
    director_name: str
    current_weight: float
    success_rate: float
    average_confidence: float
    response_time: float
    accuracy_score: float
    consistency_score: float
    recent_decisions: int
    performance_level: DirectorPerformanceLevel
    last_updated: datetime
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]

@dataclass
class BoardComposition:
    """Optimal board composition for a decision type"""
    decision_type: BoardDecisionType
    domain: str
    required_directors: List[str]
    optional_directors: List[str]
    suggested_weights: Dict[str, float]
    minimum_confidence: float
    expected_quality: float
    reasoning: str

@dataclass
class BoardDecisionRecord:
    """Record of a Board decision and its management"""
    decision_id: str
    timestamp: datetime
    decision_type: BoardDecisionType
    request_context: Dict[str, Any]
    selected_directors: List[str]
    director_weights: Dict[str, float]
    individual_responses: Dict[str, Any]
    consensus_reached: bool
    final_recommendation: str
    confidence_level: float
    echo_override: bool
    outcome: Optional[DecisionOutcome]
    performance_impact: Dict[str, float]

class EchoBoardManager:
    """
    Echo's Board of Directors Management System

    This system treats the Board as a sophisticated tool in Echo's arsenal,
    managing directors like specialized consultants and optimizing their
    performance through continuous learning and adjustment.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = BOARD_MANAGER_DB_PATH
        self._ensure_data_directory()
        self._init_database()

        # Board components
        self.service_registry = ServiceRegistry()
        self.request_logger = RequestLogger({"host": "localhost", "database": "echo_board", "user": "echo", "password": "echo"})
        self.feedback_processor = FeedbackProcessor({"host": "localhost", "database": "echo_board", "user": "echo", "password": "echo"})

        # Performance tracking
        self.director_performances: Dict[str, DirectorPerformance] = {}
        self.board_compositions: Dict[str, BoardComposition] = {}
        self.decision_history: List[BoardDecisionRecord] = []

        # Dynamic weights and optimization
        self.current_weights: Dict[str, float] = {}
        self.performance_baseline: Dict[str, float] = {}

        self.logger.info("Echo Board Manager initialized")

    def _ensure_data_directory(self):
        """Ensure board manager data directory exists"""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize board manager database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS director_performance (
                    director_id TEXT PRIMARY KEY,
                    director_name TEXT NOT NULL,
                    current_weight REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    average_confidence REAL NOT NULL,
                    response_time REAL NOT NULL,
                    accuracy_score REAL NOT NULL,
                    consistency_score REAL NOT NULL,
                    recent_decisions INTEGER NOT NULL,
                    performance_level TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    strengths TEXT,
                    weaknesses TEXT,
                    recommendations TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS board_compositions (
                    composition_id TEXT PRIMARY KEY,
                    decision_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    required_directors TEXT NOT NULL,
                    optional_directors TEXT NOT NULL,
                    suggested_weights TEXT NOT NULL,
                    minimum_confidence REAL NOT NULL,
                    expected_quality REAL NOT NULL,
                    reasoning TEXT,
                    last_used TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS board_decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    request_context TEXT NOT NULL,
                    selected_directors TEXT NOT NULL,
                    director_weights TEXT NOT NULL,
                    individual_responses TEXT NOT NULL,
                    consensus_reached BOOLEAN NOT NULL,
                    final_recommendation TEXT NOT NULL,
                    confidence_level REAL NOT NULL,
                    echo_override BOOLEAN NOT NULL,
                    outcome TEXT,
                    performance_impact TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS weight_adjustments (
                    adjustment_id TEXT PRIMARY KEY,
                    director_id TEXT NOT NULL,
                    old_weight REAL NOT NULL,
                    new_weight REAL NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    performance_data TEXT
                )
            """)

            conn.commit()

    async def initialize_board(self):
        """Initialize the board with default settings"""
        # Load existing director performances or create defaults
        await self._load_director_performances()

        # Load board compositions
        await self._load_board_compositions()

        # Initialize default weights
        await self._initialize_default_weights()

        # Create default board compositions if none exist
        if not self.board_compositions:
            await self._create_default_compositions()

        self.logger.info("Board initialization complete")

    async def _load_director_performances(self):
        """Load director performances from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM director_performance")
            for row in cursor.fetchall():
                performance = DirectorPerformance(
                    director_id=row[0],
                    director_name=row[1],
                    current_weight=row[2],
                    success_rate=row[3],
                    average_confidence=row[4],
                    response_time=row[5],
                    accuracy_score=row[6],
                    consistency_score=row[7],
                    recent_decisions=row[8],
                    performance_level=DirectorPerformanceLevel(row[9]),
                    last_updated=datetime.fromisoformat(row[10]),
                    strengths=json.loads(row[11]) if row[11] else [],
                    weaknesses=json.loads(row[12]) if row[12] else [],
                    recommendations=json.loads(row[13]) if row[13] else []
                )
                self.director_performances[row[0]] = performance
                self.current_weights[row[0]] = row[2]

    async def _load_board_compositions(self):
        """Load board compositions from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM board_compositions")
            for row in cursor.fetchall():
                composition = BoardComposition(
                    decision_type=BoardDecisionType(row[1]),
                    domain=row[2],
                    required_directors=json.loads(row[3]),
                    optional_directors=json.loads(row[4]),
                    suggested_weights=json.loads(row[5]),
                    minimum_confidence=row[6],
                    expected_quality=row[7],
                    reasoning=row[8] or ""
                )
                self.board_compositions[row[0]] = composition

    async def _initialize_default_weights(self):
        """Initialize default weights for all available directors"""
        available_directors = await self.service_registry.get_available_directors()

        for director_id in available_directors:
            if director_id not in self.current_weights:
                self.current_weights[director_id] = DEFAULT_DIRECTOR_WEIGHT

                # Create default performance record
                if director_id not in self.director_performances:
                    director_info = await self.service_registry.get_director_info(director_id)
                    performance = DirectorPerformance(
                        director_id=director_id,
                        director_name=director_info.get("name", director_id),
                        current_weight=DEFAULT_DIRECTOR_WEIGHT,
                        success_rate=0.5,
                        average_confidence=0.5,
                        response_time=5.0,
                        accuracy_score=0.5,
                        consistency_score=0.5,
                        recent_decisions=0,
                        performance_level=DirectorPerformanceLevel.AVERAGE,
                        last_updated=datetime.now(),
                        strengths=[],
                        weaknesses=[],
                        recommendations=[]
                    )
                    self.director_performances[director_id] = performance

    async def _create_default_compositions(self):
        """Create default board compositions"""
        default_compositions = [
            {
                "decision_type": BoardDecisionType.CONSULTATION,
                "domain": "general",
                "required_directors": ["quality_director"],
                "optional_directors": ["performance_director", "security_director"],
                "suggested_weights": {"quality_director": 0.8, "performance_director": 0.6, "security_director": 0.5},
                "minimum_confidence": 0.6,
                "expected_quality": 0.7,
                "reasoning": "General consultation with quality focus"
            },
            {
                "decision_type": BoardDecisionType.FULL_ANALYSIS,
                "domain": "complex_problem",
                "required_directors": ["quality_director", "performance_director", "security_director"],
                "optional_directors": ["ethics_director", "ux_director"],
                "suggested_weights": {
                    "quality_director": 0.9,
                    "performance_director": 0.8,
                    "security_director": 0.8,
                    "ethics_director": 0.6,
                    "ux_director": 0.5
                },
                "minimum_confidence": 0.8,
                "expected_quality": 0.9,
                "reasoning": "Comprehensive analysis requiring multiple perspectives"
            },
            {
                "decision_type": BoardDecisionType.EXPERT_REVIEW,
                "domain": "security",
                "required_directors": ["security_director"],
                "optional_directors": ["quality_director", "ethics_director"],
                "suggested_weights": {"security_director": 1.0, "quality_director": 0.6, "ethics_director": 0.5},
                "minimum_confidence": 0.9,
                "expected_quality": 0.95,
                "reasoning": "Security-focused review with supporting perspectives"
            }
        ]

        for comp_data in default_compositions:
            composition_id = f"{comp_data['decision_type'].value}_{comp_data['domain']}"
            composition = BoardComposition(**comp_data)
            self.board_compositions[composition_id] = composition
            await self._store_board_composition(composition_id, composition)

    async def consult_board(self,
                          request: str,
                          context: Dict[str, Any],
                          decision_type: BoardDecisionType = BoardDecisionType.CONSULTATION,
                          domain: str = "general",
                          required_confidence: float = 0.7) -> Dict[str, Any]:
        """
        Consult the Board of Directors as a tool

        Args:
            request: The request or question for the board
            context: Additional context for the decision
            decision_type: Type of board decision needed
            domain: Domain of the decision
            required_confidence: Minimum confidence required

        Returns:
            Board decision with recommendations and confidence
        """
        decision_id = f"board_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request) % 10000}"

        self.logger.info(f"Consulting board for decision: {decision_id}")

        # Record this as a learning decision
        learning_decision_id = await record_decision(
            domain=LearningDomain.BOARD_MANAGEMENT,
            context={
                "request": request,
                "context": context,
                "decision_type": decision_type.value,
                "domain": domain,
                "required_confidence": required_confidence
            },
            decision_factors={
                "board_complexity": len(context.get("complexity_indicators", [])) / 5,
                "domain_specificity": 1.0 if domain != "general" else 0.5,
                "confidence_requirement": required_confidence
            },
            decision_made="board_consultation",
            confidence=0.8
        )

        try:
            # Select optimal board composition
            board_composition = await self._select_board_composition(decision_type, domain, context)

            # Get director responses
            director_responses = await self._collect_director_responses(
                request, context, board_composition
            )

            # Synthesize board decision
            board_decision = await self._synthesize_board_decision(
                director_responses, board_composition, required_confidence
            )

            # Record the board decision
            decision_record = BoardDecisionRecord(
                decision_id=decision_id,
                timestamp=datetime.now(),
                decision_type=decision_type,
                request_context={"request": request, **context},
                selected_directors=list(director_responses.keys()),
                director_weights={d: board_composition.suggested_weights.get(d, 0.5)
                                for d in director_responses.keys()},
                individual_responses=director_responses,
                consensus_reached=board_decision["consensus_reached"],
                final_recommendation=board_decision["recommendation"],
                confidence_level=board_decision["confidence"],
                echo_override=False,
                outcome=None,  # Will be set later
                performance_impact={}
            )

            self.decision_history.append(decision_record)
            await self._store_board_decision(decision_record)

            # Prepare response
            response = {
                "decision_id": decision_id,
                "learning_decision_id": learning_decision_id,
                "recommendation": board_decision["recommendation"],
                "confidence": board_decision["confidence"],
                "consensus_reached": board_decision["consensus_reached"],
                "contributing_directors": list(director_responses.keys()),
                "director_weights": decision_record.director_weights,
                "individual_insights": board_decision["insights"],
                "board_composition": {
                    "type": decision_type.value,
                    "domain": domain,
                    "directors_used": len(director_responses),
                    "expected_quality": board_composition.expected_quality
                },
                "meta_analysis": {
                    "decision_complexity": len(context.get("complexity_indicators", [])),
                    "board_agreement_level": board_decision["agreement_level"],
                    "recommendation_strength": board_decision["recommendation_strength"]
                }
            }

            self.logger.info(f"Board consultation complete: {decision_id} (confidence: {board_decision['confidence']:.2f})")
            return response

        except Exception as e:
            self.logger.error(f"Board consultation failed: {e}")

            # Record failure
            await record_outcome(
                learning_decision_id,
                DecisionOutcome.FAILURE,
                {"error": str(e), "board_available": False},
                f"Board consultation failed: {e}"
            )

            return {
                "decision_id": decision_id,
                "error": str(e),
                "fallback_recommendation": "Board consultation unavailable - Echo proceeding independently",
                "confidence": 0.3
            }

    async def _select_board_composition(self,
                                      decision_type: BoardDecisionType,
                                      domain: str,
                                      context: Dict[str, Any]) -> BoardComposition:
        """Select optimal board composition for the decision"""
        composition_key = f"{decision_type.value}_{domain}"

        # Try to find exact match
        if composition_key in self.board_compositions:
            return self.board_compositions[composition_key]

        # Try to find by decision type
        for comp_id, composition in self.board_compositions.items():
            if composition.decision_type == decision_type:
                return composition

        # Create dynamic composition
        return await self._create_dynamic_composition(decision_type, domain, context)

    async def _create_dynamic_composition(self,
                                        decision_type: BoardDecisionType,
                                        domain: str,
                                        context: Dict[str, Any]) -> BoardComposition:
        """Create a dynamic board composition based on the context"""
        available_directors = await self.service_registry.get_available_directors()

        # Analyze context to determine best directors
        required_directors = []
        optional_directors = []
        suggested_weights = {}

        # Always include quality director for oversight
        if "quality_director" in available_directors:
            required_directors.append("quality_director")
            suggested_weights["quality_director"] = 0.8

        # Add domain-specific directors
        domain_mapping = {
            "security": "security_director",
            "performance": "performance_director",
            "ethics": "ethics_director",
            "user_experience": "ux_director"
        }

        if domain in domain_mapping and domain_mapping[domain] in available_directors:
            required_directors.append(domain_mapping[domain])
            suggested_weights[domain_mapping[domain]] = 0.9

        # Add based on complexity
        complexity_indicators = context.get("complexity_indicators", [])
        if len(complexity_indicators) > 3:
            # High complexity - need more directors
            for director in available_directors:
                if director not in required_directors:
                    optional_directors.append(director)
                    suggested_weights[director] = 0.6

        # Adjust weights based on director performance
        for director_id in required_directors + optional_directors:
            if director_id in self.director_performances:
                performance = self.director_performances[director_id]
                # Boost weight for high-performing directors
                if performance.performance_level in [DirectorPerformanceLevel.EXCELLENT, DirectorPerformanceLevel.GOOD]:
                    suggested_weights[director_id] = min(1.0, suggested_weights.get(director_id, 0.5) + 0.2)

        composition = BoardComposition(
            decision_type=decision_type,
            domain=domain,
            required_directors=required_directors,
            optional_directors=optional_directors[:3],  # Limit to top 3 optional
            suggested_weights=suggested_weights,
            minimum_confidence=0.7 if decision_type == BoardDecisionType.FULL_ANALYSIS else 0.6,
            expected_quality=0.8 if len(required_directors) > 1 else 0.7,
            reasoning=f"Dynamic composition for {decision_type.value} in {domain} domain"
        )

        return composition

    async def _collect_director_responses(self,
                                        request: str,
                                        context: Dict[str, Any],
                                        composition: BoardComposition) -> Dict[str, Any]:
        """Collect responses from selected directors"""
        director_responses = {}

        # Collect from required directors
        for director_id in composition.required_directors:
            try:
                response = await self._get_director_response(director_id, request, context)
                if response:
                    director_responses[director_id] = response
                    await self._record_director_activity(director_id, response)
            except Exception as e:
                self.logger.error(f"Error getting response from {director_id}: {e}")

        # Collect from optional directors if needed
        if len(director_responses) < 2:  # Ensure minimum board size
            for director_id in composition.optional_directors:
                if len(director_responses) >= 3:  # Reasonable maximum
                    break
                try:
                    response = await self._get_director_response(director_id, request, context)
                    if response:
                        director_responses[director_id] = response
                        await self._record_director_activity(director_id, response)
                except Exception as e:
                    self.logger.error(f"Error getting response from optional director {director_id}: {e}")

        return director_responses

    async def _get_director_response(self,
                                   director_id: str,
                                   request: str,
                                   context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get response from a specific director"""
        try:
            # This would interface with the actual director system
            # For now, simulate director response
            director = await self.service_registry.get_director(director_id)
            if not director:
                return None

            # Use the actual director's analyze method
            start_time = datetime.now()
            response = await director.analyze(request, context)
            response_time = (datetime.now() - start_time).total_seconds()

            return {
                "director_id": director_id,
                "response": response,
                "confidence": getattr(response, 'confidence', 0.7),
                "response_time": response_time,
                "timestamp": start_time.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting response from director {director_id}: {e}")
            return None

    async def _synthesize_board_decision(self,
                                       director_responses: Dict[str, Any],
                                       composition: BoardComposition,
                                       required_confidence: float) -> Dict[str, Any]:
        """Synthesize individual director responses into board decision"""
        if not director_responses:
            return {
                "recommendation": "No board input available",
                "confidence": 0.1,
                "consensus_reached": False,
                "insights": [],
                "agreement_level": 0.0,
                "recommendation_strength": 0.1
            }

        # Extract key information from responses
        recommendations = []
        confidences = []
        insights = []

        for director_id, response_data in director_responses.items():
            response = response_data["response"]
            director_weight = composition.suggested_weights.get(director_id, 0.5)

            # Extract recommendation
            recommendation = self._extract_recommendation(response)
            if recommendation:
                recommendations.append({
                    "director": director_id,
                    "recommendation": recommendation,
                    "weight": director_weight,
                    "confidence": response_data["confidence"]
                })
                confidences.append(response_data["confidence"] * director_weight)

            # Extract insights
            insight = self._extract_insight(director_id, response)
            if insight:
                insights.append(insight)

        # Calculate consensus
        consensus_reached = self._calculate_consensus(recommendations)
        agreement_level = self._calculate_agreement_level(recommendations)

        # Synthesize final recommendation
        final_recommendation = self._synthesize_recommendation(recommendations, composition)

        # Calculate overall confidence
        if confidences:
            weighted_confidence = sum(confidences) / sum(composition.suggested_weights.values())
        else:
            weighted_confidence = 0.1

        recommendation_strength = min(1.0, weighted_confidence * agreement_level)

        return {
            "recommendation": final_recommendation,
            "confidence": weighted_confidence,
            "consensus_reached": consensus_reached,
            "insights": insights,
            "agreement_level": agreement_level,
            "recommendation_strength": recommendation_strength,
            "individual_recommendations": recommendations
        }

    def _extract_recommendation(self, response: Any) -> Optional[str]:
        """Extract actionable recommendation from director response"""
        # This would parse the actual director response format
        if hasattr(response, 'recommendation'):
            return response.recommendation
        elif hasattr(response, 'analysis') and 'recommendation' in response.analysis:
            return response.analysis['recommendation']
        elif isinstance(response, dict) and 'recommendation' in response:
            return response['recommendation']
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def _extract_insight(self, director_id: str, response: Any) -> Optional[Dict[str, Any]]:
        """Extract insight from director response"""
        recommendation = self._extract_recommendation(response)
        if recommendation:
            return {
                "director": director_id,
                "insight": recommendation,
                "director_perspective": self.director_performances.get(director_id, {}).director_name or director_id
            }
        return None

    def _calculate_consensus(self, recommendations: List[Dict[str, Any]]) -> bool:
        """Calculate if consensus was reached among directors"""
        if len(recommendations) < 2:
            return True

        # Simple consensus: majority agreement or high confidence overlap
        recommendation_texts = [r["recommendation"] for r in recommendations]

        # Check for keyword overlap (simplified consensus detection)
        consensus_keywords = []
        for rec in recommendation_texts:
            words = rec.lower().split()
            consensus_keywords.extend(words)

        # If any words appear in majority of recommendations, consider consensus
        word_counts = {}
        for word in consensus_keywords:
            word_counts[word] = word_counts.get(word, 0) + 1

        majority_threshold = len(recommendations) / 2
        consensus_words = [word for word, count in word_counts.items() if count > majority_threshold]

        return len(consensus_words) > 0

    def _calculate_agreement_level(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate numerical agreement level among directors"""
        if len(recommendations) < 2:
            return 1.0

        # Calculate based on confidence overlap and recommendation similarity
        confidences = [r["confidence"] for r in recommendations]
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0

        # Lower variance = higher agreement
        confidence_agreement = max(0.0, 1.0 - confidence_variance)

        # Factor in weights
        weighted_confidences = [r["confidence"] * r["weight"] for r in recommendations]
        if weighted_confidences:
            weighted_average = statistics.mean(weighted_confidences)
            return min(1.0, confidence_agreement * weighted_average)
        else:
            return confidence_agreement

    def _synthesize_recommendation(self,
                                 recommendations: List[Dict[str, Any]],
                                 composition: BoardComposition) -> str:
        """Synthesize final recommendation from individual recommendations"""
        if not recommendations:
            return "No recommendations available from board"

        if len(recommendations) == 1:
            return recommendations[0]["recommendation"]

        # Weight recommendations by director performance and assigned weights
        weighted_recommendations = []
        for rec in recommendations:
            director_id = rec["director"]
            performance = self.director_performances.get(director_id)

            # Calculate combined weight
            base_weight = rec["weight"]
            performance_multiplier = 1.0

            if performance:
                if performance.performance_level == DirectorPerformanceLevel.EXCELLENT:
                    performance_multiplier = 1.3
                elif performance.performance_level == DirectorPerformanceLevel.GOOD:
                    performance_multiplier = 1.1
                elif performance.performance_level == DirectorPerformanceLevel.POOR:
                    performance_multiplier = 0.8
                elif performance.performance_level == DirectorPerformanceLevel.CRITICAL:
                    performance_multiplier = 0.5

            final_weight = base_weight * performance_multiplier * rec["confidence"]
            weighted_recommendations.append((rec["recommendation"], final_weight))

        # Sort by weight and synthesize
        weighted_recommendations.sort(key=lambda x: x[1], reverse=True)

        if len(weighted_recommendations) == 1:
            return weighted_recommendations[0][0]

        # Combine top recommendations
        top_rec = weighted_recommendations[0][0]
        second_rec = weighted_recommendations[1][0] if len(weighted_recommendations) > 1 else ""

        if weighted_recommendations[0][1] > weighted_recommendations[1][1] * 2:
            # Clear winner
            return top_rec
        else:
            # Synthesize multiple perspectives
            return f"{top_rec} Additionally, consider: {second_rec}"

    async def record_board_outcome(self,
                                 decision_id: str,
                                 outcome: DecisionOutcome,
                                 outcome_metrics: Dict[str, float],
                                 user_feedback: Optional[str] = None):
        """Record the outcome of a board decision for learning"""
        # Find the decision record
        decision_record = None
        for record in self.decision_history:
            if record.decision_id == decision_id:
                decision_record = record
                break

        if not decision_record:
            self.logger.error(f"Board decision {decision_id} not found")
            return

        # Update the record
        decision_record.outcome = outcome
        decision_record.performance_impact = outcome_metrics

        # Record learning outcome
        if hasattr(decision_record, 'learning_decision_id'):
            await record_outcome(
                decision_record.learning_decision_id,
                outcome,
                outcome_metrics,
                user_feedback
            )

        # Update director performances
        await self._update_director_performances(decision_record, outcome, outcome_metrics)

        # Store updated decision
        await self._store_board_decision(decision_record)

        self.logger.info(f"Board decision outcome recorded: {decision_id} -> {outcome.value}")

    async def _update_director_performances(self,
                                          decision_record: BoardDecisionRecord,
                                          outcome: DecisionOutcome,
                                          outcome_metrics: Dict[str, float]):
        """Update individual director performances based on decision outcome"""
        success = outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL]

        for director_id in decision_record.selected_directors:
            if director_id in self.director_performances:
                performance = self.director_performances[director_id]

                # Update success rate
                old_success_rate = performance.success_rate
                new_success = 1.0 if success else 0.0
                alpha = 0.1  # Learning rate
                performance.success_rate = alpha * new_success + (1 - alpha) * old_success_rate

                # Update accuracy score based on outcome metrics
                if "accuracy" in outcome_metrics:
                    performance.accuracy_score = alpha * outcome_metrics["accuracy"] + (1 - alpha) * performance.accuracy_score

                # Update recent decisions count
                performance.recent_decisions += 1

                # Recalculate performance level
                performance.performance_level = self._calculate_performance_level(performance)

                # Update weight if needed
                await self._adjust_director_weight(director_id, performance, success)

                # Update strengths/weaknesses
                await self._update_director_insights(director_id, performance, decision_record, success)

                # Store updated performance
                await self._store_director_performance(performance)

    def _calculate_performance_level(self, performance: DirectorPerformance) -> DirectorPerformanceLevel:
        """Calculate overall performance level for a director"""
        # Combine multiple metrics
        overall_score = (
            performance.success_rate * 0.4 +
            performance.accuracy_score * 0.3 +
            performance.consistency_score * 0.2 +
            min(1.0, performance.average_confidence) * 0.1
        )

        if overall_score >= 0.9:
            return DirectorPerformanceLevel.EXCELLENT
        elif overall_score >= 0.75:
            return DirectorPerformanceLevel.GOOD
        elif overall_score >= 0.5:
            return DirectorPerformanceLevel.AVERAGE
        elif overall_score >= 0.3:
            return DirectorPerformanceLevel.POOR
        else:
            return DirectorPerformanceLevel.CRITICAL

    async def _adjust_director_weight(self,
                                    director_id: str,
                                    performance: DirectorPerformance,
                                    success: bool):
        """Adjust director weight based on performance"""
        if performance.recent_decisions < MIN_DECISIONS_FOR_TUNING:
            return

        current_weight = self.current_weights.get(director_id, DEFAULT_DIRECTOR_WEIGHT)
        new_weight = current_weight

        # Adjust based on success rate
        if performance.success_rate > 0.8:
            new_weight = min(1.0, current_weight + WEIGHT_ADJUSTMENT_RATE)
        elif performance.success_rate < 0.4:
            new_weight = max(0.1, current_weight - WEIGHT_ADJUSTMENT_RATE)

        # Adjust based on recent performance
        if success and performance.performance_level in [DirectorPerformanceLevel.EXCELLENT, DirectorPerformanceLevel.GOOD]:
            new_weight = min(1.0, current_weight + WEIGHT_ADJUSTMENT_RATE * 0.5)
        elif not success and performance.performance_level in [DirectorPerformanceLevel.POOR, DirectorPerformanceLevel.CRITICAL]:
            new_weight = max(0.1, current_weight - WEIGHT_ADJUSTMENT_RATE * 0.5)

        if abs(new_weight - current_weight) > 0.01:  # Meaningful change
            old_weight = current_weight
            self.current_weights[director_id] = new_weight
            performance.current_weight = new_weight

            # Record weight adjustment
            await self._record_weight_adjustment(director_id, old_weight, new_weight, performance, success)

            self.logger.info(f"Adjusted weight for {director_id}: {old_weight:.2f} -> {new_weight:.2f}")

    async def _record_weight_adjustment(self,
                                      director_id: str,
                                      old_weight: float,
                                      new_weight: float,
                                      performance: DirectorPerformance,
                                      success: bool):
        """Record weight adjustment in database"""
        adjustment_id = f"adj_{director_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        reason = f"Performance-based adjustment: success_rate={performance.success_rate:.2f}, recent_success={success}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO weight_adjustments
                (adjustment_id, director_id, old_weight, new_weight, reason, timestamp, performance_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                adjustment_id, director_id, old_weight, new_weight, reason,
                datetime.now().isoformat(), json.dumps(asdict(performance), default=str)
            ))
            conn.commit()

    async def _update_director_insights(self,
                                      director_id: str,
                                      performance: DirectorPerformance,
                                      decision_record: BoardDecisionRecord,
                                      success: bool):
        """Update director strengths and weaknesses"""
        # Analyze decision context to identify strengths/weaknesses
        context = decision_record.request_context
        decision_type = decision_record.decision_type

        # Update strengths
        if success:
            strength = f"Effective in {decision_type.value} decisions"
            if strength not in performance.strengths:
                performance.strengths.append(strength)

            # Domain-specific strengths
            if "domain" in context:
                domain_strength = f"Strong performance in {context['domain']} domain"
                if domain_strength not in performance.strengths:
                    performance.strengths.append(domain_strength)

        # Update weaknesses
        else:
            weakness = f"Needs improvement in {decision_type.value} decisions"
            if weakness not in performance.weaknesses:
                performance.weaknesses.append(weakness)

        # Limit list sizes
        performance.strengths = performance.strengths[-5:]  # Keep latest 5
        performance.weaknesses = performance.weaknesses[-5:]  # Keep latest 5

        # Generate recommendations
        performance.recommendations = self._generate_director_recommendations(performance)

    def _generate_director_recommendations(self, performance: DirectorPerformance) -> List[str]:
        """Generate recommendations for improving director performance"""
        recommendations = []

        if performance.success_rate < 0.6:
            recommendations.append("Focus on improving decision accuracy through additional training")

        if performance.consistency_score < 0.5:
            recommendations.append("Work on providing more consistent analysis across similar decisions")

        if performance.response_time > 10.0:
            recommendations.append("Optimize response time for better board efficiency")

        if performance.average_confidence < 0.5:
            recommendations.append("Build confidence through domain expertise development")

        if performance.performance_level == DirectorPerformanceLevel.CRITICAL:
            recommendations.append("Consider performance review and potential role adjustment")

        return recommendations[:3]  # Limit to top 3

    async def _record_director_activity(self, director_id: str, response_data: Dict[str, Any]):
        """Record director activity for performance tracking"""
        if director_id in self.director_performances:
            performance = self.director_performances[director_id]

            # Update response time
            response_time = response_data.get("response_time", 5.0)
            alpha = 0.1
            performance.response_time = alpha * response_time + (1 - alpha) * performance.response_time

            # Update confidence
            confidence = response_data.get("confidence", 0.5)
            performance.average_confidence = alpha * confidence + (1 - alpha) * performance.average_confidence

            performance.last_updated = datetime.now()

    # Public API methods

    async def get_board_status(self) -> Dict[str, Any]:
        """Get comprehensive board status"""
        status = {
            "total_directors": len(self.director_performances),
            "active_directors": len([p for p in self.director_performances.values()
                                   if p.performance_level != DirectorPerformanceLevel.CRITICAL]),
            "average_performance": 0.0,
            "recent_decisions": len([d for d in self.decision_history
                                   if d.timestamp > datetime.now() - timedelta(days=7)]),
            "director_summary": {},
            "board_compositions": len(self.board_compositions),
            "system_health": "good"
        }

        if self.director_performances:
            performance_scores = []
            for performance in self.director_performances.values():
                score = (performance.success_rate + performance.accuracy_score + performance.consistency_score) / 3
                performance_scores.append(score)

                status["director_summary"][performance.director_id] = {
                    "name": performance.director_name,
                    "performance_level": performance.performance_level.value,
                    "current_weight": performance.current_weight,
                    "success_rate": performance.success_rate,
                    "recent_decisions": performance.recent_decisions
                }

            status["average_performance"] = statistics.mean(performance_scores)

            # Determine system health
            excellent_count = len([p for p in self.director_performances.values()
                                 if p.performance_level == DirectorPerformanceLevel.EXCELLENT])
            critical_count = len([p for p in self.director_performances.values()
                                if p.performance_level == DirectorPerformanceLevel.CRITICAL])

            if critical_count > len(self.director_performances) * 0.3:
                status["system_health"] = "critical"
            elif excellent_count > len(self.director_performances) * 0.5:
                status["system_health"] = "excellent"
            elif status["average_performance"] > 0.7:
                status["system_health"] = "good"
            else:
                status["system_health"] = "degraded"

        return status

    async def get_director_performance(self, director_id: str) -> Optional[DirectorPerformance]:
        """Get detailed performance data for a specific director"""
        return self.director_performances.get(director_id)

    async def optimize_board_composition(self, decision_type: BoardDecisionType, domain: str) -> BoardComposition:
        """Optimize board composition based on historical performance"""
        # Analyze historical decisions of this type
        relevant_decisions = [
            d for d in self.decision_history
            if d.decision_type == decision_type and d.outcome is not None
        ]

        if not relevant_decisions:
            # Use default composition
            return await self._create_dynamic_composition(decision_type, domain, {})

        # Analyze which directors performed best
        director_success_rates = {}
        for decision in relevant_decisions:
            success = decision.outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL]
            for director_id in decision.selected_directors:
                if director_id not in director_success_rates:
                    director_success_rates[director_id] = []
                director_success_rates[director_id].append(success)

        # Calculate success rates
        optimized_directors = {}
        for director_id, successes in director_success_rates.items():
            if len(successes) >= 3:  # Minimum sample size
                success_rate = sum(successes) / len(successes)
                optimized_directors[director_id] = success_rate

        # Sort by performance
        top_directors = sorted(optimized_directors.items(), key=lambda x: x[1], reverse=True)

        # Create optimized composition
        required_directors = [d[0] for d in top_directors[:2]]  # Top 2 performers
        optional_directors = [d[0] for d in top_directors[2:5]]  # Next 3 performers

        suggested_weights = {}
        for director_id, success_rate in top_directors[:5]:
            suggested_weights[director_id] = min(1.0, success_rate + 0.1)

        return BoardComposition(
            decision_type=decision_type,
            domain=domain,
            required_directors=required_directors,
            optional_directors=optional_directors,
            suggested_weights=suggested_weights,
            minimum_confidence=0.8,
            expected_quality=statistics.mean([sr for _, sr in top_directors[:3]]) if top_directors else 0.7,
            reasoning=f"Optimized based on {len(relevant_decisions)} historical decisions"
        )

    async def reset_director_performance(self, director_id: str) -> bool:
        """Reset a director's performance to defaults"""
        if director_id not in self.director_performances:
            return False

        # Reset to defaults
        director_info = await self.service_registry.get_director_info(director_id)
        performance = DirectorPerformance(
            director_id=director_id,
            director_name=director_info.get("name", director_id),
            current_weight=DEFAULT_DIRECTOR_WEIGHT,
            success_rate=0.5,
            average_confidence=0.5,
            response_time=5.0,
            accuracy_score=0.5,
            consistency_score=0.5,
            recent_decisions=0,
            performance_level=DirectorPerformanceLevel.AVERAGE,
            last_updated=datetime.now(),
            strengths=[],
            weaknesses=[],
            recommendations=[]
        )

        self.director_performances[director_id] = performance
        self.current_weights[director_id] = DEFAULT_DIRECTOR_WEIGHT

        await self._store_director_performance(performance)

        self.logger.info(f"Reset performance for director {director_id}")
        return True

    # Storage methods

    async def _store_director_performance(self, performance: DirectorPerformance):
        """Store director performance in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO director_performance
                (director_id, director_name, current_weight, success_rate, average_confidence,
                 response_time, accuracy_score, consistency_score, recent_decisions,
                 performance_level, last_updated, strengths, weaknesses, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                performance.director_id, performance.director_name, performance.current_weight,
                performance.success_rate, performance.average_confidence, performance.response_time,
                performance.accuracy_score, performance.consistency_score, performance.recent_decisions,
                performance.performance_level.value, performance.last_updated.isoformat(),
                json.dumps(performance.strengths), json.dumps(performance.weaknesses),
                json.dumps(performance.recommendations)
            ))
            conn.commit()

    async def _store_board_composition(self, composition_id: str, composition: BoardComposition):
        """Store board composition in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO board_compositions
                (composition_id, decision_type, domain, required_directors, optional_directors,
                 suggested_weights, minimum_confidence, expected_quality, reasoning, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                composition_id, composition.decision_type.value, composition.domain,
                json.dumps(composition.required_directors), json.dumps(composition.optional_directors),
                json.dumps(composition.suggested_weights), composition.minimum_confidence,
                composition.expected_quality, composition.reasoning, datetime.now().isoformat()
            ))
            conn.commit()

    async def _store_board_decision(self, decision: BoardDecisionRecord):
        """Store board decision in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO board_decisions
                (decision_id, timestamp, decision_type, request_context, selected_directors,
                 director_weights, individual_responses, consensus_reached, final_recommendation,
                 confidence_level, echo_override, outcome, performance_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.decision_id, decision.timestamp.isoformat(), decision.decision_type.value,
                json.dumps(decision.request_context), json.dumps(decision.selected_directors),
                json.dumps(decision.director_weights), json.dumps(decision.individual_responses, default=str),
                decision.consensus_reached, decision.final_recommendation, decision.confidence_level,
                decision.echo_override, decision.outcome.value if decision.outcome else None,
                json.dumps(decision.performance_impact)
            ))
            conn.commit()

# Global instance
_board_manager = None

def get_board_manager() -> EchoBoardManager:
    """Get global board manager instance"""
    global _board_manager
    if _board_manager is None:
        _board_manager = EchoBoardManager()
    return _board_manager

# Convenience functions
async def consult_board(request: str,
                       context: Dict[str, Any],
                       decision_type: BoardDecisionType = BoardDecisionType.CONSULTATION,
                       domain: str = "general",
                       required_confidence: float = 0.7) -> Dict[str, Any]:
    """Convenience function to consult the board"""
    manager = get_board_manager()
    return await manager.consult_board(request, context, decision_type, domain, required_confidence)

async def record_board_outcome(decision_id: str,
                              outcome: DecisionOutcome,
                              outcome_metrics: Dict[str, float],
                              user_feedback: Optional[str] = None):
    """Convenience function to record board decision outcome"""
    manager = get_board_manager()
    await manager.record_board_outcome(decision_id, outcome, outcome_metrics, user_feedback)

async def get_board_status() -> Dict[str, Any]:
    """Convenience function to get board status"""
    manager = get_board_manager()
    return await manager.get_board_status()

if __name__ == "__main__":
    # Example usage
    async def example():
        manager = get_board_manager()
        await manager.initialize_board()

        # Consult board on a complex decision
        result = await manager.consult_board(
            request="Should we implement a new caching strategy for improved performance?",
            context={
                "complexity_indicators": ["performance", "architecture", "user_impact"],
                "domain": "performance",
                "urgency": "medium",
                "resources_available": True
            },
            decision_type=BoardDecisionType.FULL_ANALYSIS,
            domain="performance",
            required_confidence=0.8
        )

        print("Board Decision:", json.dumps(result, indent=2))

        # Later, record the outcome
        await manager.record_board_outcome(
            result["decision_id"],
            DecisionOutcome.SUCCESS,
            {"implementation_success": 0.9, "performance_improvement": 0.8, "user_satisfaction": 0.85},
            "Implementation went smoothly and delivered expected benefits"
        )

        # Get board status
        status = await manager.get_board_status()
        print("Board Status:", json.dumps(status, indent=2))

    asyncio.run(example())