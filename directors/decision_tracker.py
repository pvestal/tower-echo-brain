#!/usr/bin/env python3
"""
Decision Tracker for Echo Brain Board of Directors
Implements transparent decision tracking with audit trails and confidence scoring
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger(__name__)

class DecisionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    OVERRIDDEN = "overridden"
    APPEALED = "appealed"

class EvidenceType(Enum):
    PERFORMANCE_DATA = "performance_data"
    SECURITY_ANALYSIS = "security_analysis"
    QUALITY_METRICS = "quality_metrics"
    USER_FEEDBACK = "user_feedback"
    HISTORICAL_PATTERN = "historical_pattern"
    RESOURCE_IMPACT = "resource_impact"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class Evidence:
    """Single piece of evidence supporting a decision"""
    id: str
    type: EvidenceType
    source: str  # Which director provided this
    weight: float  # 0.0 to 1.0
    data: Dict[str, Any]
    timestamp: datetime
    confidence: float
    reasoning: str

@dataclass
class DirectorEvaluation:
    """Individual director's evaluation of a task"""
    director_id: str
    director_name: str
    recommendation: str  # approve, reject, modify, escalate
    confidence: float  # 0.0 to 1.0
    reasoning: str
    evidence: List[Evidence]
    processing_time: float
    risk_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class DecisionPoint:
    """Single decision point in the timeline"""
    id: str
    task_id: str
    timestamp: datetime
    status: DecisionStatus
    description: str
    director_evaluations: List[DirectorEvaluation]
    consensus_score: float
    total_confidence: float
    evidence_summary: Dict[str, Any]
    user_feedback: Optional[str] = None

@dataclass
class TaskDecision:
    """Complete decision record for a task"""
    task_id: str
    user_id: str
    original_request: str
    submitted_at: datetime
    status: DecisionStatus
    final_recommendation: str
    consensus_score: float
    confidence_score: float
    decision_points: List[DecisionPoint]
    total_processing_time: float
    evidence_count: int
    director_participation: List[str]
    override_history: List[Dict[str, Any]]
    completion_timestamp: Optional[datetime] = None

class DecisionTracker:
    """
    Comprehensive decision tracking system for the Board of Directors
    Provides full audit trail, confidence scoring, and evidence management
    """

    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize DecisionTracker with database connection

        Args:
            db_config: Database connection parameters
        """
        self.db_config = db_config
        self.active_tasks: Dict[str, TaskDecision] = {}
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database tables if they don't exist"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Create board decision tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS board_tasks (
                    task_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    original_request TEXT NOT NULL,
                    submitted_at TIMESTAMP DEFAULT NOW(),
                    status VARCHAR(50) NOT NULL,
                    final_recommendation TEXT,
                    consensus_score FLOAT,
                    confidence_score FLOAT,
                    total_processing_time FLOAT,
                    evidence_count INTEGER,
                    director_participation TEXT[],
                    completion_timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS director_evaluations (
                    evaluation_id VARCHAR(255) PRIMARY KEY,
                    task_id VARCHAR(255) REFERENCES board_tasks(task_id),
                    decision_point_id VARCHAR(255),
                    director_id VARCHAR(255) NOT NULL,
                    director_name VARCHAR(255) NOT NULL,
                    recommendation VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    reasoning TEXT,
                    processing_time FLOAT,
                    risk_score FLOAT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_evidence (
                    evidence_id VARCHAR(255) PRIMARY KEY,
                    evaluation_id VARCHAR(255) REFERENCES director_evaluations(evaluation_id),
                    evidence_type VARCHAR(100) NOT NULL,
                    source VARCHAR(255) NOT NULL,
                    weight FLOAT NOT NULL,
                    confidence FLOAT NOT NULL,
                    reasoning TEXT,
                    data JSONB,
                    timestamp TIMESTAMP DEFAULT NOW()
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_timeline (
                    point_id VARCHAR(255) PRIMARY KEY,
                    task_id VARCHAR(255) REFERENCES board_tasks(task_id),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    status VARCHAR(50) NOT NULL,
                    description TEXT,
                    consensus_score FLOAT,
                    total_confidence FLOAT,
                    evidence_summary JSONB,
                    user_feedback TEXT
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_overrides (
                    override_id VARCHAR(255) PRIMARY KEY,
                    task_id VARCHAR(255) REFERENCES board_tasks(task_id),
                    user_id VARCHAR(255) NOT NULL,
                    override_type VARCHAR(50) NOT NULL, -- approve, reject, modify
                    original_recommendation TEXT,
                    new_recommendation TEXT,
                    reasoning TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                );
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_board_tasks_user_id ON board_tasks(user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_board_tasks_status ON board_tasks(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_director_evaluations_task_id ON director_evaluations(task_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision_timeline_task_id ON decision_timeline(task_id);")

            conn.close()
            logger.info("Decision tracker database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize decision tracker database: {e}")
            raise

    def start_task_tracking(self, task_id: str, user_id: str, original_request: str) -> TaskDecision:
        """
        Start tracking a new task decision

        Args:
            task_id: Unique task identifier
            user_id: User who submitted the task
            original_request: Original task description

        Returns:
            TaskDecision: New task decision record
        """
        task_decision = TaskDecision(
            task_id=task_id,
            user_id=user_id,
            original_request=original_request,
            submitted_at=datetime.utcnow(),
            status=DecisionStatus.PENDING,
            final_recommendation="",
            consensus_score=0.0,
            confidence_score=0.0,
            decision_points=[],
            total_processing_time=0.0,
            evidence_count=0,
            director_participation=[],
            override_history=[]
        )

        self.active_tasks[task_id] = task_decision

        # Persist to database
        self._save_task_to_db(task_decision)

        # Create initial decision point
        initial_point = DecisionPoint(
            id=str(uuid.uuid4()),
            task_id=task_id,
            timestamp=datetime.utcnow(),
            status=DecisionStatus.PENDING,
            description="Task submitted for board evaluation",
            director_evaluations=[],
            consensus_score=0.0,
            total_confidence=0.0,
            evidence_summary={}
        )

        self.add_decision_point(task_id, initial_point)

        logger.info(f"Started tracking task {task_id} for user {user_id}")
        return task_decision

    def add_director_evaluation(self, task_id: str, evaluation: DirectorEvaluation) -> bool:
        """
        Add a director's evaluation to the decision tracking

        Args:
            task_id: Task being evaluated
            evaluation: Director's evaluation

        Returns:
            bool: True if successfully added
        """
        if task_id not in self.active_tasks:
            logger.error(f"Task {task_id} not found in active tracking")
            return False

        # Add to current decision point or create new one
        current_point = self._get_latest_decision_point(task_id)
        if not current_point:
            # Create new decision point for this evaluation
            current_point = DecisionPoint(
                id=str(uuid.uuid4()),
                task_id=task_id,
                timestamp=datetime.utcnow(),
                status=DecisionStatus.IN_PROGRESS,
                description=f"Director evaluation from {evaluation.director_name}",
                director_evaluations=[],
                consensus_score=0.0,
                total_confidence=0.0,
                evidence_summary={}
            )

        current_point.director_evaluations.append(evaluation)

        # Update task participation
        task = self.active_tasks[task_id]
        if evaluation.director_id not in task.director_participation:
            task.director_participation.append(evaluation.director_id)

        # Update evidence count
        task.evidence_count += len(evaluation.evidence)

        # Recalculate consensus and confidence
        self._recalculate_scores(current_point)

        # Save to database
        self._save_evaluation_to_db(evaluation, current_point.id)

        logger.info(f"Added evaluation from {evaluation.director_name} for task {task_id}")
        return True

    def add_decision_point(self, task_id: str, decision_point: DecisionPoint) -> bool:
        """
        Add a decision point to the timeline

        Args:
            task_id: Task identifier
            decision_point: Decision point to add

        Returns:
            bool: True if successfully added
        """
        if task_id not in self.active_tasks:
            logger.error(f"Task {task_id} not found in active tracking")
            return False

        self.active_tasks[task_id].decision_points.append(decision_point)

        # Update task status if this is a status change
        if decision_point.status != self.active_tasks[task_id].status:
            self.active_tasks[task_id].status = decision_point.status

        # Save to database
        self._save_decision_point_to_db(decision_point)

        return True

    def finalize_task_decision(self, task_id: str, final_recommendation: str) -> TaskDecision:
        """
        Finalize a task decision with final recommendation

        Args:
            task_id: Task identifier
            final_recommendation: Final board recommendation

        Returns:
            TaskDecision: Completed task decision
        """
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found in active tracking")

        task = self.active_tasks[task_id]
        task.status = DecisionStatus.COMPLETED
        task.final_recommendation = final_recommendation
        task.completion_timestamp = datetime.utcnow()

        # Calculate final metrics
        self._calculate_final_metrics(task)

        # Create final decision point
        final_point = DecisionPoint(
            id=str(uuid.uuid4()),
            task_id=task_id,
            timestamp=datetime.utcnow(),
            status=DecisionStatus.COMPLETED,
            description=f"Task completed with recommendation: {final_recommendation}",
            director_evaluations=[],
            consensus_score=task.consensus_score,
            total_confidence=task.confidence_score,
            evidence_summary=self._generate_evidence_summary(task)
        )

        self.add_decision_point(task_id, final_point)

        # Update database
        self._update_task_in_db(task)

        # Remove from active tracking
        completed_task = self.active_tasks.pop(task_id)

        logger.info(f"Finalized decision for task {task_id}: {final_recommendation}")
        return completed_task

    def get_task_decision(self, task_id: str) -> Optional[TaskDecision]:
        """Get current state of a task decision"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]

        # Try to load from database
        return self._load_task_from_db(task_id)

    def get_decision_timeline(self, task_id: str) -> List[DecisionPoint]:
        """Get complete decision timeline for a task"""
        task = self.get_task_decision(task_id)
        if task:
            return task.decision_points
        return []

    def add_user_override(self, task_id: str, user_id: str, override_type: str,
                         original_rec: str, new_rec: str, reasoning: str) -> bool:
        """
        Record user override of board decision

        Args:
            task_id: Task identifier
            user_id: User making override
            override_type: Type of override (approve, reject, modify)
            original_rec: Original board recommendation
            new_rec: New user recommendation
            reasoning: User's reasoning

        Returns:
            bool: True if successfully recorded
        """
        if task_id not in self.active_tasks:
            logger.error(f"Task {task_id} not found for override")
            return False

        override_record = {
            "override_id": str(uuid.uuid4()),
            "user_id": user_id,
            "override_type": override_type,
            "original_recommendation": original_rec,
            "new_recommendation": new_rec,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat()
        }

        task = self.active_tasks[task_id]
        task.override_history.append(override_record)
        task.status = DecisionStatus.OVERRIDDEN

        # Create override decision point
        override_point = DecisionPoint(
            id=str(uuid.uuid4()),
            task_id=task_id,
            timestamp=datetime.utcnow(),
            status=DecisionStatus.OVERRIDDEN,
            description=f"User override: {override_type}",
            director_evaluations=[],
            consensus_score=task.consensus_score,
            total_confidence=task.confidence_score,
            evidence_summary={},
            user_feedback=reasoning
        )

        self.add_decision_point(task_id, override_point)

        # Save override to database
        self._save_override_to_db(override_record, task_id)

        logger.info(f"User {user_id} overrode task {task_id}: {override_type}")
        return True

    def get_board_analytics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Get analytics about board decision making

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            Dict containing analytics data
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Overall task statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_tasks,
                    AVG(consensus_score) as avg_consensus,
                    AVG(confidence_score) as avg_confidence,
                    AVG(total_processing_time) as avg_processing_time,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
                    COUNT(CASE WHEN status = 'overridden' THEN 1 END) as overridden_tasks
                FROM board_tasks
                WHERE submitted_at BETWEEN %s AND %s
            """, (start_date, end_date))

            stats = cursor.fetchone()

            # Director participation
            cursor.execute("""
                SELECT
                    director_name,
                    COUNT(*) as evaluation_count,
                    AVG(confidence) as avg_confidence,
                    AVG(risk_score) as avg_risk_score,
                    COUNT(CASE WHEN recommendation = 'approve' THEN 1 END) as approvals,
                    COUNT(CASE WHEN recommendation = 'reject' THEN 1 END) as rejections
                FROM director_evaluations de
                JOIN board_tasks bt ON de.task_id = bt.task_id
                WHERE bt.submitted_at BETWEEN %s AND %s
                GROUP BY director_name
                ORDER BY evaluation_count DESC
            """, (start_date, end_date))

            director_stats = cursor.fetchall()

            # Evidence type distribution
            cursor.execute("""
                SELECT
                    evidence_type,
                    COUNT(*) as count,
                    AVG(weight) as avg_weight,
                    AVG(confidence) as avg_confidence
                FROM decision_evidence de
                JOIN director_evaluations eval ON de.evaluation_id = eval.evaluation_id
                JOIN board_tasks bt ON eval.task_id = bt.task_id
                WHERE bt.submitted_at BETWEEN %s AND %s
                GROUP BY evidence_type
                ORDER BY count DESC
            """, (start_date, end_date))

            evidence_stats = cursor.fetchall()

            conn.close()

            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "overall_stats": dict(stats) if stats else {},
                "director_performance": [dict(row) for row in director_stats],
                "evidence_distribution": [dict(row) for row in evidence_stats]
            }

        except Exception as e:
            logger.error(f"Failed to generate board analytics: {e}")
            return {}

    def _get_latest_decision_point(self, task_id: str) -> Optional[DecisionPoint]:
        """Get the latest decision point for a task"""
        if task_id not in self.active_tasks:
            return None

        decision_points = self.active_tasks[task_id].decision_points
        if not decision_points:
            return None

        return max(decision_points, key=lambda dp: dp.timestamp)

    def _recalculate_scores(self, decision_point: DecisionPoint):
        """Recalculate consensus and confidence scores for a decision point"""
        evaluations = decision_point.director_evaluations
        if not evaluations:
            return

        # Calculate consensus score (agreement level)
        recommendations = [eval.recommendation for eval in evaluations]
        unique_recs = set(recommendations)

        if len(unique_recs) == 1:
            # Perfect consensus
            decision_point.consensus_score = 1.0
        else:
            # Calculate based on majority
            most_common = max(set(recommendations), key=recommendations.count)
            majority_count = recommendations.count(most_common)
            decision_point.consensus_score = majority_count / len(recommendations)

        # Calculate total confidence (weighted average)
        total_weight = sum(eval.confidence for eval in evaluations)
        if total_weight > 0:
            decision_point.total_confidence = total_weight / len(evaluations)

        # Update evidence summary
        decision_point.evidence_summary = self._generate_evidence_summary_for_point(decision_point)

    def _calculate_final_metrics(self, task: TaskDecision):
        """Calculate final metrics for a completed task"""
        if not task.decision_points:
            return

        # Get all evaluations across all decision points
        all_evaluations = []
        total_processing_time = 0.0

        for point in task.decision_points:
            all_evaluations.extend(point.director_evaluations)
            for eval in point.director_evaluations:
                total_processing_time += eval.processing_time

        task.total_processing_time = total_processing_time

        if all_evaluations:
            # Final consensus score (from latest decision point)
            task.consensus_score = task.decision_points[-1].consensus_score

            # Final confidence score (weighted average of all evaluations)
            total_confidence = sum(eval.confidence for eval in all_evaluations)
            task.confidence_score = total_confidence / len(all_evaluations)

    def _generate_evidence_summary(self, task: TaskDecision) -> Dict[str, Any]:
        """Generate comprehensive evidence summary for a task"""
        all_evidence = []

        for point in task.decision_points:
            for evaluation in point.director_evaluations:
                all_evidence.extend(evaluation.evidence)

        if not all_evidence:
            return {}

        # Group by evidence type
        evidence_by_type = {}
        for evidence in all_evidence:
            type_name = evidence.type.value
            if type_name not in evidence_by_type:
                evidence_by_type[type_name] = []
            evidence_by_type[type_name].append(evidence)

        # Generate summary
        summary = {
            "total_evidence_count": len(all_evidence),
            "evidence_types": {},
            "average_confidence": sum(e.confidence for e in all_evidence) / len(all_evidence),
            "key_evidence": []
        }

        for evidence_type, evidence_list in evidence_by_type.items():
            summary["evidence_types"][evidence_type] = {
                "count": len(evidence_list),
                "avg_weight": sum(e.weight for e in evidence_list) / len(evidence_list),
                "avg_confidence": sum(e.confidence for e in evidence_list) / len(evidence_list)
            }

        # Identify key evidence (highest weight and confidence)
        sorted_evidence = sorted(all_evidence,
                               key=lambda e: e.weight * e.confidence,
                               reverse=True)

        summary["key_evidence"] = [
            {
                "type": e.type.value,
                "source": e.source,
                "weight": e.weight,
                "confidence": e.confidence,
                "reasoning": e.reasoning
            }
            for e in sorted_evidence[:5]  # Top 5 pieces of evidence
        ]

        return summary

    def _generate_evidence_summary_for_point(self, decision_point: DecisionPoint) -> Dict[str, Any]:
        """Generate evidence summary for a specific decision point"""
        all_evidence = []

        for evaluation in decision_point.director_evaluations:
            all_evidence.extend(evaluation.evidence)

        if not all_evidence:
            return {}

        return {
            "evidence_count": len(all_evidence),
            "types": list(set(e.type.value for e in all_evidence)),
            "avg_confidence": sum(e.confidence for e in all_evidence) / len(all_evidence),
            "sources": list(set(e.source for e in all_evidence))
        }

    def _save_task_to_db(self, task: TaskDecision):
        """Save task decision to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO board_tasks (
                    task_id, user_id, original_request, submitted_at, status,
                    final_recommendation, consensus_score, confidence_score,
                    total_processing_time, evidence_count, director_participation,
                    completion_timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (task_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    final_recommendation = EXCLUDED.final_recommendation,
                    consensus_score = EXCLUDED.consensus_score,
                    confidence_score = EXCLUDED.confidence_score,
                    total_processing_time = EXCLUDED.total_processing_time,
                    evidence_count = EXCLUDED.evidence_count,
                    director_participation = EXCLUDED.director_participation,
                    completion_timestamp = EXCLUDED.completion_timestamp,
                    updated_at = NOW()
            """, (
                task.task_id, task.user_id, task.original_request, task.submitted_at,
                task.status.value, task.final_recommendation, task.consensus_score,
                task.confidence_score, task.total_processing_time, task.evidence_count,
                task.director_participation, task.completion_timestamp
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save task to database: {e}")

    def _update_task_in_db(self, task: TaskDecision):
        """Update existing task in database"""
        self._save_task_to_db(task)  # Uses ON CONFLICT UPDATE

    def _save_evaluation_to_db(self, evaluation: DirectorEvaluation, decision_point_id: str):
        """Save director evaluation to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            eval_id = str(uuid.uuid4())

            cursor.execute("""
                INSERT INTO director_evaluations (
                    evaluation_id, task_id, decision_point_id, director_id,
                    director_name, recommendation, confidence, reasoning,
                    processing_time, risk_score, timestamp, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                eval_id, evaluation.task_id if hasattr(evaluation, 'task_id') else None,
                decision_point_id, evaluation.director_id, evaluation.director_name,
                evaluation.recommendation, evaluation.confidence, evaluation.reasoning,
                evaluation.processing_time, evaluation.risk_score, evaluation.timestamp,
                json.dumps(evaluation.metadata) if evaluation.metadata else None
            ))

            # Save evidence
            for evidence in evaluation.evidence:
                cursor.execute("""
                    INSERT INTO decision_evidence (
                        evidence_id, evaluation_id, evidence_type, source,
                        weight, confidence, reasoning, data, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    evidence.id, eval_id, evidence.type.value, evidence.source,
                    evidence.weight, evidence.confidence, evidence.reasoning,
                    json.dumps(evidence.data), evidence.timestamp
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save evaluation to database: {e}")

    def _save_decision_point_to_db(self, decision_point: DecisionPoint):
        """Save decision point to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO decision_timeline (
                    point_id, task_id, timestamp, status, description,
                    consensus_score, total_confidence, evidence_summary, user_feedback
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                decision_point.id, decision_point.task_id, decision_point.timestamp,
                decision_point.status.value, decision_point.description,
                decision_point.consensus_score, decision_point.total_confidence,
                json.dumps(decision_point.evidence_summary), decision_point.user_feedback
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save decision point to database: {e}")

    def _save_override_to_db(self, override_record: Dict[str, Any], task_id: str):
        """Save user override to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_overrides (
                    override_id, task_id, user_id, override_type,
                    original_recommendation, new_recommendation, reasoning, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                override_record["override_id"], task_id, override_record["user_id"],
                override_record["override_type"], override_record["original_recommendation"],
                override_record["new_recommendation"], override_record["reasoning"],
                datetime.fromisoformat(override_record["timestamp"])
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save override to database: {e}")

    def _load_task_from_db(self, task_id: str) -> Optional[TaskDecision]:
        """Load task decision from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Load main task data
            cursor.execute("""
                SELECT * FROM board_tasks WHERE task_id = %s
            """, (task_id,))

            task_row = cursor.fetchone()
            if not task_row:
                return None

            # Load decision timeline
            cursor.execute("""
                SELECT * FROM decision_timeline
                WHERE task_id = %s
                ORDER BY timestamp
            """, (task_id,))

            timeline_rows = cursor.fetchall()

            # Load evaluations for each decision point
            decision_points = []
            for timeline_row in timeline_rows:
                cursor.execute("""
                    SELECT * FROM director_evaluations
                    WHERE decision_point_id = %s
                """, (timeline_row['point_id'],))

                evaluation_rows = cursor.fetchall()
                evaluations = []

                for eval_row in evaluation_rows:
                    # Load evidence for evaluation
                    cursor.execute("""
                        SELECT * FROM decision_evidence
                        WHERE evaluation_id = %s
                    """, (eval_row['evaluation_id'],))

                    evidence_rows = cursor.fetchall()
                    evidence_list = []

                    for evidence_row in evidence_rows:
                        evidence = Evidence(
                            id=evidence_row['evidence_id'],
                            type=EvidenceType(evidence_row['evidence_type']),
                            source=evidence_row['source'],
                            weight=evidence_row['weight'],
                            data=evidence_row['data'],
                            timestamp=evidence_row['timestamp'],
                            confidence=evidence_row['confidence'],
                            reasoning=evidence_row['reasoning']
                        )
                        evidence_list.append(evidence)

                    evaluation = DirectorEvaluation(
                        director_id=eval_row['director_id'],
                        director_name=eval_row['director_name'],
                        recommendation=eval_row['recommendation'],
                        confidence=eval_row['confidence'],
                        reasoning=eval_row['reasoning'],
                        evidence=evidence_list,
                        processing_time=eval_row['processing_time'],
                        risk_score=eval_row['risk_score'],
                        timestamp=eval_row['timestamp'],
                        metadata=eval_row['metadata']
                    )
                    evaluations.append(evaluation)

                decision_point = DecisionPoint(
                    id=timeline_row['point_id'],
                    task_id=timeline_row['task_id'],
                    timestamp=timeline_row['timestamp'],
                    status=DecisionStatus(timeline_row['status']),
                    description=timeline_row['description'],
                    director_evaluations=evaluations,
                    consensus_score=timeline_row['consensus_score'] or 0.0,
                    total_confidence=timeline_row['total_confidence'] or 0.0,
                    evidence_summary=timeline_row['evidence_summary'] or {},
                    user_feedback=timeline_row['user_feedback']
                )
                decision_points.append(decision_point)

            # Load override history
            cursor.execute("""
                SELECT * FROM user_overrides
                WHERE task_id = %s
                ORDER BY timestamp
            """, (task_id,))

            override_rows = cursor.fetchall()
            override_history = [dict(row) for row in override_rows]

            conn.close()

            # Reconstruct TaskDecision
            task_decision = TaskDecision(
                task_id=task_row['task_id'],
                user_id=task_row['user_id'],
                original_request=task_row['original_request'],
                submitted_at=task_row['submitted_at'],
                status=DecisionStatus(task_row['status']),
                final_recommendation=task_row['final_recommendation'] or "",
                consensus_score=task_row['consensus_score'] or 0.0,
                confidence_score=task_row['confidence_score'] or 0.0,
                decision_points=decision_points,
                total_processing_time=task_row['total_processing_time'] or 0.0,
                evidence_count=task_row['evidence_count'] or 0,
                director_participation=task_row['director_participation'] or [],
                override_history=override_history,
                completion_timestamp=task_row['completion_timestamp']
            )

            return task_decision

        except Exception as e:
            logger.error(f"Failed to load task from database: {e}")
            return None