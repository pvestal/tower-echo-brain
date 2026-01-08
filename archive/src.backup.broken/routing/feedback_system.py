#!/usr/bin/env python3
"""
Feedback System for AI Assist Board of Directors
Implements user control, feedback processing, and machine learning from user decisions
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
from src.routing.db_pool import get_db_pool
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    APPROVAL = "approval"
    REJECTION = "rejection"
    MODIFICATION = "modification"
    ESCALATION = "escalation"
    COMMENT = "comment"
    RATING = "rating"

class LearningType(Enum):
    DIRECTOR_WEIGHT_ADJUSTMENT = "director_weight_adjustment"
    PATTERN_RECOGNITION = "pattern_recognition"
    PREFERENCE_EXTRACTION = "preference_extraction"
    CONFIDENCE_CALIBRATION = "confidence_calibration"

@dataclass
class UserFeedback:
    """Individual user feedback record"""
    feedback_id: str
    task_id: str
    user_id: str
    feedback_type: FeedbackType
    content: str
    rating: Optional[float]  # 0.0 to 1.0
    timestamp: datetime
    original_recommendation: str
    user_recommendation: Optional[str]
    reasoning: str
    context: Dict[str, Any]

@dataclass
class LearningInsight:
    """Machine learning insight from user feedback"""
    insight_id: str
    learning_type: LearningType
    confidence: float
    description: str
    affected_directors: List[str]
    weight_adjustments: Dict[str, float]
    pattern_data: Dict[str, Any]
    timestamp: datetime
    validation_score: float

@dataclass
class UserBehaviorPattern:
    """Pattern extracted from user behavior"""
    pattern_id: str
    user_id: str
    pattern_type: str  # approval_bias, rejection_tendency, modification_preference
    pattern_strength: float  # 0.0 to 1.0
    context_conditions: Dict[str, Any]
    examples: List[str]  # Task IDs that demonstrate pattern
    confidence: float
    discovered_at: datetime

class FeedbackProcessor:
    """
    Processes user feedback and implements machine learning
    to improve board decision making over time
    """

    def __init__(self, db_config: Dict[str, str], model_path: str = "/tmp/feedback_models"):
        """
        Initialize FeedbackProcessor

        Args:
            db_config: Database connection parameters
            model_path: Path to store ML models
        """
        self.db_config = db_config
        self.model_path = model_path
        self.feedback_history: Dict[str, List[UserFeedback]] = {}
        self.user_patterns: Dict[str, List[UserBehaviorPattern]] = {}
        self.director_weights: Dict[str, Dict[str, float]] = {}  # user_id -> director_id -> weight

        # ML models
        self.preference_model: Optional[Pipeline] = None
        self.confidence_model: Optional[Pipeline] = None

        self._initialize_database()
        self._initialize_models()

    def _initialize_database(self):
        """Initialize feedback database tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # User feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id VARCHAR(255) PRIMARY KEY,
                    task_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    feedback_type VARCHAR(50) NOT NULL,
                    content TEXT,
                    rating FLOAT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    original_recommendation TEXT,
                    user_recommendation TEXT,
                    reasoning TEXT,
                    context JSONB,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Learning insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_insights (
                    insight_id VARCHAR(255) PRIMARY KEY,
                    learning_type VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    description TEXT,
                    affected_directors TEXT[],
                    weight_adjustments JSONB,
                    pattern_data JSONB,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    validation_score FLOAT,
                    applied BOOLEAN DEFAULT FALSE
                );
            """)

            # User behavior patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_behavior_patterns (
                    pattern_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    pattern_type VARCHAR(100) NOT NULL,
                    pattern_strength FLOAT NOT NULL,
                    context_conditions JSONB,
                    examples TEXT[],
                    confidence FLOAT,
                    discovered_at TIMESTAMP DEFAULT NOW(),
                    active BOOLEAN DEFAULT TRUE
                );
            """)

            # Director weight adjustments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS director_weight_adjustments (
                    adjustment_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    director_id VARCHAR(255) NOT NULL,
                    original_weight FLOAT,
                    adjusted_weight FLOAT,
                    adjustment_reason TEXT,
                    confidence FLOAT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    active BOOLEAN DEFAULT TRUE
                );
            """)

            # Feedback processing queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_processing_queue (
                    queue_id VARCHAR(255) PRIMARY KEY,
                    feedback_id VARCHAR(255) REFERENCES user_feedback(feedback_id),
                    processing_type VARCHAR(50) NOT NULL,
                    priority INTEGER DEFAULT 5,
                    scheduled_at TIMESTAMP DEFAULT NOW(),
                    processing_attempts INTEGER DEFAULT 0,
                    last_error TEXT,
                    processed BOOLEAN DEFAULT FALSE
                );
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback(user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_feedback_task_id ON user_feedback(task_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_behavior_patterns_user_id ON user_behavior_patterns(user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_weight_adjustments_user_id ON director_weight_adjustments(user_id);")

            conn.close()
            logger.info("Feedback processor database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize feedback database: {e}")
            raise

    def _initialize_models(self):
        """Initialize or load ML models"""
        try:
            os.makedirs(self.model_path, exist_ok=True)

            # Load or create preference model
            preference_model_file = os.path.join(self.model_path, "preference_model.pkl")
            if os.path.exists(preference_model_file):
                with open(preference_model_file, 'rb') as f:
                    self.preference_model = pickle.load(f)
                logger.info("Loaded existing preference model")
            else:
                # Create new preference model
                self.preference_model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                    ('classifier', MultinomialNB())
                ])
                logger.info("Created new preference model")

            # Load or create confidence model
            confidence_model_file = os.path.join(self.model_path, "confidence_model.pkl")
            if os.path.exists(confidence_model_file):
                with open(confidence_model_file, 'rb') as f:
                    self.confidence_model = pickle.load(f)
                logger.info("Loaded existing confidence model")
            else:
                # Create new confidence model
                self.confidence_model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
                    ('classifier', MultinomialNB())
                ])
                logger.info("Created new confidence model")

        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")

    def process_user_feedback(self, feedback: UserFeedback) -> bool:
        """
        Process user feedback and trigger learning

        Args:
            feedback: UserFeedback object to process

        Returns:
            bool: True if successfully processed
        """
        try:
            # Store feedback in database
            self._store_feedback(feedback)

            # Add to local cache
            if feedback.user_id not in self.feedback_history:
                self.feedback_history[feedback.user_id] = []
            self.feedback_history[feedback.user_id].append(feedback)

            # Queue for batch processing
            self._queue_feedback_processing(feedback)

            # Immediate processing for high-priority feedback
            if self._is_high_priority_feedback(feedback):
                self._process_feedback_immediately(feedback)

            logger.info(f"Processed feedback {feedback.feedback_id} from user {feedback.user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to process feedback {feedback.feedback_id}: {e}")
            return False

    def generate_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Generate user preference profile from feedback history

        Args:
            user_id: User identifier

        Returns:
            Dict containing user preferences
        """
        try:
            user_feedback = self._get_user_feedback_history(user_id)
            if not user_feedback:
                return {"status": "insufficient_data"}

            preferences = {
                "user_id": user_id,
                "total_feedback_count": len(user_feedback),
                "approval_rate": 0.0,
                "modification_rate": 0.0,
                "rejection_rate": 0.0,
                "average_rating": 0.0,
                "director_preferences": {},
                "context_preferences": {},
                "patterns": [],
                "confidence": 0.0
            }

            # Calculate basic statistics
            approvals = sum(1 for f in user_feedback if f.feedback_type == FeedbackType.APPROVAL)
            modifications = sum(1 for f in user_feedback if f.feedback_type == FeedbackType.MODIFICATION)
            rejections = sum(1 for f in user_feedback if f.feedback_type == FeedbackType.REJECTION)

            total = len(user_feedback)
            preferences["approval_rate"] = approvals / total
            preferences["modification_rate"] = modifications / total
            preferences["rejection_rate"] = rejections / total

            # Calculate average rating
            ratings = [f.rating for f in user_feedback if f.rating is not None]
            if ratings:
                preferences["average_rating"] = sum(ratings) / len(ratings)

            # Analyze director preferences
            director_feedback = {}
            for feedback in user_feedback:
                context = feedback.context or {}
                director_name = context.get("director_name", "unknown")

                if director_name not in director_feedback:
                    director_feedback[director_name] = {"positive": 0, "negative": 0, "total": 0}

                director_feedback[director_name]["total"] += 1

                if feedback.feedback_type in [FeedbackType.APPROVAL]:
                    director_feedback[director_name]["positive"] += 1
                elif feedback.feedback_type in [FeedbackType.REJECTION]:
                    director_feedback[director_name]["negative"] += 1

            # Calculate director preference scores
            for director, stats in director_feedback.items():
                if stats["total"] > 0:
                    preference_score = (stats["positive"] - stats["negative"]) / stats["total"]
                    preferences["director_preferences"][director] = {
                        "preference_score": preference_score,
                        "confidence": min(stats["total"] / 10.0, 1.0),  # Confidence based on sample size
                        "total_feedback": stats["total"]
                    }

            # Detect behavioral patterns
            patterns = self._detect_user_patterns(user_id, user_feedback)
            preferences["patterns"] = [
                {
                    "type": p.pattern_type,
                    "strength": p.pattern_strength,
                    "confidence": p.confidence
                }
                for p in patterns
            ]

            # Overall confidence in preferences
            preferences["confidence"] = min(total / 50.0, 1.0)  # Full confidence after 50+ feedback items

            return preferences

        except Exception as e:
            logger.error(f"Failed to generate user preferences for {user_id}: {e}")
            return {"status": "error", "message": str(e)}

    def adjust_director_weights(self, user_id: str) -> Dict[str, float]:
        """
        Adjust director weights based on user feedback patterns

        Args:
            user_id: User identifier

        Returns:
            Dict mapping director_id to adjusted weight
        """
        try:
            preferences = self.generate_user_preferences(user_id)
            if preferences.get("status") in ["insufficient_data", "error"]:
                return {}

            director_prefs = preferences.get("director_preferences", {})
            adjustments = {}

            # Calculate weight adjustments
            for director, pref_data in director_prefs.items():
                preference_score = pref_data["preference_score"]
                confidence = pref_data["confidence"]

                # Base weight is 1.0, adjust based on preference
                base_weight = 1.0
                adjustment_factor = preference_score * confidence * 0.5  # Max 50% adjustment

                adjusted_weight = max(0.1, base_weight + adjustment_factor)  # Minimum weight 0.1
                adjustments[director] = adjusted_weight

                # Store adjustment in database
                self._store_weight_adjustment(user_id, director, base_weight, adjusted_weight,
                                            f"Preference score: {preference_score:.2f}, confidence: {confidence:.2f}")

            # Cache adjustments
            self.director_weights[user_id] = adjustments

            logger.info(f"Adjusted director weights for user {user_id}: {adjustments}")
            return adjustments

        except Exception as e:
            logger.error(f"Failed to adjust director weights for {user_id}: {e}")
            return {}

    def predict_user_response(self, user_id: str, task_description: str,
                            board_recommendation: str) -> Dict[str, Any]:
        """
        Predict how user might respond to board recommendation

        Args:
            user_id: User identifier
            task_description: Task description
            board_recommendation: Board's recommendation

        Returns:
            Dict containing prediction data
        """
        try:
            if not self.preference_model:
                return {"status": "model_not_ready"}

            # Get user feedback history
            user_feedback = self._get_user_feedback_history(user_id)
            if len(user_feedback) < 5:
                return {"status": "insufficient_data"}

            # Prepare features for prediction
            feature_text = f"{task_description} {board_recommendation}"

            # Prepare training data from user history
            X_train = []
            y_train = []

            for feedback in user_feedback[-50:]:  # Use last 50 feedback items
                training_text = f"{feedback.context.get('task_description', '')} {feedback.original_recommendation}"
                X_train.append(training_text)
                y_train.append(feedback.feedback_type.value)

            if len(set(y_train)) < 2:
                return {"status": "insufficient_variety"}

            # Train model on user's historical data
            self.preference_model.fit(X_train, y_train)

            # Make prediction
            prediction = self.preference_model.predict([feature_text])[0]
            prediction_proba = self.preference_model.predict_proba([feature_text])[0]

            # Get confidence score
            confidence = max(prediction_proba)

            # Prepare result
            result = {
                "predicted_response": prediction,
                "confidence": float(confidence),
                "probability_distribution": {
                    label: float(prob)
                    for label, prob in zip(self.preference_model.classes_, prediction_proba)
                },
                "based_on_samples": len(user_feedback),
                "recommendation": self._generate_response_recommendation(prediction, confidence)
            }

            return result

        except Exception as e:
            logger.error(f"Failed to predict user response: {e}")
            return {"status": "error", "message": str(e)}

    def get_learning_insights(self, limit: int = 10) -> List[LearningInsight]:
        """
        Get recent learning insights from user feedback

        Args:
            limit: Maximum number of insights to return

        Returns:
            List of LearningInsight objects
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute("""
                SELECT * FROM learning_insights
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            insights = []
            for row in rows:
                insight = LearningInsight(
                    insight_id=row['insight_id'],
                    learning_type=LearningType(row['learning_type']),
                    confidence=row['confidence'],
                    description=row['description'],
                    affected_directors=row['affected_directors'] or [],
                    weight_adjustments=row['weight_adjustments'] or {},
                    pattern_data=row['pattern_data'] or {},
                    timestamp=row['timestamp'],
                    validation_score=row['validation_score'] or 0.0
                )
                insights.append(insight)

            return insights

        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return []

    def process_batch_feedback(self) -> int:
        """
        Process queued feedback in batch mode

        Returns:
            int: Number of feedback items processed
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get unprocessed feedback from queue
            cursor.execute("""
                SELECT fpq.*, uf.* FROM feedback_processing_queue fpq
                JOIN user_feedback uf ON fpq.feedback_id = uf.feedback_id
                WHERE fpq.processed = FALSE
                ORDER BY fpq.priority DESC, fpq.scheduled_at ASC
                LIMIT 100
            """)

            queue_items = cursor.fetchall()
            processed_count = 0

            for item in queue_items:
                try:
                    feedback = UserFeedback(
                        feedback_id=item['feedback_id'],
                        task_id=item['task_id'],
                        user_id=item['user_id'],
                        feedback_type=FeedbackType(item['feedback_type']),
                        content=item['content'] or "",
                        rating=item['rating'],
                        timestamp=item['timestamp'],
                        original_recommendation=item['original_recommendation'] or "",
                        user_recommendation=item['user_recommendation'],
                        reasoning=item['reasoning'] or "",
                        context=item['context'] or {}
                    )

                    # Process feedback
                    success = self._process_single_feedback(feedback)

                    if success:
                        # Mark as processed
                        cursor.execute("""
                            UPDATE feedback_processing_queue
                            SET processed = TRUE
                            WHERE queue_id = %s
                        """, (item['queue_id'],))
                        processed_count += 1
                    else:
                        # Increment attempt counter
                        cursor.execute("""
                            UPDATE feedback_processing_queue
                            SET processing_attempts = processing_attempts + 1,
                                last_error = %s
                            WHERE queue_id = %s
                        """, ("Processing failed", item['queue_id']))

                except Exception as e:
                    logger.error(f"Failed to process queued feedback {item['feedback_id']}: {e}")

                    cursor.execute("""
                        UPDATE feedback_processing_queue
                        SET processing_attempts = processing_attempts + 1,
                            last_error = %s
                        WHERE queue_id = %s
                    """, (str(e), item['queue_id']))

            conn.commit()
            conn.close()

            logger.info(f"Batch processed {processed_count} feedback items")
            return processed_count

        except Exception as e:
            logger.error(f"Failed to process batch feedback: {e}")
            return 0

    def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_feedback (
                    feedback_id, task_id, user_id, feedback_type, content,
                    rating, timestamp, original_recommendation, user_recommendation,
                    reasoning, context
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                feedback.feedback_id, feedback.task_id, feedback.user_id,
                feedback.feedback_type.value, feedback.content, feedback.rating,
                feedback.timestamp, feedback.original_recommendation,
                feedback.user_recommendation, feedback.reasoning,
                json.dumps(feedback.context)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            raise

    def _queue_feedback_processing(self, feedback: UserFeedback):
        """Add feedback to processing queue"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            priority = 5  # Default priority
            if feedback.feedback_type in [FeedbackType.REJECTION, FeedbackType.MODIFICATION]:
                priority = 8  # Higher priority for negative feedback

            cursor.execute("""
                INSERT INTO feedback_processing_queue (
                    queue_id, feedback_id, processing_type, priority
                ) VALUES (%s, %s, %s, %s)
            """, (
                str(uuid.uuid4()), feedback.feedback_id, "pattern_analysis", priority
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to queue feedback processing: {e}")

    def _is_high_priority_feedback(self, feedback: UserFeedback) -> bool:
        """Determine if feedback requires immediate processing"""
        return feedback.feedback_type in [FeedbackType.REJECTION, FeedbackType.ESCALATION]

    def _process_feedback_immediately(self, feedback: UserFeedback):
        """Process feedback immediately for high-priority items"""
        try:
            # Generate immediate insights
            insights = self._analyze_feedback_patterns([feedback])

            for insight in insights:
                self._store_learning_insight(insight)

            logger.info(f"Immediately processed high-priority feedback {feedback.feedback_id}")

        except Exception as e:
            logger.error(f"Failed to immediately process feedback: {e}")

    def _process_single_feedback(self, feedback: UserFeedback) -> bool:
        """Process a single feedback item"""
        try:
            # Update user patterns
            user_feedback_history = self._get_user_feedback_history(feedback.user_id)
            patterns = self._detect_user_patterns(feedback.user_id, user_feedback_history + [feedback])

            # Store new patterns
            for pattern in patterns:
                if pattern.confidence > 0.7:  # Only store high-confidence patterns
                    self._store_behavior_pattern(pattern)

            # Generate learning insights
            insights = self._analyze_feedback_patterns([feedback])
            for insight in insights:
                self._store_learning_insight(insight)

            return True

        except Exception as e:
            logger.error(f"Failed to process single feedback: {e}")
            return False

    def _get_user_feedback_history(self, user_id: str) -> List[UserFeedback]:
        """Get user's feedback history from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute("""
                SELECT * FROM user_feedback
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT 200
            """, (user_id,))

            rows = cursor.fetchall()
            conn.close()

            feedback_list = []
            for row in rows:
                feedback = UserFeedback(
                    feedback_id=row['feedback_id'],
                    task_id=row['task_id'],
                    user_id=row['user_id'],
                    feedback_type=FeedbackType(row['feedback_type']),
                    content=row['content'] or "",
                    rating=row['rating'],
                    timestamp=row['timestamp'],
                    original_recommendation=row['original_recommendation'] or "",
                    user_recommendation=row['user_recommendation'],
                    reasoning=row['reasoning'] or "",
                    context=row['context'] or {}
                )
                feedback_list.append(feedback)

            return feedback_list

        except Exception as e:
            logger.error(f"Failed to get user feedback history: {e}")
            return []

    def _detect_user_patterns(self, user_id: str, feedback_list: List[UserFeedback]) -> List[UserBehaviorPattern]:
        """Detect behavioral patterns from user feedback"""
        patterns = []

        if len(feedback_list) < 10:
            return patterns

        try:
            # Pattern 1: Approval bias
            approvals = sum(1 for f in feedback_list if f.feedback_type == FeedbackType.APPROVAL)
            approval_rate = approvals / len(feedback_list)

            if approval_rate > 0.8:
                pattern = UserBehaviorPattern(
                    pattern_id=str(uuid.uuid4()),
                    user_id=user_id,
                    pattern_type="high_approval_bias",
                    pattern_strength=approval_rate,
                    context_conditions={"minimum_feedback": 10},
                    examples=[f.task_id for f in feedback_list if f.feedback_type == FeedbackType.APPROVAL][:5],
                    confidence=min(len(feedback_list) / 50.0, 1.0),
                    discovered_at=datetime.utcnow()
                )
                patterns.append(pattern)

            # Pattern 2: Rejection tendency
            rejections = sum(1 for f in feedback_list if f.feedback_type == FeedbackType.REJECTION)
            rejection_rate = rejections / len(feedback_list)

            if rejection_rate > 0.3:
                pattern = UserBehaviorPattern(
                    pattern_id=str(uuid.uuid4()),
                    user_id=user_id,
                    pattern_type="high_rejection_tendency",
                    pattern_strength=rejection_rate,
                    context_conditions={"minimum_feedback": 10},
                    examples=[f.task_id for f in feedback_list if f.feedback_type == FeedbackType.REJECTION][:5],
                    confidence=min(len(feedback_list) / 50.0, 1.0),
                    discovered_at=datetime.utcnow()
                )
                patterns.append(pattern)

            # Pattern 3: Modification preference
            modifications = sum(1 for f in feedback_list if f.feedback_type == FeedbackType.MODIFICATION)
            modification_rate = modifications / len(feedback_list)

            if modification_rate > 0.4:
                pattern = UserBehaviorPattern(
                    pattern_id=str(uuid.uuid4()),
                    user_id=user_id,
                    pattern_type="modification_preference",
                    pattern_strength=modification_rate,
                    context_conditions={"minimum_feedback": 10},
                    examples=[f.task_id for f in feedback_list if f.feedback_type == FeedbackType.MODIFICATION][:5],
                    confidence=min(len(feedback_list) / 50.0, 1.0),
                    discovered_at=datetime.utcnow()
                )
                patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"Failed to detect user patterns: {e}")
            return []

    def _analyze_feedback_patterns(self, feedback_list: List[UserFeedback]) -> List[LearningInsight]:
        """Analyze feedback to generate learning insights"""
        insights = []

        try:
            if not feedback_list:
                return insights

            # Insight 1: Director performance correlation
            director_feedback = {}
            for feedback in feedback_list:
                director = feedback.context.get("director_name", "unknown")
                if director not in director_feedback:
                    director_feedback[director] = []
                director_feedback[director].append(feedback)

            for director, dir_feedback in director_feedback.items():
                if len(dir_feedback) >= 5:
                    approvals = sum(1 for f in dir_feedback if f.feedback_type == FeedbackType.APPROVAL)
                    approval_rate = approvals / len(dir_feedback)

                    if approval_rate < 0.5:  # Low approval rate suggests need for adjustment
                        insight = LearningInsight(
                            insight_id=str(uuid.uuid4()),
                            learning_type=LearningType.DIRECTOR_WEIGHT_ADJUSTMENT,
                            confidence=min(len(dir_feedback) / 20.0, 1.0),
                            description=f"Director {director} has low approval rate ({approval_rate:.2f})",
                            affected_directors=[director],
                            weight_adjustments={director: max(0.1, 1.0 - (1.0 - approval_rate) * 0.5)},
                            pattern_data={"approval_rate": approval_rate, "sample_size": len(dir_feedback)},
                            timestamp=datetime.utcnow(),
                            validation_score=0.0
                        )
                        insights.append(insight)

            return insights

        except Exception as e:
            logger.error(f"Failed to analyze feedback patterns: {e}")
            return []

    def _store_learning_insight(self, insight: LearningInsight):
        """Store learning insight in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO learning_insights (
                    insight_id, learning_type, confidence, description,
                    affected_directors, weight_adjustments, pattern_data,
                    timestamp, validation_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                insight.insight_id, insight.learning_type.value, insight.confidence,
                insight.description, insight.affected_directors,
                json.dumps(insight.weight_adjustments), json.dumps(insight.pattern_data),
                insight.timestamp, insight.validation_score
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store learning insight: {e}")

    def _store_behavior_pattern(self, pattern: UserBehaviorPattern):
        """Store behavior pattern in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_behavior_patterns (
                    pattern_id, user_id, pattern_type, pattern_strength,
                    context_conditions, examples, confidence, discovered_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pattern_id) DO UPDATE SET
                    pattern_strength = EXCLUDED.pattern_strength,
                    confidence = EXCLUDED.confidence,
                    discovered_at = EXCLUDED.discovered_at
            """, (
                pattern.pattern_id, pattern.user_id, pattern.pattern_type,
                pattern.pattern_strength, json.dumps(pattern.context_conditions),
                pattern.examples, pattern.confidence, pattern.discovered_at
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store behavior pattern: {e}")

    def _store_weight_adjustment(self, user_id: str, director_id: str,
                               original_weight: float, adjusted_weight: float, reason: str):
        """Store director weight adjustment"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            adjustment_id = str(uuid.uuid4())
            confidence = abs(adjusted_weight - original_weight)

            cursor.execute("""
                INSERT INTO director_weight_adjustments (
                    adjustment_id, user_id, director_id, original_weight,
                    adjusted_weight, adjustment_reason, confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                adjustment_id, user_id, director_id, original_weight,
                adjusted_weight, reason, confidence
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store weight adjustment: {e}")

    def _generate_response_recommendation(self, prediction: str, confidence: float) -> str:
        """Generate recommendation based on prediction"""
        if confidence < 0.6:
            return "Low confidence prediction - proceed with caution"

        if prediction == "approval":
            return "User likely to approve - board recommendation aligns with preferences"
        elif prediction == "rejection":
            return "User likely to reject - consider alternative approaches"
        elif prediction == "modification":
            return "User likely to request modifications - prepare alternative options"
        else:
            return "Uncertain prediction - monitor user response closely"

    def save_models(self):
        """Save trained ML models to disk"""
        try:
            if self.preference_model:
                preference_model_file = os.path.join(self.model_path, "preference_model.pkl")
                with open(preference_model_file, 'wb') as f:
                    pickle.dump(self.preference_model, f)

            if self.confidence_model:
                confidence_model_file = os.path.join(self.model_path, "confidence_model.pkl")
                with open(confidence_model_file, 'wb') as f:
                    pickle.dump(self.confidence_model, f)

            logger.info("ML models saved successfully")

        except Exception as e:
            logger.error(f"Failed to save ML models: {e}")