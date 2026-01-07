# 🔴 DEPRECATED: Use unified_router.py instead
# This file is being phased out in favor of single source of truth
# Import from: from src.routing.unified_router import unified_router

#!/usr/bin/env python3
"""
Continuous Learning System for Echo Brain
Learns from user feedback and adjusts model routing dynamically
"""

import json
import time
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ContinuousLearningSystem:
    """System that learns from user feedback to improve model routing"""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self._init_database_tables()

    def _init_database_tables(self):
        """Initialize tables for feedback tracking"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # User feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(100),
                    query TEXT,
                    model_used VARCHAR(100),
                    complexity_score INTEGER,
                    intent_detected VARCHAR(100),
                    domain_detected VARCHAR(100),
                    user_satisfaction INTEGER CHECK (user_satisfaction BETWEEN 1 AND 5),
                    feedback_text TEXT,
                    response_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Model performance trends table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_trends (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100),
                    intent_type VARCHAR(100),
                    avg_satisfaction FLOAT,
                    total_queries INTEGER,
                    avg_response_time_ms INTEGER,
                    success_rate FLOAT,
                    trend_direction VARCHAR(20), -- 'improving', 'declining', 'stable'
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Learning adjustments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_adjustments (
                    id SERIAL PRIMARY KEY,
                    adjustment_type VARCHAR(50), -- 'weight', 'threshold', 'model_preference'
                    parameter_name VARCHAR(100),
                    old_value JSONB,
                    new_value JSONB,
                    reason TEXT,
                    confidence FLOAT,
                    applied BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_model_satisfaction
                ON user_feedback(model_used, user_satisfaction)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_created
                ON user_feedback(created_at DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trends_model_intent
                ON model_performance_trends(model_name, intent_type)
            """)

            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize learning tables: {e}")

    def record_feedback(self,
                       conversation_id: str,
                       query: str,
                       model_used: str,
                       complexity_score: int,
                       intent: Optional[str],
                       domain: Optional[str],
                       satisfaction: int,
                       feedback_text: Optional[str] = None,
                       response_time_ms: Optional[int] = None) -> bool:
        """
        Record user feedback for a query

        Args:
            conversation_id: Unique conversation identifier
            query: The user's query
            model_used: Model that handled the query
            complexity_score: Calculated complexity score
            intent: Detected intent
            domain: Detected domain
            satisfaction: User satisfaction (1-5)
            feedback_text: Optional text feedback
            response_time_ms: Response time in milliseconds

        Returns:
            True if feedback recorded successfully
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_feedback
                (conversation_id, query, model_used, complexity_score,
                 intent_detected, domain_detected, user_satisfaction,
                 feedback_text, response_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (conversation_id, query, model_used, complexity_score,
                  intent, domain, satisfaction, feedback_text, response_time_ms))

            conn.commit()
            cursor.close()
            conn.close()

            # Trigger learning update
            self._update_learning_metrics()

            return True
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False

    def _update_learning_metrics(self):
        """Update performance metrics based on recent feedback"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Calculate metrics for each model-intent combination
            cursor.execute("""
                SELECT
                    model_used,
                    intent_detected,
                    AVG(user_satisfaction) as avg_satisfaction,
                    COUNT(*) as total_queries,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(CASE WHEN user_satisfaction >= 4 THEN 1 END)::FLOAT / COUNT(*) as success_rate
                FROM user_feedback
                WHERE created_at >= NOW() - INTERVAL '7 days'
                GROUP BY model_used, intent_detected
            """)

            results = cursor.fetchall()

            for row in results:
                model, intent, avg_sat, total, avg_time, success = row

                # Determine trend
                trend = self._calculate_trend(cursor, model, intent, avg_sat)

                # Update or insert performance trends
                cursor.execute("""
                    INSERT INTO model_performance_trends
                    (model_name, intent_type, avg_satisfaction, total_queries,
                     avg_response_time_ms, success_rate, trend_direction)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_name, intent_type)
                    DO UPDATE SET
                        avg_satisfaction = EXCLUDED.avg_satisfaction,
                        total_queries = EXCLUDED.total_queries,
                        avg_response_time_ms = EXCLUDED.avg_response_time_ms,
                        success_rate = EXCLUDED.success_rate,
                        trend_direction = EXCLUDED.trend_direction,
                        last_updated = CURRENT_TIMESTAMP
                """, (model, intent, avg_sat, total, avg_time, success, trend))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update learning metrics: {e}")

    def _calculate_trend(self, cursor, model: str, intent: str, current_avg: float) -> str:
        """Calculate performance trend for model-intent combination"""
        # Get previous average
        cursor.execute("""
            SELECT avg_satisfaction
            FROM model_performance_trends
            WHERE model_name = %s AND intent_type = %s
        """, (model, intent))

        result = cursor.fetchone()
        if not result:
            return 'stable'

        prev_avg = result[0]
        diff = current_avg - prev_avg

        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'

    def suggest_adjustments(self) -> List[Dict]:
        """
        Analyze performance data and suggest routing adjustments

        Returns:
            List of suggested adjustments
        """
        suggestions = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Find underperforming model-intent combinations
            cursor.execute("""
                SELECT model_name, intent_type, avg_satisfaction, success_rate
                FROM model_performance_trends
                WHERE avg_satisfaction < 3.5 OR success_rate < 0.7
                AND total_queries >= 10
            """)

            underperforming = cursor.fetchall()

            for model, intent, avg_sat, success_rate in underperforming:
                # Find better performing alternatives
                cursor.execute("""
                    SELECT model_name, avg_satisfaction, success_rate
                    FROM model_performance_trends
                    WHERE intent_type = %s
                    AND model_name != %s
                    AND avg_satisfaction > %s
                    ORDER BY avg_satisfaction DESC
                    LIMIT 1
                """, (intent, model, avg_sat))

                better = cursor.fetchone()
                if better:
                    suggestion = {
                        'type': 'model_preference',
                        'intent': intent,
                        'current_model': model,
                        'suggested_model': better[0],
                        'current_satisfaction': avg_sat,
                        'expected_satisfaction': better[1],
                        'confidence': min((better[1] - avg_sat) / 2, 0.9),
                        'reason': f"Model {better[0]} performs better for {intent} "
                                f"(satisfaction: {better[1]:.2f} vs {avg_sat:.2f})"
                    }
                    suggestions.append(suggestion)

            # Check for complexity threshold adjustments
            cursor.execute("""
                SELECT
                    complexity_score,
                    AVG(user_satisfaction) as avg_sat,
                    COUNT(*) as count
                FROM user_feedback
                WHERE created_at >= NOW() - INTERVAL '7 days'
                GROUP BY complexity_score
                HAVING COUNT(*) >= 5
                ORDER BY complexity_score
            """)

            complexity_performance = cursor.fetchall()

            # Analyze complexity thresholds
            if len(complexity_performance) >= 10:
                suggestions.extend(self._analyze_complexity_thresholds(complexity_performance))

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")

        return suggestions

    def _analyze_complexity_thresholds(self, data: List[Tuple]) -> List[Dict]:
        """Analyze complexity thresholds for optimization"""
        suggestions = []

        # Convert to numpy arrays for analysis
        complexities = np.array([d[0] for d in data])
        satisfactions = np.array([d[1] for d in data])

        # Find optimal thresholds using gradient analysis
        gradients = np.gradient(satisfactions)

        # Find significant changes in satisfaction
        threshold_indices = np.where(np.abs(gradients) > 0.1)[0]

        for idx in threshold_indices:
            if idx > 0 and idx < len(data) - 1:
                complexity_threshold = complexities[idx]
                suggestion = {
                    'type': 'threshold',
                    'parameter': 'complexity_boundary',
                    'current_value': complexity_threshold,
                    'suggested_value': complexity_threshold + 5,  # Adjust by 5 points
                    'confidence': float(np.abs(gradients[idx])),
                    'reason': f"Satisfaction gradient change detected at complexity {complexity_threshold}"
                }
                suggestions.append(suggestion)

        return suggestions

    def apply_learning(self, auto_apply: bool = False) -> Dict:
        """
        Apply learning adjustments to the system

        Args:
            auto_apply: If True, automatically apply high-confidence adjustments

        Returns:
            Dictionary with applied adjustments and results
        """
        suggestions = self.suggest_adjustments()
        applied = []
        skipped = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            for suggestion in suggestions:
                confidence = suggestion.get('confidence', 0)

                # Only auto-apply high confidence suggestions
                if not auto_apply or confidence < 0.8:
                    # Log suggestion but don't apply
                    cursor.execute("""
                        INSERT INTO learning_adjustments
                        (adjustment_type, parameter_name, old_value, new_value,
                         reason, confidence, applied)
                        VALUES (%s, %s, %s, %s, %s, %s, FALSE)
                    """, (suggestion['type'],
                          suggestion.get('parameter', suggestion.get('intent', '')),
                          json.dumps({'value': suggestion.get('current_model', suggestion.get('current_value'))}),
                          json.dumps({'value': suggestion.get('suggested_model', suggestion.get('suggested_value'))}),
                          suggestion['reason'],
                          confidence))
                    skipped.append(suggestion)
                    continue

                # Apply the adjustment
                if suggestion['type'] == 'model_preference':
                    # Update intent routing preference
                    cursor.execute("""
                        UPDATE intent_routing
                        SET preferred_model = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE intent_name = %s
                    """, (suggestion['suggested_model'], suggestion['intent']))

                    applied.append(suggestion)

                elif suggestion['type'] == 'threshold':
                    # Update complexity thresholds
                    cursor.execute("""
                        UPDATE model_routing
                        SET min_complexity = CASE
                            WHEN min_complexity > %s THEN %s
                            ELSE min_complexity
                            END,
                            max_complexity = CASE
                            WHEN max_complexity < %s THEN %s
                            ELSE max_complexity
                            END
                        WHERE is_active = TRUE
                    """, (suggestion['suggested_value'], suggestion['suggested_value'],
                          suggestion['suggested_value'], suggestion['suggested_value']))

                    applied.append(suggestion)

                # Log the applied adjustment
                cursor.execute("""
                    INSERT INTO learning_adjustments
                    (adjustment_type, parameter_name, old_value, new_value,
                     reason, confidence, applied)
                    VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                """, (suggestion['type'],
                      suggestion.get('parameter', suggestion.get('intent', '')),
                      json.dumps({'value': suggestion.get('current_model', suggestion.get('current_value'))}),
                      json.dumps({'value': suggestion.get('suggested_model', suggestion.get('suggested_value'))}),
                      suggestion['reason'],
                      confidence))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to apply learning adjustments: {e}")

        return {
            'suggestions_generated': len(suggestions),
            'adjustments_applied': len(applied),
            'adjustments_skipped': len(skipped),
            'applied': applied,
            'skipped': skipped
        }

    def get_performance_report(self, days: int = 7) -> Dict:
        """
        Generate a performance report for the learning system

        Args:
            days: Number of days to include in report

        Returns:
            Performance report dictionary
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Overall satisfaction trend
            cursor.execute("""
                SELECT
                    DATE(created_at) as date,
                    AVG(user_satisfaction) as avg_satisfaction,
                    COUNT(*) as query_count
                FROM user_feedback
                WHERE created_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE(created_at)
                ORDER BY date
            """, (days,))

            daily_trends = [
                {'date': str(row[0]), 'avg_satisfaction': row[1], 'queries': row[2]}
                for row in cursor.fetchall()
            ]

            # Model performance summary
            cursor.execute("""
                SELECT
                    model_name,
                    AVG(avg_satisfaction) as overall_satisfaction,
                    SUM(total_queries) as total_queries,
                    AVG(success_rate) as avg_success_rate
                FROM model_performance_trends
                GROUP BY model_name
                ORDER BY overall_satisfaction DESC
            """)

            model_summary = [
                {
                    'model': row[0],
                    'satisfaction': row[1],
                    'queries': row[2],
                    'success_rate': row[3]
                }
                for row in cursor.fetchall()
            ]

            # Recent adjustments
            cursor.execute("""
                SELECT
                    adjustment_type,
                    parameter_name,
                    new_value,
                    confidence,
                    applied,
                    created_at
                FROM learning_adjustments
                WHERE created_at >= NOW() - INTERVAL '%s days'
                ORDER BY created_at DESC
                LIMIT 10
            """, (days,))

            recent_adjustments = [
                {
                    'type': row[0],
                    'parameter': row[1],
                    'new_value': row[2],
                    'confidence': row[3],
                    'applied': row[4],
                    'timestamp': str(row[5])
                }
                for row in cursor.fetchall()
            ]

            cursor.close()
            conn.close()

            return {
                'report_period_days': days,
                'daily_trends': daily_trends,
                'model_performance': model_summary,
                'recent_adjustments': recent_adjustments,
                'generated_at': str(datetime.now())
            }

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {}


if __name__ == "__main__":
    # Test the continuous learning system
    db_config = {
        'host': 'localhost',
        'database': 'echo_brain',
        'user': 'patrick',
        'password': '***REMOVED***'
    }

    learning_system = ContinuousLearningSystem(db_config)

    print("🧠 Continuous Learning System Test")
    print("=" * 60)

    # Record some test feedback
    print("\n📝 Recording test feedback...")
    learning_system.record_feedback(
        conversation_id="test_001",
        query="Generate anime scene with Mei",
        model_used="gemma2:9b",
        complexity_score=75,
        intent="anime_scene",
        domain="anime_production",
        satisfaction=5,
        feedback_text="Perfect anime generation!",
        response_time_ms=1500
    )

    learning_system.record_feedback(
        conversation_id="test_002",
        query="Write Python code",
        model_used="llama3.1:8b",
        complexity_score=45,
        intent="code_generation",
        domain="programming",
        satisfaction=3,
        feedback_text="Code works but could be better",
        response_time_ms=800
    )

    # Generate suggestions
    print("\n🤔 Analyzing performance and generating suggestions...")
    suggestions = learning_system.suggest_adjustments()

    if suggestions:
        print(f"Found {len(suggestions)} suggestions:")
        for s in suggestions:
            print(f"  - {s['type']}: {s['reason']}")
            print(f"    Confidence: {s.get('confidence', 0):.2%}")
    else:
        print("No suggestions at this time (need more data)")

    # Generate report
    print("\n📊 Generating performance report...")
    report = learning_system.get_performance_report(days=7)

    if report.get('model_performance'):
        print("\nModel Performance Summary:")
        for model in report['model_performance']:
            if model['satisfaction'] is not None:
                print(f"  {model['model']}: {model['satisfaction']:.2f}/5.0 "
                      f"({model['queries']} queries)")

    print("\n✅ Continuous Learning System Test Complete!")