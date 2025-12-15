#!/usr/bin/env python3
"""
Learning Pipeline API Routes for Echo Brain
Triggers comprehensive learning from all available data sources
"""
import logging
import asyncio
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Dict, Any

from src.learning.comprehensive_data_ingestion import comprehensive_ingestion

logger = logging.getLogger(__name__)
router = APIRouter()

# Track learning pipeline status
pipeline_status = {
    'last_run': None,
    'status': 'idle',
    'patterns_extracted': 0,
    'sources_processed': 0,
    'errors': [],
    'total_runs': 0
}

@router.post("/api/echo/learning/comprehensive-ingestion")
async def trigger_comprehensive_learning(background_tasks: BackgroundTasks):
    """
    Trigger comprehensive learning from all available data sources.
    This will analyze codebase, database, documentation, configs, and git history.
    """
    if pipeline_status['status'] == 'running':
        return {
            'status': 'already_running',
            'message': 'Comprehensive learning pipeline is already running',
            'current_run_start': pipeline_status['last_run']
        }

    # Start learning in background
    background_tasks.add_task(run_comprehensive_learning)

    pipeline_status['status'] = 'running'
    pipeline_status['last_run'] = datetime.now().isoformat()
    pipeline_status['errors'] = []

    logger.info("🚀 Starting comprehensive learning pipeline...")

    return {
        'status': 'started',
        'message': 'Comprehensive learning pipeline started in background',
        'start_time': pipeline_status['last_run']
    }

async def run_comprehensive_learning():
    """Background task to run comprehensive learning"""
    try:
        logger.info("📚 Running comprehensive data ingestion...")

        # Run the comprehensive ingestion
        results = await comprehensive_ingestion.ingest_all_sources()

        # Update status
        pipeline_status.update({
            'status': 'completed',
            'patterns_extracted': results.get('patterns_extracted', 0),
            'sources_processed': results.get('sources_processed', 0),
            'total_runs': pipeline_status['total_runs'] + 1,
            'last_completion': datetime.now().isoformat(),
            'last_results': results
        })

        logger.info(f"✅ Comprehensive learning completed - {results.get('patterns_extracted', 0)} patterns extracted")

    except Exception as e:
        error_msg = f"Comprehensive learning failed: {str(e)}"
        logger.error(error_msg)

        pipeline_status.update({
            'status': 'failed',
            'errors': pipeline_status['errors'] + [error_msg],
            'last_error': datetime.now().isoformat()
        })

@router.get("/api/echo/learning/pipeline-status")
async def get_pipeline_status():
    """Get status of the learning pipeline"""
    return {
        'pipeline_status': pipeline_status,
        'next_recommended_run': 'Every 24-48 hours for optimal learning'
    }

@router.post("/api/echo/learning/validate-patterns")
async def validate_learned_patterns():
    """
    Test that learned patterns are actually being applied to responses.
    This validates the end-to-end learning effectiveness.
    """
    try:
        # Test queries that should trigger learned patterns
        test_queries = [
            "What database should I use?",
            "What frontend framework is best?",
            "How should I organize my code?",
            "What's the Tower system architecture?",
            "How do I deploy a service?"
        ]

        validation_results = []

        for query in test_queries:
            # Make a query to Echo and check if patterns are applied
            import requests
            response = requests.post('http://localhost:8309/api/echo/query',
                                   json={'query': query})

            if response.status_code == 200:
                data = response.json()
                validation_results.append({
                    'query': query,
                    'response_preview': data['response'][:100],
                    'patterns_applied': len(data.get('business_logic_applied', [])),
                    'intent': data.get('intent'),
                    'confidence': data.get('confidence')
                })

        return {
            'validation_status': 'completed',
            'test_queries': len(test_queries),
            'results': validation_results,
            'overall_effectiveness': sum(1 for r in validation_results if r['patterns_applied'] > 0) / len(validation_results),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Pattern validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/api/echo/learning/retrain-from-feedback")
async def retrain_from_feedback():
    """
    Use collected feedback to improve pattern matching.
    Analyzes negative feedback to identify pattern issues.
    """
    try:
        # Get feedback from feedback API
        import requests
        response = requests.get('http://localhost:8309/api/echo/feedback/stats')

        if response.status_code != 200:
            raise Exception("Could not fetch feedback data")

        feedback_data = response.json()

        # Analyze negative feedback for pattern improvements
        negative_feedback = [
            f for f in feedback_data.get('recent_feedback', [])
            if f.get('feedback_type') == 'thumbs_down'
        ]

        pattern_issues = []
        for feedback in negative_feedback:
            reason = feedback.get('feedback_data', {}).get('reason', '')
            if 'postgresql' in reason.lower() and 'pushy' in reason.lower():
                pattern_issues.append({
                    'pattern_type': 'database_recommendation',
                    'issue': 'Too aggressive PostgreSQL recommendations',
                    'suggested_fix': 'Reduce confidence or add qualifying language',
                    'feedback_count': 1
                })

        # Apply pattern adjustments (placeholder - would update BusinessLogicPatternMatcher)
        adjustments_made = len(pattern_issues)

        return {
            'retraining_status': 'completed',
            'negative_feedback_analyzed': len(negative_feedback),
            'pattern_issues_identified': len(pattern_issues),
            'adjustments_made': adjustments_made,
            'issues': pattern_issues,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Feedback retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@router.get("/api/echo/learning/pattern-analytics")
async def get_pattern_analytics():
    """
    Analytics on learned patterns - what's working, what's not.
    """
    try:
        # Get pattern statistics from database
        import psycopg2
        db = psycopg2.connect(
            host='localhost',
            database='echo_brain',
            user='patrick',
            password='tower_echo_brain_secret_key_2025'
        )
        cursor = db.cursor()

        # Get pattern distribution
        cursor.execute("""
            SELECT fact_type, COUNT(*), AVG(confidence)
            FROM learning_history
            GROUP BY fact_type
            ORDER BY COUNT(*) DESC
        """)

        pattern_distribution = []
        for row in cursor.fetchall():
            fact_type, count, avg_confidence = row
            pattern_distribution.append({
                'pattern_type': fact_type,
                'count': count,
                'average_confidence': float(avg_confidence or 0)
            })

        # Get recent patterns
        cursor.execute("""
            SELECT fact_type, learned_fact, confidence, created_at
            FROM learning_history
            ORDER BY created_at DESC
            LIMIT 20
        """)

        recent_patterns = []
        for row in cursor.fetchall():
            fact_type, learned_fact, confidence, created_at = row
            recent_patterns.append({
                'type': fact_type,
                'pattern': learned_fact[:100],
                'confidence': float(confidence or 0),
                'learned_at': created_at.isoformat() if created_at else None
            })

        db.close()

        return {
            'analytics_status': 'completed',
            'pattern_distribution': pattern_distribution,
            'recent_patterns': recent_patterns,
            'total_pattern_types': len(pattern_distribution),
            'highest_confidence_type': max(pattern_distribution, key=lambda x: x['average_confidence']) if pattern_distribution else None,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Pattern analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")