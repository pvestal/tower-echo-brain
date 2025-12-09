#!/usr/bin/env python3
"""
Quality Monitoring System for Echo Service
Real quality assessment, failure learning, performance metrics, error analysis, and output validation
"""

import time
import json
import sqlite3
import logging
import re
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import hashlib

logger = logging.getLogger(__name__)

class ResponseQuality(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    FAILED = 1

class ErrorCategory(Enum):
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    LOGIC_ERROR = "logic_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class QualityMetric:
    timestamp: float
    conversation_id: str
    query: str
    response: str
    model_used: str
    response_time: float
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    overall_quality: ResponseQuality
    token_count: int
    success: bool
    error_category: Optional[ErrorCategory] = None
    error_details: Optional[str] = None

class QualityMonitor:
    def __init__(self, db_path: str = "/opt/tower-echo-brain/quality_metrics.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for quality metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                conversation_id TEXT,
                query_hash TEXT,
                query TEXT,
                response_hash TEXT,
                response TEXT,
                model_used TEXT,
                response_time REAL,
                relevance_score REAL,
                completeness_score REAL,
                accuracy_score REAL,
                overall_quality INTEGER,
                token_count INTEGER,
                success BOOLEAN,
                error_category TEXT,
                error_details TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failure_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE,
                pattern_description TEXT,
                occurrence_count INTEGER,
                first_seen REAL,
                last_seen REAL,
                suggested_fix TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def assess_response_quality(self, query: str, response: str, model_used: str,
                              response_time: float, success: bool,
                              error_details: Optional[str] = None):
        """Assess the quality of a response using real analysis"""
        
        if not success:
            return QualityMetric(
                timestamp=time.time(),
                conversation_id="",
                query=query,
                response=response,
                model_used=model_used,
                response_time=response_time,
                relevance_score=0.0,
                completeness_score=0.0,
                accuracy_score=0.0,
                overall_quality=ResponseQuality.FAILED,
                token_count=0,
                success=False,
                error_category=self._categorize_error(error_details),
                error_details=error_details
            )

        relevance_score = self._assess_relevance(query, response)
        completeness_score = self._assess_completeness(query, response)
        accuracy_score = self._assess_accuracy(response)

        quality_factors = [relevance_score, completeness_score, accuracy_score]
        time_factor = max(0.5, min(1.0, 10.0 / max(response_time, 1.0)))
        quality_factors.append(time_factor)

        avg_quality = statistics.mean(quality_factors)

        if avg_quality >= 0.9:
            overall_quality = ResponseQuality.EXCELLENT
        elif avg_quality >= 0.75:
            overall_quality = ResponseQuality.GOOD
        elif avg_quality >= 0.6:
            overall_quality = ResponseQuality.AVERAGE
        elif avg_quality >= 0.4:
            overall_quality = ResponseQuality.POOR
        else:
            overall_quality = ResponseQuality.FAILED

        token_count = len(response.split()) + len(query.split())

        return QualityMetric(
            timestamp=time.time(),
            conversation_id="",
            query=query,
            response=response,
            model_used=model_used,
            response_time=response_time,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            overall_quality=overall_quality,
            token_count=token_count,
            success=success,
            error_category=None,
            error_details=error_details
        )

    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess how relevant the response is to the query"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_keywords = query_words - common_words
        response_keywords = response_words - common_words

        if not query_keywords:
            return 0.7

        overlap = len(query_keywords & response_keywords)
        relevance = overlap / len(query_keywords)

        if any(word in response.lower() for word in ['error', 'exception']) and 'error' in query.lower():
            relevance += 0.2

        return min(1.0, relevance)

    def _assess_completeness(self, query: str, response: str) -> float:
        """Assess how complete the response is"""
        response_length = len(response.strip())

        if response_length < 10:
            return 0.1
        elif response_length < 50:
            return 0.4
        elif response_length < 200:
            return 0.7
        elif response_length < 1000:
            return 0.9
        else:
            return 0.8

    def _assess_accuracy(self, response: str) -> float:
        """Assess the accuracy of the response"""
        accuracy_score = 0.7

        if any(indicator in response.lower() for indicator in ['according to', 'based on']):
            accuracy_score += 0.1

        if re.search(r'\d{4}', response):
            accuracy_score += 0.05

        if 'error' in response.lower() and 'traceback' in response.lower():
            accuracy_score = 0.3

        return max(0.0, min(1.0, accuracy_score))

    def _categorize_error(self, error_details: Optional[str]) -> ErrorCategory:
        """Categorize the type of error"""
        if not error_details:
            return ErrorCategory.UNKNOWN_ERROR

        error_lower = error_details.lower()

        if 'timeout' in error_lower:
            return ErrorCategory.TIMEOUT
        elif 'model' in error_lower:
            return ErrorCategory.MODEL_ERROR
        elif 'network' in error_lower:
            return ErrorCategory.NETWORK_ERROR
        else:
            return ErrorCategory.UNKNOWN_ERROR

    def log_interaction(self, metric: QualityMetric, conversation_id: str = ""):
        """Log an interaction to the database"""
        metric.conversation_id = conversation_id

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query_hash = hashlib.md5(metric.query.encode()).hexdigest()
        response_hash = hashlib.md5(metric.response.encode()).hexdigest()

        cursor.execute('''
            INSERT INTO quality_metrics (
                timestamp, conversation_id, query_hash, query, response_hash, response,
                model_used, response_time, relevance_score, completeness_score,
                accuracy_score, overall_quality, token_count, success,
                error_category, error_details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.timestamp, metric.conversation_id, query_hash, metric.query,
            response_hash, metric.response, metric.model_used, metric.response_time,
            metric.relevance_score, metric.completeness_score, metric.accuracy_score,
            metric.overall_quality.value, metric.token_count, metric.success,
            metric.error_category.value if metric.error_category else None,
            metric.error_details
        ))

        conn.commit()
        conn.close()

    def get_performance_metrics(self, hours: int = 24) -> Dict:
        """Get performance metrics for the specified time period"""
        since_timestamp = time.time() - (hours * 3600)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                AVG(response_time) as avg_response_time,
                COUNT(*) as total_interactions,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_interactions,
                AVG(relevance_score) as avg_relevance,
                AVG(completeness_score) as avg_completeness,
                AVG(accuracy_score) as avg_accuracy
            FROM quality_metrics
            WHERE timestamp > ?
        ''', (since_timestamp,))

        basic_stats = cursor.fetchone()

        if not basic_stats or basic_stats[1] == 0:
            return {"error": "No data available for the specified period"}

        total_interactions = basic_stats[1]
        successful_interactions = basic_stats[2] or 0
        success_rate = successful_interactions / total_interactions if total_interactions > 0 else 0.0

        conn.close()

        return {
            "period_hours": hours,
            "total_interactions": total_interactions,
            "success_rate": round(success_rate, 3),
            "avg_response_time": round(basic_stats[0] or 0, 3),
            "avg_relevance_score": round(basic_stats[3] or 0, 3),
            "avg_completeness_score": round(basic_stats[4] or 0, 3),
            "avg_accuracy_score": round(basic_stats[5] or 0, 3)
        }

    def validate_output(self, query: str, response: str, expected_format: Optional[str] = None) -> Dict[str, bool]:
        """Validate if the output meets expected criteria"""
        validation_results = {
            "has_content": len(response.strip()) > 0,
            "not_empty_response": response.strip() not in ["", "None", "null"],
            "reasonable_length": 10 <= len(response) <= 10000,
            "responds_to_query": len(set(query.lower().split()) & set(response.lower().split())) > 0
        }

        validation_results["overall_valid"] = all(validation_results.values())
        return validation_results

# Global instance
quality_monitor = QualityMonitor()
