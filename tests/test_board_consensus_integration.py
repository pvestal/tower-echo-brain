"""
Comprehensive Integration Tests for Board Consensus System

This module tests the complete board decision-making process:
- Multi-director evaluation coordination
- Consensus algorithm implementation
- Decision tracking and persistence
- Real-time feedback integration
- Performance under load

Author: Echo Brain CI/CD Pipeline
Created: 2025-09-16
"""

import pytest
import asyncio
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.service_registry import ServiceRegistry
from routing.request_logger import RequestLogger, TaskDecision
from routing.feedback_system import FeedbackSystem
from routing.security_director import SecurityDirector
from routing.quality_director import QualityDirector
from routing.performance_director import PerformanceDirector
from routing.ethics_director import EthicsDirector
from routing.ux_director import UXDirector


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
def full_board_setup():
    """Set up a complete board with all directors."""
    registry = ServiceRegistry()
    
    # Register all directors
    directors = {
        'security': SecurityDirector(),
        'quality': QualityDirector(),
        'performance': PerformanceDirector(),
        'ethics': EthicsDirector(),
        'ux': UXDirector()
    }
    
    for name, director in directors.items():
        registry.register_director(name, director)
    
    return registry, directors


@pytest.fixture
async def request_logger_with_db():
    """Set up decision tracker with mock database."""
    with patch('directors.request_logger.DatabasePool') as mock_db:
        # Configure mock database
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_db.return_value.get_connection.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_db.return_value.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        tracker = RequestLogger()
        await tracker.initialize()
        yield tracker


@pytest.fixture
def complex_task_samples():
    """Provide complex task samples for integration testing."""
    return {
        "secure_payment_system": {
            "task_id": "integration_001",
            "description": "Implement secure payment processing system",
            "code": """
import hashlib
import secrets
from cryptography.fernet import Fernet
from decimal import Decimal

class PaymentProcessor:
    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)
        self.processed_transactions = set()
    
    def process_payment(self, amount: Decimal, card_number: str, cvv: str) -> dict:
        # Generate transaction ID
        transaction_id = secrets.token_urlsafe(32)
        
        # Validate amount
        if amount <= 0 or amount > Decimal('10000'):
            raise ValueError("Invalid payment amount")
        
        # Mask card number for logging
        masked_card = '*' * (len(card_number) - 4) + card_number[-4:]
        
        # Encrypt sensitive data
        encrypted_data = self.cipher.encrypt(
            json.dumps({
                "card_number": card_number,
                "cvv": cvv,
                "amount": str(amount),
                "timestamp": datetime.now().isoformat()
            }).encode()
        )
        
        # Prevent duplicate processing
        if transaction_id in self.processed_transactions:
            raise ValueError("Duplicate transaction")
        
        self.processed_transactions.add(transaction_id)
        
        return {
            "transaction_id": transaction_id,
            "amount": amount,
            "masked_card": masked_card,
            "status": "approved",
            "timestamp": datetime.now().isoformat()
        }
            """,
            "language": "python",
            "priority": "high",
            "context": {
                "application_type": "financial",
                "compliance_required": ["PCI-DSS", "SOX"],
                "user_facing": True,
                "performance_critical": True
            }
        },
        
        "user_dashboard_ui": {
            "task_id": "integration_002",
            "description": "Create accessible user dashboard interface",
            "code": """
<!-- User Dashboard with Accessibility Features -->
<div class="dashboard" role="main" aria-label="User Dashboard">
    <header class="dashboard-header">
        <h1 id="dashboard-title">Welcome, {{user.name}}</h1>
        <nav aria-label="Dashboard Navigation">
            <ul role="menubar">
                <li role="none">
                    <a href="/profile" role="menuitem" aria-describedby="profile-desc">
                        Profile
                    </a>
                    <div id="profile-desc" class="sr-only">Manage your profile settings</div>
                </li>
                <li role="none">
                    <a href="/settings" role="menuitem" aria-describedby="settings-desc">
                        Settings
                    </a>
                    <div id="settings-desc" class="sr-only">Configure application preferences</div>
                </li>
            </ul>
        </nav>
    </header>
    
    <main class="dashboard-content">
        <section aria-labelledby="recent-activity">
            <h2 id="recent-activity">Recent Activity</h2>
            <div class="activity-list" role="log" aria-live="polite">
                <!-- Activity items will be loaded here -->
            </div>
        </section>
        
        <section aria-labelledby="quick-actions">
            <h2 id="quick-actions">Quick Actions</h2>
            <div class="action-buttons">
                <button type="button" class="btn btn-primary" 
                        aria-describedby="create-desc" onclick="createNew()">
                    Create New
                </button>
                <div id="create-desc" class="sr-only">Create a new item or document</div>
            </div>
        </section>
    </main>
</div>

<style>
.dashboard {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    font-family: system-ui, -apple-system, sans-serif;
}

.btn {
    padding: 12px 24px;
    border: none;
    border-radius: 4px;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-primary {
    background-color: #007bff;
    color: white;
}

.btn:hover, .btn:focus {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0,0,0,0);
    white-space: nowrap;
    border: 0;
}

@media (prefers-reduced-motion: reduce) {
    .btn {
        transition: none;
    }
    
    .btn:hover, .btn:focus {
        transform: none;
    }
}
</style>
            """,
            "language": "html",
            "priority": "medium",
            "context": {
                "application_type": "web_interface",
                "accessibility_required": True,
                "responsive_design": True,
                "user_facing": True
            }
        },
        
        "data_processing_algorithm": {
            "task_id": "integration_003",
            "description": "Optimize data processing for large datasets",
            "code": """
import asyncio
import aiohttp
import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, max_workers: int = 4, batch_size: int = 1000):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_large_dataset(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process large dataset efficiently using parallel processing.
        
        Args:
            data: List of data records to process
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {len(data)} records in batches of {self.batch_size}")
        
        # Split data into batches
        batches = [data[i:i + self.batch_size] 
                  for i in range(0, len(data), self.batch_size)]
        
        # Process batches in parallel
        tasks = []
        async with aiohttp.ClientSession() as session:
            for batch in batches:
                task = asyncio.create_task(
                    self._process_batch(session, batch)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        # Combine results
        combined_df = pd.concat(results, ignore_index=True)
        
        # Optimize memory usage
        combined_df = self._optimize_datatypes(combined_df)
        
        logger.info(f"Processing complete. Result shape: {combined_df.shape}")
        return combined_df
    
    async def _process_batch(self, session: aiohttp.ClientSession, 
                           batch: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a single batch of data."""
        # Simulate data enrichment from external API
        enriched_data = []
        
        for record in batch:
            # Add computed fields
            enriched_record = record.copy()
            enriched_record['processed_at'] = datetime.now().isoformat()
            enriched_record['record_hash'] = hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            enriched_data.append(enriched_record)
        
        return pd.DataFrame(enriched_data)
    
    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by selecting appropriate dtypes."""
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                except (ValueError, TypeError):
                    pass
        
        return df
            """,
            "language": "python",
            "priority": "high",
            "context": {
                "application_type": "data_processing",
                "performance_critical": True,
                "scalability_required": True,
                "memory_constrained": True
            }
        }
    }


# ============================================================================
# Board Consensus Integration Tests
# ============================================================================

class TestBoardConsensusIntegration:
    """Test the complete board consensus system."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_board_evaluation_process(self, full_board_setup, 
                                                 request_logger_with_db):
        """Test complete evaluation process with all directors."""
        registry, directors = full_board_setup
        tracker = request_logger_with_db
        
        # Submit a task for evaluation
        task_data = {
            "task_id": "consensus_test_001",
            "description": "Test multi-director evaluation",
            "code": "def secure_function(): return True",
            "language": "python",
            "priority": "medium"
        }
        
        # Get all director evaluations
        evaluations = {}
        for name, director in directors.items():
            evaluation = director.evaluate(task_data)
            evaluations[name] = evaluation
        
        # All directors should provide evaluations
        assert len(evaluations) == 5
        
        # Each evaluation should have required fields
        for director_name, evaluation in evaluations.items():
            assert "confidence" in evaluation
            assert "recommendation" in evaluation
            assert "reasoning" in evaluation
            assert isinstance(evaluation["findings"], list)
            assert 0 <= evaluation["confidence"] <= 100
            assert evaluation["recommendation"] in ["approved", "needs_review", "rejected"]
        
        # Test consensus calculation
        consensus = await self._calculate_consensus(evaluations)
        assert "overall_recommendation" in consensus
        assert "confidence_score" in consensus
        assert "agreement_level" in consensus

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complex_task_evaluation(self, full_board_setup, 
                                          complex_task_samples):
        """Test evaluation of complex, realistic tasks."""
        registry, directors = full_board_setup
        
        for task_name, task_data in complex_task_samples.items():
            evaluations = {}
            
            # Get evaluations from all directors
            for director_name, director in directors.items():
                start_time = time.time()
                evaluation = director.evaluate(task_data)
                end_time = time.time()
                
                evaluations[director_name] = evaluation
                
                # Performance check: should complete in reasonable time
                assert (end_time - start_time) < 10.0, \
                    f"{director_name} took too long on {task_name}"
            
            # Verify all evaluations are complete
            assert len(evaluations) == 5
            
            # Check for domain-specific insights
            security_eval = evaluations['security']
            quality_eval = evaluations['quality']
            performance_eval = evaluations['performance']
            ethics_eval = evaluations['ethics']
            ux_eval = evaluations['ux']
            
            # Each director should provide domain-specific findings
            assert len(security_eval["findings"]) > 0
            assert len(quality_eval["findings"]) > 0
            assert len(performance_eval["findings"]) > 0
            assert len(ethics_eval["findings"]) > 0
            assert len(ux_eval["findings"]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_consensus_with_conflicting_opinions(self, full_board_setup):
        """Test consensus when directors have conflicting opinions."""
        registry, directors = full_board_setup
        
        # Task that should produce mixed opinions
        controversial_task = {
            "task_id": "controversial_001",
            "description": "Implement user tracking for analytics",
            "code": """
def track_user_behavior(user_id, page, actions, personal_data):
    # Track everything for detailed analytics
    analytics_data = {
        'user_id': user_id,
        'page': page,
        'actions': actions,
        'personal_data': personal_data,  # Includes sensitive info
        'timestamp': datetime.now(),
        'ip_address': request.remote_addr,
        'browser_fingerprint': get_browser_fingerprint()
    }
    
    # Store in external analytics service
    send_to_analytics_service(analytics_data)
    
    return analytics_data
            """,
            "language": "python",
            "priority": "medium",
            "context": {
                "business_critical": True,
                "privacy_sensitive": True,
                "user_facing": True
            }
        }
        
        evaluations = {}
        for director_name, director in directors.items():
            evaluation = director.evaluate(controversial_task)
            evaluations[director_name] = evaluation
        
        # Should have mixed recommendations
        recommendations = [eval["recommendation"] for eval in evaluations.values()]
        unique_recommendations = set(recommendations)
        
        # Ethics director should likely flag privacy concerns
        ethics_eval = evaluations['ethics']
        assert ethics_eval["recommendation"] in ["needs_review", "rejected"]
        
        # Security director should flag data exposure risks
        security_eval = evaluations['security']
        assert any("privacy" in finding.lower() or "data" in finding.lower() 
                  for finding in security_eval["findings"])
        
        # Calculate consensus with conflicting opinions
        consensus = await self._calculate_consensus(evaluations)
        
        # Should indicate low agreement when opinions conflict
        assert consensus["agreement_level"] < 0.8  # Less than 80% agreement
        assert "needs_review" in consensus["overall_recommendation"]  # Should require review

    @pytest.mark.integration
    @pytest.mark.performance
    async def test_board_performance_under_load(self, full_board_setup):
        """Test board performance with multiple concurrent evaluations."""
        registry, directors = full_board_setup
        
        # Create multiple tasks for concurrent evaluation
        tasks = []
        for i in range(20):
            task = {
                "task_id": f"load_test_{i:03d}",
                "description": f"Load test task {i}",
                "code": f"def function_{i}(): return {i}",
                "language": "python",
                "priority": "normal"
            }
            tasks.append(task)
        
        start_time = time.time()
        
        # Process tasks concurrently
        all_results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for task in tasks:
                for director_name, director in directors.items():
                    future = executor.submit(director.evaluate, task)
                    futures.append((future, task["task_id"], director_name))
            
            # Collect results
            for future, task_id, director_name in futures:
                try:
                    result = future.result(timeout=10.0)
                    all_results.append((task_id, director_name, result))
                except Exception as e:
                    pytest.fail(f"Evaluation failed for {task_id} with {director_name}: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        expected_evaluations = len(tasks) * len(directors)  # 20 tasks * 5 directors = 100
        actual_evaluations = len(all_results)
        
        assert actual_evaluations == expected_evaluations
        assert total_time < 30.0  # Should complete within 30 seconds
        
        # Average time per evaluation should be reasonable
        avg_time_per_evaluation = total_time / actual_evaluations
        assert avg_time_per_evaluation < 2.0  # Less than 2 seconds per evaluation

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_decision_persistence(self, request_logger_with_db, full_board_setup):
        """Test that decisions are properly persisted and retrievable."""
        tracker = request_logger_with_db
        registry, directors = full_board_setup
        
        # Submit a task
        task_data = {
            "task_id": "persistence_test_001",
            "description": "Test decision persistence",
            "code": "def test(): pass",
            "language": "python",
            "user_id": "test_user"
        }
        
        # Mock the submission process
        task_id = await tracker.submit_task(
            task_data["description"],
            task_data["user_id"],
            "medium",
            task_data
        )
        
        assert task_id is not None
        
        # Retrieve the submitted task
        decision = await tracker.get_decision(task_id)
        assert decision is not None
        assert decision["task_id"] == task_id

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_feedback_integration(self, full_board_setup):
        """Test integration with feedback system."""
        registry, directors = full_board_setup
        
        # Mock feedback system
        with patch('directors.feedback_system.FeedbackSystem') as mock_feedback:
            feedback_instance = MagicMock()
            mock_feedback.return_value = feedback_instance
            
            # Evaluate a task
            task_data = {
                "task_id": "feedback_test_001",
                "description": "Test feedback integration",
                "code": "def example(): return 'test'",
                "language": "python"
            }
            
            evaluations = {}
            for director_name, director in directors.items():
                evaluation = director.evaluate(task_data)
                evaluations[director_name] = evaluation
            
            # Simulate feedback submission
            feedback_data = {
                "task_id": task_data["task_id"],
                "user_id": "test_user",
                "rating": 4,
                "comments": "Good evaluation, helpful insights",
                "director_ratings": {
                    "security": 5,
                    "quality": 4,
                    "performance": 4,
                    "ethics": 5,
                    "ux": 3
                }
            }
            
            # Verify feedback can be processed
            assert feedback_data["task_id"] == task_data["task_id"]
            assert all(1 <= rating <= 5 for rating in feedback_data["director_ratings"].values())

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _calculate_consensus(self, evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus from multiple director evaluations."""
        recommendations = []
        confidences = []
        
        for evaluation in evaluations.values():
            recommendations.append(evaluation["recommendation"])
            confidences.append(evaluation["confidence"])
        
        # Calculate agreement level
        from collections import Counter
        rec_counts = Counter(recommendations)
        most_common_rec, most_common_count = rec_counts.most_common(1)[0]
        agreement_level = most_common_count / len(recommendations)
        
        # Calculate weighted confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        # Determine overall recommendation
        if agreement_level >= 0.8:  # Strong consensus
            overall_recommendation = most_common_rec
        elif agreement_level >= 0.6:  # Moderate consensus
            overall_recommendation = "needs_review" if "needs_review" in recommendations else most_common_rec
        else:  # Weak consensus
            overall_recommendation = "needs_review"
        
        return {
            "overall_recommendation": overall_recommendation,
            "confidence_score": avg_confidence,
            "agreement_level": agreement_level,
            "individual_recommendations": recommendations,
            "individual_confidences": confidences
        }


# ============================================================================
# Cross-System Integration Tests
# ============================================================================

class TestCrossSystemIntegration:
    """Test integration between board system and external components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_endpoint_integration(self, full_board_setup):
        """Test integration with API endpoints."""
        registry, directors = full_board_setup
        
        # Mock API client
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "success"})
            
            mock_session.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            # Test that directors can work with API integrations
            task_with_api = {
                "task_id": "api_integration_001",
                "description": "Function that makes API calls",
                "code": """
async def fetch_user_data(user_id):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'/api/users/{user_id}') as response:
            return await response.json()
                """,
                "language": "python",
                "context": {
                    "api_integration": True,
                    "async_code": True
                }
            }
            
            # All directors should handle API-related code
            for director_name, director in directors.items():
                evaluation = director.evaluate(task_with_api)
                assert "confidence" in evaluation
                assert "recommendation" in evaluation

    @pytest.mark.integration
    @pytest.mark.database
    async def test_database_integration(self, full_board_setup):
        """Test integration with database operations."""
        registry, directors = full_board_setup
        
        task_with_db = {
            "task_id": "db_integration_001",
            "description": "Database operation function",
            "code": """
def get_user_by_email(email):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        user = session.query(User).filter(User.email == email).first()
        return user
    finally:
        session.close()
                """,
            "language": "python",
            "context": {
                "database_operation": True,
                "user_data": True
            }
        }
        
        # Directors should provide relevant feedback for database code
        evaluations = {}
        for director_name, director in directors.items():
            evaluation = director.evaluate(task_with_db)
            evaluations[director_name] = evaluation
        
        # Security director should check for SQL injection
        security_eval = evaluations['security']
        assert len(security_eval["findings"]) > 0
        
        # Performance director should consider query efficiency
        performance_eval = evaluations['performance']
        assert len(performance_eval["findings"]) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_long_running_evaluation(self, full_board_setup):
        """Test system behavior with long-running evaluations."""
        registry, directors = full_board_setup
        
        # Create a complex task that might take longer to evaluate
        complex_task = {
            "task_id": "long_running_001",
            "description": "Complex algorithmic implementation",
            "code": "\n".join([
                f"def complex_function_{i}():",
                f"    result = []",
                f"    for j in range(100):",
                f"        for k in range(100):",
                f"            result.append(j * k + {i})",
                f"    return result",
                ""
            ] for i in range(20)),  # 20 similar functions
            "language": "python",
            "context": {
                "algorithmic_complexity": "high",
                "code_size": "large"
            }
        }
        
        # All directors should handle large code samples
        start_time = time.time()
        
        evaluations = {}
        for director_name, director in directors.items():
            evaluation_start = time.time()
            evaluation = director.evaluate(complex_task)
            evaluation_end = time.time()
            
            evaluations[director_name] = evaluation
            
            # Individual evaluation should complete within reasonable time
            evaluation_time = evaluation_end - evaluation_start
            assert evaluation_time < 15.0, \
                f"{director_name} took {evaluation_time} seconds on complex task"
        
        total_time = time.time() - start_time
        
        # Total time should be reasonable
        assert total_time < 30.0
        
        # All evaluations should be complete and valid
        assert len(evaluations) == 5
        for director_name, evaluation in evaluations.items():
            assert "confidence" in evaluation
            assert "recommendation" in evaluation
            assert len(evaluation["findings"]) > 0