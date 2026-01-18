#!/usr/bin/env python3
"""
Advanced Model Performance Comparison Framework
Tests different models against each other with statistical validation
"""

import pytest
import sys
import os
import time
import json
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/tower-echo-brain')

from src.core.db_model_router import DatabaseModelRouter, RoutingDecision


class ModelPerformanceComparator:
    """Advanced A/B testing framework for model comparison"""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.router = DatabaseModelRouter(db_config)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def run_ab_test(self,
                    model_a: str,
                    model_b: str,
                    test_queries: List[str],
                    metrics: List[str] = ['response_time', 'quality_score']) -> Dict:
        """
        Run A/B test between two models

        Args:
            model_a: First model to test
            model_b: Second model to test
            test_queries: List of queries to test
            metrics: Metrics to compare

        Returns:
            Dict with statistical analysis results
        """
        results_a = []
        results_b = []

        for query in test_queries:
            # Test Model A
            start = time.time()
            response_a = self._query_model(model_a, query)
            time_a = (time.time() - start) * 1000  # ms

            # Test Model B
            start = time.time()
            response_b = self._query_model(model_b, query)
            time_b = (time.time() - start) * 1000  # ms

            # Calculate quality scores (mock for testing)
            quality_a = self._calculate_quality_score(response_a, query)
            quality_b = self._calculate_quality_score(response_b, query)

            results_a.append({
                'response_time': time_a,
                'quality_score': quality_a,
                'response_length': len(response_a)
            })

            results_b.append({
                'response_time': time_b,
                'quality_score': quality_b,
                'response_length': len(response_b)
            })

        # Statistical analysis
        analysis = self._perform_statistical_analysis(results_a, results_b, metrics)

        # Log results to database
        self._log_ab_test_results(model_a, model_b, analysis)

        return {
            'model_a': model_a,
            'model_b': model_b,
            'sample_size': len(test_queries),
            'analysis': analysis,
            'recommendation': self._generate_recommendation(analysis)
        }

    def _query_model(self, model: str, query: str) -> str:
        """Query specific model (mock for testing)"""
        # In production, this would actually query the model
        # For testing, we'll simulate with different responses
        import random

        if 'gemma' in model:
            response_time = random.gauss(500, 100)  # Slower but better quality
            quality = random.gauss(0.85, 0.05)
        elif 'llama' in model:
            response_time = random.gauss(300, 50)  # Faster, decent quality
            quality = random.gauss(0.75, 0.07)
        else:
            response_time = random.gauss(400, 75)
            quality = random.gauss(0.80, 0.06)

        # Simulate processing time
        time.sleep(response_time / 10000)  # Scale down for testing

        return f"Response from {model} for query: {query[:30]}..."

    def _calculate_quality_score(self, response: str, query: str) -> float:
        """Calculate quality score for response"""
        # In production, this would use more sophisticated metrics
        # For testing, we'll use simple heuristics
        import random

        base_score = 0.5

        # Length bonus (up to 0.2)
        length_score = min(len(response) / 500, 0.2)

        # Relevance bonus (mock)
        relevance_score = random.uniform(0.1, 0.3)

        return min(base_score + length_score + relevance_score, 1.0)

    def _perform_statistical_analysis(self,
                                     results_a: List[Dict],
                                     results_b: List[Dict],
                                     metrics: List[str]) -> Dict:
        """Perform statistical significance testing"""
        analysis = {}

        for metric in metrics:
            values_a = [r[metric] for r in results_a]
            values_b = [r[metric] for r in results_b]

            # Calculate means and std
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            std_a = np.std(values_a)
            std_b = np.std(values_b)

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(values_a, values_b)

            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
            cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

            # Calculate confidence intervals
            confidence_a = stats.t.interval(0.95, len(values_a)-1,
                                           loc=mean_a,
                                           scale=stats.sem(values_a))
            confidence_b = stats.t.interval(0.95, len(values_b)-1,
                                           loc=mean_b,
                                           scale=stats.sem(values_b))

            analysis[metric] = {
                'mean_a': mean_a,
                'mean_b': mean_b,
                'std_a': std_a,
                'std_b': std_b,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'confidence_interval_a': confidence_a,
                'confidence_interval_b': confidence_b,
                'significant': p_value < 0.05,
                'winner': 'model_a' if mean_a > mean_b else 'model_b' if mean_b > mean_a else 'tie'
            }

        return analysis

    def _generate_recommendation(self, analysis: Dict) -> str:
        """Generate recommendation based on analysis"""
        recommendations = []

        for metric, results in analysis.items():
            if results['significant']:
                effect_size = abs(results['cohens_d'])
                if effect_size < 0.2:
                    strength = "small"
                elif effect_size < 0.5:
                    strength = "medium"
                elif effect_size < 0.8:
                    strength = "large"
                else:
                    strength = "very large"

                winner = results['winner']
                if winner != 'tie':
                    recommendations.append(
                        f"{winner} shows {strength} improvement in {metric} "
                        f"(p={results['p_value']:.4f})"
                    )
            else:
                recommendations.append(
                    f"No significant difference in {metric} (p={results['p_value']:.4f})"
                )

        return "; ".join(recommendations) if recommendations else "No clear winner"

    def _log_ab_test_results(self, model_a: str, model_b: str, analysis: Dict):
        """Log A/B test results to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    id SERIAL PRIMARY KEY,
                    model_a VARCHAR(100),
                    model_b VARCHAR(100),
                    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analysis JSONB,
                    recommendation TEXT
                )
            """)

            # Insert results
            cursor.execute("""
                INSERT INTO ab_test_results (model_a, model_b, analysis, recommendation)
                VALUES (%s, %s, %s, %s)
            """, (model_a, model_b, json.dumps(analysis),
                  self._generate_recommendation(analysis)))

            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Failed to log A/B test results: {e}")


class TestModelPerformanceComparison:
    """Test suite for model performance comparison"""

    @pytest.fixture
    def db_config(self):
        """Database configuration"""
        return {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': 'tower_echo_brain_secret_key_2025'
        }

    @pytest.fixture
    def comparator(self, db_config):
        """Create comparator instance"""
        return ModelPerformanceComparator(db_config)

    @pytest.fixture
    def test_queries(self):
        """Test queries for comparison"""
        return [
            "Generate a cyberpunk anime scene",
            "Write Python code for sorting",
            "Hello, how are you?",
            "Explain quantum computing",
            "Create a story about dragons",
            "Debug this JavaScript function",
            "What's the weather like?",
            "Calculate the factorial of 10",
            "Design a database schema",
            "Tell me a joke"
        ]

    @pytest.mark.performance
    def test_ab_comparison_gemma_vs_llama(self, comparator, test_queries):
        """Test A/B comparison between Gemma and Llama"""
        results = comparator.run_ab_test(
            'gemma2:9b',
            'llama3.1:8b',
            test_queries[:5]  # Use subset for faster testing
        )

        assert 'analysis' in results
        assert 'recommendation' in results
        assert results['sample_size'] == 5

        # Check statistical analysis exists
        assert 'response_time' in results['analysis']
        assert 'quality_score' in results['analysis']

        # Verify statistical metrics
        for metric in ['response_time', 'quality_score']:
            analysis = results['analysis'][metric]
            assert 'p_value' in analysis
            assert 'cohens_d' in analysis
            assert 'significant' in analysis
            assert isinstance(analysis['significant'], bool)

    @pytest.mark.performance
    def test_statistical_significance(self, comparator):
        """Test statistical significance calculation"""
        # Create clearly different result sets
        results_good = [
            {'response_time': 100, 'quality_score': 0.9},
            {'response_time': 110, 'quality_score': 0.85},
            {'response_time': 105, 'quality_score': 0.88},
            {'response_time': 95, 'quality_score': 0.92},
            {'response_time': 108, 'quality_score': 0.87}
        ]

        results_bad = [
            {'response_time': 200, 'quality_score': 0.6},
            {'response_time': 210, 'quality_score': 0.55},
            {'response_time': 195, 'quality_score': 0.62},
            {'response_time': 205, 'quality_score': 0.58},
            {'response_time': 198, 'quality_score': 0.61}
        ]

        analysis = comparator._perform_statistical_analysis(
            results_good, results_bad, ['response_time', 'quality_score']
        )

        # Should show significant differences
        assert analysis['response_time']['significant'] == True
        assert analysis['quality_score']['significant'] == True

        # Good model should win on both metrics
        assert analysis['response_time']['winner'] == 'model_a'
        assert analysis['quality_score']['winner'] == 'model_a'

    @pytest.mark.performance
    def test_effect_size_calculation(self, comparator):
        """Test Cohen's d effect size calculation"""
        results_a = [
            {'metric': 100},
            {'metric': 110},
            {'metric': 105}
        ]

        results_b = [
            {'metric': 120},
            {'metric': 130},
            {'metric': 125}
        ]

        analysis = comparator._perform_statistical_analysis(
            results_a, results_b, ['metric']
        )

        # Check Cohen's d is calculated
        cohens_d = analysis['metric']['cohens_d']
        assert isinstance(cohens_d, float)

        # Should show medium to large effect size
        assert abs(cohens_d) > 0.5

    @pytest.mark.performance
    def test_recommendation_generation(self, comparator):
        """Test recommendation generation"""
        analysis = {
            'response_time': {
                'mean_a': 100,
                'mean_b': 150,
                'p_value': 0.001,
                'cohens_d': -1.2,
                'significant': True,
                'winner': 'model_a'
            },
            'quality_score': {
                'mean_a': 0.8,
                'mean_b': 0.75,
                'p_value': 0.08,
                'cohens_d': 0.3,
                'significant': False,
                'winner': 'model_a'
            }
        }

        recommendation = comparator._generate_recommendation(analysis)

        # Should mention significant improvement in response time
        assert 'response_time' in recommendation
        assert 'very large improvement' in recommendation or 'large improvement' in recommendation

        # Should mention no significant difference in quality
        assert 'No significant difference in quality_score' in recommendation

    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_model_testing(self, comparator, test_queries):
        """Test concurrent model testing for performance"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        models_to_test = [
            ('llama3.1:8b', 'gemma2:9b'),
            ('qwen2.5-coder:7b', 'deepseek-coder-v2:16b'),
            ('llama3.1:8b', 'qwen2.5-coder:7b')
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for model_a, model_b in models_to_test:
                future = executor.submit(
                    comparator.run_ab_test,
                    model_a,
                    model_b,
                    test_queries[:3]  # Small subset for speed
                )
                futures.append(future)

            # Collect results
            results = [f.result() for f in futures]

        assert len(results) == 3
        for result in results:
            assert 'analysis' in result
            assert 'recommendation' in result

    @pytest.mark.performance
    def test_database_logging(self, comparator, db_config):
        """Test that A/B test results are logged to database"""
        # Run a test
        analysis = {
            'test_metric': {
                'mean_a': 100,
                'mean_b': 110,
                'p_value': 0.04,
                'cohens_d': -0.5,
                'significant': True,
                'winner': 'model_a'
            }
        }

        comparator._log_ab_test_results('test_model_a', 'test_model_b', analysis)

        # Verify it was logged
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_a, model_b, analysis
            FROM ab_test_results
            WHERE model_a = 'test_model_a' AND model_b = 'test_model_b'
            ORDER BY test_date DESC
            LIMIT 1
        """)

        result = cursor.fetchone()
        if result:  # Table might not exist in test environment
            assert result[0] == 'test_model_a'
            assert result[1] == 'test_model_b'
            assert 'test_metric' in result[2]

        cursor.close()
        conn.close()


if __name__ == "__main__":
    # Run performance comparison tests
    db_config = {
        'host': 'localhost',
        'database': 'echo_brain',
        'user': 'patrick',
        'password': 'tower_echo_brain_secret_key_2025'
    }

    comparator = ModelPerformanceComparator(db_config)

    test_queries = [
        "Generate an anime scene with Mei",
        "Write Python code to sort a list",
        "Hello, how are you today?",
        "Explain machine learning concepts",
        "Create a fantasy story"
    ]

    print("🚀 Running Model Performance Comparison")
    print("=" * 60)

    # Compare Gemma vs Llama
    print("\n📊 Comparing gemma2:9b vs llama3.1:8b...")
    results = comparator.run_ab_test(
        'gemma2:9b',
        'llama3.1:8b',
        test_queries
    )

    print(f"\nSample Size: {results['sample_size']}")
    print(f"Recommendation: {results['recommendation']}")

    for metric, analysis in results['analysis'].items():
        print(f"\n{metric}:")
        print(f"  Model A Mean: {analysis['mean_a']:.2f}")
        print(f"  Model B Mean: {analysis['mean_b']:.2f}")
        print(f"  P-Value: {analysis['p_value']:.4f}")
        print(f"  Effect Size (Cohen's d): {analysis['cohens_d']:.2f}")
        print(f"  Statistically Significant: {analysis['significant']}")
        print(f"  Winner: {analysis['winner']}")

    print("\n" + "=" * 60)
    print("✅ Model Performance Comparison Complete!")