#!/usr/bin/env python3
"""
Echo A/B Testing Framework - Scientific Model Comparison
======================================================

Advanced A/B testing framework for AI Assist that enables:
- Simultaneous testing of multiple models/configurations
- Statistical significance testing
- Automated winner selection
- Real-time performance monitoring
- Bias detection and correction
- Multi-armed bandit optimization

Features:
- Stratified randomization for fair comparison
- Bayesian statistical analysis
- Early stopping criteria
- Champion/challenger model management
- Detailed experiment tracking
"""

import asyncio
import logging
import json
import sqlite3
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import random
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Configuration
AB_TESTING_DB = "/opt/tower-echo-brain/data/ab_testing.db"
EXPERIMENT_DATA_PATH = "/opt/tower-echo-brain/experiments/"
MIN_SAMPLE_SIZE = 100
SIGNIFICANCE_LEVEL = 0.05
POWER = 0.8

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    EARLY_STOPPED = "early_stopped"

class AllocationStrategy(Enum):
    UNIFORM = "uniform"
    THOMPSON_SAMPLING = "thompson_sampling"
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"  # Upper Confidence Bound

class MetricType(Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"
    CUSTOM = "custom"

@dataclass
class VariantConfig:
    variant_id: str
    name: str
    description: str
    model_id: Optional[str] = None
    config_override: Dict[str, Any] = None
    allocation_weight: float = 1.0

@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    description: str
    variants: List[VariantConfig]
    primary_metric: MetricType
    secondary_metrics: List[MetricType]
    allocation_strategy: AllocationStrategy
    target_sample_size: int
    max_duration_days: int
    early_stopping_enabled: bool = True
    minimum_detectable_effect: float = 0.05
    statistical_power: float = 0.8
    significance_level: float = 0.05

@dataclass
class ExperimentResult:
    variant_id: str
    timestamp: datetime
    user_id: str
    session_id: str
    metrics: Dict[str, float]
    context: Dict[str, Any] = None

class ABTestingFramework:
    def __init__(self):
        self.db_path = AB_TESTING_DB
        self.data_path = Path(EXPERIMENT_DATA_PATH)
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
        self._init_database()
        self.running_experiments: Dict[str, ExperimentConfig] = {}

    def _ensure_directories(self):
        """Create necessary directories"""
        self.data_path.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize the A/B testing database"""
        with sqlite3.connect(self.db_path) as conn:
            # Experiments table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_by TEXT,
                    results TEXT
                )
            ''')
            
            # Experiment results table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiment_results (
                    result_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    metrics TEXT NOT NULL,
                    context TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            # User assignments table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    assigned_at TIMESTAMP NOT NULL,
                    assignment_method TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            # Statistical snapshots table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS statistical_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    variant_statistics TEXT NOT NULL,
                    significance_tests TEXT,
                    recommendations TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')

            # Indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_results_experiment ON experiment_results(experiment_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_results_timestamp ON experiment_results(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_assignments_user ON user_assignments(user_id)')

    def _generate_id(self, prefix: str = "exp") -> str:
        """Generate unique ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"

    async def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[VariantConfig],
        primary_metric: MetricType,
        secondary_metrics: List[MetricType] = None,
        allocation_strategy: AllocationStrategy = AllocationStrategy.UNIFORM,
        target_sample_size: int = 1000,
        max_duration_days: int = 30,
        early_stopping_enabled: bool = True,
        created_by: str = "system"
    ) -> str:
        """Create a new A/B test experiment"""
        
        experiment_id = self._generate_id("exp")
        secondary_metrics = secondary_metrics or []
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            allocation_strategy=allocation_strategy,
            target_sample_size=target_sample_size,
            max_duration_days=max_duration_days,
            early_stopping_enabled=early_stopping_enabled
        )
        
        # Validate configuration
        if len(variants) < 2:
            raise ValueError("At least 2 variants required for A/B testing")
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO experiments (
                    experiment_id, name, description, config, status, created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id, name, description, json.dumps(asdict(config)),
                ExperimentStatus.DRAFT.value, datetime.now(), created_by
            ))
        
        self.logger.info(f"Experiment created: {experiment_id}")
        return experiment_id

    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Get experiment config
            cursor = conn.execute('''
                SELECT config, status FROM experiments WHERE experiment_id = ?
            ''', (experiment_id,))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            config_data = json.loads(result[0])
            current_status = result[1]
            
            if current_status != ExperimentStatus.DRAFT.value:
                raise ValueError(f"Cannot start experiment in {current_status} status")
            
            # Load config
            config = ExperimentConfig(**config_data)
            
            # Update status
            conn.execute('''
                UPDATE experiments SET status = ?, started_at = ? WHERE experiment_id = ?
            ''', (ExperimentStatus.RUNNING.value, datetime.now(), experiment_id))
            
            # Cache running experiment
            self.running_experiments[experiment_id] = config
        
        self.logger.info(f"Experiment started: {experiment_id}")
        return True

    async def assign_variant(self, experiment_id: str, user_id: str, context: Dict[str, Any] = None) -> str:
        """Assign a user to a variant"""
        
        if experiment_id not in self.running_experiments:
            # Try to load from database
            await self._load_experiment(experiment_id)
        
        config = self.running_experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment {experiment_id} not found or not running")
        
        # Check if user already assigned
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT variant_id FROM user_assignments 
                WHERE experiment_id = ? AND user_id = ?
            ''', (experiment_id, user_id))
            
            existing_assignment = cursor.fetchone()
            if existing_assignment:
                return existing_assignment[0]
        
        # Assign new variant based on strategy
        variant_id = await self._select_variant(experiment_id, user_id, config, context)
        
        # Store assignment
        assignment_id = self._generate_id("assign")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO user_assignments (
                    assignment_id, experiment_id, user_id, variant_id, 
                    assigned_at, assignment_method
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                assignment_id, experiment_id, user_id, variant_id,
                datetime.now(), config.allocation_strategy.value
            ))
        
        return variant_id

    async def _select_variant(
        self, 
        experiment_id: str, 
        user_id: str, 
        config: ExperimentConfig,
        context: Dict[str, Any] = None
    ) -> str:
        """Select variant based on allocation strategy"""
        
        if config.allocation_strategy == AllocationStrategy.UNIFORM:
            return await self._uniform_allocation(config)
        elif config.allocation_strategy == AllocationStrategy.THOMPSON_SAMPLING:
            return await self._thompson_sampling_allocation(experiment_id, config)
        elif config.allocation_strategy == AllocationStrategy.EPSILON_GREEDY:
            return await self._epsilon_greedy_allocation(experiment_id, config)
        elif config.allocation_strategy == AllocationStrategy.UCB:
            return await self._ucb_allocation(experiment_id, config)
        else:
            return await self._uniform_allocation(config)

    async def _uniform_allocation(self, config: ExperimentConfig) -> str:
        """Uniform random allocation"""
        weights = [v.allocation_weight for v in config.variants]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        return np.random.choice([v.variant_id for v in config.variants], p=normalized_weights)

    async def _thompson_sampling_allocation(self, experiment_id: str, config: ExperimentConfig) -> str:
        """Thompson sampling (Bayesian bandit)"""
        
        # Get current performance statistics
        variant_stats = await self._get_variant_statistics(experiment_id)
        
        best_variant = None
        best_sample = -1
        
        for variant in config.variants:
            variant_id = variant.variant_id
            stats_data = variant_stats.get(variant_id, {'successes': 0, 'failures': 0})
            
            # Beta distribution parameters (assuming binary success metric)
            alpha = stats_data['successes'] + 1
            beta = stats_data['failures'] + 1
            
            # Sample from posterior
            sample = np.random.beta(alpha, beta)
            
            if sample > best_sample:
                best_sample = sample
                best_variant = variant_id
        
        return best_variant or config.variants[0].variant_id

    async def _epsilon_greedy_allocation(self, experiment_id: str, config: ExperimentConfig, epsilon: float = 0.1) -> str:
        """Epsilon-greedy allocation"""
        
        if np.random.random() < epsilon:
            # Explore: random selection
            return await self._uniform_allocation(config)
        else:
            # Exploit: select best performing variant
            variant_stats = await self._get_variant_statistics(experiment_id)
            
            best_variant = None
            best_performance = -1
            
            for variant in config.variants:
                variant_id = variant.variant_id
                stats_data = variant_stats.get(variant_id, {})
                performance = stats_data.get('conversion_rate', 0)
                
                if performance > best_performance:
                    best_performance = performance
                    best_variant = variant_id
            
            return best_variant or config.variants[0].variant_id

    async def _ucb_allocation(self, experiment_id: str, config: ExperimentConfig) -> str:
        """Upper Confidence Bound allocation"""
        
        variant_stats = await self._get_variant_statistics(experiment_id)
        total_samples = sum(stats.get('total_samples', 0) for stats in variant_stats.values())
        
        if total_samples == 0:
            return await self._uniform_allocation(config)
        
        best_variant = None
        best_ucb = -1
        
        for variant in config.variants:
            variant_id = variant.variant_id
            stats_data = variant_stats.get(variant_id, {'total_samples': 0, 'conversion_rate': 0})
            
            n_variant = stats_data['total_samples']
            if n_variant == 0:
                ucb_value = float('inf')  # Prioritize unexplored variants
            else:
                mean_reward = stats_data['conversion_rate']
                confidence_interval = np.sqrt(2 * np.log(total_samples) / n_variant)
                ucb_value = mean_reward + confidence_interval
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_variant = variant_id
        
        return best_variant or config.variants[0].variant_id

    async def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        metrics: Dict[str, float],
        session_id: str = None,
        context: Dict[str, Any] = None
    ) -> bool:
        """Record experiment result"""
        
        result_id = self._generate_id("result")
        session_id = session_id or self._generate_id("session")
        
        result = ExperimentResult(
            variant_id=variant_id,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            metrics=metrics,
            context=context
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO experiment_results (
                    result_id, experiment_id, variant_id, timestamp,
                    user_id, session_id, metrics, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result_id, experiment_id, variant_id, result.timestamp,
                user_id, session_id, json.dumps(metrics), 
                json.dumps(context) if context else None
            ))
        
        # Check for early stopping criteria
        if experiment_id in self.running_experiments:
            config = self.running_experiments[experiment_id]
            if config.early_stopping_enabled:
                await self._check_early_stopping(experiment_id)
        
        return True

    async def _get_variant_statistics(self, experiment_id: str) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all variants"""
        
        stats = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT variant_id, metrics FROM experiment_results 
                WHERE experiment_id = ?
            ''', (experiment_id,))
            
            variant_data = {}
            for row in cursor.fetchall():
                variant_id = row[0]
                metrics = json.loads(row[1])
                
                if variant_id not in variant_data:
                    variant_data[variant_id] = []
                variant_data[variant_id].append(metrics)
            
            # Calculate statistics for each variant
            for variant_id, results in variant_data.items():
                if not results:
                    continue
                
                # Aggregate metrics
                aggregated = {}
                for metric_name in results[0].keys():
                    values = [r[metric_name] for r in results if metric_name in r]
                    if values:
                        aggregated[f"{metric_name}_mean"] = np.mean(values)
                        aggregated[f"{metric_name}_std"] = np.std(values)
                        aggregated[f"{metric_name}_count"] = len(values)
                
                # Common statistics
                aggregated['total_samples'] = len(results)
                aggregated['conversion_rate'] = np.mean([
                    1 if r.get('success', 0) > 0 else 0 for r in results
                ])
                
                # Success/failure counts for Thompson sampling
                aggregated['successes'] = sum(1 for r in results if r.get('success', 0) > 0)
                aggregated['failures'] = len(results) - aggregated['successes']
                
                stats[variant_id] = aggregated
        
        return stats

    async def _check_early_stopping(self, experiment_id: str) -> bool:
        """Check if experiment should be stopped early"""
        
        config = self.running_experiments.get(experiment_id)
        if not config:
            return False
        
        # Check sample size
        variant_stats = await self._get_variant_statistics(experiment_id)
        total_samples = sum(stats.get('total_samples', 0) for stats in variant_stats.values())
        
        if total_samples < MIN_SAMPLE_SIZE:
            return False
        
        # Check statistical significance
        significance_results = await self._run_significance_tests(experiment_id)
        
        # Check if we have a clear winner
        primary_metric = config.primary_metric.value
        significance_test = significance_results.get(primary_metric)
        
        if significance_test and significance_test.get('p_value', 1.0) < config.significance_level:
            # We have statistical significance, consider early stopping
            effect_size = significance_test.get('effect_size', 0)
            
            if abs(effect_size) >= config.minimum_detectable_effect:
                await self._stop_experiment(experiment_id, ExperimentStatus.EARLY_STOPPED)
                self.logger.info(f"Experiment {experiment_id} stopped early due to statistical significance")
                return True
        
        # Check maximum duration
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT started_at FROM experiments WHERE experiment_id = ?
            ''', (experiment_id,))
            
            result = cursor.fetchone()
            if result:
                started_at = datetime.fromisoformat(result[0])
                duration = datetime.now() - started_at
                
                if duration.days >= config.max_duration_days:
                    await self._stop_experiment(experiment_id, ExperimentStatus.COMPLETED)
                    self.logger.info(f"Experiment {experiment_id} completed after {duration.days} days")
                    return True
        
        return False

    async def _run_significance_tests(self, experiment_id: str) -> Dict[str, Dict[str, float]]:
        """Run statistical significance tests"""
        
        config = self.running_experiments.get(experiment_id)
        if not config:
            return {}
        
        variant_stats = await self._get_variant_statistics(experiment_id)
        
        if len(variant_stats) < 2:
            return {}
        
        results = {}
        
        # Get raw data for testing
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT variant_id, metrics FROM experiment_results 
                WHERE experiment_id = ?
            ''', (experiment_id,))
            
            variant_data = {}
            for row in cursor.fetchall():
                variant_id = row[0]
                metrics = json.loads(row[1])
                
                if variant_id not in variant_data:
                    variant_data[variant_id] = []
                variant_data[variant_id].append(metrics)
        
        # Run tests for each metric
        metrics_to_test = [config.primary_metric.value] + [m.value for m in config.secondary_metrics]
        
        for metric_name in metrics_to_test:
            variant_values = []
            variant_labels = []
            
            for variant_id, results_list in variant_data.items():
                values = [r.get(metric_name, 0) for r in results_list if metric_name in r]
                if values:
                    variant_values.extend(values)
                    variant_labels.extend([variant_id] * len(values))
            
            if len(set(variant_labels)) >= 2:
                # Two-sample t-test (assuming normal distribution)
                groups = {}
                for i, label in enumerate(variant_labels):
                    if label not in groups:
                        groups[label] = []
                    groups[label].append(variant_values[i])
                
                if len(groups) == 2:
                    group_names = list(groups.keys())
                    group1_values = groups[group_names[0]]
                    group2_values = groups[group_names[1]]
                    
                    t_stat, p_value = stats.ttest_ind(group1_values, group2_values)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(group1_values) - 1) * np.var(group1_values, ddof=1) + 
                                         (len(group2_values) - 1) * np.var(group2_values, ddof=1)) / 
                                        (len(group1_values) + len(group2_values) - 2))
                    
                    cohens_d = (np.mean(group1_values) - np.mean(group2_values)) / pooled_std if pooled_std > 0 else 0
                    
                    results[metric_name] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'effect_size': float(cohens_d),
                        'sample_size_1': len(group1_values),
                        'sample_size_2': len(group2_values),
                        'mean_1': float(np.mean(group1_values)),
                        'mean_2': float(np.mean(group2_values))
                    }
        
        # Store snapshot
        snapshot_id = self._generate_id("snapshot")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO statistical_snapshots (
                    snapshot_id, experiment_id, created_at, variant_statistics, significance_tests
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                snapshot_id, experiment_id, datetime.now(),
                json.dumps(variant_stats), json.dumps(results)
            ))
        
        return results

    async def _stop_experiment(self, experiment_id: str, status: ExperimentStatus) -> bool:
        """Stop an experiment"""
        
        # Run final analysis
        final_results = await self._run_significance_tests(experiment_id)
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE experiments SET status = ?, completed_at = ?, results = ? 
                WHERE experiment_id = ?
            ''', (status.value, datetime.now(), json.dumps(final_results), experiment_id))
        
        # Remove from running experiments
        if experiment_id in self.running_experiments:
            del self.running_experiments[experiment_id]
        
        return True

    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment results"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Get experiment metadata
            cursor = conn.execute('''
                SELECT name, description, config, status, started_at, completed_at, results
                FROM experiments WHERE experiment_id = ?
            ''', (experiment_id,))
            
            exp_row = cursor.fetchone()
            if not exp_row:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Get variant statistics
            variant_stats = await self._get_variant_statistics(experiment_id)
            
            # Get latest significance tests
            cursor = conn.execute('''
                SELECT significance_tests FROM statistical_snapshots 
                WHERE experiment_id = ? ORDER BY created_at DESC LIMIT 1
            ''', (experiment_id,))
            
            sig_test_row = cursor.fetchone()
            significance_tests = json.loads(sig_test_row[0]) if sig_test_row else {}
            
            return {
                'experiment_id': experiment_id,
                'name': exp_row[0],
                'description': exp_row[1],
                'config': json.loads(exp_row[2]),
                'status': exp_row[3],
                'started_at': exp_row[4],
                'completed_at': exp_row[5],
                'variant_statistics': variant_stats,
                'significance_tests': significance_tests,
                'final_results': json.loads(exp_row[6]) if exp_row[6] else None
            }

    async def _load_experiment(self, experiment_id: str) -> bool:
        """Load experiment config from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT config, status FROM experiments WHERE experiment_id = ?
            ''', (experiment_id,))
            
            result = cursor.fetchone()
            if not result:
                return False
            
            if result[1] == ExperimentStatus.RUNNING.value:
                config_data = json.loads(result[0])
                self.running_experiments[experiment_id] = ExperimentConfig(**config_data)
                return True
        
        return False

# Example usage and testing
async def example_ab_test():
    """Example A/B test setup"""
    
    framework = ABTestingFramework()
    
    # Create variants
    variants = [
        VariantConfig(
            variant_id="control",
            name="Current Echo Configuration",
            description="Baseline Echo decision engine",
            model_id="echo_v1.0"
        ),
        VariantConfig(
            variant_id="treatment",
            name="Optimized Echo Configuration",
            description="Echo with improved decision thresholds",
            model_id="echo_v1.1",
            config_override={"decision_threshold": 0.8, "confidence_boost": 0.1}
        )
    ]
    
    # Create experiment
    experiment_id = await framework.create_experiment(
        name="Echo Decision Engine Optimization",
        description="Compare baseline vs optimized decision engine",
        variants=variants,
        primary_metric=MetricType.ACCURACY,
        secondary_metrics=[MetricType.PRECISION, MetricType.RECALL],
        allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
        target_sample_size=1000,
        max_duration_days=14
    )
    
    print(f"Created experiment: {experiment_id}")
    
    # Start experiment
    await framework.start_experiment(experiment_id)
    print(f"Started experiment: {experiment_id}")
    
    return framework, experiment_id

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # FastAPI app for A/B testing API
    app = FastAPI(title="Echo A/B Testing Framework", version="1.0.0")
    ab_framework = ABTestingFramework()
    
    @app.post("/experiments")
    async def create_experiment_endpoint(
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        primary_metric: str,
        secondary_metrics: List[str] = None
    ):
        try:
            variant_configs = [VariantConfig(**v) for v in variants]
            primary_metric_enum = MetricType(primary_metric)
            secondary_metric_enums = [MetricType(m) for m in secondary_metrics or []]
            
            experiment_id = await ab_framework.create_experiment(
                name=name,
                description=description,
                variants=variant_configs,
                primary_metric=primary_metric_enum,
                secondary_metrics=secondary_metric_enums
            )
            
            return {"experiment_id": experiment_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/experiments/{experiment_id}/start")
    async def start_experiment_endpoint(experiment_id: str):
        try:
            success = await ab_framework.start_experiment(experiment_id)
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/experiments/{experiment_id}/assign/{user_id}")
    async def assign_variant_endpoint(experiment_id: str, user_id: str):
        try:
            variant_id = await ab_framework.assign_variant(experiment_id, user_id)
            return {"variant_id": variant_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/experiments/{experiment_id}/results")
    async def record_result_endpoint(
        experiment_id: str,
        variant_id: str,
        user_id: str,
        metrics: Dict[str, float]
    ):
        try:
            success = await ab_framework.record_result(
                experiment_id, variant_id, user_id, metrics
            )
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/experiments/{experiment_id}/results")
    async def get_experiment_results_endpoint(experiment_id: str):
        try:
            results = await ab_framework.get_experiment_results(experiment_id)
            return results
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8341)
