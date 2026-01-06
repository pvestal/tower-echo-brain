#!/usr/bin/env python3
"""
Echo Automated Retraining Pipeline
=================================

Intelligent automated retraining system for Echo Brain that:
- Monitors performance degradation triggers
- Orchestrates data collection and preparation
- Performs hyperparameter optimization
- Validates new models before deployment
- Manages the entire retraining lifecycle
- Integrates with drift detection and model registry

Features:
- Scheduled and triggered retraining
- Incremental learning capabilities
- Automated hyperparameter tuning
- Model validation and A/B testing
- Rollback capabilities
- Performance tracking and optimization
"""

import asyncio
import logging
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import yaml
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import optuna
import mlflow
import mlflow.sklearn

# Configuration
RETRAINING_DB = "/opt/tower-echo-brain/data/retraining_pipeline.db"
RETRAINING_DATA_PATH = "/opt/tower-echo-brain/retraining/"
MODEL_ARTIFACTS_PATH = "/opt/tower-echo-brain/retraining/artifacts/"
TRAINING_LOGS_PATH = "/opt/tower-echo-brain/retraining/logs/"

class RetrainingTrigger(Enum):
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DRIFT_DETECTION = "drift_detection"
    MANUAL = "manual"
    DATA_AVAILABILITY = "data_availability"

class RetrainingStatus(Enum):
    PENDING = "pending"
    DATA_COLLECTION = "data_collection"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelType(Enum):
    DECISION_ENGINE = "decision_engine"
    CLASSIFIER = "classifier"
    REGRESSION = "regression"
    ENSEMBLE = "ensemble"

@dataclass
class RetrainingConfig:
    model_name: str
    model_type: ModelType
    trigger_conditions: Dict[str, Any]
    training_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    hyperparameter_space: Dict[str, Any]
    deployment_config: Dict[str, Any]
    enabled: bool = True

@dataclass
class RetrainingJob:
    job_id: str
    model_name: str
    trigger: RetrainingTrigger
    trigger_reason: str
    status: RetrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config: Optional[RetrainingConfig] = None
    metrics: Dict[str, float] = None
    artifacts: Dict[str, str] = None
    error_message: Optional[str] = None

class AutomatedRetrainingPipeline:
    def __init__(self):
        self.db_path = RETRAINING_DB
        self.data_path = Path(RETRAINING_DATA_PATH)
        self.artifacts_path = Path(MODEL_ARTIFACTS_PATH)
        self.logs_path = Path(TRAINING_LOGS_PATH)
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
        self._init_database()
        self.model_configs: Dict[str, RetrainingConfig] = {}
        self.active_jobs: Dict[str, RetrainingJob] = {}

    def _ensure_directories(self):
        """Create necessary directories"""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize retraining pipeline database"""
        with sqlite3.connect(self.db_path) as conn:
            # Retraining jobs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS retraining_jobs (
                    job_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    trigger_reason TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    config TEXT,
                    metrics TEXT,
                    artifacts TEXT,
                    error_message TEXT
                )
            ''')
            
            # Model configurations table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_configs (
                    config_id TEXT PRIMARY KEY,
                    model_name TEXT UNIQUE NOT NULL,
                    config TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            ''')
            
            # Training data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    data_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    file_path TEXT,
                    data_hash TEXT,
                    sample_count INTEGER,
                    feature_count INTEGER,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES retraining_jobs (job_id)
                )
            ''')
            
            # Hyperparameter trials table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS hyperparameter_trials (
                    trial_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    trial_number INTEGER,
                    parameters TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES retraining_jobs (job_id)
                )
            ''')
            
            # Performance monitoring table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_monitoring (
                    monitor_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold_value REAL,
                    trigger_retraining BOOLEAN DEFAULT 0
                )
            ''')

            # Indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status ON retraining_jobs(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_jobs_model ON retraining_jobs(model_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_monitoring_model ON performance_monitoring(model_name)')

    def _generate_id(self, prefix: str = "retrain") -> str:
        """Generate unique ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        import hashlib
        import random
        random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"

    async def configure_model_retraining(
        self,
        model_name: str,
        model_type: ModelType,
        trigger_conditions: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None,
        validation_config: Dict[str, Any] = None,
        hyperparameter_space: Dict[str, Any] = None,
        deployment_config: Dict[str, Any] = None
    ) -> bool:
        """Configure automated retraining for a model"""
        
        # Default configurations
        trigger_conditions = trigger_conditions or {
            "performance_threshold": 0.05,  # 5% degradation
            "drift_threshold": 0.1,
            "schedule": "weekly",
            "min_data_samples": 1000
        }
        
        training_config = training_config or {
            "test_size": 0.2,
            "validation_size": 0.2,
            "random_state": 42,
            "cross_validation_folds": 5
        }
        
        validation_config = validation_config or {
            "metrics": ["accuracy", "precision", "recall", "f1_score"],
            "minimum_improvement": 0.01,
            "validation_strategy": "holdout"
        }
        
        hyperparameter_space = hyperparameter_space or self._get_default_hyperparameter_space(model_type)
        
        deployment_config = deployment_config or {
            "staging_validation": True,
            "ab_test_percentage": 10,
            "rollback_on_failure": True
        }
        
        config = RetrainingConfig(
            model_name=model_name,
            model_type=model_type,
            trigger_conditions=trigger_conditions,
            training_config=training_config,
            validation_config=validation_config,
            hyperparameter_space=hyperparameter_space,
            deployment_config=deployment_config
        )
        
        # Store configuration
        config_id = self._generate_id("config")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO model_configs (
                    config_id, model_name, config, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                config_id, model_name, json.dumps(asdict(config)),
                datetime.now(), datetime.now()
            ))
        
        self.model_configs[model_name] = config
        self.logger.info(f"Configured retraining for model: {model_name}")
        return True

    def _get_default_hyperparameter_space(self, model_type: ModelType) -> Dict[str, Any]:
        """Get default hyperparameter space for model type"""
        
        if model_type == ModelType.CLASSIFIER:
            return {
                "random_forest": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "gradient_boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                },
                "logistic_regression": {
                    "C": [0.1, 1.0, 10.0, 100.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"]
                }
            }
        else:
            return {}

    async def trigger_retraining(
        self,
        model_name: str,
        trigger: RetrainingTrigger,
        trigger_reason: str = "",
        priority: int = 1
    ) -> str:
        """Trigger a retraining job"""
        
        if model_name not in self.model_configs:
            raise ValueError(f"No configuration found for model: {model_name}")
        
        job_id = self._generate_id("job")
        
        job = RetrainingJob(
            job_id=job_id,
            model_name=model_name,
            trigger=trigger,
            trigger_reason=trigger_reason,
            status=RetrainingStatus.PENDING,
            created_at=datetime.now(),
            config=self.model_configs[model_name]
        )
        
        # Store job in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO retraining_jobs (
                    job_id, model_name, trigger_type, trigger_reason, status, created_at, config
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_id, model_name, trigger.value, trigger_reason,
                RetrainingStatus.PENDING.value, job.created_at, json.dumps(asdict(job.config))
            ))
        
        self.active_jobs[job_id] = job
        
        # Start retraining asynchronously
        asyncio.create_task(self._execute_retraining_job(job_id))
        
        self.logger.info(f"Triggered retraining job: {job_id} for model: {model_name}")
        return job_id

    async def _execute_retraining_job(self, job_id: str) -> bool:
        """Execute a complete retraining job"""
        
        job = self.active_jobs.get(job_id)
        if not job:
            self.logger.error(f"Job {job_id} not found")
            return False
        
        try:
            # Update job status
            await self._update_job_status(job_id, RetrainingStatus.DATA_COLLECTION)
            
            # Step 1: Data Collection
            training_data_path = await self._collect_training_data(job)
            if not training_data_path:
                raise Exception("Failed to collect training data")
            
            # Step 2: Data Preprocessing
            await self._update_job_status(job_id, RetrainingStatus.PREPROCESSING)
            X_train, X_val, y_train, y_val = await self._preprocess_data(job, training_data_path)
            
            # Step 3: Hyperparameter Optimization
            await self._update_job_status(job_id, RetrainingStatus.TRAINING)
            best_model, best_params, training_metrics = await self._optimize_hyperparameters(
                job, X_train, y_train, X_val, y_val
            )
            
            # Step 4: Model Validation
            await self._update_job_status(job_id, RetrainingStatus.VALIDATION)
            validation_results = await self._validate_model(job, best_model, X_val, y_val)
            
            # Step 5: Model Deployment
            if validation_results["passed"]:
                await self._update_job_status(job_id, RetrainingStatus.DEPLOYMENT)
                deployment_success = await self._deploy_model(job, best_model, best_params)
                
                if deployment_success:
                    # Update job with final results
                    job.metrics = {**training_metrics, **validation_results}
                    job.artifacts = {
                        "model_path": str(self.artifacts_path / f"{job_id}_model.pkl"),
                        "parameters_path": str(self.artifacts_path / f"{job_id}_params.json")
                    }
                    job.completed_at = datetime.now()
                    
                    await self._update_job_status(job_id, RetrainingStatus.COMPLETED)
                    self.logger.info(f"Retraining job {job_id} completed successfully")
                    return True
                else:
                    raise Exception("Model deployment failed")
            else:
                raise Exception(f"Model validation failed: {validation_results['reason']}")
        
        except Exception as e:
            job.error_message = str(e)
            await self._update_job_status(job_id, RetrainingStatus.FAILED)
            self.logger.error(f"Retraining job {job_id} failed: {e}")
            return False
        
        finally:
            # Clean up
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

    async def _collect_training_data(self, job: RetrainingJob) -> Optional[str]:
        """Collect and prepare training data"""
        
        # This is a placeholder - in real implementation, this would:
        # 1. Query the Echo learning database
        # 2. Collect recent decision outcomes
        # 3. Feature engineering
        # 4. Data quality checks
        
        data_path = self.data_path / f"{job.job_id}_training_data.csv"
        
        # Simulated data collection - replace with actual implementation
        # Get data from Echo learning system
        from src.core.echo.echo_learning_system import EchoLearningSystem
        learning_system = EchoLearningSystem()
        
        # Collect recent learning data
        recent_data = await learning_system.get_training_data(
            days=job.config.trigger_conditions.get("data_collection_days", 30),
            min_samples=job.config.trigger_conditions.get("min_data_samples", 1000)
        )
        
        if len(recent_data) < job.config.trigger_conditions.get("min_data_samples", 1000):
            self.logger.warning(f"Insufficient training data: {len(recent_data)} samples")
            return None
        
        # Save data
        df = pd.DataFrame(recent_data)
        df.to_csv(data_path, index=False)
        
        # Store data metadata
        data_id = self._generate_id("data")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO training_data (
                    data_id, job_id, data_type, file_path, sample_count, feature_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_id, job.job_id, "training", str(data_path),
                len(df), len(df.columns) - 1, datetime.now()
            ))
        
        self.logger.info(f"Collected {len(df)} training samples for job {job.job_id}")
        return str(data_path)

    async def _preprocess_data(
        self, 
        job: RetrainingJob, 
        data_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess training data"""
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Basic preprocessing - customize based on your data
        # Remove NaN values
        df = df.dropna()
        
        # Separate features and target
        target_column = job.config.training_config.get("target_column", "outcome")
        feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler for later use
        scaler_path = self.artifacts_path / f"{job.job_id}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        # Split data
        test_size = job.config.training_config.get("test_size", 0.2)
        val_size = job.config.training_config.get("validation_size", 0.2)
        random_state = job.config.training_config.get("random_state", 42)
        
        # First split: training + validation vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: training vs validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_train_val
        )
        
        self.logger.info(f"Data preprocessing completed for job {job.job_id}:")
        self.logger.info(f"  Training samples: {len(X_train)}")
        self.logger.info(f"  Validation samples: {len(X_val)}")
        self.logger.info(f"  Test samples: {len(X_test)}")
        
        return X_train, X_val, y_train, y_val

    async def _optimize_hyperparameters(
        self,
        job: RetrainingJob,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            # Select model type
            model_types = list(job.config.hyperparameter_space.keys())
            model_type = trial.suggest_categorical("model_type", model_types)
            
            # Get hyperparameters for selected model
            param_space = job.config.hyperparameter_space[model_type]
            params = {}
            
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values if v is not None):
                        # Numeric parameter
                        if all(isinstance(v, int) for v in param_values if v is not None):
                            params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                        else:
                            params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        # Categorical parameter
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Create and train model
            if model_type == "random_forest":
                model = RandomForestClassifier(**params, random_state=42)
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(**params, random_state=42)
            elif model_type == "logistic_regression":
                model = LogisticRegression(**params, random_state=42, max_iter=1000)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate primary metric (accuracy by default)
            primary_metric = job.config.validation_config.get("primary_metric", "accuracy")
            if primary_metric == "accuracy":
                score = accuracy_score(y_val, y_pred)
            elif primary_metric == "f1_score":
                score = f1_score(y_val, y_pred, average="weighted")
            else:
                score = accuracy_score(y_val, y_pred)
            
            # Store trial results
            trial_id = self._generate_id("trial")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO hyperparameter_trials (
                        trial_id, job_id, trial_number, parameters, metrics, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trial_id, job.job_id, trial.number, json.dumps({"model_type": model_type, **params}),
                    json.dumps({"score": score}), "completed", datetime.now()
                ))
            
            return score
        
        # Run optimization
        study = optuna.create_study(direction="maximize")
        n_trials = job.config.training_config.get("optimization_trials", 50)
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        model_type = best_params.pop("model_type")
        
        # Train final model with best parameters
        if model_type == "random_forest":
            best_model = RandomForestClassifier(**best_params, random_state=42)
        elif model_type == "gradient_boosting":
            best_model = GradientBoostingClassifier(**best_params, random_state=42)
        elif model_type == "logistic_regression":
            best_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
        
        best_model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        training_metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "train_f1": f1_score(y_train, y_train_pred, average="weighted"),
            "val_f1": f1_score(y_val, y_val_pred, average="weighted"),
            "best_score": study.best_value,
            "n_trials": len(study.trials)
        }
        
        self.logger.info(f"Hyperparameter optimization completed for job {job.job_id}:")
        self.logger.info(f"  Best score: {study.best_value:.4f}")
        self.logger.info(f"  Best params: {best_params}")
        
        return best_model, {"model_type": model_type, **best_params}, training_metrics

    async def _validate_model(
        self,
        job: RetrainingJob,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Validate trained model"""
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate validation metrics
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="weighted"),
            "recall": recall_score(y_val, y_pred, average="weighted"),
            "f1_score": f1_score(y_val, y_pred, average="weighted")
        }
        
        # Check if model meets minimum requirements
        required_metrics = job.config.validation_config.get("metrics", ["accuracy"])
        minimum_improvement = job.config.validation_config.get("minimum_improvement", 0.01)
        
        # Get current production model performance for comparison
        # This would integrate with your model registry
        current_performance = await self._get_current_model_performance(job.model_name)
        
        validation_passed = True
        failure_reasons = []
        
        for metric_name in required_metrics:
            current_value = metrics.get(metric_name, 0)
            baseline_value = current_performance.get(metric_name, 0)
            
            if current_value < baseline_value + minimum_improvement:
                validation_passed = False
                failure_reasons.append(
                    f"{metric_name}: {current_value:.4f} < {baseline_value + minimum_improvement:.4f} (required improvement)"
                )
        
        result = {
            "passed": validation_passed,
            "metrics": metrics,
            "baseline_metrics": current_performance,
            "improvement": {
                metric: metrics[metric] - current_performance.get(metric, 0)
                for metric in metrics.keys()
            }
        }
        
        if not validation_passed:
            result["reason"] = "; ".join(failure_reasons)
        
        self.logger.info(f"Model validation for job {job.job_id}: {'PASSED' if validation_passed else 'FAILED'}")
        return result

    async def _get_current_model_performance(self, model_name: str) -> Dict[str, float]:
        """Get current production model performance"""
        
        # This would integrate with your model registry and monitoring system
        # For now, return dummy baseline metrics
        return {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.7,
            "f1_score": 0.72
        }

    async def _deploy_model(
        self,
        job: RetrainingJob,
        model: Any,
        parameters: Dict[str, Any]
    ) -> bool:
        """Deploy validated model"""
        
        try:
            # Save model artifacts
            model_path = self.artifacts_path / f"{job.job_id}_model.pkl"
            params_path = self.artifacts_path / f"{job.job_id}_params.json"
            
            joblib.dump(model, model_path)
            with open(params_path, 'w') as f:
                json.dump(parameters, f, indent=2)
            
            # Register model in model registry
            from src.core.echo.echo_model_registry import ModelRegistryAPI, ModelType as RegistryModelType
            registry = ModelRegistryAPI()
            
            model_id = await registry.register_model(
                name=job.model_name,
                version=f"retrained_{job.job_id}",
                model_type=RegistryModelType.CLASSIFIER,  # Map appropriately
                model_artifact=model,
                description=f"Automatically retrained model - Job {job.job_id}",
                tags=["automated", "retrained"],
                hyperparameters=parameters,
                created_by="retraining_pipeline"
            )
            
            # Promote to staging first
            await registry.promote_model(model_id, RegistryModelType.STAGING)
            
            # If AB testing is enabled, start A/B test
            if job.config.deployment_config.get("ab_test_percentage", 0) > 0:
                await self._start_ab_test(job, model_id)
            else:
                # Direct promotion to production
                await registry.promote_model(model_id, RegistryModelType.PRODUCTION)
            
            self.logger.info(f"Model deployed successfully for job {job.job_id}: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model deployment failed for job {job.job_id}: {e}")
            return False

    async def _start_ab_test(self, job: RetrainingJob, new_model_id: str) -> bool:
        """Start A/B test for new model"""
        
        try:
            from src.core.echo.echo_ab_testing import ABTestingFramework, VariantConfig, MetricType
            
            ab_framework = ABTestingFramework()
            
            # Get current production model
            from src.core.echo.echo_model_registry import ModelRegistryAPI, ModelType as RegistryModelType
            registry = ModelRegistryAPI()
            current_model_id = await registry.get_production_model(RegistryModelType.CLASSIFIER)
            
            if not current_model_id:
                # No current model, promote directly
                await registry.promote_model(new_model_id, RegistryModelType.PRODUCTION)
                return True
            
            # Create A/B test variants
            variants = [
                VariantConfig(
                    variant_id="current",
                    name="Current Production Model",
                    description="Currently deployed model",
                    model_id=current_model_id,
                    allocation_weight=0.9  # 90% traffic
                ),
                VariantConfig(
                    variant_id="new",
                    name="Retrained Model",
                    description=f"Newly retrained model - Job {job.job_id}",
                    model_id=new_model_id,
                    allocation_weight=0.1  # 10% traffic
                )
            ]
            
            # Create experiment
            experiment_id = await ab_framework.create_experiment(
                name=f"Retraining A/B Test - {job.model_name}",
                description=f"Testing retrained model from job {job.job_id}",
                variants=variants,
                primary_metric=MetricType.ACCURACY,
                secondary_metrics=[MetricType.PRECISION, MetricType.RECALL],
                target_sample_size=1000,
                max_duration_days=7
            )
            
            # Start experiment
            await ab_framework.start_experiment(experiment_id)
            
            self.logger.info(f"A/B test started for job {job.job_id}: {experiment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start A/B test for job {job.job_id}: {e}")
            return False

    async def _update_job_status(self, job_id: str, status: RetrainingStatus) -> bool:
        """Update job status in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            update_fields = ["status = ?"]
            params = [status.value]
            
            if status == RetrainingStatus.TRAINING and job_id in self.active_jobs:
                update_fields.append("started_at = ?")
                params.append(datetime.now())
                self.active_jobs[job_id].started_at = datetime.now()
            
            update_fields.append("job_id = ?")
            params.append(job_id)
            
            conn.execute(f'''
                UPDATE retraining_jobs SET {', '.join(update_fields[:-1])} WHERE {update_fields[-1]}
            ''', params)
        
        if job_id in self.active_jobs:
            self.active_jobs[job_id].status = status
        
        self.logger.info(f"Job {job_id} status updated to: {status.value}")
        return True

    async def monitor_performance(self, model_name: str, metrics: Dict[str, float]) -> bool:
        """Monitor model performance and trigger retraining if needed"""
        
        if model_name not in self.model_configs:
            return False
        
        config = self.model_configs[model_name]
        performance_threshold = config.trigger_conditions.get("performance_threshold", 0.05)
        
        # Store performance metrics
        for metric_name, metric_value in metrics.items():
            monitor_id = self._generate_id("monitor")
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO performance_monitoring (
                        monitor_id, model_name, timestamp, metric_name, metric_value, threshold_value
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    monitor_id, model_name, datetime.now(), metric_name, metric_value, performance_threshold
                ))
        
        # Check if retraining should be triggered
        primary_metric = metrics.get("accuracy", 0)  # Default to accuracy
        baseline_performance = await self._get_current_model_performance(model_name)
        baseline_metric = baseline_performance.get("accuracy", 0)
        
        if baseline_metric - primary_metric > performance_threshold:
            # Performance degradation detected
            await self.trigger_retraining(
                model_name=model_name,
                trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                trigger_reason=f"Performance dropped from {baseline_metric:.4f} to {primary_metric:.4f}"
            )
            
            self.logger.warning(f"Triggered retraining for {model_name} due to performance degradation")
            return True
        
        return False

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get retraining job status"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM retraining_jobs WHERE job_id = ?
            ''', (job_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "job_id": row[0],
                "model_name": row[1],
                "trigger_type": row[2],
                "trigger_reason": row[3],
                "status": row[4],
                "created_at": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "metrics": json.loads(row[9]) if row[9] else None,
                "error_message": row[11]
            }

# Example usage
async def example_retraining_setup():
    """Example retraining setup"""
    
    pipeline = AutomatedRetrainingPipeline()
    
    # Configure retraining for Echo decision engine
    await pipeline.configure_model_retraining(
        model_name="echo_decision_engine",
        model_type=ModelType.CLASSIFIER,
        trigger_conditions={
            "performance_threshold": 0.05,
            "drift_threshold": 0.1,
            "schedule": "weekly",
            "min_data_samples": 1000
        },
        training_config={
            "test_size": 0.2,
            "validation_size": 0.2,
            "optimization_trials": 100
        }
    )
    
    # Trigger manual retraining
    job_id = await pipeline.trigger_retraining(
        model_name="echo_decision_engine",
        trigger=RetrainingTrigger.MANUAL,
        trigger_reason="Initial model training"
    )
    
    print(f"Triggered retraining job: {job_id}")
    
    # Monitor job status
    import time
    while True:
        status = await pipeline.get_job_status(job_id)
        print(f"Job status: {status['status']}")
        
        if status["status"] in ["completed", "failed"]:
            break
        
        time.sleep(10)
    
    print(f"Final job status: {status}")

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, HTTPException
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # FastAPI app for retraining pipeline API
    app = FastAPI(title="Echo Automated Retraining Pipeline", version="1.0.0")
    pipeline = AutomatedRetrainingPipeline()
    
    @app.post("/retraining/configure")
    async def configure_retraining_endpoint(
        model_name: str,
        model_type: str,
        trigger_conditions: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None
    ):
        try:
            model_type_enum = ModelType(model_type)
            success = await pipeline.configure_model_retraining(
                model_name, model_type_enum, trigger_conditions, training_config
            )
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/retraining/trigger")
    async def trigger_retraining_endpoint(
        model_name: str,
        trigger_type: str,
        trigger_reason: str = ""
    ):
        try:
            trigger_enum = RetrainingTrigger(trigger_type)
            job_id = await pipeline.trigger_retraining(model_name, trigger_enum, trigger_reason)
            return {"job_id": job_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/retraining/jobs/{job_id}")
    async def get_job_status_endpoint(job_id: str):
        try:
            status = await pipeline.get_job_status(job_id)
            if not status:
                raise HTTPException(status_code=404, detail="Job not found")
            return status
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/retraining/monitor")
    async def monitor_performance_endpoint(model_name: str, metrics: Dict[str, float]):
        try:
            triggered = await pipeline.monitor_performance(model_name, metrics)
            return {"retraining_triggered": triggered}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8343)
