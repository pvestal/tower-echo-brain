#!/usr/bin/env python3
"""
Echo MLOps Integration Layer
===========================

Central integration layer that connects all MLOps components with the existing Echo system:
- Model Registry integration
- A/B Testing coordination
- Drift Detection monitoring
- Automated Retraining triggers
- Feature Store management
- Unified MLOps orchestration

This serves as the main coordinator for all MLOps operations within Echo Brain.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Import our MLOps components
from echo_model_registry import ModelRegistryAPI, ModelType, ModelStage
from echo_ab_testing import ABTestingFramework, VariantConfig, MetricType, AllocationStrategy
from echo_drift_detector import DriftDetector, DriftMethod, DriftType, DriftSeverity
from echo_retraining_pipeline import AutomatedRetrainingPipeline, RetrainingTrigger, ModelType as RetrainingModelType
from echo_feature_store import FeatureStore, FeatureType, FeatureValue

class MLOpsEvent(Enum):
    MODEL_REGISTERED = "model_registered"
    MODEL_PROMOTED = "model_promoted"
    DRIFT_DETECTED = "drift_detected"
    EXPERIMENT_COMPLETED = "experiment_completed"
    RETRAINING_TRIGGERED = "retraining_triggered"
    RETRAINING_COMPLETED = "retraining_completed"
    FEATURE_QUALITY_DEGRADED = "feature_quality_degraded"

@dataclass
class MLOpsConfig:
    drift_monitoring_enabled: bool = True
    auto_retraining_enabled: bool = True
    ab_testing_enabled: bool = True
    feature_monitoring_enabled: bool = True
    drift_threshold: float = 0.1
    performance_threshold: float = 0.05
    quality_threshold: float = 0.8
    retraining_schedule: str = "weekly"

class EchoMLOpsOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all MLOps components
        self.model_registry = ModelRegistryAPI()
        self.ab_testing = ABTestingFramework()
        self.drift_detector = DriftDetector()
        self.retraining_pipeline = AutomatedRetrainingPipeline()
        self.feature_store = FeatureStore()
        
        # Configuration
        self.config = MLOpsConfig()
        
        # Event handlers
        self.event_handlers: Dict[MLOpsEvent, List[callable]] = {
            event: [] for event in MLOpsEvent
        }
        
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Setup default event handlers"""
        
        # Drift detection triggers retraining
        self.register_event_handler(
            MLOpsEvent.DRIFT_DETECTED,
            self._handle_drift_detected
        )
        
        # Model promotion triggers A/B testing
        self.register_event_handler(
            MLOpsEvent.MODEL_REGISTERED,
            self._handle_model_registered
        )
        
        # Experiment completion triggers model promotion
        self.register_event_handler(
            MLOpsEvent.EXPERIMENT_COMPLETED,
            self._handle_experiment_completed
        )
        
        # Feature quality degradation triggers monitoring
        self.register_event_handler(
            MLOpsEvent.FEATURE_QUALITY_DEGRADED,
            self._handle_feature_quality_degraded
        )

    def register_event_handler(self, event: MLOpsEvent, handler: callable):
        """Register an event handler"""
        self.event_handlers[event].append(handler)

    async def emit_event(self, event: MLOpsEvent, data: Dict[str, Any]):
        """Emit an MLOps event"""
        self.logger.info(f"MLOps event emitted: {event.value} - {data}")
        
        # Call all registered handlers
        for handler in self.event_handlers[event]:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event.value}: {e}")

    async def initialize_echo_mlops(self) -> bool:
        """Initialize MLOps for Echo Brain"""
        
        try:
            # Configure drift monitoring for key Echo features
            await self._setup_drift_monitoring()
            
            # Configure retraining for Echo models
            await self._setup_retraining()
            
            # Register Echo features in feature store
            await self._setup_feature_store()
            
            # Start monitoring loops
            asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("Echo MLOps initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"MLOps initialization failed: {e}")
            return False

    async def _setup_drift_monitoring(self):
        """Setup drift monitoring for Echo features"""
        
        echo_features = [
            {
                "name": "echo_decision_confidence",
                "methods": [DriftMethod.KOLMOGOROV_SMIRNOV, DriftMethod.POPULATION_STABILITY_INDEX]
            },
            {
                "name": "echo_response_time",
                "methods": [DriftMethod.JENSEN_SHANNON_DIVERGENCE]
            },
            {
                "name": "echo_decision_accuracy",
                "methods": [DriftMethod.KOLMOGOROV_SMIRNOV, DriftMethod.CHI_SQUARE]
            }
        ]
        
        for feature in echo_features:
            await self.drift_detector.configure_feature_monitoring(
                feature_name=feature["name"],
                drift_methods=feature["methods"],
                alert_threshold=self.config.drift_threshold,
                warning_threshold=self.config.drift_threshold * 2
            )
        
        self.logger.info("Drift monitoring configured for Echo features")

    async def _setup_retraining(self):
        """Setup automated retraining for Echo models"""
        
        echo_models = [
            {
                "name": "echo_decision_engine",
                "type": RetrainingModelType.CLASSIFIER,
                "trigger_conditions": {
                    "performance_threshold": self.config.performance_threshold,
                    "drift_threshold": self.config.drift_threshold,
                    "schedule": self.config.retraining_schedule,
                    "min_data_samples": 1000
                }
            },
            {
                "name": "echo_outcome_predictor",
                "type": RetrainingModelType.CLASSIFIER,
                "trigger_conditions": {
                    "performance_threshold": self.config.performance_threshold,
                    "drift_threshold": self.config.drift_threshold,
                    "schedule": "bi-weekly",
                    "min_data_samples": 500
                }
            }
        ]
        
        for model in echo_models:
            await self.retraining_pipeline.configure_model_retraining(
                model_name=model["name"],
                model_type=model["type"],
                trigger_conditions=model["trigger_conditions"]
            )
        
        self.logger.info("Automated retraining configured for Echo models")

    async def _setup_feature_store(self):
        """Setup feature store for Echo features"""
        
        echo_features = [
            {
                "name": "echo_decision_confidence",
                "description": "Confidence score of Echo's decisions",
                "type": FeatureType.NUMERICAL,
                "entity_type": "decision",
                "validation_rules": {"min_value": 0.0, "max_value": 1.0}
            },
            {
                "name": "echo_response_time",
                "description": "Time taken for Echo to respond",
                "type": FeatureType.NUMERICAL,
                "entity_type": "decision",
                "validation_rules": {"min_value": 0.0}
            },
            {
                "name": "echo_decision_accuracy",
                "description": "Accuracy of Echo's decisions",
                "type": FeatureType.NUMERICAL,
                "entity_type": "decision",
                "validation_rules": {"min_value": 0.0, "max_value": 1.0}
            },
            {
                "name": "user_interaction_frequency",
                "description": "Frequency of user interactions",
                "type": FeatureType.NUMERICAL,
                "entity_type": "user"
            },
            {
                "name": "echo_learning_rate",
                "description": "Rate of Echo's learning from feedback",
                "type": FeatureType.NUMERICAL,
                "entity_type": "session"
            }
        ]
        
        for feature in echo_features:
            await self.feature_store.register_feature(
                name=feature["name"],
                description=feature["description"],
                feature_type=feature["type"],
                entity_type=feature["entity_type"],
                validation_rules=feature.get("validation_rules", {})
            )
        
        # Create feature set for Echo decision making
        decision_features = [f["name"] for f in echo_features if f["entity_type"] == "decision"]
        await self.feature_store.create_feature_set(
            name="echo_decision_features",
            description="Features used for Echo decision making",
            features=decision_features,
            entity_type="decision"
        )
        
        self.logger.info("Feature store configured for Echo features")

    async def _monitoring_loop(self):
        """Main monitoring loop for MLOps"""
        
        while True:
            try:
                # Check for drift
                if self.config.drift_monitoring_enabled:
                    await self._check_drift()
                
                # Monitor feature quality
                if self.config.feature_monitoring_enabled:
                    await self._monitor_feature_quality()
                
                # Check experiment status
                if self.config.ab_testing_enabled:
                    await self._check_experiments()
                
                # Sleep for 5 minutes before next check
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error

    async def _check_drift(self):
        """Check for drift in Echo features"""
        
        features_to_check = [
            "echo_decision_confidence",
            "echo_response_time",
            "echo_decision_accuracy"
        ]
        
        for feature_name in features_to_check:
            try:
                alerts = await self.drift_detector.detect_drift(feature_name)
                
                for alert in alerts:
                    if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                        await self.emit_event(MLOpsEvent.DRIFT_DETECTED, {
                            "feature_name": feature_name,
                            "alert": asdict(alert)
                        })
                        
            except Exception as e:
                self.logger.warning(f"Drift check failed for {feature_name}: {e}")

    async def _monitor_feature_quality(self):
        """Monitor feature quality metrics"""
        
        features_to_monitor = [
            "echo_decision_confidence",
            "echo_response_time",
            "echo_decision_accuracy",
            "user_interaction_frequency"
        ]
        
        for feature_name in features_to_monitor:
            try:
                quality_metrics = await self.feature_store.monitor_feature_quality(feature_name)
                
                if quality_metrics.get("quality_score", 1.0) < self.config.quality_threshold:
                    await self.emit_event(MLOpsEvent.FEATURE_QUALITY_DEGRADED, {
                        "feature_name": feature_name,
                        "quality_metrics": quality_metrics
                    })
                    
            except Exception as e:
                self.logger.warning(f"Quality monitoring failed for {feature_name}: {e}")

    async def _check_experiments(self):
        """Check status of running A/B experiments"""
        
        # This would integrate with your A/B testing framework
        # to check for completed experiments
        pass

    # Event Handlers
    async def _handle_drift_detected(self, data: Dict[str, Any]):
        """Handle drift detection event"""
        
        feature_name = data["feature_name"]
        alert = data["alert"]
        
        self.logger.warning(f"Drift detected in {feature_name}: {alert['description']}")
        
        # Trigger retraining if enabled
        if self.config.auto_retraining_enabled:
            # Map feature to model
            feature_to_model = {
                "echo_decision_confidence": "echo_decision_engine",
                "echo_decision_accuracy": "echo_decision_engine",
                "echo_response_time": "echo_outcome_predictor"
            }
            
            model_name = feature_to_model.get(feature_name)
            if model_name:
                await self.retraining_pipeline.trigger_retraining(
                    model_name=model_name,
                    trigger=RetrainingTrigger.DRIFT_DETECTION,
                    trigger_reason=f"Drift detected in {feature_name}: {alert['drift_score']:.4f}"
                )
                
                self.logger.info(f"Triggered retraining for {model_name} due to drift in {feature_name}")

    async def _handle_model_registered(self, data: Dict[str, Any]):
        """Handle new model registration"""
        
        model_id = data["model_id"]
        model_name = data.get("model_name", "unknown")
        
        self.logger.info(f"New model registered: {model_id} - {model_name}")
        
        # If this is a retrained model, consider A/B testing
        if "retrained" in model_name and self.config.ab_testing_enabled:
            await self._setup_ab_test_for_model(model_id, model_name)

    async def _handle_experiment_completed(self, data: Dict[str, Any]):
        """Handle A/B experiment completion"""
        
        experiment_id = data["experiment_id"]
        winner_variant = data.get("winner_variant")
        
        self.logger.info(f"A/B experiment completed: {experiment_id}, winner: {winner_variant}")
        
        # Promote winning model to production
        if winner_variant and winner_variant != "control":
            model_id = data.get("winner_model_id")
            if model_id:
                await self.model_registry.promote_model(model_id, ModelStage.PRODUCTION)
                self.logger.info(f"Promoted winning model {model_id} to production")

    async def _handle_feature_quality_degraded(self, data: Dict[str, Any]):
        """Handle feature quality degradation"""
        
        feature_name = data["feature_name"]
        quality_metrics = data["quality_metrics"]
        
        self.logger.warning(f"Feature quality degraded for {feature_name}: {quality_metrics}")
        
        # Increase monitoring frequency or trigger investigation
        # This could integrate with alerting systems

    async def _setup_ab_test_for_model(self, model_id: str, model_name: str):
        """Setup A/B test for a new model"""
        
        try:
            # Get current production model
            current_model_id = await self.model_registry.get_production_model(ModelType.CLASSIFIER)
            
            if not current_model_id:
                # No current model, promote directly
                await self.model_registry.promote_model(model_id, ModelStage.PRODUCTION)
                return
            
            # Create A/B test variants
            variants = [
                VariantConfig(
                    variant_id="control",
                    name="Current Production Model",
                    description="Currently deployed model",
                    model_id=current_model_id,
                    allocation_weight=0.8  # 80% traffic
                ),
                VariantConfig(
                    variant_id="treatment",
                    name="New Model",
                    description=f"New model: {model_name}",
                    model_id=model_id,
                    allocation_weight=0.2  # 20% traffic
                )
            ]
            
            # Create experiment
            experiment_id = await self.ab_testing.create_experiment(
                name=f"A/B Test - {model_name}",
                description=f"Testing new model {model_id} vs current production",
                variants=variants,
                primary_metric=MetricType.ACCURACY,
                secondary_metrics=[MetricType.PRECISION, MetricType.RECALL],
                allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
                target_sample_size=1000,
                max_duration_days=7
            )
            
            # Start experiment
            await self.ab_testing.start_experiment(experiment_id)
            
            self.logger.info(f"A/B test started for model {model_id}: {experiment_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup A/B test for model {model_id}: {e}")

    # Integration methods for Echo Brain
    async def record_echo_decision(
        self,
        decision_id: str,
        confidence: float,
        response_time: float,
        accuracy: Optional[float] = None,
        user_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> bool:
        """Record an Echo decision for MLOps monitoring"""
        
        try:
            timestamp = datetime.now()
            
            # Record in feature store
            features_to_record = [
                FeatureValue(
                    feature_id="echo_decision_confidence",
                    entity_id=decision_id,
                    value=confidence,
                    timestamp=timestamp,
                    version="1.0.0"
                ),
                FeatureValue(
                    feature_id="echo_response_time",
                    entity_id=decision_id,
                    value=response_time,
                    timestamp=timestamp,
                    version="1.0.0"
                )
            ]
            
            if accuracy is not None:
                features_to_record.append(
                    FeatureValue(
                        feature_id="echo_decision_accuracy",
                        entity_id=decision_id,
                        value=accuracy,
                        timestamp=timestamp,
                        version="1.0.0"
                    )
                )
            
            # Store all features
            for feature_value in features_to_record:
                await self.feature_store.store_feature_values(
                    feature_value.feature_id, [feature_value]
                )
            
            # Record for drift monitoring
            await self.drift_detector.record_observation(
                feature_name="echo_decision_confidence",
                feature_value=confidence,
                context=context
            )
            
            await self.drift_detector.record_observation(
                feature_name="echo_response_time",
                feature_value=response_time,
                context=context
            )
            
            if accuracy is not None:
                await self.drift_detector.record_observation(
                    feature_name="echo_decision_accuracy",
                    feature_value=accuracy,
                    context=context
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record Echo decision: {e}")
            return False

    async def get_mlops_status(self) -> Dict[str, Any]:
        """Get overall MLOps status"""
        
        try:
            # Get drift summary
            drift_summary = await self.drift_detector.get_drift_summary(days=7)
            
            # Get feature quality summary
            feature_quality = {}
            features = ["echo_decision_confidence", "echo_response_time", "echo_decision_accuracy"]
            
            for feature in features:
                try:
                    quality = await self.feature_store.monitor_feature_quality(feature)
                    feature_quality[feature] = quality
                except:
                    feature_quality[feature] = {}
            
            # Get model registry summary
            models = await self.model_registry.list_models(limit=10)
            
            return {
                "drift_monitoring": {
                    "enabled": self.config.drift_monitoring_enabled,
                    "summary": drift_summary
                },
                "feature_quality": feature_quality,
                "models": {
                    "total_registered": len(models),
                    "recent_models": [
                        {
                            "model_id": m.model_id,
                            "name": m.name,
                            "version": m.version,
                            "stage": m.stage.value,
                            "created_at": m.created_at.isoformat()
                        }
                        for m in models[:5]
                    ]
                },
                "retraining": {
                    "enabled": self.config.auto_retraining_enabled,
                    "schedule": self.config.retraining_schedule
                },
                "ab_testing": {
                    "enabled": self.config.ab_testing_enabled
                },
                "configuration": asdict(self.config)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get MLOps status: {e}")
            return {"error": str(e)}

# Global orchestrator instance
mlops_orchestrator = EchoMLOpsOrchestrator()

# Integration function for Echo Brain
async def initialize_echo_mlops() -> bool:
    """Initialize MLOps for Echo Brain - called from main Echo system"""
    return await mlops_orchestrator.initialize_echo_mlops()

async def record_echo_decision_mlops(
    decision_id: str,
    confidence: float,
    response_time: float,
    accuracy: Optional[float] = None,
    user_id: Optional[str] = None,
    context: Dict[str, Any] = None
) -> bool:
    """Record Echo decision for MLOps monitoring - called from Echo decision engine"""
    return await mlops_orchestrator.record_echo_decision(
        decision_id, confidence, response_time, accuracy, user_id, context
    )

async def get_echo_mlops_status() -> Dict[str, Any]:
    """Get MLOps status - called from Echo API"""
    return await mlops_orchestrator.get_mlops_status()

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, HTTPException
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # FastAPI app for MLOps integration API
    app = FastAPI(title="Echo MLOps Integration", version="1.0.0")
    
    @app.on_event("startup")
    async def startup_event():
        await initialize_echo_mlops()
    
    @app.post("/mlops/record-decision")
    async def record_decision_endpoint(
        decision_id: str,
        confidence: float,
        response_time: float,
        accuracy: Optional[float] = None,
        user_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ):
        try:
            success = await record_echo_decision_mlops(
                decision_id, confidence, response_time, accuracy, user_id, context
            )
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/mlops/status")
    async def get_status_endpoint():
        try:
            status = await get_echo_mlops_status()
            return status
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/mlops/trigger-retraining")
    async def trigger_retraining_endpoint(
        model_name: str,
        reason: str = "Manual trigger"
    ):
        try:
            job_id = await mlops_orchestrator.retraining_pipeline.trigger_retraining(
                model_name=model_name,
                trigger=RetrainingTrigger.MANUAL,
                trigger_reason=reason
            )
            return {"job_id": job_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/mlops/drift-summary")
    async def get_drift_summary_endpoint(days: int = 7):
        try:
            summary = await mlops_orchestrator.drift_detector.get_drift_summary(days=days)
            return summary
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8345)
