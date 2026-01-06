#!/usr/bin/env python3
"""
Echo Drift Detection & Monitoring System
=======================================

Advanced drift detection system for Echo Brain that monitors:
- Data drift: Changes in input data distribution
- Concept drift: Changes in data-target relationships
- Performance drift: Degradation in model performance
- Contextual drift: Changes in environmental conditions

Features:
- Statistical drift detection (KS test, PSI, JS divergence)
- Real-time monitoring with configurable thresholds
- Automatic alert generation and escalation
- Historical trend analysis
- Integration with retraining pipeline
- Multi-dimensional drift analysis
"""

import asyncio
import logging
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Configuration
DRIFT_MONITORING_DB = "/opt/tower-echo-brain/data/drift_monitoring.db"
DRIFT_DATA_PATH = "/opt/tower-echo-brain/drift_analysis/"
REFERENCE_WINDOW_DAYS = 30
DETECTION_WINDOW_DAYS = 7
ALERT_THRESHOLD = 0.05
WARNING_THRESHOLD = 0.1

class DriftType(Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    CONTEXTUAL_DRIFT = "contextual_drift"

class DriftSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DriftMethod(Enum):
    KOLMOGOROV_SMIRNOV = "ks_test"
    POPULATION_STABILITY_INDEX = "psi"
    JENSEN_SHANNON_DIVERGENCE = "js_divergence"
    WASSERSTEIN_DISTANCE = "wasserstein"
    CHI_SQUARE = "chi_square"
    STATISTICAL_DISTANCE = "statistical_distance"

@dataclass
class DriftAlert:
    alert_id: str
    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    feature_name: str
    drift_score: float
    threshold: float
    method: DriftMethod
    description: str
    recommendations: List[str]
    metadata: Dict[str, Any] = None

@dataclass
class DriftConfig:
    feature_name: str
    drift_methods: List[DriftMethod]
    alert_threshold: float
    warning_threshold: float
    reference_window_days: int
    detection_window_days: int
    enabled: bool = True

class DriftDetector:
    def __init__(self):
        self.db_path = DRIFT_MONITORING_DB
        self.data_path = Path(DRIFT_DATA_PATH)
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
        self._init_database()
        self.feature_configs: Dict[str, DriftConfig] = {}
        self.reference_distributions: Dict[str, Dict[str, Any]] = {}

    def _ensure_directories(self):
        """Create necessary directories"""
        self.data_path.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize drift monitoring database"""
        with sqlite3.connect(self.db_path) as conn:
            # Feature monitoring table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_monitoring (
                    record_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    feature_vector TEXT,
                    prediction REAL,
                    actual_value REAL,
                    context TEXT
                )
            ''')
            
            # Drift alerts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    drift_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    drift_score REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    method TEXT NOT NULL,
                    description TEXT,
                    recommendations TEXT,
                    metadata TEXT,
                    acknowledged BOOLEAN DEFAULT 0,
                    resolved BOOLEAN DEFAULT 0
                )
            ''')
            
            # Drift statistics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS drift_statistics (
                    stat_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    feature_name TEXT NOT NULL,
                    method TEXT NOT NULL,
                    drift_score REAL NOT NULL,
                    p_value REAL,
                    reference_period_start TIMESTAMP,
                    reference_period_end TIMESTAMP,
                    detection_period_start TIMESTAMP,
                    detection_period_end TIMESTAMP,
                    sample_size_reference INTEGER,
                    sample_size_detection INTEGER
                )
            ''')
            
            # Feature configurations table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_configs (
                    config_id TEXT PRIMARY KEY,
                    feature_name TEXT UNIQUE NOT NULL,
                    config TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            ''')

            # Indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_monitoring_timestamp ON feature_monitoring(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_monitoring_feature ON feature_monitoring(feature_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON drift_alerts(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_statistics_feature ON drift_statistics(feature_name)')

    def _generate_id(self, prefix: str = "drift") -> str:
        """Generate unique ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        import hashlib
        import random
        random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"

    async def configure_feature_monitoring(
        self,
        feature_name: str,
        drift_methods: List[DriftMethod] = None,
        alert_threshold: float = ALERT_THRESHOLD,
        warning_threshold: float = WARNING_THRESHOLD,
        reference_window_days: int = REFERENCE_WINDOW_DAYS,
        detection_window_days: int = DETECTION_WINDOW_DAYS
    ) -> bool:
        """Configure drift monitoring for a feature"""
        
        drift_methods = drift_methods or [
            DriftMethod.KOLMOGOROV_SMIRNOV,
            DriftMethod.POPULATION_STABILITY_INDEX,
            DriftMethod.JENSEN_SHANNON_DIVERGENCE
        ]
        
        config = DriftConfig(
            feature_name=feature_name,
            drift_methods=drift_methods,
            alert_threshold=alert_threshold,
            warning_threshold=warning_threshold,
            reference_window_days=reference_window_days,
            detection_window_days=detection_window_days
        )
        
        # Store configuration
        config_id = self._generate_id("config")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO feature_configs (
                    config_id, feature_name, config, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                config_id, feature_name, json.dumps(asdict(config)),
                datetime.now(), datetime.now()
            ))
        
        self.feature_configs[feature_name] = config
        self.logger.info(f"Configured drift monitoring for feature: {feature_name}")
        return True

    async def record_observation(
        self,
        feature_name: str,
        feature_value: Optional[float] = None,
        feature_vector: Optional[List[float]] = None,
        prediction: Optional[float] = None,
        actual_value: Optional[float] = None,
        context: Dict[str, Any] = None
    ) -> bool:
        """Record a new observation for drift monitoring"""
        
        record_id = self._generate_id("record")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feature_monitoring (
                    record_id, timestamp, feature_name, feature_value,
                    feature_vector, prediction, actual_value, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_id, datetime.now(), feature_name, feature_value,
                json.dumps(feature_vector) if feature_vector else None,
                prediction, actual_value, json.dumps(context) if context else None
            ))
        
        # Check if we should run drift detection
        if feature_name in self.feature_configs:
            await self._maybe_detect_drift(feature_name)
        
        return True

    async def _maybe_detect_drift(self, feature_name: str) -> bool:
        """Check if it's time to run drift detection"""
        
        config = self.feature_configs[feature_name]
        
        # Check if we have enough recent data
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT COUNT(*) FROM feature_monitoring 
                WHERE feature_name = ? AND timestamp > ?
            ''', (
                feature_name,
                datetime.now() - timedelta(days=config.detection_window_days)
            ))
            
            recent_count = cursor.fetchone()[0]
            
            # Check if we have enough reference data
            cursor = conn.execute('''
                SELECT COUNT(*) FROM feature_monitoring 
                WHERE feature_name = ? AND timestamp BETWEEN ? AND ?
            ''', (
                feature_name,
                datetime.now() - timedelta(days=config.reference_window_days + config.detection_window_days),
                datetime.now() - timedelta(days=config.detection_window_days)
            ))
            
            reference_count = cursor.fetchone()[0]
        
        # Run drift detection if we have sufficient data
        if recent_count >= 50 and reference_count >= 100:  # Minimum sample sizes
            await self.detect_drift(feature_name)
            return True
        
        return False

    async def detect_drift(self, feature_name: str) -> List[DriftAlert]:
        """Run drift detection for a specific feature"""
        
        if feature_name not in self.feature_configs:
            raise ValueError(f"No configuration found for feature: {feature_name}")
        
        config = self.feature_configs[feature_name]
        alerts = []
        
        # Get reference and detection period data
        reference_data, detection_data = await self._get_drift_data(feature_name, config)
        
        if not reference_data or not detection_data:
            self.logger.warning(f"Insufficient data for drift detection: {feature_name}")
            return alerts
        
        # Run each configured drift method
        for method in config.drift_methods:
            try:
                drift_score, p_value = await self._calculate_drift_score(
                    reference_data, detection_data, method
                )
                
                # Store statistics
                await self._store_drift_statistics(
                    feature_name, method, drift_score, p_value, config
                )
                
                # Check for alerts
                severity = self._assess_drift_severity(drift_score, config)
                
                if severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    alert = await self._create_drift_alert(
                        feature_name, method, drift_score, severity, config
                    )
                    alerts.append(alert)
                
            except Exception as e:
                self.logger.error(f"Error in drift detection for {feature_name} using {method}: {e}")
        
        return alerts

    async def _get_drift_data(
        self, 
        feature_name: str, 
        config: DriftConfig
    ) -> Tuple[List[float], List[float]]:
        """Get reference and detection period data"""
        
        now = datetime.now()
        detection_start = now - timedelta(days=config.detection_window_days)
        reference_end = detection_start
        reference_start = reference_end - timedelta(days=config.reference_window_days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Reference data
            cursor = conn.execute('''
                SELECT feature_value FROM feature_monitoring 
                WHERE feature_name = ? AND timestamp BETWEEN ? AND ?
                AND feature_value IS NOT NULL
                ORDER BY timestamp
            ''', (feature_name, reference_start, reference_end))
            
            reference_data = [row[0] for row in cursor.fetchall()]
            
            # Detection data
            cursor = conn.execute('''
                SELECT feature_value FROM feature_monitoring 
                WHERE feature_name = ? AND timestamp >= ?
                AND feature_value IS NOT NULL
                ORDER BY timestamp
            ''', (feature_name, detection_start))
            
            detection_data = [row[0] for row in cursor.fetchall()]
        
        return reference_data, detection_data

    async def _calculate_drift_score(
        self,
        reference_data: List[float],
        detection_data: List[float],
        method: DriftMethod
    ) -> Tuple[float, Optional[float]]:
        """Calculate drift score using specified method"""
        
        ref_array = np.array(reference_data)
        det_array = np.array(detection_data)
        
        if method == DriftMethod.KOLMOGOROV_SMIRNOV:
            statistic, p_value = stats.ks_2samp(ref_array, det_array)
            return float(statistic), float(p_value)
        
        elif method == DriftMethod.POPULATION_STABILITY_INDEX:
            psi_score = self._calculate_psi(ref_array, det_array)
            return float(psi_score), None
        
        elif method == DriftMethod.JENSEN_SHANNON_DIVERGENCE:
            js_div = self._calculate_js_divergence(ref_array, det_array)
            return float(js_div), None
        
        elif method == DriftMethod.WASSERSTEIN_DISTANCE:
            wasserstein_dist = stats.wasserstein_distance(ref_array, det_array)
            return float(wasserstein_dist), None
        
        elif method == DriftMethod.CHI_SQUARE:
            chi2_stat, p_value = self._calculate_chi_square(ref_array, det_array)
            return float(chi2_stat), float(p_value)
        
        else:
            raise ValueError(f"Unknown drift method: {method}")

    def _calculate_psi(self, reference: np.ndarray, detection: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Calculate frequencies for both datasets
        ref_freq, _ = np.histogram(reference, bins=bin_edges)
        det_freq, _ = np.histogram(detection, bins=bin_edges)
        
        # Convert to proportions
        ref_prop = ref_freq / len(reference)
        det_prop = det_freq / len(detection)
        
        # Avoid division by zero
        ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
        det_prop = np.where(det_prop == 0, 0.0001, det_prop)
        
        # Calculate PSI
        psi = np.sum((det_prop - ref_prop) * np.log(det_prop / ref_prop))
        
        return psi

    def _calculate_js_divergence(self, reference: np.ndarray, detection: np.ndarray, bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence"""
        
        # Create histograms
        min_val = min(reference.min(), detection.min())
        max_val = max(reference.max(), detection.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        det_hist, _ = np.histogram(detection, bins=bin_edges, density=True)
        
        # Normalize to create probability distributions
        ref_hist = ref_hist / ref_hist.sum()
        det_hist = det_hist / det_hist.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        det_hist = det_hist + epsilon
        
        # Calculate JS divergence
        js_div = jensenshannon(ref_hist, det_hist)
        
        return js_div

    def _calculate_chi_square(self, reference: np.ndarray, detection: np.ndarray, bins: int = 10) -> Tuple[float, float]:
        """Calculate Chi-square test statistic"""
        
        # Create bins
        _, bin_edges = np.histogram(np.concatenate([reference, detection]), bins=bins)
        
        # Get observed frequencies
        ref_freq, _ = np.histogram(reference, bins=bin_edges)
        det_freq, _ = np.histogram(detection, bins=bin_edges)
        
        # Calculate expected frequencies
        total_ref = len(reference)
        total_det = len(detection)
        total_combined = total_ref + total_det
        
        combined_freq = ref_freq + det_freq
        expected_ref = combined_freq * (total_ref / total_combined)
        expected_det = combined_freq * (total_det / total_combined)
        
        # Avoid division by zero
        expected_ref = np.where(expected_ref == 0, 0.5, expected_ref)
        expected_det = np.where(expected_det == 0, 0.5, expected_det)
        
        # Calculate chi-square statistic
        chi2_ref = np.sum((ref_freq - expected_ref) ** 2 / expected_ref)
        chi2_det = np.sum((det_freq - expected_det) ** 2 / expected_det)
        chi2_stat = chi2_ref + chi2_det
        
        # Calculate p-value
        degrees_of_freedom = bins - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, degrees_of_freedom)
        
        return chi2_stat, p_value

    def _assess_drift_severity(self, drift_score: float, config: DriftConfig) -> DriftSeverity:
        """Assess drift severity based on score and thresholds"""
        
        if drift_score <= config.alert_threshold:
            return DriftSeverity.LOW
        elif drift_score <= config.warning_threshold:
            return DriftSeverity.MEDIUM
        elif drift_score <= config.warning_threshold * 2:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    async def _create_drift_alert(
        self,
        feature_name: str,
        method: DriftMethod,
        drift_score: float,
        severity: DriftSeverity,
        config: DriftConfig
    ) -> DriftAlert:
        """Create and store drift alert"""
        
        alert_id = self._generate_id("alert")
        
        # Generate recommendations based on drift type and severity
        recommendations = self._generate_recommendations(method, severity, drift_score)
        
        alert = DriftAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            drift_type=DriftType.DATA_DRIFT,  # Can be extended for other types
            severity=severity,
            feature_name=feature_name,
            drift_score=drift_score,
            threshold=config.alert_threshold,
            method=method,
            description=f"Drift detected in {feature_name} using {method.value}. Score: {drift_score:.4f}",
            recommendations=recommendations
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO drift_alerts (
                    alert_id, timestamp, drift_type, severity, feature_name,
                    drift_score, threshold_value, method, description, recommendations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_id, alert.timestamp, alert.drift_type.value, alert.severity.value,
                feature_name, drift_score, config.alert_threshold, method.value,
                alert.description, json.dumps(recommendations)
            ))
        
        self.logger.warning(f"Drift alert created: {alert_id} - {alert.description}")
        return alert

    def _generate_recommendations(
        self, 
        method: DriftMethod, 
        severity: DriftSeverity, 
        drift_score: float
    ) -> List[str]:
        """Generate recommendations based on drift detection"""
        
        recommendations = []
        
        if severity == DriftSeverity.CRITICAL:
            recommendations.extend([
                "Immediate model retraining recommended",
                "Consider emergency rollback to previous model version",
                "Investigate data source changes",
                "Review feature engineering pipeline"
            ])
        elif severity == DriftSeverity.HIGH:
            recommendations.extend([
                "Schedule model retraining within 24 hours",
                "Increase monitoring frequency",
                "Investigate recent data changes",
                "Consider A/B testing new model version"
            ])
        elif severity == DriftSeverity.MEDIUM:
            recommendations.extend([
                "Schedule model retraining within 1 week",
                "Monitor trend closely",
                "Review recent system changes"
            ])
        
        # Method-specific recommendations
        if method == DriftMethod.POPULATION_STABILITY_INDEX:
            if drift_score > 0.2:
                recommendations.append("PSI indicates significant population shift - review data sources")
        elif method == DriftMethod.KOLMOGOROV_SMIRNOV:
            recommendations.append("KS test indicates distribution change - analyze feature statistics")
        
        return recommendations

    async def _store_drift_statistics(
        self,
        feature_name: str,
        method: DriftMethod,
        drift_score: float,
        p_value: Optional[float],
        config: DriftConfig
    ) -> bool:
        """Store drift statistics"""
        
        stat_id = self._generate_id("stat")
        now = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO drift_statistics (
                    stat_id, timestamp, feature_name, method, drift_score, p_value,
                    reference_period_start, reference_period_end,
                    detection_period_start, detection_period_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stat_id, now, feature_name, method.value, drift_score, p_value,
                now - timedelta(days=config.reference_window_days + config.detection_window_days),
                now - timedelta(days=config.detection_window_days),
                now - timedelta(days=config.detection_window_days),
                now
            ))
        
        return True

    async def get_drift_summary(self, feature_name: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get drift monitoring summary"""
        
        summary = {
            "period_days": days,
            "features": {},
            "alerts": {
                "total": 0,
                "by_severity": {},
                "recent": []
            }
        }
        
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get alerts
            alert_query = '''
                SELECT drift_type, severity, feature_name, drift_score, timestamp, description
                FROM drift_alerts WHERE timestamp >= ?
            '''
            params = [since_date]
            
            if feature_name:
                alert_query += ' AND feature_name = ?'
                params.append(feature_name)
            
            alert_query += ' ORDER BY timestamp DESC'
            
            cursor = conn.execute(alert_query, params)
            alerts = cursor.fetchall()
            
            summary["alerts"]["total"] = len(alerts)
            
            # Count by severity
            for alert in alerts:
                severity = alert[1]
                summary["alerts"]["by_severity"][severity] = summary["alerts"]["by_severity"].get(severity, 0) + 1
            
            # Recent alerts (last 10)
            for alert in alerts[:10]:
                summary["alerts"]["recent"].append({
                    "drift_type": alert[0],
                    "severity": alert[1],
                    "feature_name": alert[2],
                    "drift_score": alert[3],
                    "timestamp": alert[4],
                    "description": alert[5]
                })
            
            # Get feature statistics
            stat_query = '''
                SELECT feature_name, method, AVG(drift_score), MAX(drift_score), COUNT(*)
                FROM drift_statistics WHERE timestamp >= ?
            '''
            if feature_name:
                stat_query += ' AND feature_name = ?'
            
            stat_query += ' GROUP BY feature_name, method'
            
            cursor = conn.execute(stat_query, params)
            stats = cursor.fetchall()
            
            for stat in stats:
                fname = stat[0]
                method = stat[1]
                avg_score = stat[2]
                max_score = stat[3]
                count = stat[4]
                
                if fname not in summary["features"]:
                    summary["features"][fname] = {}
                
                summary["features"][fname][method] = {
                    "avg_drift_score": avg_score,
                    "max_drift_score": max_score,
                    "measurements": count
                }
        
        return summary

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge a drift alert"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE drift_alerts SET acknowledged = 1 WHERE alert_id = ?
            ''', (alert_id,))
        
        self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Mark a drift alert as resolved"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE drift_alerts SET resolved = 1 WHERE alert_id = ?
            ''', (alert_id,))
        
        self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True

# Example usage
async def example_drift_monitoring():
    """Example drift monitoring setup"""
    
    detector = DriftDetector()
    
    # Configure monitoring for Echo decision confidence
    await detector.configure_feature_monitoring(
        feature_name="echo_decision_confidence",
        drift_methods=[
            DriftMethod.KOLMOGOROV_SMIRNOV,
            DriftMethod.POPULATION_STABILITY_INDEX,
            DriftMethod.JENSEN_SHANNON_DIVERGENCE
        ],
        alert_threshold=0.05,
        warning_threshold=0.1
    )
    
    # Simulate some observations
    import random
    for i in range(1000):
        # Simulate drift by changing distribution after 500 observations
        if i < 500:
            confidence = random.normalvariate(0.8, 0.1)
        else:
            confidence = random.normalvariate(0.6, 0.15)  # Drift: lower mean, higher variance
        
        await detector.record_observation(
            feature_name="echo_decision_confidence",
            feature_value=max(0, min(1, confidence)),
            prediction=random.random(),
            actual_value=random.randint(0, 1)
        )
    
    # Run drift detection
    alerts = await detector.detect_drift("echo_decision_confidence")
    
    print(f"Detected {len(alerts)} drift alerts")
    for alert in alerts:
        print(f"  - {alert.severity.value}: {alert.description}")
    
    # Get summary
    summary = await detector.get_drift_summary()
    print(f"Drift summary: {json.dumps(summary, indent=2, default=str)}")

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, HTTPException
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # FastAPI app for drift monitoring API
    app = FastAPI(title="Echo Drift Detection System", version="1.0.0")
    drift_detector = DriftDetector()
    
    @app.post("/monitoring/configure")
    async def configure_monitoring_endpoint(
        feature_name: str,
        drift_methods: List[str] = None,
        alert_threshold: float = 0.05,
        warning_threshold: float = 0.1
    ):
        try:
            methods = [DriftMethod(m) for m in drift_methods or ["ks_test", "psi"]]
            success = await drift_detector.configure_feature_monitoring(
                feature_name, methods, alert_threshold, warning_threshold
            )
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/monitoring/record")
    async def record_observation_endpoint(
        feature_name: str,
        feature_value: Optional[float] = None,
        feature_vector: Optional[List[float]] = None,
        prediction: Optional[float] = None,
        actual_value: Optional[float] = None
    ):
        try:
            success = await drift_detector.record_observation(
                feature_name, feature_value, feature_vector, prediction, actual_value
            )
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/monitoring/detect/{feature_name}")
    async def detect_drift_endpoint(feature_name: str):
        try:
            alerts = await drift_detector.detect_drift(feature_name)
            return {
                "alerts_count": len(alerts),
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "severity": alert.severity.value,
                        "drift_score": alert.drift_score,
                        "method": alert.method.value,
                        "description": alert.description
                    }
                    for alert in alerts
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/monitoring/summary")
    async def get_summary_endpoint(feature_name: Optional[str] = None, days: int = 30):
        try:
            summary = await drift_detector.get_drift_summary(feature_name, days)
            return summary
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert_endpoint(alert_id: str):
        try:
            success = await drift_detector.acknowledge_alert(alert_id)
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8342)
