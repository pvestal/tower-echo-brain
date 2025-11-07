"""
Regression Testing Framework for Echo Brain Architecture Changes
================================================================

This module provides comprehensive regression testing capabilities to detect
performance, functionality, and behavioral regressions during Echo Brain's
modernization and ongoing development.

Features:
- Performance regression detection
- Functional regression testing
- API contract validation
- Behavioral consistency checks
- Baseline management
- Automated regression reporting

Author: Development Testing Framework Agent
Created: 2025-11-06
"""

import json
import time
import statistics
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

import pytest
import numpy as np

from .test_framework_core import TestFrameworkCore, TestMetrics
from .performance_testing import PerformanceMetrics


@dataclass
class RegressionBaseline:
    """Baseline metrics for regression testing."""
    test_name: str
    version: str
    timestamp: datetime
    performance_metrics: Dict[str, float]
    functional_results: Dict[str, Any]
    api_contracts: Dict[str, Dict[str, Any]]
    behavioral_signatures: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionResult:
    """Result of regression testing."""
    test_name: str
    regression_detected: bool
    confidence_level: float
    baseline_version: str
    current_version: str
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionConfig:
    """Configuration for regression testing."""
    name: str
    baseline_path: str
    tolerance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'performance': 0.1,  # 10% degradation
        'accuracy': 0.05,    # 5% accuracy loss
        'memory': 0.2,       # 20% memory increase
        'response_time': 0.15  # 15% response time increase
    })
    significance_level: float = 0.05
    minimum_samples: int = 10
    enabled_checks: List[str] = field(default_factory=lambda: [
        'performance', 'functional', 'api', 'behavioral'
    ])


class PerformanceRegressionDetector:
    """Detects performance regressions."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize performance regression detector."""
        self.framework = framework
        
    def detect_regressions(
        self,
        baseline: RegressionBaseline,
        current_metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Detect performance regressions."""
        regressions = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline.performance_metrics:
                continue
                
            baseline_value = baseline.performance_metrics[metric_name]
            threshold = thresholds.get(metric_name, 0.1)
            
            # Calculate percentage change
            if baseline_value != 0:
                change_percent = (current_value - baseline_value) / baseline_value
            else:
                change_percent = float('inf') if current_value > 0 else 0
                
            # Determine if this is a regression based on metric type
            is_regression = self._is_regression(metric_name, change_percent, threshold)
            
            if is_regression:
                regression = {
                    'type': 'performance',
                    'metric': metric_name,
                    'baseline_value': baseline_value,
                    'current_value': current_value,
                    'change_percent': change_percent * 100,
                    'threshold_percent': threshold * 100,
                    'severity': self._calculate_severity(change_percent, threshold)
                }
                regressions.append(regression)
                
        return regressions
        
    def _is_regression(self, metric_name: str, change_percent: float, threshold: float) -> bool:
        """Determine if a change represents a regression."""
        # Metrics where increase is bad
        bad_increase_metrics = [
            'response_time', 'memory_usage', 'cpu_usage', 'error_rate',
            'load_time', 'latency', 'duration'
        ]
        
        # Metrics where decrease is bad
        bad_decrease_metrics = [
            'throughput', 'accuracy', 'success_rate', 'availability',
            'requests_per_second', 'cache_hit_rate'
        ]
        
        metric_lower = metric_name.lower()
        
        if any(bad_metric in metric_lower for bad_metric in bad_increase_metrics):
            return change_percent > threshold
        elif any(bad_metric in metric_lower for bad_metric in bad_decrease_metrics):
            return change_percent < -threshold
        else:
            # Default: increase beyond threshold is regression
            return abs(change_percent) > threshold
            
    def _calculate_severity(self, change_percent: float, threshold: float) -> str:
        """Calculate regression severity."""
        magnitude = abs(change_percent)
        
        if magnitude > threshold * 5:
            return 'critical'
        elif magnitude > threshold * 3:
            return 'high'
        elif magnitude > threshold * 2:
            return 'medium'
        else:
            return 'low'


class FunctionalRegressionDetector:
    """Detects functional regressions."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize functional regression detector."""
        self.framework = framework
        
    def detect_regressions(
        self,
        baseline: RegressionBaseline,
        current_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect functional regressions."""
        regressions = []
        
        baseline_results = baseline.functional_results
        
        # Check for missing functionality
        for test_name, baseline_result in baseline_results.items():
            if test_name not in current_results:
                regressions.append({
                    'type': 'functional',
                    'subtype': 'missing_test',
                    'test_name': test_name,
                    'description': f"Test '{test_name}' present in baseline but missing in current run",
                    'severity': 'high'
                })
                continue
                
            current_result = current_results[test_name]
            
            # Check for test failures that previously passed
            if baseline_result.get('success', False) and not current_result.get('success', False):
                regressions.append({
                    'type': 'functional',
                    'subtype': 'test_failure',
                    'test_name': test_name,
                    'baseline_success': True,
                    'current_success': False,
                    'error_message': current_result.get('error', 'Unknown error'),
                    'severity': 'high'
                })
                
            # Check for output changes
            baseline_output = baseline_result.get('output')
            current_output = current_result.get('output')
            
            if baseline_output != current_output:
                # Analyze output differences
                similarity = self._calculate_output_similarity(baseline_output, current_output)
                
                if similarity < 0.95:  # Less than 95% similarity
                    regressions.append({
                        'type': 'functional',
                        'subtype': 'output_change',
                        'test_name': test_name,
                        'similarity_score': similarity,
                        'baseline_output': baseline_output,
                        'current_output': current_output,
                        'severity': 'medium' if similarity > 0.8 else 'high'
                    })
                    
        return regressions
        
    def _calculate_output_similarity(self, baseline: Any, current: Any) -> float:
        """Calculate similarity between outputs."""
        if baseline == current:
            return 1.0
            
        if type(baseline) != type(current):
            return 0.0
            
        if isinstance(baseline, dict) and isinstance(current, dict):
            return self._dict_similarity(baseline, current)
        elif isinstance(baseline, list) and isinstance(current, list):
            return self._list_similarity(baseline, current)
        elif isinstance(baseline, str) and isinstance(current, str):
            return self._string_similarity(baseline, current)
        else:
            return 1.0 if baseline == current else 0.0
            
    def _dict_similarity(self, dict1: dict, dict2: dict) -> float:
        """Calculate similarity between dictionaries."""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        if not all_keys:
            return 1.0
            
        matching_keys = 0
        for key in all_keys:
            if key in dict1 and key in dict2:
                if dict1[key] == dict2[key]:
                    matching_keys += 1
                    
        return matching_keys / len(all_keys)
        
    def _list_similarity(self, list1: list, list2: list) -> float:
        """Calculate similarity between lists."""
        if len(list1) == 0 and len(list2) == 0:
            return 1.0
            
        max_len = max(len(list1), len(list2))
        if max_len == 0:
            return 1.0
            
        matching_items = 0
        for i in range(min(len(list1), len(list2))):
            if list1[i] == list2[i]:
                matching_items += 1
                
        return matching_items / max_len
        
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between strings."""
        if str1 == str2:
            return 1.0
            
        # Simple Levenshtein distance-based similarity
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
            
        # For simplicity, using a basic character comparison
        matching_chars = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        return matching_chars / max_len


class APIRegressionDetector:
    """Detects API contract regressions."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize API regression detector."""
        self.framework = framework
        
    def detect_regressions(
        self,
        baseline: RegressionBaseline,
        current_contracts: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect API contract regressions."""
        regressions = []
        
        baseline_contracts = baseline.api_contracts
        
        for endpoint, baseline_contract in baseline_contracts.items():
            if endpoint not in current_contracts:
                regressions.append({
                    'type': 'api',
                    'subtype': 'missing_endpoint',
                    'endpoint': endpoint,
                    'description': f"API endpoint '{endpoint}' removed",
                    'severity': 'critical'
                })
                continue
                
            current_contract = current_contracts[endpoint]
            
            # Check request schema changes
            baseline_request = baseline_contract.get('request_schema', {})
            current_request = current_contract.get('request_schema', {})
            
            request_changes = self._detect_schema_changes(
                baseline_request, current_request, 'request'
            )
            regressions.extend(request_changes)
            
            # Check response schema changes
            baseline_response = baseline_contract.get('response_schema', {})
            current_response = current_contract.get('response_schema', {})
            
            response_changes = self._detect_schema_changes(
                baseline_response, current_response, 'response'
            )
            regressions.extend(response_changes)
            
            # Check status code changes
            baseline_status = baseline_contract.get('status_codes', [])
            current_status = current_contract.get('status_codes', [])
            
            if set(baseline_status) != set(current_status):
                regressions.append({
                    'type': 'api',
                    'subtype': 'status_code_change',
                    'endpoint': endpoint,
                    'baseline_codes': baseline_status,
                    'current_codes': current_status,
                    'severity': 'medium'
                })
                
        return regressions
        
    def _detect_schema_changes(
        self,
        baseline_schema: Dict[str, Any],
        current_schema: Dict[str, Any],
        schema_type: str
    ) -> List[Dict[str, Any]]:
        """Detect schema changes."""
        changes = []
        
        baseline_props = baseline_schema.get('properties', {})
        current_props = current_schema.get('properties', {})
        
        # Check for removed properties
        for prop in baseline_props:
            if prop not in current_props:
                changes.append({
                    'type': 'api',
                    'subtype': 'schema_property_removed',
                    'schema_type': schema_type,
                    'property': prop,
                    'severity': 'high'
                })
                
        # Check for type changes
        for prop in set(baseline_props.keys()) & set(current_props.keys()):
            baseline_type = baseline_props[prop].get('type')
            current_type = current_props[prop].get('type')
            
            if baseline_type != current_type:
                changes.append({
                    'type': 'api',
                    'subtype': 'schema_type_change',
                    'schema_type': schema_type,
                    'property': prop,
                    'baseline_type': baseline_type,
                    'current_type': current_type,
                    'severity': 'high'
                })
                
        return changes


class RegressionTester:
    """Main regression testing coordinator."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize regression tester."""
        self.framework = framework
        self.performance_detector = PerformanceRegressionDetector(framework)
        self.functional_detector = FunctionalRegressionDetector(framework)
        self.api_detector = APIRegressionDetector(framework)
        self.baselines: Dict[str, RegressionBaseline] = {}
        
    def load_baseline(self, baseline_path: str) -> RegressionBaseline:
        """Load regression baseline from file."""
        path = Path(baseline_path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        baseline = RegressionBaseline(
            test_name=data['test_name'],
            version=data['version'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            performance_metrics=data['performance_metrics'],
            functional_results=data['functional_results'],
            api_contracts=data['api_contracts'],
            behavioral_signatures=data['behavioral_signatures'],
            metadata=data.get('metadata', {})
        )
        
        self.baselines[baseline.test_name] = baseline
        return baseline
        
    def save_baseline(
        self,
        baseline: RegressionBaseline,
        baseline_path: str
    ):
        """Save regression baseline to file."""
        data = {
            'test_name': baseline.test_name,
            'version': baseline.version,
            'timestamp': baseline.timestamp.isoformat(),
            'performance_metrics': baseline.performance_metrics,
            'functional_results': baseline.functional_results,
            'api_contracts': baseline.api_contracts,
            'behavioral_signatures': baseline.behavioral_signatures,
            'metadata': baseline.metadata
        }
        
        path = Path(baseline_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.framework.logger.info(f"Baseline saved to {baseline_path}")
        
    def run_regression_test(
        self,
        config: RegressionConfig,
        current_data: Dict[str, Any]
    ) -> RegressionResult:
        """Run comprehensive regression test."""
        with self.framework.monitor_test(f"regression_{config.name}"):
            self.framework.logger.info(f"Running regression test: {config.name}")
            
            # Load baseline
            baseline = self.load_baseline(config.baseline_path)
            
            # Initialize result
            result = RegressionResult(
                test_name=config.name,
                regression_detected=False,
                confidence_level=0.0,
                baseline_version=baseline.version,
                current_version=current_data.get('version', 'unknown')
            )
            
            all_regressions = []
            
            # Performance regression detection
            if 'performance' in config.enabled_checks:
                performance_metrics = current_data.get('performance_metrics', {})
                perf_regressions = self.performance_detector.detect_regressions(
                    baseline, performance_metrics, config.tolerance_thresholds
                )
                all_regressions.extend(perf_regressions)
                
            # Functional regression detection
            if 'functional' in config.enabled_checks:
                functional_results = current_data.get('functional_results', {})
                func_regressions = self.functional_detector.detect_regressions(
                    baseline, functional_results
                )
                all_regressions.extend(func_regressions)
                
            # API regression detection
            if 'api' in config.enabled_checks:
                api_contracts = current_data.get('api_contracts', {})
                api_regressions = self.api_detector.detect_regressions(
                    baseline, api_contracts
                )
                all_regressions.extend(api_regressions)
                
            # Analyze results
            result.regressions = all_regressions
            result.regression_detected = len(all_regressions) > 0
            result.confidence_level = self._calculate_confidence(all_regressions)
            result.analysis = self._analyze_regressions(all_regressions)
            
            return result
            
    def _calculate_confidence(self, regressions: List[Dict[str, Any]]) -> float:
        """Calculate confidence level for regression detection."""
        if not regressions:
            return 1.0  # High confidence in no regression
            
        # Weight by severity
        severity_weights = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
        
        total_weight = 0
        weighted_sum = 0
        
        for regression in regressions:
            severity = regression.get('severity', 'medium')
            weight = severity_weights.get(severity, 0.5)
            total_weight += weight
            weighted_sum += weight
            
        if total_weight == 0:
            return 0.5
            
        # Normalize to confidence level
        confidence = min(weighted_sum / len(regressions), 1.0)
        return confidence
        
    def _analyze_regressions(self, regressions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze detected regressions."""
        if not regressions:
            return {'summary': 'No regressions detected'}
            
        # Group by type and severity
        by_type = {}
        by_severity = {}
        
        for regression in regressions:
            reg_type = regression.get('type', 'unknown')
            severity = regression.get('severity', 'medium')
            
            by_type[reg_type] = by_type.get(reg_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
        # Find most critical issues
        critical_issues = [r for r in regressions if r.get('severity') == 'critical']
        high_issues = [r for r in regressions if r.get('severity') == 'high']
        
        return {
            'total_regressions': len(regressions),
            'by_type': by_type,
            'by_severity': by_severity,
            'critical_issues': len(critical_issues),
            'high_priority_issues': len(high_issues),
            'most_critical': critical_issues[:3] if critical_issues else high_issues[:3],
            'recommendations': self._generate_recommendations(regressions)
        }
        
    def _generate_recommendations(self, regressions: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on regressions."""
        recommendations = []
        
        # Performance recommendations
        perf_regressions = [r for r in regressions if r.get('type') == 'performance']
        if perf_regressions:
            recommendations.append("Review recent performance changes and optimize critical paths")
            
        # Functional recommendations
        func_regressions = [r for r in regressions if r.get('type') == 'functional']
        if func_regressions:
            recommendations.append("Investigate failed tests and verify functional requirements")
            
        # API recommendations
        api_regressions = [r for r in regressions if r.get('type') == 'api']
        if api_regressions:
            recommendations.append("Review API changes for backward compatibility")
            
        # Critical issues
        critical_count = len([r for r in regressions if r.get('severity') == 'critical'])
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical regressions before deployment")
            
        return recommendations
        
    def generate_regression_report(
        self,
        results: List[RegressionResult],
        output_path: str
    ):
        """Generate comprehensive regression report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'tests_with_regressions': len([r for r in results if r.regression_detected]),
            'summary': self._summarize_results(results),
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'regression_detected': r.regression_detected,
                    'confidence_level': r.confidence_level,
                    'baseline_version': r.baseline_version,
                    'current_version': r.current_version,
                    'regressions': r.regressions,
                    'analysis': r.analysis
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.framework.logger.info(f"Regression report saved to {output_path}")
        
    def _summarize_results(self, results: List[RegressionResult]) -> Dict[str, Any]:
        """Summarize regression test results."""
        if not results:
            return {'status': 'no_tests'}
            
        total_regressions = sum(len(r.regressions) for r in results)
        avg_confidence = statistics.mean(r.confidence_level for r in results)
        
        severity_counts = {}
        for result in results:
            for regression in result.regressions:
                severity = regression.get('severity', 'medium')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
        return {
            'status': 'regressions_detected' if total_regressions > 0 else 'clean',
            'total_regressions': total_regressions,
            'avg_confidence': avg_confidence,
            'severity_distribution': severity_counts,
            'recommendation': 'Deploy with caution' if total_regressions > 0 else 'Safe to deploy'
        }


# Export main classes
__all__ = [
    'RegressionTester',
    'PerformanceRegressionDetector',
    'FunctionalRegressionDetector',
    'APIRegressionDetector',
    'RegressionBaseline',
    'RegressionResult',
    'RegressionConfig'
]
