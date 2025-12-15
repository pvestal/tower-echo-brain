"""
Enhanced Testing Framework for Echo Brain Modernization
=======================================================

This comprehensive testing framework provides all the infrastructure needed
to support Echo Brain's transition to modular architecture with extensive
testing capabilities.

Main Components:
- TestFrameworkCore: Core testing infrastructure
- IntegrationTestPipeline: Service integration testing
- PerformanceTestSuite: Load and stress testing
- AITestingSuite: AI model validation
- RegressionTester: Regression detection
- TestDataManager: Test data and fixtures

Author: Development Testing Framework Agent
Created: 2025-11-06
"""

from tests.framework.test_framework_core import (
    TestFrameworkCore,
    TestMetrics,
    TestSuiteConfig
)

from tests.framework.integration_testing import (
    IntegrationTestPipeline,
    ServiceManager,
    DatabaseManager,
    ServiceConfig,
    DatabaseConfig,
    IntegrationTestResult
)

from tests.framework.performance_testing import (
    PerformanceTestSuite,
    LoadTester,
    StressTester,
    SystemMonitor,
    LoadTestConfig,
    StressTestConfig,
    PerformanceMetrics
)

from tests.framework.ai_testing_framework import (
    AITestingSuite,
    ModelAccuracyTester,
    LearningConvergenceTester,
    BoardOfDirectorsConsensusTester,
    MockModelManager,
    ModelTestCase,
    ModelEvaluationResult,
    LearningConvergenceConfig,
    ConsensusTestConfig
)

from tests.framework.test_data_management import (
    TestDataManager,
    FixtureManager,
    DatabaseManager as DataDatabaseManager,
    DataFactory,
    UserDataFactory,
    TaskDataFactory,
    AIModelDataFactory,
    TestDataConfig,
    DatabaseSnapshot
)

from tests.framework.regression_testing import (
    RegressionTester,
    PerformanceRegressionDetector,
    FunctionalRegressionDetector,
    APIRegressionDetector,
    RegressionBaseline,
    RegressionResult,
    RegressionConfig
)

__version__ = "1.0.0"

__all__ = [
    # Core framework
    'TestFrameworkCore',
    'TestMetrics',
    'TestSuiteConfig',
    
    # Integration testing
    'IntegrationTestPipeline',
    'ServiceManager',
    'DatabaseManager',
    'ServiceConfig',
    'DatabaseConfig',
    'IntegrationTestResult',
    
    # Performance testing
    'PerformanceTestSuite',
    'LoadTester',
    'StressTester',
    'SystemMonitor',
    'LoadTestConfig',
    'StressTestConfig',
    'PerformanceMetrics',
    
    # AI testing
    'AITestingSuite',
    'ModelAccuracyTester',
    'LearningConvergenceTester',
    'BoardOfDirectorsConsensusTester',
    'MockModelManager',
    'ModelTestCase',
    'ModelEvaluationResult',
    'LearningConvergenceConfig',
    'ConsensusTestConfig',
    
    # Test data management
    'TestDataManager',
    'FixtureManager',
    'DataDatabaseManager',
    'DataFactory',
    'UserDataFactory',
    'TaskDataFactory',
    'AIModelDataFactory',
    'TestDataConfig',
    'DatabaseSnapshot',
    
    # Regression testing
    'RegressionTester',
    'PerformanceRegressionDetector',
    'FunctionalRegressionDetector',
    'APIRegressionDetector',
    'RegressionBaseline',
    'RegressionResult',
    'RegressionConfig'
]
