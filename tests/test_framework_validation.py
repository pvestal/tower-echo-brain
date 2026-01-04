"""
Framework Validation Test
========================

This test validates that the comprehensive testing framework is working correctly.
"""

import pytest
import asyncio
from framework import (
    TestFrameworkCore,
    TestMetrics,
    LoadTestConfig,
    ModelTestCase,
    UserDataFactory,
    TestDataConfig
)


class TestFrameworkValidation:
    """Test the testing framework itself."""
    
    def test_framework_initialization(self):
        """Test that the framework can be initialized."""
        framework = TestFrameworkCore()
        assert framework is not None
        assert framework.config is not None
        assert framework.logger is not None
        
    def test_test_metrics_creation(self):
        """Test TestMetrics creation."""
        metrics = TestMetrics(
            test_name="test_example",
            start_time=1000.0
        )
        assert metrics.test_name == "test_example"
        assert metrics.start_time == 1000.0
        assert metrics.success is False  # Default value
        
    def test_load_test_config_creation(self):
        """Test LoadTestConfig creation."""
        config = LoadTestConfig(
            name="test_load",
            target_url="http://localhost:8309/health",
            concurrent_users=5,
            duration_seconds=30
        )
        assert config.name == "test_load"
        assert config.concurrent_users == 5
        assert config.duration_seconds == 30
        
    def test_model_test_case_creation(self):
        """Test ModelTestCase creation."""
        test_case = ModelTestCase(
            case_id="test_001",
            input_data={"query": "test"},
            expected_output=0.8,
            test_category="decision"
        )
        assert test_case.case_id == "test_001"
        assert test_case.test_category == "decision"
        
    def test_user_data_factory(self):
        """Test UserDataFactory."""
        factory = UserDataFactory()
        
        # Test single user generation
        user = factory.generate(count=1)
        assert isinstance(user, dict)
        assert 'user_id' in user
        assert 'username' in user
        assert 'email' in user
        
        # Test multiple users generation
        users = factory.generate(count=3)
        assert isinstance(users, list)
        assert len(users) == 3
        
        # Test schema
        schema = factory.get_schema()
        assert 'properties' in schema
        assert 'user_id' in schema['properties']
        
    def test_framework_monitoring(self):
        """Test framework monitoring capabilities."""
        framework = TestFrameworkCore()
        
        # Test monitoring context
        with framework.monitor_test("test_monitoring") as metrics:
            assert metrics.test_name == "test_monitoring"
            assert metrics.start_time > 0
            
        # Check that metrics were recorded
        assert len(framework.metrics) == 1
        recorded_metrics = framework.metrics[0]
        assert recorded_metrics.test_name == "test_monitoring"
        assert recorded_metrics.success is True
        assert recorded_metrics.duration is not None
        
    def test_performance_report_generation(self):
        """Test performance report generation."""
        framework = TestFrameworkCore()
        
        # Add some test metrics
        with framework.monitor_test("test_1"):
            pass
            
        with framework.monitor_test("test_2"):
            pass
            
        # Generate report
        report = framework.generate_performance_report()
        assert 'summary' in report
        assert 'performance' in report
        assert report['summary']['total_tests'] == 2
        assert report['summary']['successful'] == 2
        
    @pytest.mark.asyncio
    async def test_async_framework_features(self):
        """Test async framework features."""
        framework = TestFrameworkCore()
        
        async def async_test_function():
            await asyncio.sleep(0.1)
            return "success"
            
        # Test async monitoring
        with framework.monitor_test("async_test"):
            result = await async_test_function()
            assert result == "success"
            
        # Verify async test was recorded
        assert len(framework.metrics) == 1
        assert framework.metrics[0].success is True


if __name__ == "__main__":
    # Run a quick validation
    print("🧪 Validating Testing Framework...")
    
    # Test basic functionality
    framework = TestFrameworkCore()
    print("✅ Framework initialized successfully")
    
    # Test data factory
    factory = UserDataFactory()
    test_user = factory.generate()
    print(f"✅ Generated test user: {test_user['username']}")
    
    # Test monitoring
    with framework.monitor_test("validation_test"):
        import time
        time.sleep(0.1)
    
    print("✅ Test monitoring working")
    
    # Generate report
    report = framework.generate_performance_report()
    print(f"✅ Performance report generated: {report['summary']['total_tests']} tests")
    
    print("\n🎉 Testing Framework Validation Complete!")
    print("The comprehensive testing framework is ready for use.")
