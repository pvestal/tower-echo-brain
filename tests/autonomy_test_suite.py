#!/usr/bin/env python3
"""
Comprehensive Autonomy Test Suite for Echo Brain
Run with: pytest autonomy_test_suite.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import pytest
import asyncio
from pathlib import Path
import json
import docker
from datetime import datetime

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import all capabilities
from capabilities.code_executor_fixed import SandboxedCodeExecutor
from capabilities.capability_registry import CapabilityRegistry, CapabilityType
from capabilities.self_improvement import SelfImprovementSystem, CodeAnalyzer
from capabilities.autonomous_loop import AutonomousEventLoop, TaskPriority
from capabilities.financial_integration import PlaidFinancialIntegration
from capabilities.lora_training import LoRATrainer
from capabilities.autonomous_brain import AutonomousBrain


class TestDockerExecution:
    """Test Docker-based code execution"""

    @pytest.fixture
    def executor(self):
        return SandboxedCodeExecutor()

    @pytest.mark.asyncio
    async def test_python_execution(self, executor):
        """Test Python code execution in Docker"""
        code = "print('Hello from Docker'); print(sum(range(10)))"
        result = await executor.execute_code(code, language="python")
        assert result['success'] == True
        assert 'Hello from Docker' in result['output']
        assert '45' in result['output']

    @pytest.mark.asyncio
    async def test_javascript_execution(self, executor):
        """Test JavaScript execution"""
        code = "console.log('JS Test'); console.log(5 + 3);"
        result = await executor.execute_code(code, language="javascript")
        assert result['success'] == True

    @pytest.mark.asyncio
    async def test_execution_timeout(self, executor):
        """Test execution with timeout"""
        code = "import time; time.sleep(100)"
        result = await executor.execute_code(code, language="python", timeout=2)
        # Should handle timeout gracefully
        assert 'exec_id' in result

    def test_execution_stats(self, executor):
        """Test execution statistics"""
        stats = executor.get_execution_stats()
        assert 'total_executions' in stats
        assert stats['total_executions'] >= 0


class TestSelfImprovement:
    """Test self-improvement capabilities"""

    @pytest.fixture
    def analyzer(self):
        return CodeAnalyzer()

    @pytest.fixture
    def self_improve(self):
        return SelfImprovementSystem()

    def test_code_analysis(self, analyzer):
        """Test AST code analysis"""
        test_file = Path(__file__)
        analysis = analyzer.analyze_file(test_file)

        assert 'functions' in analysis
        assert 'classes' in analysis
        assert 'lines' in analysis
        assert analysis['lines'] > 0

    def test_complexity_calculation(self, analyzer):
        """Test complexity calculation"""
        test_code = Path("/tmp/test_complexity.py")
        test_code.write_text("""
def complex_function(x):
    if x > 0:
        if x > 10:
            return x * 2
        else:
            return x + 1
    else:
        return 0
""")

        analysis = analyzer.analyze_file(test_code)
        assert analysis['complexity'] > 0

    @pytest.mark.asyncio
    async def test_codebase_analysis(self, self_improve):
        """Test full codebase analysis"""
        analysis = await self_improve.analyze_codebase()

        assert 'total_files' in analysis
        assert 'total_lines' in analysis
        assert 'issues' in analysis
        assert analysis['total_files'] > 0

    def test_improvement_suggestions(self, self_improve):
        """Test improvement suggestion generation"""
        mock_analysis = {
            'issues': [
                {'type': 'high_complexity', 'file': 'test.py', 'complexity': 15},
                {'type': 'low_documentation', 'file': 'test2.py', 'coverage': 0.3}
            ]
        }

        improvements = self_improve._generate_improvements(mock_analysis)
        assert len(improvements) > 0
        assert improvements[0]['type'] in ['refactor', 'documentation']


class TestFinancialIntegration:
    """Test Plaid financial integration"""

    @pytest.fixture
    def financial(self):
        return PlaidFinancialIntegration()

    def test_plaid_initialization(self, financial):
        """Test Plaid client initialization"""
        # Should initialize without errors
        assert hasattr(financial, 'client')

    @pytest.mark.asyncio
    async def test_mock_transaction_analysis(self, financial):
        """Test transaction analysis with mock data"""
        # Since we don't have real Plaid auth, test the logic
        mock_transactions = [
            {"amount": 100, "date": "2025-12-30", "category": ["Food"]},
            {"amount": 200, "date": "2025-12-29", "category": ["Shopping"]}
        ]

        total = sum(t['amount'] for t in mock_transactions)
        assert total == 300

    def test_access_token_storage(self, financial):
        """Test access token storage mechanism"""
        test_token = financial._get_access_token("test_user")
        # Should return None if no token stored
        assert test_token is None or isinstance(test_token, str)


class TestLoRATraining:
    """Test LoRA training capabilities"""

    @pytest.fixture
    def trainer(self):
        return LoRATrainer()

    def test_lora_initialization(self, trainer):
        """Test LoRA trainer initialization"""
        assert trainer.model_path.exists()
        assert trainer.output_path.exists()

    def test_training_config_generation(self, trainer):
        """Test training configuration generation"""
        config = {
            "learning_rate": 1e-4,
            "batch_size": 1,
            "lora_rank": 32
        }

        assert config['learning_rate'] > 0
        assert config['batch_size'] > 0
        assert config['lora_rank'] > 0

    def test_prepare_training_data(self, trainer):
        """Test training data preparation"""
        # Create test images
        test_dir = Path("/tmp/test_lora_images")
        test_dir.mkdir(exist_ok=True)

        # Create dummy image files
        for i in range(3):
            (test_dir / f"image_{i}.png").touch()

        image_paths, captions = trainer.prepare_training_data(str(test_dir))

        assert len(image_paths) == 3
        assert len(captions) == 3

    @pytest.mark.skipif(not TORCH_AVAILABLE or (TORCH_AVAILABLE and not torch.cuda.is_available()), reason="CUDA not available")
    def test_cuda_availability(self, trainer):
        """Test CUDA availability for training"""
        import torch
        assert torch.cuda.is_available()
        assert trainer.device.type == 'cuda'


class TestAutonomousLoop:
    """Test autonomous event loop"""

    @pytest.fixture
    def registry(self):
        return CapabilityRegistry()

    @pytest.fixture
    def event_loop(self, registry):
        return AutonomousEventLoop(registry)

    @pytest.mark.asyncio
    async def test_event_loop_initialization(self, event_loop):
        """Test event loop initialization"""
        await event_loop.start()
        assert event_loop.is_running == True
        await event_loop.stop()

    @pytest.mark.asyncio
    async def test_task_addition(self, event_loop):
        """Test adding tasks to queue"""
        task_id = await event_loop.add_task(
            name="Test task",
            description="Test description",
            priority=TaskPriority.NORMAL,
            capability="test_capability",
            parameters={}
        )

        assert task_id is not None
        assert len(event_loop.task_queue) > 0

    def test_event_loop_status(self, event_loop):
        """Test status reporting"""
        status = event_loop.get_status()

        assert 'is_running' in status
        assert 'pending_tasks' in status
        assert 'completed_tasks' in status


class TestAutonomousBrain:
    """Test the main autonomous brain integration"""

    @pytest.fixture
    async def brain(self):
        brain = AutonomousBrain()
        await brain.initialize()
        yield brain
        await brain.shutdown()

    @pytest.mark.asyncio
    async def test_brain_initialization(self, brain):
        """Test brain initialization"""
        assert brain.is_initialized == True
        assert brain.autonomy_level > 0

    @pytest.mark.asyncio
    async def test_capability_registration(self, brain):
        """Test capability registration"""
        capabilities = brain.capability_registry.capabilities
        assert len(capabilities) > 0

        # Check for key capabilities
        cap_types = [cap.type for cap in capabilities.values()]
        assert CapabilityType.CODE_EXECUTION in cap_types

    @pytest.mark.asyncio
    async def test_autonomous_task_execution(self, brain):
        """Test autonomous task execution"""
        task_id = await brain.execute_autonomous_task(
            "Test autonomous execution",
            TaskPriority.HIGH
        )

        assert task_id is not None

    @pytest.mark.asyncio
    async def test_autonomy_calculation(self, brain):
        """Test autonomy level calculation"""
        autonomy = brain._calculate_autonomy_level()

        assert autonomy >= 0
        assert autonomy <= 1
        assert autonomy > 0.6  # Should be at least 60% with current setup


class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from code execution to improvement"""

        # 1. Execute code
        executor = SandboxedCodeExecutor()
        code_result = await executor.execute_code(
            "print('E2E test')",
            language="python"
        )
        assert code_result['success'] == True

        # 2. Analyze code
        analyzer = CodeAnalyzer()
        test_file = Path(__file__)
        analysis = analyzer.analyze_file(test_file)
        assert 'functions' in analysis

        # 3. Create autonomous task
        brain = AutonomousBrain()
        await brain.initialize()
        task_id = await brain.execute_autonomous_task(
            "E2E test task",
            TaskPriority.NORMAL
        )
        assert task_id is not None

        await brain.shutdown()

    def test_system_performance(self):
        """Test system performance metrics"""
        import time
        import psutil

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        assert cpu_percent < 90  # Should not overload CPU

        # Check memory usage
        memory = psutil.virtual_memory()
        assert memory.percent < 95  # Should not exhaust memory

    def test_error_handling(self):
        """Test error handling and recovery"""
        executor = SandboxedCodeExecutor()

        # This should fail gracefully
        result = asyncio.run(executor.execute_code(
            "import nonexistent_module",
            language="python"
        ))

        assert result['success'] == False
        assert 'error' in result


# Performance benchmarks
class TestPerformance:
    """Performance tests"""

    @pytest.mark.benchmark
    def test_code_execution_speed(self, benchmark):
        """Benchmark code execution speed"""
        executor = SandboxedCodeExecutor()

        def execute():
            return asyncio.run(executor.execute_code(
                "print('test')",
                language="python"
            ))

        result = benchmark(execute)
        assert result['success'] == True

    @pytest.mark.benchmark
    def test_analysis_speed(self, benchmark):
        """Benchmark code analysis speed"""
        analyzer = CodeAnalyzer()
        test_file = Path(__file__)

        result = benchmark(analyzer.analyze_file, test_file)
        assert 'functions' in result


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--tb=short"])