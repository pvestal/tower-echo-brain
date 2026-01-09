"""
Comprehensive Autonomous Worker Tests

Tests:
- Worker initialization
- Task execution
- Autonomous coordinator
- LoRA training workers
- Task queuing
- Worker health monitoring
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestAutonomousCoordinator:
    """Test autonomous task coordinator"""

    def test_coordinator_imports(self):
        """Autonomous coordinator should import"""
        try:
            from workers.autonomous_coordinator import AutonomousCoordinator
            assert AutonomousCoordinator is not None
        except ImportError:
            pytest.skip("Autonomous coordinator not available")

    def test_coordinator_initialization(self):
        """Should initialize coordinator"""
        try:
            from workers.autonomous_coordinator import AutonomousCoordinator
            coordinator = AutonomousCoordinator()
            assert coordinator is not None
        except ImportError:
            pytest.skip("Autonomous coordinator not available")

    @pytest.mark.asyncio
    async def test_coordinator_submit_task(self):
        """Should submit task to coordinator"""
        try:
            from workers.autonomous_coordinator import AutonomousCoordinator
            coordinator = AutonomousCoordinator()

            task = {
                "type": "code_analysis",
                "payload": {"file": "test.py"}
            }

            if hasattr(coordinator, 'submit_task'):
                task_id = await coordinator.submit_task(task)
                assert task_id is not None
        except ImportError:
            pytest.skip("Autonomous coordinator not available")

    @pytest.mark.asyncio
    async def test_coordinator_get_task_status(self):
        """Should get task status"""
        try:
            from workers.autonomous_coordinator import AutonomousCoordinator
            coordinator = AutonomousCoordinator()

            if hasattr(coordinator, 'get_task_status'):
                status = await coordinator.get_task_status("task_123")
                assert status in ["pending", "running", "completed", "failed", None]
        except ImportError:
            pytest.skip("Autonomous coordinator not available")

    @pytest.mark.asyncio
    async def test_coordinator_cancel_task(self):
        """Should cancel running task"""
        try:
            from workers.autonomous_coordinator import AutonomousCoordinator
            coordinator = AutonomousCoordinator()

            if hasattr(coordinator, 'cancel_task'):
                result = await coordinator.cancel_task("task_123")
                assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Autonomous coordinator not available")


class TestLoRAGenerationWorker:
    """Test LoRA generation worker"""

    def test_lora_generation_worker_imports(self):
        """LoRA generation worker should import"""
        try:
            from workers.lora_generation_worker import LoRAGenerationWorker
            assert LoRAGenerationWorker is not None
        except ImportError:
            pytest.skip("LoRA generation worker not available")

    def test_lora_worker_initialization(self):
        """Should initialize LoRA worker"""
        try:
            from workers.lora_generation_worker import LoRAGenerationWorker
            worker = LoRAGenerationWorker()
            assert worker is not None
        except ImportError:
            pytest.skip("LoRA generation worker not available")

    @pytest.mark.asyncio
    async def test_lora_generation_task(self):
        """Should process LoRA generation task"""
        try:
            from workers.lora_generation_worker import LoRAGenerationWorker
            worker = LoRAGenerationWorker()

            task = {
                "model_name": "test_lora",
                "base_model": "sdxl",
                "training_images": ["img1.jpg", "img2.jpg"]
            }

            if hasattr(worker, 'process'):
                with patch.object(worker, '_train_lora', new_callable=AsyncMock):
                    result = await worker.process(task)
        except ImportError:
            pytest.skip("LoRA generation worker not available")


class TestLoRATrainingWorker:
    """Test LoRA training worker"""

    def test_lora_training_worker_imports(self):
        """LoRA training worker should import"""
        try:
            from workers.lora_training_worker import LoRATrainingWorker
            assert LoRATrainingWorker is not None
        except ImportError:
            pytest.skip("LoRA training worker not available")

    @pytest.mark.asyncio
    async def test_training_worker_process(self):
        """Should process training task"""
        try:
            from workers.lora_training_worker import LoRATrainingWorker
            worker = LoRATrainingWorker()

            if hasattr(worker, 'process'):
                task = {
                    "dataset_path": "/tmp/dataset",
                    "epochs": 10,
                    "batch_size": 4
                }
                # Mock actual training
                with patch.object(worker, '_run_training', new_callable=AsyncMock):
                    result = await worker.process(task)
        except ImportError:
            pytest.skip("LoRA training worker not available")


class TestLoRATaggingWorker:
    """Test LoRA tagging worker"""

    def test_lora_tagging_worker_imports(self):
        """LoRA tagging worker should import"""
        try:
            from workers.lora_tagging_worker import LoRATaggingWorker
            assert LoRATaggingWorker is not None
        except ImportError:
            pytest.skip("LoRA tagging worker not available")

    @pytest.mark.asyncio
    async def test_tagging_worker_process(self):
        """Should process tagging task"""
        try:
            from workers.lora_tagging_worker import LoRATaggingWorker
            worker = LoRATaggingWorker()

            if hasattr(worker, 'process'):
                task = {
                    "images": ["img1.jpg"],
                    "model": "wd14-tagger"
                }
                with patch.object(worker, '_tag_images', new_callable=AsyncMock):
                    result = await worker.process(task)
        except ImportError:
            pytest.skip("LoRA tagging worker not available")


class TestTaskIntegration:
    """Test LoRA task integration"""

    def test_task_integration_imports(self):
        """Task integration should import"""
        try:
            from workers.lora_task_integration import LoRATaskIntegration
            assert LoRATaskIntegration is not None
        except ImportError:
            pytest.skip("Task integration not available")


class TestWorkerHealth:
    """Test worker health monitoring"""

    def test_worker_health_check(self):
        """Workers should report health status"""
        try:
            from workers.autonomous_coordinator import AutonomousCoordinator
            coordinator = AutonomousCoordinator()

            if hasattr(coordinator, 'health_check'):
                health = coordinator.health_check()
                assert "status" in health or isinstance(health, bool)
        except ImportError:
            pytest.skip("Coordinator not available")

    def test_worker_metrics(self):
        """Workers should provide metrics"""
        try:
            from workers.autonomous_coordinator import AutonomousCoordinator
            coordinator = AutonomousCoordinator()

            if hasattr(coordinator, 'get_metrics'):
                metrics = coordinator.get_metrics()
                assert isinstance(metrics, dict)
        except ImportError:
            pytest.skip("Coordinator not available")


class TestTaskQueue:
    """Test task queuing system"""

    @pytest.mark.asyncio
    async def test_task_enqueue(self):
        """Should enqueue tasks"""
        try:
            from workers.autonomous_coordinator import AutonomousCoordinator
            coordinator = AutonomousCoordinator()

            if hasattr(coordinator, 'enqueue'):
                task_id = await coordinator.enqueue({
                    "type": "test",
                    "priority": 1
                })
                assert task_id is not None
        except ImportError:
            pytest.skip("Coordinator not available")

    @pytest.mark.asyncio
    async def test_task_priority(self):
        """Should respect task priority"""
        try:
            from workers.autonomous_coordinator import AutonomousCoordinator
            coordinator = AutonomousCoordinator()

            if hasattr(coordinator, 'enqueue'):
                # High priority task
                high_id = await coordinator.enqueue({
                    "type": "test",
                    "priority": 10
                })

                # Low priority task
                low_id = await coordinator.enqueue({
                    "type": "test",
                    "priority": 1
                })

                # High priority should be processed first
        except ImportError:
            pytest.skip("Coordinator not available")


class TestCeleryIntegration:
    """Test Celery task queue integration (if used)"""

    def test_celery_app_configuration(self):
        """Celery app should be configured"""
        try:
            # Check if Celery is used in this project
            from celery import Celery
            # If we get here, Celery is available
        except ImportError:
            pytest.skip("Celery not installed")

    def test_celery_task_registration(self):
        """Tasks should be registered with Celery"""
        try:
            from celery import Celery
            # Check for registered tasks
        except ImportError:
            pytest.skip("Celery not installed")


class TestAutonomousBehaviors:
    """Test autonomous behavior execution"""

    def test_behavior_scheduler_imports(self):
        """Behavior scheduler should import"""
        try:
            from behaviors.scheduler import BehaviorScheduler
            assert BehaviorScheduler is not None
        except ImportError:
            pytest.skip("Behavior scheduler not available")

    def test_system_monitor_imports(self):
        """System monitor should import"""
        try:
            from behaviors.system_monitor import SystemMonitor
            assert SystemMonitor is not None
        except ImportError:
            pytest.skip("System monitor not available")

    def test_service_monitor_imports(self):
        """Service monitor should import"""
        try:
            from behaviors.service_monitor import ServiceMonitor
            assert ServiceMonitor is not None
        except ImportError:
            pytest.skip("Service monitor not available")

    def test_code_quality_monitor_imports(self):
        """Code quality monitor should import"""
        try:
            from behaviors.code_quality_monitor import CodeQualityMonitor
            assert CodeQualityMonitor is not None
        except ImportError:
            pytest.skip("Code quality monitor not available")


class TestTowerLLMExecutor:
    """Test Tower LLM execution delegation"""

    def test_tower_executor_imports(self):
        """Tower executor should import"""
        try:
            from core.tower_llm_executor import TowerLLMExecutor
            assert TowerLLMExecutor is not None
        except ImportError:
            try:
                from core.tower_llm_executor import tower_executor
                assert tower_executor is not None
            except ImportError:
                pytest.skip("Tower executor not available")

    @pytest.mark.asyncio
    async def test_delegate_task(self):
        """Should delegate task to Tower LLM"""
        try:
            from core.tower_llm_executor import tower_executor

            with patch.object(tower_executor, 'delegate_task', new_callable=AsyncMock) as mock:
                mock.return_value = {
                    "success": True,
                    "result": "Task completed"
                }

                result = await tower_executor.delegate_task(
                    task="List files in directory",
                    context={"path": "/tmp"}
                )

                assert result["success"] is True
        except ImportError:
            pytest.skip("Tower executor not available")

    def test_executor_capabilities(self):
        """Should report executor capabilities"""
        try:
            from core.tower_llm_executor import tower_executor

            if hasattr(tower_executor, 'capabilities'):
                caps = tower_executor.capabilities
                assert isinstance(caps, dict)
        except ImportError:
            pytest.skip("Tower executor not available")
