"""
Autonomous Brain - Main Integration Module
Integrates all autonomous capabilities for Echo Brain
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

from .capability_registry import CapabilityRegistry, CapabilityType
from .code_executor import SandboxedCodeExecutor
from .autonomous_loop import AutonomousEventLoop, TaskPriority
from .financial_integration import PlaidFinancialIntegration
from .self_improvement import SelfImprovementSystem
from .lora_training import LoRATrainer

logger = logging.getLogger(__name__)

class AutonomousBrain:
    """Main autonomous brain system integrating all capabilities"""

    def __init__(self):
        # Initialize components
        self.capability_registry = CapabilityRegistry()
        self.code_executor = SandboxedCodeExecutor()
        self.event_loop = AutonomousEventLoop(self.capability_registry)
        self.financial = PlaidFinancialIntegration()
        self.self_improvement = SelfImprovementSystem()
        self.lora_trainer = LoRATrainer()

        # Register all capabilities
        self._register_capabilities()

        # Status
        self.is_initialized = False
        self.autonomy_level = 0.0

    def _register_capabilities(self):
        """Register all autonomous capabilities"""

        # Code execution capability
        self.capability_registry.register_capability(
            name="code_execution",
            capability_type=CapabilityType.CODE_EXECUTION,
            description="Execute code in sandboxed Docker containers",
            handler=self.code_executor.execute_code,
            requirements=["module:docker"],
            permissions=["docker_access"]
        )

        # Financial capabilities
        self.capability_registry.register_capability(
            name="get_balances",
            capability_type=CapabilityType.FINANCIAL,
            description="Get account balances via Plaid",
            handler=self.financial.get_balances,
            requirements=["module:plaid"],
            permissions=["financial_read"]
        )

        self.capability_registry.register_capability(
            name="get_transactions",
            capability_type=CapabilityType.FINANCIAL,
            description="Get financial transactions",
            handler=self.financial.get_transactions,
            requirements=["module:plaid"],
            permissions=["financial_read"]
        )

        self.capability_registry.register_capability(
            name="analyze_spending",
            capability_type=CapabilityType.FINANCIAL,
            description="Analyze spending patterns",
            handler=self.financial.analyze_spending_patterns,
            requirements=["module:plaid"],
            permissions=["financial_read"]
        )

        # Self-improvement capabilities
        self.capability_registry.register_capability(
            name="analyze_codebase",
            capability_type=CapabilityType.SELF_MODIFICATION,
            description="Analyze own codebase for improvements",
            handler=self.self_improvement.analyze_codebase,
            requirements=["module:ast", "module:git"],
            permissions=["code_write"]
        )

        self.capability_registry.register_capability(
            name="apply_improvement",
            capability_type=CapabilityType.SELF_MODIFICATION,
            description="Apply code improvements",
            handler=self.self_improvement.apply_improvement,
            requirements=["module:ast", "module:git"],
            permissions=["code_write"]
        )

        # LoRA training capabilities
        self.capability_registry.register_capability(
            name="train_character_lora",
            capability_type=CapabilityType.TRAINING,
            description="Train character LoRA adapter",
            handler=self.lora_trainer.train_character_lora,
            requirements=["module:torch", "module:peft"],
            permissions=["gpu_access", "model_training"]
        )

        self.capability_registry.register_capability(
            name="train_style_lora",
            capability_type=CapabilityType.TRAINING,
            description="Train style LoRA adapter",
            handler=self.lora_trainer.train_style_lora,
            requirements=["module:torch", "module:peft"],
            permissions=["gpu_access", "model_training"]
        )

        logger.info(f"Registered {len(self.capability_registry.capabilities)} capabilities")

    async def initialize(self) -> bool:
        """Initialize the autonomous brain"""

        try:
            logger.info("Initializing Autonomous Brain...")

            # Start event loop
            await self.event_loop.start()

            # Register event handlers
            self.event_loop.register_event_handler("task_completed", self._on_task_completed)
            self.event_loop.register_event_handler("task_failed", self._on_task_failed)

            # Calculate autonomy level
            self.autonomy_level = self._calculate_autonomy_level()

            self.is_initialized = True
            logger.info(f"Autonomous Brain initialized. Autonomy level: {self.autonomy_level:.1%}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def _calculate_autonomy_level(self) -> float:
        """Calculate current autonomy level based on active capabilities"""

        stats = self.capability_registry.get_statistics()
        total = stats["total_capabilities"]
        active = stats["active_capabilities"]

        if total == 0:
            return 0.0

        # Base autonomy from active capabilities
        base_autonomy = active / total

        # Bonus for specific capabilities
        bonuses = 0.0
        if self.capability_registry.get_capability("code_execution"):
            bonuses += 0.1
        if self.capability_registry.get_capability("analyze_codebase"):
            bonuses += 0.1
        if self.capability_registry.get_capability("get_balances"):
            bonuses += 0.05
        if self.capability_registry.get_capability("train_character_lora"):
            bonuses += 0.05

        return min(1.0, base_autonomy + bonuses)

    async def _on_task_completed(self, task):
        """Handle completed tasks"""

        logger.info(f"Task completed: {task.name}")

        # Learn from successful execution
        if task.capability_required == "code_execution":
            await self.self_improvement.learn_from_execution(task.result)

    async def _on_task_failed(self, task):
        """Handle failed tasks"""

        logger.warning(f"Task failed: {task.name}")

        # Try to self-improve based on failure
        if task.capability_required == "code_execution":
            improvement_result = await self.self_improvement.learn_from_execution(task.result)

            if improvement_result.get("learned"):
                # Retry task with improvements
                await self.event_loop.add_task(
                    name=f"Retry: {task.name}",
                    description=f"Retrying after improvement: {improvement_result.get('action')}",
                    priority=TaskPriority.HIGH,
                    capability=task.capability_required,
                    parameters=task.parameters
                )

    async def execute_autonomous_task(
        self,
        task_description: str,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Execute an autonomous task

        Args:
            task_description: Natural language task description
            priority: Task priority

        Returns:
            Task ID
        """

        # Analyze task to determine required capability
        capability = self._analyze_task_requirements(task_description)

        # Extract parameters from description
        parameters = self._extract_task_parameters(task_description, capability)

        # Add to event loop
        task_id = await self.event_loop.add_task(
            name=task_description[:50],
            description=task_description,
            priority=priority,
            capability=capability,
            parameters=parameters
        )

        return task_id

    def _analyze_task_requirements(self, description: str) -> str:
        """Analyze task description to determine required capability"""

        description_lower = description.lower()

        if "code" in description_lower or "execute" in description_lower:
            return "code_execution"
        elif "balance" in description_lower or "account" in description_lower:
            return "get_balances"
        elif "transaction" in description_lower or "spending" in description_lower:
            return "get_transactions"
        elif "analyze" in description_lower and "code" in description_lower:
            return "analyze_codebase"
        elif "train" in description_lower and "lora" in description_lower:
            return "train_character_lora"
        else:
            return "code_execution"  # Default

    def _extract_task_parameters(self, description: str, capability: str) -> Dict[str, Any]:
        """Extract parameters from task description"""

        # Basic parameter extraction
        # This would need more sophisticated NLP in production

        if capability == "code_execution":
            # Look for code blocks
            if "python" in description.lower():
                return {"language": "python", "code": "print('Task executed')"}
            else:
                return {"language": "bash", "code": "echo 'Task executed'"}

        elif capability == "get_transactions":
            # Look for date ranges
            return {"days_back": 30}

        else:
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get current status of autonomous brain"""

        return {
            "initialized": self.is_initialized,
            "autonomy_level": f"{self.autonomy_level:.1%}",
            "capabilities": self.capability_registry.get_statistics(),
            "event_loop": self.event_loop.get_status(),
            "execution_history": {
                "total": len(self.code_executor.execution_history),
                "recent": self.code_executor.execution_history[-5:]
            }
        }

    async def shutdown(self):
        """Shutdown autonomous brain"""

        logger.info("Shutting down Autonomous Brain...")

        # Stop event loop
        await self.event_loop.stop()

        # Save state
        self._save_state()

        self.is_initialized = False

    def _save_state(self):
        """Save current state for persistence"""

        state = {
            "capabilities": self.capability_registry.export_capabilities(),
            "autonomy_level": self.autonomy_level,
            "execution_history": self.code_executor.execution_history[-100:],
            "training_history": self.lora_trainer.get_training_history()
        }

        state_file = Path("/opt/tower-echo-brain/data/autonomous_state.json")
        state_file.parent.mkdir(exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2))

        logger.info("State saved successfully")


# Test function for autonomous capabilities
async def test_autonomous_capabilities():
    """Test all autonomous capabilities"""

    logger.info("=" * 50)
    logger.info("TESTING AUTONOMOUS CAPABILITIES")
    logger.info("=" * 50)

    brain = AutonomousBrain()

    # Initialize
    await brain.initialize()

    # Test 1: Code execution
    logger.info("\n1. Testing code execution...")
    result = await brain.capability_registry.execute_capability(
        "code_execution",
        code="def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint([fibonacci(i) for i in range(10)])",
        language="python"
    )
    logger.info(f"Result: {result}")

    # Test 2: Self-improvement
    logger.info("\n2. Testing self-improvement...")
    analysis = await brain.capability_registry.execute_capability(
        "analyze_codebase"
    )
    logger.info(f"Codebase analysis: Files={analysis.get('total_files', 0)}, Issues={len(analysis.get('issues', []))}")

    # Test 3: Financial (will fail without auth)
    logger.info("\n3. Testing financial integration...")
    balances = await brain.capability_registry.execute_capability(
        "get_balances"
    )
    logger.info(f"Financial result: {balances}")

    # Test 4: Autonomous task
    logger.info("\n4. Testing autonomous task execution...")
    task_id = await brain.execute_autonomous_task(
        "Write and execute a Python function to calculate prime numbers",
        TaskPriority.HIGH
    )
    logger.info(f"Task submitted: {task_id}")

    # Wait for task completion
    await asyncio.sleep(10)

    # Get final status
    status = brain.get_status()
    logger.info("\nFinal Status:")
    logger.info(f"Autonomy Level: {status['autonomy_level']}")
    logger.info(f"Active Capabilities: {status['capabilities']['active_capabilities']}/{status['capabilities']['total_capabilities']}")
    logger.info(f"Event Loop: {status['event_loop']['running_tasks']} running, {status['event_loop']['completed_tasks']} completed")

    # Shutdown
    await brain.shutdown()

    logger.info("\n" + "=" * 50)
    logger.info("AUTONOMOUS CAPABILITIES TEST COMPLETE")
    logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_autonomous_capabilities())