#!/usr/bin/env python3
"""
Protocol Validation System
Validates that implementations comply with their declared protocols
"""

import logging
import asyncio
from typing import Any, Dict, List, Type, get_type_hints
from datetime import datetime

from .vector_memory import VectorMemoryInterface
from .database import AsyncDatabaseInterface, ConversationDatabaseInterface
from .task_orchestrator import TaskOrchestratorInterface
from .model_manager import ModelInterface, ModelManagerInterface
from .conversation import ConversationManagerInterface
from .security import AuthenticationInterface, AuthorizationInterface

logger = logging.getLogger(__name__)

class ProtocolValidator:
    """
    Validates protocol compliance for Echo Brain implementations
    """

    def __init__(self):
        self.validation_results: Dict[str, Dict[str, Any]] = {}

    def validate_class_protocol_compliance(self,
                                         instance: Any,
                                         protocol_class: Type) -> Dict[str, Any]:
        """
        Validate that a class instance implements all protocol methods

        Args:
            instance: The class instance to validate
            protocol_class: The protocol class to validate against

        Returns:
            Dict: Validation results with compliance status and missing methods
        """
        protocol_name = protocol_class.__name__
        instance_name = instance.__class__.__name__

        # Get all protocol methods
        protocol_methods = set()
        for attr_name in dir(protocol_class):
            if not attr_name.startswith('_'):
                attr = getattr(protocol_class, attr_name)
                if callable(attr):
                    protocol_methods.add(attr_name)

        # Check implementation methods
        instance_methods = set()
        missing_methods = set()

        for method_name in protocol_methods:
            if hasattr(instance, method_name) and callable(getattr(instance, method_name)):
                instance_methods.add(method_name)
            else:
                missing_methods.add(method_name)

        # Check type hints compliance
        type_hint_compliance = self._check_type_hints(instance, protocol_class)

        compliance_score = (len(instance_methods) / len(protocol_methods)) * 100 if protocol_methods else 100

        result = {
            "protocol": protocol_name,
            "implementation": instance_name,
            "compliance_score": compliance_score,
            "is_compliant": len(missing_methods) == 0,
            "total_methods": len(protocol_methods),
            "implemented_methods": len(instance_methods),
            "missing_methods": list(missing_methods),
            "type_hint_compliance": type_hint_compliance,
            "validated_at": datetime.now().isoformat()
        }

        self.validation_results[f"{instance_name}_{protocol_name}"] = result
        return result

    def _check_type_hints(self, instance: Any, protocol_class: Type) -> Dict[str, Any]:
        """Check type hint compliance between implementation and protocol"""
        protocol_hints = {}
        instance_hints = {}

        try:
            protocol_hints = get_type_hints(protocol_class)
            instance_hints = get_type_hints(instance.__class__)
        except Exception as e:
            logger.warning(f"Failed to get type hints: {e}")
            return {"status": "error", "error": str(e)}

        matching_hints = 0
        total_hints = len(protocol_hints)
        mismatched_hints = []

        for method_name, protocol_hint in protocol_hints.items():
            if method_name in instance_hints:
                if instance_hints[method_name] == protocol_hint:
                    matching_hints += 1
                else:
                    mismatched_hints.append({
                        "method": method_name,
                        "expected": str(protocol_hint),
                        "actual": str(instance_hints[method_name])
                    })

        return {
            "status": "success",
            "total_hints": total_hints,
            "matching_hints": matching_hints,
            "compliance_percentage": (matching_hints / total_hints * 100) if total_hints > 0 else 100,
            "mismatched_hints": mismatched_hints
        }

    async def validate_functional_compliance(self,
                                           instance: Any,
                                           protocol_class: Type) -> Dict[str, Any]:
        """
        Perform functional validation by testing key protocol methods

        Args:
            instance: The class instance to validate
            protocol_class: The protocol class to validate against

        Returns:
            Dict: Functional validation results
        """
        protocol_name = protocol_class.__name__
        instance_name = instance.__class__.__name__

        try:
            if isinstance(instance, VectorMemoryInterface):
                return await self._test_vector_memory_functions(instance)
            elif isinstance(instance, AsyncDatabaseInterface):
                return await self._test_async_database_functions(instance)
            elif isinstance(instance, TaskOrchestratorInterface):
                return await self._test_task_orchestrator_functions(instance)
            else:
                return {
                    "status": "skipped",
                    "reason": f"No functional tests defined for {protocol_name}",
                    "protocol": protocol_name,
                    "implementation": instance_name
                }
        except Exception as e:
            logger.error(f"Functional validation failed for {instance_name}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "protocol": protocol_name,
                "implementation": instance_name
            }

    async def _test_vector_memory_functions(self, memory: VectorMemoryInterface) -> Dict[str, Any]:
        """Test VectorMemory functional compliance"""
        test_results = []

        # Test 1: Get statistics
        try:
            stats = await memory.get_statistics()
            test_results.append({
                "test": "get_statistics",
                "status": "pass" if isinstance(stats, dict) else "fail",
                "result": "Returns dictionary" if isinstance(stats, dict) else "Invalid return type"
            })
        except Exception as e:
            test_results.append({
                "test": "get_statistics",
                "status": "error",
                "error": str(e)
            })

        # Test 2: Get embedding model
        try:
            model = memory.get_embedding_model()
            test_results.append({
                "test": "get_embedding_model",
                "status": "pass" if isinstance(model, str) else "fail",
                "result": f"Model: {model}" if isinstance(model, str) else "Invalid return type"
            })
        except Exception as e:
            test_results.append({
                "test": "get_embedding_model",
                "status": "error",
                "error": str(e)
            })

        # Test 3: Get vector dimensions
        try:
            dimensions = memory.get_vector_dimensions()
            test_results.append({
                "test": "get_vector_dimensions",
                "status": "pass" if isinstance(dimensions, int) and dimensions > 0 else "fail",
                "result": f"Dimensions: {dimensions}" if isinstance(dimensions, int) else "Invalid return type"
            })
        except Exception as e:
            test_results.append({
                "test": "get_vector_dimensions",
                "status": "error",
                "error": str(e)
            })

        passed_tests = sum(1 for test in test_results if test["status"] == "pass")
        total_tests = len(test_results)

        return {
            "status": "completed",
            "protocol": "VectorMemoryInterface",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "test_results": test_results
        }

    async def _test_async_database_functions(self, db: AsyncDatabaseInterface) -> Dict[str, Any]:
        """Test AsyncDatabase functional compliance"""
        test_results = []

        # Test 1: Health check
        try:
            health = await db.health_check()
            test_results.append({
                "test": "health_check",
                "status": "pass" if isinstance(health, bool) else "fail",
                "result": f"Health status: {health}" if isinstance(health, bool) else "Invalid return type"
            })
        except Exception as e:
            test_results.append({
                "test": "health_check",
                "status": "error",
                "error": str(e)
            })

        # Test 2: Get pool stats
        try:
            stats = await db.get_pool_stats()
            test_results.append({
                "test": "get_pool_stats",
                "status": "pass" if isinstance(stats, dict) else "fail",
                "result": "Returns dictionary" if isinstance(stats, dict) else "Invalid return type"
            })
        except Exception as e:
            test_results.append({
                "test": "get_pool_stats",
                "status": "error",
                "error": str(e)
            })

        passed_tests = sum(1 for test in test_results if test["status"] == "pass")
        total_tests = len(test_results)

        return {
            "status": "completed",
            "protocol": "AsyncDatabaseInterface",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "test_results": test_results
        }

    async def _test_task_orchestrator_functions(self, orchestrator: TaskOrchestratorInterface) -> Dict[str, Any]:
        """Test TaskOrchestrator functional compliance"""
        test_results = []

        # Test 1: Get statistics
        try:
            stats = await orchestrator.get_task_statistics()
            test_results.append({
                "test": "get_task_statistics",
                "status": "pass" if isinstance(stats, dict) else "fail",
                "result": "Returns dictionary" if isinstance(stats, dict) else "Invalid return type"
            })
        except Exception as e:
            test_results.append({
                "test": "get_task_statistics",
                "status": "error",
                "error": str(e)
            })

        passed_tests = sum(1 for test in test_results if test["status"] == "pass")
        total_tests = len(test_results)

        return {
            "status": "completed",
            "protocol": "TaskOrchestratorInterface",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "test_results": test_results
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        if not self.validation_results:
            return {"error": "No validation results available"}

        total_validations = len(self.validation_results)
        compliant_implementations = sum(1 for result in self.validation_results.values()
                                      if result["is_compliant"])

        avg_compliance_score = sum(result["compliance_score"] for result in self.validation_results.values()) / total_validations

        report = {
            "summary": {
                "total_validations": total_validations,
                "compliant_implementations": compliant_implementations,
                "compliance_rate": (compliant_implementations / total_validations * 100) if total_validations > 0 else 0,
                "average_compliance_score": avg_compliance_score,
                "generated_at": datetime.now().isoformat()
            },
            "detailed_results": self.validation_results,
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        for validation_id, result in self.validation_results.items():
            if not result["is_compliant"]:
                impl_name = result["implementation"]
                protocol_name = result["protocol"]
                missing_methods = result["missing_methods"]

                recommendations.append(
                    f"🔧 {impl_name}: Implement missing methods for {protocol_name}: {', '.join(missing_methods)}"
                )

            if result["compliance_score"] < 80:
                recommendations.append(
                    f"⚠️ {result['implementation']}: Low compliance score ({result['compliance_score']:.1f}%) - review protocol implementation"
                )

        if not recommendations:
            recommendations.append("✅ All implementations are protocol compliant!")

        return recommendations

# Example usage and testing functions
async def validate_echo_brain_protocols():
    """Validate all Echo Brain protocol implementations"""
    validator = ProtocolValidator()

    logger.info("🔍 Starting Echo Brain protocol validation...")

    # Import implementations for validation
    try:
        from ..echo_vector_memory import VectorMemory
        from ..db.async_database import AsyncEchoDatabase

        # Validate VectorMemory
        memory = VectorMemory()
        memory_validation = validator.validate_class_protocol_compliance(memory, VectorMemoryInterface)
        memory_functional = await validator.validate_functional_compliance(memory, VectorMemoryInterface)

        logger.info(f"VectorMemory validation: {memory_validation['compliance_score']:.1f}% compliant")

        # Validate AsyncEchoDatabase
        db = AsyncEchoDatabase()
        db_validation = validator.validate_class_protocol_compliance(db, AsyncDatabaseInterface)
        db_functional = await validator.validate_functional_compliance(db, AsyncDatabaseInterface)

        logger.info(f"AsyncEchoDatabase validation: {db_validation['compliance_score']:.1f}% compliant")

        # Generate report
        report = validator.generate_compliance_report()

        logger.info("📊 Protocol Validation Complete")
        logger.info(f"Overall compliance rate: {report['summary']['compliance_rate']:.1f}%")

        return report

    except ImportError as e:
        logger.error(f"Failed to import implementations for validation: {e}")
        return {"error": "Import failed", "details": str(e)}
    except Exception as e:
        logger.error(f"Protocol validation failed: {e}")
        return {"error": "Validation failed", "details": str(e)}

if __name__ == "__main__":
    # Run validation
    async def main():
        report = await validate_echo_brain_protocols()
        print(f"\n=== ECHO BRAIN PROTOCOL VALIDATION REPORT ===")
        print(f"Compliance Rate: {report['summary']['compliance_rate']:.1f}%")
        print(f"Average Score: {report['summary']['average_compliance_score']:.1f}%")
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")

    asyncio.run(main())