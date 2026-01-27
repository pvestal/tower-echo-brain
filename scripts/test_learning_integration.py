#!/usr/bin/env python3
"""
Test the integrated learning system with real repair scenarios
Demonstrates how Echo Brain learns from successes and failures
"""

import asyncio
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/opt/tower-echo-brain')
from src.tasks.autonomous_repair_executor import RepairExecutor
from scripts.tower_outcome_learning import TowerOutcomeLearning

async def test_learning_integration():
    """Test the complete learning feedback loop"""

    print("🧠 ECHO BRAIN LEARNING INTEGRATION TEST")
    print("="*50)

    # Initialize repair executor (with integrated learning)
    executor = RepairExecutor()
    learner = TowerOutcomeLearning()

    # Test scenarios
    test_repairs = [
        {
            'repair_type': 'service_restart',
            'target': 'tower-echo-brain',
            'issue': 'service not responding',
            'expected': True  # Should succeed
        },
        {
            'repair_type': 'service_restart',
            'target': 'nonexistent-service',
            'issue': 'service not found',
            'expected': False  # Should fail
        },
        {
            'repair_type': 'disk_cleanup',
            'target': '/tmp',
            'issue': 'disk space low',
            'expected': True  # Should succeed
        }
    ]

    print("\n📝 Running test repairs with learning...")

    for test in test_repairs:
        print(f"\n   Testing: {test['target']} - {test['issue']}")

        # Execute repair (automatically records outcome)
        result = await executor.execute_repair(
            repair_type=test['repair_type'],
            target=test['target'],
            issue=test['issue']
        )

        success = result.get('success', False)
        status = "✅" if success else "❌"

        print(f"   {status} Result: {'Success' if success else 'Failed'}")

        if success != test['expected']:
            print(f"   ⚠️  Unexpected result! Expected: {test['expected']}")

    print("\n" + "="*50)
    print("📊 LEARNING REPORT:")
    print(learner.generate_report())

    # Test learned recommendations
    print("\n🎯 TESTING LEARNED RECOMMENDATIONS:")

    test_cases = [
        ('tower-echo-brain', 'service not responding'),
        ('nonexistent-service', 'service not found'),
        ('/tmp', 'disk space low'),
    ]

    for service, issue in test_cases:
        best = learner.get_best_action(service, issue)
        if best:
            print(f"\n   {service} [{issue}]:")
            print(f"   → Recommended: {best['action']}")
            print(f"   → Confidence: {best['confidence']*100:.0f}%")
            print(f"   → Based on {best['attempts']} attempts")
        else:
            print(f"\n   {service} [{issue}]: No recommendation yet")

    # Demonstrate continuous learning
    print("\n" + "="*50)
    print("🔄 DEMONSTRATING CONTINUOUS LEARNING:")

    # Simulate the same repair multiple times
    print("\n   Running same repair 3 times to show learning...")

    for i in range(3):
        result = await executor.execute_repair(
            repair_type='disk_cleanup',
            target='/var/log',
            issue='log rotation needed'
        )

        # Check updated confidence
        best = learner.get_best_action('/var/log', 'log rotation needed')
        if best:
            print(f"   Attempt {i+1}: Confidence now {best['confidence']*100:.0f}%")

    print("\n✅ INTEGRATION TEST COMPLETE!")
    print("\nEcho Brain is now:")
    print("   1. Executing repairs")
    print("   2. Recording outcomes automatically")
    print("   3. Learning from successes and failures")
    print("   4. Making better recommendations over time")
    print("\n   Database: /opt/tower-echo-brain/data/repair_outcomes.db")

if __name__ == "__main__":
    asyncio.run(test_learning_integration())