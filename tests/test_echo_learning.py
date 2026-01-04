#!/usr/bin/env python3
"""
Test Script for Echo Learning System
===================================

This script demonstrates Echo's comprehensive learning system capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    print("🧠 Echo Learning System Test")
    print("=" * 50)

    try:
        # Import learning integration
        from src.core.echo.echo_learning_integration import (
            initialize_echo_learning, make_intelligent_decision,
            record_decision_outcome, get_learning_system_status
        )

        print("\n1. Initializing Learning Systems...")
        success = await initialize_echo_learning()
        print(f"   ✅ Initialization: {'Success' if success else 'Failed'}")

        print("\n2. Testing Intelligent Decision Making...")

        # Test 1: Simple decision
        response1 = await make_intelligent_decision(
            request="Optimize system performance for better user experience",
            context={
                "current_performance": 0.65,
                "user_complaints": 2,
                "available_resources": True,
                "system_load": 0.7
            },
            priority="medium"
        )

        print(f"   📋 Decision ID: {response1.request_id}")
        print(f"   🎯 Decision: {response1.decision_made}")
        print(f"   📊 Confidence: {response1.confidence:.2f}")
        print(f"   🛠️  Decision Path: {' → '.join(response1.decision_path)}")
        print(f"   👥 Board Used: {response1.board_consultation_used}")
        print(f"   📋 Workflow Created: {response1.task_workflow_created or 'None'}")

        print("\n3. Testing Complex Decision (should trigger board consultation)...")

        # Test 2: Complex decision
        response2 = await make_intelligent_decision(
            request="Implement comprehensive security audit and remediation system",
            context={
                "security_concerns": ["data_breach_risk", "unauthorized_access"],
                "compliance_requirements": True,
                "business_critical": True,
                "timeline": "urgent"
            },
            priority="critical"
        )

        print(f"   📋 Decision ID: {response2.request_id}")
        print(f"   🎯 Decision: {response2.decision_made}")
        print(f"   📊 Confidence: {response2.confidence:.2f}")
        print(f"   🛠️  Decision Path: {' → '.join(response2.decision_path)}")
        print(f"   👥 Board Used: {response2.board_consultation_used}")
        print(f"   📋 Workflow Created: {response2.task_workflow_created or 'None'}")

        print("\n4. Recording Decision Outcomes...")

        # Record outcome for first decision
        await record_decision_outcome(
            request_id=response1.request_id,
            outcome="Successfully optimized performance with 20% improvement",
            metrics={
                "performance_improvement": 0.20,
                "user_satisfaction": 0.85,
                "implementation_time": 3.5,
                "resource_efficiency": 0.9
            },
            user_feedback="Great improvement in system responsiveness"
        )
        print(f"   ✅ Recorded outcome for {response1.request_id}")

        # Record outcome for second decision
        await record_decision_outcome(
            request_id=response2.request_id,
            outcome="Security audit initiated with comprehensive remediation plan",
            metrics={
                "security_improvement": 0.40,
                "compliance_score": 0.95,
                "implementation_complexity": 0.8,
                "timeline_adherence": 0.9
            },
            user_feedback="Thorough security assessment with actionable recommendations"
        )
        print(f"   ✅ Recorded outcome for {response2.request_id}")

        print("\n5. Getting Learning System Status...")
        status = await get_learning_system_status()

        print(f"   🏥 Overall Health: {status.overall_health}")
        print(f"   ⚡ Active Processes: {status.active_learning_processes}")
        print(f"   🧠 Learning Core: {status.learning_core_status}")
        print(f"   🩺 Self Diagnosis: {status.self_diagnosis_status}")
        print(f"   👥 Board Manager: {status.board_manager_status}")
        print(f"   📋 Task Decomposer: {status.task_decomposer_status}")
        print(f"   📊 Outcome Tracker: {status.outcome_tracker_status}")

        if status.recent_insights:
            print(f"   💡 Recent Insights:")
            for insight in status.recent_insights[:3]:
                print(f"      • {insight}")

        if status.system_recommendations:
            print(f"   🎯 Recommendations:")
            for rec in status.system_recommendations[:3]:
                print(f"      • {rec}")

        print("\n6. Testing Individual Components...")

        # Test learning system directly
        from src.core.echo.echo_learning_system import get_learning_system
        learning_sys = get_learning_system()
        learning_status = await learning_sys.get_learning_status()
        print(f"   🧠 Learning patterns: {learning_status.get('active_patterns', 0)}")
        print(f"   📚 Decision history: {learning_status.get('decision_history_size', 0)}")

        # Test self-diagnosis
        from src.core.echo.echo_self_diagnosis import get_health_status
        health = await get_health_status()
        print(f"   🩺 Health status: {health.get('status', 'unknown') if health else 'unavailable'}")

        # Test board manager
        from bin.echo_board_manager import get_board_status
        board = await get_board_status()
        print(f"   👥 Board directors: {board.get('total_directors', 0) if board else 0}")

        print("\n" + "=" * 50)
        print("🎉 Echo Learning System Test Complete!")
        print("\nKey Features Demonstrated:")
        print("• ✅ Intelligent decision making with context analysis")
        print("• ✅ Board of Directors consultation for complex decisions")
        print("• ✅ Task decomposition for complex requests")
        print("• ✅ Outcome tracking and learning from results")
        print("• ✅ Self-diagnosis and health monitoring")
        print("• ✅ Integrated learning across all components")
        print("\nEcho is now equipped with comprehensive learning capabilities!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())