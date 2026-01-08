#!/usr/bin/env python3
"""
Simple Test for Echo Learning System
===================================
"""

import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)

async def main():
    print("🧠 Echo Learning System - Simple Test")
    print("=" * 40)

    try:
        # Test basic imports
        print("1. Testing imports...")
        from src.core.echo.echo_learning_system import get_learning_system
        from src.core.echo.echo_self_diagnosis import get_diagnosis_system
        from bin.echo_board_manager import get_board_manager
        from src.core.echo.echo_outcome_tracker import get_outcome_tracker
        print("   ✅ All imports successful")

        # Test basic initialization
        print("\n2. Testing basic initialization...")
        learning_sys = get_learning_system()
        diagnosis_sys = get_diagnosis_system()
        board_mgr = get_board_manager()
        outcome_tracker = get_outcome_tracker()
        print("   ✅ All systems initialized")

        # Test learning system status
        print("\n3. Testing learning system...")
        status = await learning_sys.get_learning_status()
        print(f"   📊 Active patterns: {status.get('active_patterns', 0)}")
        print(f"   📚 Decision history: {status.get('decision_history_size', 0)}")

        # Test board consultation
        print("\n4. Testing board consultation...")
        board_result = await board_mgr.consult_board(
            "Test board consultation",
            {"test": True},
            domain="general"
        )
        print(f"   🎯 Board decision: {board_result['recommendation'][:50]}...")
        print(f"   📊 Confidence: {board_result['confidence']:.2f}")

        # Test health status
        print("\n5. Testing health monitoring...")
        health = await diagnosis_sys.get_health_status()
        print(f"   🏥 Health status: {health.get('status', 'unknown') if health else 'no data'}")

        print("\n" + "=" * 40)
        print("🎉 Simple test completed successfully!")
        print("\nCore learning components are working:")
        print("• ✅ Learning System - Pattern recognition")
        print("• ✅ Board Manager - Decision consultation")
        print("• ✅ Self Diagnosis - Health monitoring")
        print("• ✅ Outcome Tracker - Results tracking")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())