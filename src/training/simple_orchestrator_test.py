#!/usr/bin/env python3
"""
Simple test for the unified orchestrator to verify core functionality
"""

import os
import sys
import logging
import psycopg2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database config
test_db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "database": "echo_brain",
    "user": "patrick",
    "password": os.environ.get("DB_PASSWORD", "tower_echo_brain_secret_key_2025")
}

def test_database_setup():
    """Test that orchestrator database tables exist"""
    try:
        conn = psycopg2.connect(**test_db_config)
        cursor = conn.cursor()

        logger.info("🧪 Testing Orchestrator Database Setup")

        # Check required tables
        required_tables = [
            'orchestration_checkpoints',
            'orchestration_tasks',
            'orchestration_worker_metrics',
            'orchestration_resources',
            'orchestration_analytics',
            'orchestration_errors'
        ]

        for table in required_tables:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                );
            """, (table,))

            exists = cursor.fetchone()[0]
            if exists:
                logger.info(f"✅ Table '{table}' exists")
            else:
                logger.error(f"❌ Table '{table}' missing")
                return False

        # Test we can insert data
        cursor.execute("""
            INSERT INTO orchestration_analytics (session_id, metric_name, metric_value)
            VALUES (gen_random_uuid(), 'test_metric', 1.0)
        """)

        cursor.execute("SELECT COUNT(*) FROM orchestration_analytics WHERE metric_name = 'test_metric'")
        count = cursor.fetchone()[0]

        if count > 0:
            logger.info("✅ Database write test successful")
        else:
            logger.error("❌ Database write test failed")
            return False

        # Cleanup test data
        cursor.execute("DELETE FROM orchestration_analytics WHERE metric_name = 'test_metric'")

        conn.commit()
        cursor.close()
        conn.close()

        logger.info("✅ Database setup test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

def test_orchestrator_import():
    """Test that the orchestrator can be imported"""
    try:
        # Add path for imports
        sys.path.append('/opt/tower-echo-brain/src')

        logger.info("🧪 Testing Orchestrator Import")

        from training.unified_orchestrator import (
            UnifiedTrainingOrchestrator,
            WorkerType,
            TaskPriority,
            PipelineStage
        )

        logger.info("✅ Core orchestrator classes imported successfully")

        # Test enum values
        worker_types = [wt.value for wt in WorkerType]
        priorities = [p.value for p in TaskPriority]
        stages = [s.value for s in PipelineStage]

        logger.info(f"✅ Worker types: {len(worker_types)} defined")
        logger.info(f"✅ Task priorities: {len(priorities)} defined")
        logger.info(f"✅ Pipeline stages: {len(stages)} defined")

        return True

    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_orchestrator_initialization():
    """Test orchestrator can be initialized"""
    try:
        sys.path.append('/opt/tower-echo-brain/src')

        from training.unified_orchestrator import UnifiedTrainingOrchestrator

        logger.info("🧪 Testing Orchestrator Initialization")

        # Simple test config
        test_config = {
            "max_concurrent_workers": 1,
            "enable_gpu_acceleration": False,
            "enable_deduplication": False,
            "enable_safety_checks": False,
            "websocket_port": 8313,  # Unique port for testing
            "redis_url": "redis://localhost:6379/3"  # Different DB
        }

        # Initialize orchestrator
        orchestrator = UnifiedTrainingOrchestrator(config=test_config)

        if orchestrator.session_id:
            logger.info(f"✅ Orchestrator initialized with session: {orchestrator.session_id}")
            logger.info(f"✅ Workers available: {len(orchestrator.workers)}")
            return True
        else:
            logger.error("❌ Orchestrator session ID not set")
            return False

    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 UNIFIED ORCHESTRATOR SIMPLE TEST SUITE")
    logger.info("=" * 60)

    tests = [
        ("Database Setup", test_database_setup),
        ("Orchestrator Import", test_orchestrator_import),
        ("Orchestrator Initialization", test_orchestrator_initialization)
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running: {test_name}")
        logger.info("-" * 40)

        try:
            result = test_func()
            results.append((test_name, result))

            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")

        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n📋 SIMPLE TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\n🏁 RESULTS: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Core orchestrator functionality working!")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed - Check configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)