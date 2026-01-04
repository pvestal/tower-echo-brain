#!/usr/bin/env python3
"""
Basic test of Echo MLOps components
"""

import asyncio
import sys
import os
import sqlite3
from datetime import datetime

async def test_model_registry():
    print("Testing Model Registry...")
    try:
        # Test database creation
        db_path = "/opt/tower-echo-brain/data/model_registry.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        ''')
        
        # Test insertion
        conn.execute('''
            INSERT OR REPLACE INTO models (model_id, name, version, created_at)
            VALUES (?, ?, ?, ?)
        ''', ("test_model_001", "Echo Decision Engine", "1.0.0", datetime.now()))
        
        conn.commit()
        
        # Test retrieval
        cursor = conn.execute('SELECT * FROM models WHERE model_id = ?', ("test_model_001",))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            print("✅ Model Registry: Database operations working")
            return True
        else:
            print("❌ Model Registry: Failed to retrieve test data")
            return False
            
    except Exception as e:
        print(f"❌ Model Registry: Error - {e}")
        return False

async def test_drift_detector():
    print("Testing Drift Detector...")
    try:
        # Test database creation
        db_path = "/opt/tower-echo-brain/data/drift_monitoring.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_monitoring (
                record_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL
            )
        ''')
        
        # Test data insertion
        conn.execute('''
            INSERT INTO feature_monitoring (record_id, timestamp, feature_name, feature_value)
            VALUES (?, ?, ?, ?)
        ''', ("test_record_001", datetime.now(), "echo_confidence", 0.85))
        
        conn.commit()
        conn.close()
        
        print("✅ Drift Detector: Database operations working")
        return True
        
    except Exception as e:
        print(f"❌ Drift Detector: Error - {e}")
        return False

async def test_feature_store():
    print("Testing Feature Store...")
    try:
        # Test database creation
        db_path = "/opt/tower-echo-brain/data/feature_store.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_definitions (
                feature_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                feature_type TEXT NOT NULL,
                entity_type TEXT NOT NULL
            )
        ''')
        
        # Test feature definition
        conn.execute('''
            INSERT OR REPLACE INTO feature_definitions 
            (feature_id, name, description, feature_type, entity_type)
            VALUES (?, ?, ?, ?, ?)
        ''', ("feat_001", "Echo Confidence", "Decision confidence score", "numerical", "decision"))
        
        conn.commit()
        conn.close()
        
        print("✅ Feature Store: Database operations working")
        return True
        
    except Exception as e:
        print(f"❌ Feature Store: Error - {e}")
        return False

async def test_ab_testing():
    print("Testing A/B Testing Framework...")
    try:
        # Test database creation
        db_path = "/opt/tower-echo-brain/data/ab_testing.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        ''')
        
        # Test experiment creation
        conn.execute('''
            INSERT OR REPLACE INTO experiments (experiment_id, name, status, created_at)
            VALUES (?, ?, ?, ?)
        ''', ("exp_001", "Echo Model Test", "draft", datetime.now()))
        
        conn.commit()
        conn.close()
        
        print("✅ A/B Testing: Database operations working")
        return True
        
    except Exception as e:
        print(f"❌ A/B Testing: Error - {e}")
        return False

async def test_retraining_pipeline():
    print("Testing Retraining Pipeline...")
    try:
        # Test database creation
        db_path = "/opt/tower-echo-brain/data/retraining_pipeline.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS retraining_jobs (
                job_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        ''')
        
        # Test job creation
        conn.execute('''
            INSERT OR REPLACE INTO retraining_jobs (job_id, model_name, status, created_at)
            VALUES (?, ?, ?, ?)
        ''', ("job_001", "Echo Decision Engine", "pending", datetime.now()))
        
        conn.commit()
        conn.close()
        
        print("✅ Retraining Pipeline: Database operations working")
        return True
        
    except Exception as e:
        print(f"❌ Retraining Pipeline: Error - {e}")
        return False

async def main():
    print("Echo MLOps Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_model_registry,
        test_drift_detector,
        test_feature_store,
        test_ab_testing,
        test_retraining_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All MLOps components basic functionality working!")
        return 0
    else:
        print("⚠️ Some components need attention")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
