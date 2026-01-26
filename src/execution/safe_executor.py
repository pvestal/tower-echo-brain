#!/usr/bin/env python3
"""
Safe SQL Executor for Echo Brain - Remote Installation
"""

import asyncio
import os
import psycopg2
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

class SafeSQLExecutor:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.connection = None

    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            print("✅ Connected to PostgreSQL")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def execute_with_proof(self, query: str) -> Dict[str, Any]:
        """Execute SQL with proof generation"""
        if not self.connection:
            return {"success": False, "error": "No database connection"}

        try:
            cursor = self.connection.cursor()
            start_time = datetime.now()

            cursor.execute(query)

            # Try to fetch results (for SELECT queries)
            try:
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            except:
                results = []
                column_names = []

            self.connection.commit()

            execution_time = (datetime.now() - start_time).total_seconds()

            proof = f"✅ REAL EXECUTION PROOF:\n"
            proof += f"Query: {query}\n"
            proof += f"Results: {len(results)} rows\n"
            proof += f"Columns: {column_names}\n"
            proof += f"Sample data: {results[:2] if results else 'No data'}\n"
            proof += f"Execution time: {execution_time:.3f}s"

            return {
                "success": True,
                "results": results,
                "columns": column_names,
                "proof": proof,
                "execution_time": execution_time
            }

        except Exception as e:
            self.connection.rollback()
            return {
                "success": False,
                "error": str(e),
                "proof": f"❌ EXECUTION FAILED: {str(e)}"
            }

    def test_autonomous_learning(self):
        """Test Echo's first real autonomous database operation"""
        print("🧠 Testing Autonomous Learning Capability...")

        # Create learned_preferences table if it doesn't exist
        create_table = """
        CREATE TABLE IF NOT EXISTS learned_preferences (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50),
            preference_type VARCHAR(100),
            value TEXT,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """

        result1 = self.execute_with_proof(create_table)
        print(f"📋 Table creation: {result1}")

        # Insert a real learning example
        learn_query = """
        INSERT INTO learned_preferences (user_id, preference_type, value, confidence)
        VALUES ('patrick', 'analysis_style', 'expert_detailed', 0.95)
        ON CONFLICT DO NOTHING
        RETURNING *
        """

        result2 = self.execute_with_proof(learn_query)
        print(f"🎯 Learning execution: {result2}")

        # Verify the learning
        verify_query = """
        SELECT * FROM learned_preferences
        WHERE user_id = 'patrick' AND preference_type = 'analysis_style'
        """

        result3 = self.execute_with_proof(verify_query)
        print(f"✅ Learning verification: {result3}")

        return result1, result2, result3

# Test function
def test_real_execution():
    config = {
        'host': '192.168.50.135',
        'database': 'echo_brain',
        'user': 'patrick',
        'password': os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
    }

    executor = SafeSQLExecutor(config)

    if executor.connect():
        print("🎯 TESTING ECHO'S FIRST REAL AUTONOMOUS CAPABILITY...")
        return executor.test_autonomous_learning()
    else:
        print("❌ Failed to connect to database")
        return None

if __name__ == "__main__":
    test_real_execution()
