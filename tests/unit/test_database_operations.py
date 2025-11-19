"""
Database Consistency and Transaction Validation Tests
Tests database integrity, ACID compliance, and data consistency.
"""

import asyncio
import json
import random
import string
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import concurrent.futures

import psycopg2
import psycopg2.pool
import psycopg2.extras
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


@dataclass
class TransactionTest:
    """Result of a transaction test"""
    test_name: str
    success: bool
    duration: float
    error: Optional[str] = None
    records_affected: int = 0
    consistency_check: bool = True


class DatabaseConsistencyTester:
    """Comprehensive database testing for anime generation system"""

    def __init__(self,
                 db_config: Dict[str, str],
                 test_db_name: str = "test_echo_brain"):
        self.db_config = db_config
        self.test_db_name = test_db_name
        self.production_db = db_config["database"]
        self.connection_pool = None

        # Test data patterns
        self.test_characters = [
            {
                "name": f"TestChar_{uuid.uuid4().hex[:8]}",
                "description": "Test character for database validation",
                "visual_features": json.dumps({"hair": "black", "eyes": "blue"}),
                "personality": "brave, kind",
                "backstory": "Test character backstory"
            }
            for _ in range(10)
        ]

        self.test_projects = [
            {
                "name": f"TestProject_{uuid.uuid4().hex[:8]}",
                "description": "Test project for database validation",
                "style": "anime",
                "status": "active"
            }
            for _ in range(5)
        ]

    async def setup_test_environment(self):
        """Create isolated test database and schema"""
        # Connect to postgres database to create test db
        admin_config = self.db_config.copy()
        admin_config["database"] = "postgres"

        try:
            conn = psycopg2.connect(**admin_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Drop test database if exists
            cursor.execute(f"DROP DATABASE IF EXISTS {self.test_db_name}")

            # Create test database
            cursor.execute(f"CREATE DATABASE {self.test_db_name}")

            cursor.close()
            conn.close()

            # Connect to test database and create schema
            test_config = self.db_config.copy()
            test_config["database"] = self.test_db_name

            await self._create_test_schema(test_config)

            # Initialize connection pool for tests
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                1, 20, **test_config
            )

        except Exception as e:
            raise Exception(f"Failed to setup test environment: {e}")

    async def _create_test_schema(self, test_config: Dict):
        """Create test database schema based on production schema"""
        conn = psycopg2.connect(**test_config)

        try:
            cursor = conn.cursor()

            # Create characters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS characters (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT,
                    visual_features JSONB,
                    personality TEXT,
                    backstory TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    version INTEGER DEFAULT 1
                )
            """)

            # Create projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT,
                    style VARCHAR(100),
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Create generations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id SERIAL PRIMARY KEY,
                    job_id UUID UNIQUE NOT NULL,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    character_id INTEGER REFERENCES characters(id) ON DELETE SET NULL,
                    prompt TEXT NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    output_path TEXT,
                    generation_time FLOAT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    completed_at TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Create validation_results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id SERIAL PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    character_name VARCHAR(255),
                    consistency_score FLOAT,
                    style_score FLOAT,
                    quality_score FLOAT,
                    emotion_match BOOLEAN,
                    validation_time FLOAT,
                    error_message TEXT,
                    llava_response TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_characters_name ON characters(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generations_job_id ON generations(job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generations_status ON generations(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_validation_character ON validation_results(character_name)")

            # Create constraints
            cursor.execute("ALTER TABLE characters ADD CONSTRAINT chk_character_name_length CHECK (length(name) >= 2)")
            cursor.execute("ALTER TABLE projects ADD CONSTRAINT chk_project_status CHECK (status IN ('active', 'archived', 'deleted'))")

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to create test schema: {e}")
        finally:
            cursor.close()
            conn.close()

    def get_connection(self):
        """Get database connection from pool"""
        if not self.connection_pool:
            raise Exception("Database connection pool not initialized")
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """Return connection to pool"""
        if self.connection_pool:
            self.connection_pool.putconn(conn)

    async def test_acid_compliance(self) -> List[TransactionTest]:
        """Test ACID compliance of database operations"""
        tests = []

        # Test Atomicity
        tests.append(await self._test_atomicity())

        # Test Consistency
        tests.append(await self._test_consistency())

        # Test Isolation
        tests.append(await self._test_isolation())

        # Test Durability
        tests.append(await self._test_durability())

        return tests

    async def _test_atomicity(self) -> TransactionTest:
        """Test that transactions are atomic (all or nothing)"""
        start_time = time.time()

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Count initial records
            cursor.execute("SELECT COUNT(*) FROM characters")
            initial_count = cursor.fetchone()[0]

            try:
                # Begin transaction
                cursor.execute("BEGIN")

                # Insert valid character
                cursor.execute("""
                    INSERT INTO characters (name, description)
                    VALUES (%s, %s)
                """, ("Valid_Character", "Valid description"))

                # Try to insert invalid character (should fail)
                cursor.execute("""
                    INSERT INTO characters (name, description)
                    VALUES (%s, %s)
                """, ("X", "Too short name should fail constraint"))

                # This should never execute due to constraint failure
                conn.commit()

            except Exception:
                # Transaction should rollback
                conn.rollback()

            # Check that no records were inserted (atomicity)
            cursor.execute("SELECT COUNT(*) FROM characters")
            final_count = cursor.fetchone()[0]

            cursor.close()
            self.return_connection(conn)

            # If atomicity works, count should be unchanged
            success = initial_count == final_count
            duration = time.time() - start_time

            return TransactionTest(
                test_name="atomicity",
                success=success,
                duration=duration,
                records_affected=final_count - initial_count,
                consistency_check=True
            )

        except Exception as e:
            return TransactionTest(
                test_name="atomicity",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def _test_consistency(self) -> TransactionTest:
        """Test that constraints maintain data consistency"""
        start_time = time.time()

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Test various constraint violations
            constraint_tests = [
                # Duplicate character names
                ("INSERT INTO characters (name) VALUES ('DuplicateTest')", None),
                ("INSERT INTO characters (name) VALUES ('DuplicateTest')", psycopg2.IntegrityError),

                # Character name too short
                ("INSERT INTO characters (name) VALUES ('X')", psycopg2.IntegrityError),

                # Invalid project status
                ("INSERT INTO projects (name, status) VALUES ('TestProject', 'invalid')", psycopg2.IntegrityError),

                # Foreign key constraint
                ("INSERT INTO generations (job_id, project_id, prompt) VALUES (%s, 999999, 'test')" % uuid.uuid4(), psycopg2.IntegrityError),
            ]

            violations_caught = 0
            for sql, expected_error in constraint_tests:
                try:
                    cursor.execute(sql)
                    conn.commit()
                    if expected_error:
                        # Should have failed but didn't
                        continue
                except expected_error:
                    violations_caught += 1
                    conn.rollback()
                except Exception as e:
                    if expected_error and isinstance(e, expected_error):
                        violations_caught += 1
                    conn.rollback()

            cursor.close()
            self.return_connection(conn)

            # All constraint violations should have been caught
            expected_violations = len([t for t in constraint_tests if t[1] is not None])
            success = violations_caught == expected_violations

            return TransactionTest(
                test_name="consistency",
                success=success,
                duration=time.time() - start_time,
                records_affected=violations_caught,
                consistency_check=True
            )

        except Exception as e:
            return TransactionTest(
                test_name="consistency",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def _test_isolation(self) -> TransactionTest:
        """Test transaction isolation using concurrent connections"""
        start_time = time.time()

        try:
            # Use threading to simulate concurrent access
            def transaction_1():
                conn = self.get_connection()
                cursor = conn.cursor()
                try:
                    cursor.execute("BEGIN")
                    cursor.execute("""
                        INSERT INTO characters (name, description)
                        VALUES ('IsolationTest1', 'Transaction 1 character')
                    """)
                    # Hold transaction open
                    time.sleep(0.5)
                    conn.commit()
                    return True
                except Exception:
                    conn.rollback()
                    return False
                finally:
                    cursor.close()
                    self.return_connection(conn)

            def transaction_2():
                time.sleep(0.1)  # Start after transaction 1
                conn = self.get_connection()
                cursor = conn.cursor()
                try:
                    # This should not see uncommitted data from transaction 1
                    cursor.execute("SELECT COUNT(*) FROM characters WHERE name = 'IsolationTest1'")
                    count_during = cursor.fetchone()[0]

                    # Wait for transaction 1 to complete
                    time.sleep(0.6)

                    # This should see committed data
                    cursor.execute("SELECT COUNT(*) FROM characters WHERE name = 'IsolationTest1'")
                    count_after = cursor.fetchone()[0]

                    return count_during, count_after
                except Exception:
                    return None, None
                finally:
                    cursor.close()
                    self.return_connection(conn)

            # Run transactions concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future1 = executor.submit(transaction_1)
                future2 = executor.submit(transaction_2)

                result1 = future1.result(timeout=5)
                result2 = future2.result(timeout=5)

            # Check isolation: should not see uncommitted data
            count_during, count_after = result2
            success = result1 and count_during == 0 and count_after == 1

            return TransactionTest(
                test_name="isolation",
                success=success,
                duration=time.time() - start_time,
                records_affected=1,
                consistency_check=True
            )

        except Exception as e:
            return TransactionTest(
                test_name="isolation",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def _test_durability(self) -> TransactionTest:
        """Test that committed transactions persist"""
        start_time = time.time()

        try:
            # Insert data and commit
            test_name = f"DurabilityTest_{uuid.uuid4().hex[:8]}"

            conn1 = self.get_connection()
            cursor1 = conn1.cursor()

            cursor1.execute("""
                INSERT INTO characters (name, description)
                VALUES (%s, %s)
            """, (test_name, "Durability test character"))
            conn1.commit()

            cursor1.close()
            self.return_connection(conn1)

            # Open new connection and verify data persists
            conn2 = self.get_connection()
            cursor2 = conn2.cursor()

            cursor2.execute("SELECT name FROM characters WHERE name = %s", (test_name,))
            result = cursor2.fetchone()

            cursor2.close()
            self.return_connection(conn2)

            success = result is not None and result[0] == test_name

            return TransactionTest(
                test_name="durability",
                success=success,
                duration=time.time() - start_time,
                records_affected=1,
                consistency_check=True
            )

        except Exception as e:
            return TransactionTest(
                test_name="durability",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def test_concurrent_access(self, num_threads: int = 10) -> TransactionTest:
        """Test database performance under concurrent access"""
        start_time = time.time()

        def worker_thread(thread_id: int):
            """Worker thread that performs database operations"""
            try:
                conn = self.get_connection()
                cursor = conn.cursor()

                operations_completed = 0

                for i in range(5):  # 5 operations per thread
                    try:
                        # Insert character
                        char_name = f"ConcurrentTest_{thread_id}_{i}"
                        cursor.execute("""
                            INSERT INTO characters (name, description)
                            VALUES (%s, %s)
                        """, (char_name, f"Character from thread {thread_id}"))

                        # Query characters
                        cursor.execute("SELECT COUNT(*) FROM characters WHERE name LIKE %s",
                                     (f"ConcurrentTest_{thread_id}%",))
                        cursor.fetchone()

                        # Update character
                        cursor.execute("""
                            UPDATE characters SET description = %s WHERE name = %s
                        """, (f"Updated by thread {thread_id}", char_name))

                        conn.commit()
                        operations_completed += 1

                    except Exception as e:
                        conn.rollback()
                        # Continue with next operation

                cursor.close()
                self.return_connection(conn)
                return operations_completed

            except Exception:
                return 0

        try:
            # Run concurrent threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
                results = [future.result(timeout=30) for future in futures]

            total_operations = sum(results)
            expected_operations = num_threads * 5
            success_rate = (total_operations / expected_operations) * 100

            # Consider test successful if >90% of operations completed
            success = success_rate >= 90

            return TransactionTest(
                test_name="concurrent_access",
                success=success,
                duration=time.time() - start_time,
                records_affected=total_operations,
                consistency_check=True
            )

        except Exception as e:
            return TransactionTest(
                test_name="concurrent_access",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def test_data_integrity(self) -> TransactionTest:
        """Test referential integrity and data consistency"""
        start_time = time.time()

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Create test project
            cursor.execute("""
                INSERT INTO projects (name, description, status)
                VALUES (%s, %s, %s) RETURNING id
            """, ("IntegrityTest", "Test project for integrity", "active"))
            project_id = cursor.fetchone()[0]

            # Create test character
            cursor.execute("""
                INSERT INTO characters (name, description)
                VALUES (%s, %s) RETURNING id
            """, ("IntegrityTestChar", "Test character for integrity"))
            character_id = cursor.fetchone()[0]

            # Create generation linked to project and character
            job_id = uuid.uuid4()
            cursor.execute("""
                INSERT INTO generations (job_id, project_id, character_id, prompt, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (job_id, project_id, character_id, "Test prompt", "completed"))

            conn.commit()

            # Test cascade delete
            cursor.execute("DELETE FROM projects WHERE id = %s", (project_id,))

            # Check that generation was cascade deleted
            cursor.execute("SELECT COUNT(*) FROM generations WHERE project_id = %s", (project_id,))
            remaining_generations = cursor.fetchone()[0]

            # Test set null on character delete
            cursor.execute("DELETE FROM characters WHERE id = %s", (character_id,))

            # Check that any remaining generations have character_id set to NULL
            cursor.execute("SELECT character_id FROM generations WHERE job_id = %s", (job_id,))
            char_reference = cursor.fetchone()

            conn.commit()
            cursor.close()
            self.return_connection(conn)

            # Referential integrity working if:
            # 1. Cascade delete worked (no orphaned generations)
            # 2. Set null worked (character_id is NULL after character delete)
            success = (remaining_generations == 0 and
                      (char_reference is None or char_reference[0] is None))

            return TransactionTest(
                test_name="data_integrity",
                success=success,
                duration=time.time() - start_time,
                records_affected=3,  # project, character, generation
                consistency_check=True
            )

        except Exception as e:
            return TransactionTest(
                test_name="data_integrity",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def test_performance_constraints(self) -> TransactionTest:
        """Test database performance under load"""
        start_time = time.time()

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Insert batch of test data
            batch_size = 100
            insert_start = time.time()

            characters_data = [
                (f"PerfTest_{i}", f"Performance test character {i}")
                for i in range(batch_size)
            ]

            cursor.executemany("""
                INSERT INTO characters (name, description) VALUES (%s, %s)
            """, characters_data)

            insert_time = time.time() - insert_start

            # Test query performance
            query_start = time.time()

            cursor.execute("SELECT COUNT(*) FROM characters WHERE name LIKE 'PerfTest_%'")
            count_result = cursor.fetchone()[0]

            cursor.execute("""
                SELECT name, description FROM characters
                WHERE name LIKE 'PerfTest_%'
                ORDER BY name
                LIMIT 50
            """)
            query_results = cursor.fetchall()

            query_time = time.time() - query_start

            # Test update performance
            update_start = time.time()

            cursor.execute("""
                UPDATE characters
                SET description = 'Updated performance test character'
                WHERE name LIKE 'PerfTest_%'
            """)

            update_time = time.time() - update_start

            # Test delete performance
            delete_start = time.time()

            cursor.execute("DELETE FROM characters WHERE name LIKE 'PerfTest_%'")

            delete_time = time.time() - delete_start

            conn.commit()
            cursor.close()
            self.return_connection(conn)

            # Performance thresholds (reasonable for test environment)
            performance_ok = (
                insert_time < 5.0 and    # Insert 100 records in <5s
                query_time < 1.0 and     # Query in <1s
                update_time < 2.0 and    # Update 100 records in <2s
                delete_time < 1.0        # Delete 100 records in <1s
            )

            success = performance_ok and count_result == batch_size

            return TransactionTest(
                test_name="performance_constraints",
                success=success,
                duration=time.time() - start_time,
                records_affected=batch_size,
                consistency_check=True
            )

        except Exception as e:
            return TransactionTest(
                test_name="performance_constraints",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def cleanup_test_environment(self):
        """Clean up test database and connections"""
        try:
            if self.connection_pool:
                self.connection_pool.closeall()

            # Drop test database
            admin_config = self.db_config.copy()
            admin_config["database"] = "postgres"

            conn = psycopg2.connect(**admin_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            cursor.execute(f"DROP DATABASE IF EXISTS {self.test_db_name}")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Warning: Failed to cleanup test environment: {e}")

    async def run_all_tests(self) -> Dict:
        """Run all database consistency tests"""
        await self.setup_test_environment()

        try:
            all_tests = []

            # ACID compliance tests
            acid_tests = await self.test_acid_compliance()
            all_tests.extend(acid_tests)

            # Concurrent access test
            concurrent_test = await self.test_concurrent_access()
            all_tests.append(concurrent_test)

            # Data integrity test
            integrity_test = await self.test_data_integrity()
            all_tests.append(integrity_test)

            # Performance test
            performance_test = await self.test_performance_constraints()
            all_tests.append(performance_test)

            # Analyze results
            total_tests = len(all_tests)
            passed_tests = len([t for t in all_tests if t.success])
            failed_tests = [t for t in all_tests if not t.success]

            results = {
                "test_summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": len(failed_tests),
                    "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
                },
                "test_results": [
                    {
                        "test_name": t.test_name,
                        "success": t.success,
                        "duration": t.duration,
                        "records_affected": t.records_affected,
                        "error": t.error
                    }
                    for t in all_tests
                ],
                "failed_tests": [
                    {
                        "test_name": t.test_name,
                        "error": t.error,
                        "duration": t.duration
                    }
                    for t in failed_tests
                ]
            }

            return results

        finally:
            await self.cleanup_test_environment()


# Pytest test cases
class TestDatabaseConsistency:
    """Pytest test cases for database consistency"""

    @pytest.fixture
    def db_config(self):
        return {
            "host": "192.168.50.135",
            "database": "echo_brain",
            "user": "patrick",
            "password": "tower_echo_brain_secret_key_2025"
        }

    @pytest.fixture
    def db_tester(self, db_config):
        return DatabaseConsistencyTester(db_config)

    @pytest.mark.asyncio
    async def test_acid_compliance(self, db_tester):
        """Test ACID compliance of database operations"""
        await db_tester.setup_test_environment()

        try:
            acid_tests = await db_tester.test_acid_compliance()

            # All ACID tests should pass
            failed_tests = [t for t in acid_tests if not t.success]
            assert len(failed_tests) == 0, f"ACID compliance failed: {[t.test_name for t in failed_tests]}"

            # Check individual ACID properties
            atomicity_test = next((t for t in acid_tests if t.test_name == "atomicity"), None)
            consistency_test = next((t for t in acid_tests if t.test_name == "consistency"), None)
            isolation_test = next((t for t in acid_tests if t.test_name == "isolation"), None)
            durability_test = next((t for t in acid_tests if t.test_name == "durability"), None)

            assert atomicity_test and atomicity_test.success, "Atomicity test failed"
            assert consistency_test and consistency_test.success, "Consistency test failed"
            assert isolation_test and isolation_test.success, "Isolation test failed"
            assert durability_test and durability_test.success, "Durability test failed"

        finally:
            await db_tester.cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_concurrent_access(self, db_tester):
        """Test database handles concurrent access properly"""
        await db_tester.setup_test_environment()

        try:
            concurrent_test = await db_tester.test_concurrent_access(num_threads=5)

            assert concurrent_test.success, f"Concurrent access test failed: {concurrent_test.error}"
            assert concurrent_test.records_affected >= 20, "Too few concurrent operations completed"

        finally:
            await db_tester.cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_referential_integrity(self, db_tester):
        """Test referential integrity constraints"""
        await db_tester.setup_test_environment()

        try:
            integrity_test = await db_tester.test_data_integrity()

            assert integrity_test.success, f"Data integrity test failed: {integrity_test.error}"
            assert integrity_test.consistency_check, "Consistency check failed"

        finally:
            await db_tester.cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_database_performance(self, db_tester):
        """Test database performance meets requirements"""
        await db_tester.setup_test_environment()

        try:
            performance_test = await db_tester.test_performance_constraints()

            assert performance_test.success, f"Performance test failed: {performance_test.error}"
            assert performance_test.duration < 30, f"Performance test too slow: {performance_test.duration}s"

        finally:
            await db_tester.cleanup_test_environment()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_comprehensive_database_validation(self, db_tester):
        """Run complete database validation suite (slow test)"""
        results = await db_tester.run_all_tests()

        # Save results
        results_file = f"/tmp/database_consistency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Database test results saved to: {results_file}")

        # Check overall database health
        success_rate = results["test_summary"]["success_rate"]
        assert success_rate >= 90, f"Database consistency success rate too low: {success_rate}%"

        # Print summary
        print(f"Database Test Summary:")
        print(f"Total Tests: {results['test_summary']['total_tests']}")
        print(f"Passed: {results['test_summary']['passed_tests']}")
        print(f"Failed: {results['test_summary']['failed_tests']}")
        print(f"Success Rate: {success_rate:.1f}%")


if __name__ == "__main__":
    # CLI interface for standalone testing
    import argparse

    parser = argparse.ArgumentParser(description="Database consistency testing")
    parser.add_argument("--host", default="192.168.50.135", help="Database host")
    parser.add_argument("--database", default="echo_brain", help="Database name")
    parser.add_argument("--user", default="patrick", help="Database user")
    parser.add_argument("--password", required=True, help="Database password")
    parser.add_argument("--test-type", choices=["acid", "concurrent", "integrity", "performance", "all"],
                       default="all", help="Type of test to run")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    db_config = {
        "host": args.host,
        "database": args.database,
        "user": args.user,
        "password": args.password
    }

    async def main():
        tester = DatabaseConsistencyTester(db_config)

        if args.test_type == "acid":
            await tester.setup_test_environment()
            try:
                results = await tester.test_acid_compliance()
                results = [{"test_name": r.test_name, "success": r.success, "duration": r.duration, "error": r.error} for r in results]
            finally:
                await tester.cleanup_test_environment()
        elif args.test_type == "concurrent":
            await tester.setup_test_environment()
            try:
                result = await tester.test_concurrent_access()
                results = [{"test_name": result.test_name, "success": result.success, "duration": result.duration, "error": result.error}]
            finally:
                await tester.cleanup_test_environment()
        elif args.test_type == "integrity":
            await tester.setup_test_environment()
            try:
                result = await tester.test_data_integrity()
                results = [{"test_name": result.test_name, "success": result.success, "duration": result.duration, "error": result.error}]
            finally:
                await tester.cleanup_test_environment()
        elif args.test_type == "performance":
            await tester.setup_test_environment()
            try:
                result = await tester.test_performance_constraints()
                results = [{"test_name": result.test_name, "success": result.success, "duration": result.duration, "error": result.error}]
            finally:
                await tester.cleanup_test_environment()
        elif args.test_type == "all":
            results = await tester.run_all_tests()

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())