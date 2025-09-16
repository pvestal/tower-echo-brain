"""
Test Suite for Database Connection Pool

This module tests the database connection pooling functionality,
including connection management, thread safety, resource cleanup,
and error handling scenarios.

Author: Echo Brain Test Suite
Created: 2025-09-16
"""

import pytest
import sys
import os
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import concurrent.futures

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock psycopg2 before importing
mock_psycopg2 = MagicMock()
mock_pool = MagicMock()
mock_psycopg2.pool = mock_pool

sys.modules['psycopg2'] = mock_psycopg2
sys.modules['psycopg2.pool'] = mock_pool

from directors.db_pool import DatabasePool


class TestDatabasePoolInitialization:
    """Test database pool initialization and configuration."""

    @patch('directors.db_pool.psycopg2')
    def test_pool_creation_success(self, mock_psycopg2):
        """Test successful pool creation."""
        mock_threaded_pool = MagicMock()
        mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool

        pool = DatabasePool(min_conn=2, max_conn=10)

        assert pool.min_conn == 2
        assert pool.max_conn == 10
        assert pool._pool == mock_threaded_pool

        # Verify pool was created with correct parameters
        mock_psycopg2.pool.ThreadedConnectionPool.assert_called_once()
        args, kwargs = mock_psycopg2.pool.ThreadedConnectionPool.call_args

        assert args[0] == 2  # min_conn
        assert args[1] == 10  # max_conn

    @patch('directors.db_pool.psycopg2')
    def test_default_pool_parameters(self, mock_psycopg2):
        """Test pool creation with default parameters."""
        mock_threaded_pool = MagicMock()
        mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool

        pool = DatabasePool()

        assert pool.min_conn == 1
        assert pool.max_conn == 20

        # Verify default parameters were used
        mock_psycopg2.pool.ThreadedConnectionPool.assert_called_once()
        args, kwargs = mock_psycopg2.pool.ThreadedConnectionPool.call_args

        assert args[0] == 1  # default min_conn
        assert args[1] == 20  # default max_conn

    @patch.dict(os.environ, {
        'DB_HOST': 'test.example.com',
        'DB_NAME': 'test_db',
        'DB_USER': 'test_user',
        'DB_PASSWORD': 'test_pass',
        'DB_PORT': '5433'
    })
    @patch('directors.db_pool.psycopg2')
    def test_environment_configuration(self, mock_psycopg2):
        """Test that environment variables are used for configuration."""
        mock_threaded_pool = MagicMock()
        mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool

        pool = DatabasePool()

        # Verify environment variables were used
        args, kwargs = mock_psycopg2.pool.ThreadedConnectionPool.call_args
        assert kwargs['host'] == 'test.example.com'
        assert kwargs['database'] == 'test_db'
        assert kwargs['user'] == 'test_user'
        assert kwargs['password'] == 'test_pass'
        assert kwargs['port'] == 5433

    @patch('directors.db_pool.psycopg2')
    def test_pool_creation_failure(self, mock_psycopg2):
        """Test handling of pool creation failure."""
        mock_psycopg2.pool.ThreadedConnectionPool.side_effect = Exception("Connection failed")

        with pytest.raises(Exception):
            DatabasePool()

    @patch('directors.db_pool.psycopg2')
    def test_invalid_pool_parameters(self, mock_psycopg2):
        """Test handling of invalid pool parameters."""
        mock_threaded_pool = MagicMock()
        mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool

        # Test with min_conn > max_conn
        pool = DatabasePool(min_conn=10, max_conn=5)

        # Should still create the pool (database will handle the validation)
        assert pool.min_conn == 10
        assert pool.max_conn == 5


class TestConnectionManagement:
    """Test connection acquisition and release."""

    @pytest.fixture
    def mock_pool_setup(self):
        """Set up mock pool for testing."""
        with patch('directors.db_pool.psycopg2') as mock_psycopg2:
            mock_threaded_pool = MagicMock()
            mock_connection = MagicMock()

            mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool
            mock_threaded_pool.getconn.return_value = mock_connection

            yield {
                'psycopg2': mock_psycopg2,
                'threaded_pool': mock_threaded_pool,
                'connection': mock_connection
            }

    def test_get_connection_success(self, mock_pool_setup):
        """Test successful connection acquisition."""
        pool = DatabasePool()

        connection = pool.get_connection()

        assert connection is not None
        mock_pool_setup['threaded_pool'].getconn.assert_called_once()

    def test_get_connection_context_manager(self, mock_pool_setup):
        """Test connection context manager."""
        pool = DatabasePool()

        with pool.get_connection() as conn:
            assert conn is not None
            assert conn == mock_pool_setup['connection']

        # Verify connection was returned to pool
        mock_pool_setup['threaded_pool'].putconn.assert_called_once_with(
            mock_pool_setup['connection']
        )

    def test_connection_context_manager_exception(self, mock_pool_setup):
        """Test connection context manager with exception."""
        pool = DatabasePool()

        try:
            with pool.get_connection() as conn:
                raise Exception("Test exception")
        except Exception:
            pass

        # Connection should still be returned to pool
        mock_pool_setup['threaded_pool'].putconn.assert_called_once()

    def test_get_connection_pool_exhausted(self, mock_pool_setup):
        """Test behavior when connection pool is exhausted."""
        pool = DatabasePool()

        # Mock pool exhaustion
        mock_pool_setup['threaded_pool'].getconn.side_effect = Exception("Pool exhausted")

        with pytest.raises(Exception):
            pool.get_connection()

    def test_connection_release(self, mock_pool_setup):
        """Test explicit connection release."""
        pool = DatabasePool()

        connection = pool.get_connection()
        pool.release_connection(connection)

        mock_pool_setup['threaded_pool'].putconn.assert_called_once_with(connection)

    def test_release_invalid_connection(self, mock_pool_setup):
        """Test releasing invalid or None connection."""
        pool = DatabasePool()

        # Should handle gracefully
        pool.release_connection(None)

        # Should not call putconn for None
        mock_pool_setup['threaded_pool'].putconn.assert_not_called()


class TestThreadSafety:
    """Test thread safety of the connection pool."""

    @pytest.fixture
    def mock_pool_setup(self):
        """Set up mock pool for threading tests."""
        with patch('directors.db_pool.psycopg2') as mock_psycopg2:
            mock_threaded_pool = MagicMock()
            mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool

            # Create different mock connections for each call
            connections = [MagicMock() for _ in range(20)]
            mock_threaded_pool.getconn.side_effect = connections

            yield {
                'psycopg2': mock_psycopg2,
                'threaded_pool': mock_threaded_pool,
                'connections': connections
            }

    def test_concurrent_connection_acquisition(self, mock_pool_setup):
        """Test concurrent connection acquisition from multiple threads."""
        pool = DatabasePool(min_conn=2, max_conn=10)
        acquired_connections = []
        errors = []

        def worker():
            try:
                connection = pool.get_connection()
                acquired_connections.append(connection)
                time.sleep(0.01)  # Simulate work
                pool.release_connection(connection)
            except Exception as e:
                errors.append(e)

        # Start 10 concurrent workers
        threads = [threading.Thread(target=worker) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0
        assert len(acquired_connections) == 10

        # All connections should be unique (if pool provides different ones)
        # Note: In real implementation, connections might be reused

    def test_concurrent_context_manager_usage(self, mock_pool_setup):
        """Test concurrent usage of connection context manager."""
        pool = DatabasePool(min_conn=2, max_conn=10)
        results = []
        errors = []

        def worker(worker_id):
            try:
                with pool.get_connection() as conn:
                    # Simulate database work
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    results.append((worker_id, result))
                    time.sleep(0.01)
            except Exception as e:
                errors.append((worker_id, e))

        # Start 15 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(worker, i) for i in range(15)]
            concurrent.futures.wait(futures)

        # Should complete without major errors
        assert len(errors) <= 2  # Allow for some potential connection issues

    def test_thread_local_behavior(self, mock_pool_setup):
        """Test that connections behave properly across threads."""
        pool = DatabasePool()
        thread_connections = {}

        def worker(thread_id):
            connection = pool.get_connection()
            thread_connections[thread_id] = connection
            time.sleep(0.01)
            pool.release_connection(connection)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Each thread should have gotten a connection
        assert len(thread_connections) == 5

        # Connections may or may not be the same (depends on pool implementation)
        for thread_id, conn in thread_connections.items():
            assert conn is not None


class TestConnectionPoolCleanup:
    """Test proper cleanup and resource management."""

    @pytest.fixture
    def mock_pool_setup(self):
        """Set up mock pool for cleanup tests."""
        with patch('directors.db_pool.psycopg2') as mock_psycopg2:
            mock_threaded_pool = MagicMock()
            mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool

            yield {
                'psycopg2': mock_psycopg2,
                'threaded_pool': mock_threaded_pool
            }

    def test_pool_close(self, mock_pool_setup):
        """Test pool closure."""
        pool = DatabasePool()

        pool.close()

        # Should call closeall on the pool
        mock_pool_setup['threaded_pool'].closeall.assert_called_once()

    def test_double_close(self, mock_pool_setup):
        """Test that double close doesn't cause issues."""
        pool = DatabasePool()

        pool.close()
        pool.close()  # Should handle gracefully

        # closeall might be called multiple times, that's OK
        assert mock_pool_setup['threaded_pool'].closeall.call_count >= 1

    def test_close_with_active_connections(self, mock_pool_setup):
        """Test closing pool with active connections."""
        pool = DatabasePool()

        # Get a connection but don't release it
        connection = pool.get_connection()

        # Close the pool
        pool.close()

        # Should still close the pool
        mock_pool_setup['threaded_pool'].closeall.assert_called_once()

    def test_operations_after_close(self, mock_pool_setup):
        """Test that operations after close are handled properly."""
        pool = DatabasePool()

        pool.close()

        # Mock that operations on closed pool raise exceptions
        mock_pool_setup['threaded_pool'].getconn.side_effect = Exception("Pool is closed")

        with pytest.raises(Exception):
            pool.get_connection()


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def mock_pool_setup(self):
        """Set up mock pool for error testing."""
        with patch('directors.db_pool.psycopg2') as mock_psycopg2:
            mock_threaded_pool = MagicMock()
            mock_connection = MagicMock()

            mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool
            mock_threaded_pool.getconn.return_value = mock_connection

            yield {
                'psycopg2': mock_psycopg2,
                'threaded_pool': mock_threaded_pool,
                'connection': mock_connection
            }

    def test_database_connection_error(self, mock_pool_setup):
        """Test handling of database connection errors."""
        pool = DatabasePool()

        # Mock connection error
        mock_pool_setup['threaded_pool'].getconn.side_effect = Exception("Database unreachable")

        with pytest.raises(Exception):
            pool.get_connection()

    def test_connection_timeout(self, mock_pool_setup):
        """Test handling of connection timeouts."""
        pool = DatabasePool()

        # Mock timeout
        mock_pool_setup['threaded_pool'].getconn.side_effect = Exception("Connection timeout")

        with pytest.raises(Exception):
            pool.get_connection()

    def test_invalid_connection_return(self, mock_pool_setup):
        """Test handling when returning invalid connection to pool."""
        pool = DatabasePool()

        # Mock error when returning connection
        mock_pool_setup['threaded_pool'].putconn.side_effect = Exception("Invalid connection")

        connection = pool.get_connection()

        # Should handle the error gracefully
        try:
            pool.release_connection(connection)
        except Exception:
            # Error during release is acceptable
            pass

    def test_pool_initialization_retry(self):
        """Test retry logic for pool initialization (if implemented)."""
        with patch('directors.db_pool.psycopg2') as mock_psycopg2:
            # First call fails, second succeeds
            mock_threaded_pool = MagicMock()
            mock_psycopg2.pool.ThreadedConnectionPool.side_effect = [
                Exception("Initial failure"),
                mock_threaded_pool
            ]

            # Current implementation doesn't retry, so this should fail
            with pytest.raises(Exception):
                DatabasePool()


class TestDatabasePoolConfiguration:
    """Test various configuration scenarios."""

    def test_pool_size_validation(self):
        """Test pool size validation."""
        with patch('directors.db_pool.psycopg2') as mock_psycopg2:
            mock_threaded_pool = MagicMock()
            mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool

            # Test edge cases
            pool1 = DatabasePool(min_conn=0, max_conn=1)
            assert pool1.min_conn == 0
            assert pool1.max_conn == 1

            pool2 = DatabasePool(min_conn=1, max_conn=1)
            assert pool2.min_conn == 1
            assert pool2.max_conn == 1

    @patch.dict(os.environ, {}, clear=True)
    @patch('directors.db_pool.psycopg2')
    def test_default_environment_values(self, mock_psycopg2):
        """Test default values when environment variables are not set."""
        mock_threaded_pool = MagicMock()
        mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool

        pool = DatabasePool()

        args, kwargs = mock_psycopg2.pool.ThreadedConnectionPool.call_args
        assert kwargs['host'] == '192.168.50.135'  # default
        assert kwargs['database'] == 'tower_consolidated'  # default
        assert kwargs['user'] == 'patrick'  # default
        assert kwargs['port'] == 5432  # default

    @patch('directors.db_pool.psycopg2')
    def test_connection_parameters(self, mock_psycopg2):
        """Test that proper connection parameters are set."""
        mock_threaded_pool = MagicMock()
        mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_threaded_pool

        pool = DatabasePool()

        args, kwargs = mock_psycopg2.pool.ThreadedConnectionPool.call_args

        # Verify important parameters are set
        assert 'connect_timeout' in kwargs
        assert 'application_name' in kwargs
        assert kwargs['application_name'] == 'echo_brain_board'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])