#!/usr/bin/env python3
"""
Database Interface Protocols
Defines contracts for database operations and connection management
"""

from typing import Protocol, runtime_checkable, List, Dict, Optional, Any, AsyncContextManager
from datetime import datetime
import asyncio

@runtime_checkable
class DatabaseInterface(Protocol):
    """
    Protocol for synchronous database operations

    Defines standardized methods for database interactions with
    proper connection management and query execution.
    """

    def connect(self) -> bool:
        """
        Establish database connection

        Returns:
            bool: True if connection was successful
        """
        ...

    def disconnect(self) -> bool:
        """
        Close database connection

        Returns:
            bool: True if disconnection was successful
        """
        ...

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            List[Dict]: Query results as dictionaries
        """
        ...

    def execute_command(self, command: str, params: Optional[tuple] = None) -> bool:
        """
        Execute an INSERT/UPDATE/DELETE command

        Args:
            command: SQL command string
            params: Optional command parameters

        Returns:
            bool: True if command executed successfully
        """
        ...

    def execute_script(self, script: str) -> bool:
        """
        Execute a multi-statement SQL script

        Args:
            script: SQL script content

        Returns:
            bool: True if script executed successfully
        """
        ...

    def begin_transaction(self) -> bool:
        """
        Begin a database transaction

        Returns:
            bool: True if transaction started successfully
        """
        ...

    def commit_transaction(self) -> bool:
        """
        Commit current transaction

        Returns:
            bool: True if transaction committed successfully
        """
        ...

    def rollback_transaction(self) -> bool:
        """
        Rollback current transaction

        Returns:
            bool: True if transaction rolled back successfully
        """
        ...

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a table structure

        Args:
            table_name: Name of the table

        Returns:
            Dict: Table metadata including columns, types, etc.
        """
        ...

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists

        Args:
            table_name: Name of the table to check

        Returns:
            bool: True if table exists
        """
        ...

    def get_last_insert_id(self) -> Optional[int]:
        """
        Get the ID of the last inserted row

        Returns:
            Optional[int]: Last insert ID or None
        """
        ...


@runtime_checkable
class AsyncDatabaseInterface(Protocol):
    """
    Protocol for asynchronous database operations

    Defines standardized async methods for database interactions with
    connection pooling and optimized async performance.
    """

    async def initialize(self) -> bool:
        """
        Initialize database connection pool

        Returns:
            bool: True if initialization was successful
        """
        ...

    async def close(self) -> bool:
        """
        Close database connection pool

        Returns:
            bool: True if cleanup was successful
        """
        ...

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute async SELECT query

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            List[Dict]: Query results
        """
        ...

    async def execute_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Execute query and return single result

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            Optional[Dict]: Single result or None
        """
        ...

    async def execute_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute async INSERT/UPDATE/DELETE command

        Args:
            command: SQL command string
            params: Optional command parameters

        Returns:
            bool: True if command executed successfully
        """
        ...

    async def execute_batch(self, commands: List[str], params_list: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Execute multiple commands in a batch

        Args:
            commands: List of SQL commands
            params_list: Optional list of parameter dictionaries

        Returns:
            bool: True if all commands executed successfully
        """
        ...

    async def transaction(self) -> AsyncContextManager:
        """
        Get async transaction context manager

        Returns:
            AsyncContextManager: Transaction context
        """
        ...

    async def get_connection(self) -> Any:
        """
        Get a database connection from the pool

        Returns:
            Database connection object
        """
        ...

    async def health_check(self) -> bool:
        """
        Perform database health check

        Returns:
            bool: True if database is healthy
        """
        ...

    async def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics

        Returns:
            Dict: Pool statistics including active/idle connections
        """
        ...


@runtime_checkable
class ConversationDatabaseInterface(Protocol):
    """
    Protocol for conversation-specific database operations
    """

    async def log_interaction(self, query: str, response: str, model_used: str,
                            processing_time: float, user_id: str = "anonymous",
                            conversation_id: Optional[str] = None) -> str:
        """
        Log a conversation interaction

        Args:
            query: User query
            response: System response
            model_used: AI model identifier
            processing_time: Time taken to process
            user_id: User identifier
            conversation_id: Optional conversation ID

        Returns:
            str: Interaction ID
        """
        ...

    async def get_conversation_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user

        Args:
            user_id: User identifier
            limit: Maximum number of interactions

        Returns:
            List[Dict]: Conversation history
        """
        ...

    async def create_conversation(self, user_id: str, title: Optional[str] = None) -> str:
        """
        Create a new conversation

        Args:
            user_id: User identifier
            title: Optional conversation title

        Returns:
            str: Conversation ID
        """
        ...

    async def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """
        Update conversation title

        Args:
            conversation_id: Conversation identifier
            title: New title

        Returns:
            bool: True if updated successfully
        """
        ...

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and its interactions

        Args:
            conversation_id: Conversation identifier

        Returns:
            bool: True if deleted successfully
        """
        ...

    async def search_conversations(self, user_id: str, search_term: str,
                                 limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search conversations by content

        Args:
            user_id: User identifier
            search_term: Search term
            limit: Maximum results

        Returns:
            List[Dict]: Matching conversations
        """
        ...


@runtime_checkable
class LearningDatabaseInterface(Protocol):
    """
    Protocol for learning and improvement tracking database operations
    """

    async def store_learning(self, content: str, category: str, confidence: float,
                           source: str = "autonomous") -> str:
        """
        Store a learning insight

        Args:
            content: Learning content
            category: Category of learning
            confidence: Confidence score (0-1)
            source: Learning source

        Returns:
            str: Learning ID
        """
        ...

    async def get_learnings(self, category: Optional[str] = None,
                          min_confidence: float = 0.0,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get stored learnings

        Args:
            category: Optional category filter
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List[Dict]: Learning records
        """
        ...

    async def update_learning_confidence(self, learning_id: str, new_confidence: float) -> bool:
        """
        Update confidence score for a learning

        Args:
            learning_id: Learning identifier
            new_confidence: New confidence score

        Returns:
            bool: True if updated successfully
        """
        ...

    async def mark_learning_applied(self, learning_id: str, application_context: str) -> bool:
        """
        Mark a learning as applied in practice

        Args:
            learning_id: Learning identifier
            application_context: Context where learning was applied

        Returns:
            bool: True if marked successfully
        """
        ...


@runtime_checkable
class MigrationInterface(Protocol):
    """
    Protocol for database migration operations
    """

    async def get_current_version(self) -> int:
        """
        Get current database schema version

        Returns:
            int: Current schema version
        """
        ...

    async def apply_migration(self, migration_name: str, migration_sql: str) -> bool:
        """
        Apply a database migration

        Args:
            migration_name: Name of the migration
            migration_sql: SQL commands for migration

        Returns:
            bool: True if migration applied successfully
        """
        ...

    async def rollback_migration(self, migration_name: str) -> bool:
        """
        Rollback a database migration

        Args:
            migration_name: Name of migration to rollback

        Returns:
            bool: True if rollback successful
        """
        ...

    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """
        Get history of applied migrations

        Returns:
            List[Dict]: Migration history records
        """
        ...

    async def create_migration_table(self) -> bool:
        """
        Create migration tracking table

        Returns:
            bool: True if table created successfully
        """
        ...