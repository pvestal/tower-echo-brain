"""
Test Data Management and Fixture System for Echo Brain Testing
==============================================================

This module provides comprehensive test data management including fixture generation,
test data factories, database seeding, mock data creation, and test environment setup
for Echo Brain's comprehensive testing framework.

Features:
- Dynamic fixture generation
- Test data factories
- Database seeding and cleanup
- Mock data generation
- Test environment isolation
- Data versioning and snapshots

Author: Development Testing Framework Agent
Created: 2025-11-06
"""

import json
import os
import pickle
import tempfile
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union, Type, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

import pytest
import psycopg2
from faker import Faker

from .test_framework_core import TestFrameworkCore


@dataclass
class TestDataConfig:
    """Configuration for test data management."""
    name: str
    data_type: str  # 'fixture', 'mock', 'database', 'file'
    source_type: str  # 'generated', 'template', 'snapshot', 'factory'
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatabaseSnapshot:
    """Database snapshot for test isolation."""
    snapshot_id: str
    database_name: str
    timestamp: datetime
    table_data: Dict[str, List[Dict[str, Any]]]
    schema_version: str
    file_path: Optional[str] = None


class DataFactory(ABC):
    """Abstract base class for test data factories."""
    
    @abstractmethod
    def generate(self, count: int = 1, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate test data."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get data schema."""
        pass


class UserDataFactory(DataFactory):
    """Factory for generating user test data."""
    
    def __init__(self):
        """Initialize user data factory."""
        self.faker = Faker()
        
    def generate(self, count: int = 1, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate user test data."""
        users = []
        
        for i in range(count):
            user = {
                'user_id': f"test_user_{i:04d}",
                'username': self.faker.user_name(),
                'email': self.faker.email(),
                'first_name': self.faker.first_name(),
                'last_name': self.faker.last_name(),
                'created_at': self.faker.date_time_between(start_date='-1y', end_date='now').isoformat(),
                'is_active': self.faker.boolean(chance_of_getting_true=80),
                'permissions': self.faker.random_elements(
                    elements=['read', 'write', 'admin', 'board:submit', 'board:view'],
                    length=self.faker.random_int(min=1, max=3),
                    unique=True
                ),
                'profile': {
                    'bio': self.faker.text(max_nb_chars=200),
                    'location': self.faker.city(),
                    'timezone': self.faker.timezone()
                }
            }
            
            # Apply custom overrides
            for key, value in kwargs.items():
                if key in user:
                    user[key] = value
                    
            users.append(user)
            
        return users[0] if count == 1 else users
        
    def get_schema(self) -> Dict[str, Any]:
        """Get user data schema."""
        return {
            'type': 'object',
            'properties': {
                'user_id': {'type': 'string'},
                'username': {'type': 'string'},
                'email': {'type': 'string', 'format': 'email'},
                'first_name': {'type': 'string'},
                'last_name': {'type': 'string'},
                'created_at': {'type': 'string', 'format': 'date-time'},
                'is_active': {'type': 'boolean'},
                'permissions': {'type': 'array', 'items': {'type': 'string'}},
                'profile': {
                    'type': 'object',
                    'properties': {
                        'bio': {'type': 'string'},
                        'location': {'type': 'string'},
                        'timezone': {'type': 'string'}
                    }
                }
            },
            'required': ['user_id', 'username', 'email']
        }


class TaskDataFactory(DataFactory):
    """Factory for generating task test data."""
    
    def __init__(self):
        """Initialize task data factory."""
        self.faker = Faker()
        
    def generate(self, count: int = 1, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate task test data."""
        tasks = []
        
        task_types = ['code_review', 'security_audit', 'performance_test', 'ai_decision']
        priorities = ['low', 'medium', 'high', 'critical']
        statuses = ['pending', 'in_progress', 'completed', 'failed']
        
        for i in range(count):
            task = {
                'task_id': f"task_{i:06d}",
                'task_type': self.faker.random_element(task_types),
                'title': self.faker.sentence(nb_words=6),
                'description': self.faker.text(max_nb_chars=500),
                'priority': self.faker.random_element(priorities),
                'status': self.faker.random_element(statuses),
                'created_at': self.faker.date_time_between(start_date='-30d', end_date='now').isoformat(),
                'updated_at': self.faker.date_time_between(start_date='-7d', end_date='now').isoformat(),
                'assigned_to': f"user_{self.faker.random_int(min=1, max=100):03d}",
                'estimated_hours': self.faker.random_int(min=1, max=40),
                'actual_hours': self.faker.random_int(min=0, max=50),
                'tags': self.faker.random_elements(
                    elements=['backend', 'frontend', 'ai', 'security', 'performance'],
                    length=self.faker.random_int(min=1, max=3),
                    unique=True
                ),
                'context': {
                    'code': self.faker.text(max_nb_chars=1000) if self.faker.boolean() else None,
                    'files': [f"src/{self.faker.file_name()}" for _ in range(self.faker.random_int(min=0, max=5))],
                    'environment': self.faker.random_element(['development', 'staging', 'production'])
                }
            }
            
            # Apply custom overrides
            for key, value in kwargs.items():
                if key in task:
                    task[key] = value
                    
            tasks.append(task)
            
        return tasks[0] if count == 1 else tasks
        
    def get_schema(self) -> Dict[str, Any]:
        """Get task data schema."""
        return {
            'type': 'object',
            'properties': {
                'task_id': {'type': 'string'},
                'task_type': {'type': 'string'},
                'title': {'type': 'string'},
                'description': {'type': 'string'},
                'priority': {'type': 'string', 'enum': ['low', 'medium', 'high', 'critical']},
                'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'completed', 'failed']},
                'created_at': {'type': 'string', 'format': 'date-time'},
                'updated_at': {'type': 'string', 'format': 'date-time'},
                'assigned_to': {'type': 'string'},
                'estimated_hours': {'type': 'integer', 'minimum': 0},
                'actual_hours': {'type': 'integer', 'minimum': 0},
                'tags': {'type': 'array', 'items': {'type': 'string'}},
                'context': {'type': 'object'}
            },
            'required': ['task_id', 'task_type', 'title', 'priority', 'status']
        }


class AIModelDataFactory(DataFactory):
    """Factory for generating AI model test data."""
    
    def __init__(self):
        """Initialize AI model data factory."""
        self.faker = Faker()
        
    def generate(self, count: int = 1, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate AI model test data."""
        models = []
        
        model_types = ['classification', 'regression', 'clustering', 'recommendation']
        frameworks = ['tensorflow', 'pytorch', 'scikit-learn', 'xgboost']
        
        for i in range(count):
            model = {
                'model_id': f"model_{i:04d}",
                'name': f"{self.faker.word()}_{self.faker.word()}_model",
                'version': f"{self.faker.random_int(min=1, max=5)}.{self.faker.random_int(min=0, max=9)}.{self.faker.random_int(min=0, max=9)}",
                'type': self.faker.random_element(model_types),
                'framework': self.faker.random_element(frameworks),
                'accuracy': round(self.faker.random.uniform(0.7, 0.99), 4),
                'parameters': self.faker.random_int(min=1000, max=100000000),
                'training_data_size': self.faker.random_int(min=1000, max=1000000),
                'created_at': self.faker.date_time_between(start_date='-1y', end_date='now').isoformat(),
                'last_trained': self.faker.date_time_between(start_date='-30d', end_date='now').isoformat(),
                'hyperparameters': {
                    'learning_rate': round(self.faker.random.uniform(0.001, 0.1), 6),
                    'batch_size': self.faker.random_element([16, 32, 64, 128, 256]),
                    'epochs': self.faker.random_int(min=10, max=1000),
                    'dropout_rate': round(self.faker.random.uniform(0.1, 0.5), 2)
                },
                'metrics': {
                    'precision': round(self.faker.random.uniform(0.7, 0.99), 4),
                    'recall': round(self.faker.random.uniform(0.7, 0.99), 4),
                    'f1_score': round(self.faker.random.uniform(0.7, 0.99), 4),
                    'training_time_hours': round(self.faker.random.uniform(0.5, 48.0), 2)
                },
                'status': self.faker.random_element(['training', 'deployed', 'archived', 'failed'])
            }
            
            # Apply custom overrides
            for key, value in kwargs.items():
                if key in model:
                    model[key] = value
                    
            models.append(model)
            
        return models[0] if count == 1 else models
        
    def get_schema(self) -> Dict[str, Any]:
        """Get AI model data schema."""
        return {
            'type': 'object',
            'properties': {
                'model_id': {'type': 'string'},
                'name': {'type': 'string'},
                'version': {'type': 'string'},
                'type': {'type': 'string'},
                'framework': {'type': 'string'},
                'accuracy': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'parameters': {'type': 'integer', 'minimum': 0},
                'training_data_size': {'type': 'integer', 'minimum': 0},
                'created_at': {'type': 'string', 'format': 'date-time'},
                'last_trained': {'type': 'string', 'format': 'date-time'},
                'hyperparameters': {'type': 'object'},
                'metrics': {'type': 'object'},
                'status': {'type': 'string'}
            },
            'required': ['model_id', 'name', 'type', 'framework']
        }


class FixtureManager:
    """Manages test fixtures and data generation."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize fixture manager."""
        self.framework = framework
        self.factories: Dict[str, DataFactory] = {}
        self.fixtures: Dict[str, Any] = {}
        self.data_configs: Dict[str, TestDataConfig] = {}
        self.temp_directories: List[str] = []
        
        # Register built-in factories
        self._register_builtin_factories()
        
    def _register_builtin_factories(self):
        """Register built-in data factories."""
        self.register_factory('users', UserDataFactory())
        self.register_factory('tasks', TaskDataFactory())
        self.register_factory('ai_models', AIModelDataFactory())
        
    def register_factory(self, name: str, factory: DataFactory):
        """Register a data factory."""
        self.factories[name] = factory
        
    def register_fixture(self, config: TestDataConfig, data: Any):
        """Register a test fixture."""
        self.data_configs[config.name] = config
        self.fixtures[config.name] = data
        
    def generate_data(self, factory_name: str, count: int = 1, **kwargs) -> Any:
        """Generate test data using a factory."""
        if factory_name not in self.factories:
            raise ValueError(f"Factory '{factory_name}' not registered")
            
        factory = self.factories[factory_name]
        return factory.generate(count, **kwargs)
        
    def get_fixture(self, name: str) -> Any:
        """Get a test fixture."""
        if name not in self.fixtures:
            raise ValueError(f"Fixture '{name}' not found")
            
        return self.fixtures[name]
        
    def create_temp_directory(self, prefix: str = "echo_test_") -> str:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.temp_directories.append(temp_dir)
        return temp_dir
        
    def create_temp_file(
        self,
        content: str = "",
        suffix: str = ".txt",
        prefix: str = "test_"
    ) -> str:
        """Create a temporary file for testing."""
        temp_dir = self.create_temp_directory()
        temp_file = os.path.join(temp_dir, f"{prefix}{os.getpid()}{suffix}")
        
        with open(temp_file, 'w') as f:
            f.write(content)
            
        return temp_file
        
    def save_fixture_to_file(self, name: str, file_path: str):
        """Save a fixture to a file."""
        if name not in self.fixtures:
            raise ValueError(f"Fixture '{name}' not found")
            
        data = self.fixtures[name]
        
        with open(file_path, 'w') as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2, default=str)
            else:
                f.write(str(data))
                
    def load_fixture_from_file(self, name: str, file_path: str, config: TestDataConfig):
        """Load a fixture from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fixture file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
            else:
                data = f.read()
                
        self.register_fixture(config, data)
        
    def cleanup(self):
        """Clean up temporary resources."""
        for temp_dir in self.temp_directories:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                self.framework.logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
                
        self.temp_directories.clear()


class DatabaseManager:
    """Manages database operations for testing."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize database manager."""
        self.framework = framework
        self.connections: Dict[str, Any] = {}
        self.snapshots: Dict[str, DatabaseSnapshot] = {}
        
    def create_connection(
        self,
        name: str,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str
    ):
        """Create a database connection."""
        try:
            connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
            self.connections[name] = connection
            self.framework.logger.info(f"Database connection '{name}' created")
            
        except Exception as e:
            self.framework.logger.error(f"Failed to create database connection '{name}': {e}")
            raise
            
    def execute_sql(self, connection_name: str, sql: str, params: Optional[tuple] = None) -> List[tuple]:
        """Execute SQL query."""
        if connection_name not in self.connections:
            raise ValueError(f"Connection '{connection_name}' not found")
            
        connection = self.connections[connection_name]
        cursor = connection.cursor()
        
        try:
            cursor.execute(sql, params)
            
            if sql.strip().upper().startswith(('SELECT', 'WITH')):
                return cursor.fetchall()
            else:
                connection.commit()
                return []
                
        except Exception as e:
            connection.rollback()
            raise
        finally:
            cursor.close()
            
    def seed_table(
        self,
        connection_name: str,
        table_name: str,
        data: List[Dict[str, Any]]
    ):
        """Seed a table with test data."""
        if not data:
            return
            
        # Build insert statement
        columns = list(data[0].keys())
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Convert data to tuples
        values = [tuple(row[col] for col in columns) for row in data]
        
        connection = self.connections[connection_name]
        cursor = connection.cursor()
        
        try:
            cursor.executemany(sql, values)
            connection.commit()
            self.framework.logger.info(f"Seeded {len(data)} rows in table '{table_name}'")
            
        except Exception as e:
            connection.rollback()
            self.framework.logger.error(f"Failed to seed table '{table_name}': {e}")
            raise
        finally:
            cursor.close()
            
    def create_snapshot(self, connection_name: str, snapshot_id: str, tables: List[str]) -> DatabaseSnapshot:
        """Create a database snapshot."""
        table_data = {}
        
        for table in tables:
            rows = self.execute_sql(connection_name, f"SELECT * FROM {table}")
            
            # Get column names
            column_sql = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position
            """
            columns = [row[0] for row in self.execute_sql(connection_name, column_sql, (table,))]
            
            # Convert rows to dictionaries
            table_data[table] = [
                dict(zip(columns, row)) for row in rows
            ]
            
        snapshot = DatabaseSnapshot(
            snapshot_id=snapshot_id,
            database_name=connection_name,
            timestamp=datetime.now(),
            table_data=table_data,
            schema_version="1.0"
        )
        
        self.snapshots[snapshot_id] = snapshot
        return snapshot
        
    def restore_snapshot(self, snapshot_id: str, connection_name: str):
        """Restore database from snapshot."""
        if snapshot_id not in self.snapshots:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")
            
        snapshot = self.snapshots[snapshot_id]
        
        # Clear existing data
        for table in snapshot.table_data:
            self.execute_sql(connection_name, f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
            
        # Restore data
        for table, rows in snapshot.table_data.items():
            if rows:
                self.seed_table(connection_name, table, rows)
                
        self.framework.logger.info(f"Database restored from snapshot '{snapshot_id}'")
        
    def cleanup(self):
        """Clean up database connections."""
        for name, connection in self.connections.items():
            try:
                connection.close()
                self.framework.logger.info(f"Database connection '{name}' closed")
            except Exception as e:
                self.framework.logger.warning(f"Failed to close connection '{name}': {e}")
                
        self.connections.clear()


class TestDataManager:
    """Main test data management system."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize test data manager."""
        self.framework = framework
        self.fixture_manager = FixtureManager(framework)
        self.database_manager = DatabaseManager(framework)
        self.data_directory = Path("tests/data")
        self.data_directory.mkdir(exist_ok=True)
        
    def setup_test_environment(self, config: Dict[str, Any]):
        """Setup complete test environment."""
        self.framework.logger.info("Setting up test environment")
        
        # Setup fixtures
        if 'fixtures' in config:
            for fixture_config in config['fixtures']:
                self._setup_fixture(fixture_config)
                
        # Setup databases
        if 'databases' in config:
            for db_config in config['databases']:
                self._setup_database(db_config)
                
        # Generate test data
        if 'generate_data' in config:
            for data_config in config['generate_data']:
                self._generate_test_data(data_config)
                
    def _setup_fixture(self, config: Dict[str, Any]):
        """Setup a single fixture."""
        name = config['name']
        
        if config['type'] == 'generated':
            factory_name = config['factory']
            data = self.fixture_manager.generate_data(
                factory_name,
                count=config.get('count', 1),
                **config.get('params', {})
            )
        elif config['type'] == 'file':
            file_path = config['file_path']
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unknown fixture type: {config['type']}")
            
        fixture_config = TestDataConfig(
            name=name,
            data_type='fixture',
            source_type=config['type'],
            description=config.get('description', '')
        )
        
        self.fixture_manager.register_fixture(fixture_config, data)
        
    def _setup_database(self, config: Dict[str, Any]):
        """Setup a database connection and data."""
        name = config['name']
        
        self.database_manager.create_connection(
            name=name,
            host=config['host'],
            port=config['port'],
            database=config['database'],
            username=config['username'],
            password=config['password']
        )
        
        # Execute setup SQL
        if 'setup_sql' in config:
            for sql_file in config['setup_sql']:
                with open(sql_file, 'r') as f:
                    sql = f.read()
                self.database_manager.execute_sql(name, sql)
                
        # Seed tables
        if 'seed_data' in config:
            for table_config in config['seed_data']:
                table_name = table_config['table']
                factory_name = table_config['factory']
                count = table_config.get('count', 10)
                
                data = self.fixture_manager.generate_data(
                    factory_name,
                    count=count,
                    **table_config.get('params', {})
                )
                
                if not isinstance(data, list):
                    data = [data]
                    
                self.database_manager.seed_table(name, table_name, data)
                
    def _generate_test_data(self, config: Dict[str, Any]):
        """Generate and save test data."""
        factory_name = config['factory']
        output_file = config['output_file']
        count = config.get('count', 10)
        
        data = self.fixture_manager.generate_data(
            factory_name,
            count=count,
            **config.get('params', {})
        )
        
        output_path = self.data_directory / output_file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.framework.logger.info(f"Generated test data saved to {output_path}")
        
    def cleanup(self):
        """Clean up all test resources."""
        self.fixture_manager.cleanup()
        self.database_manager.cleanup()


# Export main classes
__all__ = [
    'TestDataManager',
    'FixtureManager',
    'DatabaseManager',
    'DataFactory',
    'UserDataFactory',
    'TaskDataFactory',
    'AIModelDataFactory',
    'TestDataConfig',
    'DatabaseSnapshot'
]
