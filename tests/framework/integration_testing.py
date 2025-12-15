"""
Integration Testing Pipeline for Echo Brain Modular Architecture
================================================================

This module provides comprehensive integration testing capabilities for Echo Brain's
microservice components, including service-to-service communication, database
integration, external API testing, and multi-service workflow validation.

Features:
- Microservice communication testing
- Database integration validation
- External service mocking and testing
- End-to-end workflow testing
- Service dependency management
- Health check automation

Author: Development Testing Framework Agent
Created: 2025-11-06
"""

import asyncio
import json
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from contextlib import asynccontextmanager

import pytest
import httpx
import psycopg2
from fastapi.testclient import TestClient

from tests.framework.test_framework_core import TestFrameworkCore, TestMetrics


@dataclass
class ServiceConfig:
    """Configuration for a microservice in integration testing."""
    name: str
    host: str
    port: int
    health_endpoint: str = "/health"
    required_dependencies: List[str] = field(default_factory=list)
    startup_timeout: int = 30
    shutdown_timeout: int = 10
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class DatabaseConfig:
    """Configuration for database integration testing."""
    name: str
    host: str
    port: int
    database: str
    username: str
    password: str
    schema_file: Optional[str] = None
    test_data_file: Optional[str] = None


@dataclass
class IntegrationTestResult:
    """Result container for integration tests."""
    test_name: str
    success: bool
    duration: float
    services_tested: List[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ServiceManager:
    """Manages microservice lifecycle for integration testing."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize service manager."""
        self.framework = framework
        self.services: Dict[str, ServiceConfig] = {}
        self.running_services: Set[str] = set()
        self.client_pool: Dict[str, httpx.AsyncClient] = {}
        
    def register_service(self, config: ServiceConfig):
        """Register a service configuration."""
        self.services[config.name] = config
        
    async def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not registered")
            
        config = self.services[service_name]
        
        # Check dependencies first
        for dep in config.required_dependencies:
            if dep not in self.running_services:
                await self.start_service(dep)
                
        try:
            # Create HTTP client for service
            base_url = f"http://{config.host}:{config.port}"
            self.client_pool[service_name] = httpx.AsyncClient(base_url=base_url)
            
            # Wait for service to be ready
            await self._wait_for_service_ready(service_name, config.startup_timeout)
            
            self.running_services.add(service_name)
            self.framework.logger.info(f"Service {service_name} started successfully")
            return True
            
        except Exception as e:
            self.framework.logger.error(f"Failed to start service {service_name}: {e}")
            return False
            
    async def stop_service(self, service_name: str):
        """Stop a specific service."""
        if service_name in self.client_pool:
            await self.client_pool[service_name].aclose()
            del self.client_pool[service_name]
            
        if service_name in self.running_services:
            self.running_services.remove(service_name)
            
        self.framework.logger.info(f"Service {service_name} stopped")
        
    async def _wait_for_service_ready(self, service_name: str, timeout: int):
        """Wait for service to be ready."""
        config = self.services[service_name]
        client = self.client_pool[service_name]
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = await client.get(config.health_endpoint)
                if response.status_code == 200:
                    return
            except Exception:
                pass
                
            await asyncio.sleep(1)
            
        raise TimeoutError(f"Service {service_name} not ready within {timeout} seconds")
        
    async def health_check(self, service_name: str) -> Dict[str, Any]:
        """Perform health check on a service."""
        if service_name not in self.client_pool:
            return {"status": "not_running", "error": "Service not started"}
            
        client = self.client_pool[service_name]
        config = self.services[service_name]
        
        try:
            start_time = time.time()
            response = await client.get(config.health_endpoint)
            duration = time.time() - start_time
            
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": duration,
                "response_data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def shutdown_all(self):
        """Shutdown all running services."""
        for service_name in list(self.running_services):
            await self.stop_service(service_name)


class DatabaseManager:
    """Manages database setup and teardown for integration testing."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize database manager."""
        self.framework = framework
        self.databases: Dict[str, DatabaseConfig] = {}
        self.connections: Dict[str, Any] = {}
        
    def register_database(self, config: DatabaseConfig):
        """Register a database configuration."""
        self.databases[config.name] = config
        
    async def setup_database(self, db_name: str) -> bool:
        """Setup test database."""
        if db_name not in self.databases:
            raise ValueError(f"Database {db_name} not registered")
            
        config = self.databases[db_name]
        
        try:
            # Connect to database
            connection = psycopg2.connect(
                host=config.host,
                port=config.port,
                database=config.database,
                user=config.username,
                password=config.password
            )
            
            self.connections[db_name] = connection
            
            # Load schema if provided
            if config.schema_file:
                await self._execute_sql_file(db_name, config.schema_file)
                
            # Load test data if provided
            if config.test_data_file:
                await self._execute_sql_file(db_name, config.test_data_file)
                
            self.framework.logger.info(f"Database {db_name} setup complete")
            return True
            
        except Exception as e:
            self.framework.logger.error(f"Failed to setup database {db_name}: {e}")
            return False
            
    async def cleanup_database(self, db_name: str):
        """Clean up test database."""
        if db_name in self.connections:
            self.connections[db_name].close()
            del self.connections[db_name]
            
    async def _execute_sql_file(self, db_name: str, file_path: str):
        """Execute SQL file on database."""
        connection = self.connections[db_name]
        cursor = connection.cursor()
        
        try:
            with open(file_path, 'r') as f:
                sql_content = f.read()
                cursor.execute(sql_content)
                connection.commit()
        finally:
            cursor.close()


class IntegrationTestPipeline:
    """Main integration testing pipeline."""
    
    def __init__(self, framework: TestFrameworkCore):
        """Initialize integration test pipeline."""
        self.framework = framework
        self.service_manager = ServiceManager(framework)
        self.database_manager = DatabaseManager(framework)
        self.test_results: List[IntegrationTestResult] = []
        
    def register_service(self, config: ServiceConfig):
        """Register a service for testing."""
        self.service_manager.register_service(config)
        
    def register_database(self, config: DatabaseConfig):
        """Register a database for testing."""
        self.database_manager.register_database(config)
        
    async def run_service_communication_tests(self, test_scenarios: List[Dict[str, Any]]) -> List[IntegrationTestResult]:
        """Run service-to-service communication tests."""
        results = []
        
        for scenario in test_scenarios:
            test_name = scenario['name']
            source_service = scenario['source_service']
            target_service = scenario['target_service']
            request_data = scenario.get('request_data', {})
            expected_status = scenario.get('expected_status', 200)
            
            with self.framework.monitor_test(f"service_comm_{test_name}"):
                start_time = time.time()
                errors = []
                
                try:
                    # Ensure both services are running
                    await self.service_manager.start_service(source_service)
                    await self.service_manager.start_service(target_service)
                    
                    # Get client for source service
                    source_client = self.service_manager.client_pool[source_service]
                    
                    # Execute request
                    endpoint = scenario['endpoint']
                    method = scenario.get('method', 'GET').upper()
                    
                    if method == 'GET':
                        response = await source_client.get(endpoint, params=request_data)
                    elif method == 'POST':
                        response = await source_client.post(endpoint, json=request_data)
                    elif method == 'PUT':
                        response = await source_client.put(endpoint, json=request_data)
                    elif method == 'DELETE':
                        response = await source_client.delete(endpoint)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                        
                    # Validate response
                    success = response.status_code == expected_status
                    if not success:
                        errors.append(f"Expected status {expected_status}, got {response.status_code}")
                        
                    # Additional validations
                    if 'expected_response_keys' in scenario:
                        try:
                            response_json = response.json()
                            for key in scenario['expected_response_keys']:
                                if key not in response_json:
                                    errors.append(f"Missing expected response key: {key}")
                                    success = False
                        except Exception as e:
                            errors.append(f"Failed to parse JSON response: {e}")
                            success = False
                            
                except Exception as e:
                    success = False
                    errors.append(f"Test execution failed: {e}")
                    
                duration = time.time() - start_time
                
                result = IntegrationTestResult(
                    test_name=test_name,
                    success=success,
                    duration=duration,
                    services_tested=[source_service, target_service],
                    errors=errors
                )
                
                results.append(result)
                self.test_results.append(result)
                
        return results
        
    async def run_database_integration_tests(self, test_scenarios: List[Dict[str, Any]]) -> List[IntegrationTestResult]:
        """Run database integration tests."""
        results = []
        
        for scenario in test_scenarios:
            test_name = scenario['name']
            service_name = scenario['service']
            database_name = scenario['database']
            operation = scenario['operation']  # 'create', 'read', 'update', 'delete'
            
            with self.framework.monitor_test(f"db_integration_{test_name}"):
                start_time = time.time()
                errors = []
                
                try:
                    # Setup database and service
                    await self.database_manager.setup_database(database_name)
                    await self.service_manager.start_service(service_name)
                    
                    # Execute database operation through service
                    client = self.service_manager.client_pool[service_name]
                    endpoint = scenario['endpoint']
                    
                    if operation == 'create':
                        response = await client.post(endpoint, json=scenario['data'])
                    elif operation == 'read':
                        response = await client.get(endpoint)
                    elif operation == 'update':
                        response = await client.put(endpoint, json=scenario['data'])
                    elif operation == 'delete':
                        response = await client.delete(endpoint)
                    else:
                        raise ValueError(f"Unsupported operation: {operation}")
                        
                    # Validate response
                    expected_status = scenario.get('expected_status', 200)
                    success = response.status_code == expected_status
                    
                    if not success:
                        errors.append(f"Expected status {expected_status}, got {response.status_code}")
                        
                    # Validate database state if specified
                    if 'database_validation' in scenario:
                        db_success = await self._validate_database_state(
                            database_name,
                            scenario['database_validation']
                        )
                        if not db_success:
                            success = False
                            errors.append("Database state validation failed")
                            
                except Exception as e:
                    success = False
                    errors.append(f"Database integration test failed: {e}")
                    
                duration = time.time() - start_time
                
                result = IntegrationTestResult(
                    test_name=test_name,
                    success=success,
                    duration=duration,
                    services_tested=[service_name],
                    errors=errors
                )
                
                results.append(result)
                self.test_results.append(result)
                
        return results
        
    async def run_end_to_end_workflow_tests(self, workflows: List[Dict[str, Any]]) -> List[IntegrationTestResult]:
        """Run end-to-end workflow tests."""
        results = []
        
        for workflow in workflows:
            test_name = workflow['name']
            steps = workflow['steps']
            
            with self.framework.monitor_test(f"e2e_workflow_{test_name}"):
                start_time = time.time()
                errors = []
                services_tested = set()
                workflow_context = {}
                
                try:
                    for step_idx, step in enumerate(steps):
                        step_name = step.get('name', f"step_{step_idx}")
                        service_name = step['service']
                        services_tested.add(service_name)
                        
                        # Ensure service is running
                        await self.service_manager.start_service(service_name)
                        client = self.service_manager.client_pool[service_name]
                        
                        # Prepare request data (may use context from previous steps)
                        request_data = step.get('data', {})
                        if isinstance(request_data, str) and request_data.startswith('$'):
                            # Reference to workflow context
                            context_key = request_data[1:]
                            request_data = workflow_context.get(context_key, {})
                            
                        # Execute step
                        endpoint = step['endpoint']
                        method = step.get('method', 'GET').upper()
                        
                        if method == 'GET':
                            response = await client.get(endpoint, params=request_data)
                        elif method == 'POST':
                            response = await client.post(endpoint, json=request_data)
                        elif method == 'PUT':
                            response = await client.put(endpoint, json=request_data)
                        elif method == 'DELETE':
                            response = await client.delete(endpoint)
                            
                        # Validate step response
                        expected_status = step.get('expected_status', 200)
                        if response.status_code != expected_status:
                            errors.append(f"Step {step_name}: Expected status {expected_status}, got {response.status_code}")
                            break
                            
                        # Store response data in workflow context
                        if 'store_response_as' in step:
                            try:
                                response_data = response.json()
                                workflow_context[step['store_response_as']] = response_data
                            except Exception:
                                workflow_context[step['store_response_as']] = response.text
                                
                        # Wait between steps if specified
                        if 'wait_seconds' in step:
                            await asyncio.sleep(step['wait_seconds'])
                            
                    success = len(errors) == 0
                    
                except Exception as e:
                    success = False
                    errors.append(f"Workflow execution failed: {e}")
                    
                duration = time.time() - start_time
                
                result = IntegrationTestResult(
                    test_name=test_name,
                    success=success,
                    duration=duration,
                    services_tested=list(services_tested),
                    errors=errors,
                    metrics={'workflow_context': workflow_context}
                )
                
                results.append(result)
                self.test_results.append(result)
                
        return results
        
    async def _validate_database_state(self, db_name: str, validation_config: Dict[str, Any]) -> bool:
        """Validate database state against expected conditions."""
        try:
            connection = self.database_manager.connections[db_name]
            cursor = connection.cursor()
            
            query = validation_config['query']
            expected_result = validation_config.get('expected_result')
            
            cursor.execute(query)
            actual_result = cursor.fetchall()
            
            if expected_result is not None:
                return actual_result == expected_result
            else:
                # Just check that query executed successfully
                return True
                
        except Exception as e:
            self.framework.logger.error(f"Database validation failed: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
                
    async def run_health_check_suite(self) -> Dict[str, Any]:
        """Run comprehensive health checks on all services."""
        health_results = {}
        
        for service_name in self.service_manager.services:
            if service_name in self.service_manager.running_services:
                health_results[service_name] = await self.service_manager.health_check(service_name)
            else:
                health_results[service_name] = {"status": "not_running"}
                
        return health_results
        
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration test report."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - successful_tests
        
        avg_duration = sum(r.duration for r in self.test_results) / total_tests if total_tests > 0 else 0
        
        # Group by test type
        test_types = {}
        for result in self.test_results:
            test_type = result.test_name.split('_')[0] if '_' in result.test_name else 'unknown'
            if test_type not in test_types:
                test_types[test_type] = {'total': 0, 'successful': 0, 'failed': 0}
            test_types[test_type]['total'] += 1
            if result.success:
                test_types[test_type]['successful'] += 1
            else:
                test_types[test_type]['failed'] += 1
                
        return {
            "summary": {
                "total_tests": total_tests,
                "successful": successful_tests,
                "failed": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "avg_duration": avg_duration
            },
            "test_types": test_types,
            "failed_tests": [
                {
                    "name": r.test_name,
                    "services": r.services_tested,
                    "errors": r.errors,
                    "duration": r.duration
                }
                for r in self.test_results if not r.success
            ],
            "slowest_tests": sorted(
                self.test_results,
                key=lambda r: r.duration,
                reverse=True
            )[:5]
        }
        
    async def cleanup(self):
        """Clean up all test resources."""
        await self.service_manager.shutdown_all()
        
        for db_name in list(self.database_manager.connections.keys()):
            await self.database_manager.cleanup_database(db_name)


# Export main classes
__all__ = [
    'IntegrationTestPipeline',
    'ServiceManager',
    'DatabaseManager',
    'ServiceConfig',
    'DatabaseConfig',
    'IntegrationTestResult'
]
