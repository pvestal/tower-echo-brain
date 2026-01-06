#!/usr/bin/env python3
"""
Echo Automated Test Generator
Generates comprehensive unit tests for approved code changes
Integrates with Echo's learning system for continuous improvement
"""

import os
import ast
import json
import asyncio
import sqlite3
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import importlib.util
from dataclasses import dataclass
import coverage

@dataclass
class TestCase:
    """Represents a generated test case"""
    name: str
    description: str
    test_code: str
    test_type: str  # unit, integration, functional
    complexity: str  # simple, medium, complex
    dependencies: List[str]
    expected_outcome: str

@dataclass
class TestSuite:
    """Collection of test cases for a module"""
    module_name: str
    file_path: str
    test_cases: List[TestCase]
    coverage_target: float
    priority: str

class CodeAnalyzer:
    """Analyzes code structure to generate appropriate tests"""
    
    def __init__(self):
        self.function_signatures = {}
        self.class_definitions = {}
        self.imports = []
        self.complexity_scores = {}
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file and extract testable components"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'complexity': 0,
                'testable_units': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node)
                    analysis['functions'].append(func_info)
                    if not func_info['name'].startswith('_'):  # Public functions
                        analysis['testable_units'].append(func_info)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    analysis['classes'].append(class_info)
                    analysis['testable_units'].extend(class_info['methods'])
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis['imports'].append(self._get_import_info(node))
            
            analysis['complexity'] = self._calculate_complexity(tree)
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {}
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function information for test generation"""
        args = []
        for arg in node.args.args:
            args.append({
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None
            })
        
        return {
            'name': node.name,
            'args': args,
            'returns': ast.unparse(node.returns) if node.returns else None,
            'docstring': ast.get_docstring(node),
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'decorators': [ast.unparse(dec) for dec in node.decorator_list],
            'complexity': self._function_complexity(node)
        }
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class information for test generation"""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._analyze_function(item))
        
        return {
            'name': node.name,
            'methods': methods,
            'bases': [ast.unparse(base) for base in node.bases],
            'docstring': ast.get_docstring(node)
        }
    
    def _get_import_info(self, node) -> Dict[str, Any]:
        """Extract import information"""
        if isinstance(node, ast.Import):
            return {
                'type': 'import',
                'names': [alias.name for alias in node.names]
            }
        else:  # ImportFrom
            return {
                'type': 'from_import',
                'module': node.module,
                'names': [alias.name for alias in node.names]
            }
    
    def _calculate_complexity(self, tree) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _function_complexity(self, node) -> int:
        """Calculate function-specific complexity"""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
        return complexity

class TestGenerator:
    """Generates test code based on analyzed components"""
    
    def __init__(self):
        self.test_templates = self._load_test_templates()
        self.analyzer = CodeAnalyzer()
    
    def generate_test_suite(self, file_path: str, module_name: str) -> TestSuite:
        """Generate complete test suite for a module"""
        analysis = self.analyzer.analyze_file(file_path)
        test_cases = []
        
        for unit in analysis.get('testable_units', []):
            test_cases.extend(self._generate_unit_tests(unit, module_name))
        
        # Generate integration tests
        test_cases.extend(self._generate_integration_tests(analysis, module_name))
        
        return TestSuite(
            module_name=module_name,
            file_path=file_path,
            test_cases=test_cases,
            coverage_target=0.85,  # 85% coverage target
            priority='high' if analysis.get('complexity', 0) > 10 else 'medium'
        )
    
    def _generate_unit_tests(self, unit: Dict[str, Any], module_name: str) -> List[TestCase]:
        """Generate unit tests for a specific function or method"""
        tests = []
        func_name = unit['name']
        
        # Basic functionality test
        tests.append(TestCase(
            name=f"test_{func_name}_basic_functionality",
            description=f"f"f"Test {func_name}" functionality"",
            test_code=self._generate_basic_test(unit, module_name),
            test_type="unit",
            complexity="simple",
            dependencies=[module_name],
            expected_outcome="passes"
        ))
        
        # Edge cases test
        if unit.get('args'):
            tests.append(TestCase(
                name=f"test_{func_name}_edge_cases",
                description=f"f"f"Test {func_name}" functionality"",
                test_code=self._generate_edge_case_test(unit, module_name),
                test_type="unit",
                complexity="medium",
                dependencies=[module_name],
                expected_outcome="passes"
            ))
        
        # Error handling test
        tests.append(TestCase(
            name=f"test_{func_name}_error_handling",
            description=f"f"f"Test {func_name}" functionality"",
            test_code=self._generate_error_test(unit, module_name),
            test_type="unit",
            complexity="medium",
            dependencies=[module_name],
            expected_outcome="passes"
        ))
        
        return tests
    
    def _generate_basic_test(self, unit: Dict[str, Any], module_name: str) -> str:
        """Generate basic functionality test code"""
        func_name = unit['name']
        args = unit.get('args', [])
        is_async = unit.get('is_async', False)
        
        # Generate mock arguments
        mock_args = []
        for arg in args:
            if arg['name'] != 'self':
                mock_args.append(self._generate_mock_arg(arg))
        
        test_code = f"""
def test_{func_name}_basic_functionality():
    """f"f"Test {func_name}" functionality""""
    # Arrange
    from {module_name} import {func_name}
    {chr(10).join([f"    {arg}" for arg in mock_args])}
    
    # Act
    {'result = await ' if is_async else 'result = '}{func_name}({', '.join([arg.split(' = ')[0] for arg in mock_args])})
    
    # Assert
    assert result is not None
    # Add specific assertions based on expected behavior
"""
        
        if is_async:
            test_code = f"""
@pytest.mark.asyncio
async {test_code.strip()}
"""
        
        return test_code.strip()
    
    def _generate_edge_case_test(self, unit: Dict[str, Any], module_name: str) -> str:
        """Generate edge case test code"""
        func_name = unit['name']
        args = unit.get('args', [])
        is_async = unit.get('is_async', False)
        
        test_code = f"""
def test_{func_name}_edge_cases():
    """f"f"Test {func_name}" functionality""""
    from {module_name} import {func_name}
    
    # Test with None values
    {'result = await ' if is_async else 'result = '}{func_name}(None)
    
    # Test with empty values
    {'result = await ' if is_async else 'result = '}{func_name}('')
    
    # Test with extreme values
    # Add specific edge case tests
"""
        
        if is_async:
            test_code = f"""
@pytest.mark.asyncio
async {test_code.strip()}
"""
        
        return test_code.strip()
    
    def _generate_error_test(self, unit: Dict[str, Any], module_name: str) -> str:
        """Generate error handling test code"""
        func_name = unit['name']
        is_async = unit.get('is_async', False)
        
        test_code = f"""
def test_{func_name}_error_handling():
    """f"f"Test {func_name}" functionality""""
    from {module_name} import {func_name}
    import pytest
    
    # Test invalid input types
    with pytest.raises((TypeError, ValueError)):
        {'await ' if is_async else ''}{func_name}("invalid_input")
    
    # Test missing required arguments
    with pytest.raises(TypeError):
        {'await ' if is_async else ''}{func_name}()
"""
        
        if is_async:
            test_code = f"""
@pytest.mark.asyncio
async {test_code.strip()}
"""
        
        return test_code.strip()
    
    def _generate_integration_tests(self, analysis: Dict[str, Any], module_name: str) -> List[TestCase]:
        """Generate integration tests"""
        tests = []
        
        if len(analysis.get('classes', [])) > 1:
            tests.append(TestCase(
                name=f"test_{module_name}_class_integration",
                description=f"Test integration between classes in {module_name}",
                test_code=self._generate_class_integration_test(analysis, module_name),
                test_type="integration",
                complexity="complex",
                dependencies=[module_name],
                expected_outcome="passes"
            ))
        
        return tests
    
    def _generate_class_integration_test(self, analysis: Dict[str, Any], module_name: str) -> str:
        """Generate class integration test"""
        classes = analysis.get('classes', [])
        if len(classes) < 2:
            return "# No integration test needed - insufficient classes"
        
        return f"""
def test_{module_name}_class_integration():
    """Test integration between classes"""
    from {module_name} import {', '.join([cls['name'] for cls in classes[:2]])}
    
    # Test class interaction
    obj1 = {classes[0]['name']}()
    obj2 = {classes[1]['name']}()
    
    # Add specific integration test logic
    assert obj1 is not None
    assert obj2 is not None
"""
    
    def _generate_mock_arg(self, arg: Dict[str, Any]) -> str:
        """Generate mock argument based on type annotation"""
        arg_name = arg['name']
        annotation = arg.get('annotation')
        
        if annotation == 'str':
            return f'{arg_name} = "test_string"'
        elif annotation == 'int':
            return f'{arg_name} = 42'
        elif annotation == 'float':
            return f'{arg_name} = 3.14'
        elif annotation == 'bool':
            return f'{arg_name} = True'
        elif annotation == 'list':
            return f'{arg_name} = ["item1", "item2"]'
        elif annotation == 'dict':
            return f'{arg_name} = {{"key": "value"}}'
        else:
            return f'{arg_name} = None  # Mock for {annotation or "unknown type"}'
    
    def _load_test_templates(self) -> Dict[str, str]:
        """Load test templates for different scenarios"""
        return {
            'basic': 'def test_{name}(): pass',
            'async': '@pytest.mark.asyncio\nasync def test_{name}(): pass',
            'parametrized': '@pytest.mark.parametrize("input,expected", [(1, 1)])\ndef test_{name}(input, expected): pass'
        }

class TestExecutor:
    """Executes generated tests and collects results"""
    
    def __init__(self):
        self.coverage = coverage.Coverage()
        self.results = {}
    
    async def run_test_suite(self, test_suite: TestSuite) -> Dict[str, Any]:
        """Execute a test suite and return results"""
        test_file_path = self._write_test_file(test_suite)
        
        try:
            # Start coverage
            self.coverage.start()
            
            # Run tests using pytest
            import subprocess
            result = subprocess.run([
                'python', '-m', 'pytest', test_file_path, '-v', '--json-report'
            ], capture_output=True, text=True, cwd=os.path.dirname(test_file_path))
            
            # Stop coverage
            self.coverage.stop()
            self.coverage.save()
            
            # Analyze results
            return {
                'test_suite': test_suite.module_name,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'coverage': self._get_coverage_report(),
                'timestamp': datetime.now().isoformat(),
                'success': result.returncode == 0
            }
            
        except Exception as e:
            return {
                'test_suite': test_suite.module_name,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def _write_test_file(self, test_suite: TestSuite) -> str:
        """Write test suite to a file"""
        test_dir = Path(f"/opt/tower-echo-brain/tests/generated")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = test_dir / f"test_{test_suite.module_name}.py"
        
        content = f"""
# Generated test file for {test_suite.module_name}
# Generated at: {datetime.now().isoformat()}

import pytest
import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

{chr(10).join([test_case.test_code for test_case in test_suite.test_cases])}
"""
        
        with open(test_file, 'w') as f:
            f.write(content)
        
        return str(test_file)
    
    def _get_coverage_report(self) -> Dict[str, Any]:
        """Get coverage report"""
        try:
            return {
                'percentage': self.coverage.report(),
                'missing_lines': {},  # Could be expanded
                'covered_lines': {}   # Could be expanded
            }
        except:
            return {'percentage': 0, 'missing_lines': {}, 'covered_lines': {}}

class EchoTestGeneratorService:
    """Main service for automated test generation"""
    
    def __init__(self):
        self.generator = TestGenerator()
        self.executor = TestExecutor()
        self.db_path = '/opt/tower-echo-brain/data/test_results.db'
        self._init_database()
    
    def _init_database(self):
        """Initialize test results database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_name TEXT,
                    test_suite_hash TEXT,
                    results TEXT,
                    coverage_percentage REAL,
                    success BOOLEAN,
                    timestamp TEXT,
                    board_decision_id TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS generated_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_name TEXT,
                    test_name TEXT,
                    test_code TEXT,
                    test_type TEXT,
                    complexity TEXT,
                    generated_at TEXT
                )
            ''')
    
    async def generate_and_run_tests(self, file_path: str, module_name: str, board_decision_id: str = None) -> Dict[str, Any]:
        """Generate tests for a file and run them"""
        try:
            # Generate test suite
            test_suite = self.generator.generate_test_suite(file_path, module_name)
            
            # Save generated tests
            self._save_generated_tests(test_suite)
            
            # Run tests
            results = await self.executor.run_test_suite(test_suite)
            
            # Save results
            self._save_test_results(test_suite, results, board_decision_id)
            
            return {
                'success': True,
                'test_suite': {
                    'module_name': test_suite.module_name,
                    'test_count': len(test_suite.test_cases),
                    'coverage_target': test_suite.coverage_target
                },
                'results': results,
                'board_decision_id': board_decision_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'module_name': module_name,
                'board_decision_id': board_decision_id
            }
    
    def _save_generated_tests(self, test_suite: TestSuite):
        """Save generated tests to database"""
        with sqlite3.connect(self.db_path) as conn:
            for test_case in test_suite.test_cases:
                conn.execute('''
                    INSERT INTO generated_tests 
                    (module_name, test_name, test_code, test_type, complexity, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    test_suite.module_name,
                    test_case.name,
                    test_case.test_code,
                    test_case.test_type,
                    test_case.complexity,
                    datetime.now().isoformat()
                ))
    
    def _save_test_results(self, test_suite: TestSuite, results: Dict[str, Any], board_decision_id: str):
        """Save test results to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO test_runs 
                (module_name, test_suite_hash, results, coverage_percentage, success, timestamp, board_decision_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_suite.module_name,
                hash(str(test_suite.test_cases)),
                json.dumps(results),
                results.get('coverage', {}).get('percentage', 0),
                results.get('success', False),
                datetime.now().isoformat(),
                board_decision_id
            ))
    
    async def get_test_history(self, module_name: str = None) -> List[Dict[str, Any]]:
        """Get test execution history"""
        with sqlite3.connect(self.db_path) as conn:
            if module_name:
                cursor = conn.execute('''
                    SELECT * FROM test_runs WHERE module_name = ? 
                    ORDER BY timestamp DESC LIMIT 50
                ''', (module_name,))
            else:
                cursor = conn.execute('''
                    SELECT * FROM test_runs ORDER BY timestamp DESC LIMIT 50
                ''')
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    async def get_coverage_metrics(self) -> Dict[str, Any]:
        """Get overall coverage metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT 
                    AVG(coverage_percentage) as avg_coverage,
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_runs
                FROM test_runs
                WHERE timestamp > datetime('now', '-7 days')
            ''')
            
            result = cursor.fetchone()
            return {
                'average_coverage': result[0] or 0,
                'total_runs': result[1] or 0,
                'successful_runs': result[2] or 0,
                'success_rate': (result[2] / result[1] * 100) if result[1] > 0 else 0
            }

# REST API endpoints for integration
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Echo Test Generator", version="1.0.0")
test_service = EchoTestGeneratorService()

class TestGenerationRequest(BaseModel):
    file_path: str
    module_name: str
    board_decision_id: Optional[str] = None

@app.post("/api/generate-tests")
async def generate_tests(request: TestGenerationRequest, background_tasks: BackgroundTasks):
    """Generate and run tests for a module"""
    try:
        result = await test_service.generate_and_run_tests(
            request.file_path,
            request.module_name,
            request.board_decision_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-history/{module_name}")
async def get_test_history(module_name: str):
    """Get test history for a module"""
    return await test_service.get_test_history(module_name)

@app.get("/api/coverage-metrics")
async def get_coverage_metrics():
    """Get overall coverage metrics"""
    return await test_service.get_coverage_metrics()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "echo_test_generator", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8340)
