"""
Code Intelligence for Echo Brain
Parses and indexes the Tower codebase for semantic understanding.
NOT text search - actual code comprehension.
"""

import ast
import asyncio
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import asyncpg
import logging
from datetime import datetime

from .schemas import (
    CodeFile, CodeSymbol, CodeDependency, APIEndpoint,
    DependencyGraph, CodeIssue, CodeLocation, SymbolType
)

logger = logging.getLogger(__name__)


class PythonCodeAnalyzer:
    """Analyzes Python source code using AST"""

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a Python file and extract semantic information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)

            symbols = []
            dependencies = []
            endpoints = []
            issues = []

            # Extract symbols
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.append({
                        'name': node.name,
                        'type': SymbolType.FUNCTION,
                        'line_start': node.lineno,
                        'line_end': node.end_lineno or node.lineno,
                        'signature': self._get_function_signature(node),
                        'docstring': ast.get_docstring(node),
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    })

                    # Check for FastAPI endpoints
                    if self._is_fastapi_endpoint(node):
                        endpoint_info = self._extract_endpoint_info(node)
                        if endpoint_info:
                            endpoints.append(endpoint_info)

                elif isinstance(node, ast.ClassDef):
                    symbols.append({
                        'name': node.name,
                        'type': SymbolType.CLASS,
                        'line_start': node.lineno,
                        'line_end': node.end_lineno or node.lineno,
                        'signature': f"class {node.name}",
                        'docstring': ast.get_docstring(node),
                        'base_classes': [self._get_name(base) for base in node.bases]
                    })

                    # Extract methods
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            symbols.append({
                                'name': f"{node.name}.{item.name}",
                                'type': SymbolType.METHOD,
                                'line_start': item.lineno,
                                'line_end': item.end_lineno or item.lineno,
                                'signature': self._get_function_signature(item),
                                'docstring': ast.get_docstring(item),
                                'parent_class': node.name,
                                'is_async': isinstance(item, ast.AsyncFunctionDef)
                            })

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append({
                            'module': alias.name,
                            'names': [alias.name],
                            'is_relative': False,
                            'line': node.lineno
                        })

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        names = [alias.name for alias in node.names]
                        dependencies.append({
                            'module': node.module,
                            'names': names,
                            'is_relative': node.level > 0,
                            'line': node.lineno
                        })

            return {
                'symbols': symbols,
                'dependencies': dependencies,
                'endpoints': endpoints,
                'issues': issues,
                'line_count': len(content.splitlines())
            }

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'symbols': [],
                'dependencies': [],
                'endpoints': [],
                'issues': [{'severity': 'error', 'message': str(e), 'line': 1}],
                'line_count': 0
            }

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []

        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation_string(arg.annotation)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            vararg = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg += f": {self._get_annotation_string(node.args.vararg.annotation)}"
            args.append(vararg)

        # **kwargs
        if node.args.kwarg:
            kwarg = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg += f": {self._get_annotation_string(node.args.kwarg.annotation)}"
            args.append(kwarg)

        signature = f"{node.name}({', '.join(args)})"

        # Return annotation
        if node.returns:
            signature += f" -> {self._get_annotation_string(node.returns)}"

        return signature

    def _get_annotation_string(self, annotation) -> str:
        """Convert annotation AST to string"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return repr(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{self._get_annotation_string(annotation.value)}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            value = self._get_annotation_string(annotation.value)
            slice_val = self._get_annotation_string(annotation.slice)
            return f"{value}[{slice_val}]"
        else:
            return "Any"

    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)

    def _is_fastapi_endpoint(self, node: ast.FunctionDef) -> bool:
        """Check if function is a FastAPI endpoint"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
                        return True
                elif isinstance(decorator.func, ast.Name):
                    if decorator.func.id in ['get', 'post', 'put', 'delete', 'patch']:
                        return True
        return False

    def _extract_endpoint_info(self, node: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Extract FastAPI endpoint information"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                method = decorator.func.attr.upper()

                # Extract path from first argument
                if decorator.args and isinstance(decorator.args[0], ast.Constant):
                    path = decorator.args[0].value

                    # Extract parameters from function signature
                    params = {}
                    for arg in node.args.args[1:]:  # Skip 'self' or first arg
                        param_info = {'name': arg.arg}
                        if arg.annotation:
                            param_info['type'] = self._get_annotation_string(arg.annotation)
                        params[arg.arg] = param_info

                    return {
                        'method': method,
                        'path': path,
                        'function': node.name,
                        'parameters': params,
                        'line': node.lineno
                    }
        return None


class CodeIntelligence:
    """
    Parses and indexes the Tower codebase for semantic understanding.
    NOT text search - actual code comprehension.
    """

    def __init__(self, db_config: Dict[str, str] = None):
        self.db_config = db_config or {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": ""
        }
        self._pool = None
        self.analyzer = PythonCodeAnalyzer()

    async def get_db_pool(self):
        """Get or create database connection pool"""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=2,
                max_size=10,
                timeout=10
            )
        return self._pool

    async def close(self):
        """Clean up connections"""
        if self._pool:
            await self._pool.close()

    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    def _should_index_file(self, file_path: str) -> bool:
        """Check if file should be indexed"""
        path = Path(file_path)

        # Skip hidden files and directories
        if any(part.startswith('.') for part in path.parts):
            return False

        # Only Python files for now
        if path.suffix != '.py':
            return False

        # Skip __pycache__ and test files for initial implementation
        if '__pycache__' in str(path) or 'test' in path.name.lower():
            return False

        return True

    async def index_codebase(self, paths: List[str]) -> Dict[str, Any]:
        """
        Parse Python files into structured knowledge:
        - Classes and their methods
        - Functions and their signatures
        - Import dependencies
        - API endpoints (FastAPI routes)
        - Configuration files
        """
        pool = await self.get_db_pool()
        results = {
            'files_processed': 0,
            'symbols_indexed': 0,
            'endpoints_found': 0,
            'dependencies_mapped': 0,
            'errors': []
        }

        try:
            async with pool.acquire() as conn:
                for base_path in paths:
                    if os.path.isfile(base_path):
                        if self._should_index_file(base_path):
                            await self._index_file(conn, base_path, results)
                    elif os.path.isdir(base_path):
                        for root, dirs, files in os.walk(base_path):
                            # Remove hidden directories from dirs to prevent os.walk from entering them
                            dirs[:] = [d for d in dirs if not d.startswith('.')]

                            for file in files:
                                file_path = os.path.join(root, file)
                                if self._should_index_file(file_path):
                                    await self._index_file(conn, file_path, results)

        except Exception as e:
            logger.error(f"Error during codebase indexing: {e}")
            results['errors'].append(str(e))

        return results

    async def _index_file(self, conn, file_path: str, results: Dict[str, Any]):
        """Index a single file"""
        try:
            # Check if file needs re-indexing
            current_hash = self._calculate_hash(file_path)

            existing = await conn.fetchrow(
                "SELECT id, content_hash FROM code_files WHERE path = $1",
                file_path
            )

            if existing and existing['content_hash'] == current_hash:
                logger.debug(f"File {file_path} already up to date")
                return

            # Analyze the file
            analysis = self.analyzer.analyze_file(file_path)

            if existing:
                file_id = existing['id']
                # Update existing file
                await conn.execute(
                    """UPDATE code_files
                       SET content_hash = $2, last_indexed = $3, line_count = $4
                       WHERE id = $1""",
                    file_id, current_hash, datetime.now(), analysis['line_count']
                )

                # Clear existing symbols, dependencies, and endpoints
                await conn.execute("DELETE FROM code_symbols WHERE file_id = $1", file_id)
                await conn.execute("DELETE FROM code_dependencies WHERE from_file_id = $1", file_id)
                await conn.execute("DELETE FROM code_endpoints WHERE file_id = $1", file_id)
            else:
                # Insert new file
                file_id = await conn.fetchval(
                    """INSERT INTO code_files (path, content_hash, last_indexed, line_count, language)
                       VALUES ($1, $2, $3, $4, 'python') RETURNING id""",
                    file_path, current_hash, datetime.now(), analysis['line_count']
                )

            # Insert symbols
            for symbol in analysis['symbols']:
                await conn.execute(
                    """INSERT INTO code_symbols
                       (file_id, name, symbol_type, line_start, line_end, signature, docstring)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    file_id, symbol['name'], symbol['type'], symbol['line_start'],
                    symbol['line_end'], symbol.get('signature'), symbol.get('docstring')
                )
                results['symbols_indexed'] += 1

            # Insert dependencies
            for dep in analysis['dependencies']:
                await conn.execute(
                    """INSERT INTO code_dependencies
                       (from_file_id, to_module, import_names, is_relative)
                       VALUES ($1, $2, $3, $4)""",
                    file_id, dep['module'], dep['names'], dep['is_relative']
                )
                results['dependencies_mapped'] += 1

            # Insert endpoints
            for endpoint in analysis['endpoints']:
                await conn.execute(
                    """INSERT INTO code_endpoints
                       (file_id, http_method, path_pattern, function_name, parameters)
                       VALUES ($1, $2, $3, $4, $5)""",
                    file_id, endpoint['method'], endpoint['path'],
                    endpoint['function'], endpoint['parameters']
                )
                results['endpoints_found'] += 1

            results['files_processed'] += 1
            logger.info(f"Indexed {file_path}: {len(analysis['symbols'])} symbols, {len(analysis['dependencies'])} deps")

        except Exception as e:
            error_msg = f"Error indexing {file_path}: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

    async def find_definition(self, name: str) -> List[CodeLocation]:
        """Find where a function/class is defined"""
        pool = await self.get_db_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT cf.path, cs.line_start, cs.line_end, cs.symbol_type
                   FROM code_symbols cs
                   JOIN code_files cf ON cs.file_id = cf.id
                   WHERE cs.name = $1 OR cs.name LIKE $2
                   ORDER BY cs.symbol_type, cf.path""",
                name, f"%.{name}"
            )

            return [
                CodeLocation(
                    file_path=row['path'],
                    line_start=row['line_start'],
                    line_end=row['line_end']
                )
                for row in rows
            ]

    async def get_callers(self, function_name: str) -> List[CodeLocation]:
        """Find all places that call this function (basic implementation)"""
        # This would require more sophisticated analysis of function calls
        # For now, return empty list - can be enhanced later
        return []

    async def get_dependencies(self, file_path: str) -> DependencyGraph:
        """What does this file import/depend on"""
        pool = await self.get_db_pool()

        async with pool.acquire() as conn:
            # Get file ID
            file_record = await conn.fetchrow(
                "SELECT id FROM code_files WHERE path = $1", file_path
            )

            if not file_record:
                return DependencyGraph(file_path=file_path, dependencies=[], dependents=[])

            file_id = file_record['id']

            # Get dependencies (what this file imports)
            dep_rows = await conn.fetch(
                "SELECT to_module FROM code_dependencies WHERE from_file_id = $1",
                file_id
            )
            dependencies = [row['to_module'] for row in dep_rows]

            # Get dependents (what files import this)
            # This is simplified - would need module name mapping for full accuracy
            filename = os.path.splitext(os.path.basename(file_path))[0]
            dep_rows = await conn.fetch(
                """SELECT cf.path FROM code_dependencies cd
                   JOIN code_files cf ON cd.from_file_id = cf.id
                   WHERE cd.to_module LIKE $1""",
                f"%{filename}%"
            )
            dependents = [row['path'] for row in dep_rows]

            return DependencyGraph(
                file_path=file_path,
                dependencies=dependencies,
                dependents=dependents
            )

    async def find_endpoints(self) -> List[APIEndpoint]:
        """Find all FastAPI/Flask routes"""
        pool = await self.get_db_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT ce.*, cf.path
                   FROM code_endpoints ce
                   JOIN code_files cf ON ce.file_id = cf.id
                   ORDER BY cf.path, ce.path_pattern"""
            )

            return [
                APIEndpoint(
                    id=row['id'],
                    file_id=row['file_id'],
                    http_method=row['http_method'],
                    path_pattern=row['path_pattern'],
                    function_name=row['function_name'],
                    parameters=row['parameters'] or {}
                )
                for row in rows
            ]

    async def analyze_for_issues(self, file_path: str) -> List[CodeIssue]:
        """Static analysis for common problems"""
        # Basic implementation - can be enhanced with more sophisticated analysis
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                # Check for common issues
                if 'TODO' in line or 'FIXME' in line:
                    issues.append(CodeIssue(
                        severity='info',
                        message='TODO/FIXME comment found',
                        location=CodeLocation(file_path=file_path, line_start=i, line_end=i),
                        suggestion='Consider addressing this TODO item'
                    ))

                if 'print(' in line and not line.strip().startswith('#'):
                    issues.append(CodeIssue(
                        severity='warning',
                        message='Debug print statement found',
                        location=CodeLocation(file_path=file_path, line_start=i, line_end=i),
                        suggestion='Consider using logging instead of print'
                    ))

        except Exception as e:
            issues.append(CodeIssue(
                severity='error',
                message=f'Error analyzing file: {e}',
                location=CodeLocation(file_path=file_path, line_start=1, line_end=1)
            ))

        return issues

    async def search_symbols(self, query: str, symbol_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search code symbols by name"""
        pool = await self.get_db_pool()

        conditions = ["cs.name ILIKE $1"]
        params = [f"%{query}%"]

        if symbol_type:
            conditions.append("cs.symbol_type = $2")
            params.append(symbol_type)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT cs.*, cf.path
                   FROM code_symbols cs
                   JOIN code_files cf ON cs.file_id = cf.id
                   WHERE {' AND '.join(conditions)}
                   ORDER BY cs.name
                   LIMIT 50""",
                *params
            )

            return [
                {
                    'name': row['name'],
                    'type': row['symbol_type'],
                    'file_path': row['path'],
                    'line_start': row['line_start'],
                    'line_end': row['line_end'],
                    'signature': row['signature'],
                    'docstring': row['docstring']
                }
                for row in rows
            ]


# Singleton instance
_code_intelligence = None

def get_code_intelligence() -> CodeIntelligence:
    """Get or create singleton instance"""
    global _code_intelligence
    if not _code_intelligence:
        _code_intelligence = CodeIntelligence()
    return _code_intelligence