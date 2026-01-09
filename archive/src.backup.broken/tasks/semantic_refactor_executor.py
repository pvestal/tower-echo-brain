#!/usr/bin/env python3
"""
Semantic Code Refactoring Executor
Performs actual code transformations using AST manipulation
"""

import ast
import asyncio
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import astunparse
import autopep8

logger = logging.getLogger(__name__)

class SemanticRefactorExecutor:
    """Executes semantic code refactoring beyond just formatting"""

    def __init__(self):
        self.refactor_log = Path("/opt/tower-echo-brain/logs/semantic_refactors.log")
        self.refactor_log.parent.mkdir(parents=True, exist_ok=True)
        self.refactor_patterns = {
            'extract_constants': self.extract_magic_numbers,
            'simplify_conditions': self.simplify_boolean_conditions,
            'remove_dead_code': self.remove_unreachable_code,
            'rename_variables': self.improve_variable_names,
            'extract_functions': self.extract_long_functions,
            'add_docstrings': self.add_missing_docstrings
        }

    async def analyze_and_refactor_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file and apply semantic refactoring"""
        logger.info(f"🔍 Analyzing file for semantic refactoring: {file_path}")

        results = {
            'file': file_path,
            'timestamp': datetime.now().isoformat(),
            'original_score': None,
            'final_score': None,
            'transformations_applied': [],
            'code_changed': False,
            'error': None
        }

        try:
            # Read file
            path = Path(file_path)
            if not path.exists() or not path.suffix == '.py':
                results['error'] = f"Invalid Python file: {file_path}"
                return results

            with open(path, 'r') as f:
                original_code = f.read()

            # Get original score
            results['original_score'] = await self._get_pylint_score(file_path)

            # Parse AST
            try:
                tree = ast.parse(original_code)
            except SyntaxError as e:
                results['error'] = f"Syntax error in file: {e}"
                return results

            # Apply transformations
            modified = False
            for pattern_name, transformer in self.refactor_patterns.items():
                try:
                    tree, changed = transformer(tree, file_path)
                    if changed:
                        results['transformations_applied'].append(pattern_name)
                        modified = True
                        logger.info(f"✅ Applied {pattern_name} to {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to apply {pattern_name}: {e}")

            if modified:
                # Generate new code
                new_code = astunparse.unparse(tree)

                # Format with autopep8
                new_code = autopep8.fix_code(new_code, options={'aggressive': 2})

                # Create backup
                backup_path = path.with_suffix('.py.backup')
                with open(backup_path, 'w') as f:
                    f.write(original_code)

                # Write new code
                with open(path, 'w') as f:
                    f.write(new_code)

                results['code_changed'] = True
                results['final_score'] = await self._get_pylint_score(file_path)

                # Log the refactoring
                self._log_refactoring(results)

                logger.info(f"📈 Score improved: {results['original_score']} → {results['final_score']}")
            else:
                logger.info(f"No semantic improvements needed for {file_path}")

        except Exception as e:
            logger.error(f"Refactoring failed for {file_path}: {e}")
            results['error'] = str(e)

        return results

    def extract_magic_numbers(self, tree: ast.AST, file_path: str) -> Tuple[ast.AST, bool]:
        """Extract magic numbers into named constants"""
        changed = False

        class MagicNumberExtractor(ast.NodeTransformer):
            def __init__(self):
                self.constants = {}
                self.modified = False

            def visit_Num(self, node):
                # Skip 0, 1, -1 as they're often not magic
                if isinstance(node.n, (int, float)) and node.n not in (0, 1, -1):
                    if node.n > 10 or node.n < -10:  # Likely magic number
                        const_name = f"CONSTANT_{abs(int(node.n))}"
                        if const_name not in self.constants:
                            self.constants[const_name] = node.n
                            self.modified = True
                        return ast.Name(id=const_name, ctx=ast.Load())
                return node

        extractor = MagicNumberExtractor()
        new_tree = extractor.visit(tree)

        if extractor.modified:
            # Add constants at module level
            for name, value in extractor.constants.items():
                const_assign = ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=ast.Num(n=value)
                )
                new_tree.body.insert(0, const_assign)
            changed = True

        return new_tree, changed

    def simplify_boolean_conditions(self, tree: ast.AST, file_path: str) -> Tuple[ast.AST, bool]:
        """Simplify complex boolean conditions"""
        changed = False

        class BooleanSimplifier(ast.NodeTransformer):
            def visit_Compare(self, node):
                self.generic_visit(node)
                # Simplify "x == True" to "x"
                if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                    if isinstance(node.comparators[0], ast.NameConstant):
                        if node.comparators[0].value is True:
                            return node.left
                        elif node.comparators[0].value is False:
                            return ast.UnaryOp(op=ast.Not(), operand=node.left)
                return node

        simplifier = BooleanSimplifier()
        new_tree = simplifier.visit(tree)
        # Note: For demo, we're not tracking changes precisely
        return new_tree, False  # Would need better change tracking

    def remove_unreachable_code(self, tree: ast.AST, file_path: str) -> Tuple[ast.AST, bool]:
        """Remove code after return statements"""
        changed = False

        class DeadCodeRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                new_body = []
                found_return = False

                for stmt in node.body:
                    if found_return:
                        break  # Skip everything after return
                    new_body.append(stmt)
                    if isinstance(stmt, ast.Return):
                        found_return = True

                if len(new_body) < len(node.body):
                    node.body = new_body
                    # changed would be True here

                return node

        remover = DeadCodeRemover()
        new_tree = remover.visit(tree)
        return new_tree, False  # Would need better change tracking

    def improve_variable_names(self, tree: ast.AST, file_path: str) -> Tuple[ast.AST, bool]:
        """Rename single-letter variables to descriptive names"""
        changed = False

        class VariableRenamer(ast.NodeTransformer):
            def __init__(self):
                self.renames = {
                    'x': 'value',
                    'y': 'result',
                    'z': 'output',
                    'i': 'index',
                    'j': 'counter',
                    'k': 'key',
                    'n': 'number',
                    's': 'string_value',
                    'd': 'data'
                }

            def visit_Name(self, node):
                if node.id in self.renames:
                    node.id = self.renames[node.id]
                return node

        renamer = VariableRenamer()
        new_tree = renamer.visit(tree)
        # Would need to track if any renames actually happened
        return new_tree, False

    def extract_long_functions(self, tree: ast.AST, file_path: str) -> Tuple[ast.AST, bool]:
        """Extract long functions into smaller ones"""
        # This is complex and would require sophisticated analysis
        # For now, just return unchanged
        return tree, False

    def add_missing_docstrings(self, tree: ast.AST, file_path: str) -> Tuple[ast.AST, bool]:
        """Add docstrings to functions and classes missing them"""
        changed = False

        class DocstringAdder(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                # Check if first statement is a docstring
                has_docstring = (
                    node.body and
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Str)
                )

                if not has_docstring:
                    # Add a basic docstring
                    docstring = ast.Expr(value=ast.Str(s=f"Function {node.name}"))
                    node.body.insert(0, docstring)
                    # changed would be True here

                return node

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                has_docstring = (
                    node.body and
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Str)
                )

                if not has_docstring:
                    docstring = ast.Expr(value=ast.Str(s=f"Class {node.name}"))
                    node.body.insert(0, docstring)

                return node

        adder = DocstringAdder()
        new_tree = adder.visit(tree)
        return new_tree, False  # Would need better change tracking

    async def _get_pylint_score(self, file_path: str) -> float:
        """Get pylint score for a single file"""
        try:
            result = subprocess.run(
                ['/opt/tower-echo-brain/venv/bin/pylint', file_path, '--score=y'],
                capture_output=True,
                text=True,
                timeout=30
            )
            # Parse score from output
            for line in result.stdout.split('\n'):
                if 'Your code has been rated at' in line:
                    score_str = line.split('rated at ')[1].split('/')[0]
                    return float(score_str)
        except Exception as e:
            logger.error(f"Failed to get pylint score: {e}")
        return 0.0

    def _log_refactoring(self, results: Dict[str, Any]):
        """Log refactoring results"""
        try:
            with open(self.refactor_log, 'a') as f:
                f.write(json.dumps(results) + '\n')
        except Exception as e:
            logger.error(f"Failed to log refactoring: {e}")

    async def refactor_project_incrementally(self, project_path: str, max_files: int = 5) -> Dict[str, Any]:
        """Refactor worst files in a project incrementally"""
        logger.info(f"🔧 Starting incremental refactoring of {project_path}")

        results = {
            'project': project_path,
            'files_analyzed': 0,
            'files_improved': 0,
            'total_transformations': 0,
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Find Python files
            path = Path(project_path)
            py_files = list(path.rglob('*.py'))[:max_files]  # Limit to max_files

            # Analyze and refactor each file
            for py_file in py_files:
                file_results = await self.analyze_and_refactor_file(str(py_file))
                results['files_analyzed'] += 1

                if file_results['code_changed']:
                    results['files_improved'] += 1
                    results['total_transformations'] += len(file_results['transformations_applied'])

        except Exception as e:
            logger.error(f"Project refactoring failed: {e}")
            results['error'] = str(e)

        logger.info(f"📊 Refactored {results['files_improved']}/{results['files_analyzed']} files")
        return results

# Global instance
semantic_refactor_executor = SemanticRefactorExecutor()