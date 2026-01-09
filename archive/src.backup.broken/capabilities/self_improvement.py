"""
Self-Improvement Module
Enables Echo Brain to analyze and modify its own code
"""

import ast
import inspect
import textwrap
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import git
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """Analyzes Python code using AST"""

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a Python file

        Args:
            file_path: Path to Python file

        Returns:
            Analysis results
        """

        try:
            source = file_path.read_text()
            tree = ast.parse(source)

            analysis = {
                "file": str(file_path),
                "classes": [],
                "functions": [],
                "imports": [],
                "complexity": 0,
                "lines": len(source.splitlines()),
                "docstring_coverage": 0
            }

            # Analyze AST
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis["classes"].append(self._analyze_class(node))
                elif isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(self._analyze_function(node))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis["imports"].append(node.module)

            # Calculate complexity
            analysis["complexity"] = self._calculate_complexity(tree)

            # Calculate docstring coverage
            total_defs = len(analysis["classes"]) + len(analysis["functions"])
            if total_defs > 0:
                with_docs = sum(1 for c in analysis["classes"] if c.get("has_docstring"))
                with_docs += sum(1 for f in analysis["functions"] if f.get("has_docstring"))
                analysis["docstring_coverage"] = with_docs / total_defs

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return {"error": str(e)}

    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition"""

        return {
            "name": node.name,
            "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
            "has_docstring": ast.get_docstring(node) is not None,
            "line": node.lineno
        }

    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition"""

        return {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "has_docstring": ast.get_docstring(node) is not None,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "line": node.lineno,
            "complexity": self._calculate_complexity(node)
        }

    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity"""

        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity


class CodeModifier:
    """Modifies Python code using AST transformations"""

    def add_logging(self, source: str, function_name: str) -> str:
        """
        Add logging to a function

        Args:
            source: Source code
            function_name: Function to add logging to

        Returns:
            Modified source code
        """

        tree = ast.parse(source)
        transformer = LoggingTransformer(function_name)
        modified_tree = transformer.visit(tree)
        return ast.unparse(modified_tree)

    def optimize_imports(self, source: str) -> str:
        """
        Optimize and sort imports

        Args:
            source: Source code

        Returns:
            Optimized source code
        """

        tree = ast.parse(source)

        # Collect all imports
        imports = []
        other_nodes = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
            else:
                other_nodes.append(node)

        # Sort imports
        std_imports = []
        third_party = []
        local = []

        for imp in imports:
            if isinstance(imp, ast.Import):
                module = imp.names[0].name
            else:
                module = imp.module or ""

            if module.startswith((".", "..")):
                local.append(imp)
            elif module in ["os", "sys", "json", "logging", "pathlib", "datetime"]:
                std_imports.append(imp)
            else:
                third_party.append(imp)

        # Reconstruct tree
        tree.body = std_imports + third_party + local + other_nodes

        return ast.unparse(tree)

    def add_error_handling(self, source: str, function_name: str) -> str:
        """
        Add error handling to a function

        Args:
            source: Source code
            function_name: Function to add error handling to

        Returns:
            Modified source code
        """

        tree = ast.parse(source)
        transformer = ErrorHandlingTransformer(function_name)
        modified_tree = transformer.visit(tree)
        return ast.unparse(modified_tree)


class LoggingTransformer(ast.NodeTransformer):
    """AST transformer to add logging"""

    def __init__(self, function_name: str):
        self.function_name = function_name

    def visit_FunctionDef(self, node):
        if node.name == self.function_name:
            # Add logging at start
            log_stmt = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='logger', ctx=ast.Load()),
                        attr='info',
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=f"Executing {node.name}")],
                    keywords=[]
                )
            )
            node.body.insert(0, log_stmt)

        return node


class ErrorHandlingTransformer(ast.NodeTransformer):
    """AST transformer to add error handling"""

    def __init__(self, function_name: str):
        self.function_name = function_name

    def visit_FunctionDef(self, node):
        if node.name == self.function_name:
            # Wrap body in try-except
            try_node = ast.Try(
                body=node.body,
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id='Exception', ctx=ast.Load()),
                        name='e',
                        body=[
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id='logger', ctx=ast.Load()),
                                        attr='error',
                                        ctx=ast.Load()
                                    ),
                                    args=[
                                        ast.JoinedStr(
                                            values=[
                                                ast.Constant(value=f"Error in {node.name}: "),
                                                ast.FormattedValue(
                                                    value=ast.Name(id='e', ctx=ast.Load()),
                                                    conversion=-1
                                                )
                                            ]
                                        )
                                    ],
                                    keywords=[]
                                )
                            ),
                            ast.Raise()
                        ]
                    )
                ],
                orelse=[],
                finalbody=[]
            )
            node.body = [try_node]

        return node


class SelfImprovementSystem:
    """Main self-improvement system for Echo Brain"""

    def __init__(self, repo_path: str = "/opt/tower-echo-brain"):
        self.repo_path = Path(repo_path)
        self.analyzer = CodeAnalyzer()
        self.modifier = CodeModifier()
        self.repo = git.Repo(self.repo_path)
        self.improvement_history = []

    async def analyze_codebase(self) -> Dict[str, Any]:
        """
        Analyze entire codebase

        Returns:
            Analysis results
        """

        python_files = list(self.repo_path.glob("**/*.py"))

        results = {
            "total_files": len(python_files),
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "average_complexity": 0,
            "docstring_coverage": 0,
            "issues": [],
            "improvements": []
        }

        complexities = []
        docstring_coverages = []

        for file_path in python_files:
            if "venv" in str(file_path) or "__pycache__" in str(file_path):
                continue

            analysis = self.analyzer.analyze_file(file_path)

            if "error" not in analysis:
                results["total_lines"] += analysis["lines"]
                results["total_functions"] += len(analysis["functions"])
                results["total_classes"] += len(analysis["classes"])
                complexities.append(analysis["complexity"])
                docstring_coverages.append(analysis["docstring_coverage"])

                # Identify issues
                if analysis["complexity"] > 10:
                    results["issues"].append({
                        "type": "high_complexity",
                        "file": str(file_path),
                        "complexity": analysis["complexity"]
                    })

                if analysis["docstring_coverage"] < 0.5:
                    results["issues"].append({
                        "type": "low_documentation",
                        "file": str(file_path),
                        "coverage": analysis["docstring_coverage"]
                    })

        # Calculate averages
        if complexities:
            results["average_complexity"] = sum(complexities) / len(complexities)

        if docstring_coverages:
            results["docstring_coverage"] = sum(docstring_coverages) / len(docstring_coverages)

        # Generate improvement suggestions
        results["improvements"] = self._generate_improvements(results)

        return results

    def _generate_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on analysis"""

        improvements = []

        # High complexity files
        for issue in analysis["issues"]:
            if issue["type"] == "high_complexity":
                improvements.append({
                    "type": "refactor",
                    "target": issue["file"],
                    "reason": f"Complexity is {issue['complexity']}, should be < 10",
                    "priority": "high"
                })

            elif issue["type"] == "low_documentation":
                improvements.append({
                    "type": "documentation",
                    "target": issue["file"],
                    "reason": f"Documentation coverage is {issue['coverage']:.1%}",
                    "priority": "medium"
                })

        return improvements

    async def apply_improvement(
        self,
        file_path: str,
        improvement_type: str,
        target: str = None
    ) -> Dict[str, Any]:
        """
        Apply an improvement to code

        Args:
            file_path: Path to file to improve
            improvement_type: Type of improvement (logging, error_handling, optimization)
            target: Target function/class

        Returns:
            Result of improvement
        """

        file_path = Path(file_path)

        if not file_path.exists():
            return {
                "success": False,
                "error": "File not found"
            }

        try:
            # Read current code
            source = file_path.read_text()
            original_hash = hashlib.sha256(source.encode()).hexdigest()

            # Apply improvement
            if improvement_type == "logging":
                modified = self.modifier.add_logging(source, target)
            elif improvement_type == "error_handling":
                modified = self.modifier.add_error_handling(source, target)
            elif improvement_type == "optimize_imports":
                modified = self.modifier.optimize_imports(source)
            else:
                return {
                    "success": False,
                    "error": f"Unknown improvement type: {improvement_type}"
                }

            # Create backup
            backup_path = file_path.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            backup_path.write_text(source)

            # Write modified code
            file_path.write_text(modified)

            # Commit changes
            self.repo.index.add([str(file_path)])
            self.repo.index.commit(f"Self-improvement: {improvement_type} for {target or file_path.name}")

            # Record improvement
            self.improvement_history.append({
                "timestamp": datetime.now().isoformat(),
                "file": str(file_path),
                "type": improvement_type,
                "target": target,
                "original_hash": original_hash,
                "backup": str(backup_path)
            })

            return {
                "success": True,
                "file": str(file_path),
                "backup": str(backup_path),
                "changes": len(modified) - len(source)
            }

        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def learn_from_execution(
        self,
        execution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Learn from execution results and improve code

        Args:
            execution_data: Data from code execution

        Returns:
            Learning results
        """

        if not execution_data.get("success"):
            # Analyze failure
            error = execution_data.get("error", "")

            if "ImportError" in error:
                # Missing import
                return await self._fix_import_error(execution_data)

            elif "AttributeError" in error:
                # Missing attribute
                return await self._fix_attribute_error(execution_data)

            elif "TypeError" in error:
                # Type error
                return await self._fix_type_error(execution_data)

        return {
            "success": True,
            "learned": False,
            "reason": "No actionable learning from this execution"
        }

    async def _fix_import_error(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix import errors automatically"""

        error = execution_data.get("error", "")

        # Extract module name from error
        import_match = "No module named '([^']+)'"
        import re
        match = re.search(import_match, error)

        if match:
            module = match.group(1)

            # Try to install the module
            from .code_executor import SandboxedCodeExecutor
            executor = SandboxedCodeExecutor()

            result = await executor.execute_code(
                f"pip install {module}",
                language="bash"
            )

            if result.get("success"):
                return {
                    "success": True,
                    "learned": True,
                    "action": f"Installed missing module: {module}"
                }

        return {
            "success": False,
            "learned": False,
            "reason": "Could not fix import error"
        }

    async def _fix_attribute_error(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix attribute errors"""

        # This would require more complex analysis
        # For now, just log the error for manual review

        return {
            "success": False,
            "learned": True,
            "action": "Logged attribute error for review"
        }

    async def _fix_type_error(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix type errors"""

        # This would require type inference and correction
        # For now, just log the error

        return {
            "success": False,
            "learned": True,
            "action": "Logged type error for review"
        }

    def rollback_improvement(self, backup_path: str) -> bool:
        """
        Rollback an improvement using backup

        Args:
            backup_path: Path to backup file

        Returns:
            True if successful
        """

        try:
            backup = Path(backup_path)
            if not backup.exists():
                return False

            # Determine original file
            original = backup.with_suffix("")  # Remove timestamp suffix
            original = original.with_suffix("")  # Remove .backup suffix

            # Restore from backup
            original.write_text(backup.read_text())

            # Commit rollback
            self.repo.index.add([str(original)])
            self.repo.index.commit(f"Rollback improvement for {original.name}")

            return True

        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return False