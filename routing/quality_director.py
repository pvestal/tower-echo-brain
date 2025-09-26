"""
Quality Director for Echo Brain Board of Directors System

This module provides comprehensive code quality evaluation capabilities including
cyclomatic complexity analysis, code duplication detection, naming conventions
checking, documentation coverage assessment, test coverage evaluation, and
SOLID principles compliance.

Author: Echo Brain Board of Directors System
Created: 2025-09-16
Version: 1.0.0
"""

import logging
import re
import ast
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from .base_director import DirectorBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityDirector(DirectorBase):
    """
    Quality Director specializing in code quality assessment and improvement recommendations.

    This director provides comprehensive code quality analysis including:
    - Cyclomatic complexity analysis
    - Code duplication detection
    - Naming conventions verification
    - Documentation coverage assessment
    - Test coverage evaluation
    - SOLID principles compliance checking
    - Clean code principles assessment
    - Code smell detection
    - Refactoring recommendations
    """

    def __init__(self):
        """Initialize the Quality Director with code quality expertise."""
        super().__init__(
            name="QualityDirector",
            expertise="Code Quality, Software Architecture, Clean Code, SOLID Principles, Refactoring",
            version="1.0.0"
        )

        # Initialize quality-specific configurations
        self.complexity_thresholds = {
            "function": {"low": 5, "medium": 10, "high": 15},
            "class": {"low": 20, "medium": 40, "high": 60},
            "module": {"low": 50, "medium": 100, "high": 200}
        }

        self.naming_patterns = {
            "python": {
                "function": r"^[a-z_][a-z0-9_]*$",
                "class": r"^[A-Z][a-zA-Z0-9]*$",
                "constant": r"^[A-Z][A-Z0-9_]*$",
                "variable": r"^[a-z_][a-z0-9_]*$"
            },
            "javascript": {
                "function": r"^[a-z][a-zA-Z0-9]*$",
                "class": r"^[A-Z][a-zA-Z0-9]*$",
                "constant": r"^[A-Z][A-Z0-9_]*$",
                "variable": r"^[a-z][a-zA-Z0-9]*$"
            }
        }

        self.code_smell_patterns = self._initialize_code_smell_patterns()

        logger.info(f"QualityDirector initialized with {len(self.knowledge_base['quality_best_practices'])} best practices")

    def evaluate(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a task from a code quality perspective.

        Args:
            task (Dict[str, Any]): Task information including code, description, requirements
            context (Dict[str, Any]): Additional context including user info, system state

        Returns:
            Dict[str, Any]: Comprehensive quality evaluation result
        """
        try:
            # Extract code and relevant information
            code_content = task.get("code", "")
            task_type = task.get("type", "unknown")
            description = task.get("description", "")
            language = task.get("language", "python")

            # Perform comprehensive quality analysis
            quality_metrics = self._perform_quality_analysis(code_content, language)

            # Calculate overall quality score
            quality_score = self._calculate_quality_score(quality_metrics)

            # Detect code smells
            code_smells = self.detect_code_smells(code_content, language)

            # Analyze complexity
            complexity_analysis = self.analyze_complexity(code_content, language)

            # Check naming conventions
            naming_issues = self.check_naming_conventions(code_content, language)

            # Assess documentation
            documentation_assessment = self.assess_documentation(code_content, language)

            # Evaluate test coverage (if test code is provided)
            test_coverage = self.evaluate_test_coverage(task, context)

            # Check SOLID principles compliance
            solid_compliance = self._check_solid_compliance(code_content, language)

            # Determine confidence based on analysis completeness
            confidence_factors = {
                "code_availability": 0.9 if code_content else 0.2,
                "analysis_depth": 0.8,
                "pattern_recognition": 0.7,
                "context_completeness": 0.8 if context.get("requirements") else 0.4
            }
            confidence = self.calculate_confidence(confidence_factors)

            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                quality_metrics, code_smells, complexity_analysis, naming_issues,
                documentation_assessment, solid_compliance
            )

            # Create detailed reasoning
            reasoning_factors = [
                f"Overall quality score: {quality_score:.1f}/10",
                f"Detected {len(code_smells)} code smells",
                f"Cyclomatic complexity: {complexity_analysis.get('average_complexity', 0):.1f}",
                f"Documentation coverage: {documentation_assessment.get('coverage_percentage', 0):.1f}%",
                f"Naming convention compliance: {100 - len(naming_issues):.1f}%",
                f"SOLID principles adherence: {solid_compliance.get('overall_score', 0):.1f}/5"
            ]

            reasoning = self.generate_reasoning(
                f"Quality analysis completed with score {quality_score:.1f}/10",
                reasoning_factors,
                context
            )

            # Determine risk factors
            risk_factors = self._identify_quality_risks(quality_metrics, code_smells, complexity_analysis)

            # Update metrics
            evaluation_result = {
                "assessment": f"Code quality evaluation completed with score {quality_score:.1f}/10",
                "confidence": confidence,
                "reasoning": reasoning,
                "recommendations": recommendations,
                "risk_factors": risk_factors,
                "quality_metrics": quality_metrics,
                "quality_score": quality_score,
                "code_smells": code_smells,
                "complexity_analysis": complexity_analysis,
                "naming_issues": naming_issues,
                "documentation_assessment": documentation_assessment,
                "solid_compliance": solid_compliance,
                "estimated_effort": self._estimate_refactoring_effort(quality_metrics, code_smells)
            }

            self.update_metrics(evaluation_result)

            return evaluation_result

        except Exception as e:
            logger.error(f"Error in quality evaluation: {str(e)}")
            return {
                "assessment": "Quality evaluation failed due to technical error",
                "confidence": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "recommendations": ["Review code structure and try again"],
                "risk_factors": ["Unable to perform quality analysis"],
                "quality_score": 0.0,
                "estimated_effort": "Unknown - analysis failed"
            }

    def analyze_complexity(self, code_content: str, language: str = "python") -> Dict[str, Any]:
        """
        Analyze cyclomatic complexity of the provided code.

        Args:
            code_content (str): Source code to analyze
            language (str): Programming language

        Returns:
            Dict[str, Any]: Complexity analysis results
        """
        if not code_content.strip():
            return {"average_complexity": 0, "functions": [], "classes": [], "total_complexity": 0}

        complexity_results = {
            "functions": [],
            "classes": [],
            "total_complexity": 0,
            "average_complexity": 0,
            "high_complexity_items": []
        }

        try:
            if language.lower() == "python":
                complexity_results = self._analyze_python_complexity(code_content)
            else:
                # Generic complexity analysis using simple heuristics
                complexity_results = self._analyze_generic_complexity(code_content)

            return complexity_results

        except Exception as e:
            logger.error(f"Error analyzing complexity: {str(e)}")
            return {"error": str(e), "average_complexity": 0}

    def check_naming_conventions(self, code_content: str, language: str = "python") -> List[Dict[str, Any]]:
        """
        Check naming convention compliance.

        Args:
            code_content (str): Source code to check
            language (str): Programming language

        Returns:
            List[Dict[str, Any]]: List of naming convention violations
        """
        violations = []

        if not code_content.strip():
            return violations

        try:
            patterns = self.naming_patterns.get(language.lower(), self.naming_patterns["python"])

            # Check function names
            function_matches = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
            for func_name in function_matches:
                if not re.match(patterns["function"], func_name):
                    violations.append({
                        "type": "function_naming",
                        "item": func_name,
                        "expected_pattern": patterns["function"],
                        "severity": "medium",
                        "line": self._find_line_number(code_content, f"def {func_name}")
                    })

            # Check class names
            class_matches = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]', code_content)
            for class_name in class_matches:
                if not re.match(patterns["class"], class_name):
                    violations.append({
                        "type": "class_naming",
                        "item": class_name,
                        "expected_pattern": patterns["class"],
                        "severity": "medium",
                        "line": self._find_line_number(code_content, f"class {class_name}")
                    })

            # Check constant names (ALL_CAPS assignments)
            constant_matches = re.findall(r'^([A-Z][A-Z0-9_]*)\s*=', code_content, re.MULTILINE)
            for const_name in constant_matches:
                if not re.match(patterns["constant"], const_name):
                    violations.append({
                        "type": "constant_naming",
                        "item": const_name,
                        "expected_pattern": patterns["constant"],
                        "severity": "low",
                        "line": self._find_line_number(code_content, f"{const_name} =")
                    })

        except Exception as e:
            logger.error(f"Error checking naming conventions: {str(e)}")
            violations.append({
                "type": "analysis_error",
                "item": "naming_check",
                "error": str(e),
                "severity": "high"
            })

        return violations

    def assess_documentation(self, code_content: str, language: str = "python") -> Dict[str, Any]:
        """
        Assess documentation coverage and quality.

        Args:
            code_content (str): Source code to assess
            language (str): Programming language

        Returns:
            Dict[str, Any]: Documentation assessment results
        """
        if not code_content.strip():
            return {"coverage_percentage": 0, "documented_functions": 0, "total_functions": 0}

        assessment = {
            "coverage_percentage": 0.0,
            "documented_functions": 0,
            "total_functions": 0,
            "documented_classes": 0,
            "total_classes": 0,
            "module_docstring": False,
            "quality_issues": []
        }

        try:
            # Check for module-level docstring
            if code_content.strip().startswith('"""') or code_content.strip().startswith("'''"):
                assessment["module_docstring"] = True

            # Find all functions and check for docstrings
            function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):'
            functions = re.finditer(function_pattern, code_content)

            for func_match in functions:
                assessment["total_functions"] += 1
                func_start = func_match.end()

                # Look for docstring after function definition
                remaining_code = code_content[func_start:]
                lines = remaining_code.split('\n')

                # Skip empty lines and find first non-empty line
                for line in lines[1:6]:  # Check first few lines only
                    line = line.strip()
                    if line and (line.startswith('"""') or line.startswith("'''")):
                        assessment["documented_functions"] += 1
                        break
                    elif line and not line.startswith('#'):
                        break  # Hit code, no docstring found

            # Find all classes and check for docstrings
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]'
            classes = re.finditer(class_pattern, code_content)

            for class_match in classes:
                assessment["total_classes"] += 1
                class_start = class_match.end()

                # Look for docstring after class definition
                remaining_code = code_content[class_start:]
                lines = remaining_code.split('\n')

                for line in lines[1:6]:  # Check first few lines only
                    line = line.strip()
                    if line and (line.startswith('"""') or line.startswith("'''")):
                        assessment["documented_classes"] += 1
                        break
                    elif line and not line.startswith('#'):
                        break

            # Calculate overall coverage
            total_items = assessment["total_functions"] + assessment["total_classes"]
            documented_items = assessment["documented_functions"] + assessment["documented_classes"]

            if total_items > 0:
                assessment["coverage_percentage"] = (documented_items / total_items) * 100

            # Add module docstring to coverage if present
            if assessment["module_docstring"] and total_items > 0:
                assessment["coverage_percentage"] = min(100, assessment["coverage_percentage"] + 10)

            # Identify quality issues
            if assessment["coverage_percentage"] < 50:
                assessment["quality_issues"].append("Low documentation coverage")

            if not assessment["module_docstring"]:
                assessment["quality_issues"].append("Missing module-level docstring")

        except Exception as e:
            logger.error(f"Error assessing documentation: {str(e)}")
            assessment["quality_issues"].append(f"Documentation analysis error: {str(e)}")

        return assessment

    def evaluate_test_coverage(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate test coverage if test code is available.

        Args:
            task (Dict[str, Any]): Task information
            context (Dict[str, Any]): Additional context

        Returns:
            Dict[str, Any]: Test coverage evaluation
        """
        coverage_info = {
            "has_tests": False,
            "test_files": [],
            "estimated_coverage": 0.0,
            "test_quality": "unknown",
            "recommendations": []
        }

        # Check if test code is provided
        test_code = task.get("test_code", "")
        if test_code:
            coverage_info["has_tests"] = True
            coverage_info = self._analyze_test_code(test_code, task.get("code", ""))

        # Check for test files in context
        files = context.get("files", [])
        test_files = [f for f in files if self._is_test_file(f)]
        coverage_info["test_files"] = test_files

        if test_files:
            coverage_info["has_tests"] = True
            if not test_code:
                coverage_info["estimated_coverage"] = min(50.0, len(test_files) * 10)

        # Generate recommendations
        if not coverage_info["has_tests"]:
            coverage_info["recommendations"].append("Add unit tests to improve code reliability")
        elif coverage_info["estimated_coverage"] < 80:
            coverage_info["recommendations"].append("Increase test coverage to at least 80%")

        return coverage_info

    def detect_code_smells(self, code_content: str, language: str = "python") -> List[Dict[str, Any]]:
        """
        Detect code smells in the provided code.

        Args:
            code_content (str): Source code to analyze
            language (str): Programming language

        Returns:
            List[Dict[str, Any]]: List of detected code smells
        """
        smells = []

        if not code_content.strip():
            return smells

        try:
            # Check for long functions
            smells.extend(self._detect_long_methods(code_content))

            # Check for large classes
            smells.extend(self._detect_large_classes(code_content))

            # Check for duplicate code
            smells.extend(self._detect_duplicate_code(code_content))

            # Check for long parameter lists
            smells.extend(self._detect_long_parameter_lists(code_content))

            # Check for deep nesting
            smells.extend(self._detect_deep_nesting(code_content))

            # Check for magic numbers
            smells.extend(self._detect_magic_numbers(code_content))

            # Check for commented code
            smells.extend(self._detect_commented_code(code_content))

        except Exception as e:
            logger.error(f"Error detecting code smells: {str(e)}")
            smells.append({
                "type": "analysis_error",
                "description": f"Code smell detection failed: {str(e)}",
                "severity": "high",
                "line": 0
            })

        return smells

    def load_knowledge(self) -> Dict[str, List[str]]:
        """
        Load quality-specific knowledge base.

        Returns:
            Dict[str, List[str]]: Quality knowledge base
        """
        return {
            "quality_best_practices": [
                # Clean Code Principles (10 items)
                "Write self-documenting code with meaningful names",
                "Keep functions small and focused on single responsibility",
                "Use consistent formatting and indentation throughout",
                "Eliminate code duplication through proper abstraction",
                "Write comprehensive unit tests for all public methods",
                "Use dependency injection for better testability",
                "Follow SOLID principles for maintainable architecture",
                "Implement proper error handling and logging",
                "Use version control with meaningful commit messages",
                "Refactor regularly to improve code structure",

                # Documentation (5 items)
                "Document public APIs with clear examples",
                "Maintain up-to-date README files",
                "Use inline comments sparingly and meaningfully",
                "Document complex business logic and algorithms",
                "Keep documentation close to the code it describes",

                # Testing (8 items)
                "Aim for at least 80% code coverage",
                "Write tests before implementation (TDD)",
                "Use descriptive test names that explain behavior",
                "Keep tests independent and deterministic",
                "Test edge cases and error conditions",
                "Use mocks and stubs for external dependencies",
                "Implement integration tests for critical workflows",
                "Run tests automatically in CI/CD pipeline",

                # Architecture (7 items)
                "Design with loose coupling and high cohesion",
                "Separate concerns into distinct layers",
                "Use design patterns appropriately",
                "Plan for scalability from the beginning",
                "Implement proper database indexing",
                "Use caching strategically for performance",
                "Design APIs with versioning in mind"
            ],

            "code_smell_patterns": [
                # Method-level smells (8 items)
                "Long Method: Methods with more than 20-30 lines",
                "Long Parameter List: Methods with more than 4-5 parameters",
                "Duplicate Code: Identical or nearly identical code blocks",
                "Large Class: Classes with too many responsibilities",
                "God Object: Classes that know too much or do too much",
                "Feature Envy: Methods using more features of another class",
                "Data Clumps: Groups of data items that appear together",
                "Primitive Obsession: Overuse of primitive types",

                # Class-level smells (6 items)
                "Refused Bequest: Subclasses that don't use inherited methods",
                "Inappropriate Intimacy: Classes that know too much about each other",
                "Message Chains: Long chains of method calls",
                "Middle Man: Classes that delegate most of their work",
                "Speculative Generality: Unused abstract classes or methods",
                "Temporary Field: Instance variables set only in certain circumstances",

                # Code organization smells (6 items)
                "Dead Code: Unused variables, parameters, methods, or classes",
                "Comments: Excessive commenting often indicates unclear code",
                "Divergent Change: One class changes for multiple reasons",
                "Shotgun Surgery: One change affects many classes",
                "Parallel Inheritance: Adding subclass requires adding to multiple hierarchies",
                "Lazy Class: Classes that don't do enough to justify their existence"
            ],

            "quality_risk_factors": [
                # Maintainability risks (5 items)
                "High cyclomatic complexity increases bug probability",
                "Lack of test coverage makes refactoring dangerous",
                "Poor naming conventions reduce code readability",
                "Tight coupling makes components hard to change",
                "Missing documentation hinders knowledge transfer",

                # Performance risks (5 items)
                "Inefficient algorithms in critical paths",
                "Memory leaks from unclosed resources",
                "Excessive object creation in loops",
                "Unoptimized database queries",
                "Lack of caching for expensive operations",

                # Security risks (5 items)
                "Input validation missing or insufficient",
                "Hardcoded credentials or sensitive data",
                "Improper error handling exposes system information",
                "Missing authorization checks",
                "Vulnerable dependencies not updated"
            ],

            "refactoring_strategies": [
                # Code structure improvements (5 items)
                "Extract Method: Break large methods into smaller ones",
                "Extract Class: Move related data and methods to new class",
                "Move Method: Relocate methods to more appropriate classes",
                "Rename Method/Variable: Use more descriptive names",
                "Replace Magic Number: Use named constants instead",

                # Architecture improvements (5 items)
                "Introduce Parameter Object: Group related parameters",
                "Replace Conditional with Polymorphism: Use inheritance instead",
                "Form Template Method: Extract common algorithm structure",
                "Replace Inheritance with Delegation: Favor composition",
                "Introduce Null Object: Eliminate null checks with objects"
            ]
        }

    # Private helper methods for quality analysis

    def _perform_quality_analysis(self, code_content: str, language: str) -> Dict[str, Any]:
        """Perform comprehensive quality analysis."""
        metrics = {
            "lines_of_code": len(code_content.split('\n')) if code_content else 0,
            "complexity_score": 0,
            "duplication_percentage": 0,
            "documentation_coverage": 0,
            "naming_compliance": 0,
            "test_coverage": 0
        }

        if not code_content.strip():
            return metrics

        try:
            # Analyze complexity
            complexity_result = self.analyze_complexity(code_content, language)
            metrics["complexity_score"] = complexity_result.get("average_complexity", 0)

            # Check documentation
            doc_result = self.assess_documentation(code_content, language)
            metrics["documentation_coverage"] = doc_result.get("coverage_percentage", 0)

            # Check naming conventions
            naming_issues = self.check_naming_conventions(code_content, language)
            total_identifiers = self._count_identifiers(code_content)
            if total_identifiers > 0:
                metrics["naming_compliance"] = max(0, (total_identifiers - len(naming_issues)) / total_identifiers * 100)

            # Estimate duplication
            metrics["duplication_percentage"] = self._estimate_duplication(code_content)

        except Exception as e:
            logger.error(f"Error in quality analysis: {str(e)}")

        return metrics

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from metrics."""
        if not metrics:
            return 0.0

        # Weighted scoring system
        weights = {
            "complexity_score": 0.25,      # Lower complexity is better
            "documentation_coverage": 0.20,
            "naming_compliance": 0.15,
            "duplication_percentage": 0.20,  # Lower duplication is better
            "test_coverage": 0.20
        }

        total_score = 0.0

        # Complexity score (inverse - lower is better)
        complexity = metrics.get("complexity_score", 0)
        if complexity <= 5:
            complexity_score = 10
        elif complexity <= 10:
            complexity_score = 7
        elif complexity <= 15:
            complexity_score = 4
        else:
            complexity_score = 1
        total_score += complexity_score * weights["complexity_score"]

        # Documentation coverage (0-100% -> 0-10 points)
        doc_coverage = metrics.get("documentation_coverage", 0)
        total_score += (doc_coverage / 10) * weights["documentation_coverage"]

        # Naming compliance (0-100% -> 0-10 points)
        naming = metrics.get("naming_compliance", 0)
        total_score += (naming / 10) * weights["naming_compliance"]

        # Duplication (inverse - lower is better, 0-100% -> 10-0 points)
        duplication = metrics.get("duplication_percentage", 0)
        duplication_score = max(0, 10 - (duplication / 10))
        total_score += duplication_score * weights["duplication_percentage"]

        # Test coverage (0-100% -> 0-10 points)
        test_coverage = metrics.get("test_coverage", 0)
        total_score += (test_coverage / 10) * weights["test_coverage"]

        return min(10.0, max(0.0, total_score))

    def _analyze_python_complexity(self, code_content: str) -> Dict[str, Any]:
        """Analyze Python code complexity using AST."""
        complexity_results = {
            "functions": [],
            "classes": [],
            "total_complexity": 0,
            "average_complexity": 0,
            "high_complexity_items": []
        }

        try:
            tree = ast.parse(code_content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    func_info = {
                        "name": node.name,
                        "complexity": complexity,
                        "line": node.lineno
                    }
                    complexity_results["functions"].append(func_info)
                    complexity_results["total_complexity"] += complexity

                    if complexity > self.complexity_thresholds["function"]["medium"]:
                        complexity_results["high_complexity_items"].append(func_info)

                elif isinstance(node, ast.ClassDef):
                    # Calculate class complexity as sum of method complexities
                    class_complexity = 0
                    methods = []

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_complexity = self._calculate_cyclomatic_complexity(item)
                            class_complexity += method_complexity
                            methods.append({
                                "name": item.name,
                                "complexity": method_complexity,
                                "line": item.lineno
                            })

                    class_info = {
                        "name": node.name,
                        "complexity": class_complexity,
                        "methods": methods,
                        "line": node.lineno
                    }
                    complexity_results["classes"].append(class_info)

                    if class_complexity > self.complexity_thresholds["class"]["medium"]:
                        complexity_results["high_complexity_items"].append(class_info)

            # Calculate average
            total_items = len(complexity_results["functions"]) + len(complexity_results["classes"])
            if total_items > 0:
                complexity_results["average_complexity"] = complexity_results["total_complexity"] / total_items

        except Exception as e:
            logger.error(f"Error in Python complexity analysis: {str(e)}")
            complexity_results["error"] = str(e)

        return complexity_results

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for an AST node."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points that increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1

        return complexity

    def _analyze_generic_complexity(self, code_content: str) -> Dict[str, Any]:
        """Generic complexity analysis using pattern matching."""
        complexity = 1  # Base complexity

        # Count decision points
        decision_patterns = [
            r'\bif\b', r'\bwhile\b', r'\bfor\b', r'\bswitch\b', r'\bcase\b',
            r'\bcatch\b', r'\btry\b', r'\b&&\b', r'\b\|\|\b'
        ]

        for pattern in decision_patterns:
            complexity += len(re.findall(pattern, code_content, re.IGNORECASE))

        return {
            "functions": [{"name": "unknown", "complexity": complexity, "line": 0}],
            "classes": [],
            "total_complexity": complexity,
            "average_complexity": complexity,
            "high_complexity_items": []
        }

    def _detect_long_methods(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect methods that are too long."""
        smells = []
        lines = code_content.split('\n')

        # Find function definitions and measure their length
        for i, line in enumerate(lines, 1):
            if re.match(r'^\s*def\s+', line):
                func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
                if func_match:
                    func_name = func_match.group(1)

                    # Count lines until next function or class
                    func_lines = 1
                    j = i
                    indent_level = len(line) - len(line.lstrip())

                    while j < len(lines):
                        next_line = lines[j]
                        if (next_line.strip() and
                            len(next_line) - len(next_line.lstrip()) <= indent_level and
                            (re.match(r'^\s*def\s+', next_line) or re.match(r'^\s*class\s+', next_line))):
                            break
                        func_lines += 1
                        j += 1

                    if func_lines > 30:  # Threshold for long method
                        smells.append({
                            "type": "long_method",
                            "description": f"Method '{func_name}' is too long ({func_lines} lines)",
                            "severity": "medium",
                            "line": i,
                            "item": func_name,
                            "metric_value": func_lines
                        })

        return smells

    def _detect_large_classes(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect classes that are too large."""
        smells = []
        lines = code_content.split('\n')

        for i, line in enumerate(lines, 1):
            if re.match(r'^\s*class\s+', line):
                class_match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
                if class_match:
                    class_name = class_match.group(1)

                    # Count methods in class
                    method_count = 0
                    j = i
                    indent_level = len(line) - len(line.lstrip())

                    while j < len(lines):
                        next_line = lines[j]
                        if (next_line.strip() and
                            len(next_line) - len(next_line.lstrip()) <= indent_level and
                            re.match(r'^\s*class\s+', next_line)):
                            break
                        if re.match(r'^\s*def\s+', next_line):
                            method_count += 1
                        j += 1

                    if method_count > 20:  # Threshold for large class
                        smells.append({
                            "type": "large_class",
                            "description": f"Class '{class_name}' has too many methods ({method_count})",
                            "severity": "medium",
                            "line": i,
                            "item": class_name,
                            "metric_value": method_count
                        })

        return smells

    def _detect_duplicate_code(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect potential duplicate code blocks."""
        smells = []
        lines = [line.strip() for line in code_content.split('\n') if line.strip()]

        # Simple duplicate detection - look for identical sequences of 3+ lines
        for i in range(len(lines) - 3):
            sequence = lines[i:i+3]
            for j in range(i + 3, len(lines) - 2):
                if lines[j:j+3] == sequence:
                    smells.append({
                        "type": "duplicate_code",
                        "description": f"Potential duplicate code found (lines {i+1}-{i+3} and {j+1}-{j+3})",
                        "severity": "medium",
                        "line": i + 1,
                        "item": f"lines_{i+1}_{j+1}",
                        "duplicate_lines": sequence
                    })
                    break  # Only report first duplicate of each sequence

        return smells

    def _detect_long_parameter_lists(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect functions with too many parameters."""
        smells = []

        # Find function definitions and count parameters
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)'
        for match in re.finditer(func_pattern, code_content):
            func_name = match.group(1)
            params = match.group(2)

            if params.strip():
                param_count = len([p.strip() for p in params.split(',') if p.strip()])
                if param_count > 5:  # Threshold for too many parameters
                    line_num = code_content[:match.start()].count('\n') + 1
                    smells.append({
                        "type": "long_parameter_list",
                        "description": f"Function '{func_name}' has too many parameters ({param_count})",
                        "severity": "medium",
                        "line": line_num,
                        "item": func_name,
                        "metric_value": param_count
                    })

        return smells

    def _detect_deep_nesting(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect deeply nested code structures."""
        smells = []
        lines = code_content.split('\n')

        for i, line in enumerate(lines, 1):
            if line.strip():
                indent_level = (len(line) - len(line.lstrip())) // 4  # Assuming 4-space indentation
                if indent_level > 4:  # Threshold for deep nesting
                    smells.append({
                        "type": "deep_nesting",
                        "description": f"Deep nesting detected (level {indent_level})",
                        "severity": "medium",
                        "line": i,
                        "item": f"line_{i}",
                        "metric_value": indent_level
                    })

        return smells

    def _detect_magic_numbers(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect magic numbers in code."""
        smells = []

        # Find numeric literals (excluding common ones like 0, 1, -1)
        number_pattern = r'\b(?<![\.\w])(?:[2-9]|[1-9]\d+)(?![\.\w])\b'

        for match in re.finditer(number_pattern, code_content):
            number = match.group(0)
            line_num = code_content[:match.start()].count('\n') + 1

            # Skip numbers in comments or strings
            line_start = code_content.rfind('\n', 0, match.start()) + 1
            line_end = code_content.find('\n', match.end())
            if line_end == -1:
                line_end = len(code_content)
            line_content = code_content[line_start:line_end]

            # Simple check if number is in comment or string
            before_number = line_content[:match.start() - line_start]
            if '#' in before_number or '"' in before_number or "'" in before_number:
                continue

            smells.append({
                "type": "magic_number",
                "description": f"Magic number '{number}' should be replaced with named constant",
                "severity": "low",
                "line": line_num,
                "item": number,
                "suggestion": f"Define constant for value {number}"
            })

        return smells

    def _detect_commented_code(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect commented out code."""
        smells = []
        lines = code_content.split('\n')

        # Look for comments that look like code
        code_patterns = [
            r'#\s*(def|class|if|for|while|try|import|from)\s+',
            r'#\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[=\(]',
            r'#\s*(print|return|raise)\s*\('
        ]

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                for pattern in code_patterns:
                    if re.search(pattern, line_stripped):
                        smells.append({
                            "type": "commented_code",
                            "description": "Commented out code should be removed",
                            "severity": "low",
                            "line": i,
                            "item": f"line_{i}",
                            "code_snippet": line_stripped[:50] + "..." if len(line_stripped) > 50 else line_stripped
                        })
                        break

        return smells

    def _check_solid_compliance(self, code_content: str, language: str) -> Dict[str, Any]:
        """Check SOLID principles compliance."""
        compliance = {
            "single_responsibility": {"score": 5.0, "violations": []},
            "open_closed": {"score": 5.0, "violations": []},
            "liskov_substitution": {"score": 5.0, "violations": []},
            "interface_segregation": {"score": 5.0, "violations": []},
            "dependency_inversion": {"score": 5.0, "violations": []},
            "overall_score": 5.0
        }

        if not code_content.strip():
            return compliance

        try:
            # Simple heuristic checks for SOLID violations

            # Single Responsibility - check class size and method count
            large_classes = self._detect_large_classes(code_content)
            if large_classes:
                violation_count = len(large_classes)
                compliance["single_responsibility"]["score"] = max(1.0, 5.0 - violation_count * 0.5)
                compliance["single_responsibility"]["violations"] = [
                    f"Large class detected: {cls['item']}" for cls in large_classes
                ]

            # Open/Closed - look for modification patterns (hard to detect statically)
            # This is a placeholder - real implementation would need more context

            # Interface Segregation - look for large interfaces (many abstract methods)
            # This is a placeholder for more sophisticated analysis

            # Dependency Inversion - look for direct imports vs injection patterns
            direct_imports = len(re.findall(r'^\s*import\s+\w+', code_content, re.MULTILINE))
            if direct_imports > 10:  # Arbitrary threshold
                compliance["dependency_inversion"]["score"] = max(2.0, 5.0 - (direct_imports - 10) * 0.1)
                compliance["dependency_inversion"]["violations"].append(
                    f"Many direct imports ({direct_imports}) may indicate tight coupling"
                )

            # Calculate overall score
            scores = [data["score"] for data in compliance.values() if isinstance(data, dict) and "score" in data]
            compliance["overall_score"] = sum(scores) / len(scores) if scores else 5.0

        except Exception as e:
            logger.error(f"Error checking SOLID compliance: {str(e)}")

        return compliance

    def _generate_quality_recommendations(self, quality_metrics: Dict[str, Any],
                                        code_smells: List[Dict[str, Any]],
                                        complexity_analysis: Dict[str, Any],
                                        naming_issues: List[Dict[str, Any]],
                                        documentation_assessment: Dict[str, Any],
                                        solid_compliance: Dict[str, Any]) -> List[str]:
        """Generate specific quality improvement recommendations."""
        recommendations = []

        # Complexity recommendations
        if complexity_analysis.get("average_complexity", 0) > 10:
            recommendations.append("Reduce cyclomatic complexity by breaking down complex methods")

        # Documentation recommendations
        doc_coverage = documentation_assessment.get("coverage_percentage", 0)
        if doc_coverage < 50:
            recommendations.append("Improve documentation coverage - aim for at least 80% of public methods")

        # Code smell recommendations
        smell_types = {}
        for smell in code_smells:
            smell_type = smell.get("type", "unknown")
            smell_types[smell_type] = smell_types.get(smell_type, 0) + 1

        for smell_type, count in smell_types.items():
            if smell_type == "long_method":
                recommendations.append(f"Refactor {count} long methods using Extract Method pattern")
            elif smell_type == "duplicate_code":
                recommendations.append(f"Eliminate {count} instances of duplicate code through abstraction")
            elif smell_type == "magic_number":
                recommendations.append(f"Replace {count} magic numbers with named constants")

        # Naming convention recommendations
        if naming_issues:
            recommendations.append(f"Fix {len(naming_issues)} naming convention violations")

        # SOLID principle recommendations
        overall_solid = solid_compliance.get("overall_score", 5.0)
        if overall_solid < 4.0:
            recommendations.append("Review class design to better follow SOLID principles")

        # General recommendations based on quality score
        quality_score = quality_metrics.get("quality_score", self._calculate_quality_score(quality_metrics))
        if quality_score < 5:
            recommendations.append("Consider comprehensive refactoring to improve overall code quality")
        elif quality_score < 7:
            recommendations.append("Focus on incremental improvements in testing and documentation")

        return recommendations

    def _identify_quality_risks(self, quality_metrics: Dict[str, Any],
                              code_smells: List[Dict[str, Any]],
                              complexity_analysis: Dict[str, Any]) -> List[str]:
        """Identify quality-related risks."""
        risks = []

        # High complexity risks
        high_complexity_items = complexity_analysis.get("high_complexity_items", [])
        if high_complexity_items:
            risks.append(f"High complexity in {len(high_complexity_items)} items increases bug risk")

        # Critical code smells
        critical_smells = [s for s in code_smells if s.get("severity") == "high"]
        if critical_smells:
            risks.append(f"{len(critical_smells)} critical code smells require immediate attention")

        # Low test coverage
        test_coverage = quality_metrics.get("test_coverage", 0)
        if test_coverage < 50:
            risks.append("Low test coverage makes refactoring risky")

        # High duplication
        duplication = quality_metrics.get("duplication_percentage", 0)
        if duplication > 20:
            risks.append("High code duplication increases maintenance burden")

        return risks

    def _estimate_refactoring_effort(self, quality_metrics: Dict[str, Any],
                                   code_smells: List[Dict[str, Any]]) -> str:
        """Estimate effort required for refactoring."""
        quality_score = quality_metrics.get("quality_score", self._calculate_quality_score(quality_metrics))
        smell_count = len(code_smells)
        complexity_score = quality_metrics.get("complexity_score", 0)

        # Calculate effort based on multiple factors
        effort_points = 0

        if quality_score < 3:
            effort_points += 8  # Major refactoring needed
        elif quality_score < 5:
            effort_points += 5  # Significant improvements needed
        elif quality_score < 7:
            effort_points += 2  # Minor improvements needed

        effort_points += min(smell_count * 0.5, 5)  # Each smell adds effort

        if complexity_score > 15:
            effort_points += 3  # High complexity adds effort

        # Convert to human-readable estimate
        if effort_points >= 10:
            return "High (2-4 weeks of focused refactoring)"
        elif effort_points >= 6:
            return "Medium (1-2 weeks of improvements)"
        elif effort_points >= 3:
            return "Low (2-5 days of cleanup)"
        else:
            return "Minimal (few hours of minor fixes)"

    def _initialize_code_smell_patterns(self) -> Dict[str, Any]:
        """Initialize code smell detection patterns."""
        return {
            "long_method_threshold": 30,
            "large_class_threshold": 20,
            "long_parameter_list_threshold": 5,
            "deep_nesting_threshold": 4,
            "duplicate_block_size": 3
        }

    def _find_line_number(self, code_content: str, search_text: str) -> int:
        """Find line number of given text in code."""
        lines = code_content.split('\n')
        for i, line in enumerate(lines, 1):
            if search_text in line:
                return i
        return 0

    def _count_identifiers(self, code_content: str) -> int:
        """Count total identifiers in code for naming compliance calculation."""
        # Simple estimation - count function, class, and variable definitions
        identifiers = 0
        identifiers += len(re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code_content))
        identifiers += len(re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', code_content))
        identifiers += len(re.findall(r'^([A-Z][A-Z0-9_]*)\s*=', code_content, re.MULTILINE))
        return max(1, identifiers)  # Avoid division by zero

    def _estimate_duplication(self, code_content: str) -> float:
        """Estimate code duplication percentage."""
        lines = [line.strip() for line in code_content.split('\n') if line.strip()]
        if len(lines) < 6:
            return 0.0

        duplicate_lines = 0
        checked = set()

        for i in range(len(lines) - 2):
            if i in checked:
                continue

            sequence = tuple(lines[i:i+3])
            for j in range(i + 3, len(lines) - 2):
                if j in checked:
                    continue

                if tuple(lines[j:j+3]) == sequence:
                    duplicate_lines += 3
                    checked.update(range(i, i+3))
                    checked.update(range(j, j+3))
                    break

        return (duplicate_lines / len(lines)) * 100 if lines else 0.0

    def _analyze_test_code(self, test_code: str, source_code: str) -> Dict[str, Any]:
        """Analyze provided test code for coverage estimation."""
        test_info = {
            "has_tests": True,
            "estimated_coverage": 0.0,
            "test_quality": "unknown",
            "test_methods": [],
            "recommendations": []
        }

        # Count test methods
        test_methods = re.findall(r'def\s+(test_[a-zA-Z0-9_]*)', test_code)
        test_info["test_methods"] = test_methods

        # Simple coverage estimation based on test method count vs source functions
        source_functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', source_code)
        if source_functions and test_methods:
            coverage_ratio = len(test_methods) / len(source_functions)
            test_info["estimated_coverage"] = min(100.0, coverage_ratio * 100)

        # Assess test quality
        if "assert" in test_code.lower() or "expect" in test_code.lower():
            test_info["test_quality"] = "good"
        elif test_methods:
            test_info["test_quality"] = "basic"
        else:
            test_info["test_quality"] = "poor"

        return test_info

    def _is_test_file(self, filename: str) -> bool:
        """Check if a filename indicates a test file."""
        test_patterns = [
            r'test.*\.py$', r'.*test\.py$', r'.*_test\.py$',
            r'test.*\.js$', r'.*test\.js$', r'.*\.test\.js$',
            r'.*\.spec\.js$', r'.*\.spec\.ts$'
        ]

        for pattern in test_patterns:
            if re.search(pattern, filename.lower()):
                return True
        return False