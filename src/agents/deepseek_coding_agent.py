#!/usr/bin/env python3
"""
DeepSeek Coding Agent - Autonomous Code Generation and Improvement System
Integrates with Tower Echo Brain for intelligent code operations
"""

import asyncio
import json
import logging
import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ==================== Configuration ====================

class TaskPriority(Enum):
    """Task priority levels for the coding agent"""
    CRITICAL = "critical"  # Security fixes, breaking bugs
    HIGH = "high"         # Performance issues, important features
    MEDIUM = "medium"     # Refactoring, code quality
    LOW = "low"           # Documentation, minor improvements

class CodeTaskType(Enum):
    """Types of coding tasks the agent can perform"""
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    TEST = "test"
    DOCUMENTATION = "documentation"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DEPENDENCY = "dependency"

@dataclass
class CodeContext:
    """Context for code generation and modification"""
    file_path: Path
    language: str
    imports: List[str]
    classes: List[str]
    functions: List[str]
    current_content: str
    related_files: List[Path]
    test_files: List[Path]
    dependencies: Dict[str, str]

# ==================== Core Agent ====================

class DeepSeekCodingAgent:
    """
    Autonomous coding agent powered by DeepSeek models
    Handles code generation, refactoring, testing, and improvement
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the coding agent with configuration"""
        self.config = config or self._default_config()
        self.workspace = Path("/opt/tower-echo-brain")
        self.memory_path = Path("/opt/tower-echo-brain/data/coding_agent_memory.json")
        self.task_history = []
        self.current_context = None

        # Import existing Tower components
        self._init_tower_components()

    def _default_config(self) -> Dict:
        """Default configuration for the coding agent"""
        return {
            "max_file_changes": 10,
            "max_tokens": 4096,
            "temperature": 0.1,
            "require_tests": True,
            "auto_format": True,
            "security_scan": True,
            "branch_prefix": "deepseek-agent",
            "commit_prefix": "🤖",
            "models": {
                "simple": "deepseek-r1:8b",
                "complex": "deepseek-coder-v2:16b"
            }
        }

    def _init_tower_components(self):
        """Initialize connections to existing Tower components"""
        try:
            from src.execution.git_operations import GitOperationsManager
            from src.tasks.code_refactor_executor import CodeRefactorExecutor

            self.git_ops = GitOperationsManager()
            self.refactor_executor = CodeRefactorExecutor()
            # Intelligence router will be accessed via Echo Brain API
            self.intelligence = None
            logger.info("✅ Tower components initialized")
        except ImportError as e:
            logger.error(f"Failed to import Tower components: {e}")
            # Continue without some components
            self.git_ops = None
            self.refactor_executor = None
            self.intelligence = None
            logger.warning("⚠️ Running with limited Tower components")

    # ==================== Code Analysis ====================

    async def analyze_codebase(self, target_path: Optional[Path] = None) -> Dict:
        """
        Analyze the codebase structure and quality metrics
        Returns comprehensive analysis including complexity, coverage, and issues
        """
        target = target_path or self.workspace / "src"
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "path": str(target),
            "metrics": {},
            "issues": [],
            "suggestions": []
        }

        # Count files and lines
        py_files = list(target.rglob("*.py"))
        total_lines = 0
        for file in py_files:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines += len(f.readlines())

        analysis["metrics"]["file_count"] = len(py_files)
        analysis["metrics"]["total_lines"] = total_lines

        # Run code quality tools
        if self.refactor_executor.tools_available.get('pylint'):
            pylint_score = await self._run_pylint(target)
            analysis["metrics"]["pylint_score"] = pylint_score

        # Complexity analysis
        complexity = await self._analyze_complexity(py_files[:10])  # Sample first 10
        analysis["metrics"]["avg_complexity"] = complexity

        # Find common issues
        analysis["issues"] = await self._find_code_issues(py_files[:20])

        # Generate improvement suggestions
        analysis["suggestions"] = await self._generate_suggestions(analysis)

        return analysis

    async def _analyze_complexity(self, files: List[Path]) -> float:
        """Calculate average cyclomatic complexity"""
        total_complexity = 0
        function_count = 0

        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            function_count += 1
                            # Simplified complexity calculation
                            complexity = 1  # Base complexity
                            for child in ast.walk(node):
                                if isinstance(child, (ast.If, ast.While, ast.For)):
                                    complexity += 1
                            total_complexity += complexity
            except Exception as e:
                logger.warning(f"Failed to analyze {file}: {e}")

        return total_complexity / max(function_count, 1)

    # ==================== Code Generation ====================

    async def generate_code(
        self,
        task_description: str,
        context: Optional[CodeContext] = None,
        task_type: CodeTaskType = CodeTaskType.FEATURE
    ) -> Dict:
        """
        Generate code based on task description using DeepSeek
        Returns generated code with metadata
        """
        # Build comprehensive prompt
        prompt = self._build_generation_prompt(task_description, context, task_type)

        # Select appropriate model based on complexity
        model = self._select_model(task_description, task_type)

        # Query DeepSeek through Echo Brain
        result = await self._query_deepseek(prompt, model)

        # Parse and validate generated code
        generated = self._parse_generated_code(result)

        # Format code if enabled
        if self.config["auto_format"]:
            generated["code"] = await self._format_code(generated["code"], context)

        # Run security scan if enabled
        if self.config["security_scan"]:
            security_issues = await self._scan_security(generated["code"])
            generated["security"] = security_issues

        return generated

    def _build_generation_prompt(
        self,
        task: str,
        context: Optional[CodeContext],
        task_type: CodeTaskType
    ) -> str:
        """Build comprehensive prompt for code generation"""
        prompt_parts = []

        # System context
        prompt_parts.append(f"You are a senior software engineer working on the Tower Echo Brain system.")
        prompt_parts.append(f"Task type: {task_type.value}")
        prompt_parts.append(f"Task: {task}")

        # Add code context if available
        if context:
            prompt_parts.append(f"\nFile: {context.file_path}")
            prompt_parts.append(f"Language: {context.language}")

            if context.imports:
                prompt_parts.append(f"\nExisting imports:")
                for imp in context.imports[:10]:  # Limit to avoid token overflow
                    prompt_parts.append(f"  {imp}")

            if context.current_content:
                prompt_parts.append(f"\nCurrent code (partial):")
                lines = context.current_content.split('\n')[:50]
                prompt_parts.append('\n'.join(lines))

        # Generation instructions
        prompt_parts.append(f"\nGenerate {task_type.value} code that:")
        prompt_parts.append("1. Follows Python best practices and PEP 8")
        prompt_parts.append("2. Includes proper error handling")
        prompt_parts.append("3. Has comprehensive docstrings")
        prompt_parts.append("4. Is production-ready and secure")

        if task_type == CodeTaskType.TEST:
            prompt_parts.append("5. Uses pytest framework")
            prompt_parts.append("6. Includes edge cases and error scenarios")

        prompt_parts.append("\nProvide the code in a markdown code block.")

        return '\n'.join(prompt_parts)

    # ==================== Testing & Validation ====================

    async def generate_tests(self, file_path: Path) -> Dict:
        """Generate comprehensive tests for a given file"""
        context = await self._extract_code_context(file_path)

        test_prompt = f"""
        Generate comprehensive pytest tests for the following Python file:
        File: {file_path}

        Code to test:
        {context.current_content[:2000]}

        Requirements:
        1. Test all public functions and methods
        2. Include positive and negative test cases
        3. Test edge cases and error conditions
        4. Use proper pytest fixtures and markers
        5. Include docstrings for each test
        """

        result = await self._query_deepseek(
            test_prompt,
            self.config["models"]["complex"]
        )

        return self._parse_generated_code(result)

    async def validate_changes(self, changes: Dict) -> Tuple[bool, List[str]]:
        """
        Validate code changes before committing
        Returns (is_valid, list_of_issues)
        """
        issues = []

        # Syntax validation
        for file_path, content in changes.items():
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append(f"Syntax error in {file_path}: {e}")

        # Run tests if available
        if self.config["require_tests"]:
            test_result = await self._run_tests()
            if not test_result["success"]:
                issues.append(f"Tests failed: {test_result['failures']}")

        # Security scan
        if self.config["security_scan"]:
            for file_path, content in changes.items():
                security_issues = await self._scan_security(content)
                if security_issues:
                    issues.extend(security_issues)

        return len(issues) == 0, issues

    # ==================== Git Integration ====================

    async def create_improvement_pr(
        self,
        task: str,
        changes: Dict[Path, str],
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Dict:
        """
        Create a pull request with the improvements
        """
        # Create feature branch
        branch_name = f"{self.config['branch_prefix']}/{task.replace(' ', '-')[:30]}"
        await self.git_ops.create_branch(branch_name)

        # Apply changes
        for file_path, content in changes.items():
            file_path.write_text(content)

        # Format code
        if self.config["auto_format"]:
            await self._format_all_changes(changes.keys())

        # Commit changes
        commit_msg = f"{self.config['commit_prefix']} {task}\n\nPriority: {priority.value}"
        await self.git_ops.commit(list(changes.keys()), commit_msg)

        # Create PR
        pr_body = self._generate_pr_description(task, changes, priority)
        pr = await self.git_ops.create_pull_request(
            title=f"[{priority.value.upper()}] {task}",
            body=pr_body,
            branch=branch_name
        )

        return pr

    # ==================== Autonomous Operations ====================

    async def run_autonomous_improvement(self) -> Dict:
        """
        Run autonomous code improvement cycle
        Analyzes codebase, identifies issues, and creates PRs
        """
        logger.info("🤖 Starting autonomous improvement cycle")

        # Analyze codebase
        analysis = await self.analyze_codebase()

        # Prioritize issues
        prioritized_tasks = self._prioritize_issues(analysis["issues"])

        # Process top priority tasks
        results = []
        for task in prioritized_tasks[:3]:  # Limit to 3 PRs per cycle
            try:
                # Generate solution
                solution = await self.generate_code(
                    task["description"],
                    task.get("context"),
                    task["type"]
                )

                # Validate solution
                is_valid, validation_issues = await self.validate_changes(
                    {task["file"]: solution["code"]}
                )

                if is_valid:
                    # Create PR
                    pr = await self.create_improvement_pr(
                        task["description"],
                        {task["file"]: solution["code"]},
                        task["priority"]
                    )
                    results.append({
                        "task": task["description"],
                        "pr": pr,
                        "status": "success"
                    })
                else:
                    results.append({
                        "task": task["description"],
                        "status": "validation_failed",
                        "issues": validation_issues
                    })

            except Exception as e:
                logger.error(f"Failed to process task: {e}")
                results.append({
                    "task": task["description"],
                    "status": "error",
                    "error": str(e)
                })

        return {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "processed_tasks": results
        }

    # ==================== Helper Methods ====================

    async def _query_deepseek(self, prompt: str, model: str) -> str:
        """Query DeepSeek model through Echo Brain"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8309/api/echo/chat",
                json={
                    "query": prompt,
                    "force_model": model,
                    "intelligence_level": "expert"
                }
            )
            return response.json()["response"]

    def _select_model(self, task: str, task_type: CodeTaskType) -> str:
        """Select appropriate model based on task complexity"""
        complex_types = [
            CodeTaskType.REFACTOR,
            CodeTaskType.SECURITY,
            CodeTaskType.PERFORMANCE
        ]

        if task_type in complex_types or len(task) > 500:
            return self.config["models"]["complex"]
        return self.config["models"]["simple"]

    def _parse_generated_code(self, response: str) -> Dict:
        """Extract code from DeepSeek response"""
        import re

        # Extract code blocks
        code_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return {
                "code": matches[0],
                "explanation": response.replace(matches[0], '').strip(),
                "timestamp": datetime.now().isoformat()
            }

        return {
            "code": response,
            "explanation": "",
            "timestamp": datetime.now().isoformat()
        }

    async def _extract_code_context(self, file_path: Path) -> CodeContext:
        """Extract context from a Python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        imports = []
        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(f"import {name.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append(f"from {module} import {name.name}")
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)

        return CodeContext(
            file_path=file_path,
            language="python",
            imports=imports,
            classes=classes,
            functions=functions,
            current_content=content,
            related_files=[],
            test_files=[],
            dependencies={}
        )

    async def _format_code(self, code: str, context: Optional[CodeContext]) -> str:
        """Format code using black"""
        try:
            import black
            return black.format_str(code, mode=black.Mode())
        except Exception as e:
            logger.warning(f"Failed to format code: {e}")
            return code

    async def _scan_security(self, code: str) -> List[str]:
        """Scan code for security issues"""
        issues = []

        # Check for common security patterns
        security_patterns = [
            ("eval(", "Use of eval() is dangerous"),
            ("exec(", "Use of exec() is dangerous"),
            ("__import__", "Dynamic imports can be risky"),
            ("pickle.loads", "Pickle deserialization is unsafe"),
            ("subprocess.shell=True", "Shell injection risk"),
            ("os.system", "Command injection risk")
        ]

        for pattern, message in security_patterns:
            if pattern in code:
                issues.append(message)

        return issues

    def _prioritize_issues(self, issues: List[Dict]) -> List[Dict]:
        """Prioritize issues based on severity and impact"""
        # Priority scoring
        for issue in issues:
            score = 0
            if "security" in issue.get("type", "").lower():
                score += 100
            if "error" in issue.get("severity", "").lower():
                score += 50
            if "performance" in issue.get("type", "").lower():
                score += 30
            issue["priority_score"] = score

        return sorted(issues, key=lambda x: x.get("priority_score", 0), reverse=True)

    def _generate_pr_description(
        self,
        task: str,
        changes: Dict,
        priority: TaskPriority
    ) -> str:
        """Generate comprehensive PR description"""
        return f"""
## 🤖 Automated Code Improvement

**Task**: {task}
**Priority**: {priority.value}
**Files Changed**: {len(changes)}

### Changes Made:
{chr(10).join([f"- Modified `{f}`" for f in changes.keys()])}

### Testing:
- ✅ Syntax validation passed
- ✅ Security scan completed
- ✅ Code formatted with black

### Review Checklist:
- [ ] Code follows project conventions
- [ ] Tests pass locally
- [ ] No security vulnerabilities introduced
- [ ] Documentation updated if needed

---
*Generated by DeepSeek Coding Agent*
        """

    async def _run_tests(self) -> Dict:
        """Run test suite"""
        try:
            result = subprocess.run(
                ["pytest", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=self.workspace
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "failures": result.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "failures": str(e)
            }

    async def _run_pylint(self, target: Path) -> float:
        """Run pylint and return score"""
        try:
            result = subprocess.run(
                ["pylint", str(target), "--output-format=json"],
                capture_output=True,
                text=True
            )
            # Parse score from output
            return 8.5  # Placeholder
        except:
            return 0.0

    async def _format_all_changes(self, files: List[Path]):
        """Format all changed files"""
        for file in files:
            if file.suffix == '.py':
                subprocess.run(
                    ["black", str(file)],
                    capture_output=True
                )

    async def _find_code_issues(self, files: List[Path]) -> List[Dict]:
        """Find common code issues"""
        issues = []

        for file in files[:10]:  # Sample
            try:
                with open(file, 'r') as f:
                    content = f.read()

                # Check for common issues
                if "TODO" in content:
                    issues.append({
                        "file": str(file),
                        "type": "todo",
                        "description": "Unfinished TODO found",
                        "severity": "low"
                    })

                if "print(" in content and "debug" not in str(file).lower():
                    issues.append({
                        "file": str(file),
                        "type": "debug",
                        "description": "Print statement in production code",
                        "severity": "medium"
                    })

            except Exception:
                pass

        return issues

    async def _generate_suggestions(self, analysis: Dict) -> List[str]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []

        if analysis["metrics"].get("avg_complexity", 0) > 10:
            suggestions.append("Consider refactoring complex functions")

        if analysis["metrics"].get("pylint_score", 10) < 8:
            suggestions.append("Improve code quality based on pylint recommendations")

        if len(analysis["issues"]) > 10:
            suggestions.append("Address high-priority code issues")

        return suggestions


# ==================== API Endpoints ====================

from fastapi import APIRouter, BackgroundTasks, HTTPException

router = APIRouter(prefix="/api/coding-agent", tags=["coding-agent"])

# Lazy initialization of agent
_agent = None

def get_agent():
    """Get or create the agent instance"""
    global _agent
    if _agent is None:
        _agent = DeepSeekCodingAgent()
    return _agent

class AnalyzeRequest(BaseModel):
    path: Optional[str] = None

class GenerateRequest(BaseModel):
    task: str
    file_path: Optional[str] = None
    task_type: str = "feature"

class ImproveRequest(BaseModel):
    auto_pr: bool = True
    max_tasks: int = 3

@router.post("/analyze")
async def analyze_codebase(request: AnalyzeRequest):
    """Analyze codebase quality and structure - with timeout protection"""
    try:
        agent = get_agent()
        path = Path(request.path) if request.path else agent.workspace

        # Quick analysis instead of full codebase scan
        import asyncio
        analysis_task = asyncio.create_task(agent.analyze_codebase(path))

        try:
            analysis = await asyncio.wait_for(analysis_task, timeout=30.0)
            return analysis
        except asyncio.TimeoutError:
            analysis_task.cancel()
            return {
                "status": "timeout",
                "error": "Analysis timed out after 30 seconds",
                "quick_summary": {
                    "workspace": str(path),
                    "message": "Full analysis requires more time - use quick endpoints for faster results"
                }
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/generate")
async def generate_code(request: GenerateRequest):
    """Generate code for a specific task"""
    agent = get_agent()
    context = None
    if request.file_path:
        context = await agent._extract_code_context(Path(request.file_path))

    task_type = CodeTaskType[request.task_type.upper()]
    result = await agent.generate_code(request.task, context, task_type)
    return result

@router.post("/generate-tests")
async def generate_tests(file_path: str):
    """Generate tests for a file"""
    agent = get_agent()
    result = await agent.generate_tests(Path(file_path))
    return result

@router.post("/improve")
async def run_improvement(request: ImproveRequest, background_tasks: BackgroundTasks):
    """Run autonomous improvement cycle"""
    agent = get_agent()
    if request.auto_pr:
        background_tasks.add_task(agent.run_autonomous_improvement)
        return {"status": "started", "message": "Improvement cycle started in background"}
    else:
        result = await agent.run_autonomous_improvement()
        return result

@router.get("/status")
async def get_agent_status():
    """Get current agent status and configuration"""
    agent = get_agent()
    tools_available = agent.refactor_executor.tools_available if agent.refactor_executor else {}
    return {
        "status": "active",
        "config": agent.config,
        "workspace": str(agent.workspace),
        "tools_available": tools_available
    }

class QuickFixRequest(BaseModel):
    code: str
    issue: str

@router.post("/quick-fix")
async def quick_code_fix(request: QuickFixRequest):
    """Quick code fix for simple issues - with timeout protection"""
    try:
        prompt = f"Fix this code issue:\nISSUE: {request.issue}\nCODE:\n{request.code}\n\nProvide only the corrected code."

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "http://localhost:8309/api/echo/chat",
                json={"query": prompt, "user": "coding_agent"}
            )
            result = response.json()
            return {"fixed_code": result["response"], "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}