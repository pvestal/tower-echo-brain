#!/usr/bin/env python3
"""
Code Reviewer - Quality analysis and auto-fixing
Self-review loop for autonomous code modifications
"""

import asyncio
import ast
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class CodeReviewer:
    """Analyze code quality with pylint, black, and custom checks"""
    
    def __init__(self):
        self.quality_threshold = 7.0  # Minimum score to pass (0-10 scale)
        self.checks_available = self._detect_available_tools()
        logger.info(f"CodeReviewer initialized. Available tools: {self.checks_available}")
    
    def _detect_available_tools(self) -> Dict[str, bool]:
        """Detect which quality tools are installed"""
        tools = {
            'pylint': self._check_command('pylint --version'),
            'black': self._check_command('black --version'),
            'ruff': self._check_command('ruff --version'),
            'mypy': self._check_command('mypy --version')
        }
        return tools
    
    def _check_command(self, cmd: str) -> bool:
        """Check if command is available"""
        try:
            subprocess.run(cmd.split(), capture_output=True, timeout=2)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def review_file(self, path: str) -> Dict[str, Any]:
        """
        Comprehensive code review
        
        Returns:
            Dict with score, issues, suggestions, security_flags
        """
        result = {
            'success': False,
            'path': path,
            'score': 0.0,
            'passed': False,
            'issues': [],
            'suggestions': [],
            'security_flags': [],
            'complexity_score': None,
            'timestamp': None
        }
        
        try:
            file_path = Path(path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Syntax validation
            syntax_valid, syntax_error = await self._validate_syntax(path, content)
            if not syntax_valid:
                result['issues'].append({
                    'type': 'syntax_error',
                    'severity': 'critical',
                    'message': syntax_error
                })
                result['score'] = 0.0
                return result
            
            # Security checks
            security_issues = await self._security_check(content)
            result['security_flags'] = security_issues
            if security_issues:
                result['issues'].extend([
                    {'type': 'security', 'severity': 'high', 'message': issue}
                    for issue in security_issues
                ])
            
            # Code quality analysis
            if self.checks_available['pylint']:
                pylint_score, pylint_issues = await self._run_pylint(path)
                result['score'] = pylint_score
                result['issues'].extend(pylint_issues)
            else:
                # Fallback: basic quality score
                result['score'] = await self._basic_quality_score(content)
            
            # Complexity analysis
            result['complexity_score'] = await self._complexity_analysis(content)
            
            # Style checks
            if self.checks_available['black']:
                style_issues = await self._check_style(path)
                if style_issues:
                    result['suggestions'].extend(style_issues)
            
            # Determine if passed
            result['passed'] = (
                result['score'] >= self.quality_threshold and
                len(result['security_flags']) == 0
            )
            
            result['success'] = True
            logger.info(f"Review complete: {path} scored {result['score']:.2f}/10")
            
        except Exception as e:
            result['issues'].append({
                'type': 'review_error',
                'severity': 'critical',
                'message': str(e)
            })
            logger.error(f"Error reviewing {path}: {e}")
        
        return result
    
    async def _validate_syntax(self, path: str, content: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax"""
        try:
            ast.parse(content)
            return (True, None)
        except SyntaxError as e:
            return (False, f"Line {e.lineno}: {e.msg}")
    
    async def _security_check(self, content: str) -> List[str]:
        """Check for security issues"""
        issues = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']', 'Hardcoded password detected'),
            (r'api[_-]?key\s*=\s*["\'][^"\']', 'Hardcoded API key detected'),
            (r'secret\s*=\s*["\'][^"\']', 'Hardcoded secret detected'),
            (r'token\s*=\s*["\'][A-Za-z0-9]{20,}', 'Hardcoded token detected'),
        ]
        
        for pattern, message in secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(message)
        
        # Check for unsafe operations
        unsafe_patterns = [
            (r'eval\(', 'Unsafe eval() usage'),
            (r'exec\(', 'Unsafe exec() usage'),
            (r'__import__\(', 'Dynamic import detected'),
            (r'shell=True', 'Shell injection risk (shell=True)'),
        ]
        
        for pattern, message in unsafe_patterns:
            if re.search(pattern, content):
                issues.append(message)
        
        return issues
    
    async def _run_pylint(self, path: str) -> Tuple[float, List[Dict]]:
        """Run pylint and parse results"""
        try:
            result = subprocess.run(
                ['pylint', '--output-format=json', '--score=y', str(path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse JSON output
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    issues = [
                        {
                            'type': item.get('type', 'unknown'),
                            'severity': item.get('message-id', ''),
                            'message': item.get('message', ''),
                            'line': item.get('line', 0)
                        }
                        for item in data
                    ]
                except json.JSONDecodeError:
                    issues = []
            else:
                issues = []
            
            # Extract score from stderr (pylint prints score there)
            score_match = re.search(r'Your code has been rated at ([\d.]+)/10', result.stderr)
            score = float(score_match.group(1)) if score_match else 5.0
            
            return (score, issues)
            
        except Exception as e:
            logger.warning(f"Pylint error: {e}")
            return (5.0, [])
    
    async def _basic_quality_score(self, content: str) -> float:
        """Fallback quality scoring without pylint"""
        score = 10.0
        
        lines = content.split('\n')
        
        # Deduct points for common issues
        if len(lines) < 10:
            score -= 1.0  # Too short
        
        # Check for docstrings
        has_module_docstring = content.strip().startswith('"""')
        if not has_module_docstring:
            score -= 1.0
        
        # Check function docstrings
        functions = re.findall(r'def\s+\w+\([^)]*\):', content)
        docstrings = re.findall(r'def\s+\w+\([^)]*\):\s*"""', content)
        if functions and len(docstrings) / len(functions) < 0.5:
            score -= 1.0
        
        # Check line length
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines > len(lines) * 0.2:
            score -= 1.0
        
        # Check imports
        if 'import *' in content:
            score -= 0.5  # Wildcard imports
        
        return max(0.0, score)
    
    async def _complexity_analysis(self, content: str) -> Dict[str, Any]:
        """Analyze code complexity"""
        try:
            tree = ast.parse(content)
            
            # Count classes, functions, branches
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            branches = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.If, ast.For, ast.While)))
            
            # Calculate average function length
            func_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if func_nodes:
                avg_func_length = sum(
                    len(ast.get_source_segment(content, node).split('\n'))
                    for node in func_nodes
                ) / len(func_nodes)
            else:
                avg_func_length = 0
            
            return {
                'classes': classes,
                'functions': functions,
                'branches': branches,
                'avg_function_length': round(avg_func_length, 1)
            }
            
        except Exception as e:
            logger.warning(f"Complexity analysis error: {e}")
            return {}
    
    async def _check_style(self, path: str) -> List[str]:
        """Check style with black"""
        try:
            result = subprocess.run(
                ['black', '--check', '--quiet', str(path)],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return ["Code formatting does not match black style"]
            
        except Exception as e:
            logger.warning(f"Black check error: {e}")
        
        return []
    
    async def auto_fix_common_issues(self, path: str) -> Dict[str, Any]:
        """Automatically fix common issues"""
        result = {
            'success': False,
            'path': path,
            'fixes_applied': [],
            'error': None
        }
        
        try:
            file_path = Path(path)
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix trailing whitespace
            lines = content.split('\n')
            lines = [line.rstrip() for line in lines]
            content = '\n'.join(lines)
            if content != original_content:
                result['fixes_applied'].append('Removed trailing whitespace')
            
            # Ensure file ends with newline
            if content and not content.endswith('\n'):
                content += '\n'
                result['fixes_applied'].append('Added final newline')
            
            # Write back if changes made
            if result['fixes_applied']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            result['success'] = True
            logger.info(f"Auto-fixed {len(result['fixes_applied'])} issues in {path}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error auto-fixing {path}: {e}")
        
        return result
    
    async def format_with_black(self, path: str) -> Dict[str, Any]:
        """Format code with black"""
        result = {
            'success': False,
            'path': path,
            'formatted': False,
            'error': None
        }
        
        if not self.checks_available['black']:
            result['error'] = 'black not installed'
            return result
        
        try:
            subprocess.run(
                ['black', '--quiet', str(path)],
                check=True,
                timeout=30
            )
            
            result['success'] = True
            result['formatted'] = True
            logger.info(f"Formatted {path} with black")
            
        except subprocess.CalledProcessError as e:
            result['error'] = f"Black failed: {e}"
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error formatting {path}: {e}")
        
        return result

# Singleton instance
_code_reviewer = None

def get_code_reviewer() -> CodeReviewer:
    """Get singleton CodeReviewer instance"""
    global _code_reviewer
    if _code_reviewer is None:
        _code_reviewer = CodeReviewer()
    return _code_reviewer
