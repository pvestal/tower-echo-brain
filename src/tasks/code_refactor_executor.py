#!/usr/bin/env python3
"""
Autonomous Code Refactoring System
Monitors code quality and performs safe refactoring
"""

import asyncio
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class CodeRefactorExecutor:
    """Executes autonomous code refactoring with safety checks"""
    
    def __init__(self):
        self.refactor_log = Path("/opt/tower-echo-brain/logs/code_refactors.log")
        self.refactor_log.parent.mkdir(parents=True, exist_ok=True)
        self.tools_available = self._check_tools()
        
    def _check_tools(self) -> Dict[str, bool]:
        """Check which code quality tools are installed"""
        tools = {}
        for tool in ['pylint', 'black', 'ruff', 'mypy']:
            try:
                subprocess.run([tool, '--version'], capture_output=True, check=True, timeout=5)
                tools[tool] = True
                logger.info(f"✅ {tool} available")
            except Exception:
                tools[tool] = False
                logger.warning(f"❌ {tool} not available")
        return tools
    
    async def analyze_code_quality(self, project_path: str) -> Dict[str, Any]:
        """Analyze code quality for a project"""
        logger.info(f"🔍 Analyzing code quality: {project_path}")
        
        results = {
            'project': project_path,
            'timestamp': datetime.now().isoformat(),
            'issues': [],
            'pylint_score': None,
            'file_count': 0
        }
        
        # Count Python files
        path = Path(project_path)
        if path.exists():
            py_files = list(path.rglob('*.py'))
            results['file_count'] = len(py_files)
            logger.info(f"📊 Found {len(py_files)} Python files in {project_path}")
        
        # Run pylint if available
        if self.tools_available.get('pylint') and results['file_count'] > 0:
            try:
                result = subprocess.run(
                    ['pylint', project_path, '--output-format=json', '--exit-zero'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                # Parse pylint output
                if result.stdout:
                    try:
                        pylint_data = json.loads(result.stdout)
                        results['issues'] = pylint_data
                        results['pylint_score'] = self._calculate_pylint_score(pylint_data)
                        logger.info(f"📈 Pylint score: {results['pylint_score']}/10")
                    except json.JSONDecodeError:
                        logger.error("Failed to parse pylint output")
            except Exception as e:
                logger.error(f"Pylint analysis failed: {e}")
        
        # Log results
        self._log_analysis(results)
        return results
    
    def _calculate_pylint_score(self, pylint_data: List[Dict]) -> float:
        """Calculate overall code quality score from pylint issues"""
        if not pylint_data:
            return 10.0
        
        # Count severity of issues
        error_count = sum(1 for issue in pylint_data if issue.get('type') == 'error')
        warning_count = sum(1 for issue in pylint_data if issue.get('type') == 'warning')
        
        # Simple scoring: deduct points for errors and warnings
        score = 10.0
        score -= (error_count * 0.5)
        score -= (warning_count * 0.2)
        
        return max(0.0, score)
    
    async def auto_format_code(self, file_path: str) -> Dict[str, Any]:
        """Auto-format code using black"""
        if not self.tools_available.get('black'):
            return {'success': False, 'error': 'black not installed'}
        
        try:
            result = subprocess.run(
                ['black', file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'success': result.returncode == 0,
                'file': file_path,
                'output': result.stdout
            }
        except Exception as e:
            logger.error(f"Black formatting failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _log_analysis(self, results: Dict[str, Any]):
        """Log analysis results to file"""
        try:
            with open(self.refactor_log, 'a') as f:
                log_entry = {
                    'timestamp': results['timestamp'],
                    'project': results['project'],
                    'pylint_score': results['pylint_score'],
                    'file_count': results['file_count'],
                    'issue_count': len(results['issues'])
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log analysis: {e}")

# Global instance
code_refactor_executor = CodeRefactorExecutor()
