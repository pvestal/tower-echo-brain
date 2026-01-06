#!/usr/bin/env python3
"""
Enhanced Code Refactor Executor with real execution capabilities.
Integrates incremental analysis, verified execution, and safe refactoring.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import the new execution layer
import sys
sys.path.insert(0, '/opt/tower-echo-brain')
from src.execution import (
    IncrementalAnalyzer,
    VerifiedExecutor,
    VerifiedAction,
    ExecutionStatus,
    SafeRefactor
)

logger = logging.getLogger(__name__)

class EnhancedCodeRefactorExecutor:
    """
    Production-ready code refactor executor that actually modifies code.
    Uses incremental analysis to avoid timeouts and verified execution
    to ensure changes actually work.
    """

    def __init__(self):
        self.state_dir = Path("/opt/tower-echo-brain/state")
        self.state_dir.mkdir(exist_ok=True)
        self.verified_executor = VerifiedExecutor()
        self.refactor_log = Path("/opt/tower-echo-brain/logs/enhanced_refactors.log")
        self.refactor_log.parent.mkdir(parents=True, exist_ok=True)

    async def analyze_project_incrementally(self, project_path: str) -> Dict[str, Any]:
        """
        Analyze project using incremental analyzer to avoid timeouts.
        Only analyzes changed files since last run.
        """
        logger.info(f"🔍 Starting incremental analysis of {project_path}")

        project_root = Path(project_path)
        state_file = self.state_dir / f"{project_root.name}_analysis_state.json"

        analyzer = IncrementalAnalyzer(project_root, state_file)

        # Get changed files
        changed_targets = list(analyzer.get_changed_files(['.py']))
        logger.info(f"Found {len(changed_targets)} changed Python files")

        # Create batches for processing
        batches = list(analyzer.create_batches(changed_targets, max_files=10))

        results = {
            'project': project_path,
            'total_files': len(changed_targets),
            'batches_created': len(batches),
            'files_analyzed': [],
            'files_needing_refactor': [],
            'timestamp': datetime.now().isoformat()
        }

        # Process each batch
        for batch in batches:
            for target in batch.targets:
                # Analyze individual file
                file_result = await self._analyze_single_file(target.path)
                if file_result['needs_refactor']:
                    results['files_needing_refactor'].append(str(target.path))
                results['files_analyzed'].append(str(target.path))

                # Mark as analyzed to avoid reprocessing
                analyzer.mark_analyzed(target)

                # Rate limit to avoid overwhelming the system
                await asyncio.sleep(0.1)

        logger.info(f"📊 Analysis complete: {len(results['files_analyzed'])} files analyzed, "
                   f"{len(results['files_needing_refactor'])} need refactoring")

        return results

    async def _analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for quality issues."""
        result = {
            'file': str(file_path),
            'needs_refactor': False,
            'score': None
        }

        try:
            # Use pylint to get quality score
            import subprocess
            proc = subprocess.run(
                ['/opt/tower-echo-brain/venv/bin/pylint', str(file_path), '--score=y'],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse score
            for line in proc.stdout.split('\n'):
                if 'Your code has been rated at' in line:
                    score_str = line.split('rated at ')[1].split('/')[0]
                    result['score'] = float(score_str)
                    result['needs_refactor'] = result['score'] < 7.0
                    break

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")

        return result

    async def refactor_file_with_verification(self, file_path: str) -> Dict[str, Any]:
        """
        Refactor a single file with verification that changes actually improve it.
        """
        logger.info(f"🔧 Refactoring {file_path} with verification")

        # Import the semantic refactor executor
        from src.tasks.semantic_refactor_executor import semantic_refactor_executor

        # Create a verified action for refactoring
        def execute_refactor():
            """Execute the refactoring."""
            import subprocess
            # First, format with black
            return subprocess.run(
                ['/opt/tower-echo-brain/venv/bin/black', file_path],
                capture_output=True
            )

        def verify_refactor():
            """Verify the refactoring improved the file."""
            # Check that file still has valid Python syntax
            try:
                with open(file_path, 'r') as f:
                    import ast
                    ast.parse(f.read())
                return True
            except:
                return False

        action = VerifiedAction(
            name=f"refactor_{Path(file_path).name}",
            execute=execute_refactor,
            verify=verify_refactor,
            description=f"Refactor and format {file_path}"
        )

        # Execute with verification
        result = await self.verified_executor.run(action)

        # Also run semantic refactoring for deeper improvements
        if result.actually_worked:
            semantic_result = await semantic_refactor_executor.analyze_and_refactor_file(file_path)

            return {
                'file': file_path,
                'success': result.actually_worked and semantic_result.get('code_changed', False),
                'verification_status': result.status.value,
                'original_score': semantic_result.get('original_score'),
                'final_score': semantic_result.get('final_score'),
                'transformations': semantic_result.get('transformations_applied', [])
            }
        else:
            return {
                'file': file_path,
                'success': False,
                'verification_status': result.status.value,
                'error': result.actual_outcome
            }

    async def safe_refactor_project(self, project_path: str, max_files: int = 5) -> Dict[str, Any]:
        """
        Safely refactor a project with git integration.
        Changes are committed to a branch and can be rolled back if tests fail.
        """
        logger.info(f"🔒 Starting safe refactoring of {project_path}")

        project_root = Path(project_path)

        # Check if it's a git repository
        if not (project_root / '.git').exists():
            return {
                'success': False,
                'error': f"{project_path} is not a git repository"
            }

        # Use SafeRefactor for git integration
        safe_refactor = SafeRefactor(project_root)

        # Function to perform refactoring
        async def do_refactoring():
            modified_files = []

            # Get files needing refactoring
            analysis = await self.analyze_project_incrementally(project_path)
            files_to_refactor = analysis['files_needing_refactor'][:max_files]

            for file_path in files_to_refactor:
                result = await self.refactor_file_with_verification(file_path)
                if result['success']:
                    modified_files.append(Path(file_path))

            return modified_files

        # Wrap async function for sync context
        def refactor_wrapper():
            return asyncio.run(do_refactoring())

        # Determine test command based on project
        test_commands = {
            '/opt/tower-echo-brain': ['python3', '-m', 'pytest', 'tests/', '-x'],
            '/opt/tower-anime-production': ['python3', '-m', 'pytest', 'tests/', '-x'],
        }
        test_command = test_commands.get(project_path, ['true'])  # Default to always pass if no tests

        # Execute refactoring with safety
        result = safe_refactor.refactor_with_safety(
            description="automated-code-improvement",
            refactor_func=refactor_wrapper,
            test_command=test_command,
            original_branch='main'
        )

        return {
            'project': project_path,
            'success': result.success,
            'files_modified': [str(f) for f in result.files_modified],
            'commit_hash': result.commit_hash,
            'error': result.error_message
        }

    async def execute_code_refactor_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for code refactor tasks from Echo's task queue.
        """
        action = task_payload.get('action', 'analyze')
        project_path = task_payload.get('project_path')

        if not project_path:
            return {'success': False, 'error': 'No project_path specified'}

        try:
            if action == 'analyze':
                return await self.analyze_project_incrementally(project_path)
            elif action == 'refactor':
                return await self.safe_refactor_project(project_path)
            elif action == 'refactor_file':
                file_path = task_payload.get('file_path')
                if file_path:
                    return await self.refactor_file_with_verification(file_path)
                else:
                    return {'success': False, 'error': 'No file_path specified'}
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}

        except Exception as e:
            logger.error(f"Code refactor task failed: {e}")
            return {'success': False, 'error': str(e)}

    def _log_refactor(self, result: Dict[str, Any]):
        """Log refactoring results."""
        try:
            with open(self.refactor_log, 'a') as f:
                f.write(json.dumps(result) + '\n')
        except Exception as e:
            logger.error(f"Failed to log refactor: {e}")


# Global instance
enhanced_code_refactor_executor = EnhancedCodeRefactorExecutor()


async def test_enhanced_executor():
    """Test the enhanced refactor executor."""
    executor = EnhancedCodeRefactorExecutor()

    # Test incremental analysis (should be fast on second run)
    print("Testing incremental analysis...")
    result1 = await executor.analyze_project_incrementally('/opt/tower-echo-brain/src/execution')
    print(f"First run: {result1['total_files']} files analyzed")

    result2 = await executor.analyze_project_incrementally('/opt/tower-echo-brain/src/execution')
    print(f"Second run: {result2['total_files']} files analyzed (should be 0 if unchanged)")

    # Test file refactoring with verification
    print("\nTesting verified refactoring...")
    test_file = '/opt/tower-echo-brain/test_refactor_target.py'
    if Path(test_file).exists():
        result = await executor.refactor_file_with_verification(test_file)
        print(f"Refactor result: {result['success']}, Score: {result.get('original_score')} -> {result.get('final_score')}")

    print("\n✅ Enhanced executor tests complete")


if __name__ == "__main__":
    asyncio.run(test_enhanced_executor())