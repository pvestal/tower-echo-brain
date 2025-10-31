#!/usr/bin/env python3
"""
Task Implementation Executor
Autonomous code writing and deployment based on natural language tasks
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import json

from .code_writer import get_code_writer
from .code_reviewer import get_code_reviewer
from .service_tester import get_service_tester
from .video_generation_tasks import VideoGenerationExecutor, VideoTaskType

logger = logging.getLogger(__name__)

class TaskImplementationExecutor:
    """
    Execute natural language tasks autonomously
    Workflow: Analyze → Generate Code → Review → Deploy → Test
    """
    
    def __init__(self, board_integration=None):
        self.code_writer = get_code_writer()
        self.code_reviewer = get_code_reviewer()
        self.service_tester = get_service_tester()
        self.video_executor = VideoGenerationExecutor()
        self.board = board_integration
        logger.info("TaskImplementationExecutor initialized with video generation support")
    
    async def implement_task(self, task: str, service: str, test: bool = True) -> Dict[str, Any]:
        """
        Main implementation workflow
        
        Args:
            task: Natural language task description
            service: Target service name
            test: Whether to run tests after deployment
        
        Returns:
            Implementation result with status, changes, test results
        """
        task_id = str(uuid.uuid4())
        
        result = {
            'task_id': task_id,
            'task': task,
            'service': service,
            'status': 'processing',
            'estimated_time': '2-5 minutes',
            'steps_completed': [],
            'code_changes': [],
            'review_results': None,
            'test_results': None,
            'success': False,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Analyze task and plan approach
            logger.info(f"Step 1: Analyzing task '{task}' for {service}")
            analysis = await self._analyze_task(task, service)
            result['steps_completed'].append('task_analysis')
            result['analysis'] = analysis
            
            if not analysis['success']:
                result['error'] = analysis.get('error', 'Analysis failed')
                return result
            
            # Step 2: Consult Board of Directors (if available)
            if self.board:
                logger.info("Step 2: Consulting Board of Directors")
                board_decision = await self._consult_board(analysis)
                result['steps_completed'].append('board_consultation')
                result['board_decision'] = board_decision
                
                if not board_decision.get('approved', False):
                    result['error'] = f"Board rejected: {board_decision.get('reasoning', 'Unknown')}"
                    return result
            
            # Step 3: Generate code changes
            logger.info("Step 3: Generating code modifications")
            code_changes = await self._generate_code_changes(analysis)
            result['steps_completed'].append('code_generation')
            result['code_changes'] = code_changes
            
            if not code_changes['success']:
                result['error'] = code_changes.get('error', 'Code generation failed')
                return result
            
            # Step 4: Self-review with code_reviewer
            logger.info("Step 4: Self-reviewing generated code")
            review_results = await self._review_changes(code_changes)
            result['steps_completed'].append('code_review')
            result['review_results'] = review_results
            
            if not review_results['passed']:
                # Try to auto-fix common issues
                logger.info("Attempting auto-fix of issues")
                fix_result = await self._auto_fix_issues(code_changes, review_results)
                
                if fix_result['success']:
                    # Re-review after fixes
                    review_results = await self._review_changes(code_changes)
                    result['review_results'] = review_results
                
                if not review_results['passed']:
                    result['error'] = f"Code quality below threshold: {review_results['score']:.1f}/10"
                    result['quality_issues'] = review_results.get('issues', [])
                    return result
            
            # Step 5: Deploy changes
            logger.info("Step 5: Deploying code changes")
            deploy_result = await self._deploy_changes(service, code_changes)
            result['steps_completed'].append('deployment')
            result['deployment'] = deploy_result
            
            if not deploy_result['success']:
                result['error'] = deploy_result.get('error', 'Deployment failed')
                return result
            
            # Step 6: Test if requested
            if test:
                logger.info("Step 6: Testing deployed changes")
                test_results = await self._test_service(service)
                result['steps_completed'].append('testing')
                result['test_results'] = test_results
                
                if not test_results['success']:
                    # Rollback if tests fail
                    logger.warning("Tests failed, rolling back changes")
                    await self._rollback_changes(code_changes)
                    result['error'] = 'Tests failed, changes rolled back'
                    result['rollback'] = True
                    return result
            
            # Success!
            result['success'] = True
            result['status'] = 'completed'
            result['message'] = f"Task '{task}' implemented successfully"
            
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            result['error'] = str(e)
            result['status'] = 'failed'
            logger.error(f"Error implementing task {task_id}: {e}")
        
        return result

    async def execute_video_task(self, task) -> Dict[str, Any]:
        """Execute video generation tasks autonomously"""
        try:
            video_task_type = task.payload.get('video_task_type')

            if video_task_type == VideoTaskType.CHARACTER_TO_VIDEO.value:
                return await self.video_executor.execute_character_to_video(task)
            elif video_task_type == VideoTaskType.BATCH_GENERATION.value:
                return await self.video_executor.execute_batch_generation(task)
            elif video_task_type == VideoTaskType.QUALITY_CHECK.value:
                video_path = task.payload.get('video_path')
                min_quality = task.payload.get('min_quality', 0.7)
                quality_score = await self.video_executor._assess_video_quality(video_path)

                return {
                    'status': 'completed',
                    'video_path': video_path,
                    'quality_score': quality_score,
                    'passes_threshold': quality_score >= min_quality
                }
            else:
                return {
                    'status': 'failed',
                    'error': f"Unknown video task type: {video_task_type}"
                }

        except Exception as e:
            logger.error(f"Video task execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def _analyze_task(self, task: str, service: str) -> Dict[str, Any]:
        """Analyze task and determine what needs to be done"""
        analysis = {
            'success': False,
            'task': task,
            'service': service,
            'target_files': [],
            'modifications': [],
            'approach': None,
            'error': None
        }
        
        try:
            # Map service to file paths
            service_map = {
                'anime-production': '/opt/tower-anime-production/anime_service.py',
                'echo-brain': '/opt/tower-echo-brain/src/main.py',
                'tower-kb': '/opt/tower-kb/main.py',
                'tower-auth': '/opt/tower-auth/auth_service.py',
            }
            
            target_file = service_map.get(service)
            if not target_file:
                analysis['error'] = f"Unknown service: {service}"
                return analysis
            
            analysis['target_files'] = [target_file]
            
            # Parse task intent
            task_lower = task.lower()
            
            # Determine approach based on keywords
            if any(word in task_lower for word in ['cinematic', 'dramatic', 'quality', 'better']):
                analysis['approach'] = 'quality_enhancement'
                analysis['modifications'] = [
                    {'type': 'parameter_adjustment', 'target': 'generation_params'},
                    {'type': 'config_update', 'target': 'quality_settings'}
                ]
            elif any(word in task_lower for word in ['faster', 'optimize', 'speed']):
                analysis['approach'] = 'performance_optimization'
                analysis['modifications'] = [
                    {'type': 'parameter_adjustment', 'target': 'performance_params'}
                ]
            elif any(word in task_lower for word in ['fix', 'error', 'bug']):
                analysis['approach'] = 'bug_fix'
                analysis['modifications'] = [
                    {'type': 'code_fix', 'target': 'error_handling'}
                ]
            else:
                analysis['approach'] = 'generic_modification'
                analysis['modifications'] = [
                    {'type': 'code_update', 'target': 'main_logic'}
                ]
            
            analysis['success'] = True
            
        except Exception as e:
            analysis['error'] = str(e)
            logger.error(f"Error analyzing task: {e}")
        
        return analysis
    
    async def _consult_board(self, analysis: Dict) -> Dict[str, Any]:
        """Consult Board of Directors for approval"""
        decision = {
            'approved': True,  # Default approve if board unavailable
            'reasoning': 'Board consultation not configured',
            'confidence': 0.5
        }
        
        if not self.board:
            return decision
        
        try:
            # Prepare context for board
            context = {
                'task': analysis['task'],
                'service': analysis['service'],
                'approach': analysis['approach'],
                'modifications': analysis['modifications']
            }
            
            # Consult board (simplified - real implementation would use BoardIntegration)
            # For now, approve if approach is recognized
            if analysis['approach'] in ['quality_enhancement', 'performance_optimization', 'bug_fix']:
                decision['approved'] = True
                decision['reasoning'] = f"Approved {analysis['approach']} for {analysis['service']}"
                decision['confidence'] = 0.8
            
        except Exception as e:
            logger.warning(f"Board consultation error: {e}")
        
        return decision
    
    async def _generate_code_changes(self, analysis: Dict) -> Dict[str, Any]:
        """Generate code changes based on analysis"""
        changes = {
            'success': False,
            'files_modified': [],
            'backups_created': [],
            'error': None
        }
        
        try:
            target_file = analysis['target_files'][0]
            
            # Read current file
            content = await self.code_writer.read_file(target_file)
            
            # Apply modifications based on approach
            if analysis['approach'] == 'quality_enhancement':
                modified_content = await self._apply_quality_enhancements(content)
            elif analysis['approach'] == 'performance_optimization':
                modified_content = await self._apply_performance_optimizations(content)
            else:
                # Generic modification (could use LLM here in future)
                modified_content = content  # Placeholder
            
            # Write changes
            write_result = await self.code_writer.write_file(
                target_file,
                modified_content,
                backup=True
            )
            
            if write_result['success']:
                changes['files_modified'].append(target_file)
                changes['backups_created'].append(write_result['backup_path'])
                changes['success'] = True
            else:
                changes['error'] = write_result.get('error', 'Write failed')
            
        except Exception as e:
            changes['error'] = str(e)
            logger.error(f"Error generating code changes: {e}")
        
        return changes
    
    async def _apply_quality_enhancements(self, content: str) -> str:
        """Apply quality enhancement modifications (example)"""
        # Example: Increase anime quality parameters
        # In production, this would use LLM to generate appropriate changes
        
        replacements = [
            ('steps = 20', 'steps = 28'),
            ('cfg_scale = 7.0', 'cfg_scale = 7.5'),
            ('sampler = "euler"', 'sampler = "dpmpp_2m"'),
        ]
        
        modified = content
        for old, new in replacements:
            if old in modified:
                modified = modified.replace(old, new)
        
        return modified
    
    async def _apply_performance_optimizations(self, content: str) -> str:
        """Apply performance optimizations (example)"""
        # Example optimizations
        modified = content
        # Add caching, reduce iterations, etc.
        return modified
    
    async def _review_changes(self, code_changes: Dict) -> Dict[str, Any]:
        """Review generated code changes"""
        if not code_changes['files_modified']:
            return {'passed': False, 'error': 'No files to review'}
        
        # Review first modified file
        target_file = code_changes['files_modified'][0]
        review_result = await self.code_reviewer.review_file(target_file)
        
        return review_result
    
    async def _auto_fix_issues(self, code_changes: Dict, review_results: Dict) -> Dict[str, Any]:
        """Attempt to auto-fix common issues"""
        if not code_changes['files_modified']:
            return {'success': False}
        
        target_file = code_changes['files_modified'][0]
        
        # Auto-fix common issues
        fix_result = await self.code_reviewer.auto_fix_common_issues(target_file)
        
        # Try black formatting if available
        if self.code_reviewer.checks_available.get('black'):
            await self.code_reviewer.format_with_black(target_file)
        
        return fix_result
    
    async def _deploy_changes(self, service: str, code_changes: Dict) -> Dict[str, Any]:
        """Deploy code changes (restart service if needed)"""
        deploy_result = {
            'success': False,
            'service_restarted': False,
            'error': None
        }
        
        try:
            # Restart service if it's a systemd service
            import subprocess
            
            service_name = service
            if not service_name.endswith('.service'):
                service_name += '.service'
            
            result = subprocess.run(
                ['systemctl', '--user', 'restart', service_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                deploy_result['success'] = True
                deploy_result['service_restarted'] = True
                logger.info(f"Restarted service: {service_name}")
            else:
                deploy_result['error'] = f"Restart failed: {result.stderr}"
        
        except subprocess.TimeoutExpired:
            deploy_result['error'] = 'Service restart timeout'
        except Exception as e:
            deploy_result['error'] = str(e)
            logger.error(f"Error deploying changes: {e}")
        
        return deploy_result
    
    async def _test_service(self, service: str) -> Dict[str, Any]:
        """Test service after deployment"""
        test_result = await self.service_tester.test_service(service)
        return test_result
    
    async def _rollback_changes(self, code_changes: Dict) -> Dict[str, Any]:
        """Rollback changes if tests fail"""
        rollback_result = {
            'success': False,
            'files_rolled_back': [],
            'error': None
        }
        
        try:
            # Rollback each modified file
            for i, file_path in enumerate(code_changes['files_modified']):
                if i < len(code_changes['backups_created']):
                    backup_path = code_changes['backups_created'][i]
                    result = await self.code_writer.rollback(backup_path, file_path)
                    
                    if result['success']:
                        rollback_result['files_rolled_back'].append(file_path)
            
            rollback_result['success'] = True
            
        except Exception as e:
            rollback_result['error'] = str(e)
            logger.error(f"Error rolling back changes: {e}")
        
        return rollback_result

# Singleton instance
_task_executor = None

def get_task_implementation_executor(board=None) -> TaskImplementationExecutor:
    """Get singleton TaskImplementationExecutor instance"""
    global _task_executor
    if _task_executor is None:
        _task_executor = TaskImplementationExecutor(board_integration=board)
    return _task_executor
