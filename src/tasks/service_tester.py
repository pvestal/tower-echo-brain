#!/usr/bin/env python3
"""
Service Tester - Test service modifications and verify quality
Automated testing for autonomous code changes
"""

import asyncio
import aiohttp
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import psutil

logger = logging.getLogger(__name__)

class ServiceTester:
    """Test service modifications and verify output quality"""
    
    def __init__(self):
        self.test_timeout = 300  # 5 minutes max per test
        self.quality_checks = {
            'anime': self._verify_anime_quality,
            'comfyui': self._verify_comfyui_quality,
            'echo': self._verify_echo_quality,
        }
        logger.info("ServiceTester initialized")
    
    async def test_service(self, service_name: str, test_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Test a service after modification
        
        Args:
            service_name: Service to test (e.g., 'anime-production', 'echo-brain')
            test_params: Optional test parameters
        
        Returns:
            Dict with success, test_results, quality_metrics
        """
        result = {
            'success': False,
            'service': service_name,
            'test_params': test_params,
            'status_before': None,
            'status_after': None,
            'quality_metrics': None,
            'output_files': [],
            'error': None
        }
        
        try:
            # Check service status before test
            result['status_before'] = await self._check_service_status(service_name)
            
            # Run service-specific test
            if 'anime' in service_name.lower():
                test_result = await self.test_anime_service(test_params or {})
            elif 'echo' in service_name.lower():
                test_result = await self.test_echo_service(test_params or {})
            elif 'comfyui' in service_name.lower():
                test_result = await self.test_comfyui_service(test_params or {})
            else:
                test_result = await self._generic_service_test(service_name, test_params or {})
            
            result.update(test_result)
            
            # Check service status after test
            result['status_after'] = await self._check_service_status(service_name)
            
            result['success'] = test_result.get('success', False)
            logger.info(f"Service test complete: {service_name} - {'PASS' if result['success'] else 'FAIL'}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error testing {service_name}: {e}")
        
        return result
    
    async def test_anime_service(self, params: Dict) -> Dict[str, Any]:
        """Test anime production service"""
        result = {
            'success': False,
            'test_type': 'anime_generation',
            'quality_metrics': {},
            'output_files': [],
            'error': None
        }
        
        try:
            # Default test parameters
            test_prompt = params.get('prompt', 'test anime scene with dramatic lighting')
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://127.0.0.1:8328/api/anime/generate',
                    json={'prompt': test_prompt},
                    timeout=aiohttp.ClientTimeout(total=self.test_timeout)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result['output_files'] = [data.get('output_path', '')]
                        
                        # Verify quality
                        if result['output_files'][0]:
                            quality = await self._verify_anime_quality(result['output_files'][0])
                            result['quality_metrics'] = quality
                            result['success'] = quality.get('passed', False)
                    else:
                        result['error'] = f"API returned {resp.status}"
        
        except asyncio.TimeoutError:
            result['error'] = f"Test timed out after {self.test_timeout}s"
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Anime test error: {e}")
        
        return result
    
    async def test_echo_service(self, params: Dict) -> Dict[str, Any]:
        """Test Echo Brain service"""
        result = {
            'success': False,
            'test_type': 'echo_chat',
            'quality_metrics': {},
            'error': None
        }
        
        try:
            test_message = params.get('message', 'test message for quality check')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://127.0.0.1:8309/api/echo/chat',
                    json={'message': test_message},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Verify response quality
                        quality = await self._verify_echo_quality(data)
                        result['quality_metrics'] = quality
                        result['success'] = quality.get('passed', False)
                    else:
                        result['error'] = f"API returned {resp.status}"
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Echo test error: {e}")
        
        return result
    
    async def test_comfyui_service(self, params: Dict) -> Dict[str, Any]:
        """Test ComfyUI service"""
        result = {
            'success': False,
            'test_type': 'comfyui_health',
            'error': None
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'http://127.0.0.1:8188/',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    result['success'] = (resp.status == 200)
                    if not result['success']:
                        result['error'] = f"ComfyUI returned {resp.status}"
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    async def _generic_service_test(self, service_name: str, params: Dict) -> Dict[str, Any]:
        """Generic service health check"""
        result = {
            'success': False,
            'test_type': 'health_check',
            'error': None
        }
        
        try:
            # Try to find service port from common patterns
            port_map = {
                'tower-kb': 8307,
                'tower-auth': 8088,
                'tower-dashboard': 8080,
            }
            
            port = port_map.get(service_name)
            if not port:
                result['error'] = f"Unknown service: {service_name}"
                return result
            
            async with aiohttp.ClientSession() as session:
                # Try health endpoint
                for path in ['/health', '/api/health', '/']:
                    try:
                        async with session.get(
                            f'http://127.0.0.1:{port}{path}',
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as resp:
                            if resp.status == 200:
                                result['success'] = True
                                return result
                    except:
                        continue
            
            result['error'] = 'Service not responding'
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    async def _verify_anime_quality(self, file_path: str) -> Dict[str, Any]:
        """Verify anime video quality"""
        quality = {
            'passed': False,
            'file_exists': False,
            'file_size_mb': 0,
            'resolution': None,
            'duration': None,
            'issues': []
        }
        
        try:
            path = Path(file_path)
            
            if not path.exists():
                quality['issues'].append('Output file not found')
                return quality
            
            quality['file_exists'] = True
            quality['file_size_mb'] = round(path.stat().st_size / (1024 * 1024), 2)
            
            # Use ffprobe to check video properties
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=width,height,duration',
                     '-of', 'json', str(path)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if data.get('streams'):
                        stream = data['streams'][0]
                        width = stream.get('width', 0)
                        height = stream.get('height', 0)
                        duration = float(stream.get('duration', 0))
                        
                        quality['resolution'] = f"{width}x{height}"
                        quality['duration'] = round(duration, 2)
                        
                        # Quality checks
                        if width < 512 or height < 512:
                            quality['issues'].append('Resolution too low')
                        if duration < 2.0:
                            quality['issues'].append('Video too short')
                        if quality['file_size_mb'] < 1:
                            quality['issues'].append('File size suspiciously small')
            
            except subprocess.TimeoutExpired:
                quality['issues'].append('ffprobe timeout')
            except Exception as e:
                quality['issues'].append(f'ffprobe error: {e}')
            
            # Pass if no issues
            quality['passed'] = (len(quality['issues']) == 0)
            
        except Exception as e:
            quality['issues'].append(f'Quality check error: {e}')
        
        return quality
    
    async def _verify_echo_quality(self, response_data: Dict) -> Dict[str, Any]:
        """Verify Echo response quality"""
        quality = {
            'passed': False,
            'has_response': False,
            'response_length': 0,
            'issues': []
        }
        
        try:
            response_text = response_data.get('response', '')
            
            if not response_text:
                quality['issues'].append('Empty response')
                return quality
            
            quality['has_response'] = True
            quality['response_length'] = len(response_text)
            
            # Quality checks
            if len(response_text) < 10:
                quality['issues'].append('Response too short')
            
            if 'error' in response_data:
                quality['issues'].append(f"Error in response: {response_data['error']}")
            
            # Pass if no issues
            quality['passed'] = (len(quality['issues']) == 0)
            
        except Exception as e:
            quality['issues'].append(f'Quality check error: {e}')
        
        return quality
    
    async def _verify_comfyui_quality(self, output_path: str) -> Dict[str, Any]:
        """Verify ComfyUI output quality"""
        # Similar to anime quality check
        return await self._verify_anime_quality(output_path)
    
    async def _check_service_status(self, service_name: str) -> Dict[str, Any]:
        """Check systemd service status"""
        status = {
            'running': False,
            'pid': None,
            'memory_mb': 0,
            'cpu_percent': 0,
            'error': None
        }
        
        try:
            # Normalize service name
            service_file = service_name
            if not service_file.endswith('.service'):
                service_file += '.service'
            
            # Check systemd status
            result = subprocess.run(
                ['systemctl', '--user', 'is-active', service_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            status['running'] = (result.stdout.strip() == 'active')
            
            # Get process info if running
            if status['running']:
                result = subprocess.run(
                    ['systemctl', '--user', 'show', service_file, '--property=MainPID'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                pid_match = result.stdout.strip().split('=')
                if len(pid_match) == 2 and pid_match[1].isdigit():
                    pid = int(pid_match[1])
                    status['pid'] = pid
                    
                    try:
                        proc = psutil.Process(pid)
                        status['memory_mb'] = round(proc.memory_info().rss / (1024 * 1024), 2)
                        status['cpu_percent'] = proc.cpu_percent(interval=0.1)
                    except psutil.NoSuchProcess:
                        pass
        
        except subprocess.TimeoutExpired:
            status['error'] = 'Status check timeout'
        except Exception as e:
            status['error'] = str(e)
        
        return status
    
    async def rollback_changes(self, backup_path: str, target_path: str) -> Dict[str, Any]:
        """Rollback changes if tests fail"""
        from .code_writer import get_code_writer
        
        code_writer = get_code_writer()
        return await code_writer.rollback(backup_path, target_path)
    
    async def verify_quality(self, file_path: str, expected: Dict) -> tuple[bool, Dict]:
        """
        Verify output quality against expectations
        
        Args:
            file_path: Path to output file
            expected: Dict of expected properties (resolution, duration, etc.)
        
        Returns:
            (passed, quality_metrics)
        """
        # Detect file type
        path = Path(file_path)
        if path.suffix in ['.mp4', '.avi', '.mov']:
            quality = await self._verify_anime_quality(file_path)
        else:
            quality = {'passed': path.exists(), 'file_exists': path.exists()}
        
        # Check against expectations
        if expected:
            for key, value in expected.items():
                if key in quality and quality[key] != value:
                    quality['passed'] = False
                    quality.setdefault('issues', []).append(
                        f"Expected {key}={value}, got {quality[key]}"
                    )
        
        return (quality['passed'], quality)

# Singleton instance
_service_tester = None

def get_service_tester() -> ServiceTester:
    """Get singleton ServiceTester instance"""
    global _service_tester
    if _service_tester is None:
        _service_tester = ServiceTester()
    return _service_tester
