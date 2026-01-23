#!/usr/bin/env python3
"""
Test Suite for Content Generation Bridge
=========================================

Comprehensive testing of the autonomous content generation system.
Tests integration between Echo Brain agents, ComfyUI workflows, and the anime production pipeline.

Test Categories:
- Unit tests for individual components
- Integration tests for full workflows
- Performance and stress tests
- Error handling and recovery tests

Author: Claude Code & Echo Brain System
Date: January 2026
"""

import asyncio
import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import unittest
from unittest.mock import Mock, patch, AsyncMock

import pytest
import httpx
import psycopg2

from content_generation_bridge import ContentGenerationBridge
from autonomous_content_coordinator import AutonomousContentCoordinator
from workflow_generator import FramePackWorkflowGenerator

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestContentGenerationBridge(unittest.IsolatedAsyncioTestCase):
    """Test cases for ContentGenerationBridge"""

    async def asyncSetUp(self):
        """Set up test environment"""
        self.test_config = {
            'echo_brain_url': 'http://localhost:8309',
            'comfyui_url': 'http://localhost:8188',
            'lora_path': '/tmp/test_loras',
            'output_path': '/tmp/test_output',
            'workflow_timeout': 60,
            'quality_threshold': 0.5,
            'max_concurrent_jobs': 1
        }

        # Create test directories
        Path(self.test_config['lora_path']).mkdir(parents=True, exist_ok=True)
        Path(self.test_config['output_path']).mkdir(parents=True, exist_ok=True)

        self.bridge = ContentGenerationBridge(self.test_config)

    async def test_scene_data_retrieval(self):
        """Test scene data retrieval from database"""
        # Mock database connection
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = Mock()
            mock_cursor.fetchone.return_value = {
                'id': 'test-scene-123',
                'title': 'Test Scene',
                'description': 'A test scene for validation',
                'characters': 'Mei, Hiroshi',
                'frame_count': 120,
                'fps': 24
            }
            mock_connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            scene_data = self.bridge._get_scene_data('test-scene-123')

            self.assertIsNotNone(scene_data)
            self.assertEqual(scene_data['id'], 'test-scene-123')
            self.assertEqual(scene_data['title'], 'Test Scene')

    @patch('httpx.AsyncClient.post')
    async def test_echo_brain_agent_integration(self, mock_post):
        """Test integration with Echo Brain agents"""
        # Mock successful agent response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'task': 'Generate workflow spec',
            'response': 'Workflow specification generated successfully',
            'agent_used': 'coding',
            'model': 'deepseek-coder-v2:16b'
        }
        mock_post.return_value = mock_response

        scene_data = {
            'id': 'test-scene',
            'description': 'Test scene description',
            'visual_description': 'Beautiful anime landscape',
            'characters': 'Mei',
            'shot_list': {},
            'character_lora_mapping': {'Mei': 'mei_character_v1'}
        }

        workflow_spec = await self.bridge._generate_workflow_with_agents(scene_data)

        self.assertIsInstance(workflow_spec, dict)
        self.assertIn('model', workflow_spec)
        mock_post.assert_called_once()

    @patch('httpx.AsyncClient.post')
    async def test_comfyui_workflow_submission(self, mock_post):
        """Test ComfyUI workflow submission"""
        # Mock successful ComfyUI response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'prompt_id': 'test-job-123'}
        mock_post.return_value = mock_response

        test_workflow = {'nodes': [], 'links': []}
        job_id = await self.bridge._submit_comfyui_workflow(test_workflow)

        self.assertEqual(job_id, 'test-job-123')
        mock_post.assert_called_once()

    def test_lora_selection_system(self):
        """Test LoRA selection based on character mapping"""
        # Create test LoRA files
        test_loras = ['mei_character_v1.safetensors', 'hiroshi_optimized_v1.safetensors']
        lora_path = Path(self.test_config['lora_path'])

        for lora_name in test_loras:
            (lora_path / lora_name).touch()

        character_mapping = {
            'Mei': 'mei_character_v1',
            'Hiroshi': 'hiroshi_optimized_v1',
            'Unknown': 'nonexistent_lora'
        }

        lora_config = self.bridge._select_character_loras(character_mapping)

        self.assertIn('Mei', lora_config)
        self.assertIn('Hiroshi', lora_config)
        self.assertNotIn('Unknown', lora_config)

        # Verify LoRA configuration structure
        mei_config = lora_config['Mei']
        self.assertEqual(mei_config['name'], 'mei_character_v1')
        self.assertEqual(mei_config['strength'], 0.8)

    async def test_quality_validation(self):
        """Test content quality validation system"""
        # Mock execution result with video outputs
        execution_result = {
            'success': True,
            'job_id': 'test-job',
            'outputs': {
                '23': {
                    'videos': [
                        {'filename': 'test_output.mp4', 'type': 'temp'}
                    ]
                }
            }
        }

        # Create mock output file
        output_path = Path(self.test_config['output_path']) / 'test_output.mp4'
        output_path.write_bytes(b'mock video data' * 100)  # Create file with reasonable size

        validation_result = await self.bridge._validate_generated_content(execution_result)

        self.assertTrue(validation_result['passed'])
        self.assertGreater(validation_result['quality_score'], 0.0)
        self.assertEqual(len(validation_result['issues']), 0)

    async def test_ssot_tracking_integration(self):
        """Test SSOT tracking system integration"""
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = Mock()
            mock_connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            # Test tracking creation
            tracking_id = self.bridge._create_ssot_tracking('test-scene', 'scene_generation')
            self.assertIsNotNone(tracking_id)

            # Test tracking update
            self.bridge._update_ssot_tracking(tracking_id, 'completed', {'result': 'success'})

            # Verify database calls
            self.assertEqual(mock_cursor.execute.call_count, 2)  # Create + Update


class TestWorkflowGenerator(unittest.TestCase):
    """Test cases for FramePackWorkflowGenerator"""

    def setUp(self):
        """Set up test environment"""
        self.generator = FramePackWorkflowGenerator()

    def test_basic_workflow_generation(self):
        """Test basic workflow generation"""
        workflow = self.generator.generate_anime_workflow(
            prompt="A beautiful anime girl in a garden",
            negative_prompt="blurry, low quality",
            parameters={
                'width': 704,
                'height': 544,
                'frame_count': 120,
                'fps': 24
            }
        )

        # Validate workflow structure
        self.assertIn('nodes', workflow)
        self.assertIn('links', workflow)
        self.assertIn('id', workflow)
        self.assertIn('version', workflow)

        # Check for essential nodes
        node_types = [node['type'] for node in workflow['nodes']]
        essential_types = ['DualCLIPLoader', 'VAELoader', 'LoadFramePackModel']

        for node_type in essential_types:
            self.assertIn(node_type, node_types, f"Missing essential node type: {node_type}")

    def test_lora_integration(self):
        """Test LoRA integration in workflows"""
        lora_configs = [
            {'name': 'mei_character_v1.safetensors', 'strength': 0.8, 'character': 'Mei'},
            {'name': 'hiroshi_optimized_v1.safetensors', 'strength': 0.7, 'character': 'Hiroshi'}
        ]

        workflow = self.generator.generate_anime_workflow(
            prompt="Mei and Hiroshi in conversation",
            lora_configs=lora_configs
        )

        # Check for LoRA loader nodes
        lora_nodes = [node for node in workflow['nodes'] if node['type'] == 'LoraLoader']
        self.assertEqual(len(lora_nodes), len(lora_configs))

        # Verify LoRA configuration
        for i, lora_node in enumerate(lora_nodes):
            expected_config = lora_configs[i]
            widgets = lora_node['widgets_values']
            self.assertEqual(widgets[0], expected_config['name'])
            self.assertEqual(widgets[1], expected_config['strength'])

    def test_image_conditioning_workflow(self):
        """Test workflow generation with start/end images"""
        workflow = self.generator.generate_anime_workflow(
            prompt="Animated sequence",
            start_image="start.png",
            end_image="end.png",
            parameters={'width': 512, 'height': 512}
        )

        # Check for image loading nodes
        load_image_nodes = [node for node in workflow['nodes'] if node['type'] == 'LoadImage']
        self.assertGreaterEqual(len(load_image_nodes), 2)

        # Check for image resize nodes
        resize_nodes = [node for node in workflow['nodes'] if node['type'] == 'ImageResize+']
        self.assertGreaterEqual(len(resize_nodes), 2)

        # Check for VAE encode nodes
        vae_encode_nodes = [node for node in workflow['nodes'] if node['type'] == 'VAEEncode']
        self.assertGreaterEqual(len(vae_encode_nodes), 2)


class TestAutonomousContentCoordinator(unittest.IsolatedAsyncioTestCase):
    """Test cases for AutonomousContentCoordinator"""

    async def asyncSetUp(self):
        """Set up test environment"""
        self.test_config = {
            'echo_brain_url': 'http://localhost:8309',
            'comfyui_url': 'http://localhost:8188',
            'max_concurrent_scenes': 2,
            'quality_threshold': 0.7,
            'agent_timeout': 30,
            'workflow_timeout': 300
        }
        self.coordinator = AutonomousContentCoordinator(self.test_config)

    @patch('httpx.AsyncClient.post')
    async def test_scene_analysis_with_reasoning_agent(self, mock_post):
        """Test scene analysis using ReasoningAgent"""
        # Mock ReasoningAgent response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Scene analysis: Complex scene with multiple characters requiring high-quality generation',
            'agent_used': 'reasoning',
            'model': 'deepseek-r1:8b'
        }
        mock_post.return_value = mock_response

        scene_data = {
            'id': 'test-scene',
            'title': 'Test Scene',
            'description': 'A complex anime scene',
            'visual_description': 'Beautiful landscape with characters',
            'characters': 'Mei, Hiroshi',
            'frame_count': 120,
            'fps': 24,
            'shot_list': {},
            'character_lora_mapping': {}
        }

        analysis = await self.coordinator._analyze_scene_with_reasoning_agent(scene_data)

        self.assertIn('scene_id', analysis)
        self.assertIn('complexity_assessment', analysis)
        self.assertIn('agent_analysis', analysis)
        mock_post.assert_called_once()

    @patch('httpx.AsyncClient.post')
    async def test_description_enhancement_with_narration_agent(self, mock_post):
        """Test description enhancement using NarrationAgent"""
        # Mock NarrationAgent response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Enhanced description: A serene moonlit garden with cherry blossoms, soft lighting, cinematic composition',
            'agent_used': 'narration',
            'model': 'gemma2:9b'
        }
        mock_post.return_value = mock_response

        scene_data = {
            'description': 'A garden scene',
            'visual_description': 'Characters in garden'
        }

        analysis = {
            'agent_analysis': {
                'response': 'Scene requires atmospheric lighting and detailed backgrounds'
            }
        }

        enhancement = await self.coordinator._enhance_scene_description(scene_data, analysis)

        self.assertIn('enhanced_description', enhancement)
        self.assertIn('original_description', enhancement)
        mock_post.assert_called_once()

    def test_scene_complexity_assessment(self):
        """Test scene complexity assessment logic"""
        # High complexity scene
        high_complexity_scene = {
            'frame_count': 300,  # > 10 seconds
            'characters': 'Mei, Hiroshi, Kenji',  # Multiple characters
        }
        complexity = self.coordinator._assess_scene_complexity(high_complexity_scene)
        self.assertEqual(complexity, 'high')

        # Medium complexity scene
        medium_complexity_scene = {
            'frame_count': 120,  # 5 seconds
            'characters': 'Mei, Hiroshi, Kenji',  # Multiple characters
        }
        complexity = self.coordinator._assess_scene_complexity(medium_complexity_scene)
        self.assertEqual(complexity, 'medium')

        # Low complexity scene
        low_complexity_scene = {
            'frame_count': 120,  # 5 seconds
            'characters': 'Mei',  # Single character
        }
        complexity = self.coordinator._assess_scene_complexity(low_complexity_scene)
        self.assertEqual(complexity, 'low')

    def test_generation_batch_creation(self):
        """Test generation batch creation logic"""
        scenes = [
            {'id': f'scene-{i}', 'scene_number': i} for i in range(7)
        ]

        batches = self.coordinator._create_generation_batches(scenes, max_concurrent=3)

        self.assertEqual(len(batches), 3)  # 7 scenes / 3 per batch = 3 batches
        self.assertEqual(len(batches[0]), 3)  # First batch full
        self.assertEqual(len(batches[1]), 3)  # Second batch full
        self.assertEqual(len(batches[2]), 1)  # Third batch partial

    def test_duration_estimation(self):
        """Test generation duration estimation"""
        scenes = [{'id': f'scene-{i}'} for i in range(5)]
        estimated_duration = self.coordinator._estimate_generation_duration(scenes)

        # Should be 5 scenes * 360 seconds = 1800 seconds
        self.assertEqual(estimated_duration, 1800)


class TestIntegrationScenarios(unittest.IsolatedAsyncioTestCase):
    """Integration tests for complete workflows"""

    async def asyncSetUp(self):
        """Set up integration test environment"""
        self.bridge = ContentGenerationBridge()
        self.coordinator = AutonomousContentCoordinator()

    @pytest.mark.integration
    @patch('psycopg2.connect')
    @patch('httpx.AsyncClient.post')
    async def test_end_to_end_scene_generation(self, mock_http, mock_db):
        """Test complete scene generation workflow"""
        # Mock database responses
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {
            'id': 'integration-test-scene',
            'title': 'Integration Test Scene',
            'description': 'A test scene for end-to-end validation',
            'visual_description': 'Beautiful anime landscape',
            'characters': 'Mei',
            'frame_count': 120,
            'fps': 24,
            'character_lora_mapping': {'Mei': 'mei_character_v1'}
        }
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock Echo Brain agent responses
        agent_responses = [
            # ReasoningAgent response
            Mock(status_code=200, json=lambda: {
                'response': 'Scene analysis complete',
                'agent_used': 'reasoning'
            }),
            # NarrationAgent response
            Mock(status_code=200, json=lambda: {
                'response': 'Enhanced visual description',
                'agent_used': 'narration'
            }),
            # CodingAgent response
            Mock(status_code=200, json=lambda: {
                'response': 'Workflow specification generated',
                'agent_used': 'coding'
            }),
            # ComfyUI submission response
            Mock(status_code=200, json=lambda: {'prompt_id': 'test-job-123'})
        ]
        mock_http.side_effect = agent_responses

        # Mock ComfyUI execution monitoring
        with patch.object(self.bridge, '_monitor_workflow_execution') as mock_monitor:
            mock_monitor.return_value = {
                'success': True,
                'job_id': 'test-job-123',
                'outputs': {'23': {'videos': [{'filename': 'test_output.mp4'}]}},
                'execution_time': 300
            }

            # Execute end-to-end test
            result = await self.coordinator.autonomous_scene_generation('integration-test-scene')

            # Verify result structure
            self.assertIn('scene_id', result)
            self.assertIn('success', result)
            self.assertTrue(result['success'])

    @pytest.mark.stress
    async def test_concurrent_generation_stress(self):
        """Stress test with multiple concurrent generations"""
        # Create multiple mock scenes
        scene_ids = [f'stress-test-scene-{i}' for i in range(5)]

        with patch.object(self.coordinator, 'autonomous_scene_generation') as mock_generation:
            mock_generation.return_value = {'success': True, 'scene_id': 'test'}

            # Execute concurrent generations
            tasks = [
                self.coordinator.autonomous_scene_generation(scene_id)
                for scene_id in scene_ids
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all completed without exceptions
            for result in results:
                self.assertIsInstance(result, dict)
                self.assertTrue(result.get('success', False))

    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        with patch.object(self.bridge, '_get_scene_data') as mock_get_scene:
            # Test scene not found
            mock_get_scene.return_value = None

            with self.assertRaises(ValueError):
                await self.bridge.process_scene_generation_request('nonexistent-scene')

        # Test network error recovery
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = httpx.RequestError("Network error")

            with self.assertRaises(Exception):
                await self.coordinator._call_echo_brain_agent("test query", "reasoning")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance and benchmark tests"""

    def test_workflow_generation_performance(self):
        """Benchmark workflow generation performance"""
        generator = FramePackWorkflowGenerator()

        import time
        start_time = time.time()

        for i in range(10):
            workflow = generator.generate_anime_workflow(
                prompt=f"Test scene {i}",
                parameters={'width': 704, 'height': 544}
            )

        end_time = time.time()
        average_time = (end_time - start_time) / 10

        # Should generate workflows quickly (< 0.1 seconds each)
        self.assertLess(average_time, 0.1, f"Workflow generation too slow: {average_time:.3f}s")

    def test_memory_usage_stability(self):
        """Test memory usage stability during operations"""
        import gc
        import sys

        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform operations
        generator = FramePackWorkflowGenerator()
        for i in range(50):
            workflow = generator.generate_anime_workflow(f"Test {i}")
            del workflow

        # Check memory usage after cleanup
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage should not grow significantly
        growth_rate = (final_objects - initial_objects) / initial_objects
        self.assertLess(growth_rate, 0.1, f"Memory usage grew too much: {growth_rate:.2%}")


def run_test_suite():
    """Run the complete test suite"""
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestContentGenerationBridge))
    suite.addTest(unittest.makeSuite(TestWorkflowGenerator))
    suite.addTest(unittest.makeSuite(TestAutonomousContentCoordinator))
    suite.addTest(unittest.makeSuite(TestPerformanceBenchmarks))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


async def run_integration_tests():
    """Run integration tests separately"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestIntegrationScenarios))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Content Generation Bridge Test Suite")
    print("=" * 60)

    # Run unit tests
    print("\n📝 Unit Tests")
    print("-" * 30)
    unit_success = run_test_suite()

    # Run integration tests
    print("\n🔗 Integration Tests")
    print("-" * 30)
    integration_success = asyncio.run(run_integration_tests())

    # Summary
    print("\n📊 Test Summary")
    print("-" * 30)
    print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
    print(f"Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")

    if unit_success and integration_success:
        print("\n🎉 All tests passed! Content Generation Bridge is ready for production.")
        exit(0)
    else:
        print("\n⚠️  Some tests failed. Review the output above for details.")
        exit(1)