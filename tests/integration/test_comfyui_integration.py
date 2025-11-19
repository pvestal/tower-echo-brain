"""
ComfyUI Integration Reliability Tests
Tests the stability and reliability of ComfyUI service integration.
"""

import asyncio
import json
import os
import time
import uuid
import websockets
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import base64

import httpx
import pytest
from PIL import Image


@dataclass
class ComfyUITestResult:
    """Result of a ComfyUI integration test"""
    test_name: str
    success: bool
    duration: float
    error: Optional[str] = None
    workflow_id: Optional[str] = None
    output_files: List[str] = None
    metadata: Optional[Dict] = None


class ComfyUIIntegrationTester:
    """Comprehensive testing for ComfyUI integration"""

    def __init__(self,
                 comfyui_endpoint: str = "http://***REMOVED***:8188",
                 timeout: int = 300):
        self.endpoint = comfyui_endpoint
        self.timeout = timeout
        self.ws_endpoint = comfyui_endpoint.replace("http", "ws")

        # Test workflows for different scenarios
        self.test_workflows = {
            "simple_image": {
                "workflow": self._get_simple_image_workflow(),
                "expected_outputs": 1,
                "max_duration": 60
            },
            "anime_character": {
                "workflow": self._get_anime_character_workflow(),
                "expected_outputs": 1,
                "max_duration": 90
            },
            "batch_generation": {
                "workflow": self._get_batch_workflow(),
                "expected_outputs": 4,
                "max_duration": 180
            },
            "video_generation": {
                "workflow": self._get_video_workflow(),
                "expected_outputs": 1,
                "max_duration": 300
            }
        }

    def _get_simple_image_workflow(self) -> Dict:
        """Simple image generation workflow for basic testing"""
        return {
            "3": {
                "inputs": {
                    "seed": 42,
                    "steps": 10,
                    "cfg": 7,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "v1-5-pruned-emaonly.ckpt"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": "simple test image, basic style",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": "bad quality, blurry",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "test_simple",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

    def _get_anime_character_workflow(self) -> Dict:
        """Anime character generation workflow"""
        return {
            "3": {
                "inputs": {
                    "seed": 12345,
                    "steps": 20,
                    "cfg": 8,
                    "sampler_name": "dpmpp_2m_karras",
                    "scheduler": "karras",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "animePastelDream_softBakedVae.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": 768,
                    "height": 768,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": "anime character, detailed face, high quality, masterpiece",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": "low quality, bad anatomy, blurry, distorted",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "test_anime",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

    def _get_batch_workflow(self) -> Dict:
        """Batch generation workflow for testing throughput"""
        workflow = self._get_simple_image_workflow()
        # Modify to generate 4 images
        workflow["5"]["inputs"]["batch_size"] = 4
        workflow["9"]["inputs"]["filename_prefix"] = "test_batch"
        return workflow

    def _get_video_workflow(self) -> Dict:
        """Video generation workflow for testing video capabilities"""
        return {
            # Simplified video workflow - would need actual AnimateDiff nodes
            "3": {
                "inputs": {
                    "seed": 54321,
                    "steps": 15,
                    "cfg": 7,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "v1-5-pruned-emaonly.ckpt"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 8  # 8 frames
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": "anime character walking, motion",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": "static, still, low quality",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "test_video",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

    async def test_service_availability(self) -> ComfyUITestResult:
        """Test basic ComfyUI service availability"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Test basic endpoint
                response = await client.get(f"{self.endpoint}")
                if response.status_code != 200:
                    raise Exception(f"Service unavailable: {response.status_code}")

                # Test system stats endpoint
                response = await client.get(f"{self.endpoint}/system_stats")
                if response.status_code == 200:
                    stats = response.json()
                    metadata = {
                        "system_stats": stats,
                        "gpu_available": "gpu" in str(stats).lower()
                    }
                else:
                    metadata = {"system_stats": "unavailable"}

                return ComfyUITestResult(
                    test_name="service_availability",
                    success=True,
                    duration=time.time() - start_time,
                    metadata=metadata
                )

        except Exception as e:
            return ComfyUITestResult(
                test_name="service_availability",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def test_workflow_execution(self, workflow_name: str) -> ComfyUITestResult:
        """Test execution of a specific workflow"""
        if workflow_name not in self.test_workflows:
            return ComfyUITestResult(
                test_name=f"workflow_{workflow_name}",
                success=False,
                duration=0,
                error=f"Unknown workflow: {workflow_name}"
            )

        start_time = time.time()
        workflow_config = self.test_workflows[workflow_name]
        workflow = workflow_config["workflow"]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Submit workflow
                response = await client.post(f"{self.endpoint}/prompt", json={"prompt": workflow})

                if response.status_code != 200:
                    raise Exception(f"Failed to submit workflow: {response.status_code} - {response.text}")

                result = response.json()
                prompt_id = result.get("prompt_id")

                if not prompt_id:
                    raise Exception("No prompt_id returned")

                # Wait for completion
                output_files = await self._wait_for_workflow_completion(
                    prompt_id,
                    max_wait=workflow_config["max_duration"]
                )

                if not output_files:
                    raise Exception("No output files generated")

                expected_outputs = workflow_config["expected_outputs"]
                if len(output_files) < expected_outputs:
                    raise Exception(f"Expected {expected_outputs} outputs, got {len(output_files)}")

                # Validate output files
                valid_files = []
                for file_path in output_files:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        valid_files.append(file_path)

                if not valid_files:
                    raise Exception("No valid output files generated")

                return ComfyUITestResult(
                    test_name=f"workflow_{workflow_name}",
                    success=True,
                    duration=time.time() - start_time,
                    workflow_id=prompt_id,
                    output_files=valid_files,
                    metadata={
                        "expected_outputs": expected_outputs,
                        "actual_outputs": len(valid_files),
                        "workflow_config": workflow_name
                    }
                )

        except Exception as e:
            return ComfyUITestResult(
                test_name=f"workflow_{workflow_name}",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def _wait_for_workflow_completion(self, prompt_id: str, max_wait: int = 300) -> List[str]:
        """Wait for workflow completion and return output file paths"""
        start_time = time.time()
        output_files = []

        try:
            # Connect to WebSocket for real-time updates
            ws_url = f"{self.ws_endpoint}/ws"

            async with websockets.connect(ws_url) as websocket:
                while time.time() - start_time < max_wait:
                    try:
                        # Check for completion via HTTP API
                        async with httpx.AsyncClient(timeout=10) as client:
                            response = await client.get(f"{self.endpoint}/history/{prompt_id}")

                            if response.status_code == 200:
                                history = response.json()
                                if prompt_id in history:
                                    workflow_data = history[prompt_id]
                                    outputs = workflow_data.get("outputs", {})

                                    # Look for output files
                                    for node_id, node_output in outputs.items():
                                        if "images" in node_output:
                                            for img_info in node_output["images"]:
                                                filename = img_info.get("filename")
                                                if filename:
                                                    # Construct full path
                                                    output_path = f"/mnt/1TB-storage/ComfyUI/output/{filename}"
                                                    if os.path.exists(output_path):
                                                        output_files.append(output_path)

                                    if output_files:
                                        return output_files

                        await asyncio.sleep(2)

                    except Exception:
                        # Continue waiting
                        await asyncio.sleep(2)

        except Exception:
            # Fallback to HTTP polling only
            while time.time() - start_time < max_wait:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        response = await client.get(f"{self.endpoint}/history/{prompt_id}")
                        if response.status_code == 200:
                            history = response.json()
                            if prompt_id in history:
                                workflow_data = history[prompt_id]
                                if workflow_data.get("status", {}).get("completed", False):
                                    # Extract output files
                                    outputs = workflow_data.get("outputs", {})
                                    for node_id, node_output in outputs.items():
                                        if "images" in node_output:
                                            for img_info in node_output["images"]:
                                                filename = img_info.get("filename")
                                                if filename:
                                                    output_path = f"/mnt/1TB-storage/ComfyUI/output/{filename}"
                                                    if os.path.exists(output_path):
                                                        output_files.append(output_path)
                                    return output_files

                    await asyncio.sleep(5)

                except Exception:
                    await asyncio.sleep(5)

        return output_files

    async def test_concurrent_workflows(self, num_concurrent: int = 3) -> ComfyUITestResult:
        """Test concurrent workflow execution"""
        start_time = time.time()

        try:
            # Submit multiple workflows concurrently
            tasks = []
            for i in range(num_concurrent):
                # Use simple workflow for concurrent testing
                task = self._submit_concurrent_workflow(i)
                tasks.append(task)

            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze results
            successful_workflows = 0
            failed_workflows = 0
            total_outputs = 0
            errors = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_workflows += 1
                    errors.append(f"Workflow {i}: {str(result)}")
                elif result.get("success", False):
                    successful_workflows += 1
                    total_outputs += len(result.get("output_files", []))
                else:
                    failed_workflows += 1
                    errors.append(f"Workflow {i}: {result.get('error', 'Unknown error')}")

            success_rate = (successful_workflows / num_concurrent) * 100
            success = success_rate >= 70  # Allow some failures under load

            return ComfyUITestResult(
                test_name="concurrent_workflows",
                success=success,
                duration=time.time() - start_time,
                metadata={
                    "concurrent_workflows": num_concurrent,
                    "successful_workflows": successful_workflows,
                    "failed_workflows": failed_workflows,
                    "success_rate": success_rate,
                    "total_outputs": total_outputs,
                    "errors": errors
                }
            )

        except Exception as e:
            return ComfyUITestResult(
                test_name="concurrent_workflows",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def _submit_concurrent_workflow(self, workflow_index: int) -> Dict:
        """Submit a single workflow for concurrent testing"""
        try:
            workflow = self._get_simple_image_workflow()
            # Modify seed for variation
            workflow["3"]["inputs"]["seed"] = 1000 + workflow_index
            workflow["9"]["inputs"]["filename_prefix"] = f"concurrent_test_{workflow_index}"

            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(f"{self.endpoint}/prompt", json={"prompt": workflow})

                if response.status_code != 200:
                    return {"success": False, "error": f"Submit failed: {response.status_code}"}

                result = response.json()
                prompt_id = result.get("prompt_id")

                if not prompt_id:
                    return {"success": False, "error": "No prompt_id returned"}

                # Wait for completion with shorter timeout for concurrent test
                output_files = await self._wait_for_workflow_completion(prompt_id, max_wait=90)

                return {
                    "success": len(output_files) > 0,
                    "prompt_id": prompt_id,
                    "output_files": output_files,
                    "workflow_index": workflow_index
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_resource_cleanup(self) -> ComfyUITestResult:
        """Test that ComfyUI properly cleans up resources"""
        start_time = time.time()

        try:
            # Get initial system stats
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{self.endpoint}/system_stats")
                initial_stats = response.json() if response.status_code == 200 else {}

                # Submit and complete a workflow
                workflow_result = await self.test_workflow_execution("simple_image")

                if not workflow_result.success:
                    raise Exception(f"Workflow execution failed: {workflow_result.error}")

                # Wait a moment for cleanup
                await asyncio.sleep(10)

                # Get stats after workflow
                response = await client.get(f"{self.endpoint}/system_stats")
                final_stats = response.json() if response.status_code == 200 else {}

                # Check for resource leaks (basic heuristics)
                cleanup_success = True
                cleanup_notes = []

                if "gpu" in initial_stats and "gpu" in final_stats:
                    initial_gpu = initial_stats.get("gpu", {}).get("memory_used", 0)
                    final_gpu = final_stats.get("gpu", {}).get("memory_used", 0)

                    # Allow some increase but not excessive
                    gpu_increase = final_gpu - initial_gpu
                    if gpu_increase > 1000:  # More than 1GB increase
                        cleanup_success = False
                        cleanup_notes.append(f"GPU memory increased by {gpu_increase}MB")

                return ComfyUITestResult(
                    test_name="resource_cleanup",
                    success=cleanup_success,
                    duration=time.time() - start_time,
                    metadata={
                        "initial_stats": initial_stats,
                        "final_stats": final_stats,
                        "cleanup_notes": cleanup_notes,
                        "workflow_completed": workflow_result.success
                    }
                )

        except Exception as e:
            return ComfyUITestResult(
                test_name="resource_cleanup",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def test_error_recovery(self) -> ComfyUITestResult:
        """Test ComfyUI error handling and recovery"""
        start_time = time.time()

        try:
            # Submit invalid workflow
            invalid_workflow = {
                "1": {
                    "inputs": {
                        "invalid_parameter": "this should fail"
                    },
                    "class_type": "NonexistentNode"
                }
            }

            async with httpx.AsyncClient(timeout=30) as client:
                # Submit invalid workflow
                response = await client.post(f"{self.endpoint}/prompt", json={"prompt": invalid_workflow})

                error_handled = False
                if response.status_code != 200:
                    # Server properly rejected invalid workflow
                    error_handled = True
                else:
                    # Check if workflow failed gracefully
                    result = response.json()
                    prompt_id = result.get("prompt_id")

                    if prompt_id:
                        # Wait and check if error is reported
                        await asyncio.sleep(5)
                        history_response = await client.get(f"{self.endpoint}/history/{prompt_id}")

                        if history_response.status_code == 200:
                            history = history_response.json()
                            if prompt_id in history:
                                status = history[prompt_id].get("status", {})
                                if "error" in status or not status.get("completed", False):
                                    error_handled = True

                # Test that service is still responsive after error
                service_responsive = False
                try:
                    stats_response = await client.get(f"{self.endpoint}/system_stats")
                    service_responsive = stats_response.status_code == 200
                except:
                    pass

                # Test successful workflow after error
                recovery_successful = False
                if service_responsive:
                    recovery_result = await self.test_workflow_execution("simple_image")
                    recovery_successful = recovery_result.success

                overall_success = error_handled and service_responsive and recovery_successful

                return ComfyUITestResult(
                    test_name="error_recovery",
                    success=overall_success,
                    duration=time.time() - start_time,
                    metadata={
                        "error_handled": error_handled,
                        "service_responsive": service_responsive,
                        "recovery_successful": recovery_successful
                    }
                )

        except Exception as e:
            return ComfyUITestResult(
                test_name="error_recovery",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def run_integration_test_suite(self) -> Dict:
        """Run complete ComfyUI integration test suite"""
        all_tests = []

        # Test service availability
        availability_test = await self.test_service_availability()
        all_tests.append(availability_test)

        if not availability_test.success:
            # If service is down, skip other tests
            return {
                "test_summary": {
                    "total_tests": 1,
                    "passed_tests": 0,
                    "failed_tests": 1,
                    "success_rate": 0,
                    "service_available": False
                },
                "test_results": [self._test_result_to_dict(availability_test)]
            }

        # Test individual workflows
        for workflow_name in ["simple_image", "anime_character"]:
            workflow_test = await self.test_workflow_execution(workflow_name)
            all_tests.append(workflow_test)

        # Test concurrent execution (only if basic workflows work)
        successful_workflows = [t for t in all_tests if t.success and "workflow" in t.test_name]
        if len(successful_workflows) >= 1:
            concurrent_test = await self.test_concurrent_workflows(num_concurrent=2)
            all_tests.append(concurrent_test)

        # Test resource cleanup
        cleanup_test = await self.test_resource_cleanup()
        all_tests.append(cleanup_test)

        # Test error recovery
        recovery_test = await self.test_error_recovery()
        all_tests.append(recovery_test)

        # Analyze results
        total_tests = len(all_tests)
        passed_tests = len([t for t in all_tests if t.success])
        failed_tests = [t for t in all_tests if not t.success]

        results = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": len(failed_tests),
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                "service_available": availability_test.success
            },
            "test_results": [self._test_result_to_dict(t) for t in all_tests],
            "failed_tests": [
                {
                    "test_name": t.test_name,
                    "error": t.error,
                    "duration": t.duration
                }
                for t in failed_tests
            ]
        }

        return results

    def _test_result_to_dict(self, result: ComfyUITestResult) -> Dict:
        """Convert test result to dictionary for serialization"""
        return {
            "test_name": result.test_name,
            "success": result.success,
            "duration": result.duration,
            "error": result.error,
            "workflow_id": result.workflow_id,
            "output_files": result.output_files or [],
            "metadata": result.metadata
        }


# Pytest test cases
class TestComfyUIIntegration:
    """Pytest test cases for ComfyUI integration"""

    @pytest.fixture
    def comfyui_tester(self):
        return ComfyUIIntegrationTester()

    @pytest.mark.asyncio
    async def test_comfyui_service_availability(self, comfyui_tester):
        """Test ComfyUI service is available and responding"""
        result = await comfyui_tester.test_service_availability()

        assert result.success, f"ComfyUI service not available: {result.error}"
        assert result.duration < 10, f"Service response too slow: {result.duration}s"

        if result.metadata:
            assert "system_stats" in result.metadata, "No system stats returned"

    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self, comfyui_tester):
        """Test execution of simple image generation workflow"""
        result = await comfyui_tester.test_workflow_execution("simple_image")

        assert result.success, f"Simple workflow failed: {result.error}"
        assert result.duration < 120, f"Workflow too slow: {result.duration}s"
        assert result.output_files, "No output files generated"

        # Verify output file exists and is valid
        output_file = result.output_files[0]
        assert os.path.exists(output_file), f"Output file missing: {output_file}"
        assert os.path.getsize(output_file) > 1000, "Output file too small (likely corrupt)"

    @pytest.mark.asyncio
    async def test_anime_workflow_execution(self, comfyui_tester):
        """Test execution of anime character generation workflow"""
        result = await comfyui_tester.test_workflow_execution("anime_character")

        assert result.success, f"Anime workflow failed: {result.error}"
        assert result.duration < 180, f"Anime workflow too slow: {result.duration}s"
        assert result.output_files, "No output files generated"

        # Verify image can be opened
        output_file = result.output_files[0]
        try:
            with Image.open(output_file) as img:
                assert img.width > 0 and img.height > 0, "Invalid image dimensions"
        except Exception as e:
            pytest.fail(f"Generated image is corrupt: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_workflow_handling(self, comfyui_tester):
        """Test ComfyUI handles concurrent workflows properly"""
        result = await comfyui_tester.test_concurrent_workflows(num_concurrent=2)

        assert result.success, f"Concurrent test failed: {result.error}"
        assert result.duration < 300, f"Concurrent test too slow: {result.duration}s"

        if result.metadata:
            success_rate = result.metadata.get("success_rate", 0)
            assert success_rate >= 50, f"Concurrent success rate too low: {success_rate}%"

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, comfyui_tester):
        """Test ComfyUI error handling and recovery capabilities"""
        result = await comfyui_tester.test_error_recovery()

        assert result.success, f"Error recovery test failed: {result.error}"

        if result.metadata:
            assert result.metadata.get("error_handled", False), "Errors not properly handled"
            assert result.metadata.get("service_responsive", False), "Service not responsive after error"
            assert result.metadata.get("recovery_successful", False), "Recovery after error failed"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_comprehensive_comfyui_validation(self, comfyui_tester):
        """Run complete ComfyUI integration test suite (slow test)"""
        results = await comfyui_tester.run_integration_test_suite()

        # Save results
        results_file = f"/tmp/comfyui_integration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"ComfyUI integration results saved to: {results_file}")

        # Check overall integration health
        success_rate = results["test_summary"]["success_rate"]
        assert success_rate >= 70, f"ComfyUI integration success rate too low: {success_rate}%"
        assert results["test_summary"]["service_available"], "ComfyUI service not available"

        # Print summary
        print(f"ComfyUI Integration Summary:")
        print(f"Total Tests: {results['test_summary']['total_tests']}")
        print(f"Passed: {results['test_summary']['passed_tests']}")
        print(f"Failed: {results['test_summary']['failed_tests']}")
        print(f"Success Rate: {success_rate:.1f}%")


if __name__ == "__main__":
    # CLI interface for standalone testing
    import argparse

    parser = argparse.ArgumentParser(description="ComfyUI integration testing")
    parser.add_argument("--endpoint", default="http://***REMOVED***:8188", help="ComfyUI endpoint")
    parser.add_argument("--test-type", choices=["availability", "simple", "anime", "concurrent", "recovery", "all"],
                       default="all", help="Type of test to run")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--concurrent-jobs", type=int, default=2, help="Number of concurrent jobs for concurrent test")

    args = parser.parse_args()

    async def main():
        tester = ComfyUIIntegrationTester(args.endpoint)

        if args.test_type == "availability":
            result = await tester.test_service_availability()
            results = {"test_result": tester._test_result_to_dict(result)}
        elif args.test_type == "simple":
            result = await tester.test_workflow_execution("simple_image")
            results = {"test_result": tester._test_result_to_dict(result)}
        elif args.test_type == "anime":
            result = await tester.test_workflow_execution("anime_character")
            results = {"test_result": tester._test_result_to_dict(result)}
        elif args.test_type == "concurrent":
            result = await tester.test_concurrent_workflows(args.concurrent_jobs)
            results = {"test_result": tester._test_result_to_dict(result)}
        elif args.test_type == "recovery":
            result = await tester.test_error_recovery()
            results = {"test_result": tester._test_result_to_dict(result)}
        elif args.test_type == "all":
            results = await tester.run_integration_test_suite()

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())