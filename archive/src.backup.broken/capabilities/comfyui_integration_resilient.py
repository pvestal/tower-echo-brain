#!/usr/bin/env python3
"""
Resilient ComfyUI Workflow Submission Module
Enhanced with circuit breaker patterns for improved reliability
"""

import json
import aiohttp
import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import time

from ..resilience.service_breakers import get_comfyui_breaker
from ..resilience.fallback_handlers import get_fallback_handler

logger = logging.getLogger(__name__)


class ResilientComfyUIIntegration:
    """
    Circuit breaker protected ComfyUI integration
    Provides automatic failure detection and graceful degradation
    """

    def __init__(
        self,
        comfyui_url: str = "http://localhost:8188",
        output_path: str = "/opt/tower-echo-brain/data/outputs/"
    ):
        self.comfyui_url = comfyui_url
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.client_id = str(uuid.uuid4())

        # Get circuit breaker and fallback handler
        self.circuit_breaker = get_comfyui_breaker()
        self.fallback_handler = get_fallback_handler()

        logger.info(f"Initialized resilient ComfyUI integration with circuit breaker protection")

    async def load_workflow_template(self) -> Dict[str, Any]:
        """Load a workflow template (no external service call)"""

        # Simple text-to-image workflow template
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 7,
                    "denoise": 1,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": 42,
                    "steps": 20
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "animagine-xl-3.0.safetensors"
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": 512,
                    "width": 512
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": "a beautiful anime girl with blue hair, echo brain avatar, digital art"
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": "bad quality, blurry"
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "echo_brain_comfyui",
                    "images": ["8", 0]
                }
            }
        }

        return workflow

    async def inject_parameters(
        self,
        workflow: Dict[str, Any],
        prompt: str = None,
        negative_prompt: str = None,
        seed: int = None,
        steps: int = None,
        lora_name: str = None
    ) -> Dict[str, Any]:
        """Inject custom parameters into workflow (no external service call)"""

        # Update positive prompt
        if prompt and "6" in workflow:
            workflow["6"]["inputs"]["text"] = prompt

        # Update negative prompt
        if negative_prompt and "7" in workflow:
            workflow["7"]["inputs"]["text"] = negative_prompt

        # Update seed
        if seed is not None and "3" in workflow:
            workflow["3"]["inputs"]["seed"] = seed

        # Update steps
        if steps is not None and "3" in workflow:
            workflow["3"]["inputs"]["steps"] = steps

        # Add LoRA if specified
        if lora_name:
            workflow["10"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_clip": 1,
                    "strength_model": 0.8,
                    "clip": ["4", 1],
                    "model": ["4", 0]
                }
            }
            # Update connections to use LoRA
            workflow["3"]["inputs"]["model"] = ["10", 0]
            workflow["6"]["inputs"]["clip"] = ["10", 1]
            workflow["7"]["inputs"]["clip"] = ["10", 1]

        return workflow

    async def _submit_workflow_internal(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to submit workflow to ComfyUI (protected by circuit breaker)"""

        # Prepare prompt request
        prompt_data = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        # Submit to ComfyUI
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.comfyui_url}/prompt",
                json=prompt_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "prompt_id": result.get("prompt_id"),
                        "number": result.get("number"),
                        "response": result
                    }
                else:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"ComfyUI returned {response.status}: {error_text}"
                    )

    async def submit_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Submit workflow to ComfyUI with circuit breaker protection"""

        try:
            # Define fallback for workflow submission
            async def submit_fallback(*args, **kwargs):
                return await self.fallback_handler.execute_fallback(
                    "comfyui", "submit_workflow",
                    workflow=workflow
                )

            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call(
                self._submit_workflow_internal,
                workflow,
                fallback=submit_fallback
            )

            return result

        except Exception as e:
            logger.error(f"Failed to submit workflow even with fallback: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_attempted": True
            }

    async def _poll_for_completion_internal(
        self,
        prompt_id: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """Internal method to poll ComfyUI for completion (protected by circuit breaker)"""

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                # Check history for our prompt
                async with session.get(
                    f"{self.comfyui_url}/history/{prompt_id}"
                ) as response:
                    if response.status == 200:
                        history = await response.json()

                        if prompt_id in history:
                            prompt_history = history[prompt_id]

                            # Check if completed
                            if prompt_history.get("outputs"):
                                # Find output images
                                outputs = []
                                for node_id, node_output in prompt_history["outputs"].items():
                                    if "images" in node_output:
                                        for image_info in node_output["images"]:
                                            outputs.append({
                                                "filename": image_info["filename"],
                                                "subfolder": image_info.get("subfolder", ""),
                                                "type": image_info.get("type", "output")
                                            })

                                return {
                                    "success": True,
                                    "completed": True,
                                    "outputs": outputs,
                                    "execution_time": time.time() - start_time
                                }
                    else:
                        # HTTP error during polling
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"Polling failed with {response.status}: {error_text}"
                        )

                # Wait before next poll
                await asyncio.sleep(2)

            # Timeout reached
            raise asyncio.TimeoutError(f"Polling timeout after {timeout} seconds")

    async def poll_for_completion(
        self,
        prompt_id: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """Poll ComfyUI for workflow completion with circuit breaker protection"""

        try:
            # Define fallback for polling
            async def poll_fallback(*args, **kwargs):
                return await self.fallback_handler.execute_fallback(
                    "comfyui", "poll_for_completion",
                    prompt_id=prompt_id,
                    timeout=timeout
                )

            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call(
                self._poll_for_completion_internal,
                prompt_id,
                timeout,
                fallback=poll_fallback
            )

            return result

        except Exception as e:
            logger.error(f"Failed to poll for completion even with fallback: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": timeout,
                "fallback_attempted": True
            }

    async def _download_output_internal(
        self,
        filename: str,
        subfolder: str = "",
        output_type: str = "output"
    ) -> Optional[Path]:
        """Internal method to download output (protected by circuit breaker)"""

        async with aiohttp.ClientSession() as session:
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": output_type
            }

            async with session.get(
                f"{self.comfyui_url}/view",
                params=params
            ) as response:
                if response.status == 200:
                    # Save to output directory
                    output_file = self.output_path / filename
                    content = await response.read()
                    output_file.write_bytes(content)

                    logger.info(f"Downloaded output to {output_file}")
                    return output_file
                else:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Download failed with {response.status}: {error_text}"
                    )

    async def download_output(
        self,
        filename: str,
        subfolder: str = "",
        output_type: str = "output"
    ) -> Optional[Path]:
        """Download generated output with circuit breaker protection"""

        try:
            # Define fallback for download
            async def download_fallback(*args, **kwargs):
                fallback_result = await self.fallback_handler.execute_fallback(
                    "comfyui", "download_output",
                    filename=filename,
                    subfolder=subfolder,
                    output_type=output_type
                )
                return None  # Fallback cannot actually download files

            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call(
                self._download_output_internal,
                filename,
                subfolder,
                output_type,
                fallback=download_fallback
            )

            return result

        except Exception as e:
            logger.error(f"Failed to download output even with fallback: {e}")
            return None

    async def check_service_health(self) -> Dict[str, Any]:
        """Check ComfyUI service health"""

        try:
            async def health_check():
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.comfyui_url}/system_stats") as response:
                        if response.status == 200:
                            stats = await response.json()
                            return {
                                "success": True,
                                "status": "healthy",
                                "system": stats.get("system", {}),
                                "url": self.comfyui_url
                            }
                        else:
                            return {
                                "success": False,
                                "status": "unhealthy",
                                "error": f"HTTP {response.status}",
                                "url": self.comfyui_url
                            }

            # Use circuit breaker for health check
            result = await self.circuit_breaker.call(health_check)
            return result

        except Exception as e:
            logger.error(f"ComfyUI health check failed: {e}")
            return {
                "success": False,
                "status": "unreachable",
                "error": str(e),
                "url": self.comfyui_url
            }

    async def execute_workflow_end_to_end(
        self,
        prompt: str = "a beautiful anime girl with blue hair, echo brain avatar",
        steps: int = 20,
        enable_fallback: bool = True
    ) -> Dict[str, Any]:
        """Execute a complete workflow from submission to download with resilience"""

        workflow_start_time = time.time()

        logger.info(f"Starting resilient workflow execution...")

        # Check service health first
        health = await self.check_service_health()
        if not health.get("success"):
            logger.warning(f"ComfyUI health check failed: {health}")

        # Load and configure workflow
        logger.info(f"Loading workflow template...")
        workflow = await self.load_workflow_template()

        logger.info(f"Injecting parameters...")
        workflow = await self.inject_parameters(
            workflow,
            prompt=prompt,
            steps=steps,
            seed=int(time.time())  # Random seed based on time
        )

        # Submit workflow with circuit breaker protection
        logger.info(f"Submitting to ComfyUI at {self.comfyui_url} with circuit breaker protection...")
        submission = await self.submit_workflow(workflow)

        if not submission.get('success'):
            logger.error(f"Workflow submission failed: {submission}")
            return {
                **submission,
                "workflow_completed": False,
                "total_execution_time": time.time() - workflow_start_time,
                "circuit_breaker_state": self.circuit_breaker.state.value
            }

        if submission.get('fallback'):
            # Fallback response - workflow was queued
            return {
                **submission,
                "workflow_completed": False,
                "workflow_queued": True,
                "total_execution_time": time.time() - workflow_start_time,
                "circuit_breaker_state": self.circuit_breaker.state.value
            }

        prompt_id = submission.get('prompt_id')
        logger.info(f"✅ Submitted with prompt_id: {prompt_id}")

        # Poll for completion with circuit breaker protection
        logger.info(f"Polling for completion with circuit breaker protection...")
        completion = await self.poll_for_completion(prompt_id, timeout=120)

        if not completion.get('success') or not completion.get('completed'):
            logger.error(f"Workflow completion failed: {completion}")
            return {
                **completion,
                "workflow_completed": False,
                "prompt_id": prompt_id,
                "total_execution_time": time.time() - workflow_start_time,
                "circuit_breaker_state": self.circuit_breaker.state.value
            }

        if completion.get('fallback'):
            # Fallback response during completion
            return {
                **completion,
                "workflow_completed": False,
                "prompt_id": prompt_id,
                "total_execution_time": time.time() - workflow_start_time,
                "circuit_breaker_state": self.circuit_breaker.state.value
            }

        logger.info(f"✅ Generation completed in {completion['execution_time']:.1f} seconds")

        # Download outputs with circuit breaker protection
        downloaded_files = []
        for output in completion.get('outputs', []):
            logger.info(f"Downloading {output['filename']} with circuit breaker protection...")
            file_path = await self.download_output(
                output['filename'],
                output.get('subfolder', ''),
                output.get('type', 'output')
            )

            if file_path:
                downloaded_files.append(str(file_path))

        total_execution_time = time.time() - workflow_start_time

        return {
            "success": True,
            "workflow_completed": True,
            "prompt_id": prompt_id,
            "generation_time": completion['execution_time'],
            "total_execution_time": total_execution_time,
            "outputs": downloaded_files,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "workflow_submitted": True,
            "resilience_enabled": True
        }

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return self.circuit_breaker.get_metrics()


async def test_resilient_comfyui_integration():
    """Test resilient ComfyUI workflow submission"""

    print("=" * 60)
    print("RESILIENT COMFYUI INTEGRATION TEST")
    print("=" * 60)

    integrator = ResilientComfyUIIntegration()

    # Check circuit breaker status
    cb_status = integrator.get_circuit_breaker_status()
    print(f"\nCircuit Breaker Status: {cb_status['state']}")
    print(f"Total Requests: {cb_status['metrics']['total_requests']}")
    print(f"Failure Rate: {cb_status['metrics']['failure_rate']}%")

    # Test service health
    print("\nTesting service health...")
    health = await integrator.check_service_health()
    if health['success']:
        print(f"✅ ComfyUI is healthy: {health['status']}")
    else:
        print(f"❌ ComfyUI health check failed: {health['error']}")

    # Execute workflow
    print("\nExecuting resilient workflow end-to-end...")
    result = await integrator.execute_workflow_end_to_end(
        prompt="echo brain autonomous AI avatar, digital art, blue theme, resilient test",
        steps=10  # Reduced steps for faster testing
    )

    if result.get('success') and result.get('workflow_completed'):
        print(f"\n✅ RESILIENT WORKFLOW EXECUTION SUCCESSFUL!")
        print(f"   Prompt ID: {result['prompt_id']}")
        print(f"   Generation time: {result['generation_time']:.1f} seconds")
        print(f"   Total execution time: {result['total_execution_time']:.1f} seconds")
        print(f"   Circuit breaker state: {result['circuit_breaker_state']}")
        print(f"   Outputs saved: {len(result['outputs'])} files")
        for output in result['outputs']:
            file_path = Path(output)
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"     - {file_path.name} ({size_kb:.1f} KB)")
    elif result.get('workflow_queued'):
        print(f"\n⏳ WORKFLOW QUEUED FOR RETRY")
        print(f"   Service unavailable - workflow queued for processing when service recovers")
        print(f"   Circuit breaker state: {result['circuit_breaker_state']}")
    else:
        print(f"\n❌ RESILIENT WORKFLOW EXECUTION FAILED")
        print(f"   Error: {result.get('error', 'Unknown error')}")
        print(f"   Circuit breaker state: {result.get('circuit_breaker_state', 'unknown')}")

    # Final circuit breaker status
    final_cb_status = integrator.get_circuit_breaker_status()
    print(f"\nFinal Circuit Breaker Status:")
    print(f"   State: {final_cb_status['state']}")
    print(f"   Total Requests: {final_cb_status['metrics']['total_requests']}")
    print(f"   Success Rate: {final_cb_status['metrics']['success_rate']}%")
    print(f"   Failure Rate: {final_cb_status['metrics']['failure_rate']}%")

    print("\n" + "=" * 60)
    return result.get('success', False)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_resilient_comfyui_integration())