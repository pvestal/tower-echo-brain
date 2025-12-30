"""
ComfyUI Workflow Submission Module
Submits workflows to ComfyUI and retrieves generated outputs
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

logger = logging.getLogger(__name__)

class ComfyUIIntegration:
    """Integration with ComfyUI for image/video generation"""

    def __init__(
        self,
        comfyui_url: str = "http://localhost:8188",
        output_path: str = "/opt/tower-echo-brain/data/outputs/"
    ):
        self.comfyui_url = comfyui_url
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.client_id = str(uuid.uuid4())

    async def load_workflow_template(self) -> Dict[str, Any]:
        """Load a workflow template"""

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
        """Inject custom parameters into workflow"""

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

    async def submit_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Submit workflow to ComfyUI API"""

        try:
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
                        return {
                            "success": False,
                            "error": f"ComfyUI returned {response.status}: {error_text}"
                        }

        except Exception as e:
            logger.error(f"Failed to submit workflow: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def poll_for_completion(
        self,
        prompt_id: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """Poll ComfyUI for workflow completion"""

        start_time = time.time()

        try:
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

                    # Wait before next poll
                    await asyncio.sleep(2)

                return {
                    "success": False,
                    "error": f"Timeout after {timeout} seconds",
                    "execution_time": timeout
                }

        except Exception as e:
            logger.error(f"Polling failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def download_output(
        self,
        filename: str,
        subfolder: str = "",
        output_type: str = "output"
    ) -> Optional[Path]:
        """Download generated output from ComfyUI"""

        try:
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
                        logger.error(f"Failed to download: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    async def execute_workflow_end_to_end(
        self,
        prompt: str = "a beautiful anime girl with blue hair, echo brain avatar",
        steps: int = 20
    ) -> Dict[str, Any]:
        """Execute a complete workflow from submission to download"""

        print(f"Loading workflow template...")
        workflow = await self.load_workflow_template()

        print(f"Injecting parameters...")
        workflow = await self.inject_parameters(
            workflow,
            prompt=prompt,
            steps=steps,
            seed=int(time.time())  # Random seed based on time
        )

        print(f"Submitting to ComfyUI at {self.comfyui_url}...")
        submission = await self.submit_workflow(workflow)

        if not submission['success']:
            return submission

        prompt_id = submission['prompt_id']
        print(f"✅ Submitted with prompt_id: {prompt_id}")

        print(f"Polling for completion...")
        completion = await self.poll_for_completion(prompt_id, timeout=60)

        if not completion.get('success'):
            return completion

        print(f"✅ Generation completed in {completion['execution_time']:.1f} seconds")

        # Download outputs
        downloaded_files = []
        for output in completion.get('outputs', []):
            print(f"Downloading {output['filename']}...")
            file_path = await self.download_output(
                output['filename'],
                output.get('subfolder', ''),
                output.get('type', 'output')
            )

            if file_path:
                downloaded_files.append(str(file_path))

        return {
            "success": True,
            "prompt_id": prompt_id,
            "execution_time": completion['execution_time'],
            "outputs": downloaded_files,
            "workflow_submitted": True
        }


async def test_comfyui_integration():
    """Test ComfyUI workflow submission"""

    print("=" * 60)
    print("COMFYUI WORKFLOW SUBMISSION TEST")
    print("=" * 60)

    integrator = ComfyUIIntegration()

    # Check if ComfyUI is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{integrator.comfyui_url}/system_stats") as response:
                if response.status == 200:
                    stats = await response.json()
                    print(f"\n✅ ComfyUI is running")
                    print(f"   System: {stats.get('system', {}).get('os', 'Unknown')}")
                    print(f"   Python: {stats.get('system', {}).get('python_version', 'Unknown')}")
                else:
                    print(f"\n❌ ComfyUI not responding (status {response.status})")
                    print("   Please start ComfyUI first: cd /home/patrick/ComfyUI && python main.py")
                    return False
    except Exception as e:
        print(f"\n❌ Cannot connect to ComfyUI: {e}")
        print("   Please start ComfyUI first")
        return False

    # Execute workflow
    print("\nExecuting workflow end-to-end...")
    result = await integrator.execute_workflow_end_to_end(
        prompt="echo brain autonomous AI avatar, digital art, blue theme",
        steps=10  # Reduced steps for faster testing
    )

    if result['success']:
        print(f"\n✅ WORKFLOW EXECUTION SUCCESSFUL!")
        print(f"   Prompt ID: {result['prompt_id']}")
        print(f"   Execution time: {result['execution_time']:.1f} seconds")
        print(f"   Outputs saved:")
        for output in result['outputs']:
            file_path = Path(output)
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"     - {file_path.name} ({size_kb:.1f} KB)")
    else:
        print(f"\n❌ WORKFLOW EXECUTION FAILED: {result.get('error')}")

    print("\n" + "=" * 60)
    return result.get('success', False)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_comfyui_integration())