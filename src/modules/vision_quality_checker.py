#!/usr/bin/env python3
"""Vision-based quality checking for generated frames"""

import asyncio
import base64
import io
import json
import logging
import requests
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class VisionQualityChecker:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.vision_model = "llava:13b"  # or bakllava, llava-llama3
        self.quality_threshold = 0.7
        self.retry_limit = 3
        
    async def check_frame_quality(self, image_path: str) -> Dict:
        """Check frame quality using vision model"""
        try:
            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # Prepare vision prompt
            prompt = """Analyze this image for quality issues. Check for:
            1. Blur or lack of sharpness
            2. Artifacts or corruption
            3. Composition problems
            4. Color/lighting issues
            5. Character consistency
            
            Rate quality from 0-10 and list any issues found.
            Respond in JSON format: {"score": X, "issues": [], "regenerate": true/false}"""
            
            # Call vision model
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # Parse the response
                try:
                    quality_data = json.loads(result.get('response', '{}'))
                    return {
                        "passed": quality_data.get('score', 0) >= 7,
                        "score": quality_data.get('score', 0),
                        "issues": quality_data.get('issues', []),
                        "regenerate": quality_data.get('regenerate', False)
                    }
                except:
                    # Fallback to basic check
                    return self._basic_quality_check(image_path)
            else:
                logger.error(f"Vision model error: {response.status_code}")
                return self._basic_quality_check(image_path)
                
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return self._basic_quality_check(image_path)
    
    def _basic_quality_check(self, image_path: str) -> Dict:
        """Basic quality check without vision model"""
        try:
            img = Image.open(image_path)
            
            # Check basic criteria
            issues = []
            
            # Size check
            if img.width < 512 or img.height < 512:
                issues.append("Low resolution")
            
            # Check if mostly black/white
            arr = np.array(img)
            mean_val = arr.mean()
            if mean_val < 10:
                issues.append("Too dark")
            elif mean_val > 245:
                issues.append("Too bright")
            
            # Check variance (detect blank/flat images)
            variance = arr.var()
            if variance < 100:
                issues.append("Low detail/flat image")
            
            score = 10 - len(issues) * 2.5
            
            return {
                "passed": len(issues) == 0,
                "score": max(0, score),
                "issues": issues,
                "regenerate": len(issues) > 1
            }
        except Exception as e:
            logger.error(f"Basic check failed: {e}")
            return {
                "passed": False,
                "score": 0,
                "issues": ["Failed to analyze"],
                "regenerate": True
            }
    
    async def validate_batch(self, image_paths: List[str]) -> List[Dict]:
        """Validate multiple images"""
        results = []
        for path in image_paths:
            result = await self.check_frame_quality(path)
            results.append({
                "path": path,
                "quality": result
            })
            
            if not result["passed"]:
                logger.warning(f"Frame failed quality: {path} - {result['issues']}")
        
        return results

class RegenerativeVideoGenerator:
    """Video generator with quality checking and regeneration"""
    
    def __init__(self, comfyui_url="http://localhost:8188"):
        self.comfyui_url = comfyui_url
        self.quality_checker = VisionQualityChecker()
        self.max_retries = 3
        
    async def generate_with_quality_check(self, prompt: str, workflow: Dict) -> str:
        """Generate frame with quality checking and retry"""
        
        for attempt in range(self.max_retries):
            # Generate frame
            response = requests.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow}
            )
            
            if response.status_code != 200:
                logger.error(f"Generation failed: {response.text}")
                continue
            
            prompt_id = response.json().get("prompt_id")
            
            # Wait for completion
            await asyncio.sleep(10)
            
            # Get output path (simplified - would need proper tracking)
            output_dir = Path("/home/patrick/ComfyUI/output")
            latest_file = max(output_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, default=None)
            
            if not latest_file:
                logger.error("No output file found")
                continue
            
            # Check quality
            quality = await self.quality_checker.check_frame_quality(str(latest_file))
            
            if quality["passed"]:
                logger.info(f"Frame passed quality check: {quality['score']}/10")
                return str(latest_file)
            else:
                logger.warning(f"Attempt {attempt+1} failed quality: {quality['issues']}")
                
                # Modify prompt based on issues
                if "blur" in str(quality['issues']).lower():
                    workflow = self._adjust_for_sharpness(workflow)
                elif "dark" in str(quality['issues']).lower():
                    workflow = self._adjust_brightness(workflow)
                elif "composition" in str(quality['issues']).lower():
                    workflow = self._improve_composition(workflow)
        
        logger.error(f"Failed to generate acceptable frame after {self.max_retries} attempts")
        return None
    
    def _adjust_for_sharpness(self, workflow: Dict) -> Dict:
        """Adjust workflow for sharper output"""
        # Increase steps, adjust sampler
        if "5" in workflow:  # KSampler node
            workflow["5"]["inputs"]["steps"] = 30
            workflow["5"]["inputs"]["cfg"] = 8
        return workflow
    
    def _adjust_brightness(self, workflow: Dict) -> Dict:
        """Adjust for brightness issues"""
        # Modify prompt
        if "2" in workflow:  # Positive prompt
            workflow["2"]["inputs"]["text"] += ", bright lighting, well lit"
        return workflow
    
    def _improve_composition(self, workflow: Dict) -> Dict:
        """Improve composition"""
        if "2" in workflow:
            workflow["2"]["inputs"]["text"] += ", centered composition, rule of thirds"
        return workflow

# Integration with Echo Brain
class EchoVisionIntegration:
    """Integrate vision checking into Echo's pipeline"""
    
    def __init__(self):
        self.generator = RegenerativeVideoGenerator()
        self.database_url = "postgresql://echo:password@localhost/echo_brain"
        
    async def generate_quality_video(self, project_name: str, prompts: List[str], style: str):
        """Generate video with quality assurance"""
        
        successful_frames = []
        failed_frames = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating frame {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Create workflow
            workflow = self._create_workflow(prompt, style)
            
            # Generate with quality checking
            frame_path = await self.generator.generate_with_quality_check(prompt, workflow)
            
            if frame_path:
                successful_frames.append(frame_path)
                # Save to database for learning
                await self._save_success_pattern(prompt, workflow, frame_path)
            else:
                failed_frames.append(prompt)
                # Log failure for analysis
                await self._log_failure(prompt, workflow)
        
        # Create video from successful frames
        if successful_frames:
            video_path = await self._compile_video(successful_frames, project_name)
            return {
                "success": True,
                "video_path": video_path,
                "frames_generated": len(successful_frames),
                "frames_failed": len(failed_frames),
                "quality_pass_rate": len(successful_frames) / len(prompts)
            }
        else:
            return {
                "success": False,
                "error": "No frames passed quality check"
            }
    
    def _create_workflow(self, prompt: str, style: str) -> Dict:
        """Create ComfyUI workflow"""
        return {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "animagine_xl_3.1.safetensors"}},
            "2": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": f"{style} style, {prompt}, high quality, detailed", "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "bad quality, blurry, artifacts", "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "5": {"class_type": "KSampler",
                  "inputs": {"seed": 42, "steps": 25, "cfg": 7.5,
                           "sampler_name": "dpmpp_2m", "scheduler": "karras",
                           "denoise": 1, "model": ["1", 0],
                           "positive": ["2", 0], "negative": ["3", 0],
                           "latent_image": ["4", 0]}},
            "6": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage",
                  "inputs": {"images": ["6", 0],
                           "filename_prefix": f"quality_checked"}}
        }
    
    async def _save_success_pattern(self, prompt, workflow, frame_path):
        """Save successful patterns for learning"""
        # Save to database
        logger.info(f"Saving successful pattern to database")
    
    async def _log_failure(self, prompt, workflow):
        """Log failures for analysis"""
        logger.warning(f"Logging failure for analysis")
    
    async def _compile_video(self, frames, project_name):
        """Compile frames into video"""
        import subprocess
        output_path = f"/home/patrick/Videos/{project_name}_quality.mp4"
        
        # Create frame list
        with open("/tmp/quality_frames.txt", "w") as f:
            for frame in frames:
                f.write(f"file '{frame}'\n")
                f.write("duration 2\n")
        
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", "/tmp/quality_frames.txt",
            "-vf", "scale=1920:1080,fps=24",
            "-c:v", "libx264", "-crf", "18",
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True)
        return output_path

# Test the system
if __name__ == "__main__":
    async def test():
        checker = VisionQualityChecker()
        
        # Test with existing frames
        test_frames = [
            "/home/patrick/ComfyUI/output/goblin_slayer_cyberpunk_proper_00001_.png"
        ]
        
        for frame in test_frames:
            if Path(frame).exists():
                result = await checker.check_frame_quality(frame)
                print(f"Frame: {frame}")
                print(f"Quality: {result}")
        
        # Test generation with quality
        echo = EchoVisionIntegration()
        result = await echo.generate_quality_video(
            "test_quality",
            ["cyberpunk warrior in neon city"],
            "anime"
        )
        print(f"Generation result: {result}")
    
    asyncio.run(test())
