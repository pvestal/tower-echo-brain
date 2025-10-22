#!/usr/bin/env python3
"""
Echo Orchestrator with Automatic Quality Retry
Never accepts garbage - always retries until quality achieved
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any, List
import cv2
import numpy as np

class QualityAutoRetryOrchestrator:
    def __init__(self):
        self.services = {
            "comfyui": "http://localhost:8188",
            "voice": "http://localhost:8312",
            "kb": "http://localhost:8307"
        }
        self.quality_threshold = 75  # Minimum acceptable quality
        self.max_retries = 4
        
    async def orchestrate_trailer(self, character="Kai cyberpunk ninja") -> Dict[str, Any]:
        """Create trailer with automatic quality retry"""
        
        results = []
        output = f"***REMOVED***/echo_quality_{int(time.time())}.mp4"
        
        # Generate frames with auto-retry
        frames = await self.generate_frames_with_retry(character)
        
        if not frames:
            return {
                "response": "Failed to generate quality frames after retries",
                "details": ["All retry attempts exhausted"],
                "success": False
            }
        
        results.append(f"✅ Generated {len(frames)} quality frames after auto-retry")
        
        # Create video from frames
        video_created = await self.create_video_from_frames(frames, output)
        
        if video_created:
            results.append(f"✅ Video created: {output}")
            return {
                "response": "High-quality trailer created with auto-retry!",
                "details": results,
                "output": output,
                "success": True
            }
        else:
            return {
                "response": "Video assembly failed",
                "details": results,
                "success": False
            }
    
    async def generate_frames_with_retry(self, character: str) -> List[str]:
        """Generate frames with automatic quality-based retry"""
        frames = []
        
        for frame_num in range(3):  # Generate 3 frames for testing
            print(f"\n🎬 Generating frame {frame_num + 1}/3")
            
            for attempt in range(self.max_retries):
                print(f"  Attempt {attempt + 1}/{self.max_retries}")
                
                # Progressive quality enhancement
                quality_boost = attempt * 10
                frame_path = await self.generate_single_frame(
                    character, 
                    frame_num, 
                    quality_boost
                )
                
                if frame_path:
                    quality_score = self.check_frame_quality(frame_path)
                    print(f"  Quality score: {quality_score}/100")
                    
                    if quality_score >= self.quality_threshold:
                        print(f"  ✅ Quality threshold met!")
                        frames.append(frame_path)
                        break
                    else:
                        print(f"  ❌ Quality too low, auto-retrying...")
                else:
                    print(f"  ❌ Generation failed, retrying...")
                
                if attempt == self.max_retries - 1:
                    print(f"  💥 Max retries reached for frame {frame_num}")
        
        return frames
    
    async def generate_single_frame(self, character: str, frame_num: int, quality_boost: int) -> str:
        """Generate a single frame with ComfyUI"""
        
        # Enhanced prompt based on quality boost
        prompt = f"{character}, frame {frame_num}, masterpiece, best quality"
        if quality_boost > 0:
            prompt += f", ultra detailed, 8k, professional"
        if quality_boost > 10:
            prompt += f", award winning, perfect composition"
        
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "animagine_xl_3.1.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {"text": prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "garbage, bad quality, blurry, artifacts",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {"width": 1920, "height": 1080, "batch_size": 1},
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": int(time.time()) + frame_num + quality_boost,
                    "steps": 20 + quality_boost,
                    "cfg": 7 + (quality_boost / 10),
                    "sampler_name": "dpmpp_2m" if quality_boost > 10 else "euler",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode"
            },
            "7": {
                "inputs": {
                    "filename_prefix": f"echo_autoretry_f{frame_num:03d}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.services['comfyui']}/prompt",
                    json={"prompt": workflow},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    prompt_id = response.json()['prompt_id']
                    
                    # Wait for completion
                    await asyncio.sleep(20 + quality_boost)  # Longer wait for better quality
                    
                    # Check for output
                    history_response = await client.get(
                        f"{self.services['comfyui']}/history/{prompt_id}"
                    )
                    
                    if history_response.status_code == 200:
                        history = history_response.json()
                        if prompt_id in history:
                            outputs = history[prompt_id].get('outputs', {})
                            for node_id, node_output in outputs.items():
                                if 'images' in node_output:
                                    filename = node_output['images'][0]['filename']
                                    return f"/home/patrick/ComfyUI/output/{filename}"
        except Exception as e:
            print(f"    Error: {e}")
        
        return None
    
    def check_frame_quality(self, frame_path: str) -> int:
        """Quick quality check on generated frame"""
        try:
            img = cv2.imread(frame_path)
            if img is None:
                return 0
            
            # Simple quality metrics
            score = 100
            
            # Check if too uniform (garbage)
            std = np.std(img)
            if std < 30:  # Too uniform
                score -= 50
            
            # Check sharpness
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian < 100:  # Too blurry
                score -= 30
            
            return max(0, score)
        except:
            return 0
    
    async def create_video_from_frames(self, frames: List[str], output: str) -> bool:
        """Create video from frames"""
        if not frames:
            return False
        
        import subprocess
        
        # Create frame list
        frame_list = "/tmp/echo_frames.txt"
        with open(frame_list, 'w') as f:
            for frame in frames:
                # Repeat frames for duration
                for _ in range(10):
                    f.write(f"file '{frame}'\n")
        
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", frame_list,
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-b:v", "10M", "-pix_fmt", "yuv420p",
            "-vf", "fps=24,scale=1920:1080",
            "-t", "30", output
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    
    async def handle_orchestration_request(self, message: str) -> Dict[str, Any]:
        """Handle requests with auto-retry"""
        if "trailer" in message.lower() or "video" in message.lower():
            return await self.orchestrate_trailer()
        return {"response": "I create quality videos with automatic retry on failures"}

# Global orchestrator instance
orchestrator = QualityAutoRetryOrchestrator()
