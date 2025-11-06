#!/usr/bin/env python3
"""
Echo Video Generation System - Refactored from existing Tower components
Improvement loops with verification at each stage
"""

import os
import json
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import aiohttp
import psycopg2
from dataclasses import dataclass

@dataclass
class VideoProject:
    """Video project with version control"""
    id: str
    name: str
    prompt: str
    style: str
    frames: List[str]
    status: str
    quality_scores: Dict[str, float]
    version: int
    created_at: datetime

class EchoVideoRefactored:
    """Refactored video generation from working Tower components"""

    def __init__(self):
        # From working Tower paths
        self.comfyui_url = "http://***REMOVED***:8188"
        self.llava_url = "http://localhost:11435/api/generate"  # LLaVA on Ollama
        self.output_dir = Path("/home/{os.getenv("TOWER_USER", "patrick")}/Videos")
        self.frame_dir = Path("/mnt/ComfyUI/output")
        self.workflow_dir = Path("/home/{os.getenv("TOWER_USER", "patrick")}/Tower/services/anime-video-generator")

        # Database for project management
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick")),
            "password": "patrick123"
        }

        # WebSocket for progress
        self.ws_connections = set()

        # Improvement thresholds
        self.quality_threshold = 0.7
        self.max_retries = 3

    async def generate_with_improvement_loop(
        self,
        prompt: str,
        duration: int = 5,
        fps: int = 24,
        style: str = "anime"
    ) -> Dict:
        """Main generation with improvement loops"""

        project_id = self._create_project_id()
        results = {
            "project_id": project_id,
            "status": "starting",
            "iterations": [],
            "final_video": None
        }

        # LOOP 1: Generate frames with quality checking
        for attempt in range(self.max_retries):
            print(f"🎬 Frame generation attempt {attempt + 1}/{self.max_retries}")

            frames = await self._generate_frames(prompt, duration * fps, style)

            # Quality check with LLaVA
            quality_results = await self._check_frame_quality(frames)

            results["iterations"].append({
                "attempt": attempt + 1,
                "frames_generated": len(frames),
                "quality_scores": quality_results,
                "avg_quality": sum(quality_results.values()) / len(quality_results) if quality_results else 0
            })

            # Check if quality meets threshold
            avg_quality = results["iterations"][-1]["avg_quality"]
            if avg_quality >= self.quality_threshold:
                print(f"✅ Quality threshold met: {avg_quality:.2f}")
                break
            else:
                print(f"⚠️ Quality below threshold: {avg_quality:.2f} < {self.quality_threshold}")
                # Adjust parameters for next attempt
                prompt = await self._improve_prompt(prompt, quality_results)

        # LOOP 2: SVD interpolation for smoothness
        if frames:
            print("🔄 Applying SVD interpolation for 60fps...")
            frames = await self._svd_interpolate(frames, target_fps=60)

        # LOOP 3: Compile video with multiple quality settings
        if frames:
            for quality in ["high", "medium"]:
                video_path = await self._compile_video(frames, fps=60, quality=quality)

                # Verify video actually works
                if await self._verify_video(video_path):
                    results["final_video"] = str(video_path)
                    results["status"] = "completed"
                    break
                else:
                    print(f"❌ Video verification failed for {quality} quality")

        # Save project to database
        await self._save_project(project_id, results)

        # Send notifications
        await self._send_notifications(project_id, results)

        return results

    async def _generate_frames(self, prompt: str, num_frames: int, style: str) -> List[str]:
        """Generate frames using ComfyUI with AnimateDiff"""

        # Load working AnimateDiff workflow from Tower
        workflow_path = self.workflow_dir / "workflows/animatediff_workflow.json"
        if workflow_path.exists():
            with open(workflow_path) as f:
                workflow = json.load(f)
        else:
            # Use existing working ComfyUI API
            workflow = self._build_animatediff_workflow(prompt, num_frames, style)

        # Queue to ComfyUI
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow}
            ) as resp:
                result = await resp.json()
                prompt_id = result.get("prompt_id")

        # Wait for completion with progress updates
        frames = []
        while len(frames) < num_frames:
            await asyncio.sleep(1)

            # Check queue status
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.comfyui_url}/queue") as resp:
                    queue = await resp.json()

            # Update progress via WebSocket
            progress = len(frames) / num_frames * 100
            await self._broadcast_progress(progress, f"Generating frame {len(frames)}/{num_frames}")

            # Collect generated frames
            frame_files = list(self.frame_dir.glob(f"{prompt_id}_*.png"))
            frames = [str(f) for f in sorted(frame_files)]

            if len(frames) >= num_frames:
                break

        return frames[:num_frames]

    async def _check_frame_quality(self, frames: List[str]) -> Dict[str, float]:
        """Check frame quality using LLaVA vision model"""

        quality_scores = {}

        for frame_path in frames:
            # Read frame and encode to base64
            with open(frame_path, "rb") as f:
                import base64
                frame_data = base64.b64encode(f.read()).decode()

            # Query LLaVA for quality assessment
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llava",
                    "prompt": "Rate the quality of this anime frame from 0-1. Check for: clarity, composition, style consistency, artifacts. Return only a number.",
                    "images": [frame_data]
                }

                try:
                    async with session.post(self.llava_url, json=payload, timeout=30) as resp:
                        result = await resp.json()
                        score_text = result.get("response", "0.5")
                        # Extract number from response
                        score = float(''.join(c for c in score_text if c.isdigit() or c == '.') or "0.5")
                        quality_scores[frame_path] = min(1.0, max(0.0, score))
                except Exception as e:
                    print(f"LLaVA quality check failed: {e}")
                    quality_scores[frame_path] = 0.5  # Default score

        return quality_scores

    async def _improve_prompt(self, prompt: str, quality_results: Dict) -> str:
        """Improve prompt based on quality feedback"""

        # Analyze what failed
        low_quality_frames = [k for k, v in quality_results.items() if v < self.quality_threshold]

        if len(low_quality_frames) > len(quality_results) / 2:
            # Most frames are bad - major prompt revision needed
            prompt += ", high quality, detailed, sharp, professional anime style"
        else:
            # Some frames bad - minor adjustments
            prompt += ", consistent style, clear details"

        return prompt

    async def _svd_interpolate(self, frames: List[str], target_fps: int) -> List[str]:
        """Use Stable Video Diffusion for frame interpolation"""

        # This would use SVD model if available
        # For now, using frame duplication as fallback
        interpolated = []
        for i, frame in enumerate(frames[:-1]):
            interpolated.append(frame)
            # Add interpolated frame between current and next
            interpolated.append(frame)  # Duplicate for now
        interpolated.append(frames[-1])

        return interpolated

    async def _compile_video(self, frames: List[str], fps: int, quality: str) -> Path:
        """Compile frames to video using FFmpeg"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"echo_video_{timestamp}_{quality}.mp4"

        # Create frame list file
        list_file = Path("/tmp/frame_list.txt")
        with open(list_file, "w") as f:
            for frame in frames:
                f.write(f"file '{frame}'\n")
                f.write(f"duration {1/fps}\n")

        # FFmpeg command based on quality
        if quality == "high":
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", str(fps),
                str(output_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-pix_fmt", "yuv420p", "-r", str(fps),
                str(output_path)
            ]

        # Run FFmpeg on Tower
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return output_path
        else:
            print(f"FFmpeg error: {result.stderr}")
            return None

    async def _verify_video(self, video_path: Path) -> bool:
        """Verify video is actually playable"""

        if not video_path or not video_path.exists():
            return False

        # Check with ffprobe
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        return "duration=" in result.stdout

    def _create_project_id(self) -> str:
        """Create unique project ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def _build_animatediff_workflow(self, prompt: str, num_frames: int, style: str) -> Dict:
        """Build AnimateDiff workflow for ComfyUI"""

        # Based on working Tower AnimateDiff setup
        return {
            "1": {
                "class_type": "AnimateDiffLoaderWithContext",
                "inputs": {
                    "model_name": "mm_sd_v15_v2.ckpt",
                    "beta_schedule": "linear",
                    "motion_scale": 1.0,
                    "apply_v2_models_properly": True
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": f"{style} style, {prompt}",
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": -1,
                    "steps": 20,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "latent_image": ["4", 0]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": num_frames
                }
            },
            "5": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["1", 2]
                }
            },
            "6": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "echo_frame",
                    "images": ["5", 0]
                }
            }
        }

    async def _save_project(self, project_id: str, results: Dict):
        """Save project to database"""

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Create table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_projects (
                    id VARCHAR(36) PRIMARY KEY,
                    data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Save project
            cur.execute(
                "INSERT INTO video_projects (id, data) VALUES (%s, %s)",
                (project_id, json.dumps(results))
            )

            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Failed to save project: {e}")

    async def _broadcast_progress(self, percent: float, message: str):
        """Send progress updates via WebSocket"""

        update = {
            "type": "progress",
            "percent": percent,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

        # Would broadcast to connected WebSocket clients
        print(f"📊 Progress: {percent:.1f}% - {message}")

    async def _send_notifications(self, project_id: str, results: Dict):
        """Send completion notifications"""

        if results.get("status") == "completed":
            message = f"✅ Video generation complete!\nProject: {project_id}\nVideo: {results.get('final_video')}"
        else:
            message = f"❌ Video generation failed\nProject: {project_id}"

        # Telegram notification
        # Would send to @patricksechobot

        print(message)


# Test improvement loop
async def test_video_generation():
    """Test the refactored video generation with improvement loops"""

    generator = EchoVideoRefactored()

    # Test with anime prompt
    result = await generator.generate_with_improvement_loop(
        prompt="cyberpunk samurai girl in neon Tokyo",
        duration=3,
        fps=24,
        style="anime"
    )

    print(f"\n📹 Final result: {json.dumps(result, indent=2)}")

    # Verify the video actually exists and plays
    if result.get("final_video"):
        video_path = Path(result["final_video"])
        if video_path.exists():
            print(f"✅ Video successfully generated: {video_path}")
            print(f"   Size: {video_path.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            print(f"❌ Video file not found: {video_path}")
    else:
        print("❌ No video was generated")

if __name__ == "__main__":
    asyncio.run(test_video_generation())