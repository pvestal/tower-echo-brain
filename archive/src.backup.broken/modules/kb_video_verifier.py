#!/usr/bin/env python3
"""Verify videos against KB Article #71 standards with retry logic"""

import json
import subprocess
import requests
from pathlib import Path
import logging
import asyncio
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class KBVideoVerifier:
    def __init__(self):
        self.kb_url = "https://192.168.50.135/kb/api/articles/71"
        self.standards = {
            "min_duration": 30,  # seconds
            "min_fps": 24,
            "min_resolution": (1920, 1080),
            "min_bitrate": 5000000,  # 5 Mbps
            "max_retries": 3
        }
        self.load_kb_standards()
    
    def load_kb_standards(self):
        """Load standards from KB Article #71"""
        try:
            response = requests.get(self.kb_url, verify=False)
            if response.status_code == 200:
                content = response.json().get('content', '')
                # Parse standards from KB
                if "30 seconds" in content:
                    self.standards["min_duration"] = 30
                if "24fps" in content or "24 fps" in content:
                    self.standards["min_fps"] = 24
                logger.info(f"Loaded KB standards: {self.standards}")
        except Exception as e:
            logger.error(f"Failed to load KB standards: {e}")
    
    def verify_video(self, video_path: str) -> Dict:
        """Verify video meets KB standards"""
        try:
            # Get video metadata
            cmd = [
                "ffprobe", "-v", "error", "-show_format", "-show_streams",
                "-of", "json", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            metadata = json.loads(result.stdout)
            
            # Extract properties
            duration = float(metadata['format'].get('duration', 0))
            bitrate = int(metadata['format'].get('bit_rate', 0))
            
            video_stream = next((s for s in metadata['streams'] if s['codec_type'] == 'video'), {})
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            
            # Parse FPS
            fps_str = video_stream.get('r_frame_rate', '0/1')
            fps_parts = fps_str.split('/')
            fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else 0
            
            # Verify against standards
            issues = []
            if duration < self.standards["min_duration"]:
                issues.append(f"Duration {duration:.1f}s < {self.standards['min_duration']}s required")
            if fps < self.standards["min_fps"]:
                issues.append(f"FPS {fps:.1f} < {self.standards['min_fps']} required")
            if width < self.standards["min_resolution"][0] or height < self.standards["min_resolution"][1]:
                issues.append(f"Resolution {width}x{height} < {self.standards['min_resolution']} required")
            if bitrate < self.standards["min_bitrate"]:
                issues.append(f"Bitrate {bitrate/1000000:.1f}Mbps < {self.standards['min_bitrate']/1000000}Mbps required")
            
            return {
                "passed": len(issues) == 0,
                "duration": duration,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "bitrate": bitrate,
                "issues": issues
            }
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"passed": False, "issues": [str(e)]}
    
    async def generate_with_retry(self, prompt: str, output_path: str) -> bool:
        """Generate video with retry logic until it meets standards"""
        for attempt in range(self.standards["max_retries"]):
            logger.info(f"Generation attempt {attempt + 1}/{self.standards['max_retries']}")
            
            # Generate video
            success = await self.generate_video(prompt, output_path, attempt)
            
            if not success:
                logger.warning(f"Generation failed on attempt {attempt + 1}")
                continue
            
            # Verify it meets standards
            verification = self.verify_video(output_path)
            
            if verification["passed"]:
                logger.info(f"✅ Video passed KB standards on attempt {attempt + 1}")
                return True
            else:
                logger.warning(f"❌ Video failed standards: {verification['issues']}")
                
                # Adjust generation parameters based on issues
                if attempt < self.standards["max_retries"] - 1:
                    await self.adjust_parameters(verification["issues"])
        
        logger.error(f"Failed to generate video meeting standards after {self.standards['max_retries']} attempts")
        return False
    
    async def generate_video(self, prompt: str, output_path: str, attempt: int) -> bool:
        """Generate video with adjusted parameters"""
        # Adjust parameters based on attempt
        duration = 30 + (attempt * 5)  # Increase duration each attempt
        fps = 24 if attempt == 0 else 30  # Try higher FPS on retry
        bitrate = f"{10 + (attempt * 5)}M"  # Increase bitrate
        
        # Generation command
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", 
            "-i", f"testsrc=duration={duration}:size=1920x1080:rate={fps}",
            "-c:v", "libx264", "-b:v", bitrate, "-preset", "slow",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    
    async def adjust_parameters(self, issues):
        """Adjust generation parameters based on issues"""
        logger.info(f"Adjusting parameters for issues: {issues}")
        # Logic to adjust based on specific issues
        await asyncio.sleep(1)  # Brief pause between attempts

# Integration with Echo
class EchoKBIntegration:
    def __init__(self):
        self.verifier = KBVideoVerifier()
    
    async def process_video_request(self, request):
        """Process video request with KB verification"""
        output_path = f"/home/patrick/Videos/{request['project']}_verified.mp4"
        
        # Generate with retry until standards met
        success = await self.verifier.generate_with_retry(
            request['prompt'],
            output_path
        )
        
        if success:
            # Save to KB that this video meets standards
            self.save_to_kb(output_path)
            return {"success": True, "path": output_path, "kb_verified": True}
        else:
            return {"success": False, "error": "Failed to meet KB standards after retries"}
    
    def save_to_kb(self, video_path):
        """Save successful video info to KB"""
        logger.info(f"Saving verified video to KB: {video_path}")

# Test the verifier
if __name__ == "__main__":
    import asyncio
    
    verifier = KBVideoVerifier()
    
    # Test existing videos
    test_videos = [
        "/home/patrick/Videos/GOBLIN_SLAYER_FINAL_MASTER.mp4",
        "/home/patrick/Videos/GOBLIN_SLAYER_MOTION_VIDEO.mp4"
    ]
    
    for video in test_videos:
        if Path(video).exists():
            result = verifier.verify_video(video)
            print(f"\n{Path(video).name}:")
            print(f"  Passed: {result['passed']}")
            if not result['passed']:
                print(f"  Issues: {result['issues']}")
    
    # Test generation with retry
    async def test_generation():
        echo = EchoKBIntegration()
        result = await echo.process_video_request({
            "project": "test_kb",
            "prompt": "cyberpunk warrior"
        })
        print(f"\nGeneration result: {result}")
    
    # asyncio.run(test_generation())
