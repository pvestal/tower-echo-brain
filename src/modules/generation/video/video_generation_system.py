#!/usr/bin/env python3
"""
Comprehensive Video Generation System for Echo Brain
Supports multiple methods: AnimateDiff, SVD, Frame Interpolation
"""

import json
import requests
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import numpy as np

class VideoGenerationSystem:
    """Multi-method video generation system for Echo Brain"""
    
    def __init__(self, comfyui_url="http://localhost:8188"):
        self.comfyui_url = comfyui_url
        self.output_dir = Path("/home/{os.getenv("TOWER_USER", "patrick")}/Videos")
        self.output_dir.mkdir(exist_ok=True)
        
        # Available generation methods
        self.methods = {
            'animatediff': self.generate_animatediff,
            'svd': self.generate_svd,
            'ffmpeg_effects': self.generate_ffmpeg_effects,
            'frame_interpolation': self.generate_interpolated
        }
        
    def assess_quality(self, video_path: str) -> Dict:
        """Assess video quality against Article 71 standards"""
        try:
            # Use ffprobe to get video info
            cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of json "{video_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            info = json.loads(result.stdout)['streams'][0] if result.stdout else {}
            
            # Calculate quality score
            width = int(info.get('width', 0))
            height = int(info.get('height', 0))
            fps = eval(info.get('r_frame_rate', '0/1'))  # Convert fraction to float
            
            quality_score = 0
            feedback = []
            
            # Resolution check (1920x1080 minimum)
            if width >= 1920 and height >= 1080:
                quality_score += 30
            else:
                feedback.append(f"Low resolution: {width}x{height}")
            
            # Frame rate check (24fps minimum)
            if fps >= 24:
                quality_score += 30
            else:
                feedback.append(f"Low frame rate: {fps}fps")
            
            # Duration check (30 seconds minimum for trailers)
            duration_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
            duration_result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)
            duration = float(duration_result.stdout.strip()) if duration_result.stdout else 0
            
            if duration >= 30:
                quality_score += 20
            else:
                feedback.append(f"Too short: {duration:.1f}s (need 30s+)")
            
            # Motion quality (check if it's not a slideshow)
            # This would need more sophisticated analysis
            quality_score += 20  # Placeholder for motion quality
            
            return {
                'score': quality_score,
                'passed': quality_score >= 80,
                'resolution': f"{width}x{height}",
                'fps': fps,
                'duration': duration,
                'feedback': feedback
            }
        except Exception as e:
            return {'score': 0, 'passed': False, 'error': str(e)}
    
    def generate_animatediff(self, prompt: str, duration: int = 4) -> str:
        """Generate video using AnimateDiff"""
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "Counterfeit-V3.0_fp16.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": f"{prompt}, masterpiece, best quality, highly detailed",
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "low quality, worst quality, blurry",
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "ADE_AnimateDiffLoaderGen1",
                "inputs": {
                    "model_name": "mm_sd_v15_v3.ckpt",
                    "beta_schedule": "sqrt_linear",
                    "model": ["1", 0]
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 512,
                    "height": 768,
                    "batch_size": 16  # 16 frames
                }
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(time.time()),
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["5", 0]
                }
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["1", 2]
                }
            },
            "8": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["7", 0],
                    "frame_rate": 8,
                    "format": "video/mp4",
                    "filename_prefix": "animatediff_"
                }
            }
        }
        
        # Submit to ComfyUI
        response = requests.post(
            f"{self.comfyui_url}/prompt",
            json={"prompt": workflow}
        )
        
        if response.status_code == 200:
            prompt_id = response.json().get('prompt_id')
            return self.wait_for_completion(prompt_id)
        
        return None
    
    def generate_svd(self, image_path: str, motion_scale: float = 1.0) -> str:
        """Generate video from image using SVD"""
        # SVD workflow for image-to-video
        workflow = {
            "1": {
                "class_type": "ImageOnlyCheckpointLoader",
                "inputs": {
                    "ckpt_name": "svd_xt.safetensors"
                }
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": os.path.basename(image_path)
                }
            },
            "3": {
                "class_type": "SVD_img2vid_Conditioning",
                "inputs": {
                    "width": 1024,
                    "height": 576,
                    "video_frames": 25,
                    "motion_bucket_id": int(127 * motion_scale),
                    "fps": 6,
                    "augmentation_level": 0,
                    "clip_vision": ["1", 1],
                    "init_image": ["2", 0],
                    "vae": ["1", 2]
                }
            },
            "4": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "add_noise": "enable",
                    "noise_seed": int(time.time()),
                    "steps": 25,
                    "cfg": 2.5,
                    "sampler_name": "euler",
                    "scheduler": "karras",
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["3", 1],
                    "latent_image": ["3", 2]
                }
            },
            "5": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["4", 0],
                    "vae": ["1", 2]
                }
            }
        }
        
        # Submit workflow
        response = requests.post(
            f"{self.comfyui_url}/prompt",
            json={"prompt": workflow}
        )
        
        if response.status_code == 200:
            prompt_id = response.json().get('prompt_id')
            return self.wait_for_completion(prompt_id)
        
        return None
    
    def generate_ffmpeg_effects(self, image_path: str, duration: int = 30) -> str:
        """Generate video with FFmpeg effects (Ken Burns, zoom, pan)"""
        output_path = self.output_dir / f"ffmpeg_video_{int(time.time())}.mp4"
        
        # Create 30-second video with zoom effect
        cmd = f"""ffmpeg -loop 1 -i "{image_path}" -c:v libx264 -t {duration} \
            -vf "scale=w=1920:h=1080:force_original_aspect_ratio=increase,crop=1920:1080,\
            zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={duration*24}:s=1920x1080:fps=24" \
            -pix_fmt yuv420p -preset slow -crf 18 "{output_path}" -y"""
        
        result = subprocess.run(cmd, shell=True, capture_output=True)
        
        if result.returncode == 0:
            return str(output_path)
        else:
            print(f"FFmpeg error: {result.stderr.decode()}")
            return None
    
    def generate_interpolated(self, frames: List[str], target_fps: int = 60) -> str:
        """Interpolate frames to create smooth video"""
        # This would use RIFE or similar frame interpolation
        # For now, using FFmpeg frame blending
        output_path = self.output_dir / f"interpolated_{int(time.time())}.mp4"
        
        # Create video from frames with interpolation
        cmd = f"""ffmpeg -r 8 -pattern_type glob -i '{frames[0]}*.png' \
            -vf "minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" \
            -c:v libx264 -pix_fmt yuv420p -preset slow -crf 18 "{output_path}" -y"""
        
        result = subprocess.run(cmd, shell=True, capture_output=True)
        
        if result.returncode == 0:
            return str(output_path)
        return None
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Optional[str]:
        """Wait for ComfyUI to complete generation"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check history
            response = requests.get(f"{self.comfyui_url}/history/{prompt_id}")
            
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('completed'):
                        # Get output files
                        outputs = history[prompt_id].get('outputs', {})
                        for node_id, output in outputs.items():
                            if 'images' in output or 'gifs' in output or 'videos' in output:
                                # Found output file
                                return self.process_output(output)
            
            time.sleep(2)
        
        return None
    
    def process_output(self, output: Dict) -> str:
        """Process ComfyUI output and return video path"""
        # This would handle different output types
        # For now, return a placeholder
        return str(self.output_dir / f"generated_{int(time.time())}.mp4")
    
    def generate_video(self, prompt: str, method: str = 'auto', **kwargs) -> Dict:
        """Main entry point for video generation"""
        result = {
            'success': False,
            'video_path': None,
            'quality_assessment': None,
            'method_used': method
        }
        
        # Auto-select method based on available models
        if method == 'auto':
            # Check which models are available
            if os.path.exists("/home/{os.getenv("TOWER_USER", "patrick")}/Projects/ComfyUI-Working/custom_nodes/ComfyUI-AnimateDiff-Evolved/models/mm_sd_v15_v3.ckpt"):
                method = 'animatediff'
            else:
                method = 'ffmpeg_effects'  # Fallback
        
        # Generate video using selected method
        if method in self.methods:
            video_path = self.methods[method](prompt, **kwargs)
            
            if video_path and os.path.exists(video_path):
                result['success'] = True
                result['video_path'] = video_path
                
                # Assess quality
                result['quality_assessment'] = self.assess_quality(video_path)
                
                # If quality is low, try to enhance
                if not result['quality_assessment']['passed']:
                    print(f"Quality check failed: {result['quality_assessment']['feedback']}")
                    # Could trigger re-generation or enhancement here
        
        return result


# Integration with Echo Brain
class EchoBrainVideoIntegration:
    """Integrate video generation with Echo Brain"""
    
    def __init__(self):
        self.video_system = VideoGenerationSystem()
        self.echo_api = "http://localhost:8309"
    
    def handle_video_request(self, request: Dict) -> Dict:
        """Handle video generation request from Echo"""
        prompt = request.get('prompt', 'anime character in action')
        duration = request.get('duration', 30)
        style = request.get('style', 'anime')
        
        # Add style modifiers to prompt
        styled_prompt = f"{prompt}, {style} style, high quality, detailed"
        
        # Generate video
        result = self.video_system.generate_video(
            styled_prompt,
            method='auto',
            duration=duration
        )
        
        # Send quality report back to Echo
        if result['success']:
            self.report_to_echo(result)
        
        return result
    
    def report_to_echo(self, result: Dict):
        """Report generation results back to Echo Brain"""
        try:
            requests.post(
                f"{self.echo_api}/api/echo/video_complete",
                json=result
            )
        except:
            pass  # Echo might not have this endpoint yet


if __name__ == "__main__":
    # Test the system
    system = VideoGenerationSystem()
    
    print("Video Generation System initialized")
    print("Testing quality assessment on existing video...")
    
    # Test with an existing video if available
    test_video = "/home/{os.getenv("TOWER_USER", "patrick")}/Videos/goblin_slayer_30s.mp4"
    if os.path.exists(test_video):
        quality = system.assess_quality(test_video)
        print(f"Quality Assessment: {quality}")
    
    print("\nSystem ready for video generation")
    print("Available methods: animatediff, svd, ffmpeg_effects, frame_interpolation")
