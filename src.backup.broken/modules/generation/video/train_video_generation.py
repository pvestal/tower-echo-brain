#!/usr/bin/env python3
"""Train Echo on successful video generation patterns"""

import json
import requests
import time
import subprocess
from pathlib import Path

class EchoVideoTrainer:
    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.successful_workflows = []
        
    def test_comfyui(self):
        """Test if ComfyUI is running"""
        try:
            r = requests.get(f"{self.comfyui_url}/system_stats", timeout=2)
            return r.status_code == 200
        except:
            return False
    
    def generate_frame(self, prompt, seed=42):
        """Generate a single frame with proven workflow"""
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", 
                  "inputs": {"ckpt_name": "animagine_xl_3.1.safetensors"}},
            "2": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "bad quality", "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "5": {"class_type": "KSampler",
                  "inputs": {"seed": seed, "steps": 20, "cfg": 7, 
                           "sampler_name": "euler", "scheduler": "normal",
                           "denoise": 1, "model": ["1", 0], 
                           "positive": ["2", 0], "negative": ["3", 0],
                           "latent_image": ["4", 0]}},
            "6": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage",
                  "inputs": {"images": ["6", 0], 
                           "filename_prefix": f"echo_training_{seed}"}}
        }
        
        try:
            r = requests.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow})
            if r.status_code == 200:
                self.successful_workflows.append(workflow)
                return r.json().get("prompt_id")
        except Exception as e:
            print(f"Error: {e}")
        return None
    
    def create_video(self, frame_pattern, output_path):
        """Create video from frames"""
        cmd = [
            "ffmpeg", "-y", "-framerate", "2", 
            "-pattern_type", "glob", "-i", frame_pattern,
            "-c:v", "libx264", "-r", "24", "-pix_fmt", "yuv420p",
            "-vf", "scale=1920:1080", output_path
        ]
        subprocess.run(cmd, capture_output=True)
        return Path(output_path).exists()
    
    def train_echo(self):
        """Train Echo on successful patterns"""
        print("🧠 Training Echo on video generation...")
        
        # Test prompts for training
        prompts = [
            "cyberpunk warrior, neon city, rain",
            "futuristic samurai, holographic katana",
            "tech ninja, digital smoke, action pose",
            "cyber priestess, glowing magic, beautiful"
        ]
        
        if not self.test_comfyui():
            print("❌ ComfyUI not running")
            return False
        
        # Generate training frames
        for i, prompt in enumerate(prompts):
            print(f"Generating: {prompt[:30]}...")
            self.generate_frame(prompt, seed=1000+i)
            time.sleep(5)
        
        # Save successful patterns
        with open("/opt/tower-echo-brain/successful_workflows.json", "w") as f:
            json.dump(self.successful_workflows, f, indent=2)
        
        print(f"✅ Saved {len(self.successful_workflows)} successful workflows")
        return True

if __name__ == "__main__":
    trainer = EchoVideoTrainer()
    trainer.train_echo()
