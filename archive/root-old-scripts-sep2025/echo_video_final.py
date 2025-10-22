#!/usr/bin/env python3
"""Echo Video Generation - FINAL WORKING VERSION"""
import subprocess
import time
from pathlib import Path
import uuid

def generate_video_simple(prompt="goblin slayer cyberpunk", duration=30):
    """Generate video using the script that ACTUALLY WORKED earlier"""
    
    # Use the script that successfully generated video earlier
    script_path = "/opt/tower-anime-production/generate_anime_video.py"
    
    if not Path(script_path).exists():
        print("Creating working video generation script...")
        script = '''#!/usr/bin/env python3
import json
import requests
import time
import uuid
import subprocess
import os
from pathlib import Path

COMFYUI_API = "http://127.0.0.1:8188"
OUTPUT_DIR = "***REMOVED***/ComfyUI-Working/output"
VIDEO_DIR = "***REMOVED***"

def generate_frames(base_prompt, num_frames=10):
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "animagine_xl_3.1.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": base_prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality, low quality", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0, "model": ["1", 0],
            "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]
        }},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "echo_frame", "images": ["6", 0]}}
    }
    
    frames = []
    for i in range(num_frames):
        workflow["5"]["inputs"]["seed"] = 42 + i * 100
        workflow["7"]["inputs"]["filename_prefix"] = f"echo_{uuid.uuid4().hex[:8]}"
        
        response = requests.post(f"{COMFYUI_API}/prompt", json={"prompt": workflow})
        if response.status_code == 200:
            frames.append(workflow["7"]["inputs"]["filename_prefix"])
            print(f"Frame {i+1}/{num_frames} queued")
            time.sleep(3)
    
    return frames

def create_video(pattern, output_name, duration=30):
    frame_count = len(list(Path(OUTPUT_DIR).glob(pattern)))
    if frame_count == 0:
        return None
        
    fps = max(1, frame_count / duration)
    
    cmd = [
        "ffmpeg", "-y", "-pattern_type", "glob", "-i", f"{OUTPUT_DIR}/{pattern}",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-vf", f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps={fps}",
        "-pix_fmt", "yuv420p", f"{VIDEO_DIR}/{output_name}.mp4"
    ]
    
    subprocess.run(cmd, check=True)
    return f"{VIDEO_DIR}/{output_name}.mp4"

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "goblin slayer cyberpunk"
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    print(f"Generating {num_frames} frames...")
    frames = generate_frames(prompt, num_frames)
    
    print(f"Waiting for generation...")
    time.sleep(num_frames * 5 + 10)
    
    print(f"Creating video...")
    video_path = create_video("echo_*.png", f"echo_video_{uuid.uuid4().hex[:8]}", duration)
    print(f"Video created: {video_path}")
'''
        Path(script_path).parent.mkdir(exist_ok=True)
        with open(script_path, "w") as f:
            f.write(script)
        Path(script_path).chmod(0o755)
    
    # Run the script
    cmd = ["python3", script_path, prompt, "10", str(duration)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"SUCCESS: {result.stdout}")
        return {"status": "success", "output": result.stdout}
    else:
        print(f"ERROR: {result.stderr}")
        return {"status": "error", "error": result.stderr}

if __name__ == "__main__":
    print("Generating Goblin Slayer video with Echo...")
    result = generate_video_simple("goblin slayer cyberpunk armor, neon city, rain", 20)
    print(f"Result: {result}")
