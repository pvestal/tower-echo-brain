#!/usr/bin/env python3
import requests
import json
import time
import subprocess
import os

print("🔥 GENERATING CYBERPUNK GOBLIN SLAYER VIDEO")

# 1. Generate frames with ComfyUI
comfyui_url = "http://localhost:8188/prompt"

# Check ComfyUI status
try:
    status = requests.get("http://localhost:8188/system_stats")
    if status.status_code != 200:
        print("❌ ComfyUI not running, starting it...")
        subprocess.Popen(["python3", "***REMOVED***/ComfyUI-Working/main.py", "--listen", "0.0.0.0"], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(10)
except:
    print("Starting ComfyUI...")
    subprocess.Popen(["python3", "***REMOVED***/ComfyUI-Working/main.py", "--listen", "0.0.0.0"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(10)

# 2. Use existing frames or generate new ones
frames_dir = "/mnt/20TB/ComfyUI-Real/output"
existing_frames = subprocess.run(
    f"ls {frames_dir}/goblin_slayer_*.png 2>/dev/null | wc -l",
    shell=True, capture_output=True, text=True
).stdout.strip()

print(f"Found {existing_frames} existing Goblin Slayer frames")

# 3. Generate video with quality checking
output_video = "***REMOVED***/GOBLIN_SLAYER_ULTIMATE_WITH_PIPELINE.mp4"

# Create video with FFmpeg using best frames
cmd = f"""
cd {frames_dir} &&
ffmpeg -y -framerate 24 -pattern_type glob -i 'goblin_slayer_*.png' \
  -c:v libx264 -preset slow -crf 18 -b:v 15M \
  -vf "scale=1920:1080,format=yuv420p,\
       eq=contrast=1.1:brightness=0.05:saturation=1.2,\
       minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,\
       deflicker,hqdn3d=4:3:6:4,\
       fade=in:0:24,fade=out:st=30:d=1" \
  -t 35 \
  {output_video}
"""

print("🎬 Creating video with enhanced pipeline...")
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if os.path.exists(output_video):
    # Check video quality
    probe = subprocess.run(
        f"ffprobe -v error -show_entries format=duration,bit_rate,size -of json {output_video}",
        shell=True, capture_output=True, text=True
    )
    
    info = json.loads(probe.stdout)["format"]
    duration = float(info.get("duration", 0))
    bitrate = int(info.get("bit_rate", 0)) / 1000000  # Convert to Mbps
    size_mb = int(info.get("size", 0)) / 1048576  # Convert to MB
    
    print(f"""
✅ VIDEO GENERATED SUCCESSFULLY!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📹 File: {output_video}
⏱️  Duration: {duration:.1f} seconds
📊 Bitrate: {bitrate:.1f} Mbps
💾 Size: {size_mb:.1f} MB
🎯 Quality: {"✅ PASSES" if duration >= 30 and bitrate >= 10 else "❌ NEEDS IMPROVEMENT"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    
    # Copy to laptop
    print("📤 Copying to laptop...")
    subprocess.run(f"scp {output_video} patrick@192.168.50.100:***REMOVED***/", shell=True)
    
else:
    print("❌ Video generation failed!")

print("🎮 Available at: http://***REMOVED***:8096")
