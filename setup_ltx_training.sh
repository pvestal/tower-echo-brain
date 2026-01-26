#!/bin/bash
# Setup LTX LoRA training environment using ai-toolkit

set -e

echo "Setting up LTX LoRA training environment..."

# Create directory for ai-toolkit
AI_TOOLKIT_DIR="/opt/tower-lora-studio/ai-toolkit"
mkdir -p "$AI_TOOLKIT_DIR"
cd "$AI_TOOLKIT_DIR"

# Clone ai-toolkit if not exists
if [ ! -d ".git" ]; then
    echo "Cloning ai-toolkit..."
    git clone https://github.com/ostris/ai-toolkit.git .
else
    echo "Updating ai-toolkit..."
    git pull
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install diffusers transformers accelerate safetensors

# Create config directory
CONFIG_DIR="/opt/tower-lora-studio/configs/ltx"
mkdir -p "$CONFIG_DIR"

# Create a basic LTX training config template
cat > "$CONFIG_DIR/ltx_lora_template.yaml" << 'EOF'
job: extension
config:
  name: "ltx_custom_lora"
  process:
    - type: ltx_video_lora

model:
  name_or_path: "Lightricks/LTX-Video"
  revision: "refs/pr/18"
  quantize: true

network:
  type: "lora"
  linear: 16
  linear_alpha: 16

datasets:
  - folder_path: "/opt/tower-lora-studio/datasets/ltx_training"
    caption_ext: "txt"
    caption_dropout_rate: 0.05
    resolution: [768, 512, 121]

train:
  batch_size: 1
  steps: 500
  gradient_accumulation_steps: 1
  train_unet: true
  train_text_encoder: false
  gradient_checkpointing: true
  noise_scheduler: "ddpm"
  optimizer: "adamw8bit"
  lr: 1e-4

sample:
  sampler: "ddpm"
  sample_every: 100
  width: 768
  height: 512
  frames: 121
  prompts:
    - "a woman in cowgirl position"

save:
  dtype: "float16"
  save_every: 100
  filename_prefix: "ltx_lora"
EOF

echo "Creating LTX training wrapper..."
cat > "/opt/tower-lora-studio/train_ltx_lora.py" << 'EOF'
#!/usr/bin/env python3
"""
LTX LoRA Training Wrapper
Integrates ai-toolkit with Tower infrastructure
"""

import sys
import os
import subprocess
import json
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append('/opt/tower-lora-studio/ai-toolkit')

def prepare_ltx_dataset(video_path, output_dir, fps=8):
    """Extract frames from video for LTX training"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(output_dir / "frame_%04d.png")
    ]

    subprocess.run(cmd, check=True)

    # Create caption file
    caption = "a woman in intimate position, adult content"
    for frame in output_dir.glob("*.png"):
        caption_file = frame.with_suffix(".txt")
        caption_file.write_text(caption)

    logger.info(f"Prepared {len(list(output_dir.glob('*.png')))} frames for training")

def train_ltx_lora(config_path):
    """Run LTX LoRA training using ai-toolkit"""

    # Activate venv and run training
    activate_cmd = "source /opt/tower-lora-studio/ai-toolkit/venv/bin/activate"
    train_cmd = f"python run.py {config_path}"

    full_cmd = f"{activate_cmd} && cd /opt/tower-lora-studio/ai-toolkit && {train_cmd}"

    process = subprocess.Popen(
        full_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        executable="/bin/bash"
    )

    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.strip())

    process.wait()
    return process.returncode == 0

if __name__ == "__main__":
    # Quick test with existing video
    test_video = "/mnt/1TB-storage/ComfyUI/output/custom_lora_test_00001.mp4"
    dataset_dir = "/opt/tower-lora-studio/datasets/ltx_training"

    logger.info("Preparing dataset...")
    prepare_ltx_dataset(test_video, dataset_dir)

    logger.info("Starting LTX LoRA training...")
    config_path = "/opt/tower-lora-studio/configs/ltx/ltx_lora_template.yaml"

    if train_ltx_lora(config_path):
        logger.info("Training completed successfully")
    else:
        logger.error("Training failed")
EOF

chmod +x "/opt/tower-lora-studio/train_ltx_lora.py"

echo "Setup complete!"
echo "To train LTX LoRA: python3 /opt/tower-lora-studio/train_ltx_lora.py"