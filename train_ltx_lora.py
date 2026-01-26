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
