#!/usr/bin/env python3
"""
LORA Training Worker
Trains LORA models using kohya_ss
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any
import subprocess
import shutil

logger = logging.getLogger(__name__)

class LoraTrainingWorker:
    """Worker that trains LORA models using kohya_ss"""

    def __init__(self, kohya_path: str = "/opt/kohya_ss"):
        self.kohya_path = Path(kohya_path)
        self.output_base = Path("/mnt/1TB-storage/ComfyUI/models/loras")

    async def train_lora_model(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a LORA model using kohya_ss

        Args:
            task_payload: {
                'training_dir': str - Directory with tagged images,
                'output_name': str - LORA filename,
                'base_model': str - Path to base model (e.g., DreamShaper 8),
                'character_name': str,
                'network_dim': int (default 32),
                'network_alpha': int (default 16),
                'epochs': int (default 15),
                'learning_rate': float (default 1e-4)
            }

        Returns:
            Dict with training results
        """
        training_dir = Path(task_payload.get('training_dir'))
        output_name = task_payload.get('output_name')
        base_model = task_payload.get('base_model', '/mnt/1TB-storage/ComfyUI/models/checkpoints/dreamshaper_8.safetensors')
        character_name = task_payload.get('character_name')

        # Training hyperparameters
        network_dim = task_payload.get('network_dim', 32)
        network_alpha = task_payload.get('network_alpha', 16)
        epochs = task_payload.get('epochs', 15)
        learning_rate = task_payload.get('learning_rate', 1e-4)

        logger.info(f"🧠 Starting LORA training for {character_name}")
        logger.info(f"📁 Training directory: {training_dir}")
        logger.info(f"🎯 Output name: {output_name}")

        if not training_dir.exists():
            raise ValueError(f"Training directory does not exist: {training_dir}")

        # Ensure output directory exists
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Setup kohya training directory structure
        kohya_dataset_dir = await self._prepare_kohya_dataset(training_dir, character_name)

        # Create training config
        config_file = await self._create_training_config(
            dataset_dir=kohya_dataset_dir,
            output_name=output_name,
            base_model=base_model,
            network_dim=network_dim,
            network_alpha=network_alpha,
            epochs=epochs,
            learning_rate=learning_rate
        )

        # Check if kohya_ss is available
        if not self.kohya_path.exists():
            logger.warning(f"⚠️ kohya_ss not found at {self.kohya_path}")
            logger.warning("Simulating training for development...")
            return await self._simulate_training(output_name, character_name)

        # Execute training
        try:
            result = await self._execute_training(config_file, output_name)
            return result

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {
                'status': 'failed',
                'character_name': character_name,
                'output_name': output_name,
                'error': str(e)
            }

    async def _prepare_kohya_dataset(self, training_dir: Path, character_name: str) -> Path:
        """Prepare dataset in kohya_ss expected format"""

        # Kohya expects structure: dataset/character_name_num_repeats/images
        kohya_dataset = Path(f"/tmp/kohya_dataset_{character_name}")
        image_dir = kohya_dataset / f"{character_name}_10_character"

        # Create directories
        image_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and tags to kohya dataset directory
        images_source = training_dir / 'images'
        if images_source.exists():
            for img_file in images_source.glob('*'):
                if img_file.suffix in ['.png', '.jpg', '.txt']:
                    shutil.copy2(img_file, image_dir / img_file.name)

        logger.info(f"📦 Prepared kohya dataset at: {kohya_dataset}")
        return kohya_dataset

    async def _create_training_config(self, dataset_dir: Path, output_name: str,
                                     base_model: str, network_dim: int,
                                     network_alpha: int, epochs: int,
                                     learning_rate: float) -> Path:
        """Create kohya_ss training configuration file"""

        output_path = self.output_base / f"{output_name}.safetensors"

        config = {
            "pretrained_model_name_or_path": base_model,
            "output_dir": str(self.output_base),
            "output_name": output_name,
            "train_data_dir": str(dataset_dir),
            "resolution": "768,768",
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_train_epochs": epochs,
            "learning_rate": learning_rate,
            "lr_scheduler": "cosine_with_restarts",
            "lr_warmup_steps": 100,
            "network_module": "networks.lora",
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_train_unet_only": False,
            "network_train_text_encoder_only": False,
            "mixed_precision": "fp16",
            "save_precision": "fp16",
            "seed": 42,
            "cache_latents": True,
            "prior_loss_weight": 1.0,
            "max_token_length": 225,
            "caption_extension": ".txt",
            "keep_tokens": 1,
            "xformers": True,
            "min_snr_gamma": 5
        }

        config_file = Path(f"/tmp/lora_training_config_{output_name}.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"📝 Created training config: {config_file}")
        return config_file

    async def _execute_training(self, config_file: Path, output_name: str) -> Dict[str, Any]:
        """Execute kohya_ss training"""

        logger.info("🚀 Starting kohya_ss training process...")

        # Check for GPU availability
        gpu_check = subprocess.run(['nvidia-smi'], capture_output=True)
        if gpu_check.returncode != 0:
            raise Exception("No NVIDIA GPU detected. LORA training requires GPU.")

        # Construct training command
        train_script = self.kohya_path / "train_network.py"

        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")

        # Training command
        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "1",
            str(train_script),
            f"--config_file={config_file}"
        ]

        # Execute training
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.kohya_path)
        )

        # Monitor training output
        training_log = []
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            line_text = line.decode().strip()
            training_log.append(line_text)

            # Log progress
            if any(keyword in line_text.lower() for keyword in ['epoch', 'step', 'loss']):
                logger.info(f"Training: {line_text}")

        # Wait for completion
        await process.wait()

        # Check result
        output_path = self.output_base / f"{output_name}.safetensors"

        if process.returncode == 0 and output_path.exists():
            logger.info(f"✅ Training completed successfully!")
            return {
                'status': 'success',
                'output_name': output_name,
                'lora_path': str(output_path),
                'training_log': training_log[-20:]  # Last 20 lines
            }
        else:
            raise Exception(f"Training failed with return code {process.returncode}")

    async def _simulate_training(self, output_name: str, character_name: str) -> Dict[str, Any]:
        """Simulate training for development/testing"""

        logger.info("🎭 Simulating LORA training (development mode)...")

        # Simulate training delay
        await asyncio.sleep(5)

        # Create a placeholder file
        output_path = self.output_base / f"{output_name}.safetensors.placeholder"
        output_path.touch()

        logger.info(f"✅ Training simulation complete!")

        return {
            'status': 'simulated',
            'character_name': character_name,
            'output_name': output_name,
            'lora_path': str(output_path),
            'note': 'This was a simulated training. kohya_ss not found.'
        }


# Task handler function for integration with Echo task queue
async def handle_lora_training_task(task) -> Dict[str, Any]:
    """Handler function for LORA_TRAINING task type"""
    worker = LoraTrainingWorker()
    return await worker.train_lora_model(task.payload)
