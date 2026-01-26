#!/usr/bin/env python3
"""
Tower LTX LoRA Trainer - Proper diffusers implementation
Integrated with existing Tower anime production system
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
import subprocess

# Import diffusers components
from diffusers import LTXPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
from diffusers.optimization import get_scheduler
from transformers import T5EncoderModel, T5TokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
import safetensors.torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'anime_production',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE',
    'port': 5432
}

class VideoFrameDataset(Dataset):
    """Dataset for video frames with proper preprocessing"""

    def __init__(self, frames_dir: Path, caption_file: Path, resolution=(768, 512), max_frames=None):
        self.frames_dir = Path(frames_dir)
        self.resolution = resolution
        self.caption = caption_file.read_text().strip() if caption_file.exists() else ""

        # Get frame files
        frame_patterns = ["*.png", "*.jpg", "*.jpeg"]
        self.frame_files = []
        for pattern in frame_patterns:
            self.frame_files.extend(sorted(self.frames_dir.glob(pattern)))

        if max_frames:
            self.frame_files = self.frame_files[:max_frames]

        if not self.frame_files:
            raise ValueError(f"No frames found in {self.frames_dir}")

        logger.info(f"Dataset initialized with {len(self.frame_files)} frames")

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.frame_files[idx]).convert("RGB")
        image = image.resize(self.resolution, Image.LANCZOS)

        # Convert to tensor and normalize to [-1, 1]
        image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        return {
            "pixel_values": image_tensor,
            "caption": self.caption,
            "frame_path": str(self.frame_files[idx])
        }


class TowerLTXTrainer:
    """Main trainer integrated with Tower anime production"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_path = "Lightricks/LTXVideo"

        # Paths from Tower system
        self.output_dir = Path("/mnt/1TB-storage/models/loras")
        self.dataset_dir = Path("/opt/tower-lora-studio/datasets")
        self.logs_dir = Path("/opt/tower-lora-studio/logs")
        self.logs_dir.mkdir(exist_ok=True)

        # Training config
        self.resolution = (768, 512)  # LTX native resolution

        # Model components
        self.pipeline = None
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.noise_scheduler = None

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**DB_CONFIG)

    def load_models(self):
        """Load LTX models and components"""
        logger.info("Loading LTX models...")

        # Load full pipeline first to get components
        try:
            self.pipeline = LTXPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                variant="fp16"
            )

            # Extract components
            self.transformer = self.pipeline.transformer.to(self.device)
            self.vae = self.pipeline.vae.to(self.device)
            self.text_encoder = self.pipeline.text_encoder.to(self.device)
            self.tokenizer = self.pipeline.tokenizer
            self.noise_scheduler = self.pipeline.scheduler

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def setup_lora(self, rank: int = 32, alpha: int = 32):
        """Setup LoRA configuration for the transformer"""
        logger.info(f"Setting up LoRA with rank={rank}, alpha={alpha}")

        # Target modules for LTX transformer
        target_modules = [
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "norm1", "norm2", "ff.net.0", "ff.net.2"
        ]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.DIFFUSION
        )

        # Apply LoRA to transformer
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.train()

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.transformer.parameters())
        logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def prepare_dataset_from_videos(self, lora_name: str, video_sources: list, fps: int = 4):
        """Extract frames from video sources for training"""
        dataset_path = self.dataset_dir / lora_name
        frames_dir = dataset_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preparing dataset for {lora_name} from {len(video_sources)} videos")

        total_frames = 0
        for i, video_path in enumerate(video_sources):
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue

            # Extract frames with ffmpeg
            output_pattern = str(frames_dir / f"video_{i:02d}_frame_%04d.png")
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"fps={fps},scale={self.resolution[0]}:{self.resolution[1]}:force_original_aspect_ratio=1,pad={self.resolution[0]}:{self.resolution[1]}:(ow-iw)/2:(oh-ih)/2",
                "-q:v", "2",
                output_pattern
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    new_frames = len(list(frames_dir.glob(f"video_{i:02d}_*.png")))
                    total_frames += new_frames
                    logger.info(f"Extracted {new_frames} frames from {Path(video_path).name}")
                else:
                    logger.error(f"FFmpeg failed: {result.stderr}")
            except Exception as e:
                logger.error(f"Frame extraction failed: {e}")

        logger.info(f"Total frames prepared: {total_frames}")
        return dataset_path, total_frames

    def train_lora(self,
                   lora_name: str,
                   trigger_word: str,
                   base_prompt: str,
                   dataset_path: Path,
                   num_epochs: int = 100,
                   batch_size: int = 1,
                   learning_rate: float = 1e-5,
                   save_every: int = 20):
        """Train the LoRA with proper diffusers implementation"""

        logger.info(f"Starting training for {lora_name}")

        # Setup models if not loaded
        if self.transformer is None:
            self.load_models()
            self.setup_lora()

        # Prepare caption
        caption_file = dataset_path / "caption.txt"
        full_caption = f"{trigger_word}, {base_prompt}"
        caption_file.write_text(full_caption)

        # Create dataset
        frames_dir = dataset_path / "frames"
        dataset = VideoFrameDataset(frames_dir, caption_file, self.resolution)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=len(dataloader) * num_epochs
        )

        # Training loop
        self.transformer.train()
        global_step = 0
        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in progress_bar:
                # Move to device
                pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float16)
                captions = batch["caption"]

                # Encode text
                with torch.no_grad():
                    text_inputs = self.tokenizer(
                        captions,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)

                    encoder_hidden_states = self.text_encoder(
                        text_inputs.input_ids,
                        attention_mask=text_inputs.attention_mask
                    )[0]

                # Encode to latents
                with torch.no_grad():
                    latents = self.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device
                ).long()

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                model_pred = self.transformer(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )[0]

                # Calculate loss
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()

                # Update metrics
                epoch_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })

                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint_path = self.output_dir / f"{lora_name}_step_{global_step}.safetensors"
                    self.save_lora_weights(checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

            # Save if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = self.output_dir / f"{lora_name}_best.safetensors"
                self.save_lora_weights(best_path)

        # Final save
        final_path = self.output_dir / f"{lora_name}.safetensors"
        self.save_lora_weights(final_path)

        logger.info(f"Training complete! Final model: {final_path}")
        return final_path

    def save_lora_weights(self, output_path: Path):
        """Save LoRA weights to safetensors"""
        lora_state_dict = {}

        # Extract LoRA weights
        for name, param in self.transformer.named_parameters():
            if "lora" in name and param.requires_grad:
                lora_state_dict[name] = param.cpu().detach()

        # Save with metadata
        metadata = {
            "format": "pt",
            "lora_rank": str(self.transformer.peft_config['default'].r),
            "lora_alpha": str(self.transformer.peft_config['default'].lora_alpha),
            "target_modules": str(self.transformer.peft_config['default'].target_modules),
            "created_at": datetime.now().isoformat()
        }

        safetensors.torch.save_file(lora_state_dict, output_path, metadata=metadata)
        logger.info(f"Saved LoRA weights: {output_path} ({len(lora_state_dict)} tensors)")

    async def process_training_queue(self):
        """Process items from the training queue"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get next pending job
            cursor.execute("""
                SELECT q.id, q.definition_id, d.name, d.trigger_word, d.base_prompt, d.is_nsfw
                FROM training_queue q
                JOIN lora_definitions d ON q.definition_id = d.id
                WHERE q.status = 'pending'
                ORDER BY q.priority DESC, q.scheduled_at
                LIMIT 1
            """)

            job = cursor.fetchone()
            if not job:
                logger.info("No pending training jobs")
                return None

            logger.info(f"Processing training job: {job['name']}")

            # Update status
            cursor.execute("""
                UPDATE training_queue
                SET status = 'processing', started_at = NOW()
                WHERE id = %s
            """, (job['id'],))

            cursor.execute("""
                INSERT INTO lora_training_status (definition_id, status, training_started_at)
                VALUES (%s, 'training', NOW())
                ON CONFLICT (definition_id) DO UPDATE
                SET status = 'training', training_started_at = NOW(), updated_at = NOW()
            """, (job['definition_id'],))

            conn.commit()

            # Get video sources for dataset
            video_sources = self.get_video_sources_for_lora(job['name'], job['is_nsfw'])

            if not video_sources:
                logger.warning(f"No video sources found for {job['name']}")
                video_sources = ["/mnt/1TB-storage/ComfyUI/output/custom_lora_test_00001.mp4"]  # Fallback

            try:
                # Prepare dataset
                dataset_path, frame_count = self.prepare_dataset_from_videos(
                    job['name'], video_sources
                )

                if frame_count < 10:
                    raise ValueError(f"Insufficient frames ({frame_count}) for training")

                # Train the LoRA
                model_path = self.train_lora(
                    lora_name=job['name'],
                    trigger_word=job['trigger_word'],
                    base_prompt=job['base_prompt'],
                    dataset_path=dataset_path,
                    num_epochs=50 if job['is_nsfw'] else 30,
                    learning_rate=1e-5
                )

                # Update success status
                cursor.execute("""
                    UPDATE lora_training_status
                    SET status = 'completed',
                        model_path = %s,
                        training_completed_at = NOW(),
                        updated_at = NOW()
                    WHERE definition_id = %s
                """, (str(model_path), job['definition_id']))

                cursor.execute("""
                    UPDATE training_queue
                    SET status = 'completed', completed_at = NOW()
                    WHERE id = %s
                """, (job['id'],))

                conn.commit()

                logger.info(f"✅ Training completed: {job['name']}")
                return {"success": True, "model_path": str(model_path), "job": job}

            except Exception as e:
                logger.error(f"Training failed: {e}")

                # Update failure status
                cursor.execute("""
                    UPDATE lora_training_status
                    SET status = 'failed',
                        error_message = %s,
                        updated_at = NOW()
                    WHERE definition_id = %s
                """, (str(e), job['definition_id']))

                cursor.execute("""
                    UPDATE training_queue
                    SET status = 'failed', error_message = %s
                    WHERE id = %s
                """, (str(e), job['id']))

                conn.commit()
                return {"success": False, "error": str(e), "job": job}

        finally:
            conn.close()

    def get_video_sources_for_lora(self, lora_name: str, is_nsfw: bool) -> list:
        """Get relevant video sources for training dataset"""
        # This would connect to existing video database
        # For now, return some examples based on type

        base_videos = [
            "/mnt/1TB-storage/ComfyUI/output/anime_Mei_1769446107_00001.mp4",
            "/mnt/1TB-storage/ComfyUI/output/anime_Mei_1769445185_00001.mp4"
        ]

        if is_nsfw:
            # Add NSFW video sources
            base_videos.extend([
                "/mnt/1TB-storage/ComfyUI/output/custom_lora_test_00001.mp4"
            ])

        return base_videos


async def main():
    """Main training worker"""
    trainer = TowerLTXTrainer()

    logger.info("🎬 Tower LTX LoRA Trainer Starting...")

    # Process one job
    result = await trainer.process_training_queue()

    if result:
        if result["success"]:
            logger.info(f"✅ Training successful: {result['model_path']}")
        else:
            logger.error(f"❌ Training failed: {result['error']}")
    else:
        logger.info("No training jobs to process")


if __name__ == "__main__":
    asyncio.run(main())