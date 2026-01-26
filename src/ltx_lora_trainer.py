#!/usr/bin/env python3
"""
LTX LoRA Training Pipeline using Diffusers
Proper implementation for training LTX LoRAs
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from diffusers.models.autoencoders import AutoencoderKLLTXVideo
from transformers import T5EncoderModel, T5TokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
import safetensors.torch
from pathlib import Path
import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from PIL import Image
import numpy as np
from tqdm import tqdm
import subprocess
from datetime import datetime
import asyncio

logging.basicConfig(level=logging.INFO)
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
    """Dataset for video frames"""

    def __init__(self, frames_dir: Path, caption_text: str, resolution=(512, 768)):
        self.frames_dir = Path(frames_dir)
        self.caption = caption_text
        self.resolution = resolution

        # Get all frame files
        self.frame_files = sorted(self.frames_dir.glob("*.png"))
        if not self.frame_files:
            self.frame_files = sorted(self.frames_dir.glob("*.jpg"))

        logger.info(f"Found {len(self.frame_files)} frames")

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.frame_files[idx]).convert("RGB")
        image = image.resize(self.resolution, Image.LANCZOS)

        # Convert to tensor
        image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        return {
            "pixel_values": image_tensor,
            "caption": self.caption
        }


class LTXLoRATrainer:
    """Main trainer class for LTX LoRAs"""

    def __init__(self, model_path: str = "Lightricks/LTXVideo", device: str = "cuda"):
        self.device = device
        self.model_path = model_path

        # Paths
        self.output_dir = Path("/mnt/1TB-storage/models/loras")
        self.dataset_dir = Path("/opt/tower-lora-studio/datasets")

        logger.info("Initializing LTX LoRA trainer...")

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**DB_CONFIG)

    def load_models(self):
        """Load LTX models"""
        logger.info("Loading LTX models...")

        # Load pipeline
        self.pipeline = LTXPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
        ).to(self.device)

        # Get individual components
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer

        logger.info("Models loaded successfully")

    def setup_lora(self, rank: int = 16, alpha: int = 16):
        """Setup LoRA configuration"""
        logger.info(f"Setting up LoRA with rank={rank}, alpha={alpha}")

        # Configure LoRA for transformer
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.DIFFUSION
        )

        # Apply LoRA to transformer
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.train()

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.transformer.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def prepare_dataset(self, lora_name: str, video_sources: list = None):
        """Prepare training dataset from videos"""
        dataset_path = self.dataset_dir / lora_name

        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)

            if video_sources:
                logger.info(f"Extracting frames from {len(video_sources)} videos...")

                for video in video_sources:
                    if not Path(video).exists():
                        continue

                    # Extract frames
                    cmd = [
                        "ffmpeg", "-i", str(video),
                        "-vf", "fps=4",  # 4 fps for training
                        "-q:v", "2",
                        str(dataset_path / f"frame_%04d.png")
                    ]
                    subprocess.run(cmd, capture_output=True)

        return dataset_path

    def train(self,
             lora_name: str,
             trigger_word: str,
             base_prompt: str,
             dataset_path: Path,
             num_epochs: int = 10,
             batch_size: int = 1,
             learning_rate: float = 1e-4):
        """Train the LoRA"""

        logger.info(f"Starting training for {lora_name}")

        # Load models if not loaded
        if not hasattr(self, 'transformer'):
            self.load_models()
            self.setup_lora()

        # Create dataset
        dataset = VideoFrameDataset(
            dataset_path,
            f"{trigger_word}, {base_prompt}"
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=learning_rate
        )

        # Training loop
        self.transformer.train()
        progress_bar = tqdm(total=num_epochs * len(dataloader))

        for epoch in range(num_epochs):
            epoch_loss = 0

            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                pixel_values = batch["pixel_values"].to(self.device)
                captions = batch["caption"]

                # Encode text
                text_inputs = self.tokenizer(
                    captions,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(
                        text_inputs.input_ids,
                        attention_mask=text_inputs.attention_mask
                    )[0]

                # Encode images to latents
                with torch.no_grad():
                    latents = self.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 1000, (latents.shape[0],),
                    device=self.device
                ).long()

                noisy_latents = self.pipeline.scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Predict noise
                model_pred = self.transformer(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # Calculate loss
                loss = F.mse_loss(model_pred, noise)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        progress_bar.close()

        # Save LoRA weights
        output_path = self.output_dir / f"{lora_name}.safetensors"
        self.save_lora_weights(output_path)

        logger.info(f"Training complete! Model saved to {output_path}")
        return output_path

    def save_lora_weights(self, output_path: Path):
        """Save only the LoRA weights"""
        lora_state_dict = {}

        for name, param in self.transformer.named_parameters():
            if "lora" in name:
                lora_state_dict[name] = param.cpu()

        # Save using safetensors
        safetensors.torch.save_file(lora_state_dict, output_path)
        logger.info(f"Saved LoRA weights to {output_path}")

    async def train_from_queue(self):
        """Process training queue"""
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get next item from queue
        cursor.execute("""
            SELECT q.id, q.definition_id, d.name, d.trigger_word, d.base_prompt
            FROM training_queue q
            JOIN lora_definitions d ON q.definition_id = d.id
            WHERE q.status = 'pending'
            ORDER BY q.priority DESC, q.scheduled_at
            LIMIT 1
        """)

        job = cursor.fetchone()
        if not job:
            logger.info("No pending jobs in queue")
            return

        logger.info(f"Processing job: {job['name']}")

        # Update status to processing
        cursor.execute("""
            UPDATE training_queue
            SET status = 'processing', started_at = NOW()
            WHERE id = %s
        """, (job['id'],))

        # Create/update training status
        cursor.execute("""
            INSERT INTO lora_training_status (definition_id, status, training_started_at)
            VALUES (%s, 'training', NOW())
            ON CONFLICT (definition_id) DO UPDATE
            SET status = 'training', training_started_at = NOW()
        """, (job['definition_id'],))

        conn.commit()

        try:
            # Prepare dataset
            dataset_path = self.prepare_dataset(job['name'])

            # Train
            model_path = self.train(
                lora_name=job['name'],
                trigger_word=job['trigger_word'],
                base_prompt=job['base_prompt'],
                dataset_path=dataset_path,
                num_epochs=5  # Quick training for testing
            )

            # Update status to completed
            cursor.execute("""
                UPDATE lora_training_status
                SET status = 'completed',
                    model_path = %s,
                    training_completed_at = NOW()
                WHERE definition_id = %s
            """, (str(model_path), job['definition_id']))

            cursor.execute("""
                UPDATE training_queue
                SET status = 'completed', completed_at = NOW()
                WHERE id = %s
            """, (job['id'],))

            conn.commit()
            logger.info(f"✅ Training completed for {job['name']}")

        except Exception as e:
            logger.error(f"Training failed: {e}")

            # Update status to failed
            cursor.execute("""
                UPDATE lora_training_status
                SET status = 'failed', error_message = %s
                WHERE definition_id = %s
            """, (str(e), job['definition_id']))

            cursor.execute("""
                UPDATE training_queue
                SET status = 'failed', error_message = %s
                WHERE id = %s
            """, (str(e), job['id'],))

            conn.commit()

        finally:
            conn.close()


async def main():
    """Main entry point"""
    trainer = LTXLoRATrainer()

    # Process one job from queue
    await trainer.train_from_queue()


if __name__ == "__main__":
    asyncio.run(main())