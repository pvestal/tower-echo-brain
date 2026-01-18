"""
LoRA Training Pipeline
Enables training of LoRA adapters for image generation models
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

try:
    import torch
    from PIL import Image
    import numpy as np
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model, TaskType
    import accelerate
    from torch.utils.data import Dataset, DataLoader
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not available - LoRA training features disabled")
    # Create dummy Dataset class
    class Dataset:
        pass

class ImageCaptionDataset(Dataset):
    """Dataset for image-caption pairs"""

    def __init__(self, image_paths: List[str], captions: List[str], size: int = 512):
        self.image_paths = image_paths
        self.captions = captions
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image.resize((self.size, self.size), Image.LANCZOS)
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return {
            "image": image,
            "caption": self.captions[idx]
        }


class LoRATrainer:
    """Trains LoRA adapters for Stable Diffusion models"""

    def __init__(
        self,
        model_path: str = "/mnt/1TB-storage/models/checkpoints/",
        output_path: str = "/mnt/1TB-storage/models/loras/"
    ):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        if ML_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        self.training_history = []

    def prepare_training_data(
        self,
        image_folder: str,
        metadata_file: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Prepare training data from folder

        Args:
            image_folder: Folder containing training images
            metadata_file: Optional JSON file with captions

        Returns:
            Lists of image paths and captions
        """

        image_folder = Path(image_folder)
        image_paths = []
        captions = []

        # Load metadata if provided
        metadata = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        # Collect images and captions
        for img_path in image_folder.glob("*.{png,jpg,jpeg}"):
            image_paths.append(str(img_path))

            # Get caption from metadata or filename
            if str(img_path.name) in metadata:
                caption = metadata[str(img_path.name)]
            else:
                # Generate caption from filename
                caption = img_path.stem.replace("_", " ").replace("-", " ")

            captions.append(caption)

        logger.info(f"Prepared {len(image_paths)} images for training")
        return image_paths, captions

    async def train_character_lora(
        self,
        character_name: str,
        image_folder: str,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a LoRA for a specific character

        Args:
            character_name: Name of the character
            image_folder: Folder containing character images
            base_model: Base model to train on
            training_config: Training configuration

        Returns:
            Training results
        """

        if not ML_AVAILABLE:
            return {
                "success": False,
                "error": "ML libraries not available - install torch, transformers, diffusers, and peft"
            }

        try:
            # Default training config
            config = {
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "lora_rank": 4,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "resolution": 512
            }

            if training_config:
                config.update(training_config)

            # Prepare data
            image_paths, captions = self.prepare_training_data(image_folder)

            if not image_paths:
                return {
                    "success": False,
                    "error": "No training images found"
                }

            # Create dataset
            dataset = ImageCaptionDataset(
                image_paths,
                captions,
                size=config["resolution"]
            )

            dataloader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=True
            )

            # Load base model components
            logger.info(f"Loading base model: {base_model}")

            # Load tokenizer and text encoder
            tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")

            # Load VAE and UNet
            vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")

            # Move to device
            vae.to(self.device)
            text_encoder.to(self.device)
            unet.to(self.device)

            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.DIFFUSION,
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]
            )

            # Apply LoRA to UNet
            unet = get_peft_model(unet, lora_config)
            unet.train()

            # Setup optimizer
            optimizer = torch.optim.AdamW(
                unet.parameters(),
                lr=config["learning_rate"]
            )

            # Training loop
            logger.info(f"Starting training for {character_name}")
            training_losses = []

            for epoch in range(config["num_epochs"]):
                epoch_loss = 0
                for batch_idx, batch in enumerate(dataloader):
                    # Encode images
                    images = batch["image"].to(self.device)
                    with torch.no_grad():
                        latents = vae.encode(images).latent_dist.sample()
                        latents = latents * 0.18215

                    # Encode text
                    captions = batch["caption"]
                    text_inputs = tokenizer(
                        captions,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )

                    with torch.no_grad():
                        text_embeddings = text_encoder(
                            text_inputs.input_ids.to(self.device)
                        ).last_hidden_state

                    # Add noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, 1000, (latents.shape[0],),
                        device=self.device
                    ).long()

                    # Forward pass
                    noise_pred = unet(
                        latents + noise,
                        timesteps,
                        encoder_hidden_states=text_embeddings
                    ).sample

                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)

                    # Backward pass
                    loss.backward()

                    if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(dataloader)
                training_losses.append(avg_loss)
                logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")

            # Save LoRA weights
            lora_name = f"{character_name}_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_file = self.output_path / f"{lora_name}.safetensors"

            # Save the LoRA weights
            unet.save_pretrained(str(output_file.parent / lora_name))

            # Record training
            training_record = {
                "lora_name": lora_name,
                "character": character_name,
                "base_model": base_model,
                "training_images": len(image_paths),
                "epochs": config["num_epochs"],
                "final_loss": training_losses[-1],
                "output_path": str(output_file),
                "timestamp": datetime.now().isoformat(),
                "config": config
            }

            self.training_history.append(training_record)

            # Save to database
            await self._save_to_database(training_record)

            return {
                "success": True,
                "lora_name": lora_name,
                "output_path": str(output_file),
                "final_loss": training_losses[-1],
                "training_history": training_losses
            }

        except Exception as e:
            logger.error(f"Failed to train LoRA: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def train_style_lora(
        self,
        style_name: str,
        image_folder: str,
        style_description: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a LoRA for a specific art style

        Args:
            style_name: Name of the style
            image_folder: Folder containing style reference images
            style_description: Description of the style
            **kwargs: Additional training configuration

        Returns:
            Training results
        """

        # Modify captions to include style description
        image_paths, base_captions = self.prepare_training_data(image_folder)

        if style_description:
            # Prepend style description to all captions
            captions = [f"{style_description}, {cap}" for cap in base_captions]
        else:
            captions = base_captions

        # Create temporary metadata
        metadata_file = Path("/tmp") / f"{style_name}_metadata.json"
        metadata = {Path(img).name: cap for img, cap in zip(image_paths, captions)}
        metadata_file.write_text(json.dumps(metadata))

        # Train as character LoRA with style-specific settings
        return await self.train_character_lora(
            character_name=style_name,
            image_folder=image_folder,
            training_config=kwargs.get("training_config", {
                "learning_rate": 5e-5,  # Lower LR for styles
                "num_epochs": 15,  # More epochs for style learning
                "lora_rank": 8  # Higher rank for complex styles
            })
        )

    async def fine_tune_existing_lora(
        self,
        lora_path: str,
        additional_images: str,
        fine_tune_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune an existing LoRA with additional data

        Args:
            lora_path: Path to existing LoRA
            additional_images: Folder with additional training images
            fine_tune_config: Fine-tuning configuration

        Returns:
            Fine-tuning results
        """

        try:
            # Load existing LoRA
            lora_path = Path(lora_path)
            if not lora_path.exists():
                return {
                    "success": False,
                    "error": "LoRA file not found"
                }

            # TODO: Load existing LoRA weights and continue training
            # This would require loading the saved LoRA state

            return {
                "success": False,
                "error": "Fine-tuning not yet implemented"
            }

        except Exception as e:
            logger.error(f"Failed to fine-tune LoRA: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def evaluate_lora(
        self,
        lora_path: str,
        test_prompts: List[str],
        base_model: str = "runwayml/stable-diffusion-v1-5"
    ) -> Dict[str, Any]:
        """
        Evaluate a trained LoRA

        Args:
            lora_path: Path to LoRA file
            test_prompts: Prompts to test
            base_model: Base model to use

        Returns:
            Evaluation results
        """

        try:
            # Load pipeline with LoRA
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float16
            ).to(self.device)

            # TODO: Load LoRA weights into pipeline
            # pipe.load_lora_weights(lora_path)

            results = []
            for prompt in test_prompts:
                # Generate image
                with torch.no_grad():
                    image = pipe(prompt, num_inference_steps=20).images[0]

                # Save image
                output_path = Path("/tmp") / f"lora_test_{hashlib.sha256(prompt.encode()).hexdigest()[:8]}.png"
                image.save(output_path)

                results.append({
                    "prompt": prompt,
                    "image_path": str(output_path)
                })

            return {
                "success": True,
                "results": results
            }

        except Exception as e:
            logger.error(f"Failed to evaluate LoRA: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _save_to_database(self, training_record: Dict[str, Any]):
        """Save training record to database"""

        try:
            import asyncpg

            # Connect to database
            conn = await asyncpg.connect(
                host='localhost',
                database='tower_consolidated',
                user='patrick',
                password='tower_echo_brain_secret_key_2025'
            )

            # Create table if not exists
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS lora_training_history (
                    id SERIAL PRIMARY KEY,
                    lora_name VARCHAR(255) UNIQUE,
                    character_name VARCHAR(255),
                    base_model VARCHAR(255),
                    training_images INTEGER,
                    epochs INTEGER,
                    final_loss FLOAT,
                    output_path TEXT,
                    config JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')

            # Insert record
            await conn.execute('''
                INSERT INTO lora_training_history (
                    lora_name, character_name, base_model,
                    training_images, epochs, final_loss,
                    output_path, config
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (lora_name) DO UPDATE SET
                    final_loss = $6,
                    config = $8
            ''', training_record['lora_name'], training_record['character'],
                training_record['base_model'], training_record['training_images'],
                training_record['epochs'], training_record['final_loss'],
                training_record['output_path'], json.dumps(training_record['config']))

            await conn.close()

        except Exception as e:
            logger.error(f"Failed to save to database: {e}")

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history