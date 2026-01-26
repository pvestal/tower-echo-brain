#!/usr/bin/env python3
"""
Quick test to create a simple LTX LoRA without full training
Just to validate the pipeline works
"""

import torch
import safetensors.torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_minimal_ltx_lora(output_path: str, rank: int = 4):
    """Create a minimal LTX LoRA for testing"""

    # LTX LoRA structure based on existing models
    lora_weights = {}

    # Create minimal weights for transformer blocks (matching LTX structure)
    for block_idx in range(28):  # LTX has 28 transformer blocks
        for attn_type in ["attn1", "attn2"]:
            for proj in ["to_k", "to_q", "to_v", "to_out.0"]:
                # Create LoRA A and B matrices
                if "to_out" in proj:
                    # Output projection
                    lora_weights[f"diffusion_model.transformer_blocks.{block_idx}.{attn_type}.{proj}.lora_A.weight"] = torch.randn(rank, 3072) * 0.01
                    lora_weights[f"diffusion_model.transformer_blocks.{block_idx}.{attn_type}.{proj}.lora_B.weight"] = torch.randn(3072, rank) * 0.01
                else:
                    # K, Q, V projections
                    lora_weights[f"diffusion_model.transformer_blocks.{block_idx}.{attn_type}.{proj}.lora_A.weight"] = torch.randn(rank, 3072) * 0.01
                    lora_weights[f"diffusion_model.transformer_blocks.{block_idx}.{attn_type}.{proj}.lora_B.weight"] = torch.randn(3072, rank) * 0.01

    # Add some final layer weights
    lora_weights["diffusion_model.proj_out.lora_A.weight"] = torch.randn(rank, 3072) * 0.01
    lora_weights["diffusion_model.proj_out.lora_B.weight"] = torch.randn(512, rank) * 0.01

    # Save the LoRA
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_file(lora_weights, output_file)
    logger.info(f"Created minimal LTX LoRA at: {output_file}")
    logger.info(f"Total tensors: {len(lora_weights)}")

    # Calculate size
    total_params = sum(w.numel() for w in lora_weights.values())
    size_mb = (total_params * 4) / (1024 * 1024)  # float32 size
    logger.info(f"Approximate size: {size_mb:.2f} MB")

    return output_file

if __name__ == "__main__":
    # Create a test LoRA
    test_lora_path = "/mnt/1TB-storage/models/loras/test_cowgirl_ltx_minimal.safetensors"
    create_minimal_ltx_lora(test_lora_path, rank=4)

    print(f"\nTest LoRA created at: {test_lora_path}")
    print("You can now test this with your LTX pipeline!")