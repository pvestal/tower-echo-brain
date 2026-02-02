#!/usr/bin/env python3
"""
Download BETTER models from CivitAI and Hugging Face
Models chosen based on:
- Multi-character capability
- Asian face accuracy
- High ratings and download counts
"""
import requests
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = "/mnt/1TB-storage/ComfyUI/models/checkpoints"

# Top-rated models we SHOULD have
RECOMMENDED_MODELS = {
    # BEST for Asian faces and photorealistic
    "majicMIX_realistic_v7": {
        "url": "https://civitai.com/api/download/models/176425",  # MajicMix Realistic v7
        "filename": "majicmixRealistic_v7.safetensors",
        "description": "Top Asian face model, photorealistic, NSFW capable",
        "size_gb": 2.0
    },
    "ghostMix_v2": {
        "url": "https://civitai.com/api/download/models/76907",  # GhostMix v2.0
        "filename": "ghostmix_v20.safetensors",
        "description": "Best 2.5D model, high compatibility",
        "size_gb": 2.0
    },
    "epicRealism_v5": {
        "url": "https://civitai.com/api/download/models/134065",  # epiCRealism v5
        "filename": "epicrealism_naturalSin.safetensors",
        "description": "Top photorealistic model, great for multiple people",
        "size_gb": 2.0
    },

    # SDXL models for better multi-character
    "juggernautXL_v9": {
        "url": "https://civitai.com/api/download/models/456194",  # Juggernaut XL v9
        "filename": "juggernautXL_v9.safetensors",
        "description": "SDXL model, excellent for multiple characters",
        "size_gb": 6.5
    },

    # Anime models
    "meinaMix_v11": {
        "url": "https://civitai.com/api/download/models/119057",  # MeinaMix v11
        "filename": "meinamix_meinaV11.safetensors",
        "description": "Top anime model, better than counterfeit",
        "size_gb": 2.0
    }
}

# Models to DELETE (they suck)
MODELS_TO_DELETE = [
    "deliberate_v2.safetensors",  # 0 bytes, broken
    "realisticVision_v51.safetensors",  # Old and bad at multi-character
    "Counterfeit-V2.5.safetensors",  # Old version
]

def download_model(model_info, model_name):
    """Download a model from CivitAI"""

    filepath = os.path.join(MODELS_DIR, model_info["filename"])

    # Check if already exists
    if os.path.exists(filepath):
        logger.info(f"✓ {model_name} already exists")
        return True

    logger.info(f"📥 Downloading {model_name} ({model_info['size_gb']}GB)...")
    logger.info(f"   {model_info['description']}")

    try:
        # Download with streaming
        response = requests.get(model_info["url"], stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if int(percent) % 10 == 0:
                            logger.info(f"   Progress: {percent:.0f}%")

        logger.info(f"✅ Downloaded {model_name}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def cleanup_bad_models():
    """Delete or archive bad models"""

    archive_dir = os.path.join(MODELS_DIR, "archived_bad")
    os.makedirs(archive_dir, exist_ok=True)

    for model_file in MODELS_TO_DELETE:
        filepath = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(filepath):
            archive_path = os.path.join(archive_dir, model_file)
            os.rename(filepath, archive_path)
            logger.info(f"🗑️  Archived bad model: {model_file}")

def main():
    logger.info("=" * 60)
    logger.info("BETTER MODEL DOWNLOADER")
    logger.info("=" * 60)

    # First, clean up bad models
    logger.info("\n🧹 Cleaning up bad models...")
    cleanup_bad_models()

    # Show what we're getting
    logger.info("\n📋 Models to download/verify:")
    for name, info in RECOMMENDED_MODELS.items():
        logger.info(f"  • {name}: {info['description']}")

    logger.info("\n⚠️  This will download ~15GB of models")
    logger.info("Continue? (y/n)")

    if input().lower() != 'y':
        logger.info("Cancelled")
        return

    # Download each model
    logger.info("\n📥 Starting downloads...")
    for model_name, model_info in RECOMMENDED_MODELS.items():
        download_model(model_info, model_name)

    logger.info("\n✅ DONE!")
    logger.info("New models available in ComfyUI:")
    for name, info in RECOMMENDED_MODELS.items():
        logger.info(f"  • {info['filename']}")

if __name__ == "__main__":
    main()