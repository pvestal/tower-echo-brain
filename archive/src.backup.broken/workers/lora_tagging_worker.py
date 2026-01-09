#!/usr/bin/env python3
"""
LORA Tagging Worker
Creates .txt tag files for all generated training images
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import json

logger = logging.getLogger(__name__)

class LoraTaggingWorker:
    """Worker that creates caption/tag files for training images"""

    def __init__(self):
        pass

    async def tag_training_images(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create .txt tag files for all images in directory

        Args:
            task_payload: {
                'image_dir': str - Directory containing images,
                'character_tags': List[str] - Base character tags,
                'character_name': str,
                'description': str (optional)
            }

        Returns:
            Dict with tagging results
        """
        image_dir = Path(task_payload.get('image_dir'))
        character_tags = task_payload.get('character_tags', [])
        character_name = task_payload.get('character_name')
        description = task_payload.get('description', '')

        logger.info(f"🏷️ Starting LORA image tagging for {character_name}")
        logger.info(f"📁 Image directory: {image_dir}")

        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")

        # Find all PNG images
        image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))

        if not image_files:
            raise ValueError(f"No images found in directory: {image_dir}")

        logger.info(f"📸 Found {len(image_files)} images to tag")

        # Load generation metadata if available
        metadata_file = image_dir.parent / 'generation_metadata.json'
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        tagged_files = []
        failed_tags = []

        for image_file in image_files:
            try:
                # Extract image index from filename
                image_index = self._extract_image_index(image_file.name)

                # Get specific metadata for this image if available
                image_metadata = self._find_image_metadata(metadata, image_index)

                # Generate tags for this image
                tags = self._generate_tags(
                    character_name=character_name,
                    character_tags=character_tags,
                    description=description,
                    image_metadata=image_metadata
                )

                # Create .txt file with same name as image
                txt_file = image_file.with_suffix('.txt')
                with open(txt_file, 'w') as f:
                    f.write(tags)

                tagged_files.append({
                    'image': str(image_file),
                    'tags_file': str(txt_file),
                    'tags': tags
                })

                logger.info(f"✅ Tagged: {image_file.name}")

            except Exception as e:
                failed_tags.append({
                    'image': str(image_file),
                    'error': str(e)
                })
                logger.error(f"❌ Failed to tag {image_file.name}: {e}")

        result = {
            'character_name': character_name,
            'image_directory': str(image_dir),
            'total_images': len(image_files),
            'tagged_successfully': len(tagged_files),
            'failed_tags': len(failed_tags),
            'tagged_files': tagged_files,
            'failures': failed_tags
        }

        # Save tagging results
        tagging_results_file = image_dir.parent / 'tagging_results.json'
        with open(tagging_results_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"🎉 LORA image tagging complete: {len(tagged_files)}/{len(image_files)} successful")
        return result

    def _extract_image_index(self, filename: str) -> int:
        """Extract index number from filename like 'character_001.png'"""
        try:
            # Extract number from filename
            parts = filename.split('_')
            if len(parts) >= 2:
                index_str = parts[-1].split('.')[0]
                return int(index_str)
        except:
            pass
        return 0

    def _find_image_metadata(self, metadata: Dict, image_index: int) -> Dict:
        """Find specific metadata for an image by index"""
        if not metadata or 'generated_images' not in metadata:
            return {}

        for img in metadata.get('generated_images', []):
            if isinstance(img, dict) and img.get('index') == image_index:
                return img.get('metadata', {})

        return {}

    def _generate_tags(self, character_name: str, character_tags: List[str],
                       description: str, image_metadata: Dict) -> str:
        """
        Generate comma-separated tags for an image

        Format: character_name, gender, [features], [pose], [expression], quality_tags
        """
        tags = []

        # Add character name (main trigger word)
        tags.append(character_name)

        # Add base character tags
        tags.extend(character_tags)

        # Add description elements if provided
        if description:
            # Extract key descriptors from description
            tags.append(description.split(',')[0].strip())  # First descriptor

        # Add image-specific metadata if available
        if image_metadata:
            if 'pose' in image_metadata:
                tags.append(image_metadata['pose'])
            if 'expression' in image_metadata:
                tags.append(image_metadata['expression'])
            if 'angle' in image_metadata:
                tags.append(image_metadata['angle'])
            if 'lighting' in image_metadata:
                tags.append(image_metadata['lighting'])
            if 'background' in image_metadata:
                tags.append(image_metadata['background'])

        # Add quality tags
        quality_tags = ['high quality', 'detailed', 'best quality']
        tags.extend(quality_tags)

        # Join tags with commas
        return ', '.join(tags)


# Task handler function for integration with Echo task queue
async def handle_lora_tagging_task(task) -> Dict[str, Any]:
    """Handler function for LORA_TAGGING task type"""
    worker = LoraTaggingWorker()
    return await worker.tag_training_images(task.payload)
