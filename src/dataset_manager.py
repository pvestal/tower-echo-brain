#!/usr/bin/env python3
"""
Dataset Manager - Collect and prepare training data from Tower sources
Integrates with existing anime production videos and scene database
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
import subprocess
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'anime_production',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE',
    'port': 5432
}

class DatasetManager:
    """Manage training datasets from Tower anime production videos"""

    def __init__(self):
        self.output_base = Path("/mnt/1TB-storage/ComfyUI/output")
        self.datasets_base = Path("/opt/tower-lora-studio/datasets")

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**DB_CONFIG)

    def collect_videos_for_concept(self, concept_type: str, concept_name: str, limit: int = 50):
        """Collect videos from Tower productions matching concept"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Map concept to search terms
            search_terms = self.get_search_terms_for_concept(concept_type, concept_name)

            # Find videos from scene generations
            videos = []

            # Search in scene_generations table
            cursor.execute("""
                SELECT DISTINCT prompt_id, generated_at, scene_type
                FROM scene_generations
                WHERE status = 'completed'
                AND (scene_type ILIKE ANY(%s) OR
                     EXISTS (SELECT 1 FROM unnest(%s) term
                            WHERE scene_type ILIKE '%%' || term || '%%'))
                ORDER BY generated_at DESC
                LIMIT %s
            """, (search_terms, search_terms, limit))

            scene_results = cursor.fetchall()

            # Convert to video paths
            for scene in scene_results:
                # Find corresponding video files
                video_pattern = f"anime_*{scene['prompt_id']}*.mp4"
                potential_videos = list(self.output_base.glob(video_pattern))

                for video_path in potential_videos:
                    if video_path.stat().st_size > 10000:  # At least 10KB
                        videos.append({
                            "path": str(video_path),
                            "scene_type": scene['scene_type'],
                            "generated_at": scene['generated_at'],
                            "prompt_id": scene['prompt_id']
                        })

            logger.info(f"Found {len(videos)} videos for {concept_name}")
            return videos

        finally:
            conn.close()

    def get_search_terms_for_concept(self, concept_type: str, concept_name: str):
        """Get search terms based on concept type and name"""

        concept_mappings = {
            # Poses
            "cowgirl_position": ["intimate", "nsfw", "transformation"],
            "missionary_position": ["intimate", "nsfw"],
            "doggy_style": ["intimate", "nsfw"],
            "standing_sex": ["intimate", "nsfw"],
            "spooning": ["intimate", "nsfw"],

            # Violence
            "sword_slash": ["action", "fight", "combat"],
            "blood_splatter": ["action", "fight", "violence"],
            "decapitation": ["action", "fight", "violence"],
            "dismemberment": ["action", "fight", "violence"],
            "gore_explosion": ["action", "fight", "violence"],

            # Actions
            "kissing": ["intimate", "romance", "nsfw"],
            "masturbation": ["nsfw"],
            "orgasm": ["nsfw", "intimate"],

            # Styles
            "cyberpunk": ["transformation", "action"],
            "anime_style": ["dialogue", "action", "intimate"],
            "realistic": ["dialogue", "intimate"],

            # Scenes
            "bedroom": ["intimate", "nsfw"],
            "bathroom": ["nsfw"],
            "outdoors": ["action", "dialogue"],

            # Weapons
            "katana": ["action", "fight"],
            "energy_blade": ["action", "fight", "transformation"],
            "chainsaw": ["action", "fight", "violence"]
        }

        return concept_mappings.get(concept_name, ["dialogue", "action"])

    def prepare_training_dataset(self,
                                concept_name: str,
                                video_sources: list,
                                fps: int = 4,
                                max_frames_per_video: int = 50):
        """Prepare training dataset from collected videos"""

        dataset_path = self.datasets_base / concept_name
        frames_dir = dataset_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "concept_name": concept_name,
            "created_at": datetime.now().isoformat(),
            "video_sources": [],
            "total_frames": 0,
            "fps": fps
        }

        total_frames = 0

        for i, video_info in enumerate(video_sources):
            video_path = video_info["path"]
            if not Path(video_path).exists():
                continue

            # Extract frames with quality settings
            output_pattern = str(frames_dir / f"src{i:03d}_frame_%04d.png")

            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps={fps},scale=768:512:force_original_aspect_ratio=1,pad=768:512:(ow-iw)/2:(oh-ih)/2,eq=contrast=1.1:brightness=0.05:saturation=1.2",
                "-q:v", "1",  # High quality
                "-frames:v", str(max_frames_per_video),
                output_pattern
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    # Count extracted frames
                    new_frames = len(list(frames_dir.glob(f"src{i:03d}_*.png")))
                    total_frames += new_frames

                    metadata["video_sources"].append({
                        "original_path": video_path,
                        "scene_type": video_info.get("scene_type", "unknown"),
                        "frames_extracted": new_frames,
                        "prompt_id": video_info.get("prompt_id")
                    })

                    logger.info(f"Extracted {new_frames} frames from {Path(video_path).name}")

                else:
                    logger.error(f"FFmpeg failed for {video_path}: {result.stderr}")

            except subprocess.TimeoutExpired:
                logger.error(f"FFmpeg timeout for {video_path}")
            except Exception as e:
                logger.error(f"Frame extraction failed: {e}")

        metadata["total_frames"] = total_frames

        # Save metadata
        metadata_file = dataset_path / "dataset_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        logger.info(f"Dataset prepared: {concept_name} with {total_frames} frames from {len(video_sources)} videos")
        return dataset_path, total_frames

    def enhance_dataset_quality(self, dataset_path: Path):
        """Post-process frames for better quality"""
        frames_dir = dataset_path / "frames"

        if not frames_dir.exists():
            return

        # Apply quality enhancements using ImageMagick
        frames = list(frames_dir.glob("*.png"))
        logger.info(f"Enhancing {len(frames)} frames...")

        for frame_path in frames:
            try:
                # Apply sharpening and contrast enhancement
                cmd = [
                    "convert", str(frame_path),
                    "-unsharp", "0x1+1.0+0.05",  # Gentle sharpening
                    "-modulate", "100,110,100",   # Increase saturation
                    str(frame_path)  # Overwrite
                ]

                subprocess.run(cmd, capture_output=True, timeout=10)

            except Exception as e:
                logger.warning(f"Failed to enhance {frame_path}: {e}")

    def create_caption_file(self, dataset_path: Path, concept_name: str, base_prompt: str):
        """Create caption file for training"""

        # Get concept details from database
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT trigger_word, base_prompt, description
                FROM lora_definitions
                WHERE name = %s
            """, (concept_name,))

            concept = cursor.fetchone()

            if concept:
                trigger_word = concept['trigger_word']
                prompt = concept['base_prompt'] or base_prompt

                # Create comprehensive caption
                caption = f"{trigger_word}, {prompt}, high quality, detailed, professional"

                caption_file = dataset_path / "caption.txt"
                caption_file.write_text(caption)

                logger.info(f"Created caption: {caption}")

        finally:
            conn.close()

    def build_dataset_for_lora(self, lora_name: str):
        """Complete dataset building process for a LoRA"""
        logger.info(f"Building dataset for LoRA: {lora_name}")

        # Get LoRA definition
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT d.*, c.name as category_name
                FROM lora_definitions d
                JOIN lora_categories c ON d.category_id = c.id
                WHERE d.name = %s
            """, (lora_name,))

            lora_def = cursor.fetchone()

            if not lora_def:
                raise ValueError(f"LoRA definition not found: {lora_name}")

            # Collect videos
            videos = self.collect_videos_for_concept(
                lora_def['category_name'],
                lora_name,
                limit=20  # Start with manageable dataset
            )

            if not videos:
                logger.warning(f"No videos found for {lora_name}, using fallback")
                videos = [{
                    "path": "/mnt/1TB-storage/ComfyUI/output/custom_lora_test_00001.mp4",
                    "scene_type": "fallback",
                    "generated_at": datetime.now(),
                    "prompt_id": "fallback"
                }]

            # Prepare dataset
            dataset_path, frame_count = self.prepare_training_dataset(
                lora_name,
                videos,
                fps=6 if lora_def['is_nsfw'] else 4,  # Higher FPS for NSFW content
                max_frames_per_video=30
            )

            # Enhance quality
            self.enhance_dataset_quality(dataset_path)

            # Create caption
            self.create_caption_file(
                dataset_path,
                lora_name,
                lora_def['base_prompt']
            )

            logger.info(f"✅ Dataset ready: {dataset_path} ({frame_count} frames)")
            return dataset_path, frame_count

        finally:
            conn.close()


def main():
    """Test dataset building"""
    manager = DatasetManager()

    # Test with cowgirl position
    dataset_path, frame_count = manager.build_dataset_for_lora("cowgirl_position")

    print(f"Dataset created: {dataset_path}")
    print(f"Total frames: {frame_count}")


if __name__ == "__main__":
    main()