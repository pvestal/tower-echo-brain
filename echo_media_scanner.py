#!/usr/bin/env python3
"""
Echo Brain Media Scanner - Local Media File Analysis
Scans and analyzes local media files for Echo Brain training
Processes images and videos in /home/patrick/Videos and /home/patrick/Pictures
"""

import os
import sys
import json
import hashlib
import asyncio
import aiohttp
import psycopg2
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image, ExifTags
import subprocess

# Media file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EchoMediaScanner:
    """Media scanner for Echo Brain local file analysis"""
    
    def __init__(self):
        self.db_config = {
            'host': '192.168.50.135',
            'user': 'patrick',
            'database': 'tower_consolidated',
            'port': 5432
        }
        self.ollama_url = "http://localhost:11434/api/generate"
        self.processed_files = 0
        self.skipped_files = 0
        self.errors = 0
        
    def get_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash of file for deduplication"""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None

    def extract_image_metadata(self, image_path: str) -> Dict:
        """Extract EXIF metadata from images"""
        metadata = {
            'width': None,
            'height': None,
            'camera_model': None,
            'date_taken': None,
            'file_size': None
        }
        
        try:
            # Get file size
            metadata['file_size'] = os.path.getsize(image_path)
            
            # Extract EXIF data
            with Image.open(image_path) as img:
                metadata['width'] = img.width
                metadata['height'] = img.height
                
                exif_data = img.getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        if tag == "Make":
                            camera_make = str(value).strip()
                        elif tag == "Model":
                            camera_model = str(value).strip()
                            if 'camera_make' in locals():
                                metadata['camera_model'] = f"{camera_make} {camera_model}"
                            else:
                                metadata['camera_model'] = camera_model
                        elif tag == "DateTime":
                            try:
                                metadata['date_taken'] = datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                            except:
                                pass
                                
        except Exception as e:
            logger.error(f"Error extracting metadata from {image_path}: {e}")
            
        return metadata

    def extract_video_metadata(self, video_path: str) -> Dict:
        """Extract metadata from video files using ffprobe"""
        metadata = {
            'width': None,
            'height': None,
            'duration_seconds': None,
            'fps': None,
            'resolution': None,
            'codec': None,
            'file_size': None
        }
        
        try:
            metadata['file_size'] = os.path.getsize(video_path)
            
            # Use ffprobe to get video metadata
            cmd = [
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract video stream info
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        metadata['width'] = stream.get('width')
                        metadata['height'] = stream.get('height')
                        metadata['codec'] = stream.get('codec_name')
                        
                        # Calculate FPS
                        fps_str = stream.get('r_frame_rate', '0/1')
                        if '/' in fps_str:
                            num, den = fps_str.split('/')
                            if int(den) > 0:
                                metadata['fps'] = float(num) / float(den)
                        
                        # Resolution string
                        if metadata['width'] and metadata['height']:
                            metadata['resolution'] = f"{metadata['width']}x{metadata['height']}"
                        break
                
                # Duration from format info
                format_info = data.get('format', {})
                duration = format_info.get('duration')
                if duration:
                    metadata['duration_seconds'] = int(float(duration))
                    
        except Exception as e:
            logger.error(f"Error extracting video metadata from {video_path}: {e}")
            
        return metadata

    async def analyze_with_ollama(self, file_path: str, is_video: bool = False) -> Dict:
        """Analyze media file using Ollama vision models"""
        analysis_result = {
            'scene_description': None,
            'categories': [],
            'emotions': [],
            'people': [],
            'locations': [],
            'events': []
        }
        
        try:
            # For images, analyze directly
            if not is_video:
                prompt = """Analyze this image and provide:
1. A detailed scene description
2. Categories/tags that apply
3. Any emotions visible
4. People (generic descriptions like 'child', 'adult', etc.)
5. Locations or settings visible
6. Events or activities happening

Provide a structured response focusing on what's actually visible in the image."""

                # Read image as base64 for Ollama
                with open(file_path, 'rb') as f:
                    import base64
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                
                payload = {
                    "model": "llava:latest",  # Vision model
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.ollama_url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get('response', '')
                            
                            # Parse the response to extract structured data
                            analysis_result['scene_description'] = response_text[:500]  # Limit length
                            
                            # Simple keyword extraction for categories
                            keywords = self.extract_keywords(response_text)
                            analysis_result['categories'] = keywords[:10]  # Limit to 10
                            
            else:
                # For videos, extract a frame and analyze that
                # This is a simplified approach - could be enhanced with multiple frames
                analysis_result['scene_description'] = f"Video file: {os.path.basename(file_path)}"
                analysis_result['categories'] = ['video']
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path} with Ollama: {e}")
            analysis_result['scene_description'] = f"Analysis failed: {str(e)}"
            
        return analysis_result

    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from analysis text"""
        # Simple keyword extraction - could be enhanced with NLP
        common_categories = [
            'person', 'people', 'child', 'adult', 'family',
            'indoor', 'outdoor', 'nature', 'building', 'street',
            'food', 'animal', 'car', 'house', 'tree', 'water',
            'sky', 'portrait', 'landscape', 'document', 'text'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        for keyword in common_categories:
            if keyword in text_lower:
                found_keywords.append(keyword)
                
        return found_keywords

    def file_already_processed(self, file_hash: str) -> bool:
        """Check if file was already processed"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("SELECT id FROM echo_media_insights WHERE file_hash = %s", (file_hash,))
            result = cur.fetchone()
            
            cur.close()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Database error checking file hash: {e}")
            return False

    def save_to_database(self, file_path: str, file_hash: str, metadata: Dict, analysis: Dict, is_video: bool = False):
        """Save media analysis results to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Prepare data for insertion
            insert_data = {
                'file_path': file_path,
                'file_hash': file_hash,
                'date_taken': metadata.get('date_taken'),
                'camera_model': metadata.get('camera_model'),
                'file_size': metadata.get('file_size'),
                'width': metadata.get('width'),
                'height': metadata.get('height'),
                'scene_description': analysis.get('scene_description'),
                'categories': analysis.get('categories', []),
                'emotions': analysis.get('emotions', []),
                'people': analysis.get('people', []),
                'locations': analysis.get('locations', []),
                'events': analysis.get('events', []),
                'learned_by_echo': True,
                'is_video': is_video,
                'duration_seconds': metadata.get('duration_seconds') if is_video else None,
                'fps': metadata.get('fps') if is_video else None,
                'resolution': metadata.get('resolution') if is_video else None,
                'codec': metadata.get('codec') if is_video else None
            }
            
            # Insert query
            insert_query = """
                INSERT INTO echo_media_insights (
                    file_path, file_hash, date_taken, camera_model, file_size, width, height,
                    scene_description, categories, emotions, people, locations, events,
                    learned_by_echo, is_video, duration_seconds, fps, resolution, codec
                ) VALUES (
                    %(file_path)s, %(file_hash)s, %(date_taken)s, %(camera_model)s, %(file_size)s,
                    %(width)s, %(height)s, %(scene_description)s, %(categories)s, %(emotions)s,
                    %(people)s, %(locations)s, %(events)s, %(learned_by_echo)s, %(is_video)s,
                    %(duration_seconds)s, %(fps)s, %(resolution)s, %(codec)s
                )
            """
            
            cur.execute(insert_query, insert_data)
            conn.commit()
            
            cur.close()
            conn.close()
            
            logger.info(f"Saved analysis for: {os.path.basename(file_path)}")
            self.processed_files += 1
            
        except Exception as e:
            logger.error(f"Error saving to database for {file_path}: {e}")
            self.errors += 1

    async def process_file(self, file_path: str) -> bool:
        """Process a single media file"""
        try:
            # Check if it's an image or video
            file_ext = Path(file_path).suffix.lower()
            is_video = file_ext in VIDEO_EXTENSIONS
            is_image = file_ext in IMAGE_EXTENSIONS
            
            if not (is_image or is_video):
                logger.debug(f"Skipping non-media file: {file_path}")
                return False
            
            # Generate file hash
            file_hash = self.get_file_hash(file_path)
            if not file_hash:
                logger.error(f"Could not generate hash for: {file_path}")
                self.errors += 1
                return False
            
            # Check if already processed
            if self.file_already_processed(file_hash):
                logger.debug(f"Already processed: {os.path.basename(file_path)}")
                self.skipped_files += 1
                return False
            
            logger.info(f"Processing {'video' if is_video else 'image'}: {os.path.basename(file_path)}")
            
            # Extract metadata
            if is_video:
                metadata = self.extract_video_metadata(file_path)
            else:
                metadata = self.extract_image_metadata(file_path)
            
            # Analyze content with Ollama (only for images for now)
            if is_image:
                analysis = await self.analyze_with_ollama(file_path, is_video=False)
            else:
                # Basic video analysis - could be enhanced
                analysis = {
                    'scene_description': f"Video: {os.path.basename(file_path)}",
                    'categories': ['video'],
                    'emotions': [],
                    'people': [],
                    'locations': [],
                    'events': []
                }
            
            # Save to database
            self.save_to_database(file_path, file_hash, metadata, analysis, is_video)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.errors += 1
            return False

    async def scan_directory(self, directory_path: str, max_files: Optional[int] = None) -> Dict:
        """Scan a directory for media files and process them"""
        logger.info(f"Starting scan of directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return {"error": f"Directory not found: {directory_path}"}
        
        # Get all media files in directory
        media_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file_path).suffix.lower()
                if file_ext in IMAGE_EXTENSIONS or file_ext in VIDEO_EXTENSIONS:
                    media_files.append(file_path)
        
        logger.info(f"Found {len(media_files)} media files")
        
        # Limit files if specified
        if max_files:
            media_files = media_files[:max_files]
            logger.info(f"Limited to first {max_files} files")
        
        # Process files with concurrency control
        semaphore = asyncio.Semaphore(3)  # Process 3 files concurrently
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_file(file_path)
        
        # Process all files
        tasks = [process_with_semaphore(file_path) for file_path in media_files]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update learning schedule status
        self.update_learning_schedule(directory_path)
        
        return {
            "directory": directory_path,
            "total_files": len(media_files),
            "processed": self.processed_files,
            "skipped": self.skipped_files,
            "errors": self.errors
        }

    def update_learning_schedule(self, directory_path: str):
        """Update the learning schedule with processing results"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Update the schedule entry
            update_query = """
                UPDATE echo_learning_schedule 
                SET files_processed = %s,
                    insights_extracted = %s,
                    last_processed = %s,
                    status = 'completed'
                WHERE source_path = %s
            """
            
            cur.execute(update_query, (
                self.processed_files, 
                self.processed_files, 
                datetime.now(), 
                directory_path
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Updated learning schedule for: {directory_path}")
            
        except Exception as e:
            logger.error(f"Error updating learning schedule: {e}")

async def main():
    """Main function to run the media scanner"""
    scanner = EchoMediaScanner()
    
    # Directories to scan from learning schedule
    directories_to_scan = [
        "/home/patrick/Videos",
        "/home/patrick/Pictures"
    ]
    
    total_results = {
        "total_processed": 0,
        "total_skipped": 0,
        "total_errors": 0,
        "directories": {}
    }
    
    # Process each directory
    for directory in directories_to_scan:
        logger.info(f"\n=== Processing {directory} ===")
        
        # Reset counters for each directory
        scanner.processed_files = 0
        scanner.skipped_files = 0
        scanner.errors = 0
        
        # Scan directory (process all files)
        result = await scanner.scan_directory(directory)
        
        # Accumulate results
        total_results["directories"][directory] = result
        total_results["total_processed"] += scanner.processed_files
        total_results["total_skipped"] += scanner.skipped_files
        total_results["total_errors"] += scanner.errors
        
        logger.info(f"Directory {directory} results: {result}")
    
    # Print final summary
    print(f"\n=== SCAN COMPLETE ===")
    print(f"Total files processed: {total_results['total_processed']}")
    print(f"Total files skipped: {total_results['total_skipped']}")
    print(f"Total errors: {total_results['total_errors']}")
    
    for directory, result in total_results["directories"].items():
        print(f"\n{directory}:")
        print(f"  Files found: {result.get('total_files', 0)}")
        print(f"  Processed: {result.get('processed', 0)}")
        print(f"  Skipped: {result.get('skipped', 0)}")
        print(f"  Errors: {result.get('errors', 0)}")

if __name__ == "__main__":
    asyncio.run(main())