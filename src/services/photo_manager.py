#!/usr/bin/env python3
"""
Echo Google Photos Manager with LLaVA Vision
Manages local and cloud photos with deduplication and AI analysis
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import aiofiles
import aiohttp
from PIL import Image
import imagehash
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import exifread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Photos API scope
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']


class PhotoDeduplicationDB:
    """
    SQLite database for tracking photo duplicates and metadata
    """

    def __init__(self, db_path: str = "/home/patrick/.echo_photos.db"):
        self.db_path = db_path
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database with tables"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Photos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                file_hash TEXT,
                perceptual_hash TEXT,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                date_taken TIMESTAMP,
                location TEXT,
                camera_model TEXT,
                google_photo_id TEXT,
                echo_analyzed BOOLEAN DEFAULT 0,
                echo_analysis TEXT,
                categories TEXT,
                people_detected TEXT,
                quality_score REAL,
                is_duplicate BOOLEAN DEFAULT 0,
                duplicate_group_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Duplicate groups table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS duplicate_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                master_photo_id INTEGER,
                total_duplicates INTEGER,
                space_wasted INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (master_photo_id) REFERENCES photos(id)
            )
        """)

        # Analysis history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER,
                expert_type TEXT,
                analysis_text TEXT,
                confidence REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (photo_id) REFERENCES photos(id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON photos(file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON photos(perceptual_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date_taken ON photos(date_taken)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_echo_analyzed ON photos(echo_analyzed)")

        self.conn.commit()

    def add_photo(self, photo_data: Dict) -> int:
        """Add photo to database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO photos (
                file_path, file_hash, perceptual_hash, file_size,
                width, height, date_taken, location, camera_model,
                google_photo_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            photo_data.get('file_path'),
            photo_data.get('file_hash'),
            photo_data.get('perceptual_hash'),
            photo_data.get('file_size'),
            photo_data.get('width'),
            photo_data.get('height'),
            photo_data.get('date_taken'),
            photo_data.get('location'),
            photo_data.get('camera_model'),
            photo_data.get('google_photo_id')
        ))
        self.conn.commit()
        return cursor.lastrowid

    def find_duplicates(self, threshold: int = 5) -> List[Dict]:
        """Find duplicate photos based on perceptual hash"""
        cursor = self.conn.cursor()

        # Find photos with similar perceptual hashes
        cursor.execute("""
            SELECT p1.id, p1.file_path, p1.perceptual_hash, p1.file_size,
                   p2.id, p2.file_path, p2.perceptual_hash, p2.file_size
            FROM photos p1
            JOIN photos p2 ON p1.id < p2.id
            WHERE p1.perceptual_hash IS NOT NULL
              AND p2.perceptual_hash IS NOT NULL
              AND p1.is_duplicate = 0
              AND p2.is_duplicate = 0
        """)

        duplicates = []
        for row in cursor.fetchall():
            # Calculate hamming distance between perceptual hashes
            hash1 = imagehash.hex_to_hash(row[2])
            hash2 = imagehash.hex_to_hash(row[6])
            distance = hash1 - hash2

            if distance <= threshold:
                duplicates.append({
                    'photo1_id': row[0],
                    'photo1_path': row[1],
                    'photo1_size': row[3],
                    'photo2_id': row[4],
                    'photo2_path': row[5],
                    'photo2_size': row[7],
                    'similarity': 100 - (distance * 10)  # Convert to similarity percentage
                })

        return duplicates

    def mark_duplicate(self, photo_id: int, group_id: int):
        """Mark a photo as duplicate"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE photos
            SET is_duplicate = 1, duplicate_group_id = ?
            WHERE id = ?
        """, (group_id, photo_id))
        self.conn.commit()

    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()

        stats = {}

        # Total photos
        cursor.execute("SELECT COUNT(*) FROM photos")
        stats['total_photos'] = cursor.fetchone()[0]

        # Analyzed photos
        cursor.execute("SELECT COUNT(*) FROM photos WHERE echo_analyzed = 1")
        stats['analyzed_photos'] = cursor.fetchone()[0]

        # Duplicates found
        cursor.execute("SELECT COUNT(*) FROM photos WHERE is_duplicate = 1")
        stats['duplicates_found'] = cursor.fetchone()[0]

        # Space wasted by duplicates
        cursor.execute("""
            SELECT SUM(file_size) FROM photos WHERE is_duplicate = 1
        """)
        result = cursor.fetchone()[0]
        stats['space_wasted_bytes'] = result if result else 0
        stats['space_wasted_mb'] = stats['space_wasted_bytes'] / (1024 * 1024)

        # Photos by year
        cursor.execute("""
            SELECT strftime('%Y', date_taken) as year, COUNT(*)
            FROM photos
            WHERE date_taken IS NOT NULL
            GROUP BY year
            ORDER BY year
        """)
        stats['photos_by_year'] = dict(cursor.fetchall())

        return stats


class EchoPhotoAnalyzer:
    """
    Analyzes photos using Echo's LLaVA vision capabilities
    """

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.model = "llava:7b"
        self.db = PhotoDeduplicationDB()

    async def analyze_photo(self, photo_path: str) -> Dict:
        """
        Comprehensive photo analysis with LLaVA
        """
        try:
            # Load and prepare image
            with open(photo_path, 'rb') as f:
                image_data = f.read()

            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Multi-aspect analysis prompt
            prompt = """Analyze this photo comprehensively:

1. SCENE: What is the main subject or scene?
2. PEOPLE: Are there people? If yes, describe general appearance (no names).
3. LOCATION: Where was this likely taken?
4. TIME: What time of day/season does it appear to be?
5. QUALITY: Rate photo quality (composition, lighting, focus) from 1-10.
6. MEMORIES: What kind of memory or event might this represent?
7. CATEGORIES: Suggest 3-5 categories (e.g., family, travel, nature, food, etc.)
8. EMOTION: What emotion or mood does this photo convey?
9. KEEP/DELETE: Should this photo be kept? Consider quality, uniqueness, and memory value.

Provide structured analysis."""

            # Call Ollama API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": [image_base64],
                        "stream": False
                    }
                ) as response:
                    result = await response.json()

            analysis = result.get('response', '')

            # Parse and structure the analysis
            structured = self._parse_analysis(analysis)

            # Update database
            self.db.conn.execute("""
                UPDATE photos
                SET echo_analyzed = 1,
                    echo_analysis = ?,
                    categories = ?,
                    quality_score = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
            """, (
                analysis,
                json.dumps(structured.get('categories', [])),
                structured.get('quality_score', 5),
                photo_path
            ))
            self.db.conn.commit()

            return {
                'success': True,
                'analysis': structured,
                'raw_analysis': analysis
            }

        except Exception as e:
            logger.error(f"Failed to analyze {photo_path}: {e}")
            return {'success': False, 'error': str(e)}

    def _parse_analysis(self, analysis_text: str) -> Dict:
        """Parse LLaVA response into structured data"""
        structured = {
            'scene': '',
            'people_detected': False,
            'location': '',
            'time': '',
            'quality_score': 5,
            'memory_type': '',
            'categories': [],
            'emotion': '',
            'keep_recommendation': True
        }

        lines = analysis_text.split('\n')
        for line in lines:
            line_lower = line.lower()

            if 'scene:' in line_lower:
                structured['scene'] = line.split(':', 1)[1].strip()
            elif 'people:' in line_lower:
                structured['people_detected'] = 'yes' in line_lower or 'person' in line_lower
            elif 'location:' in line_lower:
                structured['location'] = line.split(':', 1)[1].strip()
            elif 'quality:' in line_lower:
                # Extract number from quality rating
                import re
                match = re.search(r'(\d+)', line)
                if match:
                    structured['quality_score'] = int(match.group(1))
            elif 'categories:' in line_lower:
                # Extract categories
                cats = line.split(':', 1)[1].strip()
                structured['categories'] = [c.strip() for c in cats.split(',')]
            elif 'keep' in line_lower and 'delete' in line_lower:
                structured['keep_recommendation'] = 'keep' in line_lower

        return structured


class GooglePhotosSync:
    """
    Syncs with Google Photos API
    """

    def __init__(self, credentials_path: str = None):
        self.creds = None
        self.service = None
        self.credentials_path = credentials_path or "/home/patrick/.google_photos_creds.json"

    def authenticate(self):
        """Authenticate with Google Photos"""
        # Token storage
        token_path = Path.home() / '.google_photos_token.json'

        if token_path.exists():
            self.creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        # If no valid credentials, get new ones
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                self.creds = flow.run_local_server(port=0)

            # Save credentials
            with open(token_path, 'w') as token:
                token.write(self.creds.to_json())

        self.service = build('photoslibrary', 'v1', credentials=self.creds)

    async def list_photos(self, page_size: int = 100) -> List[Dict]:
        """List photos from Google Photos"""
        photos = []
        next_page_token = None

        while True:
            results = self.service.mediaItems().list(
                pageSize=page_size,
                pageToken=next_page_token
            ).execute()

            items = results.get('mediaItems', [])
            photos.extend(items)

            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break

        return photos


class EchoPhotoManager:
    """
    Main manager that coordinates everything
    """

    def __init__(self):
        self.db = PhotoDeduplicationDB()
        self.analyzer = EchoPhotoAnalyzer()
        self.google_sync = GooglePhotosSync()
        self.local_paths = [
            "/home/patrick/CloudMedia/Patrick-Complete-Media/Photos",
            "/home/patrick/Documents/google_takeout_extracted",
            "/home/patrick/Pictures"
        ]

    async def scan_local_photos(self) -> Dict:
        """Scan all local photo directories"""
        stats = {'scanned': 0, 'new': 0, 'errors': 0}

        for base_path in self.local_paths:
            if not Path(base_path).exists():
                continue

            for photo_path in Path(base_path).rglob('*'):
                if photo_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic', '.webp']:
                    stats['scanned'] += 1

                    try:
                        # Calculate hashes
                        file_hash = self._calculate_file_hash(photo_path)
                        perceptual_hash = self._calculate_perceptual_hash(photo_path)

                        # Get metadata
                        metadata = self._extract_metadata(photo_path)

                        # Add to database
                        photo_data = {
                            'file_path': str(photo_path),
                            'file_hash': file_hash,
                            'perceptual_hash': perceptual_hash,
                            'file_size': photo_path.stat().st_size,
                            **metadata
                        }

                        self.db.add_photo(photo_data)
                        stats['new'] += 1

                    except Exception as e:
                        logger.error(f"Error processing {photo_path}: {e}")
                        stats['errors'] += 1

        return stats

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _calculate_perceptual_hash(self, file_path: Path) -> str:
        """Calculate perceptual hash for image similarity"""
        try:
            img = Image.open(file_path)
            return str(imagehash.phash(img))
        except:
            return None

    def _extract_metadata(self, file_path: Path) -> Dict:
        """Extract EXIF metadata from photo"""
        metadata = {
            'width': None,
            'height': None,
            'date_taken': None,
            'camera_model': None,
            'location': None
        }

        try:
            # Get image dimensions
            with Image.open(file_path) as img:
                metadata['width'], metadata['height'] = img.size

            # Get EXIF data
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f)

                # Date taken
                if 'EXIF DateTimeOriginal' in tags:
                    date_str = str(tags['EXIF DateTimeOriginal'])
                    metadata['date_taken'] = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')

                # Camera model
                if 'Image Model' in tags:
                    metadata['camera_model'] = str(tags['Image Model'])

                # GPS location
                if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                    lat = tags['GPS GPSLatitude']
                    lon = tags['GPS GPSLongitude']
                    metadata['location'] = f"{lat},{lon}"

        except Exception as e:
            logger.debug(f"Could not extract metadata from {file_path}: {e}")

        return metadata

    async def find_and_report_duplicates(self) -> Dict:
        """Find all duplicates and generate report"""
        duplicates = self.db.find_duplicates(threshold=5)

        # Group duplicates
        duplicate_groups = {}
        for dup in duplicates:
            group_key = tuple(sorted([dup['photo1_id'], dup['photo2_id']]))
            if group_key not in duplicate_groups:
                duplicate_groups[group_key] = []
            duplicate_groups[group_key].append(dup)

        # Calculate space savings
        total_space_wasted = sum(min(d['photo1_size'], d['photo2_size'])
                                 for d in duplicates)

        report = {
            'total_duplicate_pairs': len(duplicates),
            'duplicate_groups': len(duplicate_groups),
            'space_wasted_mb': total_space_wasted / (1024 * 1024),
            'duplicates': duplicates[:10]  # First 10 for review
        }

        return report

    async def analyze_unprocessed_photos(self, limit: int = 10):
        """Analyze photos that haven't been processed by Echo yet"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT file_path FROM photos
            WHERE echo_analyzed = 0
            LIMIT ?
        """, (limit,))

        photos = cursor.fetchall()
        results = []

        for (photo_path,) in photos:
            logger.info(f"Analyzing: {photo_path}")
            result = await self.analyzer.analyze_photo(photo_path)
            results.append({
                'path': photo_path,
                'success': result['success'],
                'analysis': result.get('analysis', {})
            })

        return results


# CLI interface
async def main():
    """Main entry point"""
    manager = EchoPhotoManager()

    print("🖼️ Echo Photo Manager Starting...")

    # Scan local photos
    print("\n📁 Scanning local photos...")
    scan_stats = await manager.scan_local_photos()
    print(f"  Scanned: {scan_stats['scanned']}")
    print(f"  New: {scan_stats['new']}")
    print(f"  Errors: {scan_stats['errors']}")

    # Get database stats
    db_stats = manager.db.get_stats()
    print(f"\n📊 Database Stats:")
    print(f"  Total photos: {db_stats['total_photos']}")
    print(f"  Analyzed: {db_stats['analyzed_photos']}")
    print(f"  Duplicates: {db_stats['duplicates_found']}")
    print(f"  Space wasted: {db_stats['space_wasted_mb']:.2f} MB")

    # Find duplicates
    print("\n🔍 Finding duplicates...")
    dup_report = await manager.find_and_report_duplicates()
    print(f"  Duplicate pairs: {dup_report['total_duplicate_pairs']}")
    print(f"  Space recoverable: {dup_report['space_wasted_mb']:.2f} MB")

    # Analyze some photos
    print("\n🧠 Analyzing photos with Echo LLaVA...")
    analyses = await manager.analyze_unprocessed_photos(limit=5)
    for analysis in analyses:
        if analysis['success']:
            print(f"  ✅ {Path(analysis['path']).name}")
            if analysis['analysis']:
                print(f"     Categories: {', '.join(analysis['analysis'].get('categories', []))}")
                print(f"     Quality: {analysis['analysis'].get('quality_score', 0)}/10")
                print(f"     Keep: {'Yes' if analysis['analysis'].get('keep_recommendation') else 'No'}")


if __name__ == "__main__":
    asyncio.run(main())