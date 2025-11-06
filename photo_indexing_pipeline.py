#!/usr/bin/env python3
'''Google Photos indexing pipeline for Echo'''

import os
import sqlite3
import hashlib
import json
from pathlib import Path
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import exifread
import requests
from PIL import Image
import imagehash

class PhotoIndexer:
    def __init__(self):
        self.photos_path = Path('/mnt/10TB2/Google_Takeout_2025/Takeout/Google Photos')
        self.db_path = '/opt/tower-echo-brain/photos.db'
        self.pg_conn = psycopg2.connect(
            host='localhost',
            database='postgres',
            user=os.getenv("TOWER_USER", "patrick"),
            password='patrick123'
        )
        self.init_database()
        
    def init_database(self):
        '''Create tables for photo indexing'''
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                file_hash TEXT,
                perceptual_hash TEXT,
                size INTEGER,
                width INTEGER,
                height INTEGER,
                date_taken TEXT,
                location TEXT,
                metadata TEXT,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            
            
            
        ''')
        conn.commit()
        conn.close()
        
    def calculate_file_hash(self, filepath):
        '''Calculate SHA256 hash of file'''
        sha256_hash = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def calculate_perceptual_hash(self, filepath):
        '''Calculate perceptual hash for image deduplication'''
        try:
            img = Image.open(filepath)
            return str(imagehash.average_hash(img))
        except:
            return None
            
    def extract_metadata(self, filepath):
        '''Extract EXIF metadata from photos'''
        metadata = {}
        try:
            with open(filepath, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                for tag in tags.keys():
                    if tag not in ['JPEGThumbnail', 'TIFFThumbnail']:
                        metadata[tag] = str(tags[tag])
        except:
            pass
        return json.dumps(metadata)
    
    def index_photos(self):
        '''Index all photos in the Google Takeout directory'''
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        indexed = 0
        skipped = 0
        duplicates = 0
        
        print(f'🔍 Indexing photos from {self.photos_path}')
        
        for root, dirs, files in os.walk(self.photos_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov')):
                    filepath = Path(root) / file
                    
                    # Check if already indexed
                    cursor.execute('SELECT id FROM photos WHERE file_path = ?', (str(filepath),))
                    if cursor.fetchone():
                        skipped += 1
                        continue
                    
                    try:
                        # Calculate hashes
                        file_hash = self.calculate_file_hash(filepath)
                        perceptual_hash = self.calculate_perceptual_hash(filepath)
                        
                        # Check for duplicates
                        cursor.execute('SELECT file_path FROM photos WHERE file_hash = ?', (file_hash,))
                        duplicate = cursor.fetchone()
                        if duplicate:
                            duplicates += 1
                            print(f'  Duplicate found: {filepath.name} == {Path(duplicate[0]).name}')
                            continue
                        
                        # Get file info
                        stat = filepath.stat()
                        metadata = self.extract_metadata(filepath)
                        
                        # Get image dimensions
                        width, height = 0, 0
                        try:
                            with Image.open(filepath) as img:
                                width, height = img.size
                        except:
                            pass
                        
                        # Insert into database
                        cursor.execute('''
                            INSERT INTO photos (file_path, file_hash, perceptual_hash, size, 
                                              width, height, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (str(filepath), file_hash, perceptual_hash, stat.st_size, 
                             width, height, metadata))
                        
                        indexed += 1
                        if indexed % 100 == 0:
                            conn.commit()
                            print(f'  Indexed: {indexed}, Skipped: {skipped}, Duplicates: {duplicates}')
                            
                    except Exception as e:
                        print(f'  Error indexing {filepath}: {e}')
        
        conn.commit()
        conn.close()
        
        # Send to Knowledge Base
        self.send_to_kb(indexed, duplicates)
        
        print(f'\n✅ Indexing complete!')
        print(f'  Total indexed: {indexed}')
        print(f'  Duplicates found: {duplicates}')
        print(f'  Already indexed: {skipped}')
        
        return indexed, duplicates, skipped
    
    def send_to_kb(self, indexed, duplicates):
        '''Send indexing results to Knowledge Base'''
        try:
            kb_article = {
                'title': f'Photo Indexing Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                'content': f'''
# Google Photos Index Update

**Indexed**: {indexed} new photos
**Duplicates Found**: {duplicates}
**Database**: /opt/tower-echo-brain/photos.db

## Features Enabled:
- Fast duplicate detection via hash
- Visual similarity search via perceptual hash
- Metadata search (date, location, camera info)
- Integration with AI Assist for memory queries

## Next Steps:
- Query photos via Echo: "Show me photos from [date/location]"
- Find duplicates: "Find duplicate photos"
- Search by content: "Find photos with [person/object]"
''',
                'category': 'System Reports',
                'tags': ['photos', 'indexing', 'google-photos', 'echo-integration']
            }
            
            response = requests.post(
                'https://localhost/kb/api/articles',
                json=kb_article,
                verify=False
            )
            if response.status_code == 200:
                print(f'  📚 Indexing report saved to Knowledge Base')
        except Exception as e:
            print(f'  ⚠️  Could not save to KB: {e}')

if __name__ == '__main__':
    indexer = PhotoIndexer()
    indexer.index_photos()
