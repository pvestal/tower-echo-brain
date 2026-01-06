#!/usr/bin/env python3
import os, sqlite3, hashlib, json
from pathlib import Path
from datetime import datetime
from PIL import Image
import imagehash

class PhotoIndexer:
    def __init__(self):
        self.photos_path = Path('/mnt/10TB2/Google_Takeout_2025/Takeout/Google Photos')
        self.db_path = '/opt/tower-echo-brain/photos.db'
        self.init_database()
        
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY, file_path TEXT UNIQUE, file_hash TEXT,
            perceptual_hash TEXT, size INTEGER, width INTEGER, height INTEGER,
            date_taken TEXT, metadata TEXT, indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_hash ON photos(file_hash)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_phash ON photos(perceptual_hash)')
        conn.commit()
        conn.close()
        
    def calculate_file_hash(self, filepath):
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def index_photos(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        indexed = duplicates = 0
        
        print(f'🔍 Indexing {self.photos_path}')
        for root, _, files in os.walk(self.photos_path):
            for file in files[:100]:  # Limit to 100 for testing
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4')):
                    filepath = Path(root) / file
                    cursor.execute('SELECT id FROM photos WHERE file_path = ?', (str(filepath),))
                    if cursor.fetchone(): continue
                    
                    try:
                        file_hash = self.calculate_file_hash(filepath)
                        cursor.execute('SELECT file_path FROM photos WHERE file_hash = ?', (file_hash,))
                        if cursor.fetchone():
                            duplicates += 1
                            continue
                        
                        size = filepath.stat().st_size
                        width = height = 0
                        phash = None
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            try:
                                with Image.open(filepath) as img:
                                    width, height = img.size
                                    phash = str(imagehash.average_hash(img))
                            except: pass
                        
                        cursor.execute('''INSERT INTO photos (file_path, file_hash, perceptual_hash,
                            size, width, height) VALUES (?, ?, ?, ?, ?, ?)''',
                            (str(filepath), file_hash, phash, size, width, height))
                        indexed += 1
                        
                        if indexed % 10 == 0:
                            conn.commit()
                            print(f'  Indexed: {indexed}, Duplicates: {duplicates}')
                    except Exception as e:
                        print(f'  Error: {e}')
                        
        conn.commit()
        conn.close()
        print(f'✅ Complete! Indexed: {indexed}, Duplicates: {duplicates}')
        return indexed, duplicates

if __name__ == '__main__':
    PhotoIndexer().index_photos()
