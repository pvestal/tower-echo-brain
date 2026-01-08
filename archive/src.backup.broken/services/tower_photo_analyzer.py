#!/usr/bin/env python3
"""
Tower Photo Analyzer - Runs on Tower with LLaVA
Analyzes Google Photos from /mnt/10TB2/Google_Takeout_2025
"""

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
import base64
import aiohttp
from typing import Dict, List

class TowerPhotoAnalyzer:
    def __init__(self):
        self.photo_paths = [
            '/mnt/10TB2/Google_Takeout_2025',
            '/mnt/10TB1/GooglePhotos',
            '/mnt/10TB1/Google_Photos_Master'
        ]
        self.db_path = '/opt/tower-echo-brain/data/photos.db'
        self.ollama_url = 'http://localhost:11434'
        self.init_db()
        
    def init_db(self):
        Path(self.db_path).parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                hash TEXT,
                size INTEGER,
                analyzed BOOLEAN DEFAULT 0,
                analysis TEXT,
                quality_score REAL,
                categories TEXT
            )
        ''')
        self.conn.commit()
    
    async def scan_photos(self):
        total = 0
        for base_path in self.photo_paths:
            if not Path(base_path).exists():
                continue
            for photo in Path(base_path).rglob('*.jpg'):
                self.conn.execute(
                    'INSERT OR IGNORE INTO photos (path, size) VALUES (?, ?)',
                    (str(photo), photo.stat().st_size)
                )
                total += 1
        self.conn.commit()
        return total
    
    async def analyze_with_llava(self, photo_path: str):
        with open(photo_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{self.ollama_url}/api/generate',
                json={
                    'model': 'llava:7b',
                    'prompt': 'Describe this photo. What do you see? Rate quality 1-10.',
                    'images': [image_base64],
                    'stream': False
                }
            ) as resp:
                result = await resp.json()
                return result.get('response', '')

async def main():
    analyzer = TowerPhotoAnalyzer()
    print(f'Scanning photos on Tower...')
    count = await analyzer.scan_photos()
    print(f'Found {count} photos')
    
    # Get stats
    cursor = analyzer.conn.execute('SELECT COUNT(*) FROM photos')
    total = cursor.fetchone()[0]
    print(f'Database has {total} photos')

if __name__ == '__main__':
    asyncio.run(main())
EOF'
