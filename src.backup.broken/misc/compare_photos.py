#!/usr/bin/env python3
'''Simple photo comparison tool - ask Echo to run this'''

import os
import sqlite3
from pathlib import Path

# Check local photos
db = sqlite3.connect('/opt/tower-echo-brain/photos.db')
cursor = db.cursor()
cursor.execute('SELECT COUNT(*), SUM(size)/1024/1024/1024 FROM photos')
local_count, local_gb = cursor.fetchone()
db.close()

# Check takeout photos
takeout_path = Path('/mnt/10TB2/Google_Takeout_2025/Takeout/Google Photos')
takeout_files = list(takeout_path.rglob('*.jpg')) + list(takeout_path.rglob('*.mp4'))
takeout_count = len(takeout_files)

# Check gphotos-sync directory
api_path = Path('/mnt/10TB2/Google_Photos_API')
if api_path.exists():
    api_files = list(api_path.rglob('*.jpg')) + list(api_path.rglob('*.mp4'))
    api_count = len(api_files)
else:
    api_count = 0

print(f'''
📊 GOOGLE PHOTOS STATUS:

Local Database:
  • Indexed: {local_count:,} photos
  • Size: {local_gb:.1f} GB
  • Location: /opt/tower-echo-brain/photos.db

Google Takeout:
  • Files: {takeout_count:,} photos/videos
  • Location: {takeout_path}

Google Photos API Sync:
  • Files: {api_count:,} photos/videos
  • Location: {api_path}

Comparison:
  • Takeout has {takeout_count - local_count:,} more files than indexed
  • To index remaining: python /opt/tower-echo-brain/photo_indexer_fixed.py
  • To sync from cloud: gphotos-sync (needs OAuth setup)
''')
