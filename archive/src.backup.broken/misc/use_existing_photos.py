#!/usr/bin/env python3
'''Since OAuth is blocked, use the 14,301 photos you already have'''

import os
from pathlib import Path

takeout_path = Path('/mnt/10TB2/Google_Takeout_2025/Takeout/Google Photos')
photos = list(takeout_path.rglob('*.jpg')) + list(takeout_path.rglob('*.jpeg'))
videos = list(takeout_path.rglob('*.mp4')) + list(takeout_path.rglob('*.mov'))

print(f'''
✅ YOU ALREADY HAVE YOUR PHOTOS!

Google Takeout Photos Available:
• Photos: {len(photos):,} files
• Videos: {len(videos):,} files  
• Total: {len(photos) + len(videos):,} files
• Location: {takeout_path}

Why fight with OAuth when you have 14,301 files already downloaded?

To use them:
1. Index for search: python3 /opt/tower-echo-brain/photo_indexer_fixed.py
2. View in file browser: {takeout_path}
3. Sync to other locations: rsync -av {takeout_path}/ /destination/

The OAuth hassle is because Google requires app verification for web apps.
Your photos are already on Tower. Use them!
''')
