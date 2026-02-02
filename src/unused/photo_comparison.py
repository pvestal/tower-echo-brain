from fastapi import APIRouter
import sqlite3
import os
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import requests
import json

router = APIRouter()

@router.post('/api/photos/compare')
async def compare_photos(request: dict):
    '''Compare Google Photos cloud vs local files'''
    
    # Local photos from database
    conn = sqlite3.connect('/opt/tower-echo-brain/photos.db')
    cursor = conn.cursor()
    cursor.execute('SELECT file_path, file_hash, size FROM photos')
    local_photos = {row[1]: {'path': row[0], 'size': row[2]} for row in cursor.fetchall()}
    conn.close()
    
    # Get Google Photos from API
    creds_path = '/home/patrick/.config/gphotos-sync/credentials.json'
    cloud_photos = []
    
    if os.path.exists(creds_path):
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
            creds = Credentials(
                token=creds_data.get('access_token'),
                refresh_token=creds_data.get('refresh_token'),
                token_uri='https://oauth2.googleapis.com/token',
                client_id=creds_data.get('client_id'),
                client_secret=creds_data.get('client_secret')
            )
            
        if creds.expired:
            creds.refresh(Request())
            
        headers = {'Authorization': f'Bearer {creds.token}'}
        response = requests.get(
            'https://photoslibrary.googleapis.com/v1/mediaItems',
            headers=headers,
            params={'pageSize': 100}
        )
        
        if response.status_code == 200:
            cloud_items = response.json().get('mediaItems', [])
            cloud_photos = [item['filename'] for item in cloud_items]
    
    # Find differences
    local_count = len(local_photos)
    cloud_count = len(cloud_photos)
    
    # Check what's missing locally
    missing_locally = []
    for cloud_file in cloud_photos:
        found = False
        for local_hash, local_data in local_photos.items():
            if Path(local_data['path']).name == cloud_file:
                found = True
                break
        if not found:
            missing_locally.append(cloud_file)
    
    return {
        'local_photos': local_count,
        'cloud_photos': cloud_count,
        'indexed_size_mb': sum(p['size'] for p in local_photos.values()) / 1024 / 1024,
        'missing_locally': missing_locally[:10],  # First 10
        'missing_count': len(missing_locally),
        'sync_status': 'synced' if len(missing_locally) == 0 else 'out_of_sync',
        'recommendation': f'Download {len(missing_locally)} photos from cloud' if missing_locally else 'All photos synced'
    }

@router.post('/api/photos/sync')
async def trigger_sync(request: dict):
    '''Trigger full sync between cloud and local'''
    import subprocess
    
    # Kill existing sync processes
    os.system('pkill -f gphotos-sync')
    
    # Start new sync
    result = subprocess.Popen([
        '/home/patrick/.local/bin/gphotos-sync',
        '--secret', '/home/patrick/.config/gphotos-sync/client_secret.json',
        '--use-hardlinks',
        '--album-date-by-first-photo',
        '/mnt/10TB2/Google_Photos_API'
    ])
    
    return {
        'status': 'sync_started',
        'pid': result.pid,
        'message': 'Google Photos sync initiated. Check back for progress.'
    }
