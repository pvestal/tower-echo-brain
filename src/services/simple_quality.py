import subprocess
import json
from pathlib import Path

def assess_video_quality(video_path):
    """Simple video quality assessment using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,bit_rate',
            '-of', 'json', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        stream = info.get('streams', [{}])[0]
        
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))
        bitrate = int(stream.get('bit_rate', 0) or 0)
        
        score = 0
        # Resolution (50 points max)
        if width >= 1024: score += 50
        elif width >= 512: score += 30
        else: score += 10
        
        # Bitrate (30 points max)
        if bitrate > 2000000: score += 30
        elif bitrate > 1000000: score += 20
        else: score += 10
        
        # File size (20 points)
        if Path(video_path).stat().st_size > 100000: score += 20
        
        return min(100, score)
    except:
        return 50

# Test on Goblin Slayer video
video = '***REMOVED***/AI_Generated/echo_video_1758230481_Goblin_Slayer_epic_b.mp4'
score = assess_video_quality(video)
print(f'Goblin Slayer Quality Score: {score}/100')
