
# Auto-fix permissions for Jellyfin (learned from Patrick's feedback)
import os
import stat

def set_jellyfin_permissions(video_path):
    """Set proper permissions for media server access"""
    try:
        os.chmod(video_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        print(f"✅ Set Jellyfin permissions on {video_path}")
        return True
    except Exception as e:
        print(f"❌ Permission fix failed: {e}")
        return False

# Call this after every video generation
# set_jellyfin_permissions(output_video_path)
