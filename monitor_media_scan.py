#!/usr/bin/env python3
"""
Echo Brain Media Scanner Progress Monitor
Monitors the progress of local media file analysis
"""

import psycopg2
import time
from datetime import datetime
import sys

class MediaScanMonitor:
    def __init__(self):
        self.db_config = {
            'host': '***REMOVED***',
            'user': 'patrick',
            'database': 'tower_consolidated',
            'port': 5432
        }
        self.last_count = 0
        
    def get_scan_progress(self):
        """Get current scanning progress from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Get overall progress
            cur.execute("""
                SELECT 
                    COUNT(*) as total_analyzed,
                    COUNT(*) FILTER (WHERE is_video = true) as videos,
                    COUNT(*) FILTER (WHERE is_video = false) as images,
                    COUNT(*) FILTER (WHERE file_path LIKE '***REMOVED***%') as local_videos,
                    COUNT(*) FILTER (WHERE file_path LIKE '/home/patrick/Pictures%') as local_images
                FROM echo_media_insights 
                WHERE learned_by_echo = true;
            """)
            
            progress = cur.fetchone()
            
            # Get recent activity (files processed in last 5 minutes)
            cur.execute("""
                SELECT COUNT(*) as recent_files
                FROM echo_media_insights 
                WHERE learned_by_echo = true 
                AND created_at > NOW() - INTERVAL '5 minutes';
            """)
            
            recent = cur.fetchone()
            
            # Get learning schedule status
            cur.execute("""
                SELECT source_path, files_processed, insights_extracted, status, last_processed
                FROM echo_learning_schedule 
                WHERE source_path IN ('***REMOVED***', '/home/patrick/Pictures')
                ORDER BY source_path;
            """)
            
            schedule_status = cur.fetchall()
            
            cur.close()
            conn.close()
            
            return {
                'total_analyzed': progress[0],
                'total_videos': progress[1], 
                'total_images': progress[2],
                'local_videos': progress[3],
                'local_images': progress[4],
                'recent_files': recent[0],
                'schedule_status': schedule_status
            }
            
        except Exception as e:
            print(f"Database error: {e}")
            return None
    
    def monitor_progress(self, interval=30):
        """Monitor progress with regular updates"""
        print("Echo Brain Media Scanner Progress Monitor")
        print("=" * 50)
        
        try:
            while True:
                progress = self.get_scan_progress()
                if not progress:
                    time.sleep(interval)
                    continue
                
                # Clear screen for updates
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Current Progress:")
                print(f"Total Files Analyzed: {progress['total_analyzed']:,}")
                print(f"  - Videos: {progress['total_videos']:,}")
                print(f"  - Images: {progress['total_images']:,}")
                print(f"\nLocal Files Analyzed:")
                print(f"  - Videos: {progress['local_videos']:,} / ~1,049")
                print(f"  - Images: {progress['local_images']:,} / ~18,077")
                
                # Calculate rates
                current_count = progress['total_analyzed']
                if self.last_count > 0:
                    rate = (current_count - self.last_count) / (interval / 60)  # files per minute
                    if rate > 0:
                        print(f"Processing Rate: {rate:.1f} files/minute")
                self.last_count = current_count
                
                print(f"Recent Activity: {progress['recent_files']} files in last 5 minutes")
                
                print(f"\nLearning Schedule Status:")
                for status in progress['schedule_status']:
                    path, processed, insights, status_text, last_proc = status
                    print(f"  {path}: {processed} files, {insights} insights [{status_text}]")
                    if last_proc:
                        print(f"    Last processed: {last_proc}")
                
                print("-" * 50)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            
if __name__ == "__main__":
    monitor = MediaScanMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        progress = monitor.get_scan_progress()
        if progress:
            print(f"Total analyzed: {progress['total_analyzed']:,}")
            print(f"Local videos: {progress['local_videos']:,}")  
            print(f"Local images: {progress['local_images']:,}")
    else:
        monitor.monitor_progress()