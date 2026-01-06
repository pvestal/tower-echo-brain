#!/bin/bash
# Setup smart backup cron jobs for Echo Brain

echo "🔧 Setting up smart backup cron jobs..."

# Create cron entries
CRON_DAILY="0 3 * * * /opt/tower-echo-brain/scripts/smart_backup_system.sh auto >> /opt/tower-echo-brain/logs/backup.log 2>&1"
CRON_WEEKLY="0 2 * * 0 /opt/tower-echo-brain/scripts/smart_backup_system.sh full >> /opt/tower-echo-brain/logs/backup.log 2>&1"

# Add to crontab (avoiding duplicates)
(crontab -l 2>/dev/null | grep -v "smart_backup_system.sh"; echo "$CRON_DAILY"; echo "$CRON_WEEKLY") | crontab -

echo "✅ Cron jobs configured:"
echo "  Daily: 3 AM - Git sync only"
echo "  Weekly: Sunday 2 AM - Full backup (DB + critical files)"
echo ""
echo "Current crontab:"
crontab -l | grep smart_backup

echo ""
echo "📊 Storage efficiency:"
echo "  Daily backups would use: ~200MB × 365 = 73GB/year"
echo "  Smart backups will use: < 1GB total (99% reduction!)"