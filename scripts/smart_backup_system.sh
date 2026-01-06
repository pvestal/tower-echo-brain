#!/bin/bash
# Smart Backup System for Echo Brain
# Leverages git/GitHub for code, focuses on data backups

set -e

BACKUP_DIR="/opt/tower-echo-brain/backups"
DB_NAME="echo_brain"
DB_HOST="***REMOVED***"
DB_USER="patrick"
DB_PASS="***REMOVED***"
GITHUB_REPO="git@github.com:pvestal/tower-echo-brain.git"
LOG_FILE="/opt/tower-echo-brain/logs/backup.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log_message() {
    echo -e "$1" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 1. Git Push - Primary code backup (no local storage needed)
git_backup() {
    log_message "${GREEN}📦 Starting Git backup...${NC}"
    cd /opt/tower-echo-brain

    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        log_message "${YELLOW}⚠️  Uncommitted changes found, committing...${NC}"
        git add -A
        git commit -m "Automated backup: $(date '+%Y-%m-%d %H:%M:%S')" || true
    fi

    # Push to GitHub
    if git push origin main 2>/dev/null; then
        log_message "${GREEN}✅ Code pushed to GitHub${NC}"
    else
        log_message "${RED}❌ Git push failed - check SSH keys${NC}"
        return 1
    fi
}

# 2. Database Backup - Weekly with rotation (keep 4 weeks)
database_backup() {
    log_message "${GREEN}🗄️  Starting database backup...${NC}"

    # Create backup directory if needed
    mkdir -p "$BACKUP_DIR/database"

    # Weekly backup filename
    BACKUP_FILE="$BACKUP_DIR/database/echo_brain_$(date +%Y_week%V).sql.gz"

    # Perform backup
    PGPASSWORD="$DB_PASS" pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
        --no-owner --clean --if-exists | gzip > "$BACKUP_FILE"

    if [ $? -eq 0 ]; then
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log_message "${GREEN}✅ Database backup complete: $BACKUP_FILE ($SIZE)${NC}"

        # Rotate old backups (keep only 4 weeks)
        find "$BACKUP_DIR/database" -name "echo_brain_*.sql.gz" -mtime +28 -delete
        log_message "${GREEN}🔄 Rotated old database backups (kept last 4 weeks)${NC}"
    else
        log_message "${RED}❌ Database backup failed${NC}"
        return 1
    fi
}

# 3. Critical Data Backup - Only unique data not in git
critical_data_backup() {
    log_message "${GREEN}💾 Backing up critical data...${NC}"

    mkdir -p "$BACKUP_DIR/critical"

    # Backup items that aren't in git and can't be recreated
    CRITICAL_ITEMS=(
        "/opt/tower-echo-brain/logs/production_echo.log"  # Current log only
        "/opt/tower-echo-brain/.env"                       # Environment config
        "/opt/vault/data"                                  # Vault secrets
    )

    for item in "${CRITICAL_ITEMS[@]}"; do
        if [ -e "$item" ]; then
            BASENAME=$(basename "$item")
            cp -r "$item" "$BACKUP_DIR/critical/${BASENAME}.$(date +%Y%m%d)" 2>/dev/null || true
        fi
    done

    # Rotate critical backups (keep 2 weeks)
    find "$BACKUP_DIR/critical" -mtime +14 -delete
    log_message "${GREEN}✅ Critical data backed up${NC}"
}

# 4. Backup Status Report
backup_status() {
    log_message "${GREEN}📊 Backup Status Report${NC}"

    # Git status
    cd /opt/tower-echo-brain
    LAST_COMMIT=$(git log -1 --format="%ci" 2>/dev/null || echo "Unknown")
    log_message "  Last Git commit: $LAST_COMMIT"

    # Database backups
    DB_COUNT=$(find "$BACKUP_DIR/database" -name "*.sql.gz" 2>/dev/null | wc -l)
    DB_SIZE=$(du -sh "$BACKUP_DIR/database" 2>/dev/null | cut -f1 || echo "0")
    log_message "  Database backups: $DB_COUNT files, Total: $DB_SIZE"

    # Total backup size
    TOTAL_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "0")
    log_message "  Total backup size: $TOTAL_SIZE"

    # Disk space available
    DISK_AVAIL=$(df -h /opt | tail -1 | awk '{print $4}')
    log_message "  Disk space available: $DISK_AVAIL"
}

# 5. Recovery Test (verify backups are valid)
test_recovery() {
    log_message "${GREEN}🧪 Testing backup integrity...${NC}"

    # Test database backup
    LATEST_DB=$(ls -t "$BACKUP_DIR/database"/*.sql.gz 2>/dev/null | head -1)
    if [ -n "$LATEST_DB" ]; then
        if gunzip -t "$LATEST_DB" 2>/dev/null; then
            log_message "${GREEN}✅ Database backup is valid${NC}"
        else
            log_message "${RED}❌ Database backup is corrupted!${NC}"
        fi
    fi

    # Test git connectivity
    cd /opt/tower-echo-brain
    if git ls-remote origin HEAD &>/dev/null; then
        log_message "${GREEN}✅ GitHub repository accessible${NC}"
    else
        log_message "${RED}❌ Cannot reach GitHub repository${NC}"
    fi
}

# Main execution
main() {
    log_message "${GREEN}🚀 Smart Backup System Started - $(date)${NC}"

    # Determine backup type based on day of week
    DAY=$(date +%u)

    if [ "$1" == "full" ] || [ "$DAY" == "7" ]; then
        # Sunday or manual full backup
        log_message "${YELLOW}📅 Running WEEKLY full backup${NC}"
        git_backup
        database_backup
        critical_data_backup
    else
        # Daily - just push code changes
        log_message "${YELLOW}📅 Running DAILY git sync${NC}"
        git_backup
    fi

    # Always run status and test
    backup_status
    test_recovery

    log_message "${GREEN}✨ Backup complete - $(date)${NC}"
    echo ""
}

# Handle command line args
case "${1:-auto}" in
    full)
        main "full"
        ;;
    status)
        backup_status
        ;;
    test)
        test_recovery
        ;;
    restore)
        echo "For restoration instructions, see: /opt/tower-echo-brain/RECOVERY_GUIDE.md"
        ;;
    *)
        main "auto"
        ;;
esac