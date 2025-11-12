#!/bin/bash

# Echo Brain Deployment Script
# Usage: ./scripts/deployment/deploy.sh [environment]

set -euo pipefail

ENVIRONMENT=${1:-production}
SERVICE_NAME="tower-echo-brain"
SERVICE_PATH="/opt/tower-echo-brain"
BACKUP_DIR="/opt/backups/echo-brain"
LOG_FILE="/tmp/echo-brain-deployment.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

create_backup() {
    log "Creating backup..."
    BACKUP_NAME="backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        systemctl stop "$SERVICE_NAME"
        log "Service stopped for backup"
    fi
    
    cp -r "$SERVICE_PATH" "$BACKUP_DIR/$BACKUP_NAME"
    log "Backup created: $BACKUP_DIR/$BACKUP_NAME"
}

deploy() {
    log "Starting deployment to $ENVIRONMENT..."
    
    cd "$SERVICE_PATH"
    
    # Pull latest changes
    log "Pulling latest changes..."
    git pull origin main
    
    # Update dependencies
    log "Updating dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Run database migrations if needed
    log "Checking for database migrations..."
    if [ -f "migrations/migrate.py" ]; then
        python migrations/migrate.py
        log "Database migrations completed"
    fi
    
    # Start service
    log "Starting service..."
    systemctl start "$SERVICE_NAME"
    
    # Verify deployment
    log "Verifying deployment..."
    sleep 5
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log "✅ Deployment successful - service is running"
    else
        log "❌ Deployment failed - service not running"
        return 1
    fi
}

rollback() {
    log "Rolling back deployment..."
    
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        systemctl stop "$SERVICE_NAME"
        rm -rf "$SERVICE_PATH.rollback"
        mv "$SERVICE_PATH" "$SERVICE_PATH.rollback"
        cp -r "$BACKUP_DIR/$LATEST_BACKUP" "$SERVICE_PATH"
        systemctl start "$SERVICE_NAME"
        log "Rollback completed using backup: $LATEST_BACKUP"
    else
        log "No backup found for rollback"
        return 1
    fi
}

check_health() {
    log "Checking service health..."
    
    for i in {1..10}; do
        if curl -s -f http://localhost:8309/api/echo/health > /dev/null; then
            log "✅ Health check passed"
            return 0
        fi
        log "Health check attempt $i/10..."
        sleep 2
    done
    
    log "❌ Health check failed"
    return 1
}

main() {
    log "Echo Brain Deployment Script - Environment: $ENVIRONMENT"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        create_backup
    fi
    
    if deploy; then
        if check_health; then
            log "🎉 Deployment completed successfully!"
        else
            log "Health check failed - considering rollback"
            if [ "$ENVIRONMENT" = "production" ]; then
                rollback
            fi
        fi
    else
        log "Deployment failed"
        exit 1
    fi
}

main "$@"
