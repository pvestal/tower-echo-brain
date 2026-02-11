#!/bin/bash
# Echo Brain Ingestion Health Check
# Monitors the echo-brain-ingest service and sends alerts on failures

set -euo pipefail

SERVICE_NAME="echo-brain-ingest"
LOG_FILE="/var/log/echo-brain-health.log"
ALERT_FILE="/tmp/echo-brain-alert-sent"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | sudo tee -a "$LOG_FILE"
}

check_service_health() {
    local failed=false

    # Check if service has failed
    if systemctl is-failed "$SERVICE_NAME" >/dev/null 2>&1; then
        failed=true
        log_message "CRITICAL: $SERVICE_NAME service has failed"
    fi

    # Check last run time (should run daily)
    local last_run=$(systemctl show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value)
    if [[ -n "$last_run" && "$last_run" != "n/a" ]]; then
        local last_run_epoch=$(date -d "$last_run" +%s)
        local current_epoch=$(date +%s)
        local hours_since=$((($current_epoch - $last_run_epoch) / 3600))

        if [[ $hours_since -gt 30 ]]; then  # Alert if no run in 30+ hours
            failed=true
            log_message "CRITICAL: $SERVICE_NAME last ran $hours_since hours ago"
        fi
    fi

    # Send alert if failed and haven't sent one recently
    if $failed && [[ ! -f "$ALERT_FILE" || $(find "$ALERT_FILE" -mmin +60 2>/dev/null) ]]; then
        send_alert
        touch "$ALERT_FILE"
    elif ! $failed && [[ -f "$ALERT_FILE" ]]; then
        log_message "INFO: $SERVICE_NAME is healthy again"
        rm -f "$ALERT_FILE"
    fi
}

send_alert() {
    local status=$(systemctl status "$SERVICE_NAME" --no-pager -l)
    log_message "ALERT: Sending notification about $SERVICE_NAME failure"

    # Log to Echo Brain notifications (if service is running)
    if systemctl is-active tower-echo-brain >/dev/null 2>&1; then
        curl -s -X POST http://localhost:8309/api/echo/notifications \
            -H "Content-Type: application/json" \
            -d "{\"type\": \"system_alert\", \"title\": \"Echo Brain Ingestion Failed\", \"message\": \"The $SERVICE_NAME service has failed. Check logs for details. Status: ${status:0:200}...\", \"priority\": \"high\"}" \
            || true
    fi

    # Also check Vault token expiry as a potential cause
    if command -v vault >/dev/null 2>&1; then
        export VAULT_ADDR="http://127.0.0.1:8200"
        if ! vault token lookup >/dev/null 2>&1; then
            log_message "WARNING: Vault token may be expired or invalid"
        fi
    fi
}

# Main execution
check_service_health