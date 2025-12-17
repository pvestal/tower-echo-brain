#!/bin/bash
"""
Creates systemd timer to replace broken cron job for Echo Brain Learning Pipeline.

This script:
1. Removes the broken cron job pointing to non-existent file
2. Creates systemd service and timer for reliable execution
3. Enables and starts the timer for automatic learning pipeline execution
"""

set -e

# Configuration
SERVICE_NAME="echo-learning-pipeline"
TIMER_NAME="echo-learning-pipeline.timer"
SCRIPT_PATH="/opt/tower-echo-brain/services/learning_pipeline/scripts/run_pipeline.py"
CONFIG_PATH="/opt/tower-echo-brain/services/learning_pipeline/config/production.yaml"
LOG_PATH="/opt/tower-echo-brain/logs/learning_pipeline.log"

echo "Setting up Echo Brain Learning Pipeline systemd timer..."

# Ensure log directory exists
sudo mkdir -p "$(dirname "$LOG_PATH")"
sudo chown patrick:patrick "$(dirname "$LOG_PATH")"

# Make script executable
sudo chmod +x "$SCRIPT_PATH"

# Remove broken cron job
echo "Removing broken cron job..."
(crontab -l 2>/dev/null | grep -v "echo_learning_pipeline.py" || true) | crontab -

# Create systemd service file
echo "Creating systemd service file..."
sudo tee "/etc/systemd/system/$SERVICE_NAME.service" > /dev/null << EOF
[Unit]
Description=Echo Brain Learning Pipeline
Documentation=https://github.com/your-repo/echo-brain
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=oneshot
User=patrick
Group=patrick
WorkingDirectory=/opt/tower-echo-brain/services/learning_pipeline
Environment=PYTHONPATH=/opt/tower-echo-brain/services/learning_pipeline/src
Environment=ECHO_BRAIN_DB_PASSWORD=***REMOVED***
ExecStart=/usr/bin/python3 $SCRIPT_PATH --config $CONFIG_PATH --log-level INFO
StandardOutput=append:$LOG_PATH
StandardError=append:$LOG_PATH
TimeoutSec=3600
Restart=no

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/tower-echo-brain/logs /tmp/claude /home/patrick/.claude/conversations
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
MemoryDenyWriteExecute=false
RestrictRealtime=true
RestrictNamespaces=true
LockPersonality=true

[Install]
WantedBy=multi-user.target
EOF

# Create systemd timer file
echo "Creating systemd timer file..."
sudo tee "/etc/systemd/system/$TIMER_NAME" > /dev/null << EOF
[Unit]
Description=Run Echo Brain Learning Pipeline every 6 hours
Requires=$SERVICE_NAME.service

[Timer]
# Run every 6 hours, offset by 30 minutes to avoid conflicts
OnCalendar=00/6:30:00
RandomizedDelaySec=900
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Reload systemd configuration
echo "Reloading systemd configuration..."
sudo systemctl daemon-reload

# Enable and start the timer
echo "Enabling and starting the timer..."
sudo systemctl enable "$TIMER_NAME"
sudo systemctl start "$TIMER_NAME"

# Create manual execution script
echo "Creating manual execution script..."
sudo tee "/opt/tower-echo-brain/scripts/run_learning_pipeline_manual.sh" > /dev/null << 'EOF'
#!/bin/bash
# Manual execution script for Echo Brain Learning Pipeline

set -e

SCRIPT_PATH="/opt/tower-echo-brain/services/learning_pipeline/scripts/run_pipeline.py"
CONFIG_PATH="/opt/tower-echo-brain/services/learning_pipeline/config/production.yaml"

echo "Running Echo Brain Learning Pipeline manually..."
echo "Started at: $(date)"

cd /opt/tower-echo-brain/services/learning_pipeline
export PYTHONPATH=/opt/tower-echo-brain/services/learning_pipeline/src
export ECHO_BRAIN_DB_PASSWORD=***REMOVED***

python3 "$SCRIPT_PATH" --config "$CONFIG_PATH" --log-level INFO "$@"

echo "Completed at: $(date)"
EOF

sudo chmod +x "/opt/tower-echo-brain/scripts/run_learning_pipeline_manual.sh"
sudo chown patrick:patrick "/opt/tower-echo-brain/scripts/run_learning_pipeline_manual.sh"

# Create health check script
echo "Creating health check script..."
sudo tee "/opt/tower-echo-brain/scripts/check_learning_pipeline_health.sh" > /dev/null << 'EOF'
#!/bin/bash
# Health check script for Echo Brain Learning Pipeline

set -e

SCRIPT_PATH="/opt/tower-echo-brain/services/learning_pipeline/scripts/run_pipeline.py"
CONFIG_PATH="/opt/tower-echo-brain/services/learning_pipeline/config/production.yaml"

echo "Echo Brain Learning Pipeline Health Check"
echo "========================================"

cd /opt/tower-echo-brain/services/learning_pipeline
export PYTHONPATH=/opt/tower-echo-brain/services/learning_pipeline/src
export ECHO_BRAIN_DB_PASSWORD=***REMOVED***

# Check dependencies
echo "Checking dependencies..."
python3 "$SCRIPT_PATH" --config "$CONFIG_PATH" --check-deps

echo ""

# Check health
echo "Performing health check..."
python3 "$SCRIPT_PATH" --config "$CONFIG_PATH" --health-check

echo ""

# Check timer status
echo "Systemd timer status:"
systemctl status echo-learning-pipeline.timer --no-pager

echo ""

# Check recent logs
echo "Recent pipeline logs (last 20 lines):"
tail -20 /opt/tower-echo-brain/logs/learning_pipeline.log 2>/dev/null || echo "No logs found"
EOF

sudo chmod +x "/opt/tower-echo-brain/scripts/check_learning_pipeline_health.sh"
sudo chown patrick:patrick "/opt/tower-echo-brain/scripts/check_learning_pipeline_health.sh"

# Show status
echo ""
echo "Echo Brain Learning Pipeline systemd timer setup completed!"
echo ""
echo "Service status:"
sudo systemctl status "$SERVICE_NAME.service" --no-pager --lines=0 || true
echo ""
echo "Timer status:"
sudo systemctl status "$TIMER_NAME" --no-pager
echo ""
echo "Next scheduled runs:"
systemctl list-timers "$TIMER_NAME" --no-pager
echo ""
echo "Manual execution: /opt/tower-echo-brain/scripts/run_learning_pipeline_manual.sh"
echo "Health check: /opt/tower-echo-brain/scripts/check_learning_pipeline_health.sh"
echo "Logs: tail -f $LOG_PATH"
echo ""
echo "To run immediately: sudo systemctl start $SERVICE_NAME.service"
echo "To stop timer: sudo systemctl stop $TIMER_NAME"
echo "To disable timer: sudo systemctl disable $TIMER_NAME"