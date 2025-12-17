#!/bin/bash
"""
Install script for Echo Brain Learning Pipeline systemd service and timer.
Replaces the broken cron job with a proper systemd timer.
"""

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="echo-learning-pipeline"

echo "Installing Echo Brain Learning Pipeline systemd service and timer..."

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]] && [[ -z "$SUDO_USER" ]]; then
    echo "Error: Please run with sudo, not as root directly"
    exit 1
fi

# Ensure we have the service files
if [[ ! -f "$SCRIPT_DIR/${SERVICE_NAME}.service" ]]; then
    echo "Error: Service file not found: $SCRIPT_DIR/${SERVICE_NAME}.service"
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/${SERVICE_NAME}.timer" ]]; then
    echo "Error: Timer file not found: $SCRIPT_DIR/${SERVICE_NAME}.timer"
    exit 1
fi

# Create log directory
echo "Creating log directory..."
sudo mkdir -p /opt/tower-echo-brain/logs
sudo chown patrick:patrick /opt/tower-echo-brain/logs

# Install service and timer files
echo "Installing systemd files..."
sudo cp "$SCRIPT_DIR/${SERVICE_NAME}.service" "/etc/systemd/system/"
sudo cp "$SCRIPT_DIR/${SERVICE_NAME}.timer" "/etc/systemd/system/"

# Set correct permissions
sudo chmod 644 "/etc/systemd/system/${SERVICE_NAME}.service"
sudo chmod 644 "/etc/systemd/system/${SERVICE_NAME}.timer"

# Reload systemd
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable and start the timer
echo "Enabling and starting the timer..."
sudo systemctl enable "${SERVICE_NAME}.timer"
sudo systemctl start "${SERVICE_NAME}.timer"

# Check status
echo "Checking timer status..."
sudo systemctl status "${SERVICE_NAME}.timer" --no-pager -l

# Show next run times
echo ""
echo "Timer schedule:"
sudo systemctl list-timers "${SERVICE_NAME}.timer" --no-pager

# Test the service (dry run)
echo ""
echo "Testing the service (dry run)..."
if sudo systemctl start "${SERVICE_NAME}.service" --no-block; then
    echo "Service test initiated. Check logs with:"
    echo "  sudo journalctl -u ${SERVICE_NAME}.service -f"
else
    echo "Warning: Service test failed. Check configuration."
fi

# Show useful commands
echo ""
echo "Installation complete! Useful commands:"
echo ""
echo "  # Check timer status:"
echo "  sudo systemctl status ${SERVICE_NAME}.timer"
echo ""
echo "  # View logs:"
echo "  sudo journalctl -u ${SERVICE_NAME}.service -f"
echo ""
echo "  # Manual run:"
echo "  sudo systemctl start ${SERVICE_NAME}.service"
echo ""
echo "  # Stop/disable timer:"
echo "  sudo systemctl stop ${SERVICE_NAME}.timer"
echo "  sudo systemctl disable ${SERVICE_NAME}.timer"
echo ""
echo "  # Check broken cron job (should be removed):"
echo "  crontab -l | grep echo_learning_pipeline"

# Check for broken cron job
echo ""
echo "Checking for broken cron job..."
if crontab -l 2>/dev/null | grep -q "echo_learning_pipeline"; then
    echo "⚠️  WARNING: Old cron job still exists!"
    echo "   Remove it with: crontab -e"
    echo "   Look for lines containing 'echo_learning_pipeline'"
else
    echo "✅ No old cron job found"
fi

echo ""
echo "🎉 Echo Brain Learning Pipeline systemd timer is now active!"
echo "   It will run every 6 hours to process new conversations and KB articles."