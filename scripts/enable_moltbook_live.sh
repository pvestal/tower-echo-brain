#!/bin/bash
# Enable live Moltbook API mode
# Run this when you receive your API keys (after 48 hours approval)

echo "=== Enabling Live Moltbook API Mode ==="
echo

# Backup current .env
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)

# Update Moltbook configuration
echo "Updating .env with live API keys..."
sed -i 's/MOLTBOOK_DRY_RUN=true/MOLTBOOK_DRY_RUN=false/' .env
sed -i 's/MOLTBOOK_AGENT_API_KEY=YOUR_PENDING_AGENT_KEY_HERE/MOLTBOOK_AGENT_API_KEY=YOUR_ACTUAL_AGENT_KEY/' .env
sed -i 's/MOLTBOOK_APP_API_KEY=YOUR_PENDING_APP_KEY_HERE/MOLTBOOK_APP_API_KEY=YOUR_ACTUAL_APP_KEY/' .env

echo "✅ Updated .env with live mode settings"
echo
echo "Please edit .env to replace placeholder keys with your actual API keys:"
echo "1. MOLTBOOK_AGENT_API_KEY=your_actual_agent_key"
echo "2. MOLTBOOK_APP_API_KEY=your_actual_app_key"
echo
echo "After updating keys, restart Echo Brain:"
echo "  sudo systemctl restart tower-echo-brain"
echo
echo "Test the live integration:"
echo "  curl -s http://localhost:8309/api/echo/moltbook/status | python3 -m json.tool"
