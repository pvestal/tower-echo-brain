# Echo Brain Enterprise Integrations - Deployment Guide

## Overview

This document outlines the deployment strategy for Echo Brain's new enterprise integrations: Google Calendar, Home Assistant, and unified notification services.

## =Ë Pre-Deployment Checklist

### Infrastructure Requirements
- [ ] PostgreSQL 15+ running and accessible
- [ ] Redis 7+ for caching and session management
- [ ] Python 3.12+ environment
- [ ] Node.js 18+ for frontend builds
- [ ] Sufficient disk space for logs and artifacts

### Service Dependencies
- [ ] Echo Brain core services running (port 8309)
- [ ] Tower Dashboard accessible (port 8080)
- [ ] Knowledge Base service running (port 8307)
- [ ] Database migrations completed

### Security Validation
- [ ] All secrets properly stored in Vault or environment variables
- [ ] No hardcoded credentials in configuration files
- [ ] SSL/TLS certificates valid and current
- [ ] Network security groups configured

## =€ Deployment Process

### Step 1: Backup Current System
```bash
# Create backup of current Echo Brain state
cd /opt/tower-echo-brain
sudo systemctl stop tower-echo-brain
cp -r /opt/tower-echo-brain /opt/tower-echo-brain.backup.$(date +%Y%m%d_%H%M%S)

# Backup database
pg_dump -h ***REMOVED*** -U patrick -d tower_consolidated > echo_brain_backup_$(date +%Y%m%d_%H%M%S).sql
```

### Step 2: Merge Integration Branch
```bash
# Switch to main branch
git checkout main
git pull origin main

# Merge enterprise integrations
git merge feature/enterprise-integrations

# Push merged changes
git push origin main
```

### Step 3: Install Dependencies
```bash
# Install new Python dependencies
pip install -r requirements.txt

# Install new Node.js dependencies
npm install

# Update system packages if needed
sudo apt update
sudo apt upgrade -y
```

### Step 4: Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Configure integration settings
cat >> .env << 'EOF'
# Google Calendar Integration
GOOGLE_CALENDAR_ENABLED=true
GOOGLE_CALENDAR_CREDENTIALS_PATH=/opt/tower-credentials/google_calendar_credentials.json

# Home Assistant Integration
HOME_ASSISTANT_ENABLED=true
HOME_ASSISTANT_URL=http://home-assistant.local:8123
HOME_ASSISTANT_TOKEN=your_ha_token_here

# Notification Services
NTFY_SERVER_URL=https://ntfy.sh
NTFY_DEFAULT_TOPIC=echo-brain-notifications
NTFY_AUTH_TOKEN=your_ntfy_token_here

# Telegram Integration (existing)
TELEGRAM_BOT_TOKEN=your_telegram_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
EOF
```

### Step 5: Database Migrations
```bash
# Run any required database migrations
python scripts/migrate_database.py

# Verify database integrity
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://patrick:***REMOVED***@***REMOVED***/tower_consolidated')
cur = conn.cursor()
cur.execute('SELECT version();')
print('Database connection successful:', cur.fetchone())
conn.close()
"
```

### Step 6: Integration Testing
```bash
# Run comprehensive integration tests
python tests/test_integration_pipeline.py

# Test individual services
curl http://localhost:8309/api/calendar/status
curl http://localhost:8309/api/home/status
curl http://localhost:8309/api/notifications/status
```

### Step 7: Service Restart
```bash
# Restart Echo Brain with new integrations
sudo systemctl start tower-echo-brain
sudo systemctl status tower-echo-brain

# Check logs for successful startup
tail -f /opt/tower-echo-brain/logs/echo_brain.log
```

## =' Configuration Management

### Google Calendar Setup
1. Create Google Cloud Project
2. Enable Calendar API
3. Create service account credentials
4. Download credentials JSON file
5. Store in `/opt/tower-credentials/google_calendar_credentials.json`

### Home Assistant Setup
1. Generate long-lived access token in Home Assistant
2. Configure Home Assistant URL in environment
3. Test connection: `curl -H "Authorization: Bearer TOKEN" http://HA_URL/api/`

### Notification Services Setup
1. **NTFY**: Register topic at https://ntfy.sh or setup self-hosted server
2. **Telegram**: Existing bot token and chat ID
3. **Email**: Configure SMTP settings in Vault

## =Ê Monitoring and Validation

### Health Checks
```bash
# API Health Checks
curl http://localhost:8309/api/health
curl http://localhost:8309/api/calendar/status
curl http://localhost:8309/api/home/status
curl http://localhost:8309/api/notifications/status

# Service Status
sudo systemctl status tower-echo-brain
sudo systemctl status tower-dashboard
sudo systemctl status tower-kb
```

### Performance Monitoring
```bash
# Check memory usage
ps aux | grep echo
free -h

# Check disk usage
df -h /opt/tower-echo-brain

# Check log file sizes
du -sh /opt/tower-echo-brain/logs/*
```

### Integration Validation
```bash
# Test Google Calendar integration
curl -X POST http://localhost:8309/api/calendar/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my events today?"}'

# Test Home Assistant integration
curl -X POST http://localhost:8309/api/home/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all lights"}'

# Test notification service
curl -X POST http://localhost:8309/api/notifications/send \
  -H "Content-Type: application/json" \
  -d '{"message": "Echo Brain integration test", "type": "info"}'
```

## =¨ Rollback Procedure

If issues occur during deployment:

### Quick Rollback
```bash
# Stop services
sudo systemctl stop tower-echo-brain

# Restore backup
sudo rm -rf /opt/tower-echo-brain
sudo mv /opt/tower-echo-brain.backup.TIMESTAMP /opt/tower-echo-brain

# Restore database if needed
psql -h ***REMOVED*** -U patrick -d tower_consolidated < echo_brain_backup_TIMESTAMP.sql

# Restart services
sudo systemctl start tower-echo-brain
```

### Git Rollback
```bash
# Revert to previous commit
git checkout main
git revert HEAD --no-edit
git push origin main

# Restart services
sudo systemctl restart tower-echo-brain
```

## =È Post-Deployment Tasks

1. **Update Documentation**: Ensure all integration endpoints are documented
2. **Training**: Update user guides with new integration capabilities
3. **Monitoring Setup**: Configure alerts for new service failures
4. **Performance Baseline**: Establish performance metrics for new integrations
5. **Security Review**: Conduct security audit of new endpoints

## = Troubleshooting

### Common Issues

#### Integration Import Errors
```bash
# Check Python path and dependencies
python -c "import sys; print('\n'.join(sys.path))"
pip list | grep -E "(fastapi|aiohttp|google|psycopg2)"
```

#### Database Connection Issues
```bash
# Test database connectivity
pg_isready -h ***REMOVED*** -U patrick
psql -h ***REMOVED*** -U patrick -d tower_consolidated -c "SELECT 1;"
```

#### Service Startup Failures
```bash
# Check detailed service logs
journalctl -u tower-echo-brain -f
tail -f /opt/tower-echo-brain/logs/echo_brain.log
```

#### Permission Issues
```bash
# Fix ownership
sudo chown -R echo:echo /opt/tower-echo-brain
sudo chmod +x /opt/tower-echo-brain/scripts/*.sh
```

### Integration-Specific Issues

#### Google Calendar API Errors
- Verify credentials file exists and is readable
- Check Google Cloud project has Calendar API enabled
- Validate service account has proper permissions

#### Home Assistant Connection Errors
- Verify Home Assistant is accessible from Echo Brain server
- Check token validity and permissions
- Ensure Home Assistant API is enabled

#### Notification Delivery Failures
- Test each notification channel individually
- Verify API tokens and endpoints
- Check network connectivity to external services

## =Þ Support Contacts

- **Infrastructure**: System Administrator
- **Database**: DBA Team
- **Integrations**: Development Team
- **Security**: Security Team

## =Ú Additional Resources

- [Echo Brain API Documentation](http://***REMOVED***:8307/api/docs)
- [Google Calendar API Documentation](https://developers.google.com/calendar/api)
- [Home Assistant API Documentation](https://developers.home-assistant.io/docs/api/rest/)
- [NTFY Documentation](https://docs.ntfy.sh/)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-18
**Next Review**: 2026-02-18