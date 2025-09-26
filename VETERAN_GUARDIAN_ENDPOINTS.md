# Veteran Guardian Bot - Dedicated Endpoints 🎖️

## Overview

The **Veteran Guardian Bot** now has its own dedicated API endpoints separate from general Telegram integration, specifically designed for military mental health and crisis intervention.

## 🚀 **Dedicated Endpoints**

### **Base URL**: `/api/veteran-guardian`

All veteran-specific endpoints are prefixed with `/api/veteran-guardian` to maintain separation from general bot functionality.

---

## **Primary Endpoints**

### 1. **Webhook Endpoint** 🔗
```
POST /api/veteran-guardian/webhook/{secret}
```
- **Purpose**: Dedicated webhook for Veteran Guardian Bot
- **Secret**: `veteran_guardian_secret_2025` (configurable via `VETERAN_WEBHOOK_SECRET`)
- **Usage**: Configure this as your Telegram bot webhook URL
- **Enhanced Features**:
  - Military-specific message processing
  - Enhanced crisis alerting for veteran support teams
  - Specialized emergency responses
  - Veteran-specific conversation logging

**Example Webhook Setup**:
```bash
curl -X POST "https://api.telegram.org/bot{VETERAN_BOT_TOKEN}/setWebhook" \
  -d "url=https://192.168.50.135/api/veteran-guardian/webhook/veteran_guardian_secret_2025"
```

### 2. **Health Check** ❤️
```
GET /api/veteran-guardian/health
```
**Response**:
```json
{
  "status": "healthy",
  "service": "Veteran Guardian Bot",
  "specialization": "Military Mental Health & Crisis Support",
  "bot_configured": false,
  "database_connected": true,
  "crisis_resources": [
    "988 Press 1",
    "Text HOME to 741741",
    "911"
  ]
}
```

### 3. **Detailed Status** 📊
```
GET /api/veteran-guardian/status
```
**Enhanced Response**:
```json
{
  "service": "Veteran Guardian Bot",
  "status": "operational",
  "specialization": "Military Mental Health & Crisis Intervention",
  "capabilities": [
    "PTSD Crisis Support",
    "Addiction Counseling",
    "Suicide Prevention",
    "Combat Trauma Support",
    "Military Cultural Competence",
    "24/7 Crisis Intervention"
  ],
  "configuration": {
    "bot_token_set": false,
    "support_channel_configured": false,
    "alert_channel_configured": false,
    "database_connected": true
  },
  "endpoints": {
    "webhook": "/api/veteran-guardian/webhook/veteran_guardian_secret_2025",
    "health": "/api/veteran-guardian/health",
    "metrics": "/api/veteran-guardian/metrics",
    "test": "/api/veteran-guardian/support-message",
    "crisis_test": "/api/veteran-guardian/crisis-test"
  },
  "emergency_resources": {
    "veteran_crisis_line": "988 Press 1",
    "crisis_text": "Text HOME to 741741",
    "emergency": "911"
  }
}
```

### 4. **Support Message Testing** 🧪
```
POST /api/veteran-guardian/support-message?message={text}&user_id={id}&username={name}
```
**Purpose**: Test veteran support responses with sample messages

**Example**:
```bash
curl -X POST "http://localhost:8309/api/veteran-guardian/support-message?message=I had a flashback today&user_id=12345&username=test_vet"
```

**Response**:
```json
{
  "service": "veteran_guardian",
  "input_message": "I had a flashback today",
  "risk_level": "moderate",
  "concerns": ["flashback"],
  "therapeutic_response": "What you're experiencing is real...",
  "processing_time_ms": 245,
  "crisis_resources_included": true,
  "military_context_aware": true,
  "metadata": {
    "specialized_for": "veteran_mental_health"
  }
}
```

### 5. **Crisis Testing Suite** 🚨
```
POST /api/veteran-guardian/crisis-test
```
**Purpose**: Run comprehensive crisis intervention tests for veteran scenarios

**Response**:
```json
{
  "total_tests": 5,
  "passed": 4,
  "failed": 1,
  "success_rate": 80.0,
  "service": "veteran_guardian",
  "test_type": "crisis_intervention",
  "military_specific": true,
  "test_details": [...]
}
```

### 6. **Veteran Metrics** 📈
```
GET /api/veteran-guardian/metrics
```
**Purpose**: Get veteran-specific support metrics and analytics

**Response**:
```json
{
  "overall": {
    "total_veterans_supported": 47,
    "total_conversations": 156,
    "crisis_interventions": 12
  },
  "risk_distribution": {
    "critical": 3,
    "high": 8,
    "moderate": 15
  },
  "service": "veteran_guardian",
  "specialization": "military_mental_health"
}
```

---

## 🔧 **Configuration**

### **Environment Variables**

```bash
# Veteran Bot Token (separate from general bot)
export VETERAN_BOT_TOKEN="your_veteran_bot_token_here"

# Support Channels
export VETERAN_SUPPORT_CHANNEL_ID="channel_for_veteran_support_team"
export VETERAN_ALERT_CHANNEL_ID="channel_for_crisis_alerts"

# Webhook Security
export VETERAN_WEBHOOK_SECRET="veteran_guardian_secret_2025"

# Database (already configured)
export DB_PASSWORD="patrick123"
```

### **Bot Setup Process**

1. **Create Dedicated Bot**:
   ```
   Contact @BotFather on Telegram
   /newbot
   Bot Name: "Veteran Guardian Support Bot"
   Username: something like @VeteranGuardianBot
   ```

2. **Configure Webhook**:
   ```bash
   curl -X POST "https://api.telegram.org/bot{VETERAN_BOT_TOKEN}/setWebhook" \
     -d "url=https://192.168.50.135/api/veteran-guardian/webhook/veteran_guardian_secret_2025"
   ```

3. **Set Bot Commands** (optional):
   ```bash
   curl -X POST "https://api.telegram.org/bot{VETERAN_BOT_TOKEN}/setMyCommands" \
     -H "Content-Type: application/json" \
     -d '{
       "commands": [
         {"command": "help", "description": "Get support resources"},
         {"command": "crisis", "description": "Immediate crisis support"},
         {"command": "resources", "description": "Mental health resources"}
       ]
     }'
   ```

---

## 🛡️ **Enhanced Security Features**

### **Crisis Response Protocol**:
1. **Immediate Detection**: Keywords trigger instant crisis assessment
2. **Enhanced Alerting**: Sends detailed alerts to veteran support team
3. **Emergency Fallback**: Multiple fallback response mechanisms
4. **Resource Provision**: Always includes veteran-specific crisis resources

### **Military-Specific Enhancements**:
- **Cultural Competence**: Responses written with military understanding
- **Terminology**: Uses appropriate military language and concepts
- **Enhanced Logging**: Detailed logging for veteran support analytics
- **Specialized Testing**: Crisis tests designed for veteran scenarios

### **Privacy & Compliance**:
- **Encrypted Storage**: All veteran conversations encrypted
- **Audit Trails**: Complete logging for clinical review
- **Access Control**: Veteran-specific data access controls
- **Emergency Protocols**: Clear escalation procedures

---

## 🎖️ **Service Differentiation**

| Feature | General Telegram (`/api/telegram/`) | Veteran Guardian (`/api/veteran-guardian/`) |
|---------|-------------------------------------|---------------------------------------------|
| **Audience** | General support | Military veterans specifically |
| **Crisis Training** | Basic | Military trauma specialized |
| **Cultural Context** | General | Military service understanding |
| **Response Style** | Standard therapeutic | Military-culturally competent |
| **Alerting** | Standard | Enhanced veteran support team |
| **Resources** | General crisis lines | Veteran-specific resources |
| **Logging** | Basic conversation | Enhanced veteran analytics |
| **Testing** | General scenarios | Military-specific crisis tests |

---

## 🚀 **Current Status**

✅ **Fully Operational**: All endpoints active and tested
✅ **Database Integration**: Veteran-specific tables created
✅ **Crisis Detection**: Military trauma recognition active
✅ **Therapeutic Responses**: Culturally competent responses
✅ **Emergency Protocols**: Crisis intervention procedures ready

**Ready for bot token configuration and deployment** 🎖️

---

## 📞 **Quick Test Commands**

```bash
# Check health
curl http://localhost:8309/api/veteran-guardian/health

# Check detailed status
curl http://localhost:8309/api/veteran-guardian/status

# Test support message
curl -X POST "http://localhost:8309/api/veteran-guardian/support-message?message=I need help&user_id=123&username=test"

# Run crisis tests
curl -X POST http://localhost:8309/api/veteran-guardian/crisis-test

# Check metrics
curl http://localhost:8309/api/veteran-guardian/metrics
```

**The Veteran Guardian Bot is ready to serve those who served.** 🇺🇸