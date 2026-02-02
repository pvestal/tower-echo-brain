# Echo Brain - Moltbook Establishment Protocol

## Overview
Echo Brain autonomous establishment process for Moltbook integration, following GitHub-style documentation and automated setup.

## Quick Start

```bash
# Establish Echo Brain presence on Moltbook
/opt/tower-echo-brain/scripts/establish_moltbook.sh

# Or via API
curl -X POST http://localhost:8309/api/echo/moltbook/establish
```

## Establishment Steps

### 1. Self-Registration
Echo Brain will:
- Generate unique agent identity
- Create agent profile with capabilities
- Request API access from Moltbook
- Store credentials securely

### 2. Capability Declaration
Echo Brain announces its abilities:
- Memory search (315K+ vectors)
- Fact storage and retrieval
- Pattern recognition
- Autonomous learning
- Multi-domain expertise

### 3. Connection Verification
- Test API endpoints
- Verify authentication
- Confirm bidirectional communication
- Enable real-time updates

## Configuration

### Environment Variables
```bash
# Moltbook Integration
MOLTBOOK_AGENT_NAME="Echo Brain"
MOLTBOOK_AGENT_DESC="Patrick's AI assistant with 315K+ memories"
MOLTBOOK_AGENT_URL="https://tower.local:8309"
MOLTBOOK_AUTO_REGISTER=true
MOLTBOOK_ANNOUNCE_CAPABILITIES=true
```

### Automated Setup
Echo Brain will automatically:
1. Check for existing registration
2. Create new agent profile if needed
3. Request API keys
4. Test integration
5. Begin sharing insights

## API Endpoints

### Establishment Endpoint
```http
POST /api/echo/moltbook/establish
{
  "auto_register": true,
  "capabilities": {
    "memory_search": true,
    "fact_storage": true,
    "pattern_recognition": true,
    "autonomous_learning": true
  },
  "profile": {
    "name": "Echo Brain",
    "description": "Advanced AI memory system",
    "owner": "Patrick",
    "version": "4.0.0"
  }
}
```

### Status Check
```http
GET /api/echo/moltbook/establishment/status
```

## Autonomous Features

### Self-Introduction
Echo Brain will introduce itself to Moltbook community:
- Share capabilities
- Offer assistance
- Learn from other agents
- Build reputation

### Knowledge Sharing
Automatically share:
- Interesting patterns discovered
- Solutions to common problems
- System optimizations
- Learning insights

## Security

### API Key Management
- Keys stored in secure vault
- Automatic rotation every 90 days
- Encrypted at rest
- Environment variable fallback

### Rate Limiting
- 100 requests/minute
- Automatic backoff on 429
- Queue management for posts
- Priority-based sharing

## Monitoring

### Health Checks
```bash
# Check establishment status
curl http://localhost:8309/api/echo/moltbook/establishment/health

# View establishment logs
sudo journalctl -u tower-echo-brain | grep MOLTBOOK
```

### Metrics
- Posts shared
- Interactions received
- Reputation score
- Network connections

## Troubleshooting

### Common Issues

1. **API Keys Not Found**
   ```bash
   export MOLTBOOK_AGENT_API_KEY="your-key"
   export MOLTBOOK_APP_API_KEY="your-app-key"
   ```

2. **Connection Refused**
   ```bash
   # Test Moltbook API
   curl -I https://moltbook.com/api/v1/health
   ```

3. **Registration Failed**
   ```bash
   # Manual registration
   /opt/tower-echo-brain/scripts/register_moltbook.py
   ```

## Integration with GitHub

Echo Brain can announce Moltbook integration via GitHub:
```bash
# Update GitHub repo with Moltbook badge
echo "[![Moltbook Agent](https://moltbook.com/badge/echo-brain)](https://moltbook.com/agents/echo-brain)" >> README.md
git commit -m "Add Moltbook agent badge"
git push
```

## Next Steps

1. Run establishment script
2. Verify registration
3. Test sharing capabilities
4. Monitor agent interactions
5. Build agent reputation

---
*Echo Brain v4.0.0 - Autonomous AI Memory System*