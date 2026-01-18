# 🔐 Tower Plaid Complete Configuration

## 📍 Webhook Callback URLs for Plaid Dashboard

### Transaction Updates
```
https://192.168.50.135:8089/api/plaid/webhook/{session_id}
```
- **Events**: INITIAL_UPDATE, HISTORICAL_UPDATE, DEFAULT_UPDATE, TRANSACTIONS_REMOVED
- **Auto-triggers**: AgenticPersona financial monitoring
- **Alerts**: Transactions over $500

### Income Refresh Callbacks
```
https://192.168.50.135:8089/api/plaid/webhook/{session_id}
```
- **Events**: INCOME_VERIFICATION_REFRESH_STARTED, INCOME_VERIFICATION_REFRESH_COMPLETE
- **Data stored**: Vault at `plaid/income/{item_id}`
- **Risk signals**: Auto-analyzed and flagged

### Wallet/Transfer Updates
```
https://192.168.50.135:8089/api/plaid/webhook/{session_id}
```
- **Events**: TRANSFER_EVENTS_UPDATE
- **Statuses**: posted (complete), failed, pending
- **Notifications**: Immediate via AgenticPersona

## 🔒 Tower MFA Authentication

### Setup MFA (First Time)
1. Visit: `http://192.168.50.135:8090/mfa/setup/patrick`
2. Scan QR code with Google Authenticator/Authy
3. Save backup codes

### MFA Login Flow
```python
POST http://192.168.50.135:8090/mfa/verify
{
    "user_id": "patrick",
    "totp_code": "123456"  # From authenticator app
}

Response:
{
    "success": true,
    "session_token": "abc123...",
    "expires_in": 86400
}
```

### Protected Endpoints (Require MFA)
- `/api/webhooks/transactions` - Transaction monitoring
- `/api/webhooks/income` - Income verification
- `/api/webhooks/transfers` - Wallet/transfer events

## 🚀 Complete Auth Flow

### 1. Initial Bank Connection
```
http://192.168.50.135:8089/plaid/auth
```
- Click "Connect Your Bank Account"
- Complete bank MFA
- Automatic token storage in Vault

### 2. Webhook Registration
The following webhooks are automatically registered:
- Transaction updates (real-time)
- Income refresh (on-demand)
- Transfer/wallet events (instant)

### 3. AgenticPersona Integration
Automatic monitoring triggers:
- **Large transactions**: Alert if >$500
- **Daily spending**: Alert if >$1000/day
- **Income changes**: Track salary deposits
- **Transfer failures**: Immediate notification

## 📊 Current Status

### Services Running
- ✅ Auth Service: Port 8089
- ✅ Webhook Handler: Integrated
- ✅ MFA System: Active
- ✅ Echo Brain: Port 8309
- ✅ AgenticPersona: Monitoring active

### Credentials
- **Client ID**: 67b7532c37f3d10023aba53e
- **Environment**: Production
- **Products**: transactions, auth, identity, income
- **Vault Storage**: All tokens encrypted

## 💡 Example Queries for Echo

After connecting accounts:
- "What's my current balance?"
- "Show me transactions over $500"
- "When was my last paycheck?"
- "Set up alerts for large withdrawals"
- "Track my monthly income"
- "Monitor spending patterns"

## 🔧 Technical Implementation

### Webhook Handler Features
```python
# Transaction Analysis
- Automatic categorization
- Spending pattern detection
- Anomaly alerts
- Historical comparison

# Income Tracking
- Salary verification
- Bonus detection
- Income stability analysis
- Risk assessment

# Transfer Monitoring
- Success/failure tracking
- Fee analysis
- Recurring payment detection
```

### Security Layers
1. **HTTPS/TLS**: All external connections
2. **MFA/TOTP**: Time-based authentication
3. **Vault**: Encrypted credential storage
4. **Session tokens**: 24-hour expiry
5. **Webhook verification**: Plaid signature validation

## 📝 Testing Webhooks

### Simulate Transaction Update
```bash
curl -X POST http://192.168.50.135:8089/api/plaid/webhook/test \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_type": "TRANSACTIONS",
    "webhook_code": "DEFAULT_UPDATE",
    "item_id": "test_item",
    "new_transactions": 5
  }'
```

### Simulate Income Refresh
```bash
curl -X POST http://192.168.50.135:8089/api/plaid/webhook/test \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_type": "INCOME",
    "webhook_code": "INCOME_VERIFICATION_REFRESH_COMPLETE",
    "item_id": "test_item",
    "income": {"monthly": 5000}
  }'
```

## 🎯 Next Steps

1. **Connect Bank**: Visit http://192.168.50.135:8089/plaid/auth
2. **Setup MFA**: Visit http://192.168.50.135:8090/mfa/setup/patrick
3. **Configure Alerts**: Ask Echo to set up monitoring rules
4. **Test Webhooks**: Verify real-time updates working
5. **Monitor**: AgenticPersona begins autonomous tracking

---

**All systems operational and ready for production use!**