#!/bin/bash
# Database Credentials Vault Migration Script
# Usage: Run this script with Vault token to migrate database password to Vault

export VAULT_ADDR='http://127.0.0.1:8200'

echo "=== Database Credentials Vault Migration ==="
echo ""

# Check if Vault token is provided
if [ -z "$VAULT_TOKEN" ]; then
    echo "ERROR: VAULT_TOKEN environment variable not set"
    echo "Please run: export VAULT_TOKEN='your-vault-token'"
    exit 1
fi

# Read current database password from .env
DB_PASSWORD=$(grep '^DATABASE_URL' /opt/tower-echo-brain/.env | cut -d':' -f3 | cut -d'@' -f1)

if [ -z "$DB_PASSWORD" ]; then
    echo "ERROR: Could not extract password from .env"
    exit 1
fi

echo "✅ Found database password in .env"
echo ""

# Store in Vault
echo "📦 Storing database credentials in Vault..."
vault kv put secret/tower/database \
    host=localhost \
    database=echo_brain \
    user=patrick \
    password="$DB_PASSWORD"

if [ $? -eq 0 ]; then
    echo "✅ Successfully stored credentials in Vault"
    echo ""
    echo "Next steps:"
    echo "1. Update database.py to read from Vault"
    echo "2. Test database connection"
    echo "3. Remove password from .env file"
else
    echo "❌ Failed to store credentials in Vault"
    exit 1
fi
