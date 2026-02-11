#!/bin/bash
# Fetch database password from HashiCorp Vault
# This script is sourced by other scripts to set DB_PASSWORD environment variable

set -euo pipefail

export VAULT_ADDR="http://127.0.0.1:8200"

# Use existing VAULT_TOKEN environment variable (set by systemd service)
if [[ -z "${VAULT_TOKEN:-}" ]]; then
    echo "ERROR: VAULT_TOKEN environment variable not set." >&2
    echo "For manual runs: export VAULT_TOKEN=\$(vault print token)" >&2
    exit 1
fi

# Check if Vault is available
if ! curl -s "$VAULT_ADDR/v1/sys/health" >/dev/null 2>&1; then
    echo "ERROR: Vault is not available at $VAULT_ADDR" >&2
    exit 1
fi

# Fetch credentials from Vault
if vault_response=$(vault kv get -field=password secret/echo-brain/database 2>/dev/null); then
    export DB_PASSWORD="$vault_response"
    export PGPASSWORD="$vault_response"
else
    echo "ERROR: Failed to fetch database password from Vault" >&2
    echo "Ensure you're authenticated: vault login" >&2
    exit 1
fi