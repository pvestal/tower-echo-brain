#!/bin/bash

echo "🔐 VAULT FAMILY SECRETS STRUCTURE"
echo "=================================="

# This script sets up HashiCorp Vault for multi-user family secrets management

export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='***REMOVED***'

# ================================
# ENABLE ENGINES & AUTH METHODS
# ================================

echo "📦 Setting up Vault engines..."

# Enable userpass auth for family members
vault auth enable userpass

# Enable AppRole for Echo Brain service authentication
vault auth enable approle

# Enable KV v2 for secrets
vault secrets enable -path=family kv-v2
vault secrets enable -path=users kv-v2
vault secrets enable -path=shared kv-v2

# ================================
# CREATE VAULT POLICIES
# ================================

echo "📋 Creating access policies..."

# Admin policy (Patrick) - full access
cat > /tmp/admin-policy.hcl << 'EOF'
# Admin can do everything
path "*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Audit log access
path "sys/audit/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Can manage other users
path "auth/userpass/users/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
EOF

vault policy write admin-policy /tmp/admin-policy.hcl

# Adult family member policy
cat > /tmp/adult-policy.hcl << 'EOF'
# Own user secrets - full access
path "users/data/{{identity.entity.aliases.auth_userpass_*.name}}/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "users/metadata/{{identity.entity.aliases.auth_userpass_*.name}}/*" {
  capabilities = ["read", "list"]
}

# Shared family secrets - read/write
path "shared/data/family/*" {
  capabilities = ["create", "read", "update", "list"]
}

# Read family calendar and shopping
path "family/data/calendar/*" {
  capabilities = ["read", "list"]
}

path "family/data/shopping/*" {
  capabilities = ["create", "read", "update", "list"]
}

# Cannot access other users' secrets
path "users/data/+/*" {
  capabilities = ["deny"]
}
EOF

vault policy write adult-policy /tmp/adult-policy.hcl

# Child policy - restricted access
cat > /tmp/child-policy.hcl << 'EOF'
# Own limited secrets
path "users/data/{{identity.entity.aliases.auth_userpass_*.name}}/preferences" {
  capabilities = ["read", "update"]
}

path "users/data/{{identity.entity.aliases.auth_userpass_*.name}}/games/*" {
  capabilities = ["read", "list"]
}

# Shared family - read only
path "shared/data/family/calendar" {
  capabilities = ["read"]
}

path "shared/data/family/photos/*" {
  capabilities = ["read", "list"]
}

# No access to sensitive data
path "users/data/*/passwords/*" {
  capabilities = ["deny"]
}

path "users/data/*/api_keys/*" {
  capabilities = ["deny"]
}

path "family/data/financial/*" {
  capabilities = ["deny"]
}
EOF

vault policy write child-policy /tmp/child-policy.hcl

# Echo Brain service policy
cat > /tmp/echo-service-policy.hcl << 'EOF'
# Echo can read user preferences for personalization
path "users/data/*/preferences" {
  capabilities = ["read"]
}

# Echo can store conversation metadata
path "family/data/echo/conversations/*" {
  capabilities = ["create", "read", "update"]
}

# Echo can read shared family data
path "shared/data/family/*" {
  capabilities = ["read", "list"]
}

# Echo can access service credentials
path "family/data/services/*" {
  capabilities = ["read"]
}

# Echo CANNOT access user passwords or sensitive keys
path "users/data/*/passwords/*" {
  capabilities = ["deny"]
}

path "users/data/*/api_keys/*" {
  capabilities = ["deny"]
}
EOF

vault policy write echo-service-policy /tmp/echo-service-policy.hcl

# ================================
# CREATE USER ACCOUNTS
# ================================

echo "👥 Creating family member accounts..."

# Patrick (admin)
vault write auth/userpass/users/patrick \
    password="secure_admin_password" \
    policies="admin-policy"

# Partner (adult)
vault write auth/userpass/users/partner \
    password="partner_password" \
    policies="adult-policy"

# Children (if applicable)
vault write auth/userpass/users/child1 \
    password="simple_pin_1234" \
    policies="child-policy"

# Echo Brain service account (AppRole)
vault write auth/approle/role/echo-brain \
    token_policies="echo-service-policy" \
    token_ttl="1h" \
    token_max_ttl="24h"

# Get Echo Brain credentials
echo "🤖 Echo Brain service credentials:"
vault read -field=role_id auth/approle/role/echo-brain/role-id > /tmp/echo-role-id
vault write -field=secret_id -f auth/approle/role/echo-brain/secret-id > /tmp/echo-secret-id

# ================================
# INITIALIZE SECRET STRUCTURE
# ================================

echo "📁 Creating secret structure..."

# Patrick's secrets
vault kv put users/patrick/api_keys \
    openai="sk-..." \
    github="ghp_..." \
    google="..."

vault kv put users/patrick/passwords \
    email="encrypted_password" \
    server="encrypted_password"

vault kv put users/patrick/preferences \
    theme="dark" \
    model="llama3.1:70b" \
    notifications="true"

# Partner's secrets
vault kv put users/partner/preferences \
    theme="light" \
    model="llama3.2:3b" \
    notifications="true"

# Shared family secrets
vault kv put shared/family/wifi \
    ssid="FamilyNetwork" \
    password="family_wifi_password"

vault kv put shared/family/streaming \
    netflix="family_account" \
    disney="family_account" \
    spotify="family_plan"

vault kv put shared/family/calendar \
    google_calendar_id="family@gmail.com" \
    apple_calendar="family@icloud.com"

# Service credentials (for Echo to use)
vault kv put family/services/apple_music \
    team_id="7XY5SYJMAP" \
    key_id="9M85DX285V" \
    private_key_path="/opt/tower-auth/keys/apple_music.p8"

vault kv put family/services/google \
    api_key="..." \
    oauth_client_id="..." \
    oauth_client_secret="..."

vault kv put family/services/openai \
    api_key="..." \
    organization="..."

# ================================
# AUDIT CONFIGURATION
# ================================

echo "📊 Enabling audit logging..."

# Enable file audit (logs all Vault access)
vault audit enable file file_path=/opt/vault/logs/audit.log

# Enable syslog audit for critical events
vault audit enable syslog tag="vault" facility="LOCAL7"

# ================================
# ACCESS EXAMPLES
# ================================

echo "
📚 VAULT ACCESS EXAMPLES:
========================

1. Patrick accessing his API keys:
   vault login -method=userpass username=patrick
   vault kv get users/patrick/api_keys

2. Partner accessing shared wifi:
   vault login -method=userpass username=partner
   vault kv get shared/family/wifi

3. Child accessing preferences:
   vault login -method=userpass username=child1
   vault kv get users/child1/preferences

4. Echo Brain service authentication:
   ROLE_ID=\$(cat /tmp/echo-role-id)
   SECRET_ID=\$(cat /tmp/echo-secret-id)
   TOKEN=\$(vault write -field=token auth/approle/login role_id=\$ROLE_ID secret_id=\$SECRET_ID)
   VAULT_TOKEN=\$TOKEN vault kv get family/services/apple_music

5. Admin accessing another user's data (with audit):
   vault login -method=userpass username=patrick
   vault kv get users/partner/preferences
   # This will be logged in audit log
"

# ================================
# PYTHON ACCESS EXAMPLE
# ================================

cat > /tmp/vault_family_access.py << 'EOF'
#!/usr/bin/env python3
"""
Example: Accessing Vault from Echo Brain with user context
"""

import hvac

class FamilyVaultAccess:
    def __init__(self):
        self.vault_addr = 'http://127.0.0.1:8200'
        self.client = hvac.Client(url=self.vault_addr)

    def authenticate_user(self, username: str, password: str):
        """Authenticate a family member"""
        self.client.auth.userpass.login(
            username=username,
            password=password
        )
        return self.client.is_authenticated()

    def authenticate_echo_service(self, role_id: str, secret_id: str):
        """Authenticate Echo Brain service"""
        response = self.client.auth.approle.login(
            role_id=role_id,
            secret_id=secret_id
        )
        self.client.token = response['auth']['client_token']
        return True

    def get_user_preferences(self, username: str):
        """Get user preferences (if authorized)"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=f'{username}/preferences',
                mount_point='users'
            )
            return response['data']['data']
        except Exception as e:
            return {"error": f"Access denied: {e}"}

    def get_shared_secret(self, path: str):
        """Get shared family secret"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=f'family/{path}',
                mount_point='shared'
            )
            return response['data']['data']
        except Exception as e:
            return {"error": f"Access denied: {e}"}

    def audit_admin_access(self, admin_user: str, target_user: str, reason: str):
        """Log admin access to user data"""
        # This would be logged automatically by Vault audit
        # But we can add additional application-level logging
        audit_entry = {
            "admin": admin_user,
            "target": target_user,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        # Store in application audit log
        return audit_entry


# Example usage
if __name__ == "__main__":
    vault = FamilyVaultAccess()

    # Patrick accessing his own data
    vault.authenticate_user("patrick", "secure_admin_password")
    print("Patrick's preferences:", vault.get_user_preferences("patrick"))

    # Echo service accessing family calendar
    with open('/tmp/echo-role-id', 'r') as f:
        role_id = f.read().strip()
    with open('/tmp/echo-secret-id', 'r') as f:
        secret_id = f.read().strip()

    vault.authenticate_echo_service(role_id, secret_id)
    print("Family calendar:", vault.get_shared_secret("calendar"))
EOF

echo "✅ Vault family structure configured!"
echo "Each family member has isolated secrets with appropriate access levels."