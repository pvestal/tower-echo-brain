# HashiCorp Vault Policy for Google Takeout Service
# Grants necessary permissions for credential management

# Google Takeout specific secrets
path "secret/google/takeout/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Google OAuth2 credentials
path "secret/google/oauth2/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Database credentials for takeout service
path "secret/database/takeout/*" {
  capabilities = ["read"]
}

# Encryption keys for sensitive data
path "secret/encryption/takeout/*" {
  capabilities = ["create", "read", "update", "list"]
}

# PKI for certificate management
path "pki/cert/takeout" {
  capabilities = ["read"]
}

# Transit backend for encryption operations
path "transit/encrypt/takeout" {
  capabilities = ["update"]
}

path "transit/decrypt/takeout" {
  capabilities = ["update"]
}

path "transit/datakey/plaintext/takeout" {
  capabilities = ["update"]
}

# Dynamic secrets for temporary credentials
path "gcp/token/takeout-service" {
  capabilities = ["read"]
}

# Auth token lookup and renewal
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}