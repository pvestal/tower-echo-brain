"""
HashiCorp Vault integration for Echo Brain
Stores API keys, database credentials, and other secrets securely
"""
import os
import hvac
from typing import Optional, Dict, Any
from functools import lru_cache

class VaultService:
    def __init__(self):
        self.vault_addr = os.getenv("VAULT_ADDR", "http://localhost:8200")

        # Try to get token from environment or file
        self.vault_token = os.getenv("VAULT_TOKEN")
        if not self.vault_token:
            token_file = os.path.expanduser("~/.vault-token")
            if os.path.exists(token_file):
                with open(token_file) as f:
                    self.vault_token = f.read().strip()

        if not self.vault_token:
            raise ValueError("VAULT_TOKEN environment variable or ~/.vault-token file required")

        self.mount_point = os.getenv("VAULT_MOUNT", "secret")

        self.client = hvac.Client(url=self.vault_addr, token=self.vault_token)

        if not self.client.is_authenticated():
            raise ConnectionError("Failed to authenticate with Vault")

    def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Get a secret from Vault KV v2"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.mount_point
            )
            return response["data"]["data"]
        except Exception as e:
            print(f"Failed to read secret {path}: {e}")
            return None

    def set_secret(self, path: str, data: Dict[str, Any]) -> bool:
        """Store a secret in Vault KV v2"""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            print(f"Failed to write secret {path}: {e}")
            return False

    def list_secrets(self, path: str = "") -> list:
        """List secrets at a path"""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path=path,
                mount_point=self.mount_point
            )
            return response["data"]["keys"]
        except Exception as e:
            print(f"Failed to list secrets at {path}: {e}")
            return []

    def delete_secret(self, path: str) -> bool:
        """Delete a secret"""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path,
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            print(f"Failed to delete secret {path}: {e}")
            return False

    # Convenience methods for common secrets
    def get_api_key(self, service: str) -> Optional[str]:
        """Get an API key for a service"""
        secret = self.get_secret(f"api_keys/{service}")
        return secret.get("key") if secret else None

    def get_db_credentials(self, db_name: str) -> Optional[Dict[str, str]]:
        """Get database credentials"""
        return self.get_secret(f"databases/{db_name}")

@lru_cache()
def get_vault_service() -> VaultService:
    return VaultService()