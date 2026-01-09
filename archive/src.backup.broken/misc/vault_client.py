#!/usr/bin/env python3
"""
HashiCorp Vault Client for Echo Brain Service
Provides secure access to credentials and secrets
"""

import os
import hvac
import json
from typing import Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EchoVaultClient:
    """HashiCorp Vault client for Echo Brain service"""
    
    def __init__(self):
        self.vault_addr = os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200')
        self.vault_dir = Path('/opt/tower-echo-brain/vault')
        self.role_id_path = self.vault_dir / 'role-id'
        self.secret_id_path = self.vault_dir / 'secret-id'
        
        self.client = hvac.Client(url=self.vault_addr)
        self._authenticate()
        
    def _authenticate(self):
        """Authenticate with Vault using AppRole"""
        try:
            # Read role_id and secret_id from files
            if not self.role_id_path.exists() or not self.secret_id_path.exists():
                logger.error("Echo AppRole credentials not found")
                return False
                
            role_id = self.role_id_path.read_text().strip()
            secret_id = self.secret_id_path.read_text().strip()
            
            # Authenticate using AppRole
            auth_response = self.client.auth.approle.login(
                role_id=role_id,
                secret_id=secret_id
            )
            
            # Set the client token
            self.client.token = auth_response['auth']['client_token']
            
            logger.info("✅ Echo service authenticated with Vault")
            return True
            
        except Exception as e:
            logger.error(f"❌ Echo Vault authentication failed: {str(e)}")
            return False
    
    def get_oauth_credentials(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get OAuth credentials for a provider"""
        try:
            secret_path = f'oauth/{provider}'
            response = self.client.secrets.kv.v2.read_secret_version(
                path=provider,
                mount_point='oauth'
            )
            
            if response and 'data' in response and 'data' in response['data']:
                credentials = response['data']['data']
                logger.info(f"✅ Retrieved {provider} OAuth credentials from Vault")
                return credentials
            else:
                logger.warning(f"⚠️  No credentials found for {provider}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving {provider} credentials: {str(e)}")
            return None
    
    def get_secret(self, path: str, mount_point: str = 'services') -> Optional[Dict[str, Any]]:
        """Get a secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=mount_point
            )
            
            if response and 'data' in response and 'data' in response['data']:
                return response['data']['data']
            return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving secret {path}: {str(e)}")
            return None
    
    def store_secret(self, path: str, data: Dict[str, Any], mount_point: str = 'services') -> bool:
        """Store a secret in Vault"""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
                mount_point=mount_point
            )
            
            logger.info(f"✅ Stored secret at {mount_point}/{path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing secret {path}: {str(e)}")
            return False
    
    def health_check(self) -> bool:
        """Check if Vault is healthy and accessible"""
        try:
            return self.client.sys.is_initialized() and not self.client.sys.is_sealed()
        except Exception as e:
            logger.error(f"❌ Vault health check failed: {str(e)}")
            return False
    
    def renew_token(self) -> bool:
        """Renew the current Vault token"""
        try:
            self.client.auth.token.renew_self()
            logger.info("✅ Successfully renewed Echo Vault token")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to renew Echo Vault token: {str(e)}")
            return False

# Singleton instance
_echo_vault_client = None

def get_echo_vault_client() -> EchoVaultClient:
    """Get singleton Echo vault client instance"""
    global _echo_vault_client
    if _echo_vault_client is None:
        _echo_vault_client = EchoVaultClient()
    return _echo_vault_client

def get_oauth_config(provider: str) -> Optional[Dict[str, Any]]:
    """Get OAuth configuration for a provider"""
    echo_vault = get_echo_vault_client()
    return echo_vault.get_oauth_credentials(provider)

def get_secret(path: str, mount_point: str = 'services') -> Optional[Dict[str, Any]]:
    """Get a secret from Vault"""
    echo_vault = get_echo_vault_client()
    return echo_vault.get_secret(path, mount_point)

def store_secret(path: str, data: Dict[str, Any], mount_point: str = 'services') -> bool:
    """Store a secret in Vault"""
    echo_vault = get_echo_vault_client()
    return echo_vault.store_secret(path, data, mount_point)

if __name__ == "__main__":
    # Test Echo Vault connection
    print("🧪 Testing Echo Vault connection...")
    
    try:
        vault = get_echo_vault_client()
        if vault.health_check():
            print("✅ Vault is healthy and accessible")
            
            # Test OAuth credential retrieval
            for provider in ['google', 'github', 'apple', 'spotify', 'plaid']:
                config = get_oauth_config(provider)
                if config and config.get('client_id'):
                    print(f"✅ {provider}: Configuration available")
                else:
                    print(f"⚠️  {provider}: No configuration found")
                    
        else:
            print("❌ Vault is not healthy")
            
    except Exception as e:
        print(f"❌ Echo Vault connection failed: {str(e)}")
        print("💡 Make sure Vault is running and Echo AppRole is configured")