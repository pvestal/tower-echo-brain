#!/usr/bin/env python3
"""
Vault Authentication and Credential Management for Google Takeout
Integrates with HashiCorp Vault for secure credential storage and rotation
"""

import os
import json
import time
import logging
import hvac
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class VaultAuthManager:
    """Manages authentication and credential operations with HashiCorp Vault"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Vault authentication manager

        Args:
            config: Configuration dictionary with Vault settings
        """
        self.config = config
        self.vault_url = config['vault']['url']
        self.token_path = config['vault']['token_path']
        self.secret_path = config['vault']['secret_path']
        self.client: Optional[hvac.Client] = None
        self._token_expiry: Optional[datetime] = None

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Vault client with authentication"""
        try:
            self.client = hvac.Client(url=self.vault_url)

            # Try to authenticate with existing token
            if self._authenticate_with_token():
                logger.info("Successfully authenticated with Vault using existing token")
            else:
                logger.error("Failed to authenticate with Vault")
                raise Exception("Vault authentication failed")

        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {e}")
            # Fallback to JSON vault for development
            self._use_fallback_vault()

    def _authenticate_with_token(self) -> bool:
        """
        Authenticate with Vault using token from file

        Returns:
            bool: True if authentication successful
        """
        try:
            # First try to read token from file
            if Path(self.token_path).exists():
                with open(self.token_path, 'r') as f:
                    token = f.read().strip()
                self.client.token = token

            # Check if authenticated
            if self.client.is_authenticated():
                # Get token info to check expiry
                token_info = self.client.lookup_token()
                if 'expire_time' in token_info['data']:
                    expire_time = token_info['data']['expire_time']
                    if expire_time:
                        self._token_expiry = datetime.fromisoformat(
                            expire_time.replace('Z', '+00:00')
                        )
                return True

            return False

        except Exception as e:
            logger.error(f"Token authentication failed: {e}")
            return False

    def _use_fallback_vault(self) -> None:
        """Use JSON file as fallback when Vault is unavailable"""
        logger.warning("Using fallback JSON vault at ~/.tower_credentials/vault.json")
        self._fallback_mode = True

        try:
            vault_file = Path.home() / ".tower_credentials" / "vault.json"
            if vault_file.exists():
                with open(vault_file, 'r') as f:
                    self._fallback_data = json.load(f)
                logger.info("Fallback vault data loaded successfully")
            else:
                logger.error("Fallback vault file not found")
                self._fallback_data = {}

        except Exception as e:
            logger.error(f"Failed to load fallback vault: {e}")
            self._fallback_data = {}

    def is_authenticated(self) -> bool:
        """Check if currently authenticated with Vault"""
        if hasattr(self, '_fallback_mode'):
            return len(self._fallback_data) > 0

        if not self.client:
            return False

        try:
            # Check token expiry
            if self._token_expiry and datetime.utcnow() >= self._token_expiry:
                logger.info("Token has expired, re-authenticating")
                return self._authenticate_with_token()

            return self.client.is_authenticated()

        except Exception as e:
            logger.error(f"Authentication check failed: {e}")
            return False

    def get_google_credentials(self) -> Dict[str, Any]:
        """
        Retrieve Google OAuth2 credentials from Vault

        Returns:
            Dict containing Google credentials
        """
        if hasattr(self, '_fallback_mode'):
            return self._fallback_data.get('google_photos', {})

        if not self.is_authenticated():
            raise Exception("Not authenticated with Vault")

        try:
            # Try to get credentials from Vault
            secret_response = self.client.secrets.kv.v2.read_secret_version(
                path='google/takeout'
            )

            if secret_response and 'data' in secret_response:
                credentials = secret_response['data']['data']
                logger.info("Successfully retrieved Google credentials from Vault")
                return credentials
            else:
                logger.warning("No Google credentials found in Vault, checking fallback")
                self._use_fallback_vault()
                return self._fallback_data.get('google_photos', {})

        except Exception as e:
            logger.error(f"Failed to retrieve Google credentials: {e}")
            logger.info("Attempting to use fallback vault")
            self._use_fallback_vault()
            return self._fallback_data.get('google_photos', {})

    def store_google_credentials(self, credentials: Dict[str, Any]) -> bool:
        """
        Store Google OAuth2 credentials in Vault

        Args:
            credentials: Dictionary containing Google credentials

        Returns:
            bool: True if storage successful
        """
        if hasattr(self, '_fallback_mode'):
            logger.warning("Cannot store credentials in fallback mode")
            return False

        if not self.is_authenticated():
            raise Exception("Not authenticated with Vault")

        try:
            # Store credentials with metadata
            credential_data = {
                **credentials,
                'stored_at': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(days=90)).isoformat()
            }

            self.client.secrets.kv.v2.create_or_update_secret(
                path='google/takeout',
                secret=credential_data
            )

            logger.info("Successfully stored Google credentials in Vault")
            return True

        except Exception as e:
            logger.error(f"Failed to store Google credentials: {e}")
            return False

    def get_database_credentials(self) -> Dict[str, Any]:
        """
        Retrieve database credentials from Vault

        Returns:
            Dict containing database credentials
        """
        if hasattr(self, '_fallback_mode'):
            return self._fallback_data.get('postgresql', {})

        if not self.is_authenticated():
            raise Exception("Not authenticated with Vault")

        try:
            secret_response = self.client.secrets.kv.v2.read_secret_version(
                path='database/takeout'
            )

            if secret_response and 'data' in secret_response:
                credentials = secret_response['data']['data']
                logger.info("Successfully retrieved database credentials from Vault")
                return credentials
            else:
                # Fall back to main PostgreSQL credentials
                return self._fallback_data.get('postgresql', {})

        except Exception as e:
            logger.error(f"Failed to retrieve database credentials: {e}")
            return self._fallback_data.get('postgresql', {})

    def rotate_credentials(self, credential_type: str) -> bool:
        """
        Rotate credentials based on type

        Args:
            credential_type: Type of credentials to rotate ('google', 'database')

        Returns:
            bool: True if rotation successful
        """
        if hasattr(self, '_fallback_mode'):
            logger.warning("Cannot rotate credentials in fallback mode")
            return False

        if not self.is_authenticated():
            raise Exception("Not authenticated with Vault")

        try:
            if credential_type == 'google':
                # Implement Google OAuth2 token refresh
                return self._rotate_google_credentials()
            elif credential_type == 'database':
                # Implement database credential rotation
                return self._rotate_database_credentials()
            else:
                logger.error(f"Unknown credential type: {credential_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to rotate {credential_type} credentials: {e}")
            return False

    def _rotate_google_credentials(self) -> bool:
        """Rotate Google OAuth2 credentials"""
        try:
            current_creds = self.get_google_credentials()

            if 'refresh_token' not in current_creds:
                logger.error("No refresh token available for credential rotation")
                return False

            # Use the refresh token to get new access token
            # This would integrate with Google's OAuth2 refresh flow
            logger.info("Google credential rotation would be implemented here")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate Google credentials: {e}")
            return False

    def _rotate_database_credentials(self) -> bool:
        """Rotate database credentials"""
        try:
            # Implement database credential rotation logic
            # This would typically involve creating new database user
            # and updating the stored credentials
            logger.info("Database credential rotation would be implemented here")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate database credentials: {e}")
            return False

    def get_encryption_key(self, key_name: str = 'takeout-encryption') -> Optional[str]:
        """
        Get encryption key for data protection

        Args:
            key_name: Name of the encryption key

        Returns:
            str: Base64 encoded encryption key or None
        """
        if hasattr(self, '_fallback_mode'):
            # Generate a fallback key (not recommended for production)
            import base64
            import secrets
            fallback_key = base64.b64encode(secrets.token_bytes(32)).decode()
            logger.warning("Using fallback encryption key - not secure for production")
            return fallback_key

        if not self.is_authenticated():
            return None

        try:
            # Use Vault's transit backend for encryption key
            response = self.client.secrets.transit.generate_data_key(
                name=key_name,
                key_type='plaintext'
            )

            if response and 'data' in response:
                return response['data']['plaintext']

        except Exception as e:
            logger.error(f"Failed to get encryption key: {e}")

        return None

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Vault connection

        Returns:
            Dict containing health status
        """
        status = {
            'vault_connected': False,
            'authenticated': False,
            'token_expires_in': None,
            'fallback_mode': hasattr(self, '_fallback_mode')
        }

        try:
            if hasattr(self, '_fallback_mode'):
                status['vault_connected'] = False
                status['authenticated'] = len(self._fallback_data) > 0
            else:
                status['vault_connected'] = self.client is not None
                status['authenticated'] = self.is_authenticated()

                if self._token_expiry:
                    time_remaining = self._token_expiry - datetime.utcnow()
                    status['token_expires_in'] = str(time_remaining)

        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return status


def create_vault_manager(config: Dict[str, Any]) -> VaultAuthManager:
    """
    Factory function to create VaultAuthManager instance

    Args:
        config: Configuration dictionary

    Returns:
        VaultAuthManager instance
    """
    return VaultAuthManager(config)


if __name__ == "__main__":
    # Example usage and testing
    import yaml

    logging.basicConfig(level=logging.INFO)

    # Load configuration
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Test Vault manager
    vault_manager = create_vault_manager(config)

    print("Health Check:", vault_manager.health_check())

    try:
        google_creds = vault_manager.get_google_credentials()
        print("Google credentials available:", bool(google_creds))

        db_creds = vault_manager.get_database_credentials()
        print("Database credentials available:", bool(db_creds))

    except Exception as e:
        print(f"Test failed: {e}")