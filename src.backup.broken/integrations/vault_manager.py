#!/usr/bin/env python3
"""
Vault Manager for Echo Brain
Centralized credential management with creator oversight
"""

import os
import json
import logging
import hvac
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class VaultManager:
    """Manages all credentials and secrets for Echo Brain"""

    def __init__(self):
        self.vault_addr = os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")
        self.vault_token = None
        self.client = None
        self.credentials_cache = {}
        self.access_log = []

        # Patrick is the creator - full access
        self.creator_id = "patrick"
        self.is_initialized = False

    async def initialize(self):
        """Initialize Vault connection"""
        try:
            # Try to get token from file first
            token_file = "/opt/tower-echo-brain/.vault-token"
            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    self.vault_token = f.read().strip()
            else:
                # Try environment variable
                self.vault_token = os.environ.get("VAULT_TOKEN")

            if not self.vault_token:
                # Try to read from common locations
                common_locations = [
                    "/root/.vault-token",
                    "/home/patrick/.vault-token",
                    "/opt/vault/.vault-token"
                ]
                for location in common_locations:
                    if os.path.exists(location):
                        try:
                            with open(location, 'r') as f:
                                self.vault_token = f.read().strip()
                                break
                        except:
                            continue

            if self.vault_token:
                self.client = hvac.Client(url=self.vault_addr, token=self.vault_token)

                # Check if authenticated
                if self.client.is_authenticated():
                    logger.info("✅ Vault connection established")
                    self.is_initialized = True
                    await self._load_all_credentials()
                    return True
                else:
                    logger.warning("⚠️ Vault token is invalid")
            else:
                logger.warning("⚠️ No Vault token found")

        except Exception as e:
            logger.error(f"❌ Vault initialization failed: {e}")

        return False

    async def _load_all_credentials(self):
        """Load all available credentials from Vault"""
        try:
            # Define credential paths to retrieve
            credential_paths = [
                "secret/tower/telegram",
                "secret/tower/gmail",
                "secret/tower/google",
                "secret/tower/apple",
                "secret/tower/plaid",
                "secret/tower/spotify",
                "secret/tower/openai",
                "secret/tower/anthropic",
                "secret/tower/database",
                "secret/tokens/google",
                "secret/tokens/apple",
                "secret/tokens/spotify"
            ]

            for path in credential_paths:
                try:
                    # Read secret from Vault
                    response = self.client.secrets.kv.v2.read_secret_version(
                        path=path.replace("secret/", "")
                    )
                    if response:
                        service_name = path.split("/")[-1]
                        self.credentials_cache[service_name] = response['data']['data']
                        logger.info(f"✅ Loaded {service_name} credentials")

                        # Log access for oversight
                        self._log_access("load", service_name, "system_init")

                except Exception as e:
                    logger.debug(f"Could not load {path}: {e}")

        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")

    def get_telegram_credentials(self) -> Dict[str, str]:
        """Get Telegram bot credentials"""
        self._log_access("get", "telegram", "telegram_integration")

        # Check cache first
        if "telegram" in self.credentials_cache:
            return self.credentials_cache["telegram"]

        # Try to fetch from Vault
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path="tower/telegram"
            )
            if response:
                creds = response['data']['data']
                self.credentials_cache["telegram"] = creds
                return creds
        except:
            pass

        # Fallback to environment
        return {
            "bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            "chat_id": os.environ.get("TELEGRAM_CHAT_ID", "")
        }

    def get_email_credentials(self) -> Dict[str, str]:
        """Get email/SMTP credentials"""
        self._log_access("get", "gmail", "email_integration")

        if "gmail" in self.credentials_cache:
            return self.credentials_cache["gmail"]

        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path="tower/gmail"
            )
            if response:
                creds = response['data']['data']
                self.credentials_cache["gmail"] = creds
                return creds
        except:
            pass

        return {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": "587",
            "username": "patrick.vestal@gmail.com",
            "password": os.environ.get("SMTP_PASSWORD", "")
        }

    def get_google_credentials(self) -> Dict[str, str]:
        """Get Google API credentials"""
        self._log_access("get", "google", "google_integration")

        if "google" in self.credentials_cache:
            return self.credentials_cache["google"]

        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path="tower/google"
            )
            if response:
                return response['data']['data']
        except:
            pass

        return {}

    def get_all_credentials(self, requester: str = None) -> Dict[str, Any]:
        """Get all credentials (creator only)"""
        # Only Patrick can get all credentials
        if requester != self.creator_id:
            self._log_access("denied", "all_credentials", requester)
            return {"error": "Access denied. Creator privileges required."}

        self._log_access("get_all", "all_credentials", requester)
        return self.credentials_cache

    def _log_access(self, action: str, resource: str, requester: str):
        """Log credential access for oversight"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "resource": resource,
            "requester": requester
        }
        self.access_log.append(log_entry)

        # Keep last 1000 entries
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]

    def get_access_log(self, requester: str = None) -> list:
        """Get credential access log (creator only)"""
        if requester != self.creator_id:
            return []
        return self.access_log

    async def store_credential(self, service: str, data: Dict[str, str],
                              requester: str = None) -> bool:
        """Store new credential in Vault (creator only)"""
        if requester != self.creator_id:
            self._log_access("denied_store", service, requester)
            return False

        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=f"tower/{service}",
                secret=data
            )
            self.credentials_cache[service] = data
            self._log_access("store", service, requester)
            logger.info(f"✅ Stored {service} credentials")
            return True
        except Exception as e:
            logger.error(f"Failed to store {service} credentials: {e}")
            return False

# Singleton instance
vault_manager = VaultManager()

async def get_vault_manager() -> VaultManager:
    """Get initialized Vault manager"""
    if not vault_manager.is_initialized:
        await vault_manager.initialize()
    return vault_manager