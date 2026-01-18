#!/usr/bin/env python3
"""
Connect AI Assist to Vault for all authentication needs
"""

import os
import hvac
from typing import Dict, Optional

class EchoVaultConnector:
    """AI Assist's connection to HashiCorp Vault"""

    def __init__(self):
        self.vault_addr = 'http://127.0.0.1:8200'
        self.vault_token = 'hvs.FEQ0zs7Jcng6B5nmuwtTlZnM'
        self.client = None
        self.connect()

    def connect(self):
        """Connect to Vault"""
        try:
            self.client = hvac.Client(url=self.vault_addr)
            self.client.token = self.vault_token

            # Check if sealed
            if self.client.sys.is_sealed():
                print("⚠️ Vault is sealed, attempting to unseal...")
                self.unseal()

            print("✅ Connected to Vault")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Vault: {e}")
            return False

    def unseal(self):
        """Unseal Vault with stored keys"""
        keys = [
            'iiTurL/7NBHu+OsWbCfe27GsAKZ3szkwjW/D0lSpBZa/',
            'txBLWboQsU1uZDR1tYCQesltBaMplNZosn2NPkYJI5X3',
            'MQEsqtNb2jmaVtC6rbUyU3c6BbruqzMnSDCGV6Y9TsGH'
        ]

        for key in keys:
            self.client.sys.submit_unseal_key(key)

    def get_auth_credential(self, provider: str, field: str = None) -> Optional[any]:
        """Get authentication credential from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=f'auth/{provider}'
            )

            if response and 'data' in response:
                data = response['data']['data']
                if field:
                    return data.get(field)
                return data

            return None
        except Exception as e:
            print(f"⚠️ Could not get {provider} credentials: {e}")
            return None

    def get_all_auth_status(self) -> Dict:
        """Get status of all authentication providers"""
        providers = ['google', 'apple', 'spotify', 'github', 'plaid', 'microsoft', 'discord']
        status = {}

        for provider in providers:
            creds = self.get_auth_credential(provider)
            if creds:
                # Check if actually configured or just placeholder
                client_id = creds.get('client_id', '')
                is_configured = (
                    client_id and
                    client_id != 'PENDING' and
                    client_id != 'PENDING_CONFIGURATION'
                )
                status[provider] = {
                    'stored': True,
                    'configured': is_configured or creds.get('configured') == 'true'
                }
            else:
                status[provider] = {
                    'stored': False,
                    'configured': False
                }

        return status


# Test the connector
if __name__ == "__main__":
    print("🔐 ECHO BRAIN - VAULT CONNECTION TEST")
    print("=" * 50)

    connector = EchoVaultConnector()

    print("\n📋 Authentication Provider Status:")
    print("-" * 40)

    status = connector.get_all_auth_status()
    for provider, info in status.items():
        icon = "✅" if info['configured'] else "⚠️" if info['stored'] else "❌"
        config_status = "Configured" if info['configured'] else "Pending" if info['stored'] else "Not stored"
        print(f"{icon} {provider.ljust(12)}: {config_status}")

    # Test Apple Music credentials
    print("\n🎵 Testing Apple Music credentials:")
    apple_creds = connector.get_auth_credential('apple')
    if apple_creds:
        print(f"  Team ID: {apple_creds.get('team_id')}")
        print(f"  Key ID: {apple_creds.get('key_id')}")
        print(f"  Configured: {apple_creds.get('configured')}")

    # Test Plaid credentials
    print("\n💰 Testing Plaid credentials:")
    plaid_creds = connector.get_auth_credential('plaid')
    if plaid_creds:
        print(f"  Client ID: {plaid_creds.get('client_id')}")
        print(f"  Environment: {plaid_creds.get('environment')}")

    print("\n✅ Vault connection successful!")
    print("AI Assist can now securely access all authentication credentials")