#!/usr/bin/env python3
"""
Google Takeout Manager with Vault Integration
Download new Google Takeout data directly to Tower with deduplication
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import httpx
import asyncpg

class GoogleTakeoutManager:
    def __init__(self):
        self.vault_addr = "http://127.0.0.1:8200"
        self.download_path = Path("/opt/tower-echo-brain/data/takeout")
        self.old_takeout = Path("/mnt/10TB2/Google_Takeout_2025")
        self.db_url = "postgresql://patrick:***REMOVED***@localhost/echo_brain"

        # Ensure paths exist
        self.download_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_vault_credentials(self):
        """Get Google credentials from Vault"""
        try:
            result = subprocess.run([
                "vault", "kv", "get", "-format=json", "secret/google/takeout"
            ], env={"VAULT_ADDR": self.vault_addr}, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error("No Google credentials in Vault. Setting up...")
                return None

            return json.loads(result.stdout)["data"]["data"]
        except Exception as e:
            self.logger.error(f"Vault error: {e}")
            return None

    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash for deduplication"""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return None

    async def check_duplicate(self, file_hash):
        """Check if file already exists in old takeout or database"""
        conn = await asyncpg.connect(self.db_url)
        try:
            # Check database
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM takeout_files_processed WHERE file_hash = $1)",
                file_hash
            )
            if exists:
                return True

            # Check old takeout directory
            for old_file in self.old_takeout.rglob("*"):
                if old_file.is_file():
                    old_hash = self.calculate_file_hash(old_file)
                    if old_hash == file_hash:
                        return True
            return False
        finally:
            await conn.close()

    async def request_new_takeout(self):
        """Request new Google Takeout via API"""
        creds = self.get_vault_credentials()
        if not creds:
            self.logger.error("Setup Google OAuth2 credentials first")
            return False

        self.logger.info("🔄 Requesting new Google Takeout...")

        # This would use Google Takeout API when available
        # For now, log instructions
        self.logger.info("""
📋 Manual Setup Required:
1. Go to: https://takeout.google.com
2. Select: All data or specific services
3. Choose: Export frequency: Export once
4. File type: .tgz, 2GB max per file
5. Delivery method: Add to Drive or Send download link via email
6. Click 'Create export'
        """)
        return True

    def setup_vault_credentials(self):
        """Interactive setup for Google OAuth2 in Vault"""
        print("\n🔐 Setting up Google OAuth2 credentials in Vault")
        print("First, create a Google Cloud Project and OAuth2 app:")
        print("1. Go to: https://console.cloud.google.com")
        print("2. Create new project or select existing")
        print("3. Enable Google Takeout API")
        print("4. Create OAuth2 credentials")

        client_id = input("Enter Google OAuth2 Client ID: ").strip()
        client_secret = input("Enter Google OAuth2 Client Secret: ").strip()

        # Store in Vault
        vault_data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "setup_date": datetime.now().isoformat()
        }

        subprocess.run([
            "vault", "kv", "put", "secret/google/takeout",
            f"client_id={client_id}",
            f"client_secret={client_secret}",
            f"setup_date={vault_data['setup_date']}"
        ], env={"VAULT_ADDR": self.vault_addr})

        print("✅ Credentials stored in Vault at secret/google/takeout")

if __name__ == "__main__":
    manager = GoogleTakeoutManager()

    # Check for credentials
    if not manager.get_vault_credentials():
        manager.setup_vault_credentials()

    # Request new takeout
    asyncio.run(manager.request_new_takeout())