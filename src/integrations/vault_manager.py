"""
Vault Manager with graceful fallback
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import hvac, but don't fail if not available
try:
    import hvac
    HAS_HVAC = True
except ImportError:
    HAS_HVAC = False
    logger.warning("⚠️ hvac package not installed. Vault integration disabled.")

class VaultManager:
    """Vault manager with fallback to local storage when hvac is unavailable"""
    
    def __init__(self):
        self.enabled = HAS_HVAC and os.getenv('USE_VAULT', 'false').lower() == 'true'
        if self.enabled:
            self.client = hvac.Client(url=os.getenv('VAULT_ADDR', 'http://localhost:8200'))
        else:
            self.client = None
            logger.info("Vault integration disabled or hvac not available")
    
    async def get_secret(self, path: str) -> Optional[Dict]:
        if not self.enabled or not self.client:
            logger.debug(f"Vault disabled, returning empty secret for {path}")
            return None
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response.get('data', {}).get('data', {})
        except Exception as e:
            logger.warning(f"Failed to read secret {path}: {e}")
            return None

# Singleton instance
_vault_manager_instance = None

async def get_vault_manager():
    """Get or create vault manager instance"""
    global _vault_manager_instance
    if _vault_manager_instance is None:
        _vault_manager_instance = VaultManager()
    return _vault_manager_instance
