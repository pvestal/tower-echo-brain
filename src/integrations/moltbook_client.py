"""
Moltbook API Client for Echo Brain.
Requires API keys from https://www.moltbook.com/developers
"""
import aiohttp
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MoltbookClient:
    """Client for interacting with the Moltbook API."""
    
    def __init__(self, agent_api_key: str, app_api_key: str):
        """
        Initialize the client.
        
        Args:
            agent_api_key: The API key for the Echo Brain agent (to act on Moltbook).
            app_api_key: Your developer app key (moltdev_...) to verify tokens.
        """
        self.agent_api_key = agent_api_key
        self.app_api_key = app_api_key
        self.base_url = "https://moltbook.com/api/v1"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def ensure_session(self):
        """Ensure an HTTP session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def generate_identity_token(self) -> Optional[str]:
        """Step A: Have the agent generate a temporary identity token."""
        await self.ensure_session()
        url = f"{self.base_url}/agents/me/identity-token"
        headers = {"Authorization": f"Bearer {self.agent_api_key}"}
        
        try:
            async with self.session.post(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('token')
                else:
                    logger.error(f"Failed to get identity token: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Error generating identity token: {e}")
            return None
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Step B: Verify a token from an agent (for your other services)."""
        await self.ensure_session()
        url = f"{self.base_url}/agents/verify-identity"
        headers = {"X-Moltbook-App-Key": self.app_api_key}
        
        try:
            async with self.session.post(url, headers=headers, json={"token": token}) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Token verification failed: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    async def close(self):
        """Clean up the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

# Example of how to use it (this part won't run when imported)
if __name__ == "__main__":
    import asyncio
    async def example():
        # Replace with your actual keys
        client = MoltbookClient(
            agent_api_key="YOUR_AGENT_KEY_HERE",
            app_api_key="YOUR_APP_KEY_HERE"
        )
        try:
            token = await client.generate_identity_token()
            if token:
                print(f"Got identity token: {token[:20]}...")
                # Verify it works
                verified = await client.verify_token(token)
                if verified:
                    print(f"Token verified for agent: {verified.get('agent_name')}")
        finally:
            await client.close()
    
    asyncio.run(example())
