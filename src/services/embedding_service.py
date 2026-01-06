"""
OpenAI Embedding Service with HashiCorp Vault integration
"""
import os
import asyncio
from typing import List, Optional
from functools import lru_cache
import hashlib

import httpx
import asyncpg

# Try HashiCorp Vault first, fall back to environment
def get_openai_key() -> str:
    # Check environment first
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    # Try HashiCorp Vault
    try:
        import hvac
        vault_addr = os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")

        # Check environment variable first
        token = os.getenv("VAULT_TOKEN")

        # Fall back to token file
        if not token:
            token_path = os.path.expanduser("~/.vault-token")
            if os.path.exists(token_path):
                with open(token_path) as f:
                    token = f.read().strip()

        if token:
            client = hvac.Client(url=vault_addr, token=token)
            if client.is_authenticated():
                secret = client.secrets.kv.v2.read_secret_version(
                    path="api_keys/openai",
                    mount_point="secret"
                )
                return secret["data"]["data"]["api_key"]
    except Exception as e:
        print(f"Vault lookup failed: {e}")

    raise ValueError("OPENAI_API_KEY not found in environment or Vault")

class EmbeddingService:
    def __init__(self):
        self.api_key = get_openai_key()
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
        self.base_url = "https://api.openai.com/v1/embeddings"
        self.db_pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database pool for caching"""
        db_url = os.getenv("DATABASE_URL",
            "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost:5432/tower_consolidated")
        self.db_pool = await asyncpg.create_pool(db_url)

    async def close(self):
        """Clean up resources"""
        if self.db_pool:
            await self.db_pool.close()

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text, with caching"""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """Embed multiple texts with caching"""
        if not texts:
            return []

        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        # Check cache
        if use_cache and self.db_pool:
            for i, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                async with self.db_pool.acquire() as conn:
                    cached = await conn.fetchrow(
                        "SELECT embedding FROM embedding_cache WHERE text_hash = $1",
                        text_hash
                    )
                    if cached:
                        results[i] = cached['embedding']
                    else:
                        texts_to_embed.append(text)
                        indices_to_embed.append(i)
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))

        # Embed uncached texts
        if texts_to_embed:
            embeddings = await self._call_openai(texts_to_embed)

            for idx, embedding in zip(indices_to_embed, embeddings):
                results[idx] = embedding

                # Cache the result
                if use_cache and self.db_pool:
                    text_hash = hashlib.md5(texts[idx].encode()).hexdigest()
                    async with self.db_pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO embedding_cache (text_hash, text, model, embedding, dimensions, created_at)
                            VALUES ($1, $2, $3, $4, $5, NOW())
                            ON CONFLICT (text_hash) DO NOTHING
                        """, text_hash, texts[idx], self.model, embedding, self.dimensions)

        return results

    async def _call_openai(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI API"""
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": texts,
                    "model": self.model,
                    "dimensions": self.dimensions
                }
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]

async def create_embedding_service() -> EmbeddingService:
    """Factory function to create initialized embedding service"""
    service = EmbeddingService()
    await service.initialize()
    return service