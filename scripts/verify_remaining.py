"""
Verify all remaining tasks are complete
"""
import os
import sys
import asyncio
import subprocess

# Import qdrant_client BEFORE adding src to path to avoid naming conflict
from qdrant_client import QdrantClient

sys.path.insert(0, '/opt/tower-echo-brain/src')

def check(name: str, condition: bool, details: str = ""):
    status = "✅" if condition else "❌"
    print(f"{status} {name}")
    if details:
        print(f"   {details}")
    return condition

async def main():
    print("=" * 60)
    print("Echo Brain Remaining Tasks Verification")
    print("=" * 60)

    results = []

    # 1. Check HashiCorp Vault
    print("\n### HashiCorp Vault ###")
    vault_running = os.system("pgrep -f 'vault server' > /dev/null 2>&1") == 0
    results.append(check("Vault process running", vault_running))

    vault_responding = os.system("curl -s http://localhost:8200/v1/sys/health > /dev/null 2>&1") == 0
    results.append(check("Vault API responding", vault_responding))

    vault_service_exists = os.path.exists("/opt/tower-echo-brain/src/services/vault_service.py")
    results.append(check("VaultService created", vault_service_exists))

    # 2. Check code collection
    print("\n### Code Collection ###")
    qdrant = QdrantClient(host="localhost", port=6333)

    try:
        code_info = qdrant.get_collection("code")
        code_count = code_info.points_count
        results.append(check(f"Code collection populated", code_count > 0, f"{code_count} vectors"))
    except:
        results.append(check("Code collection exists", False))

    # 3. Check llm_service.py
    print("\n### LLM Service ###")
    llm_exists = os.path.exists("/opt/tower-echo-brain/src/services/llm_service.py")
    results.append(check("llm_service.py created", llm_exists))

    if llm_exists:
        with open("/opt/tower-echo-brain/src/services/llm_service.py") as f:
            content = f.read()
            has_async = "async def" in content and "aiohttp" in content
            results.append(check("LLM service uses async", has_async))

    # 4. Check fact_extractor.py
    print("\n### Fact Extractor ###")
    extractor_exists = os.path.exists("/opt/tower-echo-brain/src/memory/fact_extractor.py")
    results.append(check("fact_extractor.py created", extractor_exists))

    # 5. Test code search
    print("\n### Code Search Test ###")
    if code_count > 0:
        from services.embedding_service import create_embedding_service
        embedding_service = await create_embedding_service()
        query_vec = await embedding_service.embed_single("embedding service OpenAI")
        from qdrant_client.models import Distance, VectorParams
        results_q = qdrant.query_points(
            collection_name="code",
            query=query_vec,
            limit=3
        ).points
        found_relevant = any("embedding" in r.payload.get("filepath", "").lower() for r in results_q)
        results.append(check("Code search finds relevant files", found_relevant))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 ALL REMAINING TASKS COMPLETE!")
    else:
        print("\n⚠️  Some tasks still need attention")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)