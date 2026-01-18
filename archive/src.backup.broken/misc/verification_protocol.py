#!/usr/bin/env python3
"""
VERIFICATION PROTOCOL - Test everything before claiming it works
"""

import asyncio
import asyncpg
import httpx
import subprocess
import json
from datetime import datetime

class VerificationProtocol:
    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "tower_echo_brain_secret_key_2025"
        }

    async def verify_conversation_persistence(self) -> bool:
        """ACTUALLY test if conversations save"""
        test_id = f"verify_{int(datetime.now().timestamp())}"

        # 1. Send test conversation
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8309/api/echo/query",
                json={
                    "query": f"VERIFICATION_TEST_{test_id}",
                    "conversation_id": test_id
                }
            )
            if response.status_code != 200:
                print(f"❌ API FAILED: {response.status_code}")
                return False

        # 2. Check database
        await asyncio.sleep(1)
        conn = await asyncpg.connect(**self.db_config)
        try:
            row = await conn.fetchrow(
                "SELECT * FROM echo_unified_interactions WHERE conversation_id = $1",
                test_id
            )
            if not row:
                print(f"❌ CONVERSATION NOT SAVED")
                return False
            print(f"✅ Conversation saved: {test_id}")
            return True
        finally:
            await conn.close()

    async def verify_learning_pipeline(self) -> bool:
        """ACTUALLY test if learning pipeline works"""
        # Count before
        conn = await asyncpg.connect(**self.db_config)
        try:
            before_facts = await conn.fetchval("SELECT COUNT(*) FROM learning_history")
            before_vectors = await self.get_vector_count()

            # Run pipeline
            result = subprocess.run([
                "python3", "/home/patrick/Tower/echo_learning_pipeline.py"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ PIPELINE FAILED: {result.stderr}")
                return False

            # Count after
            after_facts = await conn.fetchval("SELECT COUNT(*) FROM learning_history")
            after_vectors = await self.get_vector_count()

            facts_added = after_facts - before_facts
            vectors_added = after_vectors - before_vectors

            if facts_added <= 0:
                print(f"❌ NO FACTS LEARNED: {facts_added}")
                return False

            if vectors_added <= 0:
                print(f"❌ NO VECTORS CREATED: {vectors_added}")
                return False

            print(f"✅ Pipeline worked: +{facts_added} facts, +{vectors_added} vectors")
            return True

        finally:
            await conn.close()

    async def get_vector_count(self) -> int:
        """Get current vector count"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:6333/collections/learning_facts")
            if response.status_code == 200:
                return response.json()["result"]["points_count"]
            return 0

    async def verify_all_endpoints(self) -> bool:
        """Test all API endpoints actually respond"""
        endpoints = [
            "/api/echo/health",
            "/api/echo/system/metrics",
            "/api/echo/db/stats"
        ]

        async with httpx.AsyncClient() as client:
            for endpoint in endpoints:
                response = await client.get(f"http://localhost:8309{endpoint}")
                if response.status_code != 200:
                    print(f"❌ ENDPOINT BROKEN: {endpoint} -> {response.status_code}")
                    return False
                print(f"✅ Endpoint works: {endpoint}")
        return True

    async def run_full_verification(self):
        """Run complete verification suite"""
        print("=" * 60)
        print("VERIFICATION PROTOCOL - TESTING EVERYTHING")
        print("=" * 60)

        tests = [
            ("Conversation Persistence", self.verify_conversation_persistence()),
            ("Learning Pipeline", self.verify_learning_pipeline()),
            ("API Endpoints", self.verify_all_endpoints())
        ]

        results = {}
        for name, test in tests:
            print(f"\n🧪 Testing {name}...")
            try:
                result = await test
                results[name] = result
                if result:
                    print(f"✅ {name}: PASSED")
                else:
                    print(f"❌ {name}: FAILED")
            except Exception as e:
                print(f"❌ {name}: ERROR - {e}")
                results[name] = False

        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)

        passed = sum(results.values())
        total = len(results)

        for test, result in results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test}: {status}")

        if passed == total:
            print(f"\n🎉 ALL {total} TESTS PASSED - EVERYTHING ACTUALLY WORKS")
        else:
            print(f"\n💀 {total-passed} TESTS FAILED - SHIT IS BROKEN")

        return passed == total

if __name__ == "__main__":
    protocol = VerificationProtocol()
    result = asyncio.run(protocol.run_full_verification())
    exit(0 if result else 1)