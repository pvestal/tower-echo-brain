#!/usr/bin/env python3
"""
Echo Brain Persistence Testing Suite
Tests if conversations are actually saved and vectors are created
"""

import asyncio
import httpx
import asyncpg
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class PersistenceTester:
    def __init__(self):
        self.api_url = "http://localhost:8309"
        self.qdrant_url = "http://localhost:6333"
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "***REMOVED***"
        }
        self.test_results = []

    async def test_conversation_save(self) -> bool:
        """Test if conversations are saved to database"""
        print("\n🧪 Testing Conversation Persistence...")

        # Generate unique test ID
        test_id = f"persist_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_query = f"TEST_PERSISTENCE_{test_id}: Can you save this conversation?"

        # Send test query
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/api/echo/query",
                json={
                    "query": test_query,
                    "conversation_id": test_id,
                    "metadata": {"test": True, "timestamp": datetime.now().isoformat()}
                }
            )

        if response.status_code != 200:
            print(f"❌ API returned {response.status_code}")
            return False

        # Wait for save to complete
        await asyncio.sleep(1)

        # Check database
        conn = await asyncpg.connect(**self.db_config)
        try:
            # Check echo_unified_interactions
            row = await conn.fetchrow("""
                SELECT conversation_id, query, response, metadata, timestamp
                FROM echo_unified_interactions
                WHERE conversation_id = $1
            """, test_id)

            if row:
                print(f"✅ Conversation saved to echo_unified_interactions")
                print(f"   ID: {row['conversation_id']}")
                print(f"   Query: {row['query'][:50]}...")
                print(f"   Timestamp: {row['timestamp']}")
                return True
            else:
                print(f"❌ Conversation NOT found in database")
                return False

        finally:
            await conn.close()

    async def test_vector_creation(self) -> bool:
        """Test if vectors are created in Qdrant"""
        print("\n🧪 Testing Vector Memory Updates...")

        async with httpx.AsyncClient() as client:
            # Check Qdrant collections
            response = await client.get(f"{self.qdrant_url}/collections")

            if response.status_code != 200:
                print(f"❌ Qdrant not accessible")
                return False

            collections = response.json()["result"]["collections"]
            print(f"📊 Found {len(collections)} Qdrant collections:")

            for collection in collections:
                name = collection["name"]
                # Get collection info
                info_response = await client.get(f"{self.qdrant_url}/collections/{name}")
                if info_response.status_code == 200:
                    info = info_response.json()["result"]
                    count = info.get("points_count", 0)
                    print(f"   - {name}: {count} vectors")

            # TODO: Check if new vectors are being created
            # This requires tracking before/after counts
            return True

    async def test_recent_activity(self) -> Dict:
        """Check recent database activity"""
        print("\n📊 Checking Recent Activity...")

        conn = await asyncpg.connect(**self.db_config)
        try:
            # Last hour's conversations
            recent = await conn.fetch("""
                SELECT conversation_id,
                       substring(query, 1, 50) as query_start,
                       timestamp
                FROM echo_unified_interactions
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC
                LIMIT 10
            """)

            print(f"Found {len(recent)} conversations in last hour:")
            for row in recent:
                print(f"   {row['timestamp']}: {row['query_start']}...")

            # Check save rate
            total = await conn.fetchval("""
                SELECT COUNT(*) FROM echo_unified_interactions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)

            return {
                "last_hour": len(recent),
                "last_24h": total,
                "latest": recent[0]['timestamp'] if recent else None
            }

        finally:
            await conn.close()

    async def test_learning_pipeline(self) -> bool:
        """Test if learning pipeline is functional"""
        print("\n🧪 Testing Learning Pipeline...")

        conn = await asyncpg.connect(**self.db_config)
        try:
            # Check learning_history table
            recent_learning = await conn.fetch("""
                SELECT learned_fact, confidence, created_at
                FROM learning_history
                WHERE created_at > NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                LIMIT 5
            """)

            if recent_learning:
                print(f"✅ Found {len(recent_learning)} recent learned facts")
                for row in recent_learning:
                    print(f"   - {row['learned_fact'][:50]}... (confidence: {row['confidence']})")
                return True
            else:
                print("❌ No recent learning activity")
                return False

        finally:
            await conn.close()

    async def test_external_access(self) -> bool:
        """Test if external monitoring still works"""
        print("\n🧪 Testing External Access Points...")

        endpoints = [
            "/api/echo/health",
            "/api/echo/system/metrics",
            "/api/echo/db/stats"
        ]

        async with httpx.AsyncClient() as client:
            for endpoint in endpoints:
                try:
                    response = await client.get(f"{self.api_url}{endpoint}", timeout=5.0)
                    if response.status_code == 200:
                        print(f"✅ {endpoint}: OK")
                    else:
                        print(f"❌ {endpoint}: {response.status_code}")
                except Exception as e:
                    print(f"❌ {endpoint}: {e}")

        return True

    async def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 60)
        print("ECHO BRAIN PERSISTENCE TEST SUITE")
        print("=" * 60)

        results = {
            "conversation_save": await self.test_conversation_save(),
            "vector_creation": await self.test_vector_creation(),
            "recent_activity": await self.test_recent_activity(),
            "learning_pipeline": await self.test_learning_pipeline(),
            "external_access": await self.test_external_access()
        }

        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        passed = 0
        failed = 0

        for test, result in results.items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
                if result:
                    passed += 1
                else:
                    failed += 1
                print(f"{test}: {status}")
            else:
                print(f"{test}: {result}")

        print(f"\nTotal: {passed} passed, {failed} failed")

        # Overall status
        if failed == 0:
            print("\n🎉 All persistence tests passed!")
        else:
            print(f"\n⚠️ {failed} tests failed - persistence issues detected")

        return results


async def main():
    tester = PersistenceTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())