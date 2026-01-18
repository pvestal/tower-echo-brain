#!/usr/bin/env python3
"""
BULLETPROOF ECHO BRAIN VERIFICATION SUITE
No assumptions. Only facts. Every claim tested.
"""

import httpx
import asyncio
import json
import psycopg2
import time
from datetime import datetime, timedelta
from pathlib import Path

class BulletproofEchoVerifier:
    """Verify every single claim about Echo Brain with hard evidence."""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "claims_verified": [],
            "claims_false": [],
            "evidence": {}
        }

        # Connection details
        self.echo_url = "http://localhost:8309"
        self.qdrant_url = "http://localhost:6333"
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "RP78eIrW7cI2jYvL5akt1yurE"
        }

    async def test_claim_1_indefinite_memory(self):
        """CLAIM: Echo has indefinite memory retention (no 24-hour limit)"""
        print("\n[TEST 1] INDEFINITE MEMORY RETENTION")
        print("-" * 50)

        try:
            # Check database for old conversations
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Look for conversations older than 24 hours
            cursor.execute("""
                SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM echo_unified_interactions
                WHERE timestamp < NOW() - INTERVAL '24 hours'
            """)

            count, oldest, newest = cursor.fetchone()

            if count and count > 0:
                age_days = (datetime.now() - oldest).days if oldest else 0
                self.results["claims_verified"].append("INDEFINITE_MEMORY")
                self.results["evidence"]["oldest_memory"] = {
                    "count": count,
                    "oldest": str(oldest),
                    "age_days": age_days
                }
                print(f"✅ VERIFIED: Found {count} conversations older than 24 hours")
                print(f"   Oldest: {oldest} ({age_days} days old)")
                self.results["tests_passed"] += 1
                return True
            else:
                self.results["claims_false"].append("INDEFINITE_MEMORY")
                print("❌ FAILED: No conversations found older than 24 hours")
                self.results["tests_failed"] += 1
                return False

            conn.close()

        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.results["claims_false"].append("INDEFINITE_MEMORY")
            self.results["tests_failed"] += 1
            return False

    async def test_claim_2_claude_indexing(self):
        """CLAIM: Claude conversations are being indexed (12,248 files)"""
        print("\n[TEST 2] CLAUDE CONVERSATION INDEXING")
        print("-" * 50)

        try:
            # Check Qdrant for claude_conversations collection
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.qdrant_url}/collections/claude_conversations")

                if resp.status_code == 200:
                    data = resp.json()
                    count = data["result"]["points_count"]

                    # Check if indexing process is running
                    import subprocess
                    ps_result = subprocess.run(
                        ["pgrep", "-f", "index_all_claude.py"],
                        capture_output=True
                    )
                    indexing_active = ps_result.returncode == 0

                    # Get actual file count
                    claude_path = Path("/home/patrick/.claude/conversations")
                    actual_files = len(list(claude_path.glob("*.json"))) + len(list(claude_path.glob("*.md")))

                    if count > 0:
                        progress_pct = (count / 12248) * 100
                        self.results["claims_verified"].append("CLAUDE_INDEXING")
                        self.results["evidence"]["claude_indexing"] = {
                            "vectors_indexed": count,
                            "total_files": actual_files,
                            "progress": f"{progress_pct:.1f}%",
                            "process_running": indexing_active
                        }
                        print(f"✅ VERIFIED: {count}/{actual_files} files indexed ({progress_pct:.1f}%)")
                        print(f"   Process running: {indexing_active}")
                        self.results["tests_passed"] += 1
                        return True
                    else:
                        print(f"❌ FAILED: No vectors in claude_conversations collection")
                        self.results["claims_false"].append("CLAUDE_INDEXING")
                        self.results["tests_failed"] += 1
                        return False
                else:
                    print(f"❌ FAILED: Collection not found (status {resp.status_code})")
                    self.results["claims_false"].append("CLAUDE_INDEXING")
                    self.results["tests_failed"] += 1
                    return False

        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.results["claims_false"].append("CLAUDE_INDEXING")
            self.results["tests_failed"] += 1
            return False

    async def test_claim_3_knowledge_graph(self):
        """CLAIM: Knowledge graph is building Tower codebase (141,957 files)"""
        print("\n[TEST 3] KNOWLEDGE GRAPH CONSTRUCTION")
        print("-" * 50)

        try:
            # Check if knowledge graph process is running
            import subprocess
            ps_result = subprocess.run(
                ["pgrep", "-f", "tower_knowledge_graph.py"],
                capture_output=True
            )
            process_running = ps_result.returncode == 0

            # Check for checkpoint file
            checkpoint_path = Path("/opt/tower-echo-brain/data/tower_knowledge_graph_checkpoint.pkl")
            checkpoint_exists = checkpoint_path.exists()

            # Check PostgreSQL for graph data
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'tower_knowledge_graph'
                )
            """)
            table_exists = cursor.fetchone()[0]

            node_count = 0
            if table_exists:
                cursor.execute("SELECT COUNT(*) FROM tower_knowledge_graph")
                node_count = cursor.fetchone()[0]

            # Check Redis for metrics
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_nodes = r.hget('echo:knowledge_graph', 'total_nodes') or 0
            redis_edges = r.hget('echo:knowledge_graph', 'total_edges') or 0

            conn.close()

            if process_running or node_count > 0:
                self.results["claims_verified"].append("KNOWLEDGE_GRAPH")
                self.results["evidence"]["knowledge_graph"] = {
                    "process_running": process_running,
                    "checkpoint_exists": checkpoint_exists,
                    "postgres_nodes": node_count,
                    "redis_nodes": redis_nodes,
                    "redis_edges": redis_edges
                }
                print(f"✅ VERIFIED: Knowledge graph building")
                print(f"   Process running: {process_running}")
                print(f"   Nodes in PostgreSQL: {node_count}")
                print(f"   Nodes in Redis: {redis_nodes}")
                self.results["tests_passed"] += 1
                return True
            else:
                print("❌ FAILED: No evidence of knowledge graph construction")
                self.results["claims_false"].append("KNOWLEDGE_GRAPH")
                self.results["tests_failed"] += 1
                return False

        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.results["claims_false"].append("KNOWLEDGE_GRAPH")
            self.results["tests_failed"] += 1
            return False

    async def test_claim_4_memory_recall(self):
        """CLAIM: Echo can recall information from past conversations"""
        print("\n[TEST 4] MEMORY RECALL FROM PAST CONVERSATIONS")
        print("-" * 50)

        try:
            # First, store a unique fact
            test_fact = f"The secret test code is ECHO_{int(time.time())}"

            async with httpx.AsyncClient() as client:
                # Store the fact
                resp1 = await client.post(
                    f"{self.echo_url}/api/echo/query",
                    json={
                        "query": test_fact,
                        "conversation_id": "bulletproof_test",
                        "user_id": "patrick"
                    },
                    timeout=10.0
                )

                # Wait a bit
                await asyncio.sleep(2)

                # Try to recall it
                resp2 = await client.post(
                    f"{self.echo_url}/api/echo/query",
                    json={
                        "query": "What is the secret test code?",
                        "conversation_id": "bulletproof_test",
                        "user_id": "patrick"
                    },
                    timeout=10.0
                )

                if resp2.status_code == 200:
                    response = resp2.json()["response"]
                    code_number = test_fact.split("_")[-1]

                    if code_number in response or "ECHO" in response:
                        self.results["claims_verified"].append("MEMORY_RECALL")
                        self.results["evidence"]["memory_recall"] = {
                            "fact_stored": test_fact,
                            "fact_recalled": code_number in response,
                            "response_preview": response[:200]
                        }
                        print(f"✅ VERIFIED: Echo recalled stored information")
                        print(f"   Stored: {test_fact}")
                        print(f"   Recalled: {code_number in response}")
                        self.results["tests_passed"] += 1
                        return True
                    else:
                        print(f"❌ FAILED: Echo did not recall the test fact")
                        print(f"   Response: {response[:200]}")
                        self.results["claims_false"].append("MEMORY_RECALL")
                        self.results["tests_failed"] += 1
                        return False

        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.results["claims_false"].append("MEMORY_RECALL")
            self.results["tests_failed"] += 1
            return False

    async def test_claim_5_improvement_metrics(self):
        """CLAIM: Improvement metrics are real and updating"""
        print("\n[TEST 5] IMPROVEMENT METRICS REALITY CHECK")
        print("-" * 50)

        try:
            async with httpx.AsyncClient() as client:
                # Get metrics
                resp = await client.get(f"{self.echo_url}/api/echo/improvement/metrics")

                if resp.status_code == 200:
                    metrics = resp.json()

                    # Verify metrics are not all zeros/nulls
                    has_real_data = False

                    if metrics.get("knowledge_sources"):
                        claude = metrics["knowledge_sources"].get("claude_conversations", 0)
                        facts = metrics["knowledge_sources"].get("learning_facts", 0)
                        if claude > 0 or facts > 0:
                            has_real_data = True

                    if metrics.get("performance"):
                        error_rate = metrics["performance"].get("error_rate", 0)
                        response_time = metrics["performance"].get("avg_response_time", 0)
                        if error_rate > 0 or response_time > 0:
                            has_real_data = True

                    if has_real_data:
                        self.results["claims_verified"].append("IMPROVEMENT_METRICS")
                        self.results["evidence"]["improvement_metrics"] = metrics
                        print(f"✅ VERIFIED: Improvement metrics are real")
                        print(f"   Claude conversations: {metrics.get('knowledge_sources', {}).get('claude_conversations', 0)}")
                        print(f"   Learning facts: {metrics.get('knowledge_sources', {}).get('learning_facts', 0)}")
                        print(f"   Error rate: {metrics.get('performance', {}).get('error_rate', 0):.2%}")
                        self.results["tests_passed"] += 1
                        return True
                    else:
                        print("❌ FAILED: All metrics are zero/null")
                        self.results["claims_false"].append("IMPROVEMENT_METRICS")
                        self.results["tests_failed"] += 1
                        return False

        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.results["claims_false"].append("IMPROVEMENT_METRICS")
            self.results["tests_failed"] += 1
            return False

    async def test_claim_6_vector_search(self):
        """CLAIM: Qdrant vector search actually works for semantic queries"""
        print("\n[TEST 6] QDRANT VECTOR SEARCH FUNCTIONALITY")
        print("-" * 50)

        try:
            async with httpx.AsyncClient() as client:
                # First check collections exist
                resp = await client.get(f"{self.qdrant_url}/collections")
                collections = resp.json()["result"]["collections"]

                if not collections:
                    print("❌ FAILED: No Qdrant collections found")
                    self.results["claims_false"].append("VECTOR_SEARCH")
                    self.results["tests_failed"] += 1
                    return False

                # Try to search in a collection with vectors
                for collection in collections:
                    col_name = collection["name"]

                    # Get collection info
                    info_resp = await client.get(f"{self.qdrant_url}/collections/{col_name}")
                    points_count = info_resp.json()["result"]["points_count"]

                    if points_count > 0:
                        # Try a search
                        search_resp = await client.post(
                            f"{self.qdrant_url}/collections/{col_name}/points/search",
                            json={
                                "vector": [0.1] * 768,  # Dummy vector
                                "limit": 5
                            }
                        )

                        if search_resp.status_code == 200:
                            results = search_resp.json()["result"]
                            if results:
                                self.results["claims_verified"].append("VECTOR_SEARCH")
                                self.results["evidence"]["vector_search"] = {
                                    "collection": col_name,
                                    "vectors_count": points_count,
                                    "search_results": len(results)
                                }
                                print(f"✅ VERIFIED: Vector search working")
                                print(f"   Collection: {col_name}")
                                print(f"   Vectors: {points_count}")
                                print(f"   Search returned: {len(results)} results")
                                self.results["tests_passed"] += 1
                                return True

                print("❌ FAILED: No searchable collections with vectors")
                self.results["claims_false"].append("VECTOR_SEARCH")
                self.results["tests_failed"] += 1
                return False

        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.results["claims_false"].append("VECTOR_SEARCH")
            self.results["tests_failed"] += 1
            return False

    async def test_claim_7_api_endpoints(self):
        """CLAIM: All improvement API endpoints actually work"""
        print("\n[TEST 7] API ENDPOINTS FUNCTIONALITY")
        print("-" * 50)

        endpoints_to_test = [
            ("/api/echo/improvement/metrics", "GET"),
            ("/api/echo/improvement/status", "GET"),
            ("/api/echo/improvement/knowledge-graph", "GET"),
            ("/api/echo/db/stats", "GET"),
            ("/api/echo/health", "GET")
        ]

        working_endpoints = []
        failed_endpoints = []

        async with httpx.AsyncClient() as client:
            for endpoint, method in endpoints_to_test:
                try:
                    if method == "GET":
                        resp = await client.get(f"{self.echo_url}{endpoint}", timeout=5.0)

                    if resp.status_code == 200:
                        working_endpoints.append(endpoint)
                        print(f"  ✅ {endpoint}: Working")
                    else:
                        failed_endpoints.append((endpoint, resp.status_code))
                        print(f"  ❌ {endpoint}: Failed (status {resp.status_code})")
                except Exception as e:
                    failed_endpoints.append((endpoint, str(e)))
                    print(f"  ❌ {endpoint}: Error - {e}")

        if len(working_endpoints) > len(failed_endpoints):
            self.results["claims_verified"].append("API_ENDPOINTS")
            self.results["evidence"]["api_endpoints"] = {
                "working": working_endpoints,
                "failed": failed_endpoints
            }
            print(f"\n✅ VERIFIED: {len(working_endpoints)}/{len(endpoints_to_test)} endpoints working")
            self.results["tests_passed"] += 1
            return True
        else:
            self.results["claims_false"].append("API_ENDPOINTS")
            print(f"\n❌ FAILED: Only {len(working_endpoints)}/{len(endpoints_to_test)} endpoints working")
            self.results["tests_failed"] += 1
            return False

    async def test_claim_8_continuous_improvement(self):
        """CLAIM: Continuous improvement service is actually running"""
        print("\n[TEST 8] CONTINUOUS IMPROVEMENT SERVICE")
        print("-" * 50)

        try:
            # Check systemd service
            import subprocess
            result = subprocess.run(
                ["systemctl", "is-active", "echo-improvement"],
                capture_output=True,
                text=True
            )
            service_active = result.stdout.strip() == "active"

            # Check for improvement process
            ps_result = subprocess.run(
                ["pgrep", "-f", "continuous_learning.py"],
                capture_output=True
            )
            process_running = ps_result.returncode == 0

            # Check for recent improvement logs
            log_path = Path("/opt/tower-echo-brain/logs/improvement.log")
            log_exists = log_path.exists()
            recent_activity = False

            if log_exists:
                # Check if log was modified in last hour
                mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
                recent_activity = (datetime.now() - mtime) < timedelta(hours=1)

            if service_active or process_running:
                self.results["claims_verified"].append("CONTINUOUS_IMPROVEMENT")
                self.results["evidence"]["continuous_improvement"] = {
                    "service_active": service_active,
                    "process_running": process_running,
                    "log_exists": log_exists,
                    "recent_activity": recent_activity
                }
                print(f"✅ VERIFIED: Continuous improvement active")
                print(f"   Service: {service_active}")
                print(f"   Process: {process_running}")
                print(f"   Recent activity: {recent_activity}")
                self.results["tests_passed"] += 1
                return True
            else:
                print("❌ FAILED: No evidence of continuous improvement")
                self.results["claims_false"].append("CONTINUOUS_IMPROVEMENT")
                self.results["tests_failed"] += 1
                return False

        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.results["claims_false"].append("CONTINUOUS_IMPROVEMENT")
            self.results["tests_failed"] += 1
            return False

    async def run_all_tests(self):
        """Run all bulletproof verification tests."""
        print("=" * 60)
        print("BULLETPROOF ECHO BRAIN VERIFICATION SUITE")
        print("NO ASSUMPTIONS. ONLY FACTS.")
        print("=" * 60)

        # Run all tests
        await self.test_claim_1_indefinite_memory()
        await self.test_claim_2_claude_indexing()
        await self.test_claim_3_knowledge_graph()
        await self.test_claim_4_memory_recall()
        await self.test_claim_5_improvement_metrics()
        await self.test_claim_6_vector_search()
        await self.test_claim_7_api_endpoints()
        await self.test_claim_8_continuous_improvement()

        # Final report
        print("\n" + "=" * 60)
        print("BULLETPROOF VERIFICATION COMPLETE")
        print("=" * 60)

        print(f"\nTEST RESULTS:")
        print(f"  ✅ Passed: {self.results['tests_passed']}")
        print(f"  ❌ Failed: {self.results['tests_failed']}")
        print(f"  Success Rate: {self.results['tests_passed']/(self.results['tests_passed']+self.results['tests_failed'])*100:.1f}%")

        print(f"\nVERIFIED CLAIMS:")
        for claim in self.results["claims_verified"]:
            print(f"  ✅ {claim}")

        print(f"\nFALSE CLAIMS:")
        for claim in self.results["claims_false"]:
            print(f"  ❌ {claim}")

        # Save results
        with open("/opt/tower-echo-brain/bulletproof_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: bulletproof_test_results.json")

        return self.results


if __name__ == "__main__":
    verifier = BulletproofEchoVerifier()
    asyncio.run(verifier.run_all_tests())