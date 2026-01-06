#!/usr/bin/env python3
"""
Proper Embedding System Tests for Echo Brain
============================================
These tests actually validate the memory system works, not just that it runs.

Run: python3 test_embedding_system.py
"""

import requests
import json
import time
import random
import string
from typing import List, Tuple
from dataclasses import dataclass


OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
ECHO_BRAIN_URL = "http://localhost:8309"
EMBEDDING_MODEL = "mxbai-embed-large"
COLLECTION = "echo_memories"


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str
    duration_ms: float


class EmbeddingTestSuite:
    def __init__(self):
        self.results: List[TestResult] = []

    def run_all(self):
        """Run all test categories"""
        print("=" * 70)
        print("ECHO BRAIN EMBEDDING SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 70)

        # Test categories
        self.test_basic_connectivity()
        self.test_embedding_dimensions()
        self.test_semantic_clustering()
        self.test_retrieval_precision()
        self.test_edge_cases()
        self.test_performance()
        self.test_persistence()
        self.test_conversation_integration()

        self.print_summary()

    def record(self, name: str, passed: bool, details: str, duration_ms: float):
        self.results.append(TestResult(name, passed, details, duration_ms))
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            print(f"         {details}")

    # =========================================================================
    # CATEGORY 1: Basic Connectivity
    # =========================================================================

    def test_basic_connectivity(self):
        print("\n[1/8] BASIC CONNECTIVITY")

        # Test Ollama
        start = time.time()
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            models = [m['name'] for m in r.json().get('models', [])]
            has_embed = any(EMBEDDING_MODEL in m for m in models)
            self.record(
                "Ollama running with embedding model",
                has_embed,
                f"Models: {models}" if not has_embed else f"Found {EMBEDDING_MODEL}",
                (time.time() - start) * 1000
            )
        except Exception as e:
            self.record("Ollama running", False, str(e), 0)

        # Test Qdrant
        start = time.time()
        try:
            r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5)
            data = r.json()
            exists = data.get('result', {}).get('status') == 'green'
            vectors = data.get('result', {}).get('points_count', 0)
            dims = data.get('result', {}).get('config', {}).get('params', {}).get('vectors', {}).get('size', 0)
            self.record(
                "Qdrant collection exists",
                exists,
                f"Collection: {COLLECTION}, Vectors: {vectors}, Dims: {dims}",
                (time.time() - start) * 1000
            )
        except Exception as e:
            self.record("Qdrant collection exists", False, str(e), 0)

    # =========================================================================
    # CATEGORY 2: Embedding Dimensions
    # =========================================================================

    def test_embedding_dimensions(self):
        print("\n[2/8] EMBEDDING DIMENSIONS")

        start = time.time()
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": "test"},
                timeout=30
            )
            embedding = r.json().get('embedding', [])
            dims = len(embedding)

            self.record(
                "Embedding produces 1024 dimensions",
                dims == 1024,
                f"Got {dims} dimensions",
                (time.time() - start) * 1000
            )

            # Verify all values are valid floats
            valid = all(isinstance(x, (int, float)) and -10 < x < 10 for x in embedding)
            self.record(
                "Embedding values are valid floats",
                valid,
                f"Sample values: {embedding[:5]}",
                0
            )
        except Exception as e:
            self.record("Embedding generation", False, str(e), 0)

    # =========================================================================
    # CATEGORY 3: Semantic Clustering (The Real Test)
    # =========================================================================

    def test_semantic_clustering(self):
        """
        This is the REAL test - do similar concepts cluster together?
        We embed multiple phrases and verify similarity scores make sense.
        """
        print("\n[3/8] SEMANTIC CLUSTERING (Critical)")

        # Define test clusters - phrases within a cluster should be similar
        clusters = {
            "rv_electrical": [
                "My RV battery died and won't charge",
                "The camper's lithium batteries aren't holding power",
                "Victron inverter showing low voltage warning",
                "Solar panels not charging the coach batteries",
            ],
            "cooking": [
                "Recipe for chicken stir fry with vegetables",
                "How to make homemade pasta from scratch",
                "Best way to season a cast iron skillet",
                "Meal prep ideas for the week",
            ],
            "programming": [
                "Python async await syntax explanation",
                "How to fix a TypeScript type error",
                "FastAPI dependency injection pattern",
                "Debug segfault in C++ code",
            ],
        }

        # Generate embeddings for all phrases
        embeddings = {}
        for cluster_name, phrases in clusters.items():
            for phrase in phrases:
                try:
                    r = requests.post(
                        f"{OLLAMA_URL}/api/embeddings",
                        json={"model": EMBEDDING_MODEL, "prompt": phrase},
                        timeout=30
                    )
                    embeddings[phrase] = r.json().get('embedding', [])
                except:
                    pass

        def cosine_sim(a: List[float], b: List[float]) -> float:
            dot = sum(x*y for x, y in zip(a, b))
            norm_a = sum(x*x for x in a) ** 0.5
            norm_b = sum(x*x for x in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0

        # Test 1: Within-cluster similarity should be HIGH (> 0.6)
        within_scores = []
        for cluster_name, phrases in clusters.items():
            for i, p1 in enumerate(phrases):
                for p2 in phrases[i+1:]:
                    if p1 in embeddings and p2 in embeddings:
                        sim = cosine_sim(embeddings[p1], embeddings[p2])
                        within_scores.append(sim)

        avg_within = sum(within_scores) / len(within_scores) if within_scores else 0
        self.record(
            "Within-cluster similarity > 0.5",
            avg_within > 0.5,
            f"Average: {avg_within:.4f} (min: {min(within_scores):.4f}, max: {max(within_scores):.4f})",
            0
        )

        # Test 2: Cross-cluster similarity should be LOWER
        cross_scores = []
        cluster_names = list(clusters.keys())
        for i, c1 in enumerate(cluster_names):
            for c2 in cluster_names[i+1:]:
                for p1 in clusters[c1]:
                    for p2 in clusters[c2]:
                        if p1 in embeddings and p2 in embeddings:
                            sim = cosine_sim(embeddings[p1], embeddings[p2])
                            cross_scores.append(sim)

        avg_cross = sum(cross_scores) / len(cross_scores) if cross_scores else 0
        self.record(
            "Cross-cluster similarity < within-cluster",
            avg_cross < avg_within,
            f"Cross: {avg_cross:.4f} vs Within: {avg_within:.4f} (gap: {avg_within - avg_cross:.4f})",
            0
        )

        # Test 3: Separation ratio (within/cross should be > 1.2)
        separation = avg_within / avg_cross if avg_cross > 0 else 0
        self.record(
            "Cluster separation ratio > 1.2",
            separation > 1.2,
            f"Ratio: {separation:.4f}",
            0
        )

    # =========================================================================
    # CATEGORY 4: Retrieval Precision
    # =========================================================================

    def test_retrieval_precision(self):
        """
        Store known data, query for it, verify we get the right results.
        """
        print("\n[4/8] RETRIEVAL PRECISION")

        # Store test data
        test_data = [
            ("mem_rv_1", "Patrick's Victron MultiPlus inverter is 3000VA"),
            ("mem_rv_2", "The RV has 400Ah of lithium batteries"),
            ("mem_rv_3", "Solar array is 800 watts on the roof"),
            ("mem_cook_1", "Patrick likes spicy Thai food"),
            ("mem_cook_2", "Favorite recipe is maple dijon chicken"),
            ("mem_code_1", "Echo Brain runs on FastAPI port 8309"),
            ("mem_code_2", "Tower server has 64GB RAM"),
        ]

        # Clear and insert test points
        for point_id, text in test_data:
            try:
                # Generate embedding
                r = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": text},
                    timeout=30
                )
                embedding = r.json().get('embedding', [])

                # Upsert to Qdrant
                requests.put(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points",
                    json={
                        "points": [{
                            "id": hash(point_id) % (2**63),
                            "vector": embedding,
                            "payload": {"text": text, "test_id": point_id}
                        }]
                    },
                    timeout=10
                )
            except Exception as e:
                print(f"    Warning: Failed to store {point_id}: {e}")

        time.sleep(1)  # Let Qdrant index

        # Test queries - each should return the expected category
        queries = [
            ("What's Patrick's inverter setup?", "rv"),
            ("Tell me about the solar panels", "rv"),
            ("What food does Patrick like?", "cook"),
            ("What port does Echo Brain use?", "code"),
        ]

        correct = 0
        total = 0

        for query, expected_category in queries:
            try:
                # Generate query embedding
                r = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": query},
                    timeout=30
                )
                query_embedding = r.json().get('embedding', [])

                # Search Qdrant
                r = requests.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                    json={"vector": query_embedding, "limit": 1, "with_payload": True},
                    timeout=10
                )
                results = r.json().get('result', [])

                if results:
                    top_result = results[0].get('payload', {}).get('test_id', '')
                    matched = expected_category in top_result
                    correct += 1 if matched else 0
                    total += 1
            except:
                total += 1

        precision = correct / total if total > 0 else 0
        self.record(
            f"Query retrieval precision >= 75%",
            precision >= 0.75,
            f"{correct}/{total} queries returned correct category ({precision*100:.0f}%)",
            0
        )

    # =========================================================================
    # CATEGORY 5: Edge Cases
    # =========================================================================

    def test_edge_cases(self):
        print("\n[5/8] EDGE CASES")

        edge_cases = [
            ("Empty string", ""),
            ("Single character", "a"),
            ("Very long text", "word " * 1000),
            ("Special characters", "!@#$%^&*(){}[]|\\:;<>?,./~`"),
            ("Unicode", "你好世界 🚀 مرحبا"),
            ("Numbers only", "12345 67890"),
            ("Mixed", "Test123!@# with spaces\nand\nnewlines"),
        ]

        for name, text in edge_cases:
            start = time.time()
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": text if text else " "},
                    timeout=30
                )
                embedding = r.json().get('embedding', [])
                valid = len(embedding) == 1024
                self.record(
                    f"Edge case: {name}",
                    valid,
                    f"Got {len(embedding)} dims" if not valid else "OK",
                    (time.time() - start) * 1000
                )
            except Exception as e:
                self.record(f"Edge case: {name}", False, str(e), 0)

    # =========================================================================
    # CATEGORY 6: Performance
    # =========================================================================

    def test_performance(self):
        print("\n[6/8] PERFORMANCE")

        # Test embedding generation speed
        times = []
        for i in range(10):
            text = f"Test sentence number {i} for performance measurement"
            start = time.time()
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": text},
                    timeout=30
                )
                r.json()
                times.append((time.time() - start) * 1000)
            except:
                pass

        if times:
            avg_ms = sum(times) / len(times)
            self.record(
                "Embedding generation < 500ms avg",
                avg_ms < 500,
                f"Avg: {avg_ms:.1f}ms, Min: {min(times):.1f}ms, Max: {max(times):.1f}ms",
                avg_ms
            )

        # Test Qdrant search speed
        search_times = []
        for i in range(10):
            # Random vector for search
            vector = [random.random() for _ in range(1024)]
            start = time.time()
            try:
                r = requests.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                    json={"vector": vector, "limit": 5},
                    timeout=10
                )
                r.json()
                search_times.append((time.time() - start) * 1000)
            except:
                pass

        if search_times:
            avg_search = sum(search_times) / len(search_times)
            self.record(
                "Qdrant search < 50ms avg",
                avg_search < 50,
                f"Avg: {avg_search:.1f}ms",
                avg_search
            )

    # =========================================================================
    # CATEGORY 7: Persistence
    # =========================================================================

    def test_persistence(self):
        print("\n[7/8] PERSISTENCE")

        # Store a unique marker
        marker = f"PERSISTENCE_TEST_{int(time.time())}"

        try:
            # Generate embedding
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": marker},
                timeout=30
            )
            embedding = r.json().get('embedding', [])

            # Store it
            point_id = hash(marker) % (2**63)
            requests.put(
                f"{QDRANT_URL}/collections/{COLLECTION}/points",
                json={
                    "points": [{
                        "id": point_id,
                        "vector": embedding,
                        "payload": {"marker": marker}
                    }]
                },
                timeout=10
            )

            time.sleep(1)

            # Retrieve it
            r = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                json={"vector": embedding, "limit": 1, "with_payload": True},
                timeout=10
            )
            results = r.json().get('result', [])

            found = any(r.get('payload', {}).get('marker') == marker for r in results)
            self.record(
                "Point persists after storage",
                found,
                f"Marker: {marker[:30]}...",
                0
            )

            # Get collection stats
            r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5)
            points = r.json().get('result', {}).get('points_count', 0)
            self.record(
                "Collection has stored points",
                points > 0,
                f"Total points: {points}",
                0
            )

        except Exception as e:
            self.record("Persistence test", False, str(e), 0)

    # =========================================================================
    # CATEGORY 8: Echo Brain Integration
    # =========================================================================

    def test_conversation_integration(self):
        """
        The ultimate test - does Echo Brain actually USE the memory system?
        """
        print("\n[8/8] ECHO BRAIN INTEGRATION")

        # First, store a fact
        fact = "Patrick's favorite programming language is Python and he drives a 2022 Tundra"

        try:
            # Send message to Echo Brain that should store memory
            r = requests.post(
                f"{ECHO_BRAIN_URL}/api/echo/chat",
                json={
                    "query": f"Remember this: {fact}",
                    "session_id": "test_user"
                },
                timeout=60
            )

            if r.status_code == 200:
                self.record(
                    "Echo Brain accepts memory storage request",
                    True,
                    "Message sent successfully",
                    0
                )
            else:
                self.record(
                    "Echo Brain accepts memory storage request",
                    False,
                    f"Status: {r.status_code}",
                    0
                )

            time.sleep(2)

            # Query for that fact
            r = requests.post(
                f"{ECHO_BRAIN_URL}/api/echo/chat",
                json={
                    "query": "What truck does Patrick drive?",
                    "session_id": "test_user"
                },
                timeout=60
            )

            if r.status_code == 200:
                response = r.json().get('response', '')
                mentions_tundra = 'tundra' in response.lower()
                self.record(
                    "Echo Brain retrieves stored memory",
                    mentions_tundra,
                    f"Response mentions Tundra: {mentions_tundra}",
                    0
                )
            else:
                self.record(
                    "Echo Brain retrieves stored memory",
                    False,
                    f"Status: {r.status_code}",
                    0
                )

        except Exception as e:
            self.record("Echo Brain integration", False, str(e), 0)

    # =========================================================================
    # Summary
    # =========================================================================

    def print_summary(self):
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

        if passed < total:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ❌ {r.name}: {r.details}")

        print("\nPerformance metrics:")
        for r in self.results:
            if r.duration_ms > 0:
                print(f"  {r.name}: {r.duration_ms:.1f}ms")

        # Overall verdict
        print("\n" + "=" * 70)
        if passed == total:
            print("✅ ALL TESTS PASSED - Embedding system is properly configured")
        elif passed / total >= 0.8:
            print("⚠️  MOSTLY WORKING - Some issues to address")
        else:
            print("❌ SIGNIFICANT ISSUES - Embedding system needs fixes")
        print("=" * 70)


if __name__ == "__main__":
    suite = EmbeddingTestSuite()
    suite.run_all()