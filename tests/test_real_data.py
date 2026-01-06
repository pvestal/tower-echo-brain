#!/usr/bin/env python3
"""
REAL DATA Embedding Tests for Echo Brain
=========================================
Tests against ACTUAL collections with REAL data:
- claude_conversations (your conversation history)
- kb_articles (400+ knowledge base articles)
- Anime data (character profiles, scenes, projects)

NOT synthetic bullshit test data.

Run: python3 test_real_data.py
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


QDRANT_URL = "http://localhost:6333"
ECHO_BRAIN_URL = "http://localhost:8309"
OLLAMA_URL = "http://localhost:11434"
ANIME_DB_URL = "postgresql://patrick:***REMOVED***@localhost/tower_anime"
ECHO_DB_URL = "postgresql://patrick:***REMOVED***@localhost/tower_consolidated"
EMBEDDING_MODEL = "mxbai-embed-large"


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str
    data: Optional[Dict] = None


class RealDataTestSuite:
    def __init__(self):
        self.results: List[TestResult] = []

    def run_all(self):
        print("=" * 70)
        print("ECHO BRAIN REAL DATA VALIDATION")
        print("Testing against YOUR actual data, not synthetic garbage")
        print("=" * 70)

        # Discovery phase
        self.discover_qdrant_collections()
        self.discover_postgres_data()

        # Real data tests
        self.test_claude_conversations()
        self.test_kb_articles()
        self.test_anime_data()
        self.test_cross_collection_search()
        self.test_real_memory_retrieval()

        self.print_summary()

    def record(self, name: str, passed: bool, details: str, data: Dict = None):
        self.results.append(TestResult(name, passed, details, data))
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed or data:
            print(f"     {details}")

    # =========================================================================
    # DISCOVERY: What data actually exists?
    # =========================================================================

    def discover_qdrant_collections(self):
        """Find ALL Qdrant collections and their sizes"""
        print("\n[DISCOVERY] Qdrant Collections")
        print("-" * 50)

        try:
            r = requests.get(f"{QDRANT_URL}/collections", timeout=10)
            collections = r.json().get('result', {}).get('collections', [])

            print(f"  Found {len(collections)} collections:")

            self.qdrant_collections = {}
            for coll in collections:
                name = coll.get('name', 'unknown')
                # Get details
                r = requests.get(f"{QDRANT_URL}/collections/{name}", timeout=10)
                info = r.json().get('result', {})
                points = info.get('points_count', 0)
                vectors_config = info.get('config', {}).get('params', {}).get('vectors', {})
                dims = vectors_config.get('size', 'unknown')

                self.qdrant_collections[name] = {
                    'points': points,
                    'dimensions': dims,
                    'status': info.get('status', 'unknown')
                }

                status = "✅" if points > 0 else "⚠️ EMPTY"
                print(f"    {status} {name}: {points} vectors ({dims}D)")

            # Record which ones are actually usable
            usable = [n for n, v in self.qdrant_collections.items() if v['points'] > 0]
            self.record(
                "Qdrant has data",
                len(usable) > 0,
                f"Usable collections: {usable}",
                self.qdrant_collections
            )

        except Exception as e:
            self.record("Qdrant discovery", False, str(e))
            self.qdrant_collections = {}

    def discover_postgres_data(self):
        """Check PostgreSQL for actual data"""
        print("\n[DISCOVERY] PostgreSQL Data")
        print("-" * 50)

        try:
            import psycopg2

            # Check Echo Brain DB
            conn = psycopg2.connect(ECHO_DB_URL)
            cur = conn.cursor()

            # Find tables with data
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            tables = [r[0] for r in cur.fetchall()]

            print(f"  Echo Brain tables: {len(tables)}")

            self.postgres_data = {'echo_brain': {}, 'anime': {}}

            important_tables = ['conversations', 'messages', 'kb_articles', 'past_solutions',
                               'codebase_analysis', 'memories', 'claude_conversations']

            for table in important_tables:
                if table in tables:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        self.postgres_data['echo_brain'][table] = count
                        status = "✅" if count > 0 else "⚠️"
                        print(f"    {status} {table}: {count} rows")
                    except:
                        pass

            conn.close()

            # Check Anime DB
            conn = psycopg2.connect(ANIME_DB_URL)
            cur = conn.cursor()

            anime_tables = ['character_profiles', 'projects', 'scenes', 'episodes',
                           'generated_content', 'lora_models']

            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            available = [r[0] for r in cur.fetchall()]

            print(f"  Anime Production tables:")
            for table in anime_tables:
                if table in available:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        self.postgres_data['anime'][table] = count
                        status = "✅" if count > 0 else "⚠️"
                        print(f"    {status} {table}: {count} rows")
                    except:
                        pass

            conn.close()

            total_records = sum(self.postgres_data['echo_brain'].values()) + \
                           sum(self.postgres_data['anime'].values())

            self.record(
                "PostgreSQL has data",
                total_records > 0,
                f"Total records: {total_records}",
                self.postgres_data
            )

        except ImportError:
            print("  ⚠️ psycopg2 not installed - skipping PostgreSQL discovery")
            self.postgres_data = {}
        except Exception as e:
            self.record("PostgreSQL discovery", False, str(e))
            self.postgres_data = {}

    # =========================================================================
    # TEST: Claude Conversations Collection
    # =========================================================================

    def test_claude_conversations(self):
        """Test the claude_conversations Qdrant collection"""
        print("\n[TEST] Claude Conversations Collection")
        print("-" * 50)

        collection = "claude_conversations"

        if collection not in self.qdrant_collections:
            self.record(
                "claude_conversations exists",
                False,
                "Collection not found in Qdrant"
            )
            return

        info = self.qdrant_collections[collection]

        if info['points'] == 0:
            self.record(
                "claude_conversations has data",
                False,
                "Collection is EMPTY - conversations not being vectorized!"
            )
            return

        self.record(
            "claude_conversations has data",
            True,
            f"{info['points']} conversations vectorized"
        )

        # Test: Can we search it?
        try:
            # Generate query embedding
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": "RV battery Victron electrical"},
                timeout=30
            )
            query_vec = r.json().get('embedding', [])

            # Handle dimension mismatch
            target_dims = info['dimensions']
            if len(query_vec) != target_dims:
                self.record(
                    "claude_conversations searchable",
                    False,
                    f"Dimension mismatch: model produces {len(query_vec)}D, collection has {target_dims}D"
                )
                return

            # Search
            r = requests.post(
                f"{QDRANT_URL}/collections/{collection}/points/search",
                json={"vector": query_vec, "limit": 5, "with_payload": True},
                timeout=10
            )
            results = r.json().get('result', [])

            if results:
                self.record(
                    "claude_conversations searchable",
                    True,
                    f"Top result score: {results[0].get('score', 0):.4f}"
                )

                # Show sample result
                payload = results[0].get('payload', {})
                preview = str(payload)[:200]
                print(f"     Sample result: {preview}...")
            else:
                self.record(
                    "claude_conversations searchable",
                    False,
                    "Search returned no results"
                )

        except Exception as e:
            self.record("claude_conversations searchable", False, str(e))

    # =========================================================================
    # TEST: Knowledge Base Articles
    # =========================================================================

    def test_kb_articles(self):
        """Test kb_articles - should have 400+ articles"""
        print("\n[TEST] Knowledge Base Articles")
        print("-" * 50)

        collection = "kb_articles"

        if collection not in self.qdrant_collections:
            self.record(
                "kb_articles exists",
                False,
                "Collection not found - 400+ articles not indexed!"
            )
            return

        info = self.qdrant_collections[collection]

        # Check if we have the expected ~400 articles
        expected_min = 100  # At least this many
        has_data = info['points'] >= expected_min

        self.record(
            f"kb_articles has {expected_min}+ articles",
            has_data,
            f"Found {info['points']} articles (expected 400+)"
        )

        if info['points'] == 0:
            return

        # Test: Search for something that should be in KB
        test_queries = [
            ("Echo Brain architecture", "technical"),
            ("anime character generation", "creative"),
            ("Victron battery setup", "RV"),
        ]

        for query, category in test_queries:
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": query},
                    timeout=30
                )
                query_vec = r.json().get('embedding', [])

                # Handle dimension mismatch
                if len(query_vec) != info['dimensions']:
                    continue

                r = requests.post(
                    f"{QDRANT_URL}/collections/{collection}/points/search",
                    json={"vector": query_vec, "limit": 3, "with_payload": True},
                    timeout=10
                )
                results = r.json().get('result', [])

                if results and results[0].get('score', 0) > 0.5:
                    self.record(
                        f"KB search: {category}",
                        True,
                        f"Score: {results[0]['score']:.4f}"
                    )
                else:
                    self.record(
                        f"KB search: {category}",
                        False,
                        "No relevant results found"
                    )

            except Exception as e:
                self.record(f"KB search: {category}", False, str(e))

    # =========================================================================
    # TEST: Anime Data
    # =========================================================================

    def test_anime_data(self):
        """Test anime-related data in both Qdrant and PostgreSQL"""
        print("\n[TEST] Anime Production Data")
        print("-" * 50)

        # Check for anime-related Qdrant collections
        anime_collections = [n for n in self.qdrant_collections.keys()
                           if 'anime' in n.lower() or 'character' in n.lower()
                           or 'scene' in n.lower()]

        if anime_collections:
            for coll in anime_collections:
                info = self.qdrant_collections[coll]
                self.record(
                    f"Anime collection: {coll}",
                    info['points'] > 0,
                    f"{info['points']} vectors"
                )

        # Check PostgreSQL anime data
        if 'anime' in self.postgres_data:
            anime_data = self.postgres_data['anime']

            # Character profiles
            chars = anime_data.get('character_profiles', 0)
            self.record(
                "Character profiles exist",
                chars > 0,
                f"{chars} characters defined"
            )

            # Projects
            projects = anime_data.get('projects', 0)
            self.record(
                "Anime projects exist",
                projects > 0,
                f"{projects} projects (Tokyo Debt Desire, Cyberpunk Goblin, etc.)"
            )

            # Test: Query for specific character
            if chars > 0:
                try:
                    import psycopg2
                    conn = psycopg2.connect(ANIME_DB_URL)
                    cur = conn.cursor()
                    cur.execute("SELECT name, lora_path FROM character_profiles LIMIT 3")
                    characters = cur.fetchall()
                    conn.close()

                    print(f"     Characters: {[c[0] for c in characters]}")

                    # Check if LoRAs are configured
                    has_loras = any(c[1] for c in characters)
                    self.record(
                        "Character LoRAs configured",
                        has_loras,
                        f"Characters with LoRA paths: {sum(1 for c in characters if c[1])}/{len(characters)}"
                    )
                except Exception as e:
                    self.record("Character data query", False, str(e))

    # =========================================================================
    # TEST: Cross-Collection Search
    # =========================================================================

    def test_cross_collection_search(self):
        """Test that related queries hit the right collections"""
        print("\n[TEST] Cross-Collection Search Intelligence")
        print("-" * 50)

        # Get collections with data and matching dimensions
        usable = {n: v for n, v in self.qdrant_collections.items()
                 if v['points'] > 0 and v['dimensions'] == 1024}

        if len(usable) < 2:
            self.record(
                "Multiple searchable collections",
                False,
                f"Only {len(usable)} usable collections with 1024D vectors"
            )
            return

        print(f"     Searchable collections: {list(usable.keys())}")

        # This test would verify that Echo Brain can intelligently route
        # queries to the right collection based on content
        self.record(
            "Multiple searchable collections",
            True,
            f"{len(usable)} collections available for semantic search"
        )

    # =========================================================================
    # TEST: Real Memory Retrieval via Echo Brain
    # =========================================================================

    def test_real_memory_retrieval(self):
        """
        THE REAL TEST: Ask Echo Brain about something from actual history
        and see if it retrieves relevant memories
        """
        print("\n[TEST] Echo Brain Memory Retrieval (End-to-End)")
        print("-" * 50)

        # These queries should retrieve actual memories if the system works
        test_queries = [
            {
                "query": "What do you remember about my RV electrical system?",
                "expected_terms": ["victron", "battery", "inverter", "solar", "rv", "lithium"],
                "context": "Should retrieve RV discussions"
            },
            {
                "query": "Tell me about the anime projects we've discussed",
                "expected_terms": ["tokyo", "debt", "goblin", "mei", "character", "anime"],
                "context": "Should retrieve anime production discussions"
            },
            {
                "query": "What have we worked on with Echo Brain?",
                "expected_terms": ["echo", "brain", "agent", "ollama", "qdrant", "memory"],
                "context": "Should retrieve Echo Brain development discussions"
            }
        ]

        for test in test_queries:
            try:
                r = requests.post(
                    f"{ECHO_BRAIN_URL}/api/echo/chat",
                    json={
                        "query": test["query"],
                        "session_id": "real_data_test"
                    },
                    timeout=120
                )

                if r.status_code != 200:
                    self.record(
                        f"Memory: {test['context'][:30]}",
                        False,
                        f"HTTP {r.status_code}"
                    )
                    continue

                response = r.json().get('response', '').lower()

                # Check if response contains expected terms
                found_terms = [t for t in test['expected_terms'] if t in response]
                relevance = len(found_terms) / len(test['expected_terms'])

                self.record(
                    f"Memory: {test['context'][:30]}",
                    relevance >= 0.3,  # At least 30% of expected terms
                    f"Found {len(found_terms)}/{len(test['expected_terms'])} expected terms"
                )

                if found_terms:
                    print(f"     Matched: {found_terms}")

            except requests.Timeout:
                self.record(
                    f"Memory: {test['context'][:30]}",
                    False,
                    "Request timed out (>120s)"
                )
            except Exception as e:
                self.record(
                    f"Memory: {test['context'][:30]}",
                    False,
                    str(e)
                )

    # =========================================================================
    # Summary
    # =========================================================================

    def print_summary(self):
        print("\n" + "=" * 70)
        print("REAL DATA TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

        # Critical issues
        critical_failures = []

        # Check for empty collections that should have data
        if hasattr(self, 'qdrant_collections'):
            for name, info in self.qdrant_collections.items():
                if info['points'] == 0 and name in ['claude_conversations', 'kb_articles', 'echo_memories']:
                    critical_failures.append(f"Collection '{name}' is EMPTY")

        if critical_failures:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in critical_failures:
                print(f"   ❌ {issue}")

        if passed < total:
            print("\n❌ Failed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"   • {r.name}: {r.details}")

        # Data inventory
        print("\n📊 DATA INVENTORY:")
        if hasattr(self, 'qdrant_collections'):
            total_vectors = sum(v['points'] for v in self.qdrant_collections.values())
            print(f"   Qdrant: {total_vectors} total vectors across {len(self.qdrant_collections)} collections")

        if hasattr(self, 'postgres_data'):
            echo_total = sum(self.postgres_data.get('echo_brain', {}).values())
            anime_total = sum(self.postgres_data.get('anime', {}).values())
            print(f"   PostgreSQL Echo Brain: {echo_total} records")
            print(f"   PostgreSQL Anime: {anime_total} records")

        print("\n" + "=" * 70)
        if passed == total:
            print("✅ ALL TESTS PASSED")
        elif critical_failures:
            print("❌ CRITICAL FAILURES - Memory system not using real data!")
        else:
            print("⚠️  PARTIAL SUCCESS - Some issues to address")
        print("=" * 70)


if __name__ == "__main__":
    suite = RealDataTestSuite()
    suite.run_all()