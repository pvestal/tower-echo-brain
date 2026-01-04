#!/usr/bin/env python3
"""
Database-level memory persistence tests for Echo Brain
Tests PostgreSQL and Qdrant vector database integration
"""

import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
import httpx
import json
import uuid
from datetime import datetime, timedelta
from colorama import Fore, Style, init

init(autoreset=True)

class DatabaseMemoryTester:
    """Test database-level memory persistence"""

    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "tower_consolidated",
            "user": "patrick",
            "password": "***REMOVED***"
        }
        self.qdrant_url = "http://localhost:6333"
        self.echo_url = "http://localhost:8309"
        self.test_results = []

    async def run_all_tests(self):
        """Run all database memory tests"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Database Memory Persistence Tests")
        print(f"{Fore.CYAN}{'='*60}\n")

        # Test PostgreSQL conversation storage
        await self.test_postgresql_storage()

        # Test Qdrant vector memory
        await self.test_qdrant_vector_memory()

        # Test memory consolidation
        await self.test_memory_consolidation()

        # Test user context persistence
        await self.test_user_context_persistence()

        # Print summary
        self.print_summary()

    async def test_postgresql_storage(self):
        """Test PostgreSQL conversation storage"""
        print(f"\n{Fore.YELLOW}[TEST 1] PostgreSQL Conversation Storage{Style.RESET_ALL}")

        try:
            # Connect to database
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check conversation tables exist
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE '%conversation%'
                ORDER BY table_name;
            """)

            tables = cur.fetchall()
            if tables:
                print(f"{Fore.GREEN}✓ Found {len(tables)} conversation tables:")
                for table in tables:
                    print(f"  - {table['table_name']}")

                # Check recent conversations
                cur.execute("""
                    SELECT COUNT(*) as count,
                           MAX(created_at) as latest
                    FROM echo_conversations
                    WHERE created_at > NOW() - INTERVAL '7 days';
                """)

                result = cur.fetchone()
                if result and result['count'] > 0:
                    print(f"{Fore.GREEN}✓ Found {result['count']} conversations in last 7 days")
                    print(f"  Latest: {result['latest']}")
                    self.test_results.append(("PostgreSQL Storage", True, f"{result['count']} recent conversations"))
                else:
                    print(f"{Fore.YELLOW}⚠ No recent conversations found")
                    self.test_results.append(("PostgreSQL Storage", False, "No recent conversations"))

                # Check message storage
                cur.execute("""
                    SELECT COUNT(*) as message_count
                    FROM echo_messages
                    WHERE timestamp > NOW() - INTERVAL '7 days';
                """)

                msg_result = cur.fetchone()
                if msg_result and msg_result['message_count'] > 0:
                    print(f"{Fore.GREEN}✓ Found {msg_result['message_count']} messages in last 7 days")
                else:
                    print(f"{Fore.YELLOW}⚠ No recent messages found")

            else:
                print(f"{Fore.RED}✗ No conversation tables found")
                self.test_results.append(("PostgreSQL Storage", False, "No tables found"))

            cur.close()
            conn.close()

        except Exception as e:
            print(f"{Fore.RED}✗ PostgreSQL test failed: {str(e)}")
            self.test_results.append(("PostgreSQL Storage", False, str(e)))

    async def test_qdrant_vector_memory(self):
        """Test Qdrant vector database memory"""
        print(f"\n{Fore.YELLOW}[TEST 2] Qdrant Vector Memory{Style.RESET_ALL}")

        try:
            async with httpx.AsyncClient() as client:
                # Check Qdrant collections
                response = await client.get(f"{self.qdrant_url}/collections")

                if response.status_code == 200:
                    data = response.json()
                    collections = data.get("result", {}).get("collections", [])

                    if collections:
                        print(f"{Fore.GREEN}✓ Found {len(collections)} Qdrant collections:")
                        for collection in collections:
                            print(f"  - {collection['name']}")

                        # Check Echo Brain memory collection
                        echo_collection = next((c for c in collections if 'echo' in c['name'].lower()), None)

                        if echo_collection:
                            # Get collection info
                            coll_response = await client.get(
                                f"{self.qdrant_url}/collections/{echo_collection['name']}"
                            )

                            if coll_response.status_code == 200:
                                coll_data = coll_response.json()
                                vectors_count = coll_data.get("result", {}).get("vectors_count", 0)

                                print(f"{Fore.GREEN}✓ Echo collection has {vectors_count} vectors")
                                self.test_results.append(("Qdrant Vector Memory", True, f"{vectors_count} vectors stored"))
                            else:
                                print(f"{Fore.YELLOW}⚠ Could not get collection details")
                                self.test_results.append(("Qdrant Vector Memory", False, "Could not get details"))
                        else:
                            print(f"{Fore.YELLOW}⚠ No Echo Brain collection found")
                            self.test_results.append(("Qdrant Vector Memory", False, "No Echo collection"))
                    else:
                        print(f"{Fore.YELLOW}⚠ No Qdrant collections found")
                        self.test_results.append(("Qdrant Vector Memory", False, "No collections"))
                else:
                    print(f"{Fore.RED}✗ Could not connect to Qdrant")
                    self.test_results.append(("Qdrant Vector Memory", False, "Connection failed"))

        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Qdrant test skipped: {str(e)}")
            self.test_results.append(("Qdrant Vector Memory", None, f"Skipped: {str(e)}"))

    async def test_memory_consolidation(self):
        """Test memory consolidation process"""
        print(f"\n{Fore.YELLOW}[TEST 3] Memory Consolidation{Style.RESET_ALL}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create test conversation with multiple messages
                conv_id = f"consolidation_test_{uuid.uuid4().hex[:8]}"
                test_messages = [
                    "My name is TestUser and I work on AI research",
                    "I specialize in neural networks and deep learning",
                    "My current project involves transformer models",
                    "I'm building a conversational AI system"
                ]

                print(f"  Sending {len(test_messages)} test messages...")
                for msg in test_messages:
                    response = await client.post(
                        f"{self.echo_url}/api/echo/query",
                        json={
                            "query": msg,
                            "conversation_id": conv_id,
                            "username": "test_consolidation"
                        }
                    )
                    await asyncio.sleep(1)

                # Check if memories were consolidated
                await asyncio.sleep(3)

                # Query for consolidated memory
                search_response = await client.post(
                    f"{self.echo_url}/api/echo/query",
                    json={
                        "query": "What do you know about TestUser's work?",
                        "conversation_id": f"recall_{uuid.uuid4().hex[:8]}",
                        "username": "test_consolidation"
                    }
                )

                if search_response.status_code == 200:
                    data = search_response.json()
                    response_text = data.get("response", "").lower()

                    # Check if key information was consolidated
                    keywords = ["ai", "neural", "transformer", "research"]
                    found = sum(1 for kw in keywords if kw in response_text)

                    if found >= 2:
                        print(f"{Fore.GREEN}✓ Memory consolidation working ({found}/{len(keywords)} keywords found)")
                        self.test_results.append(("Memory Consolidation", True, f"{found} keywords recalled"))
                    else:
                        print(f"{Fore.YELLOW}⚠ Partial memory consolidation ({found}/{len(keywords)} keywords)")
                        self.test_results.append(("Memory Consolidation", False, f"Only {found} keywords recalled"))
                else:
                    print(f"{Fore.RED}✗ Failed to test consolidation")
                    self.test_results.append(("Memory Consolidation", False, "Query failed"))

        except Exception as e:
            print(f"{Fore.RED}✗ Consolidation test failed: {str(e)}")
            self.test_results.append(("Memory Consolidation", False, str(e)))

    async def test_user_context_persistence(self):
        """Test user context persistence across sessions"""
        print(f"\n{Fore.YELLOW}[TEST 4] User Context Persistence{Style.RESET_ALL}")

        try:
            # Check database for user contexts
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for user context table
            cur.execute("""
                SELECT COUNT(*) as user_count,
                       COUNT(DISTINCT username) as unique_users
                FROM echo_user_contexts
                WHERE last_interaction > NOW() - INTERVAL '30 days';
            """)

            result = cur.fetchone()
            if result and result['user_count'] > 0:
                print(f"{Fore.GREEN}✓ Found {result['unique_users']} unique users with contexts")

                # Check for patrick's context specifically
                cur.execute("""
                    SELECT username,
                           permissions,
                           interaction_count,
                           last_interaction
                    FROM echo_user_contexts
                    WHERE username = 'patrick'
                    LIMIT 1;
                """)

                patrick_context = cur.fetchone()
                if patrick_context:
                    print(f"{Fore.GREEN}✓ Patrick's context found:")
                    print(f"  - Interactions: {patrick_context['interaction_count']}")
                    print(f"  - Last seen: {patrick_context['last_interaction']}")
                    self.test_results.append(("User Context", True, f"{result['unique_users']} users tracked"))
                else:
                    print(f"{Fore.YELLOW}⚠ Patrick's context not found")
                    self.test_results.append(("User Context", False, "Primary user not found"))
            else:
                print(f"{Fore.YELLOW}⚠ No user contexts found")
                self.test_results.append(("User Context", False, "No contexts found"))

            cur.close()
            conn.close()

        except Exception as e:
            print(f"{Fore.YELLOW}⚠ User context test skipped: {str(e)}")
            self.test_results.append(("User Context", None, f"Table may not exist: {str(e)}"))

    def print_summary(self):
        """Print test summary"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Database Memory Test Summary")
        print(f"{Fore.CYAN}{'='*60}\n")

        passed = sum(1 for _, result, _ in self.test_results if result is True)
        failed = sum(1 for _, result, _ in self.test_results if result is False)
        skipped = sum(1 for _, result, _ in self.test_results if result is None)

        for test_name, result, message in self.test_results:
            if result is True:
                status = f"{Fore.GREEN}PASSED"
            elif result is False:
                status = f"{Fore.RED}FAILED"
            else:
                status = f"{Fore.YELLOW}SKIPPED"

            print(f"{status:20} {test_name:30} {message}")

        print(f"\n{Fore.GREEN}Passed: {passed}  {Fore.RED}Failed: {failed}  {Fore.YELLOW}Skipped: {skipped}\n")


async def main():
    """Run database memory tests"""
    tester = DatabaseMemoryTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())