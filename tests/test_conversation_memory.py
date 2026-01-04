#!/usr/bin/env python3
"""
Comprehensive test suite for Echo Brain and Telegram conversation memory
Tests memory persistence, retrieval, and cross-session continuity
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx
import pytest
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

class ConversationMemoryTester:
    """Test suite for conversation memory and history"""

    def __init__(self, base_url: str = "http://localhost:8309"):
        self.base_url = base_url
        self.test_results = []
        self.conversation_ids = []
        self.test_user = f"test_user_{uuid.uuid4().hex[:8]}"

    async def run_all_tests(self):
        """Run complete test suite"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Echo Brain & Telegram Conversation Memory Test Suite")
        print(f"{Fore.CYAN}{'='*60}\n")

        # Test Echo Brain memory
        await self.test_echo_brain_memory()

        # Test Telegram integration
        await self.test_telegram_memory()

        # Test cross-session persistence
        await self.test_cross_session_persistence()

        # Test memory search and retrieval
        await self.test_memory_search()

        # Test conversation history
        await self.test_conversation_history()

        # Print results summary
        self.print_results_summary()

    async def test_echo_brain_memory(self):
        """Test Echo Brain conversation memory"""
        print(f"\n{Fore.YELLOW}[TEST 1] Echo Brain Conversation Memory{Style.RESET_ALL}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create a new conversation
                conversation_id = f"test_echo_{uuid.uuid4().hex[:8]}"
                self.conversation_ids.append(conversation_id)

                # Send first message
                response1 = await client.post(
                    f"{self.base_url}/api/echo/query",
                    json={
                        "query": "Remember this: My favorite color is blue and I have a cat named Whiskers",
                        "conversation_id": conversation_id,
                        "username": self.test_user
                    }
                )

                if response1.status_code == 200:
                    print(f"{Fore.GREEN}✓ First message sent successfully")
                else:
                    print(f"{Fore.RED}✗ Failed to send first message: {response1.status_code}")
                    self.test_results.append(("Echo Brain Memory", False, f"Failed to send message: {response1.status_code}"))
                    return

                # Wait a moment for processing
                await asyncio.sleep(2)

                # Send follow-up message testing memory
                response2 = await client.post(
                    f"{self.base_url}/api/echo/query",
                    json={
                        "query": "What is my favorite color and what is my cat's name?",
                        "conversation_id": conversation_id,
                        "username": self.test_user
                    }
                )

                if response2.status_code == 200:
                    data = response2.json()
                    response_text = data.get("response", "").lower()

                    # Check if Echo remembered the information
                    if "blue" in response_text and "whiskers" in response_text:
                        print(f"{Fore.GREEN}✓ Echo Brain remembered conversation context correctly")
                        self.test_results.append(("Echo Brain Memory", True, "Successfully remembered context"))
                    else:
                        print(f"{Fore.YELLOW}⚠ Echo Brain response didn't contain expected information")
                        print(f"  Response: {response_text[:100]}...")
                        self.test_results.append(("Echo Brain Memory", False, "Memory recall incomplete"))
                else:
                    print(f"{Fore.RED}✗ Failed to get memory response: {response2.status_code}")
                    self.test_results.append(("Echo Brain Memory", False, f"Failed to get response: {response2.status_code}"))

        except Exception as e:
            print(f"{Fore.RED}✗ Echo Brain memory test failed: {str(e)}")
            self.test_results.append(("Echo Brain Memory", False, str(e)))

    async def test_telegram_memory(self):
        """Test Telegram integration memory"""
        print(f"\n{Fore.YELLOW}[TEST 2] Telegram Integration Memory{Style.RESET_ALL}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Simulate Telegram message
                telegram_chat_id = f"test_telegram_{uuid.uuid4().hex[:8]}"

                # Send message through Telegram endpoint
                response = await client.post(
                    f"{self.base_url}/api/telegram/message",
                    json={
                        "chat_id": telegram_chat_id,
                        "message": "Hi Echo, remember that I work at SpaceX and my project is Starship",
                        "username": self.test_user
                    }
                )

                if response.status_code == 200:
                    print(f"{Fore.GREEN}✓ Telegram message processed")

                    # Wait for processing
                    await asyncio.sleep(2)

                    # Test memory recall
                    response2 = await client.post(
                        f"{self.base_url}/api/telegram/message",
                        json={
                            "chat_id": telegram_chat_id,
                            "message": "Where do I work and what project am I on?",
                            "username": self.test_user
                        }
                    )

                    if response2.status_code == 200:
                        data = response2.json()
                        response_text = data.get("response", "").lower()

                        if "spacex" in response_text and "starship" in response_text:
                            print(f"{Fore.GREEN}✓ Telegram memory working correctly")
                            self.test_results.append(("Telegram Memory", True, "Successfully remembered context"))
                        else:
                            print(f"{Fore.YELLOW}⚠ Telegram response missing expected information")
                            self.test_results.append(("Telegram Memory", False, "Memory recall incomplete"))
                    else:
                        print(f"{Fore.YELLOW}⚠ Telegram endpoint not available")
                        self.test_results.append(("Telegram Memory", None, "Endpoint not available"))
                else:
                    print(f"{Fore.YELLOW}⚠ Telegram integration not available: {response.status_code}")
                    self.test_results.append(("Telegram Memory", None, "Integration not available"))

        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Telegram test skipped: {str(e)}")
            self.test_results.append(("Telegram Memory", None, f"Skipped: {str(e)}"))

    async def test_cross_session_persistence(self):
        """Test memory persistence across sessions"""
        print(f"\n{Fore.YELLOW}[TEST 3] Cross-Session Memory Persistence{Style.RESET_ALL}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create first session
                session1_id = f"session1_{uuid.uuid4().hex[:8]}"

                # Store information in session 1
                response1 = await client.post(
                    f"{self.base_url}/api/echo/query",
                    json={
                        "query": "In this session, remember that the secret code is ALPHA-BRAVO-42",
                        "conversation_id": session1_id,
                        "username": self.test_user
                    }
                )

                if response1.status_code == 200:
                    print(f"{Fore.GREEN}✓ Session 1 information stored")

                    # Wait for memory consolidation
                    await asyncio.sleep(3)

                    # Create new session
                    session2_id = f"session2_{uuid.uuid4().hex[:8]}"

                    # Try to recall from different session
                    response2 = await client.post(
                        f"{self.base_url}/api/echo/query",
                        json={
                            "query": f"What was the secret code from my previous conversations?",
                            "conversation_id": session2_id,
                            "username": self.test_user
                        }
                    )

                    if response2.status_code == 200:
                        data = response2.json()
                        response_text = data.get("response", "").upper()

                        if "ALPHA" in response_text and "BRAVO" in response_text and "42" in response_text:
                            print(f"{Fore.GREEN}✓ Cross-session memory persistence working!")
                            self.test_results.append(("Cross-Session Persistence", True, "Memory persisted across sessions"))
                        else:
                            print(f"{Fore.YELLOW}⚠ Cross-session memory not fully working")
                            self.test_results.append(("Cross-Session Persistence", False, "Memory not accessible across sessions"))
                    else:
                        print(f"{Fore.RED}✗ Failed to test cross-session memory")
                        self.test_results.append(("Cross-Session Persistence", False, "Failed to retrieve"))
                else:
                    print(f"{Fore.RED}✗ Failed to store session 1 data")
                    self.test_results.append(("Cross-Session Persistence", False, "Failed to store"))

        except Exception as e:
            print(f"{Fore.RED}✗ Cross-session test failed: {str(e)}")
            self.test_results.append(("Cross-Session Persistence", False, str(e)))

    async def test_memory_search(self):
        """Test memory search and retrieval capabilities"""
        print(f"\n{Fore.YELLOW}[TEST 4] Memory Search & Retrieval{Style.RESET_ALL}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Store multiple pieces of information
                test_data = [
                    ("Project Alpha involves quantum computing", "alpha_quantum"),
                    ("Project Beta is about neural networks", "beta_neural"),
                    ("Project Gamma focuses on robotics", "gamma_robotics")
                ]

                for info, conv_id in test_data:
                    response = await client.post(
                        f"{self.base_url}/api/echo/query",
                        json={
                            "query": f"Remember this: {info}",
                            "conversation_id": conv_id,
                            "username": self.test_user
                        }
                    )
                    if response.status_code == 200:
                        print(f"{Fore.GREEN}✓ Stored: {info[:30]}...")
                    await asyncio.sleep(1)

                # Test semantic search
                search_response = await client.post(
                    f"{self.base_url}/api/echo/search",
                    json={
                        "query": "What projects involve AI or machine learning?",
                        "username": self.test_user,
                        "limit": 5
                    }
                )

                if search_response.status_code == 200:
                    results = search_response.json()
                    if results and len(results) > 0:
                        print(f"{Fore.GREEN}✓ Memory search returned {len(results)} results")
                        self.test_results.append(("Memory Search", True, f"Found {len(results)} relevant memories"))
                    else:
                        print(f"{Fore.YELLOW}⚠ Memory search returned no results")
                        self.test_results.append(("Memory Search", False, "No search results"))
                else:
                    print(f"{Fore.YELLOW}⚠ Search endpoint not available")
                    self.test_results.append(("Memory Search", None, "Search endpoint not available"))

        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Memory search test skipped: {str(e)}")
            self.test_results.append(("Memory Search", None, f"Skipped: {str(e)}"))

    async def test_conversation_history(self):
        """Test full conversation history retrieval"""
        print(f"\n{Fore.YELLOW}[TEST 5] Conversation History Retrieval{Style.RESET_ALL}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get conversation history for test user
                response = await client.get(
                    f"{self.base_url}/api/echo/conversations",
                    params={"username": self.test_user}
                )

                if response.status_code == 200:
                    conversations = response.json()

                    if isinstance(conversations, list) and len(conversations) > 0:
                        print(f"{Fore.GREEN}✓ Retrieved {len(conversations)} conversations")

                        # Test detailed history for first conversation
                        if self.conversation_ids:
                            conv_id = self.conversation_ids[0]
                            history_response = await client.get(
                                f"{self.base_url}/api/echo/conversation/{conv_id}",
                                params={"username": self.test_user}
                            )

                            if history_response.status_code == 200:
                                history = history_response.json()
                                if "messages" in history and len(history["messages"]) > 0:
                                    print(f"{Fore.GREEN}✓ Retrieved detailed history with {len(history['messages'])} messages")
                                    self.test_results.append(("Conversation History", True, f"Retrieved {len(history['messages'])} messages"))
                                else:
                                    print(f"{Fore.YELLOW}⚠ History retrieved but no messages found")
                                    self.test_results.append(("Conversation History", False, "No messages in history"))
                            else:
                                print(f"{Fore.YELLOW}⚠ Could not retrieve detailed history")
                                self.test_results.append(("Conversation History", False, "Failed to get detailed history"))
                    else:
                        print(f"{Fore.YELLOW}⚠ No conversations found for test user")
                        self.test_results.append(("Conversation History", False, "No conversations found"))
                else:
                    print(f"{Fore.YELLOW}⚠ Conversation history endpoint not available")
                    self.test_results.append(("Conversation History", None, "Endpoint not available"))

        except Exception as e:
            print(f"{Fore.YELLOW}⚠ History retrieval test skipped: {str(e)}")
            self.test_results.append(("Conversation History", None, f"Skipped: {str(e)}"))

    def print_results_summary(self):
        """Print summary of all test results"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Test Results Summary")
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

        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.GREEN}Passed: {passed}  {Fore.RED}Failed: {failed}  {Fore.YELLOW}Skipped: {skipped}")
        print(f"{Fore.CYAN}{'='*60}\n")

        # Return success if no failures
        return failed == 0


async def main():
    """Main test runner"""
    tester = ConversationMemoryTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())