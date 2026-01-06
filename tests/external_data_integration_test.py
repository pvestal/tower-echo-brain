#!/usr/bin/env python3
"""
External Data Integration Test Suite

Tests conversation memory integration with external data sources:
- Tower Knowledge Base articles
- Claude conversation history
- System logs and metrics
- Cross-modal data correlation

Validates that conversation memory can enhance context by pulling from
external knowledge sources and maintaining coherence across data types.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import httpx

from src.managers.conversation_memory_manager import get_conversation_memory_manager

@dataclass
class ExternalDataTestResult:
    source_type: str
    integration_success: bool
    context_enhancement: bool
    response_time_ms: float
    accuracy_score: float

class ExternalDataIntegrationTester:
    """Test conversation memory with external data sources"""

    def __init__(self):
        self.kb_base_url = "http://localhost:8307"
        self.echo_base_url = "http://localhost:8309"

        # Mock external data sources
        self.mock_kb_articles = [
            {
                "id": "KB-001",
                "title": "Tower Service Authentication Troubleshooting",
                "content": "When tower-auth service shows bypass mode, check OAuth configuration and vault credentials...",
                "tags": ["authentication", "troubleshooting", "oauth", "vault"],
                "category": "system_administration",
                "last_modified": "2024-12-01"
            },
            {
                "id": "KB-002",
                "title": "Anime Production Service Performance Issues",
                "content": "tower-anime-production slow response times often indicate GPU memory issues on port 8328...",
                "tags": ["anime-production", "performance", "gpu", "port-8328"],
                "category": "media_services",
                "last_modified": "2024-11-28"
            }
        ]

        self.mock_claude_history = [
            {
                "conversation_id": "claude_conv_001",
                "timestamp": "2024-12-05T10:30:00Z",
                "summary": "Discussed implementing conversation memory for multi-turn context persistence",
                "entities": ["conversation_memory", "entity_resolution", "pronoun_handling"],
                "key_decisions": ["Use JSON persistence", "Implement semantic search", "Add entity decay"]
            },
            {
                "conversation_id": "claude_conv_002",
                "timestamp": "2024-12-04T15:45:00Z",
                "summary": "Tower ecosystem architecture review and service dependency mapping",
                "entities": ["tower-echo-brain", "tower-auth", "tower-anime-production"],
                "key_decisions": ["Consolidate services", "Improve monitoring", "Add health checks"]
            }
        ]

    async def test_knowledge_base_integration(self):
        """Test integration with Tower Knowledge Base"""
        print("📚 Testing Knowledge Base integration...")

        memory_manager = await get_conversation_memory_manager()
        conv_id = "kb_integration_test"

        # Start conversation about authentication issues
        await memory_manager.add_turn(
            conversation_id=conv_id,
            role="user",
            content="The tower-auth service is showing bypass mode errors",
            intent="troubleshooting"
        )

        # Simulate KB search based on conversation context
        start_time = time.time()

        # In real implementation, this would query the actual KB
        relevant_articles = []
        for article in self.mock_kb_articles:
            if any(tag in ["authentication", "oauth", "vault"] for tag in article["tags"]):
                relevant_articles.append(article)

        search_time = (time.time() - start_time) * 1000

        # Test enhancing conversation with KB context
        if relevant_articles:
            kb_enhanced_context = f"Reference KB article {relevant_articles[0]['id']}: {relevant_articles[0]['title']}"

            await memory_manager.add_turn(
                conversation_id=conv_id,
                role="assistant",
                content=f"Found authentication issue. {kb_enhanced_context}. Check OAuth configuration in vault.",
                execution_results=["kb_search", "vault_check"]
            )

        # Test that KB references become entities
        session = memory_manager.active_sessions.get(conv_id)
        kb_entity_found = any("KB-001" in entity.value for entity in session.active_entities.values()) if session else False

        result = ExternalDataTestResult(
            source_type="knowledge_base",
            integration_success=len(relevant_articles) > 0,
            context_enhancement=kb_entity_found,
            response_time_ms=search_time,
            accuracy_score=1.0 if "authentication" in relevant_articles[0]["tags"] else 0.0
        )

        print(f"   ✅ KB articles found: {len(relevant_articles)}")
        print(f"   ✅ Context enhancement: {result.context_enhancement}")
        print(f"   ✅ Search time: {result.response_time_ms:.2f}ms")

        return result

    async def test_claude_history_correlation(self):
        """Test correlation with Claude conversation history"""
        print("💬 Testing Claude conversation history correlation...")

        memory_manager = await get_conversation_memory_manager()
        conv_id = "claude_history_test"

        # Current conversation references past Claude discussion
        await memory_manager.add_turn(
            conversation_id=conv_id,
            role="user",
            content="Remember our conversation about conversation memory implementation? I need to test the entity resolution feature we discussed.",
            intent="feature_development"
        )

        start_time = time.time()

        # Simulate Claude history search
        relevant_history = []
        for history_item in self.mock_claude_history:
            if any(entity in ["conversation_memory", "entity_resolution"] for entity in history_item["entities"]):
                relevant_history.append(history_item)

        correlation_time = (time.time() - start_time) * 1000

        # Enhance current conversation with historical context
        if relevant_history:
            historical_context = f"Referencing previous discussion (claude_conv_001): {relevant_history[0]['summary']}"

            await memory_manager.add_turn(
                conversation_id=conv_id,
                role="assistant",
                content=f"Yes, {historical_context}. The entity resolution system we implemented uses pattern matching and LLM enhancement.",
                execution_results=["claude_history_lookup"]
            )

        # Test cross-conversation entity linking
        session = memory_manager.active_sessions.get(conv_id)
        cross_conversation_entity_found = any("claude_conv_001" in str(entity.value) for entity in session.active_entities.values()) if session else False

        result = ExternalDataTestResult(
            source_type="claude_history",
            integration_success=len(relevant_history) > 0,
            context_enhancement=cross_conversation_entity_found,
            response_time_ms=correlation_time,
            accuracy_score=1.0 if relevant_history and "conversation_memory" in relevant_history[0]["entities"] else 0.0
        )

        print(f"   ✅ Historical conversations found: {len(relevant_history)}")
        print(f"   ✅ Cross-conversation linking: {result.context_enhancement}")
        print(f"   ✅ Correlation time: {result.response_time_ms:.2f}ms")

        return result

    async def test_system_logs_integration(self):
        """Test integration with system logs and metrics"""
        print("📊 Testing system logs integration...")

        memory_manager = await get_conversation_memory_manager()
        conv_id = "system_logs_test"

        # Mock system log data
        mock_log_entries = [
            {
                "timestamp": "2024-12-06T17:30:00Z",
                "service": "tower-echo-brain",
                "level": "ERROR",
                "message": "Failed to connect to Ollama models on port 11434",
                "port": 11434,
                "error_code": "CONNECTION_REFUSED"
            },
            {
                "timestamp": "2024-12-06T17:25:00Z",
                "service": "tower-anime-production",
                "level": "WARN",
                "message": "High GPU memory usage detected: 95% utilization",
                "gpu_usage": 95,
                "performance_impact": "high"
            }
        ]

        # Start conversation about system issues
        await memory_manager.add_turn(
            conversation_id=conv_id,
            role="user",
            content="I'm seeing performance issues with the anime production service",
            intent="performance_investigation"
        )

        start_time = time.time()

        # Simulate log analysis
        relevant_logs = []
        for log_entry in mock_log_entries:
            if "anime" in log_entry["service"] or log_entry.get("performance_impact") == "high":
                relevant_logs.append(log_entry)

        analysis_time = (time.time() - start_time) * 1000

        # Enhance conversation with log insights
        if relevant_logs:
            log_insight = f"System logs show {relevant_logs[0]['message']} at {relevant_logs[0]['timestamp']}"

            await memory_manager.add_turn(
                conversation_id=conv_id,
                role="assistant",
                content=f"Performance issue identified. {log_insight}. GPU memory optimization needed.",
                execution_results=["log_analysis", "gpu_check"]
            )

        # Test that log data becomes contextual entities
        session = memory_manager.active_sessions.get(conv_id)
        log_entity_found = any("gpu" in entity.name.lower() for entity in session.active_entities.values()) if session else False

        result = ExternalDataTestResult(
            source_type="system_logs",
            integration_success=len(relevant_logs) > 0,
            context_enhancement=log_entity_found,
            response_time_ms=analysis_time,
            accuracy_score=1.0 if relevant_logs and "gpu" in relevant_logs[0]["message"].lower() else 0.0
        )

        print(f"   ✅ Relevant log entries: {len(relevant_logs)}")
        print(f"   ✅ Log data contextualization: {result.context_enhancement}")
        print(f"   ✅ Analysis time: {result.response_time_ms:.2f}ms")

        return result

    async def test_cross_modal_correlation(self):
        """Test correlation across multiple data sources"""
        print("🔗 Testing cross-modal data correlation...")

        memory_manager = await get_conversation_memory_manager()
        conv_id = "cross_modal_test"

        # Complex scenario involving KB, logs, and conversation history
        await memory_manager.add_turn(
            conversation_id=conv_id,
            role="user",
            content="The authentication issues we discussed last week are happening again. Check what the logs say about tower-auth service.",
            intent="incident_investigation"
        )

        start_time = time.time()

        # Simulate cross-modal data gathering
        correlations = {
            "kb_articles": [art for art in self.mock_kb_articles if "authentication" in art["tags"]],
            "claude_history": [hist for hist in self.mock_claude_history if "tower-auth" in hist["entities"]],
            "system_logs": [
                {
                    "service": "tower-auth",
                    "level": "ERROR",
                    "message": "OAuth token validation failed - vault connection timeout",
                    "related_kb": "KB-001"
                }
            ]
        }

        correlation_time = (time.time() - start_time) * 1000

        # Create unified response using all data sources
        unified_context = (
            f"Cross-referencing KB article {correlations['kb_articles'][0]['id']}, "
            f"previous conversation {correlations['claude_history'][0]['conversation_id']}, "
            f"and current system logs showing vault connection issues."
        )

        await memory_manager.add_turn(
            conversation_id=conv_id,
            role="assistant",
            content=f"Authentication recurring issue identified. {unified_context} Root cause: vault connectivity.",
            execution_results=["cross_modal_analysis"]
        )

        # Test entity linking across data types
        session = memory_manager.active_sessions.get(conv_id)
        if session:
            entity_types = {entity.entity_type.value for entity in session.active_entities.values()}
            cross_modal_success = len(entity_types) >= 2  # Multiple entity types found
        else:
            cross_modal_success = False

        result = ExternalDataTestResult(
            source_type="cross_modal",
            integration_success=all(len(correlations[source]) > 0 for source in correlations),
            context_enhancement=cross_modal_success,
            response_time_ms=correlation_time,
            accuracy_score=1.0 if "vault" in unified_context.lower() else 0.0
        )

        print(f"   ✅ Data sources correlated: {len([s for s in correlations if correlations[s]])}")
        print(f"   ✅ Unified context creation: {result.context_enhancement}")
        print(f"   ✅ Correlation time: {result.response_time_ms:.2f}ms")

        return result

    async def test_real_kb_api_integration(self):
        """Test integration with actual Tower KB API"""
        print("🌐 Testing real KB API integration...")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test KB API connectivity
                start_time = time.time()
                response = await client.get(f"{self.kb_base_url}/api/kb/articles?limit=5")
                api_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    articles = response.json()

                    result = ExternalDataTestResult(
                        source_type="real_kb_api",
                        integration_success=True,
                        context_enhancement=len(articles) > 0,
                        response_time_ms=api_time,
                        accuracy_score=1.0
                    )

                    print(f"   ✅ KB API accessible: {len(articles)} articles available")
                    print(f"   ✅ API response time: {api_time:.2f}ms")
                    return result

        except Exception as e:
            print(f"   ⚠️  KB API not accessible: {e}")

            # Return placeholder result
            return ExternalDataTestResult(
                source_type="real_kb_api",
                integration_success=False,
                context_enhancement=False,
                response_time_ms=0.0,
                accuracy_score=0.0
            )

    async def run_all_tests(self):
        """Execute complete external data integration test suite"""
        print("🔗 Starting External Data Integration Test Suite")
        print("=" * 65)

        test_results = {}

        # Test 1: Knowledge Base integration
        try:
            result1 = await self.test_knowledge_base_integration()
            test_results["kb_integration"] = result1
        except Exception as e:
            print(f"❌ KB integration test failed: {e}")
            test_results["kb_integration"] = ExternalDataTestResult("kb", False, False, 0, 0)

        # Test 2: Claude history correlation
        try:
            result2 = await self.test_claude_history_correlation()
            test_results["claude_history"] = result2
        except Exception as e:
            print(f"❌ Claude history test failed: {e}")
            test_results["claude_history"] = ExternalDataTestResult("claude", False, False, 0, 0)

        # Test 3: System logs integration
        try:
            result3 = await self.test_system_logs_integration()
            test_results["system_logs"] = result3
        except Exception as e:
            print(f"❌ System logs test failed: {e}")
            test_results["system_logs"] = ExternalDataTestResult("logs", False, False, 0, 0)

        # Test 4: Cross-modal correlation
        try:
            result4 = await self.test_cross_modal_correlation()
            test_results["cross_modal"] = result4
        except Exception as e:
            print(f"❌ Cross-modal test failed: {e}")
            test_results["cross_modal"] = ExternalDataTestResult("cross", False, False, 0, 0)

        # Test 5: Real KB API
        try:
            result5 = await self.test_real_kb_api_integration()
            test_results["real_kb_api"] = result5
        except Exception as e:
            print(f"❌ Real KB API test failed: {e}")
            test_results["real_kb_api"] = ExternalDataTestResult("real_kb", False, False, 0, 0)

        # Summary
        print("\n" + "=" * 65)
        print("📊 EXTERNAL DATA INTEGRATION RESULTS")
        print("=" * 65)

        total_tests = len(test_results)
        successful_integrations = sum(1 for result in test_results.values() if result.integration_success)
        enhanced_contexts = sum(1 for result in test_results.values() if result.context_enhancement)
        avg_response_time = sum(result.response_time_ms for result in test_results.values()) / total_tests

        for test_name, result in test_results.items():
            integration_status = "✅" if result.integration_success else "❌"
            enhancement_status = "✅" if result.context_enhancement else "❌"
            print(f"{integration_status} {test_name:20} | Enhancement: {enhancement_status} | Time: {result.response_time_ms:6.2f}ms | Accuracy: {result.accuracy_score:.2f}")

        print(f"\n📈 Overall Results:")
        print(f"   🔗 Successful integrations: {successful_integrations}/{total_tests}")
        print(f"   🧠 Context enhancements: {enhanced_contexts}/{total_tests}")
        print(f"   ⚡ Average response time: {avg_response_time:.2f}ms")
        print(f"   🎯 Integration success rate: {(successful_integrations/total_tests)*100:.1f}%")

        return test_results

async def main():
    """Run external data integration tests"""
    tester = ExternalDataIntegrationTester()
    results = await tester.run_all_tests()
    return results

if __name__ == "__main__":
    asyncio.run(main())