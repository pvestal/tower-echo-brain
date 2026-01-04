#!/usr/bin/env python3
"""
REAL tests for actual vector search functionality
Tests the REAL Qdrant vectors, not mocks
"""

import pytest
import sys
import os
sys.path.append('/opt/tower-echo-brain')

from src.services.real_vector_search import RealVectorSearch


class TestRealVectorSearch:
    """Test actual vector search against real Qdrant data"""

    @classmethod
    def setup_class(cls):
        """Setup real vector search instance"""
        cls.search = RealVectorSearch()

    def test_vector_search_initialization(self):
        """Test that vector search connects to Qdrant"""
        assert self.search.qdrant is not None
        assert self.search.encoder_384 is not None
        assert self.search.encoder_768 is not None

    def test_get_real_stats(self):
        """Test getting actual statistics from Qdrant"""
        stats = self.search.get_stats()

        # Should have actual collections
        assert stats['total_vectors'] > 0
        assert 'claude_conversations_4096d' in stats
        assert 'learning_facts_4096d' in stats

        # Check Claude conversations
        claude_stats = stats['claude_conversations_4096d']
        assert claude_stats['count'] > 10000  # Should have 12,228
        assert claude_stats['dimensions'] == 4096
        assert claude_stats['status'] == 'green'

    def test_search_claude_conversations(self):
        """Test searching actual Claude conversations"""
        results = self.search.search_vectors(
            "anime production system",
            "claude_conversations_4096d",
            limit=5
        )

        # Should find actual results
        assert len(results) > 0
        assert all('score' in r for r in results)
        assert all('payload' in r for r in results)
        assert all(r['collection'] == 'claude_conversations_4096d' for r in results)

    def test_search_learning_facts(self):
        """Test searching learning facts"""
        results = self.search.search_vectors(
            "Echo Brain intelligence",
            "learning_facts_4096d",
            limit=3
        )

        # Should find learning facts
        assert len(results) > 0
        for result in results:
            assert 'score' in result
            assert result['collection'] == 'learning_facts_4096d'

    def test_search_all_collections(self):
        """Test searching across all collections"""
        results = self.search.search_all_collections(
            "Tower architecture",
            limit_per_collection=2
        )

        # Should search multiple collections
        assert len(results) > 0
        collections_found = list(results.keys())

        # Should include major collections
        major_collections = ['claude_conversations_4096d', 'learning_facts_4096d']
        found_major = any(col in collections_found for col in major_collections)
        assert found_major

    def test_dimension_matching(self):
        """Test that search handles different vector dimensions"""
        # Test 384D collection
        if 'agent_memories' in self.search.get_stats():
            results_384 = self.search.search_vectors(
                "test query", "agent_memories", limit=1
            )
            # Should handle 384D without errors
            assert isinstance(results_384, list)

        # Test 768D collection
        if 'learning_facts' in self.search.get_stats():
            results_768 = self.search.search_vectors(
                "test query", "learning_facts", limit=1
            )
            # Should handle 768D without errors
            assert isinstance(results_768, list)

        # Test 4096D collection
        results_4096 = self.search.search_vectors(
            "test query", "claude_conversations_4096d", limit=1
        )
        # Should handle 4096D without errors
        assert isinstance(results_4096, list)

    def test_vector_count_accuracy(self):
        """Verify the actual vector counts match what we claim"""
        stats = self.search.get_stats()

        # Total should be around 40,554 vectors
        total = stats['total_vectors']
        assert total > 40000, f"Expected >40k vectors, got {total}"

        # Claude conversations should be 12,228
        claude_count = stats['claude_conversations_4096d']['count']
        assert claude_count > 12000, f"Expected >12k Claude conversations, got {claude_count}"

        # Learning facts should be substantial
        learning_count = stats['learning_facts_4096d']['count']
        assert learning_count > 6000, f"Expected >6k learning facts, got {learning_count}"

    def test_search_quality(self):
        """Test that search returns relevant results"""
        # Search for specific anime terms
        anime_results = self.search.search_vectors(
            "ComfyUI anime generation workflow",
            "claude_conversations_4096d",
            limit=3
        )

        if anime_results:
            # Should have reasonable scores (similarity)
            scores = [r['score'] for r in anime_results]
            assert all(s >= 0 and s <= 1 for s in scores), f"Invalid scores: {scores}"

            # Results should be ordered by score (descending)
            assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    # Run tests directly
    test = TestRealVectorSearch()
    test.setup_class()

    print("Testing vector search initialization...")
    test.test_vector_search_initialization()
    print("✅ Initialization test passed")

    print("Testing real statistics...")
    test.test_get_real_stats()
    print("✅ Statistics test passed")

    print("Testing Claude conversation search...")
    test.test_search_claude_conversations()
    print("✅ Claude conversation search passed")

    print("Testing learning facts search...")
    test.test_search_learning_facts()
    print("✅ Learning facts search passed")

    print("Testing multi-collection search...")
    test.test_search_all_collections()
    print("✅ Multi-collection search passed")

    print("Testing dimension handling...")
    test.test_dimension_matching()
    print("✅ Dimension matching passed")

    print("Testing vector count accuracy...")
    test.test_vector_count_accuracy()
    print("✅ Vector count accuracy passed")

    print("Testing search quality...")
    test.test_search_quality()
    print("✅ Search quality passed")

    print("\n🎯 ALL REAL VECTOR SEARCH TESTS PASSED")
    print("Vector search is working with actual Qdrant data!")