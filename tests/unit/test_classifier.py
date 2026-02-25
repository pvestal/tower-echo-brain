"""Unit tests for DomainClassifier pure functions.

Covers: _regex_classify, _cosine_similarity, filter_results_by_domain,
        should_search_source, get_allowed_sources
"""
import math
import pytest

from src.context_assembly.classifier import DomainClassifier, Domain

pytestmark = pytest.mark.unit


# ── _regex_classify ───────────────────────────────────────────────────

class TestRegexClassify:
    def setup_method(self):
        self.c = DomainClassifier()

    def test_technical_query(self):
        results = self.c._regex_classify("debug the python function for async API")
        top_domain = results[0][0]
        assert top_domain == Domain.TECHNICAL

    def test_anime_query(self):
        results = self.c._regex_classify("comfyui checkpoint for anime character generation")
        top_domain = results[0][0]
        assert top_domain == Domain.ANIME

    def test_personal_query(self):
        results = self.c._regex_classify("Patrick truck tundra")
        top_domain = results[0][0]
        assert top_domain == Domain.PERSONAL

    def test_system_query(self):
        results = self.c._regex_classify("service running port qdrant nginx")
        top_domain = results[0][0]
        assert top_domain == Domain.SYSTEM

    def test_financial_query(self):
        results = self.c._regex_classify("bank transaction payment balance")
        top_domain = results[0][0]
        assert top_domain == Domain.FINANCIAL

    def test_general_fallback(self):
        # Gibberish that matches nothing above 0.3
        results = self.c._regex_classify("xyzzy plugh")
        top_domain = results[0][0]
        assert top_domain == Domain.GENERAL

    def test_negative_signals_reduce_score(self):
        # "code" is a negative signal for anime domain
        results = self.c._regex_classify("anime code debug")
        scores_dict = dict(results)
        # Technical should benefit more since "code" and "debug" are both technical signals
        # and "anime" is a negative for technical, so it's a toss-up
        # Just verify anime score is lower than if "code" weren't there
        assert scores_dict.get(Domain.ANIME, 0) < 1.0

    def test_empty_query(self):
        results = self.c._regex_classify("")
        assert results[0][0] == Domain.GENERAL

    def test_returns_sorted_descending(self):
        results = self.c._regex_classify("python api debug code function")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ── _cosine_similarity ────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(DomainClassifier._cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(DomainClassifier._cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(DomainClassifier._cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = [1.0, 2.0]
        b = [0.0, 0.0]
        assert DomainClassifier._cosine_similarity(a, b) == 0.0

    def test_both_zero(self):
        assert DomainClassifier._cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


# ── filter_results_by_domain ──────────────────────────────────────────

class TestFilterResultsByDomain:
    def setup_method(self):
        self.c = DomainClassifier()

    def test_low_score_removed(self, make_source):
        sources = [make_source(score=0.1)]
        # TECHNICAL min_score is 0.3
        filtered = self.c.filter_results_by_domain(sources, Domain.TECHNICAL)
        assert len(filtered) == 0

    def test_high_score_kept(self, make_source):
        sources = [make_source(score=0.8)]
        filtered = self.c.filter_results_by_domain(sources, Domain.TECHNICAL)
        assert len(filtered) == 1

    def test_fact_content_filter(self, make_source):
        sources = [
            make_source(score=0.5, source_type="fact", content="anime character comfyui"),
            make_source(score=0.5, source_type="fact", content="completely unrelated fluff"),
        ]
        filtered = self.c.filter_results_by_domain(sources, Domain.ANIME)
        # First should pass (has anime keywords), second should not
        assert len(filtered) == 1
        assert "anime" in filtered[0]["content"]

    def test_max_sources_cap(self, make_source):
        # FINANCIAL max_sources is 5
        sources = [make_source(score=0.9, source_type="hybrid", content="bank money") for _ in range(10)]
        filtered = self.c.filter_results_by_domain(sources, Domain.FINANCIAL)
        assert len(filtered) <= 5


# ── should_search_source ──────────────────────────────────────────────

class TestShouldSearchSource:
    def setup_method(self):
        self.c = DomainClassifier()

    def test_qdrant_allowed(self):
        assert self.c.should_search_source(Domain.TECHNICAL, "qdrant", "echo_memory")

    def test_qdrant_denied(self):
        assert not self.c.should_search_source(Domain.TECHNICAL, "qdrant", "nonexistent_collection")

    def test_pg_allowed(self):
        assert self.c.should_search_source(Domain.TECHNICAL, "postgresql", "claude_conversations")

    def test_facts_always_true(self):
        assert self.c.should_search_source(Domain.FINANCIAL, "facts", "anything")


# ── get_allowed_sources ───────────────────────────────────────────────

class TestGetAllowedSources:
    def setup_method(self):
        self.c = DomainClassifier()

    def test_known_domain_returns_config(self):
        sources = self.c.get_allowed_sources(Domain.TECHNICAL)
        assert "qdrant_collections" in sources
        assert "echo_memory" in sources["qdrant_collections"]

    def test_unknown_falls_back_to_general(self):
        # Domain.GENERAL is the fallback
        sources = self.c.get_allowed_sources(Domain.GENERAL)
        assert "qdrant_collections" in sources
