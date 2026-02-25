"""Unit tests for IntelligenceEngine pure functions.

Covers: _identify_domain, _identify_knowledge_gaps, _find_common_terms,
        _calculate_confidence, _build_reasoning_chain
"""
import pytest

from src.core.intelligence_engine import (
    IntelligenceEngine,
    KnowledgeDomain,
    MemoryContext,
)

pytestmark = pytest.mark.unit


# ── _identify_domain ──────────────────────────────────────────────────

class TestIdentifyDomain:
    def setup_method(self):
        self.ie = IntelligenceEngine()

    def test_tower_system(self):
        assert self.ie._identify_domain("tower api dashboard service") == KnowledgeDomain.TOWER_SYSTEM

    def test_anime_production(self):
        assert self.ie._identify_domain("framepack comfyui animation") == KnowledgeDomain.ANIME_PRODUCTION

    def test_echo_brain(self):
        assert self.ie._identify_domain("memory vector qdrant embedding") == KnowledgeDomain.ECHO_BRAIN

    def test_programming(self):
        assert self.ie._identify_domain("python class bug fix") == KnowledgeDomain.PROGRAMMING

    def test_conversations(self):
        assert self.ie._identify_domain("discussed mentioned conversation") == KnowledgeDomain.CONVERSATIONS

    def test_facts(self):
        assert self.ie._identify_domain("what is data information") == KnowledgeDomain.FACTS

    def test_unknown_no_keywords(self):
        assert self.ie._identify_domain("xyzzy plugh nothing") == KnowledgeDomain.UNKNOWN

    def test_tiebreaker_highest_score_wins(self):
        # "tower system server service api dashboard" → 6 hits for TOWER_SYSTEM
        # vs fewer hits for other domains
        result = self.ie._identify_domain("tower system server service api dashboard")
        assert result == KnowledgeDomain.TOWER_SYSTEM


# ── _identify_knowledge_gaps ──────────────────────────────────────────

class TestIdentifyKnowledgeGaps:
    def setup_method(self):
        self.ie = IntelligenceEngine()

    def test_no_high_confidence_memories(self, make_memory_context):
        memories = [make_memory_context(score=0.3)]
        gaps = self.ie._identify_knowledge_gaps("test", memories)
        assert any("No high-confidence" in g for g in gaps)

    def test_how_no_implementation(self, make_memory_context):
        memories = [make_memory_context(content="general info about tower", score=0.8)]
        gaps = self.ie._identify_knowledge_gaps("how does it work", memories)
        assert any("implementation" in g.lower() for g in gaps)

    def test_why_no_reasoning(self, make_memory_context):
        memories = [make_memory_context(content="tower is running", score=0.8)]
        gaps = self.ie._identify_knowledge_gaps("why does it fail", memories)
        assert any("reasoning" in g.lower() or "explanation" in g.lower() for g in gaps)

    def test_when_no_timestamps(self, make_memory_context):
        memories = [make_memory_context(content="something happened", score=0.8, metadata={})]
        gaps = self.ie._identify_knowledge_gaps("when was it deployed", memories)
        assert any("temporal" in g.lower() for g in gaps)

    def test_error_no_fix(self, make_memory_context):
        memories = [make_memory_context(content="there was an error", score=0.8)]
        gaps = self.ie._identify_knowledge_gaps("error in the module", memories)
        assert any("solution" in g.lower() or "fix" in g.lower() for g in gaps)

    def test_all_satisfied_empty(self, make_memory_context):
        memories = [make_memory_context(
            content="implement this fix because of the reason, the solution works",
            score=0.9,
            metadata={"timestamp": "2026-02-01T00:00:00"}
        )]
        gaps = self.ie._identify_knowledge_gaps("plain query", memories)
        assert len(gaps) == 0

    def test_empty_memories(self):
        gaps = self.ie._identify_knowledge_gaps("how does it work", [])
        assert any("No high-confidence" in g for g in gaps)


# ── _find_common_terms ────────────────────────────────────────────────

class TestFindCommonTerms:
    def setup_method(self):
        self.ie = IntelligenceEngine()

    def test_common_words_found(self, make_memory_context):
        memories = [
            make_memory_context(content="tower server service architecture"),
            make_memory_context(content="tower server port configuration"),
            make_memory_context(content="tower dashboard deployment"),
        ]
        terms = self.ie._find_common_terms(memories)
        assert "tower" in terms

    def test_stop_words_excluded(self, make_memory_context):
        memories = [
            make_memory_context(content="the and is are with for"),
            make_memory_context(content="the and is are with for"),
        ]
        terms = self.ie._find_common_terms(memories)
        # All words are either stop words or too short
        assert len(terms) == 0

    def test_short_words_excluded(self, make_memory_context):
        memories = [
            make_memory_context(content="foo bar baz qux"),
            make_memory_context(content="foo bar baz qux"),
        ]
        terms = self.ie._find_common_terms(memories)
        # All words are <= 3 chars
        assert len(terms) == 0

    def test_empty_memories(self):
        terms = self.ie._find_common_terms([])
        assert terms == []

    def test_no_common_terms(self, make_memory_context):
        memories = [
            make_memory_context(content="alpha beta gamma delta"),
            make_memory_context(content="epsilon zeta theta iota"),
        ]
        terms = self.ie._find_common_terms(memories)
        assert len(terms) == 0

    def test_max_5_terms(self, make_memory_context):
        # Create memories where many words repeat
        content = "tower server service architecture dashboard deployment configuration"
        memories = [make_memory_context(content=content) for _ in range(5)]
        terms = self.ie._find_common_terms(memories)
        assert len(terms) <= 5


# ── _calculate_confidence ─────────────────────────────────────────────

class TestCalculateConfidence:
    def setup_method(self):
        self.ie = IntelligenceEngine()

    def test_empty_memories_returns_0(self):
        assert self.ie._calculate_confidence([], []) == 0.0

    def test_all_high_scores_near_1(self, make_memory_context):
        memories = [make_memory_context(score=0.95) for _ in range(5)]
        reasoning = ["step"] * 5
        confidence = self.ie._calculate_confidence(memories, reasoning)
        assert confidence > 0.8

    def test_low_scores_low_confidence(self, make_memory_context):
        memories = [make_memory_context(score=0.2) for _ in range(5)]
        confidence = self.ie._calculate_confidence(memories, [])
        assert confidence < 0.3

    def test_capped_at_1(self, make_memory_context):
        memories = [make_memory_context(score=1.0) for _ in range(10)]
        reasoning = ["step"] * 10
        confidence = self.ie._calculate_confidence(memories, reasoning)
        assert confidence <= 1.0

    def test_single_memory(self, make_memory_context):
        memories = [make_memory_context(score=0.6)]
        confidence = self.ie._calculate_confidence(memories, [])
        # avg_score=0.6*0.4 + high_conf_ratio=0*0.3 + consensus=0.5*0.2 + reasoning=0*0.1
        # = 0.24 + 0 + 0.1 + 0 = 0.34
        assert 0.3 < confidence < 0.4

    def test_weight_components(self, make_memory_context):
        # All score=0.8 (>0.7 threshold), 5 memories, 5 reasoning steps
        memories = [make_memory_context(score=0.8) for _ in range(5)]
        reasoning = ["step"] * 5
        confidence = self.ie._calculate_confidence(memories, reasoning)
        # avg_score=0.8*0.4=0.32, high_conf=1.0*0.3=0.3, consensus varies, reasoning=1.0*0.1=0.1
        assert confidence > 0.7


# ── _build_reasoning_chain ────────────────────────────────────────────

class TestBuildReasoningChain:
    def setup_method(self):
        self.ie = IntelligenceEngine()

    def test_no_memories(self):
        steps = self.ie._build_reasoning_chain("test", [], [])
        assert any("No directly relevant" in s for s in steps)

    def test_with_memories_shows_confidence(self, make_memory_context):
        memories = [make_memory_context(score=0.85, source="echo_memory")]
        steps = self.ie._build_reasoning_chain("test", memories, [])
        assert any("0.85" in s for s in steps)

    def test_with_gaps(self, make_memory_context):
        memories = [make_memory_context(score=0.5)]
        gaps = ["Missing implementation details"]
        steps = self.ie._build_reasoning_chain("test", memories, gaps)
        assert any("gap" in s.lower() for s in steps)

    def test_high_avg_score_assessment(self, make_memory_context):
        memories = [make_memory_context(score=0.9) for _ in range(5)]
        steps = self.ie._build_reasoning_chain("test", memories, [])
        assert any("High confidence" in s for s in steps)

    def test_low_avg_score_assessment(self, make_memory_context):
        memories = [make_memory_context(score=0.3) for _ in range(5)]
        steps = self.ie._build_reasoning_chain("test", memories, [])
        assert any("Low confidence" in s for s in steps)
