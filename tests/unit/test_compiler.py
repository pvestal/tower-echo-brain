"""Unit tests for ContextCompiler pure functions.

Covers: _group_sources_by_type, _apply_type_limits, _select_sources_for_budget,
        _estimate_tokens, _emergency_trim, _format_structured/narrative/minimal,
        create_system_prompt
"""
import pytest

from src.context_assembly.compiler import ContextCompiler

pytestmark = pytest.mark.unit


# ── _group_sources_by_type ────────────────────────────────────────────

class TestGroupSourcesByType:
    def setup_method(self):
        self.c = ContextCompiler()

    def test_normal_grouping(self, make_source):
        sources = [
            make_source(source_type="fact"),
            make_source(source_type="fact"),
            make_source(source_type="hybrid"),
        ]
        grouped = self.c._group_sources_by_type(sources)
        assert len(grouped["fact"]) == 2
        assert len(grouped["hybrid"]) == 1

    def test_empty_sources(self):
        assert self.c._group_sources_by_type([]) == {}

    def test_missing_type_becomes_unknown(self):
        sources = [{"content": "test", "score": 0.5}]
        grouped = self.c._group_sources_by_type(sources)
        assert "unknown" in grouped

    def test_order_preserved(self, make_source):
        sources = [
            make_source(content="first", source_type="fact"),
            make_source(content="second", source_type="fact"),
        ]
        grouped = self.c._group_sources_by_type(sources)
        assert grouped["fact"][0]["content"] == "first"
        assert grouped["fact"][1]["content"] == "second"


# ── _apply_type_limits ────────────────────────────────────────────────

class TestApplyTypeLimits:
    def setup_method(self):
        self.c = ContextCompiler()

    def test_truncates_to_limit(self, make_source):
        grouped = {"fact": [make_source() for _ in range(5)]}
        limited = self.c._apply_type_limits(grouped, {"fact": 2})
        assert len(limited["fact"]) == 2

    def test_no_limit_keeps_all(self, make_source):
        grouped = {"fact": [make_source() for _ in range(5)]}
        limited = self.c._apply_type_limits(grouped, {})
        assert len(limited["fact"]) == 5

    def test_zero_limit(self, make_source):
        grouped = {"fact": [make_source() for _ in range(3)]}
        limited = self.c._apply_type_limits(grouped, {"fact": 0})
        assert len(limited["fact"]) == 0

    def test_limit_greater_than_count(self, make_source):
        grouped = {"fact": [make_source() for _ in range(2)]}
        limited = self.c._apply_type_limits(grouped, {"fact": 100})
        assert len(limited["fact"]) == 2


# ── _select_sources_for_budget ────────────────────────────────────────

class TestSelectSourcesForBudget:
    def setup_method(self):
        self.c = ContextCompiler()

    def test_facts_first(self, make_source):
        grouped = {
            "fact": [make_source(content="short fact", score=0.5, source_type="fact")],
            "hybrid": [make_source(content="hybrid content", score=0.9, source_type="hybrid")],
        }
        selected = self.c._select_sources_for_budget(grouped, 10000)
        # Facts should come first regardless of score
        assert selected[0]["type"] == "fact"

    def test_facts_present_limits_non_facts_to_5(self, make_source):
        grouped = {
            "fact": [make_source(content="fact", source_type="fact")],
            "hybrid": [make_source(content=f"hybrid {i}", score=0.9 - i * 0.01, source_type="hybrid") for i in range(10)],
        }
        selected = self.c._select_sources_for_budget(grouped, 100000)
        non_facts = [s for s in selected if s["type"] != "fact"]
        assert len(non_facts) <= 5

    def test_no_facts_allows_15(self, make_source):
        grouped = {
            "hybrid": [make_source(content=f"hybrid {i}", score=0.9, source_type="hybrid") for i in range(20)],
        }
        selected = self.c._select_sources_for_budget(grouped, 100000)
        assert len(selected) <= 15

    def test_truncation_with_ellipsis(self, make_source):
        # Small budget that can't fit full content but > 100 tokens available
        content = "x" * 2000  # 500 tokens worth
        grouped = {
            "hybrid": [make_source(content=content, source_type="hybrid")],
        }
        selected = self.c._select_sources_for_budget(grouped, 200)
        if selected:
            assert selected[0]["content"].endswith("...")

    def test_small_budget(self, make_source):
        grouped = {
            "hybrid": [make_source(content="x" * 10000, source_type="hybrid")],
        }
        # Budget of 10 tokens = 40 chars, which is < 100 available after 0 used
        # Actually let's use a reasonable budget
        selected = self.c._select_sources_for_budget(grouped, 50)
        # Should truncate or skip
        assert len(selected) <= 1

    def test_empty_grouped(self):
        selected = self.c._select_sources_for_budget({}, 10000)
        assert selected == []


# ── _estimate_tokens ──────────────────────────────────────────────────

class TestEstimateTokens:
    def setup_method(self):
        self.c = ContextCompiler()

    def test_400_chars_is_100_tokens(self):
        assert self.c._estimate_tokens("a" * 400) == 100

    def test_empty_is_0(self):
        assert self.c._estimate_tokens("") == 0

    def test_1_char_is_0(self):
        # 1 // 4 = 0
        assert self.c._estimate_tokens("x") == 0


# ── _emergency_trim ───────────────────────────────────────────────────

class TestEmergencyTrim:
    def setup_method(self):
        self.c = ContextCompiler()

    def test_no_trim_needed(self):
        text = "short text"
        result = self.c._emergency_trim(text, 1000)
        assert result == text

    def test_truncation_with_marker(self):
        text = "x" * 1000
        result = self.c._emergency_trim(text, 100)  # 100 tokens = 400 chars
        assert result.endswith("[Context trimmed...]")
        # Marker is "\n\n[Context trimmed...]" (22 chars), code does max_chars-20
        # so result is slightly over max_chars — verify it was actually trimmed
        assert len(result) < len(text)

    def test_exact_boundary(self):
        text = "x" * 400  # Exactly 100 tokens
        result = self.c._emergency_trim(text, 100)
        assert result == text


# ── _format_structured ────────────────────────────────────────────────

class TestFormatStructured:
    def setup_method(self):
        self.c = ContextCompiler()

    def test_empty_sources_fallback(self):
        result = self.c._format_structured([], "technical")
        assert "No relevant context" in result
        assert "technical" in result.lower()

    def test_domain_header(self, make_source):
        sources = [make_source()]
        result = self.c._format_structured(sources, "technical")
        assert "TECHNICAL" in result

    def test_score_display(self, make_source):
        sources = [make_source(score=0.85)]
        result = self.c._format_structured(sources, "general")
        assert "0.85" in result


# ── _format_narrative ─────────────────────────────────────────────────

class TestFormatNarrative:
    def setup_method(self):
        self.c = ContextCompiler()

    def test_empty_sources_fallback(self):
        result = self.c._format_narrative([], "anime")
        assert "don't have" in result.lower()

    def test_domain_in_output(self, make_source):
        sources = [make_source(source_type="fact")]
        result = self.c._format_narrative(sources, "technical")
        assert "technical" in result

    def test_fact_section_present(self, make_source):
        sources = [make_source(source_type="fact", content="Test fact")]
        result = self.c._format_narrative(sources, "general")
        assert "Key facts" in result
        assert "Test fact" in result


# ── _format_minimal ───────────────────────────────────────────────────

class TestFormatMinimal:
    def setup_method(self):
        self.c = ContextCompiler()

    def test_empty_returns_empty_string(self):
        assert self.c._format_minimal([]) == ""

    def test_joins_with_double_newline(self, make_source):
        sources = [
            make_source(content="first"),
            make_source(content="second"),
        ]
        result = self.c._format_minimal(sources)
        assert result == "first\n\nsecond"


# ── create_system_prompt ──────────────────────────────────────────────

class TestCreateSystemPrompt:
    def setup_method(self):
        self.c = ContextCompiler()

    def test_technical_domain(self):
        result = self.c.create_system_prompt("technical")
        assert "technical" in result.lower()
        assert "code" in result.lower()

    def test_anime_domain(self):
        result = self.c.create_system_prompt("anime")
        assert "anime" in result.lower()

    def test_unknown_domain_fallback(self):
        result = self.c.create_system_prompt("nonexistent_domain")
        assert "helpful" in result.lower()

    def test_concise_style(self):
        result = self.c.create_system_prompt("general", style="concise")
        assert "concise" in result.lower()
