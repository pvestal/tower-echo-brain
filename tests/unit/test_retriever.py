"""Unit tests for ParallelRetriever pure functions.

Covers: _is_readable_text, _classify_query_type, _extract_text_search_terms,
        _fuse_hybrid_results, _apply_time_decay, _detect_conflicts
"""
import math
import pytest
from datetime import datetime, timezone, timedelta

from src.context_assembly.retriever import ParallelRetriever

pytestmark = pytest.mark.unit


# ── _is_readable_text ─────────────────────────────────────────────────

class TestIsReadableText:
    def test_normal_english_text(self):
        assert ParallelRetriever._is_readable_text("This is a normal sentence with spaces.")

    def test_code_snippet_readable(self):
        text = "def foo(bar):\n    return bar + 1  # comment"
        assert ParallelRetriever._is_readable_text(text)

    def test_empty_string(self):
        assert not ParallelRetriever._is_readable_text("")

    def test_none_returns_false(self):
        assert not ParallelRetriever._is_readable_text(None)

    def test_too_short(self):
        assert not ParallelRetriever._is_readable_text("short")

    def test_base64_rejected(self):
        # Base64 has very few spaces and dense alnum
        b64 = "SGVsbG8gV29ybGQgdGhpcyBpcyBhIGJhc2U2NCBlbmNvZGVkIHN0cmluZw=="
        assert not ParallelRetriever._is_readable_text(b64)

    def test_hex_hash_rejected(self):
        hex_hash = "a" * 64 + "b" * 64 + "c" * 64  # 192 chars of dense hex
        assert not ParallelRetriever._is_readable_text(hex_hash)

    def test_dense_alphanumeric_rejected(self):
        dense = "abcdefghij1234567890" * 10  # 200 chars, no spaces
        assert not ParallelRetriever._is_readable_text(dense)

    def test_unicode_text_readable(self):
        text = "This is a test with unicode chars like cafe and nino"
        assert ParallelRetriever._is_readable_text(text)

    def test_boundary_space_ratio_at_002(self):
        # 100 chars, need 2 spaces for exactly 0.02 ratio
        text = "a" * 49 + " " + "b" * 48 + " " + "c"  # 101 chars, 2 spaces = 0.0198...
        # This is just below 0.02 threshold since 2/101 < 0.02
        # Let's make it exactly 100 chars with 2 spaces
        text = "a" * 49 + " " + "b" * 49 + " "  # 100 chars, 2 spaces = 0.02 exactly
        # space_ratio = 2/100 = 0.02, check is < 0.02, so 0.02 passes
        # But dense alnum check will also trigger (98 alnum out of 100 = 0.98 > 0.85)
        # So this should still fail due to alnum density
        assert not ParallelRetriever._is_readable_text(text)

    def test_space_ratio_just_below_boundary(self):
        # 100 chars, 1 space = 0.01, below 0.02 threshold
        text = "a" * 50 + " " + "b" * 49
        assert not ParallelRetriever._is_readable_text(text)

    def test_mixed_content_readable(self):
        text = "Error in file src/main.py at line 42: unexpected token 'def'"
        assert ParallelRetriever._is_readable_text(text)

    def test_json_content_with_spaces(self):
        text = '{"key": "value", "list": [1, 2, 3], "nested": {"a": "b"}}'
        assert ParallelRetriever._is_readable_text(text)

    def test_exactly_10_chars_fails_alnum_density(self):
        # 10 chars, 1 space → space ratio OK (0.1), but alnum density 9/10 = 0.9 > 0.85
        text = "hello test"  # 10 chars
        assert not ParallelRetriever._is_readable_text(text)

    def test_short_sentence_with_punctuation_readable(self):
        # Enough non-alnum to pass the density check
        text = "hello, world! this is fine."
        assert ParallelRetriever._is_readable_text(text)


# ── _classify_query_type ──────────────────────────────────────────────

class TestClassifyQueryType:
    def setup_method(self):
        self.r = ParallelRetriever()

    def test_file_reference_is_keyword(self):
        result = self.r._classify_query_type("retriever.py error_code")
        assert result == ParallelRetriever.QueryType.KEYWORD

    def test_snake_case_is_keyword(self):
        result = self.r._classify_query_type("_is_readable_text")
        assert result == ParallelRetriever.QueryType.KEYWORD

    def test_how_does_architecture_is_conceptual(self):
        result = self.r._classify_query_type("how does the architecture of the context assembly work")
        assert result == ParallelRetriever.QueryType.CONCEPTUAL

    def test_long_with_conceptual_signal(self):
        result = self.r._classify_query_type(
            "explain the overall design and approach of the retriever module"
        )
        assert result == ParallelRetriever.QueryType.CONCEPTUAL

    def test_two_conceptual_signals(self):
        result = self.r._classify_query_type("how and why does this work")
        assert result == ParallelRetriever.QueryType.CONCEPTUAL

    def test_mixed_with_one_conceptual_and_medium_length(self):
        # 1 conceptual signal ("how") + word_count=6 → not enough for CONCEPTUAL,
        # but conceptual_hits > 0 blocks KEYWORD → falls through to MIXED
        result = self.r._classify_query_type("how is the tower echo system")
        assert result == ParallelRetriever.QueryType.MIXED

    def test_empty_is_mixed(self):
        result = self.r._classify_query_type("")
        assert result == ParallelRetriever.QueryType.MIXED

    def test_port_number_short_is_keyword(self):
        result = self.r._classify_query_type("port 8309")
        assert result == ParallelRetriever.QueryType.KEYWORD


# ── _extract_text_search_terms ────────────────────────────────────────

class TestExtractTextSearchTerms:
    def setup_method(self):
        self.r = ParallelRetriever()

    def test_preserves_file_identifier(self):
        result = self.r._extract_text_search_terms("find retriever.py")
        assert "retriever.py" in result

    def test_removes_stop_words(self):
        result = self.r._extract_text_search_terms("what is the purpose about this module")
        for stop in ["what", "is", "the", "about", "this"]:
            assert stop not in result.split()

    def test_limits_to_8_terms(self):
        query = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        result = self.r._extract_text_search_terms(query)
        assert len(result.split()) <= 8

    def test_empty_query(self):
        result = self.r._extract_text_search_terms("")
        assert result == ""

    def test_all_stop_words_returns_empty(self):
        result = self.r._extract_text_search_terms("what is the and or but")
        assert result == ""

    def test_deduplicates(self):
        # If an identifier also appears as a regular word
        result = self.r._extract_text_search_terms("retriever.py retriever.py module")
        words = result.split()
        assert words.count("retriever.py") == 1

    def test_preserves_safetensors_extension(self):
        result = self.r._extract_text_search_terms("model.safetensors checkpoint")
        assert "model.safetensors" in result

    def test_preserves_service_extension(self):
        result = self.r._extract_text_search_terms("tower-echo-brain.service status")
        assert "tower-echo-brain.service" in result


# ── _fuse_hybrid_results ──────────────────────────────────────────────

class TestFuseHybridResults:
    def setup_method(self):
        self.r = ParallelRetriever()

    def test_both_hit_weighted_sum(self, vector_results, text_results):
        fused = self.r._fuse_hybrid_results(vector_results, text_results, "echo_memory")
        # point_id=1 is in both: score = 0.7 * 0.9 + 0.3 * 0.8 = 0.63 + 0.24 = 0.87
        hit = next(r for r in fused if r["point_id"] == 1)
        assert abs(hit["score"] - 0.87) < 0.01

    def test_vector_only_hit(self, vector_results, text_results):
        fused = self.r._fuse_hybrid_results(vector_results, text_results, "echo_memory")
        # point_id=3 is vector-only: score = 0.7 * 0.5 = 0.35
        hit = next(r for r in fused if r["point_id"] == 3)
        assert abs(hit["score"] - 0.35) < 0.01

    def test_text_only_hit(self, vector_results, text_results):
        fused = self.r._fuse_hybrid_results(vector_results, text_results, "echo_memory")
        # point_id=4 is text-only: score = 0.3 * 0.6 = 0.18
        hit = next(r for r in fused if r["point_id"] == 4)
        assert abs(hit["score"] - 0.18) < 0.01

    def test_custom_weights(self, vector_results, text_results):
        fused = self.r._fuse_hybrid_results(
            vector_results, text_results, "echo_memory",
            vector_weight=0.4, text_weight=0.6
        )
        hit = next(r for r in fused if r["point_id"] == 1)
        expected = 0.4 * 0.9 + 0.6 * 0.8  # 0.36 + 0.48 = 0.84
        assert abs(hit["score"] - expected) < 0.01

    def test_empty_inputs(self):
        fused = self.r._fuse_hybrid_results([], [], "echo_memory")
        assert fused == []

    def test_dedup_by_point_id(self, vector_results, text_results):
        fused = self.r._fuse_hybrid_results(vector_results, text_results, "echo_memory")
        point_ids = [r["point_id"] for r in fused]
        assert len(point_ids) == len(set(point_ids))

    def test_metadata_excludes_text_content_embedding(self, vector_results, text_results):
        # Add keys that should be filtered
        vector_results[0]["payload"]["text"] = "should be excluded"
        vector_results[0]["payload"]["content"] = "should be excluded"
        vector_results[0]["payload"]["embedding"] = [0.1, 0.2]
        fused = self.r._fuse_hybrid_results(vector_results, text_results, "echo_memory")
        hit = next(r for r in fused if r["point_id"] == 1)
        assert "text" not in hit["metadata"]
        assert "content" not in hit["metadata"]
        assert "embedding" not in hit["metadata"]

    def test_internal_scoring_fields_cleaned(self, vector_results, text_results):
        fused = self.r._fuse_hybrid_results(vector_results, text_results, "echo_memory")
        for r in fused:
            assert "_vector_score" not in r
            assert "_text_score" not in r

    def test_source_field_set(self, vector_results, text_results):
        fused = self.r._fuse_hybrid_results(vector_results, text_results, "test_collection")
        for r in fused:
            assert r["source"] == "qdrant/test_collection"


# ── _apply_time_decay ─────────────────────────────────────────────────

class TestApplyTimeDecay:
    def setup_method(self):
        self.r = ParallelRetriever()

    def test_recent_content_minimal_decay(self, make_source, recent_timestamp):
        sources = [make_source(score=1.0, metadata={"ingested_at": recent_timestamp})]
        self.r._apply_time_decay(sources)
        # 1 day old: decay_factor ≈ 1 / (1 + log1p(1/30)) ≈ 0.968
        # score = 1.0 * (0.85 + 0.15 * 0.968) ≈ 0.995
        assert sources[0]["score"] > 0.99

    def test_old_content_max_15_percent_penalty(self, make_source, old_timestamp):
        sources = [make_source(score=1.0, metadata={"ingested_at": old_timestamp})]
        self.r._apply_time_decay(sources)
        # 90 days: significant decay but max 15% penalty
        assert sources[0]["score"] >= 0.85
        assert sources[0]["score"] < 1.0

    def test_very_old_logarithmic_flattening(self, make_source):
        ts = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        sources = [make_source(score=1.0, metadata={"ingested_at": ts})]
        self.r._apply_time_decay(sources)
        # Even 365 days old, score should still be >= 0.85 (max 15% penalty)
        assert sources[0]["score"] >= 0.85

    def test_access_count_above_5_exempt(self, make_source, old_timestamp):
        sources = [make_source(
            score=1.0,
            metadata={"ingested_at": old_timestamp, "access_count": 6}
        )]
        self.r._apply_time_decay(sources)
        # access_count > 5 means decay_factor = 1.0
        # score = 1.0 * (0.85 + 0.15 * 1.0) = 1.0
        assert sources[0]["score"] == 1.0

    def test_access_count_at_5_not_exempt(self, make_source, old_timestamp):
        sources = [make_source(
            score=1.0,
            metadata={"ingested_at": old_timestamp, "access_count": 5}
        )]
        self.r._apply_time_decay(sources)
        # access_count == 5 is NOT > 5, so decay applies
        assert sources[0]["score"] < 1.0

    def test_no_timestamp_unchanged(self, make_source):
        sources = [make_source(score=0.8, metadata={})]
        self.r._apply_time_decay(sources)
        assert sources[0]["score"] == 0.8

    def test_invalid_timestamp_unchanged(self, make_source):
        sources = [make_source(score=0.8, metadata={"ingested_at": "not-a-date"})]
        self.r._apply_time_decay(sources)
        assert sources[0]["score"] == 0.8

    def test_z_suffix_parsed(self, make_source):
        ts = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        sources = [make_source(score=1.0, metadata={"ingested_at": ts})]
        self.r._apply_time_decay(sources)
        # Should parse successfully and apply minimal decay
        assert sources[0]["score"] > 0.99

    def test_empty_list(self):
        # Should not raise
        self.r._apply_time_decay([])

    def test_mutates_in_place(self, make_source, old_timestamp):
        sources = [make_source(score=1.0, metadata={"ingested_at": old_timestamp})]
        original = sources[0]
        self.r._apply_time_decay(sources)
        assert sources[0] is original  # Same object mutated


# ── _detect_conflicts ─────────────────────────────────────────────────

class TestDetectConflicts:
    def setup_method(self):
        self.r = ParallelRetriever()

    def test_no_conflicts(self, make_source):
        sources = [
            make_source(source_type="fact", metadata={"subject": "A", "predicate": "is", "object": "X"}),
            make_source(source_type="fact", metadata={"subject": "B", "predicate": "is", "object": "Y"}),
        ]
        assert self.r._detect_conflicts(sources) == []

    def test_same_subject_predicate_different_objects(self, make_source):
        sources = [
            make_source(source_type="fact", metadata={"subject": "Port", "predicate": "is", "object": "8309"}),
            make_source(source_type="fact", metadata={"subject": "Port", "predicate": "is", "object": "8310"}),
        ]
        conflicts = self.r._detect_conflicts(sources)
        assert len(conflicts) == 1
        assert set(conflicts[0]["conflicting_values"]) == {"8309", "8310"}

    def test_non_fact_sources_ignored(self, make_source):
        sources = [
            make_source(source_type="hybrid", metadata={"subject": "A", "predicate": "is", "object": "X"}),
            make_source(source_type="hybrid", metadata={"subject": "A", "predicate": "is", "object": "Y"}),
        ]
        assert self.r._detect_conflicts(sources) == []

    def test_single_fact_returns_empty(self, make_source):
        sources = [
            make_source(source_type="fact", metadata={"subject": "A", "predicate": "is", "object": "X"}),
        ]
        assert self.r._detect_conflicts(sources) == []

    def test_case_insensitive_grouping(self, make_source):
        sources = [
            make_source(source_type="fact", metadata={"subject": "Port", "predicate": "Is", "object": "8309"}),
            make_source(source_type="fact", metadata={"subject": "port", "predicate": "is", "object": "8310"}),
        ]
        conflicts = self.r._detect_conflicts(sources)
        assert len(conflicts) == 1

    def test_same_object_no_conflict(self, make_source):
        sources = [
            make_source(source_type="fact", metadata={"subject": "A", "predicate": "is", "object": "X"}),
            make_source(source_type="fact", metadata={"subject": "A", "predicate": "is", "object": "X"}),
        ]
        assert self.r._detect_conflicts(sources) == []

    def test_multiple_conflict_groups(self, make_source):
        sources = [
            make_source(source_type="fact", metadata={"subject": "A", "predicate": "is", "object": "1"}),
            make_source(source_type="fact", metadata={"subject": "A", "predicate": "is", "object": "2"}),
            make_source(source_type="fact", metadata={"subject": "B", "predicate": "has", "object": "X"}),
            make_source(source_type="fact", metadata={"subject": "B", "predicate": "has", "object": "Y"}),
        ]
        conflicts = self.r._detect_conflicts(sources)
        assert len(conflicts) == 2

    def test_empty_sources(self):
        assert self.r._detect_conflicts([]) == []
