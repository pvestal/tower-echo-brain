"""Unit tests for dedup.merge_metadata pure function.

Covers: earliest timestamp, access_count sum, highest confidence,
        last_accessed update, edge cases with missing fields.
"""
import pytest
from datetime import datetime

from src.core.dedup import merge_metadata

pytestmark = pytest.mark.unit


class TestMergeMetadata:
    def test_earliest_timestamp_kept(self):
        existing = {"ingested_at": "2026-02-10T00:00:00"}
        new = {"ingested_at": "2026-02-15T00:00:00"}
        merged = merge_metadata(existing, new)
        assert merged["ingested_at"] == "2026-02-10T00:00:00"

    def test_sums_access_counts(self):
        existing = {"access_count": 3}
        new = {"access_count": 5}
        merged = merge_metadata(existing, new)
        assert merged["access_count"] == 8

    def test_highest_confidence(self):
        existing = {"confidence": 0.6}
        new = {"confidence": 0.9}
        merged = merge_metadata(existing, new)
        assert merged["confidence"] == 0.9

    def test_default_confidence_07(self):
        # Both missing confidence → default 0.7
        merged = merge_metadata({}, {})
        assert merged["confidence"] == 0.7

    def test_last_accessed_updated_to_now(self):
        merged = merge_metadata({}, {})
        # Should be a valid ISO timestamp close to now
        parsed = datetime.fromisoformat(merged["last_accessed"])
        diff = abs((datetime.now() - parsed).total_seconds())
        assert diff < 10

    def test_existing_only_timestamp(self):
        existing = {"ingested_at": "2026-01-01T00:00:00"}
        new = {}
        merged = merge_metadata(existing, new)
        assert merged["ingested_at"] == "2026-01-01T00:00:00"

    def test_new_only_timestamp(self):
        existing = {}
        new = {"ingested_at": "2026-02-01T00:00:00"}
        merged = merge_metadata(existing, new)
        assert merged["ingested_at"] == "2026-02-01T00:00:00"

    def test_neither_timestamp(self):
        merged = merge_metadata({}, {})
        # ingested_at should not be set (empty string default doesn't match)
        assert merged.get("ingested_at", "") == ""

    def test_preserves_extra_fields(self):
        existing = {"source": "docs", "custom_field": 42}
        new = {"another_field": "hello"}
        merged = merge_metadata(existing, new)
        assert merged["source"] == "docs"
        assert merged["custom_field"] == 42

    def test_zero_access_counts(self):
        merged = merge_metadata({}, {})
        assert merged["access_count"] == 0
