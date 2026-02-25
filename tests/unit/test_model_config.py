"""Unit tests for model_config.get_model().

Covers: general, embedding, coding roles and unknown role fallback.
"""
import pytest

from src.model_config import get_model

pytestmark = pytest.mark.unit


class TestGetModel:
    def test_general_returns_default(self):
        model = get_model("general")
        # Default is mistral:7b (unless env overridden)
        assert model  # Non-empty
        assert isinstance(model, str)

    def test_embedding_returns_nomic(self):
        model = get_model("embedding")
        assert model == "nomic-embed-text"

    def test_coding_returns_deepseek(self):
        model = get_model("coding")
        assert "deepseek" in model.lower()

    def test_unknown_role_returns_default(self):
        model = get_model("nonexistent_role_xyz")
        # Should fall back to _DEFAULT_MODEL
        assert model
        assert isinstance(model, str)
