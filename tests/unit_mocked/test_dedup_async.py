"""Mocked async tests for dedup async functions.

Covers: check_duplicate, bump_existing_point
Uses mocked httpx to avoid live Qdrant dependencies.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.mocked


# ── check_duplicate ───────────────────────────────────────────────────

class TestCheckDuplicate:
    async def test_found_returns_dict(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "result": [
                {"id": 42, "score": 0.98, "payload": {"content": "existing"}}
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("src.core.dedup.httpx.AsyncClient", return_value=mock_client):
            from src.core.dedup import check_duplicate
            result = await check_duplicate([0.1, 0.2, 0.3])

        assert result is not None
        assert result["id"] == 42
        assert result["score"] == 0.98

    async def test_not_found_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"result": []}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("src.core.dedup.httpx.AsyncClient", return_value=mock_client):
            from src.core.dedup import check_duplicate
            result = await check_duplicate([0.1, 0.2, 0.3])

        assert result is None

    async def test_network_error_returns_none(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post.side_effect = Exception("connection refused")

        with patch("src.core.dedup.httpx.AsyncClient", return_value=mock_client):
            from src.core.dedup import check_duplicate
            result = await check_duplicate([0.1, 0.2, 0.3])

        assert result is None


# ── bump_existing_point ───────────────────────────────────────────────

class TestBumpExistingPoint:
    async def test_success_returns_true(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("src.core.dedup.httpx.AsyncClient", return_value=mock_client):
            from src.core.dedup import bump_existing_point
            result = await bump_existing_point(42, {"access_count": 5})

        assert result is True

    async def test_failure_returns_false(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post.side_effect = Exception("server error")

        with patch("src.core.dedup.httpx.AsyncClient", return_value=mock_client):
            from src.core.dedup import bump_existing_point
            result = await bump_existing_point(42, {"access_count": 5})

        assert result is False

    async def test_correct_api_payload(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("src.core.dedup.httpx.AsyncClient", return_value=mock_client):
            from src.core.dedup import bump_existing_point
            await bump_existing_point(99, {"confidence": 0.9})

        # Verify the POST was called with correct payload
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["points"] == [99]
        assert payload["payload"]["confidence"] == 0.9
