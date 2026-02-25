"""Mocked async tests for LLMService.

Covers: generate, chat, generate_stream
Uses mocked aiohttp to avoid live Ollama dependencies.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from src.services.llm_service import LLMService, LLMResponse

pytestmark = pytest.mark.mocked


class _MockResponse:
    """Minimal mock for an aiohttp response used as async context manager."""

    def __init__(self, status=200, json_data=None, text="", content_lines=None):
        self.status = status
        self._json_data = json_data or {}
        self._text = text
        self._content_lines = content_lines or []

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text

    @property
    def content(self):
        return _AsyncLineIterator(self._content_lines)


class _AsyncLineIterator:
    """Async iterator over byte lines (simulates aiohttp content)."""
    def __init__(self, lines):
        self._lines = list(lines)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._idx]
        self._idx += 1
        return line


class _MockPostContextManager:
    """Async context manager returned by session.post()."""
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        pass


class _MockSession:
    """Minimal mock for aiohttp.ClientSession."""

    def __init__(self, resp):
        self._resp = resp
        self.post_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, url, **kwargs):
        self.post_calls.append({"url": url, **kwargs})
        return _MockPostContextManager(self._resp)

    def stream(self, method, url, **kwargs):
        """For session.stream() used in generate_stream."""
        return _MockPostContextManager(self._resp)


# ── generate ──────────────────────────────────────────────────────────

class TestGenerate:
    async def test_success_parses_response(self):
        svc = LLMService()
        resp = _MockResponse(status=200, json_data={
            "response": "Hello world",
            "model": "mistral:7b",
            "total_duration": 500_000_000,
            "eval_count": 50,
            "eval_duration": 200_000_000,
        })
        session = _MockSession(resp)

        with patch("src.services.llm_service.aiohttp.ClientSession", return_value=session):
            result = await svc.generate("test prompt")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello world"
        assert result.model == "mistral:7b"

    async def test_error_status_raises(self):
        svc = LLMService()
        resp = _MockResponse(status=500, text="Internal Server Error")
        session = _MockSession(resp)

        with patch("src.services.llm_service.aiohttp.ClientSession", return_value=session):
            with pytest.raises(RuntimeError, match="Ollama error 500"):
                await svc.generate("test prompt")

    async def test_system_prompt_included(self):
        svc = LLMService()
        resp = _MockResponse(status=200, json_data={
            "response": "ok",
            "model": "mistral:7b",
            "total_duration": 100_000_000,
            "eval_count": 10,
            "eval_duration": 50_000_000,
        })
        session = _MockSession(resp)

        with patch("src.services.llm_service.aiohttp.ClientSession", return_value=session):
            await svc.generate("test", system="You are helpful")

        payload = session.post_calls[0].get("json", {})
        assert payload["system"] == "You are helpful"

    async def test_duration_calculation(self):
        svc = LLMService()
        resp = _MockResponse(status=200, json_data={
            "response": "result",
            "model": "mistral:7b",
            "total_duration": 1_000_000_000,  # 1 second in ns
            "eval_count": 100,
            "eval_duration": 500_000_000,     # 0.5 seconds in ns
        })
        session = _MockSession(resp)

        with patch("src.services.llm_service.aiohttp.ClientSession", return_value=session):
            result = await svc.generate("test")

        # total_duration_ms = 1_000_000_000 / 1_000_000 = 1000.0
        assert abs(result.total_duration_ms - 1000.0) < 0.1
        # tokens_per_second = 100 / (500_000_000 / 1_000_000_000) = 200.0
        assert abs(result.tokens_per_second - 200.0) < 0.1


# ── chat ──────────────────────────────────────────────────────────────

class TestChat:
    async def test_message_format(self):
        svc = LLMService()
        resp = _MockResponse(status=200, json_data={
            "message": {"role": "assistant", "content": "I can help"},
            "model": "mistral:7b",
            "total_duration": 100_000_000,
            "eval_count": 10,
            "eval_duration": 50_000_000,
        })
        session = _MockSession(resp)

        with patch("src.services.llm_service.aiohttp.ClientSession", return_value=session):
            result = await svc.chat([{"role": "user", "content": "Hello"}])

        assert result.content == "I can help"

    async def test_content_extraction_from_nested(self):
        svc = LLMService()
        resp = _MockResponse(status=200, json_data={
            "message": {"role": "assistant", "content": "Nested response here"},
            "model": "mistral:7b",
            "total_duration": 100_000_000,
            "eval_count": 10,
            "eval_duration": 50_000_000,
        })
        session = _MockSession(resp)

        with patch("src.services.llm_service.aiohttp.ClientSession", return_value=session):
            result = await svc.chat([{"role": "user", "content": "test"}])

        assert result.content == "Nested response here"


# ── generate_stream ───────────────────────────────────────────────────

class TestGenerateStream:
    async def test_yields_chunks(self):
        svc = LLMService()
        content_lines = [
            json.dumps({"response": "Hello"}).encode(),
            json.dumps({"response": " world"}).encode(),
            json.dumps({"done": True}).encode(),
        ]
        resp = _MockResponse(status=200, content_lines=content_lines)
        session = _MockSession(resp)

        with patch("src.services.llm_service.aiohttp.ClientSession", return_value=session):
            collected = []
            async for chunk in svc.generate_stream("test prompt"):
                collected.append(chunk)

        assert "Hello" in collected
        assert " world" in collected
