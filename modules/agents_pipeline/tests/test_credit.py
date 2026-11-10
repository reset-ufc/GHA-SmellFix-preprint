"""Tests for gha_agents.credit — OpenRouter balance + out-of-credit detection."""

import json

import pytest
from gha_agents.credit import (
    OutOfCreditError,
    check_openrouter_balance,
    is_out_of_credit_error,
    message_signals_out_of_credit,
    preflight_credit_check,
)


class _FakeResponse:
    def __init__(self, status_code: int, body: str = ""):
        self.status_code = status_code
        self.text = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")  # noqa: TRY003

    def json(self):
        return json.loads(self.text) if self.text else {}


class _FakeHTTPError(Exception):
    """Stand-in for httpx.HTTPStatusError without depending on httpx in tests."""

    def __init__(self, response):
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def test_is_out_of_credit_error_detects_http_402():
    """402 Payment Required is the standard OpenRouter out-of-credit signal."""
    exc = _FakeHTTPError(_FakeResponse(402, "Insufficient credits"))
    assert is_out_of_credit_error(exc) is True


def test_is_out_of_credit_error_detects_429_with_credit_message():
    """OpenRouter sometimes returns 429 when credit is exhausted; body mentions credit."""
    exc = _FakeHTTPError(_FakeResponse(429, "You have insufficient credit. Add more at..."))
    assert is_out_of_credit_error(exc) is True


def test_is_out_of_credit_error_ignores_unrelated_429():
    """A regular rate-limit 429 (no credit mention) is not an out-of-credit signal."""
    exc = _FakeHTTPError(_FakeResponse(429, "Rate limit exceeded, retry in 60s"))
    assert is_out_of_credit_error(exc) is False


def test_is_out_of_credit_error_ignores_non_http_exceptions():
    """Plain exceptions without a response attribute are not out-of-credit."""
    assert is_out_of_credit_error(ValueError("boom")) is False


def test_message_signals_out_of_credit_detects_keywords():
    """The message-only path (provider error surfaced as a status content string)."""
    assert message_signals_out_of_credit("Error 402: insufficient credit") is True
    assert message_signals_out_of_credit("You are out of credit, top up") is True
    assert message_signals_out_of_credit("insufficient_credit") is True
    # Anthropic native-SDK exhausted-balance message (HTTP 400, message-only).
    assert (
        message_signals_out_of_credit(
            "Error code: 400 - Your credit balance is too low to access the "
            "Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."
        )
        is True
    )


def test_message_signals_out_of_credit_ignores_unrelated():
    """A plain rate-limit message (no credit mention) is not out-of-credit."""
    assert message_signals_out_of_credit("rate limit exceeded, retry in 60s") is False
    assert message_signals_out_of_credit("") is False


def test_openai_insufficient_quota_is_out_of_credit():
    """OpenAI empty-wallet surfaces as HTTP 429 type ``insufficient_quota`` — NOT a
    rate limit. Both the response-shaped and message-shaped seams must divert it to
    out-of-credit, else it matched the bare "429" rate-limit signal, retried every
    pair, and the run exited 0 masking total failure (review finding N1)."""
    body = (
        '{"error": {"message": "You exceeded your current quota, please check '
        'your plan and billing details.", "type": "insufficient_quota", '
        '"code": "insufficient_quota"}}'
    )
    exc = _FakeHTTPError(_FakeResponse(429, body))
    assert is_out_of_credit_error(exc) is True
    assert message_signals_out_of_credit(body) is True


def test_anthropic_spend_limit_400_is_out_of_credit():
    """Anthropic's monthly spend-limit cap surfaces as HTTP 400 invalid_request_error
    "You have reached your specified API usage limits…" — distinct from the empty-
    wallet "credit balance is too low" 400. Without recognising it the run fell
    through to a generic GhaError, never fast-failed, and burned every remaining
    cell against instant 400s (1895 in an n15 haiku stability run)."""
    msg = (
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'You have reached your specified API usage limits. You will regain "
        "access on 2026-07-01 at 00:00 UTC.'}}"
    )
    assert message_signals_out_of_credit(msg) is True


def test_payment_required_phrase_signals_out_of_credit():
    """A 402 flattened to a message string (status code gone) is still caught via
    the anchored 'payment required' phrase."""
    assert message_signals_out_of_credit("HTTP 402 Payment Required") is True


def test_bare_402_substring_no_longer_false_positives():
    """Regression: a bare '402' substring inside unrelated 429 text (epoch / token
    counts) must NOT read as out-of-credit now that the bare-number keyword is gone
    — the false positive contained a whole model mid-run (review finding)."""
    exc = _FakeHTTPError(_FakeResponse(429, "Rate limited; retry after 1402 seconds"))
    assert is_out_of_credit_error(exc) is False
    assert message_signals_out_of_credit("processed 402 tokens in 1402ms") is False


def test_check_openrouter_balance_calls_auth_key(monkeypatch):
    """GET /api/v1/auth/key with Bearer token; returns parsed usage / limit."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeResponse(
            200, json.dumps({"data": {"usage": 1.5, "limit": 5.0, "is_free_tier": False}})
        )

    balance = check_openrouter_balance(http_get=fake_get)

    assert captured["url"] == "https://openrouter.ai/api/v1/auth/key"
    assert captured["headers"] == {"Authorization": "Bearer sk-or-test"}
    assert balance == {"usage": 1.5, "limit": 5.0, "is_free_tier": False}


def test_preflight_aborts_when_balance_below_margin():
    """Raise OutOfCreditError if available balance < min_balance."""
    fake_balance = lambda: {"usage": 4.95, "limit": 5.00, "is_free_tier": False}  # noqa: E731
    with pytest.raises(OutOfCreditError):
        preflight_credit_check(min_balance=0.10, balance_fn=fake_balance)


def test_preflight_allows_free_tier_with_no_limit():
    """OpenRouter free-tier returns limit=None; preflight is a no-op."""
    fake_balance = lambda: {"usage": 0.0, "limit": None, "is_free_tier": True}  # noqa: E731
    preflight_credit_check(min_balance=0.10, balance_fn=fake_balance)


def test_preflight_passes_when_balance_above_margin():
    """Available balance well above the margin: no exception."""
    fake_balance = lambda: {"usage": 1.00, "limit": 5.00, "is_free_tier": False}  # noqa: E731
    preflight_credit_check(min_balance=0.10, balance_fn=fake_balance)


def test_preflight_skips_silently_when_api_key_missing():
    """Ollama-only experiments lack OPENROUTER_API_KEY; preflight is a no-op."""

    def raises_keyerror():
        raise KeyError("OPENROUTER_API_KEY")

    preflight_credit_check(balance_fn=raises_keyerror)
