"""Tests for gha_agents.errors classification + log_error enrichment."""

from __future__ import annotations

import json
from pathlib import Path

from gha_agents.errors import (
    GhaConfigError,
    GhaError,
    GhaModelError,
    GhaRateLimitError,
    GhaTransientError,
    classify_agent_error,
)
from gha_agents.utils import log_error, persist_raw_response


def test_classify_agent_error_maps_rate_limit_signal():
    classified = classify_agent_error(Exception("Error 429: too many requests"))
    assert isinstance(classified, GhaRateLimitError)


def test_classify_agent_error_maps_model_not_found_signal():
    classified = classify_agent_error(Exception("The configured model does not exist"))
    assert isinstance(classified, GhaModelError)


def test_classify_agent_error_maps_auth_signal():
    classified = classify_agent_error(Exception("401 Unauthorized: invalid api key"))
    assert isinstance(classified, GhaConfigError)


def test_classify_agent_error_falls_back_to_generic_study_error():
    classified = classify_agent_error(Exception("something unexpected happened"))
    assert isinstance(classified, GhaError)
    assert not isinstance(
        classified, (GhaRateLimitError, GhaModelError, GhaConfigError, GhaTransientError)
    )


# ─── transient (retryable) classification — the 429-envelope + 5xx fix ────────


def test_classify_maps_openrouter_provider_envelope_to_transient():
    """OpenRouter wraps upstream failures as a generic 'provider returned error'
    whose real 429/5xx cause never reaches str(exc). It MUST classify as a
    retryable transient error — previously it fell through to a non-retryable
    generic GhaError, so the live 429s were silently dropped as [UNKNOWN]."""
    classified = classify_agent_error(Exception("Provider returned error"))
    assert isinstance(classified, GhaTransientError)
    assert not isinstance(classified, GhaRateLimitError)


def test_classify_maps_5xx_and_connectivity_to_transient():
    for msg in (
        "503 Service Unavailable",
        "502 Bad Gateway",
        "504 Gateway Timeout",  # fast upstream response, not a client read-timeout
        "Connection reset by peer",
        "connection aborted by remote host",
    ):
        classified = classify_agent_error(Exception(msg))
        assert isinstance(classified, GhaTransientError), msg


def test_bare_read_timeout_is_not_auto_retried():
    """A bare client read-timeout is NOT transient: it only fires after the full
    request_timeout (180s), so retrying it would stall one pair for up to
    max_retries × 180s. It fails the pair and is recovered on resume instead.
    (A 504 'gateway timeout' is a fast upstream response and DOES stay transient.)"""
    for msg in ("Request timed out", "read operation timeout", "httpx.ReadTimeout"):
        assert not isinstance(classify_agent_error(Exception(msg)), GhaTransientError), msg


def test_rate_limit_is_a_retryable_transient_subclass():
    """429 stays a GhaRateLimitError for accurate logging, but IS-A transient
    so the single retry clause (``except GhaTransientError``) covers it."""
    assert issubclass(GhaRateLimitError, GhaTransientError)
    classified = classify_agent_error(Exception("Error 429: too many requests"))
    assert isinstance(classified, GhaRateLimitError)
    assert isinstance(classified, GhaTransientError)


def test_permanent_errors_are_never_transient():
    """Auth / model-not-found / generic must NOT be retryable — guards against
    burning the retry budget (and wall-clock) on a permanent failure."""
    for msg in (
        "401 Unauthorized: invalid api key",
        "the configured model does not exist",
        "something unexpected happened",
    ):
        assert not isinstance(classify_agent_error(Exception(msg)), GhaTransientError), msg


def test_model_not_found_wins_over_transient_envelope():
    """A 404 wrapped in a generic 'provider returned error' envelope classifies
    as the permanent model error, not transient — so a genuinely bad model id is
    never retried forever (order: rate → model → auth → transient)."""
    classified = classify_agent_error(
        Exception("Provider returned error: 404 model not found")
    )
    assert isinstance(classified, GhaModelError)
    assert not isinstance(classified, GhaTransientError)


def test_out_of_credit_is_not_classified_transient():
    """Sanity: a bare credit message is not a transient signal here. (The invoker
    diverts out-of-credit to OutOfCreditError BEFORE classify runs; this only
    confirms classify itself never makes a wallet-empty error look retryable.)"""
    classified = classify_agent_error(Exception("insufficient credit"))
    assert not isinstance(classified, GhaTransientError)


def test_log_error_writes_enriched_jsonl_when_typed_exception_supplied(tmp_path: Path):
    log_path = tmp_path / "error.log"
    typed = GhaRateLimitError("RateLimitError")

    log_error(
        log_path,
        "wf.yml",
        f"[Some smell] {typed}",
        category="rate_limit",
        exception_type="RateLimitError",
        target_smell="Some smell",
        model_ids={"detection": "openai/gpt-5.4-mini"},
    )

    line = log_path.read_text(encoding="utf-8").strip()
    entry = json.loads(line)

    assert entry["file"] == "wf.yml"
    assert entry["category"] == "rate_limit"
    assert entry["exception_type"] == "RateLimitError"
    assert entry["target_smell"] == "Some smell"
    assert entry["model_ids"] == {"detection": "openai/gpt-5.4-mini"}
    assert "timestamp" in entry
    assert "error" in entry  # backward compat


def test_log_error_omits_optional_fields_when_not_provided(tmp_path: Path):
    log_path = tmp_path / "error.log"
    log_error(log_path, "wf.yml", "plain string detail")
    entry = json.loads(log_path.read_text(encoding="utf-8").strip())

    for absent_key in (
        "category",
        "exception_type",
        "target_smell",
        "model_ids",
        "raw_response_path",
    ):
        assert absent_key not in entry, f"{absent_key} should be omitted when not supplied"


def test_persist_raw_response_writes_text_to_forensic_file_under_artifacts_dir(tmp_path: Path):
    artifacts_dir = tmp_path / "error_artifacts"

    raw = '{"smell_detected": "yes"} — free text the provider emitted instead of valid JSON'
    written = persist_raw_response(
        artifacts_dir,
        slug="Repo@wf@D1@rep1@somesmell@gpt",
        raw_response=raw,
    )

    assert written == artifacts_dir / "Repo@wf@D1@rep1@somesmell@gpt.raw"
    assert written.read_text(encoding="utf-8") == raw
