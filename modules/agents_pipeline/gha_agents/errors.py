"""
errors.py — Structured error handling for Agents.

Classifies exceptions from the three main failure modes:
  - Missing API key (caught at startup via env.py, re-raised here as GhaConfigError)
  - Invalid model name (provider responds with 404/not found)
  - Rate limit exceeded (provider responds with 429)

Usage in main.py:
    from errors import (
        classify_agent_error, GhaConfigError,
        GhaModelError, GhaRateLimitError,
    )
"""

from __future__ import annotations

# ─── Typed Exceptions ─────────────────────────────────────────────────────────


class GhaError(Exception):
    """Base exception for all the study pipeline errors."""


class GhaConfigError(GhaError):
    """API key is missing or malformed in .env."""


class GhaModelError(GhaError):
    """The configured model ID does not exist or has been deprecated."""


class GhaTransientError(GhaError):
    """A transient provider/infra failure that should be retried in place.

    Covers upstream hiccups OpenRouter surfaces as a generic ``"Provider
    returned error"`` (HTTP 502/503/504, connection resets, pooled-capacity
    stalls) where the discriminating detail is buried in nested ``metadata`` we
    never see in ``str(exc)``. The invoker retries these with bounded backoff
    (see :class:`~gha_agents.invoker.AgentInvoker`); a genuinely permanent
    error simply re-fails after the cap and surfaces as a per-cell failure.
    """


class GhaRateLimitError(GhaTransientError):
    """The provider's rate limit or quota has been exhausted (HTTP 429).

    A specific, retryable kind of :class:`GhaTransientError`.
    """


class GhaStructuredOutputError(GhaError):
    """The model's response could not be coerced into the agent's output_schema.

    Raised by :class:`~gha_agents.invoker.AgentInvoker` when a completed call
    returns ``content`` that is not an instance of the agent's ``output_schema``.
    Agno's parser does not raise on a coercion failure — it leaves the raw model
    string as content (JSON mode) or yields ``None`` (native structured outputs),
    typically because the provider ignored ``response_format`` / ``json_schema``
    and emitted free text. ``raw_response`` carries that unparsed payload for the
    forensic artifact.
    """

    def __init__(self, underlying_cause: str, raw_response: str | None = None) -> None:
        self.underlying_cause = underlying_cause
        self.raw_response = raw_response
        super().__init__(underlying_cause)

    def __str__(self) -> str:
        return f"Model failed to emit structured output: {self.underlying_cause}"


class LLMOutputError(GhaError):
    """The LLM returned empty, None, or structurally invalid output."""

    def __init__(self, cause: Exception | None = None) -> None:
        self.cause = cause

    def __str__(self) -> str:
        if self.cause is None:
            return "LLM returned empty response"
        return f"LLM returned malformed JSON: {self.cause}"


class WorkflowValidationError(GhaError):
    """The input YAML is not a valid GitHub Actions workflow."""

    def __init__(self, file_key: str) -> None:
        self.file_key = file_key

    def __str__(self) -> str:
        return f"Invalid workflow: {self.file_key}"


# ─── Error Classifier ─────────────────────────────────────────────────────────

# Substrings used to fingerprint provider error messages.
# These cover OpenAI, Anthropic, and Google's Gemini SDKs.
_RATE_LIMIT_SIGNALS = (
    "rate limit",
    "rate_limit",
    "429",
    "too many requests",
    "quota exceeded",
    "resource exhausted",
    "overloaded",
)

_MODEL_NOT_FOUND_SIGNALS = (
    "model not found",
    "no such model",
    "invalid model",
    "does not exist",
    "unknown model",
    "model_not_found",
    "404",
    "not_found",
    "deprecated",
)

_AUTH_SIGNALS = (
    "api key",
    "api_key",
    "authentication",
    "unauthorized",
    "401",
    "invalid key",
    "incorrect api key",
    "permission denied",
    "unauthenticated",
)

# Transient provider/infra failures distinct from an explicit rate limit: the
# upstream errored or was briefly unavailable. OpenRouter wraps these as a
# generic "provider returned error" whose real cause sits in nested metadata
# that never reaches ``str(exc)``, so we fingerprint that envelope plus the
# common 5xx / connectivity phrases. Retried in place with bounded backoff; a
# permanent error still surfaces once the cap is hit. ``overloaded`` already
# lives in the rate-limit set (checked first) so it is intentionally omitted.
#
# Deliberately EXCLUDES a bare client read-timeout ("timeout" / "timed out"):
# those only fire after the full ``request_timeout`` (180s) elapses, so retrying
# one would cost up to ``max_retries`` × 180s on a single pair — a long (though
# bounded) stall. All signals below are server-side responses that return fast,
# keeping worst-case retry latency to the backoff sum (~30s). A genuine read
# timeout fails the pair and is recovered on resume instead. ("gateway timeout"
# / 504 is a fast upstream *response*, not a client read-timeout, so it stays.)
_TRANSIENT_SIGNALS = (
    "provider returned error",
    "temporarily",
    "please retry",
    "try again",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "connection error",
    "connection reset",
    "connection aborted",
    "502",
    "503",
    "504",
)


def classify_agent_error(exc: Exception) -> GhaError:
    """
    Inspect an exception raised during an agent run and return a typed the study error.

    This covers the main failure modes across OpenAI, Anthropic, and Gemini:
      - Missing / invalid API key   → GhaConfigError
      - Invalid / deprecated model  → GhaModelError
      - Rate limit / quota (429)    → GhaRateLimitError  (retryable)
      - Transient provider / 5xx    → GhaTransientError  (retryable)
      - Any other error             → GhaError (generic)

    Args:
        exc: The raw exception caught during agent execution.

    Returns:
        A GhaError (or subclass) with a descriptive human-readable message.
    """
    msg = str(exc).lower()
    exc_type = type(exc).__name__

    if any(sig in msg for sig in _RATE_LIMIT_SIGNALS):
        return GhaRateLimitError(exc_type)

    if any(sig in msg for sig in _MODEL_NOT_FOUND_SIGNALS):
        return GhaModelError(exc_type)

    if any(sig in msg for sig in _AUTH_SIGNALS):
        return GhaConfigError(exc_type)

    # Checked after the specific classes above so a 404 / 401 wrapped in a
    # generic "provider returned error" envelope still types correctly.
    if any(sig in msg for sig in _TRANSIENT_SIGNALS):
        return GhaTransientError(exc_type)

    if isinstance(exc, (GhaConfigError, GhaModelError, GhaTransientError)):
        return exc

    # Preserve the original message, not just the type name: callers wrap a
    # framework error-status string as Exception(cause) (invoker._guard_status),
    # and keeping only exc_type collapses every distinct failure to "Exception",
    # destroying diagnosability. Fall back to the type name when there is no message.
    detail = str(exc).strip()
    return GhaError(f"{exc_type}: {detail}" if detail else exc_type)
