"""OpenRouter credit balance checks and out-of-credit detection.

Used by :mod:`gha_agents.experiment` to fast-fail when the OpenRouter
account runs out of funds rather than continuing to log spurious failures.
On out-of-credit the runner exits cleanly; the existing on-disk resume
mechanism (``execution.resume: true``) skips already-completed entries when
the user re-launches after topping up.

**Provider scope.** The pre-flight balance check here is OpenRouter-only —
it hits ``/api/v1/auth/key`` for the OpenRouter usage/limit. Native-SDK
providers (``anthropic`` / ``openai`` / ``gemini`` / ``vertex`` / ``ollama``)
have no equivalent gate: there is no portable balance endpoint exposed by their
SDKs, so an experiment that routes through them surfaces insufficient funds
only as a mid-run API error. (``vertex`` bills a Google Cloud project, so its
"balance" is the GCP billing account / credit grant — set a GCP budget alert
instead, there is no pre-flight balance call.) The invoker normalizes that signal —
response-shaped via :func:`is_out_of_credit_error` or message-shaped via
:func:`message_signals_out_of_credit` — into :class:`OutOfCreditError`, so a
mid-run top-out fast-fails the run cleanly. Recognized signals: OpenRouter /
generic HTTP 402, Anthropic's HTTP-400 "credit balance is too low" (empty wallet)
and its HTTP-400 "reached your specified API usage limits" (monthly spend-limit
cap), and OpenAI's HTTP-429 ``insufficient_quota``. NOT recognized: ``vertex`` GCP exhaustion
surfaces as HTTP 403 (billing disabled) or 429 ``RESOURCE_EXHAUSTED``, which
classify as transient and are retried — there is NO mid-run money-stop for
Vertex, so rely on a GCP budget alert. Plan headroom accordingly when you mix
providers.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

_AUTH_KEY_URL = "https://openrouter.ai/api/v1/auth/key"


class OutOfCreditError(RuntimeError):
    """Raised when the OpenRouter account has insufficient balance."""


def check_openrouter_balance(
    *,
    http_get: Callable[..., Any] | None = None,
) -> dict:
    """Query OpenRouter for the current key's usage / limit.

    Returns ``{"usage": float, "limit": float | None, "is_free_tier": bool}``.
    Reads the API key from ``OPENROUTER_API_KEY``; ``http_get`` is injectable
    so tests can stub the HTTP layer without monkeypatching httpx.
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    if http_get is None:
        import httpx

        http_get = httpx.get
    response = http_get(
        _AUTH_KEY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10.0,
    )
    response.raise_for_status()
    data = response.json().get("data", {})
    return {
        "usage": data.get("usage", 0.0),
        "limit": data.get("limit"),
        "is_free_tier": data.get("is_free_tier", False),
    }


def preflight_credit_check(
    *,
    min_balance: float = 0.20,
    balance_fn: Callable[[], dict] | None = None,
) -> None:
    """Raise :class:`OutOfCreditError` if OpenRouter has less than ``min_balance`` left.

    Free-tier accounts (``limit`` is ``None``) are skipped — there is no balance
    to check. Designed to run once before dispatching expensive runs so the
    experiment fails fast instead of mid-run.
    """
    if balance_fn is None:
        balance_fn = check_openrouter_balance
    try:
        info = balance_fn()
    except KeyError:
        # OPENROUTER_API_KEY missing — Ollama-only experiments don't need a check.
        return
    limit = info.get("limit")
    if limit is None:
        return
    available = limit - info.get("usage", 0.0)
    if available < min_balance:
        raise OutOfCreditError(
            f"OpenRouter balance ${available:.4f} is below the safety margin "
            f"${min_balance:.4f}. Top up at https://openrouter.ai/credits and re-run."
        )


_CREDIT_KEYWORDS = (
    "insufficient credit",
    "insufficient_credit",
    "out of credit",
    # Anthropic native-SDK phrasing for an exhausted balance. It surfaces as HTTP
    # 400, so the 402/429 status checks never catch it — the message is the only
    # signal ("Your credit balance is too low to access the Anthropic API.").
    "credit balance is too low",
    "purchase credits",
    # OpenAI exhausted-balance phrasing. An empty wallet surfaces as HTTP 429 with
    # type ``insufficient_quota`` ("You exceeded your current quota …") — NOT a
    # rate limit. Without these it matched the bare "429" / "quota exceeded"
    # rate-limit signals (errors.py) and was retried to the cap, so the run never
    # fast-failed: it logged every pair as a per-cell error and exited 0, masking
    # total failure on a funded-looking but empty key.
    "insufficient_quota",
    "exceeded your current quota",
    # HTTP 402 reason phrase. Anchored to the words, NOT the bare number "402": a
    # 3-digit substring collides with epoch timestamps / token counts that appear
    # in unrelated 429 bodies and raised a false OutOfCreditError that contained a
    # whole model mid-run. Genuine 402s are still caught structurally by
    # ``is_out_of_credit_error`` (status_code == 402); this keyword only backstops
    # a 402 flattened into a message string with no status code left.
    "payment required",
    # Anthropic monthly spend-limit cap. A workspace/org that reaches its configured
    # monthly spend limit gets HTTP 400 invalid_request_error "You have reached your
    # specified API usage limits. You will regain access on <1st-of-next-month>." —
    # NOT a 402/429 and distinct from the "credit balance is too low" empty-wallet
    # 400, so it matched no money/rate signal and fell through to a generic
    # GhaError: the run never fast-failed and burned every remaining cell against
    # instant 400s (an n15 haiku stability run logged 1895 of these after ~79 cells).
    # Anchored to "reached your specified" so it also covers a "monthly spend limit"
    # phrasing variant without colliding with rate-limit wording. The user raises the
    # cap at console.anthropic.com/settings/limits, then resume backfills.
    "reached your specified",
)


def is_out_of_credit_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like an OpenRouter out-of-credit response.

    Recognises both HTTP 402 (Payment Required) and 429 responses whose body
    mentions insufficient credit. Works with httpx.HTTPStatusError-style
    exceptions exposing a ``response`` attribute with ``status_code`` / ``text``.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return False
    status_code = getattr(response, "status_code", None)
    if status_code == 402:
        return True
    if status_code == 429:
        body = (getattr(response, "text", "") or "").lower()
        return any(keyword in body for keyword in _CREDIT_KEYWORDS)
    return False


def message_signals_out_of_credit(message: str) -> bool:
    """Return True when an error MESSAGE mentions out-of-credit.

    The response-shaped check in :func:`is_out_of_credit_error` needs a
    ``.response`` with a status code. When only the text survives — e.g. a
    provider error surfaced as an agent error-status ``content`` string, where
    the original ``.response`` is gone — this keyword match on the message is the
    available signal. Used by the invoker to normalize either shape into
    :class:`OutOfCreditError`.
    """
    text = (message or "").lower()
    return any(keyword in text for keyword in _CREDIT_KEYWORDS)
