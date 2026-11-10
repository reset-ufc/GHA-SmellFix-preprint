"""
env.py — Agents configuration loader.

Reads .env for per-provider API keys, one per YAML provider: OPENROUTER_API_KEY
(openrouter), ANTHROPIC_API_KEY (anthropic), OPENAI_API_KEY (openai),
GOOGLE_API_KEY (gemini), plus the optional OLLAMA_BASE_URL override for the
local Ollama endpoint (defaults to http://localhost:11434/v1). The ``vertex``
provider is the lone exception: it carries NO API key — it bills Gemini through
a Google Cloud project (Vertex AI) and authenticates via Application Default
Credentials (``gcloud auth application-default login``), needing only
GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION (see :func:`_get_vertex_config`).

Model selection is driven exclusively by the experiment YAML ``models:``
section: :mod:`gha_agents.experiment` activates a :func:`model_overrides`
contextvar carrying the resolved ``role → model_id`` map, and
:func:`get_model_id` / :func:`get_model` raise without an active override.
``DEFAULT_MODEL`` / ``AGENT_<ROLE>_MODEL`` env vars are no longer consulted —
keep configuration in version control.

Provider routing uses the YAML entry's ``provider:`` field (anthropic / openai
/ gemini / vertex / openrouter / ollama), pinned in :data:`_provider_override_var` by
the dispatcher and read by :func:`_resolve_provider`; the legacy prefix-based
fallback (``ollama/`` → ollama, else openrouter) remains for a caller that
pins no provider. Sampling parameters
(temperature/top_p/reasoning_effort/thinking_level) come from the YAML via
:func:`sampling_overrides`, not .env, so the paper's "same sampling settings"
invariant lives in version-controlled config.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from dotenv import load_dotenv

from gha_agents.config import Provider

logger = logging.getLogger("study.env")

_env_loaded = False

# Per-run model overrides: a dict role → model_id. This is the **only**
# valid source of model IDs. Parallel dispatchers (see experiment.py) set
# this inside each worker thread so concurrent runs with different models
# don't race on the process-wide state.
_model_override_var: ContextVar[dict[str, str] | None] = ContextVar(
    "study_model_override",
    default=None,
)

# Per-run sampling overrides: dict of OpenAI-compatible sampling params
# (temperature/top_p). Same contextvar pattern as model overrides so
# concurrent runs can carry different sampling configs without racing.
_sampling_override_var: ContextVar[dict[str, float | int] | None] = ContextVar(
    "study_sampling_override",
    default=None,
)

# Per-run request timeout (seconds) for the underlying HTTP client. ``None``
# means no client-side timeout (provider hangs indefinitely on slow upstreams).
_timeout_override_var: ContextVar[float | None] = ContextVar(
    "study_request_timeout",
    default=None,
)

# Per-run provider override. Pinned by the experiment dispatcher to the
# ``ResolvedRun.provider`` value so concurrent workers don't have to re-derive
# the provider from the model id. ``None`` means "infer from the model id"
# (``ollama/`` prefix → ollama, anything else → openrouter).
_provider_override_var: ContextVar[Provider | None] = ContextVar(
    "study_provider_override",
    default=None,
)

# Per-run audit-trail toggle. When True, teams.py captures every Agent.run()
# invocation as an :class:`AgentCall` in ``WorkflowState.trace_calls`` so
# main.py can persist ``traces/{slug}.json`` alongside the result JSON.
_persist_traces_var: ContextVar[bool] = ContextVar(
    "study_persist_traces",
    default=False,
)

# Native-thinking toggle. On by default (matches the default sampling regime); a
# per-model ``sampling.thinking: false`` flips it off for that run so the model
# is built without extended thinking — the only way to get a deterministic
# Anthropic run, whose thinking mode forces temperature=1. See ModelFactory.build.
_thinking_override_var: ContextVar[bool] = ContextVar(
    "study_thinking",
    default=True,
)

# Structured-output mode toggle. On by default (use_json_mode=True), reproducing
# the historical hardcoded behaviour: Agno requests a loose ``json_object`` plus
# the schema as prompt text. A per-model ``sampling.json_mode: false`` flips one
# model to Agno's native strict ``json_schema`` response_format (grammar-
# constrained decoding) — the fix for reasoning models that otherwise stream
# chain-of-thought into the message content and never emit the JSON object. Read
# at agent construction in agents.py (mirrors is_thinking_active in ModelFactory).
_json_mode_override_var: ContextVar[bool] = ContextVar(
    "study_json_mode",
    default=True,
)

# Deterministic defaults — enforced when no override is active. Keeps
# behavior predictable for a caller that binds no sampling override.
_DEFAULT_SAMPLING: dict[str, float | int] = {
    "temperature": 0.0,
    "top_p": 1.0,
}


def _ensure_env_loaded() -> None:
    """Load .env on first call. Deferred to avoid module-level side effects."""
    global _env_loaded
    if _env_loaded:
        return

    _project_root = Path(__file__).resolve().parent.parent
    _env_file = _project_root / ".env"

    if _env_file.exists():
        load_dotenv(dotenv_path=_env_file)
    else:
        load_dotenv()
        logger.warning(".env not found — relying on system environment variables")

    _env_loaded = True


_OLLAMA_PREFIX = "ollama/"
_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


def _is_ollama(model_id: str) -> bool:
    return model_id.startswith(_OLLAMA_PREFIX)


def _get_api_key() -> str:
    """Retrieve the OpenRouter API key."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OSError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return api_key


def _get_anthropic_api_key() -> str:
    """Retrieve the Anthropic API key for native Claude dispatch."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise OSError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    return api_key


def _get_openai_api_key() -> str:
    """Retrieve the OpenAI API key for native GPT dispatch."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OSError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return api_key


def _get_gemini_api_key() -> str:
    """Retrieve the Gemini API key for native Gemini dispatch.

    Reads ``GOOGLE_API_KEY`` (Agno's default env var for the Gemini client;
    see ``agno/models/google/gemini.py:170``). The explicit lookup here gives
    us a clean fast-fail before any network attempt, mirroring the OpenRouter
    / Anthropic / OpenAI helpers.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise OSError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return api_key


def _get_vertex_config() -> tuple[str, str]:
    """Project + location for Vertex AI dispatch (provider ``vertex``).

    Vertex authenticates via Application Default Credentials — there is NO API
    key. Run ``gcloud auth application-default login`` once and the google-genai
    client (``vertexai=True``) discovers the ADC automatically; this helper only
    resolves the project + location the SDK needs, fast-failing on a missing
    project exactly as the API-key helpers fast-fail on a missing key.

    Reads ``GOOGLE_CLOUD_PROJECT`` (required) and ``GOOGLE_CLOUD_LOCATION``
    (optional, defaults to ``global``) — Agno's Gemini reads these same vars when
    ``vertexai=True`` (see ``agno/models/google/gemini.py:182-186``); passing them
    explicitly keeps the fast-fail close to the build call.

    Default is ``global``, NOT a single region: Gemini 3.x (e.g. the fleet's
    ``gemini-3.5-flash``) is *listed* in regional model gardens like
    ``us-central1`` but ``generateContent`` against it there returns HTTP 404
    "not found / no access" — it only actually serves from the ``global``
    endpoint (verified live 2026-06-20: ``us-central1`` 404s, ``global`` 200s with
    ``thoughts_token_count>0``). A regional value still works for any model that
    serves there; override via ``GOOGLE_CLOUD_LOCATION`` when needed.
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise OSError(
            "GOOGLE_CLOUD_PROJECT is not set. Add it to your .env file and run "
            "`gcloud auth application-default login` "
            "(Vertex uses ADC, not an API key)."
        )
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    return project, location


_AGENT_ROLES = ("detection", "detection_validation", "refactor", "refactor_validation", "combined")


def get_model_id(agent_key: str) -> str:
    """Return the model ID for ``agent_key`` from the active YAML override.

    The contextvar override populated by :func:`model_overrides` (typically
    by :mod:`gha_agents.experiment` from the YAML ``models:`` section) is
    the only valid source. Raises if no override is active for the role.
    """
    _ensure_env_loaded()
    override = _model_override_var.get()
    if override is not None and agent_key in override:
        value = override[agent_key].strip()
        if value:
            return value
    raise ValueError(
        f"No model configured for agent '{agent_key}'. "
        f"Set it via the experiment YAML 'models:' section and run "
        f"`python -m gha_agents.experiment <yaml>`, or choose one with "
        f"`gha-smellfix --model`."
    )


@contextmanager
def model_overrides(overrides: dict[str, str] | None) -> Iterator[None]:
    """Context manager that applies per-run model overrides.

    ``overrides`` maps agent role → model ID. Roles absent from the map fall
    back to env vars. Passing ``None`` or an empty dict is a no-op.
    """
    if not overrides:
        yield
        return
    token = _model_override_var.set(dict(overrides))
    try:
        yield
    finally:
        _model_override_var.reset(token)


@contextmanager
def sampling_overrides(overrides: dict[str, float | int] | None) -> Iterator[None]:
    """Context manager that applies per-run sampling overrides.

    ``overrides`` maps OpenAI sampling param (``temperature``/``top_p``) to a
    value. Missing keys fall back to :data:`_DEFAULT_SAMPLING`.
    Passing ``None`` or an empty dict is a no-op.
    """
    if not overrides:
        yield
        return
    token = _sampling_override_var.set(dict(overrides))
    try:
        yield
    finally:
        _sampling_override_var.reset(token)


@contextmanager
def persist_traces(enabled: bool) -> Iterator[None]:
    """Context manager that toggles audit-trail capture for the duration."""
    token = _persist_traces_var.set(bool(enabled))
    try:
        yield
    finally:
        _persist_traces_var.reset(token)


def is_persist_traces_active() -> bool:
    """Return True when the active context wants per-pair traces persisted."""
    return _persist_traces_var.get()


@contextmanager
def thinking_overrides(enabled: bool) -> Iterator[None]:
    """Context manager that toggles native thinking for the duration."""
    token = _thinking_override_var.set(bool(enabled))
    try:
        yield
    finally:
        _thinking_override_var.reset(token)


def is_thinking_active() -> bool:
    """Return True when the active context wants native thinking enabled."""
    return _thinking_override_var.get()


@contextmanager
def json_mode_overrides(enabled: bool) -> Iterator[None]:
    """Context manager toggling Agno JSON-mode (vs native json_schema) output."""
    token = _json_mode_override_var.set(bool(enabled))
    try:
        yield
    finally:
        _json_mode_override_var.reset(token)


def is_json_mode_active() -> bool:
    """Return True when agents should be built with ``use_json_mode=True``."""
    return _json_mode_override_var.get()


@contextmanager
def provider_overrides(provider: Provider | None) -> Iterator[None]:
    """Context manager that pins the active provider for :func:`get_model`.

    Passing ``None`` is a no-op (preserves whichever provider the surrounding
    context already set, or the legacy prefix-based inference fallback). The
    dispatcher in :mod:`gha_agents.experiment` always sets this so worker
    threads agree on which Agno model class to instantiate without re-parsing
    the model id.
    """
    if provider is None:
        yield
        return
    token = _provider_override_var.set(provider)
    try:
        yield
    finally:
        _provider_override_var.reset(token)


@contextmanager
def request_timeout_overrides(timeout: float | None) -> Iterator[None]:
    """Context manager that applies a per-run HTTP request timeout in seconds.

    Passing ``None`` is a no-op (preserves whatever the surrounding context
    already set, including "no timeout"). Pass an explicit float to enforce
    a client-side cutoff on each LLM call.
    """
    if timeout is None:
        yield
        return
    token = _timeout_override_var.set(timeout)
    try:
        yield
    finally:
        _timeout_override_var.reset(token)


def get_all_model_ids() -> dict[str, str]:
    """Return a snapshot of all active model IDs keyed by agent role."""
    return {role: get_model_id(role) for role in _AGENT_ROLES}


def _resolve_provider(model_id: str) -> Provider:
    """Resolve the active provider for ``model_id``.

    Prefers the pinned :data:`_provider_override_var` (set by the experiment
    dispatcher from :class:`ResolvedRun.provider`). Falls back to prefix
    inference when no provider is pinned: ``ollama/`` → ``ollama``, otherwise
    ``openrouter``.
    """
    pinned = _provider_override_var.get()
    if pinned is not None:
        return pinned
    return "ollama" if _is_ollama(model_id) else "openrouter"


# Model construction (the only Agno-importing code) lives in
# :mod:`gha_agents.model_factory`. ``env`` resolves model id + provider and
# owns the run-scoped contextvars and API keys; ModelFactory turns that into a
# concrete provider model. Keeping them apart leaves ``env`` framework-free.
