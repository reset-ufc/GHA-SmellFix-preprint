"""Unit tests for ModelFactory — provider dispatch, sampling projection, thinking (#47).

Re-homed from test_env.py when model construction left env.py. Adds coverage for
native thinking enabled by default: Anthropic ``thinking=`` (which forces
temperature=1 and drops top_p) and Gemini ``include_thoughts=True``.
"""

from __future__ import annotations

import gha_agents.env as env_mod
import pytest
from gha_agents.model_factory import (
    _ANTHROPIC_THINKING_BUDGET_TOKENS,
    _GEMINI_3X_THINKING_LEVEL,
    _REASONING_BUDGET_TOKENS,
    ModelFactory,
)


def _clear_model_env(monkeypatch):
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    for role in env_mod._AGENT_ROLES:
        monkeypatch.delenv(f"AGENT_{role.upper()}_MODEL", raising=False)


def _activate_override(overrides: dict[str, str]):
    return env_mod._model_override_var.set(dict(overrides))


@pytest.fixture()
def factory() -> ModelFactory:
    return ModelFactory()


@pytest.fixture()
def fake_env(monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    _clear_model_env(monkeypatch)
    overrides = dict.fromkeys(env_mod._AGENT_ROLES, "google/gemini-flash-3")
    token = _activate_override(overrides)
    yield
    env_mod._model_override_var.reset(token)


@pytest.fixture()
def ollama_env(monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "ollama/llama3.2"))
    yield
    env_mod._model_override_var.reset(token)


@pytest.fixture()
def anthropic_env(monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "claude-sonnet-4-5-20250929"))
    yield
    env_mod._model_override_var.reset(token)


@pytest.fixture()
def openai_env(monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "gpt-5.4-mini"))
    yield
    env_mod._model_override_var.reset(token)


@pytest.fixture()
def gemini_env(monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "gemini-3.5-flash"))
    yield
    env_mod._model_override_var.reset(token)


# ─── OpenRouter / Ollama dispatch ─────────────────────────────────────────────


def test_build_openrouter_unchanged(factory, fake_env):
    model = factory.build("detection")
    assert "openrouter" in model.base_url
    assert model.id == "google/gemini-flash-3"


def test_build_openrouter_enables_native_reasoning_by_default(factory, fake_env):
    # Thinking on by default → opt into OpenRouter's reasoning channel with the
    # uniform token budget so hybrid reasoners (gemma-4) emit reasoning_content at
    # the same depth as the native providers instead of silently None. The "nitro"
    # provider.sort rides alongside as the fleet routing default.
    model = factory.build("detection")
    assert model.extra_body == {
        "reasoning": {"max_tokens": _REASONING_BUDGET_TOKENS},
        "provider": {"sort": "throughput"},
    }


def test_build_openrouter_thinking_off_excludes_reasoning(factory, fake_env):
    # Thinking off → exclude: the model may still reason internally but returns no
    # reasoning tokens, so reasoning_content stays empty by request. The "nitro"
    # provider.sort still rides alongside regardless of the thinking toggle.
    with env_mod.thinking_overrides(False):
        model = factory.build("detection")
    assert model.extra_body == {
        "reasoning": {"exclude": True},
        "provider": {"sort": "throughput"},
    }


def test_build_openrouter_nitro_provider_sort_is_default(factory, fake_env):
    # "nitro" throughput routing is baked into the factory for every OpenRouter
    # model — not driven by config/YAML — so the slugs stay routing-agnostic and
    # the slower price-sorted default never silently re-applies.
    model = factory.build("detection")
    assert model.extra_body["provider"] == {"sort": "throughput"}


def test_build_ollama_base_url_and_prefix(factory, ollama_env):
    model = factory.build("detection")
    assert "11434" in model.base_url
    assert model.id == "llama3.2"
    assert model.api_key == "ollama"


def test_build_raises_without_override(factory, monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    _clear_model_env(monkeypatch)
    with pytest.raises(ValueError, match="No model configured"):
        factory.build("detection")


def test_build_reflects_model_override(factory, fake_env):
    with env_mod.model_overrides(dict.fromkeys(env_mod._AGENT_ROLES, "acme/foo")):
        assert factory.build("detection").id == "acme/foo"


# ─── timeout propagation ──────────────────────────────────────────────────────


def test_build_propagates_timeout_to_openai_chat(factory, fake_env):
    with env_mod.request_timeout_overrides(120.0):
        model = factory.build("detection")
    assert model.timeout == 120.0


def test_build_disables_sdk_retries_when_timeout_set(factory, fake_env):
    with env_mod.request_timeout_overrides(120.0):
        model = factory.build("detection")
    assert model.max_retries == 0


def test_build_omits_timeout_when_no_override(factory, fake_env):
    model = factory.build("detection")
    assert model.timeout is None
    assert model.max_retries is None


# ─── Anthropic native dispatch + thinking ─────────────────────────────────────


def test_build_anthropic_returns_claude_instance(factory, anthropic_env):
    from agno.models.anthropic import Claude

    with env_mod.provider_overrides("anthropic"):
        model = factory.build("detection")
    assert isinstance(model, Claude)
    assert model.id == "claude-sonnet-4-5-20250929"
    assert model.api_key == "fake-anthropic-key"


def test_build_anthropic_enables_thinking_and_sets_top_p(factory, anthropic_env):
    # Thinking on by default: thinking config set, temperature dropped (Anthropic
    # rejects temperature != 1 with extended thinking — leave it unset so the model
    # uses its own default) and top_p pinned to the 0.95 floor the API accepts. The
    # Claude SDK has no seed field at all.
    with env_mod.provider_overrides("anthropic"):
        model = factory.build("detection")
    assert model.thinking == {"type": "enabled", "budget_tokens": _ANTHROPIC_THINKING_BUDGET_TOKENS}
    assert model.temperature is None
    assert not hasattr(model, "seed")
    assert model.top_p == 0.95


def test_build_anthropic_missing_key_raises(factory, monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "claude-sonnet-4-5-20250929"))
    try:
        with (
            env_mod.provider_overrides("anthropic"),
            pytest.raises(OSError, match="ANTHROPIC_API_KEY"),
        ):
            factory.build("detection")
    finally:
        env_mod._model_override_var.reset(token)


def test_build_openrouter_provider_override_unchanged(factory, fake_env):
    with env_mod.provider_overrides("openrouter"):
        model = factory.build("detection")
    assert "openrouter" in model.base_url
    assert model.id == "google/gemini-flash-3"


def test_build_ollama_provider_override_unchanged(factory, ollama_env):
    with env_mod.provider_overrides("ollama"):
        model = factory.build("detection")
    assert "11434" in model.base_url
    assert model.id == "llama3.2"


# ─── OpenAI native dispatch ───────────────────────────────────────────────────


def test_build_openai_returns_openaichat_against_openai_default(factory, openai_env):
    from agno.models.openai import OpenAIChat

    with env_mod.provider_overrides("openai"):
        model = factory.build("detection")
    assert isinstance(model, OpenAIChat)
    assert model.id == "gpt-5.4-mini"
    assert model.api_key == "fake-openai-key"
    assert model.base_url is None


def test_build_openai_forwards_reasoning_effort(factory, openai_env):
    from agno.models.openai import OpenAIResponses

    with (
        env_mod.sampling_overrides({"reasoning_effort": "minimal"}),
        env_mod.provider_overrides("openai"),
    ):
        model = factory.build("detection")
    # A reasoning model routes through the Responses API (/v1/responses), not
    # chat/completions: reasoning + function-tool structured output is rejected on
    # chat/completions ("use /v1/responses instead").
    assert isinstance(model, OpenAIResponses)
    assert model.reasoning_effort == "minimal"
    assert getattr(model, "seed", None) is None  # seed never set
    # OpenAI reasoning models reject temperature != 1 AND top_p → the factory drops
    # both so the model defaults are used (sending either would 400).
    assert model.temperature is None
    assert getattr(model, "top_p", None) is None


def test_effective_sampling_openai_reasoning_drops_temperature(factory):
    # With a reasoning_effort set, effective_sampling_for must drop temperature so
    # the recorded provenance matches what the API received (no temperature).
    with env_mod.sampling_overrides(
        {"temperature": 0.0, "top_p": 1.0, "reasoning_effort": "medium"}
    ):
        sampling = factory.effective_sampling_for("openai")
    assert "temperature" not in sampling
    assert "top_p" not in sampling  # reasoning models reject top_p (Responses API 400)
    assert sampling["reasoning_effort"] == "medium"


def test_effective_sampling_openai_without_effort_keeps_temperature(factory):
    # No reasoning_effort → ordinary OpenAI model, temperature 0.0 is honoured.
    with env_mod.sampling_overrides({"temperature": 0.0, "top_p": 1.0}):
        sampling = factory.effective_sampling_for("openai")
    assert sampling["temperature"] == 0.0


def test_build_openai_missing_key_raises(factory, monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "gpt-5.4-mini"))
    try:
        with (
            env_mod.provider_overrides("openai"),
            pytest.raises(OSError, match="OPENAI_API_KEY"),
        ):
            factory.build("detection")
    finally:
        env_mod._model_override_var.reset(token)


# ─── Gemini native dispatch + thinking ────────────────────────────────────────


def test_build_gemini_returns_gemini_instance(factory, gemini_env):
    from agno.models.google import Gemini

    with env_mod.provider_overrides("gemini"):
        model = factory.build("detection")
    assert isinstance(model, Gemini)
    assert model.id == "gemini-3.5-flash"
    assert model.api_key == "fake-google-key"


def test_build_gemini_enables_include_thoughts(factory, gemini_env):
    with env_mod.provider_overrides("gemini"):
        model = factory.build("detection")
    assert model.include_thoughts is True
    # gemini-3.5-flash is 3.x: the depth knob is the thinking_level enum, NOT the
    # 2.5-era thinking_budget (which 3.x silently ignores / rejects).
    assert model.thinking_budget is None
    assert model.thinking_level == _GEMINI_3X_THINKING_LEVEL


def test_build_gemini_25_uses_thinking_budget(factory, gemini_env, monkeypatch):
    # Family switch the other way: a 2.5-era id keeps the token budget knob and
    # sends NO thinking_level. Guards the regression in both directions.
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "gemini-2.5-flash"))
    try:
        with env_mod.provider_overrides("gemini"):
            model = factory.build("detection")
    finally:
        env_mod._model_override_var.reset(token)
    assert model.include_thoughts is True
    assert model.thinking_budget == _REASONING_BUDGET_TOKENS
    assert model.thinking_level is None


def test_build_gemini_forwards_thinking_level(factory, gemini_env):
    with (
        env_mod.sampling_overrides({"thinking_level": "low"}),
        env_mod.provider_overrides("gemini"),
    ):
        model = factory.build("detection")
    assert model.thinking_level == "low"
    assert model.seed is None  # seed dropped for every provider


def test_build_gemini_missing_key_raises(factory, monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "gemini-3.5-flash"))
    try:
        with (
            env_mod.provider_overrides("gemini"),
            pytest.raises(OSError, match="GOOGLE_API_KEY"),
        ):
            factory.build("detection")
    finally:
        env_mod._model_override_var.reset(token)


# ─── Vertex AI dispatch (Gemini on Google Cloud, ADC auth) ────────────────────


@pytest.fixture()
def vertex_env(monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-gcp-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    # vertex authenticates via ADC — neither an API key nor OpenRouter is set.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "gemini-3.5-flash"))
    yield
    env_mod._model_override_var.reset(token)


def test_build_vertex_returns_gemini_via_vertexai(factory, vertex_env):
    from agno.models.google import Gemini

    with env_mod.provider_overrides("vertex"):
        model = factory.build("detection")
    assert isinstance(model, Gemini)
    assert model.id == "gemini-3.5-flash"
    assert model.vertexai is True
    assert model.project_id == "fake-gcp-proj"
    assert model.location == "us-central1"
    # ADC, not an API key: the vertex branch passes no api_key.
    assert model.api_key is None


def test_build_vertex_does_not_misroute_to_openrouter(factory, vertex_env):
    # Regression: build()'s dispatch tail used to be an UNCONDITIONAL OpenRouter
    # return, so any provider without an explicit branch was silently built as an
    # OpenAIChat on openrouter.ai with the OpenRouter key. vertex must NOT.
    from agno.models.openai import OpenAIChat

    with env_mod.provider_overrides("vertex"):
        model = factory.build("detection")
    assert not isinstance(model, OpenAIChat)
    base_url = getattr(model, "base_url", None)
    assert base_url is None or "openrouter" not in str(base_url)


def test_build_vertex_enables_include_thoughts_and_budget(factory, vertex_env):
    with env_mod.provider_overrides("vertex"):
        model = factory.build("detection")
    assert model.include_thoughts is True
    # gemini-3.5-flash on Vertex is still 3.x → thinking_level enum, not the 2.5
    # token budget (mirrors the gemini branch via the shared family-aware helper).
    assert model.thinking_budget is None
    assert model.thinking_level == _GEMINI_3X_THINKING_LEVEL


def test_build_vertex_missing_project_raises(factory, monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _clear_model_env(monkeypatch)
    token = _activate_override(dict.fromkeys(env_mod._AGENT_ROLES, "gemini-3.5-flash"))
    try:
        with (
            env_mod.provider_overrides("vertex"),
            pytest.raises(OSError, match="GOOGLE_CLOUD_PROJECT"),
        ):
            factory.build("detection")
    finally:
        env_mod._model_override_var.reset(token)


# ─── sampling projection ──────────────────────────────────────────────────────


def test_sampling_vertex_mirrors_gemini(factory):
    # vertex serves Gemini, so its sampling-key allow-list matches gemini:
    # temperature / top_p / thinking_level kept; seed + reasoning_effort dropped.
    with env_mod.sampling_overrides(
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "thinking_level": "low",
            "reasoning_effort": "high",
        }
    ):
        sampling = factory.sampling_for("vertex")
    assert sampling["temperature"] == 0.0
    assert sampling["top_p"] == 1.0
    assert sampling["thinking_level"] == "low"
    assert "seed" not in sampling
    assert "reasoning_effort" not in sampling


def test_sampling_openai_keeps_reasoning_effort_and_drops_seed(factory):
    with env_mod.sampling_overrides(
        {"temperature": 0.0, "top_p": 1.0, "seed": 42, "reasoning_effort": "minimal"}
    ):
        sampling = factory.sampling_for("openai")
    assert sampling["reasoning_effort"] == "minimal"
    assert "seed" not in sampling  # seed dropped for every provider


def test_sampling_anthropic_drops_reasoning_effort_and_seed(factory):
    with env_mod.sampling_overrides(
        {"temperature": 0.0, "top_p": 1.0, "seed": 42, "reasoning_effort": "minimal"}
    ):
        sampling = factory.sampling_for("anthropic")
    assert sampling == {"temperature": 0.0, "top_p": 1.0}


def test_sampling_drops_none_values(factory):
    with env_mod.sampling_overrides(
        {"temperature": 0.0, "top_p": 1.0, "seed": 42, "reasoning_effort": None}
    ):
        sampling = factory.sampling_for("openai")
    assert "reasoning_effort" not in sampling


def test_sampling_no_arg_resolves_pinned_provider(factory):
    # No-arg sampling_for() projects onto the active provider pin (matches the
    # row recorded into result metadata). Anthropic takes neither reasoning knob.
    with (
        env_mod.sampling_overrides({"reasoning_effort": "minimal"}),
        env_mod.provider_overrides("anthropic"),
    ):
        sampling = factory.sampling_for()
    assert "reasoning_effort" not in sampling


def test_effective_sampling_anthropic_drops_temperature_and_sets_top_p(factory):
    # effective_sampling_for folds in the overrides build() applies: Anthropic
    # thinking drops temperature (model default) and pins top_p=0.95, so the value
    # recorded into result metadata matches what the API actually received.
    with env_mod.sampling_overrides({"temperature": 0.3, "top_p": 1.0}):
        sampling = factory.effective_sampling_for("anthropic")
    assert sampling == {"top_p": 0.95}


def test_effective_sampling_non_anthropic_records_configured_value(factory):
    # No other provider constrains sampling, so the configured value is recorded
    # verbatim — set 0.3, get 0.3.
    with env_mod.sampling_overrides({"temperature": 0.3, "top_p": 0.9}):
        sampling = factory.effective_sampling_for("openai")
    assert sampling == {"temperature": 0.3, "top_p": 0.9}


def test_effective_sampling_anthropic_thinking_off_honors_temperature(factory):
    # With thinking off, Anthropic no longer forces temperature=1: the configured
    # value is honoured and top_p is kept.
    with env_mod.sampling_overrides({"temperature": 0.0, "top_p": 1.0}):
        sampling = factory.effective_sampling_for("anthropic", thinking_on=False)
    assert sampling == {"temperature": 0.0, "top_p": 1.0}


def test_build_anthropic_thinking_off_drops_thinking_and_honors_temperature(factory, anthropic_env):
    # thinking off → no thinking kwarg, and the configured temperature/top_p pass
    # through (the deterministic-baseline path).
    with (
        env_mod.sampling_overrides({"temperature": 0.0, "top_p": 1.0}),
        env_mod.thinking_overrides(False),
        env_mod.provider_overrides("anthropic"),
    ):
        model = factory.build("detection")
    assert model.thinking is None
    assert model.temperature == 0.0
    assert model.top_p == 1.0


def test_build_gemini_thinking_off_disables_include_thoughts(factory, gemini_env):
    with env_mod.thinking_overrides(False), env_mod.provider_overrides("gemini"):
        model = factory.build("detection")
    assert model.include_thoughts is False
    assert model.thinking_budget is None
