"""Unit tests for env.py — model-ID resolution + the run-scoped contextvars.

Model construction (provider dispatch, sampling projection, thinking) moved to
ModelFactory; those tests live in test_model_factory.py. env.py is now
framework-free, so nothing here imports agno.
"""

import gha_agents.env as env_mod
import pytest


def _clear_model_env(monkeypatch):
    """Drop env-based model vars so any reliance on them surfaces as a failure."""
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    for role in env_mod._AGENT_ROLES:
        monkeypatch.delenv(f"AGENT_{role.upper()}_MODEL", raising=False)


def _activate_override(overrides: dict[str, str]):
    """Set the model-override contextvar and return the token for cleanup."""
    return env_mod._model_override_var.set(dict(overrides))


@pytest.fixture()
def fake_env(monkeypatch):
    """Configure an all-roles override mirroring a YAML model_map."""
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    _clear_model_env(monkeypatch)
    overrides = {
        "detection": "google/gemini-flash-3",
        "detection_validation": "google/gemini-flash-3",
        "refactor": "anthropic/claude-haiku-4-5",
        "refactor_validation": "google/gemini-flash-3",
        "combined": "google/gemini-flash-3",
    }
    token = _activate_override(overrides)
    yield overrides
    env_mod._model_override_var.reset(token)


# ── Model-ID resolution ───────────────────────────────────────────────────────


def test_get_model_id_returns_overridden_model(fake_env):
    assert env_mod.get_model_id("detection") == "google/gemini-flash-3"


def test_get_model_id_raises_without_override(monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    _clear_model_env(monkeypatch)
    with pytest.raises(ValueError, match="No model configured"):
        env_mod.get_model_id("combined")


def test_get_model_id_raises_when_role_missing_from_override(monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    _clear_model_env(monkeypatch)
    token = _activate_override({"detection": "only/this"})
    try:
        with pytest.raises(ValueError, match="No model configured"):
            env_mod.get_model_id("refactor")
    finally:
        env_mod._model_override_var.reset(token)


def test_get_all_model_ids_returns_dict_with_all_roles(fake_env):
    ids = env_mod.get_all_model_ids()
    assert isinstance(ids, dict)
    for role in env_mod._AGENT_ROLES:
        assert role in ids


def test_get_all_model_ids_uses_override_values(fake_env):
    ids = env_mod.get_all_model_ids()
    assert ids["detection"] == "google/gemini-flash-3"
    assert ids["refactor"] == "anthropic/claude-haiku-4-5"


# ── Model-override ContextVar ─────────────────────────────────────────────────


def test_model_overrides_applies_override(fake_env):
    assert env_mod.get_model_id("detection") == "google/gemini-flash-3"
    with env_mod.model_overrides({"detection": "override/model"}):
        assert env_mod.get_model_id("detection") == "override/model"
    assert env_mod.get_model_id("detection") == "google/gemini-flash-3"


def test_model_overrides_replaces_outer_completely(fake_env):
    # Inner override REPLACES outer (token-based set) — it does not merge.
    with env_mod.model_overrides({"detection": "only/this"}):
        assert env_mod.get_model_id("detection") == "only/this"
        with pytest.raises(ValueError, match="No model configured"):
            env_mod.get_model_id("refactor")


def test_model_overrides_empty_dict_is_noop(fake_env):
    before = env_mod.get_model_id("detection")
    with env_mod.model_overrides({}):
        assert env_mod.get_model_id("detection") == before


def test_model_overrides_none_is_noop(fake_env):
    before = env_mod.get_model_id("detection")
    with env_mod.model_overrides(None):
        assert env_mod.get_model_id("detection") == before


def test_model_overrides_nested(fake_env):
    with env_mod.model_overrides({"detection": "outer"}):
        assert env_mod.get_model_id("detection") == "outer"
        with env_mod.model_overrides({"detection": "inner"}):
            assert env_mod.get_model_id("detection") == "inner"
        assert env_mod.get_model_id("detection") == "outer"


def test_model_overrides_whitespace_value_raises(fake_env):
    with (
        env_mod.model_overrides({"detection": "   "}),
        pytest.raises(ValueError, match="No model configured"),
    ):
        env_mod.get_model_id("detection")


# ── Request-timeout ContextVar ────────────────────────────────────────────────


def test_request_timeout_overrides_none_is_noop():
    assert env_mod._timeout_override_var.get() is None
    with env_mod.request_timeout_overrides(None):
        assert env_mod._timeout_override_var.get() is None
    assert env_mod._timeout_override_var.get() is None


def test_request_timeout_overrides_sets_and_resets():
    assert env_mod._timeout_override_var.get() is None
    with env_mod.request_timeout_overrides(120.0):
        assert env_mod._timeout_override_var.get() == 120.0
    assert env_mod._timeout_override_var.get() is None


# ── Provider-override ContextVar ──────────────────────────────────────────────


def test_provider_overrides_none_is_noop():
    assert env_mod._provider_override_var.get() is None
    with env_mod.provider_overrides(None):
        assert env_mod._provider_override_var.get() is None
    assert env_mod._provider_override_var.get() is None


def test_provider_overrides_sets_and_resets():
    assert env_mod._provider_override_var.get() is None
    with env_mod.provider_overrides("anthropic"):
        assert env_mod._provider_override_var.get() == "anthropic"
    assert env_mod._provider_override_var.get() is None


def test_provider_overrides_nested():
    with env_mod.provider_overrides("anthropic"):
        assert env_mod._provider_override_var.get() == "anthropic"
        with env_mod.provider_overrides("openrouter"):
            assert env_mod._provider_override_var.get() == "openrouter"
        assert env_mod._provider_override_var.get() == "anthropic"
    assert env_mod._provider_override_var.get() is None


# ── JSON-mode ContextVar ──────────────────────────────────────────────────────


def test_json_mode_active_defaults_true():
    assert env_mod.is_json_mode_active() is True


def test_json_mode_overrides_sets_and_resets():
    assert env_mod.is_json_mode_active() is True
    with env_mod.json_mode_overrides(False):
        assert env_mod.is_json_mode_active() is False
    assert env_mod.is_json_mode_active() is True


# ─── Vertex AI credentials (provider "vertex") ────────────────────────────────


def test_get_vertex_config_returns_project_and_location(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
    assert env_mod._get_vertex_config() == ("my-gcp-proj", "europe-west4")


def test_get_vertex_config_defaults_location_to_global(monkeypatch):
    # Default is ``global``, NOT a region: Gemini 3.x (gemini-3.5-flash) is listed
    # in regional gardens like us-central1 but generateContent 404s there; it only
    # serves from the global endpoint (verified live 2026-06-20). A regional value
    # still works via the env override for models that serve there.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-proj")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    project, location = env_mod._get_vertex_config()
    assert project == "my-gcp-proj"
    assert location == "global"


def test_get_vertex_config_raises_without_project(monkeypatch):
    # Vertex uses ADC, not an API key — but the project is mandatory and must
    # fast-fail with a clear message (mirrors the API-key helpers).
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(OSError, match="GOOGLE_CLOUD_PROJECT"):
        env_mod._get_vertex_config()
