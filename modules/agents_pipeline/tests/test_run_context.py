"""Unit tests for ``RunContext`` + ``RunScope`` (#44).

The Parameter Object bundles the seven run-scoped settings; ``RunScope`` binds
them atomically (provider before sampling, honouring the read-order hazard in
``_get_sampling``) and tears them down in one shot. The existing per-setting
context managers remain the binding primitives — these tests prove the bundle
drives the same env readers a parallel matrix cell relies on.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from gha_agents.env import (
    _resolve_provider,
    _sampling_override_var,
    _timeout_override_var,
    get_model_id,
    is_json_mode_active,
    is_persist_traces_active,
    is_thinking_active,
)
from gha_agents.model_factory import ModelFactory
from gha_agents.run_context import RunContext, RunScope

# The merged→projected sampling read now lives on ModelFactory (env is
# framework-free); the read-order hazard it exercises is unchanged.
get_sampling = ModelFactory().sampling_for

_FULL = RunContext(
    provider="anthropic",
    model_overrides={"detection": "claude-x", "combined": "claude-x"},
    sampling_overrides={"temperature": 0.5},
    request_timeout=30.0,
    persist_traces=True,
    thinking=False,
    json_mode=False,
)


def test_scope_binds_all_seven_settings_for_the_readers():
    with RunScope(_FULL):
        assert get_model_id("detection") == "claude-x"
        assert _resolve_provider("ignored-model-id") == "anthropic"
        assert _timeout_override_var.get() == 30.0
        assert is_persist_traces_active() is True
        assert is_thinking_active() is False
        assert is_json_mode_active() is False
        # Provider was bound before sampling is read → projection onto the
        # anthropic key set happened. temperature carries the override.
        sampling = get_sampling()
        assert sampling == {"temperature": 0.5, "top_p": 1.0}


def test_scope_tears_down_every_setting_on_exit():
    with RunScope(_FULL):
        pass
    # Model override gone → get_model_id raises (no active override).
    with pytest.raises(ValueError):
        get_model_id("detection")
    # Provider override gone → legacy prefix inference (non-ollama → openrouter).
    assert _resolve_provider("some-model") == "openrouter"
    assert _timeout_override_var.get() is None
    assert is_persist_traces_active() is False
    assert _sampling_override_var.get() is None
    assert is_thinking_active() is True
    assert is_json_mode_active() is True


def test_scope_teardown_is_atomic_even_when_body_raises():
    sentinel = RuntimeError("boom")
    with pytest.raises(RuntimeError), RunScope(_FULL):
        assert is_persist_traces_active() is True
        raise sentinel
    # All five reset despite the exception unwinding through the scope.
    assert is_persist_traces_active() is False
    assert _timeout_override_var.get() is None
    assert _sampling_override_var.get() is None
    with pytest.raises(ValueError):
        get_model_id("detection")


def test_empty_context_is_inert():
    # A default RunContext binds nothing observable: provider falls back to
    # legacy inference, traces stay off, no model override is installed.
    with RunScope(RunContext()):
        assert _resolve_provider("ollama/llama-x") == "ollama"
        assert _resolve_provider("gpt-x") == "openrouter"
        assert is_persist_traces_active() is False
        assert _timeout_override_var.get() is None


def test_provider_bound_before_sampling_projects_reasoning_effort_away():
    # The load-bearing ordering guarantee: OpenAI keeps reasoning_effort,
    # anthropic drops it — proving sampling is read with the provider live.
    over = {"reasoning_effort": "minimal"}
    with RunScope(RunContext(provider="openai", sampling_overrides=over)):
        assert get_sampling().get("reasoning_effort") == "minimal"
    with RunScope(RunContext(provider="anthropic", sampling_overrides=over)):
        assert "reasoning_effort" not in get_sampling()


def test_run_context_is_frozen():
    ctx = RunContext(provider="openai")
    with pytest.raises(FrozenInstanceError):
        ctx.provider = "anthropic"  # type: ignore[misc]


def test_for_run_carries_the_resolved_model_and_sampling():
    from gha_agents.config import ExperimentConfig, expand_runs

    cfg = ExperimentConfig.model_validate(
        {
            "models": {
                "gpt": {
                    "id": "gpt-x",
                    "provider": "openai",
                    "sampling": {"reasoning_effort": "medium", "json_mode": False},
                }
            },
            "runs": [{"name": "one", "design": "D2", "model": "gpt"}],
        }
    )
    (spec,) = expand_runs(cfg)

    ctx = RunContext.for_run(spec, request_timeout=30.0, persist_traces=True)

    assert ctx == RunContext(
        provider="openai",
        model_overrides=spec.model_map,
        sampling_overrides=spec.sampling,
        request_timeout=30.0,
        persist_traces=True,
        thinking=True,
        json_mode=False,
    )
    assert spec.sampling["reasoning_effort"] == "medium"
