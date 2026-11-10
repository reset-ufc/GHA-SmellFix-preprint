"""Unit tests for the experiment YAML config schema and expansion."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from gha_agents.config import (
    DESIGN_TO_AGENTS,
    ExecutionSpec,
    ExperimentConfig,
    MatrixSpec,
    ModelSpec,
    ResolvedRun,
    RunSpec,
    expand_runs,
    load_config,
)
from gha_agents.errors import GhaConfigError


def _write(tmp_path: Path, body: str, name: str = "exp.yaml") -> Path:
    path = tmp_path / name
    path.write_text(dedent(body).lstrip(), encoding="utf-8")
    return path


# ─── load_config: file + YAML plumbing ────────────────────────────────────────


def test_load_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(GhaConfigError, match="Config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_empty_file_raises(tmp_path: Path):
    path = _write(tmp_path, "")
    with pytest.raises(GhaConfigError, match="is empty"):
        load_config(path)


def test_shipped_full_config_enables_persist_traces():
    """The paper run configs must ship with the audit trail on."""
    config_path = Path(__file__).parent.parent / "configs" / "full.yaml"
    cfg = load_config(config_path)
    assert cfg.execution.persist_traces is True


# ─── ModelSpec — provider routing schema ──────────────────────────────────────


def test_model_spec_accepts_known_providers():
    """Every native + routed provider we plan to dispatch through is accepted."""
    for provider in ("openrouter", "ollama", "anthropic", "openai", "gemini", "vertex"):
        ms = ModelSpec(id="vendor/model", provider=provider)
        assert ms.provider == provider


def test_model_spec_rejects_unknown_provider():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelSpec(id="vendor/model", provider="bogus")


def test_load_config_string_model_infers_openrouter(tmp_path: Path):
    """Bare string alias = back-compat → provider auto-inferred as openrouter."""
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          gemma: google/gemma-4-31b-it
        runs:
          - name: smoke
            design: D1
            model: gemma
        """,
    )
    cfg = load_config(path)
    assert isinstance(cfg.models["gemma"], ModelSpec)
    assert cfg.models["gemma"].id == "google/gemma-4-31b-it"
    assert cfg.models["gemma"].provider == "openrouter"


def test_load_config_ollama_string_infers_ollama_provider(tmp_path: Path):
    """Bare string with ollama/ prefix infers the ollama provider."""
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          qwen: ollama/qwen2.5-coder:7b
        runs:
          - name: smoke
            design: D1
            model: qwen
        """,
    )
    cfg = load_config(path)
    assert cfg.models["qwen"].provider == "ollama"


def test_load_config_explicit_anthropic_model_spec(tmp_path: Path):
    """Explicit dict form carries provider verbatim."""
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          claude:
            id: claude-sonnet-4-6
            provider: anthropic
        runs:
          - name: smoke
            design: D1
            model: claude
        """,
    )
    cfg = load_config(path)
    assert cfg.models["claude"].id == "claude-sonnet-4-6"
    assert cfg.models["claude"].provider == "anthropic"


def test_load_config_unknown_provider_rejected(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          weird:
            id: foo/bar
            provider: martian
        runs:
          - name: smoke
            design: D1
            model: weird
        """,
    )
    with pytest.raises(GhaConfigError):
        load_config(path)


def test_expand_runs_propagates_provider(tmp_path: Path):
    """ResolvedRun.provider mirrors the alias's provider for the whole run."""
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          claude:
            id: claude-sonnet-4-6
            provider: anthropic
        runs:
          - name: smoke
            design: D1
            model: claude
        """,
    )
    cfg = load_config(path)
    [run] = expand_runs(cfg)
    assert run.provider == "anthropic"


def test_expand_runs_string_model_resolves_openrouter(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          gemma: google/gemma-4-31b-it
        runs:
          - name: smoke
            design: D1
            model: gemma
        """,
    )
    cfg = load_config(path)
    [run] = expand_runs(cfg)
    assert run.provider == "openrouter"


def test_load_config_invalid_yaml_raises(tmp_path: Path):
    path = _write(tmp_path, "version: 1\nmodels: [unclosed")
    with pytest.raises(GhaConfigError, match="Invalid YAML"):
        load_config(path)


def test_load_config_non_mapping_root_raises(tmp_path: Path):
    path = _write(tmp_path, "- just_a_list\n")
    with pytest.raises(GhaConfigError, match="root must be a mapping"):
        load_config(path)


# ─── load_config: schema validation ───────────────────────────────────────────


def test_load_config_rejects_wrong_version(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        version: 2
        models:
          a: some/model
        runs:
          - name: r
            design: D1
            model: a
        """,
    )
    with pytest.raises(GhaConfigError, match="validation failed"):
        load_config(path)


def test_load_config_rejects_extra_keys(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        version: 1
        bogus: true
        models: {a: some/model}
        runs: [{name: r, design: D1, model: a}]
        """,
    )
    with pytest.raises(GhaConfigError, match="validation failed"):
        load_config(path)


def test_load_config_unknown_model_alias_in_matrix(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          a: some/model
        matrix:
          designs: [D1]
          models: [missing]
          repetitions: [1]
        """,
    )
    with pytest.raises(GhaConfigError, match="undefined aliases"):
        load_config(path)


def test_load_config_unknown_model_alias_in_runs(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          a: some/model
        runs:
          - name: r
            design: D1
            model: missing
        """,
    )
    with pytest.raises(GhaConfigError, match="undefined aliases"):
        load_config(path)


def test_load_config_rejects_dict_alias(tmp_path: Path):
    """Per-role dict-form aliases are no longer supported — only strings."""
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          per_role:
            detection: a/b
            refactor: c/d
        runs:
          - name: r
            design: D1
            model: per_role
        """,
    )
    with pytest.raises(GhaConfigError, match="validation failed"):
        load_config(path)


def test_load_config_rejects_empty_experiment(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        version: 1
        models:
          a: some/model
        """,
    )
    with pytest.raises(GhaConfigError, match="at least one run"):
        load_config(path)


def test_load_config_rejects_repetition_out_of_range(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        version: 1
        models: {a: some/model}
        matrix:
          designs: [D1]
          models: [a]
          repetitions: [0]
        """,
    )
    with pytest.raises(GhaConfigError, match="validation failed"):
        load_config(path)


# ─── expand_runs: matrix expansion ────────────────────────────────────────────


def test_expand_runs_matrix_cartesian_product():
    cfg = ExperimentConfig(
        models={"m1": "x/y", "m2": "a/b"},
        matrix=MatrixSpec(
            designs=["D1", "D2"],
            models=["m1", "m2"],
            repetitions=[1, 2],
        ),
    )
    runs = expand_runs(cfg)
    assert len(runs) == 2 * 2 * 2  # 8
    names = {r.name for r in runs}
    assert "matrix-D1-m1-rep1" in names
    assert "matrix-D2-m2-rep2" in names


def test_expand_runs_matrix_resolves_agents_from_config():
    cfg = ExperimentConfig(
        models={"m": "x/y"},
        matrix=MatrixSpec(designs=["D1", "D2"], models=["m"], repetitions=[1]),
    )
    runs = expand_runs(cfg)
    agents_by_design = {r.design: r.agents for r in runs}
    assert agents_by_design == DESIGN_TO_AGENTS


def test_expand_runs_string_alias_applies_to_all_roles():
    cfg = ExperimentConfig(
        models={"m": "x/y"},
        runs=[RunSpec(name="r", design="D1", model="m")],
    )
    [run] = expand_runs(cfg)
    assert run.model_map == dict.fromkeys(
        ("detection", "detection_validation", "refactor", "refactor_validation", "combined"),
        "x/y",
    )


# ─── per-model sampling override ──────────────────────────────────────────────


def test_expand_runs_per_model_sampling_override(tmp_path: Path):
    """A model's ``sampling`` block overrides only that model; others inherit."""
    path = _write(
        tmp_path,
        """
        sampling:
          temperature: 0.0
          top_p: 1.0
        models:
          base:
            id: a/b
            provider: openai
          tuned:
            id: c/d
            provider: openai
            sampling:
              temperature: 0.3
        matrix:
          designs: [D1]
          models: [base, tuned]
          repetitions: [1]
        """,
    )
    cfg = load_config(path)
    runs = {r.model_alias: r for r in expand_runs(cfg)}
    assert runs["base"].sampling["temperature"] == 0.0   # inherits global default
    assert runs["tuned"].sampling["temperature"] == 0.3  # overridden for this model
    assert runs["tuned"].sampling["top_p"] == 1.0        # unset key still inherited
    assert runs["base"].thinking is runs["tuned"].thinking is True
    assert runs["base"].json_mode is runs["tuned"].json_mode is True


def test_expand_runs_per_model_thinking_toggle(tmp_path: Path):
    """``sampling.thinking: false`` flips native thinking off for one model only."""
    path = _write(
        tmp_path,
        """
        models:
          claude:
            id: claude-x
            provider: anthropic
            sampling:
              thinking: false
          gpt:
            id: g/x
            provider: openai
        matrix:
          designs: [D1]
          models: [claude, gpt]
          repetitions: [1]
        """,
    )
    cfg = load_config(path)
    runs = {r.model_alias: r for r in expand_runs(cfg)}
    assert runs["claude"].thinking is False  # overridden off
    assert runs["gpt"].thinking is True      # global default on


def test_expand_runs_per_model_json_mode_toggle(tmp_path: Path):
    """``sampling.json_mode: false`` flips one model to native json_schema only."""
    path = _write(
        tmp_path,
        """
        models:
          reasoner:
            id: r/x
            provider: openrouter
            sampling:
              json_mode: false
          plain:
            id: p/x
            provider: openrouter
        matrix:
          designs: [D1]
          models: [reasoner, plain]
          repetitions: [1]
        """,
    )
    cfg = load_config(path)
    runs = {r.model_alias: r for r in expand_runs(cfg)}
    assert runs["reasoner"].json_mode is False  # overridden off → native json_schema
    assert runs["plain"].json_mode is True      # global default on


def test_model_sampling_override_rejects_unknown_key(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        models:
          m:
            id: a/b
            provider: openai
            sampling:
              bogus: 1
        matrix:
          designs: [D1]
          models: [m]
          repetitions: [1]
        """,
    )
    with pytest.raises(GhaConfigError):
        load_config(path)


def test_expand_runs_merges_defaults(tmp_path: Path):
    ground_truth = tmp_path / "o.json"
    cfg = ExperimentConfig(
        defaults={"limit": 5, "max_iterations": 7, "ground_truth_file": str(ground_truth)},  # type: ignore[arg-type]
        models={"m": "x/y"},
        runs=[RunSpec(name="r", design="D1", model="m")],
    )
    [run] = expand_runs(cfg)
    assert run.limit == 5
    assert run.max_iterations == 7
    assert run.ground_truth_file == ground_truth


def test_expand_runs_run_override_beats_defaults():
    cfg = ExperimentConfig(
        defaults={"limit": 220, "max_iterations": 3},  # type: ignore[arg-type]
        models={"m": "x/y"},
        runs=[
            RunSpec(
                name="smoke",
                design="D2",
                model="m",
                limit=1,
                max_iterations=5,
            ),
        ],
    )
    [run] = expand_runs(cfg)
    assert run.limit == 1
    assert run.max_iterations == 5


def test_expand_runs_matrix_overrides_beat_defaults():
    cfg = ExperimentConfig(
        defaults={"limit": None},  # type: ignore[arg-type]
        models={"m": "x/y"},
        matrix=MatrixSpec(
            designs=["D1"],
            models=["m"],
            repetitions=[1],
            overrides={"limit": 10},
        ),
    )
    [run] = expand_runs(cfg)
    assert run.limit == 10


def test_expand_runs_matrix_then_named():
    cfg = ExperimentConfig(
        models={"m": "x/y"},
        matrix=MatrixSpec(designs=["D1"], models=["m"], repetitions=[1]),
        runs=[RunSpec(name="extra", design="D2", model="m")],
    )
    runs = expand_runs(cfg)
    assert [r.name for r in runs] == ["matrix-D1-m-rep1", "extra"]


def test_expand_runs_empty_matrix_only_named():
    cfg = ExperimentConfig(
        models={"m": "x/y"},
        runs=[RunSpec(name="only", design="D1", model="m")],
    )
    [run] = expand_runs(cfg)
    assert run.name == "only"
    assert isinstance(run, ResolvedRun)


def test_expand_runs_appends_scenario_subdir_per_design(tmp_path: Path):
    base = tmp_path / "exp"
    cfg = ExperimentConfig(
        defaults={"output_dir": str(base)},  # type: ignore[arg-type]
        models={"m": "x/y"},
        matrix=MatrixSpec(designs=["D1", "D2"], models=["m"], repetitions=[1]),
    )
    runs = expand_runs(cfg)
    by_design = {r.design: r.output_dir for r in runs}
    assert by_design["D1"] == base / "design_d1"
    assert by_design["D2"] == base / "design_d2"


def test_expand_runs_scenario_subdir_applies_to_named_runs_too(tmp_path: Path):
    base = tmp_path / "base"
    custom = tmp_path / "custom"
    cfg = ExperimentConfig(
        defaults={"output_dir": str(base)},  # type: ignore[arg-type]
        models={"m": "x/y"},
        runs=[
            RunSpec(name="r1", design="D1", model="m"),
            RunSpec(name="r2", design="D2", model="m", output_dir=custom),
        ],
    )
    r1, r2 = expand_runs(cfg)
    assert r1.output_dir == base / "design_d1"
    assert r2.output_dir == custom / "design_d2"


# ─── ExecutionSpec.request_timeout ────────────────────────────────────────────


def test_execution_spec_request_timeout_default_is_180():
    spec = ExecutionSpec()
    assert spec.request_timeout == 180.0


def test_execution_spec_request_timeout_accepts_none():
    spec = ExecutionSpec(request_timeout=None)
    assert spec.request_timeout is None


def test_execution_spec_request_timeout_rejects_zero_and_negative():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExecutionSpec(request_timeout=0)
    with pytest.raises(ValidationError):
        ExecutionSpec(request_timeout=-5.0)


# ─── End-to-end through load_config ───────────────────────────────────────────


def test_end_to_end_matrix_plus_runs(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        version: 1
        defaults:
          ground_truth_file: ./final_ground_truth.json
          output_dir: ./output
          max_iterations: 3
        models:
          glm_free: z-ai/glm-4.5-air:free
          haiku: anthropic/claude-haiku-4-5
        matrix:
          designs: [D1, D2]
          models: [glm_free]
          repetitions: [1, 2]
        runs:
          - name: smoke_haiku
            design: D2
            model: haiku
            limit: 1
            max_iterations: 5
        execution:
          max_workers: 2
          stop_on_error: false
          resume: true
        """,
    )
    cfg = load_config(path)
    runs = expand_runs(cfg)

    assert len(runs) == 2 * 1 * 2 + 1
    assert cfg.execution.max_workers == 2

    smoke = next(r for r in runs if r.name == "smoke_haiku")
    assert smoke.design == "D2"
    assert smoke.agents == 2
    assert smoke.limit == 1
    assert smoke.max_iterations == 5
    assert smoke.model_map == dict.fromkeys(
        ("detection", "detection_validation", "refactor", "refactor_validation", "combined"),
        "anthropic/claude-haiku-4-5",
    )
