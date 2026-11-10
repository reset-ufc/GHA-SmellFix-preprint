"""
config.py — YAML experiment configuration for Agents.

Declarative spec that drives the paper Step 5 matrix (2 designs x N models x
220 ground_truth entries x K repetitions). A single YAML file defines:

  * ``defaults``  — CLI defaults (ground_truth path, output dir, limit, repetition,
    max_iterations, verbose).
  * ``models``    — alias → model ID string. The same model is applied to
    every agent role; per-role overrides are not supported.
  * ``matrix``    — cartesian product over ``designs`` x ``models`` x
    ``repetitions`` plus optional ``overrides``.
  * ``runs``      — individually-named runs appended to the matrix expansion.
  * ``execution`` — orchestrator hints (``max_workers``, ``stop_on_error``,
    ``resume``).

``load_config(path)`` parses and validates the YAML. ``expand_runs(cfg)``
flattens the matrix and named runs into a list of :class:`ResolvedRun`
instances ready for dispatch.

Designs use the paper's names: D1, the joint design (one combined agent), and D2,
the decomposed design (a detection agent, then a refactoring agent). Each run is
stored under its design: the ``design_d1/`` folder, the ``design`` field of the
result JSON and the ``@D1@`` part of the file names.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gha_agents.errors import GhaConfigError

__all__ = [
    "DESIGN_TO_AGENTS",
    "Defaults",
    "Design",
    "ExecutionSpec",
    "ExperimentConfig",
    "MatrixSpec",
    "ModelSpec",
    "Provider",
    "ResolvedRun",
    "RunSpec",
    "SamplingOverride",
    "SamplingSpec",
    "expand_runs",
    "load_config",
]

# The paper's name of each design.
Design = Literal["D1", "D2"]
Provider = Literal["openrouter", "ollama", "anthropic", "openai", "gemini", "vertex"]


class SamplingOverride(BaseModel):
    """Per-model sampling override — only the keys set here replace the global.

    Every field defaults to ``None`` meaning "inherit the global
    :class:`SamplingSpec` value". A model entry may set any subset; the rest fall
    through to the matrix-wide default. ``thinking`` toggles native thinking for
    just this model — set it ``false`` to build the model without extended
    thinking (the only way to honour a ``temperature`` override on Anthropic,
    whose thinking mode otherwise forces ``temperature=1``).
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning_effort: str | None = Field(default=None)
    thinking_level: str | None = Field(default=None)
    thinking: bool | None = Field(default=None)
    json_mode: bool | None = Field(default=None)


class ModelSpec(BaseModel):
    """Per-model entry — model id plus the provider it dispatches through.

    The provider determines which Agno model class
    :meth:`gha_agents.model_factory.ModelFactory.build`
    uses for this alias: the native SDK for ``"anthropic"`` / ``"openai"`` /
    ``"gemini"``, the local OpenAI-compatible endpoint for ``"ollama"``, and
    the OpenRouter proxy for ``"openrouter"``. Sampling profiles are
    per-provider — e.g. only OpenAI accepts a ``reasoning_effort`` knob. An
    optional ``sampling`` block overrides the matrix-wide default for this model
    alone (see :class:`SamplingOverride`).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    provider: Provider
    sampling: SamplingOverride | None = None


# Must match env.py's _AGENT_ROLES.
AGENT_ROLES: tuple[str, ...] = (
    "detection",
    "detection_validation",
    "refactor",
    "refactor_validation",
    "combined",
)

DESIGN_TO_AGENTS: dict[Design, int] = {"D1": 1, "D2": 2}


# ─── Schema models ────────────────────────────────────────────────────────────


class Defaults(BaseModel):
    """CLI-level defaults applied to every run unless overridden."""

    model_config = ConfigDict(extra="forbid")

    ground_truth_file: Path = Path("./final_ground_truth.json")
    output_dir: Path = Path("./output")
    limit: int | None = Field(default=None, ge=1)
    repetition: int = Field(default=1, ge=1, le=3)
    max_iterations: int = Field(default=3, ge=1, le=10)
    verbose: bool = False


class MatrixSpec(BaseModel):
    """Cartesian product of dimensions that expand into runs."""

    model_config = ConfigDict(extra="forbid")

    designs: list[Design] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    repetitions: list[int] = Field(default_factory=list)
    overrides: dict = Field(default_factory=dict)

    @field_validator("repetitions")
    @classmethod
    def _reps_in_range(cls, reps: list[int]) -> list[int]:
        for r in reps:
            if r < 1 or r > 3:
                raise ValueError(f"repetition must be in [1,3], got {r}")
        return reps

    def is_empty(self) -> bool:
        return not (self.designs and self.models and self.repetitions)


class RunSpec(BaseModel):
    """A single named run appended to the matrix expansion."""

    model_config = ConfigDict(extra="forbid")

    name: str
    design: Design
    model: str
    repetition: int = Field(default=1, ge=1, le=3)
    limit: int | None = Field(default=None, ge=1)
    max_iterations: int | None = Field(default=None, ge=1, le=10)
    ground_truth_file: Path | None = None
    output_dir: Path | None = None


class ExecutionSpec(BaseModel):
    """Orchestrator hints for the dispatcher."""

    model_config = ConfigDict(extra="forbid")

    max_workers: int = Field(default=1, ge=1)
    stop_on_error: bool = False
    resume: bool = True
    request_timeout: float | None = Field(default=180.0, gt=0)
    persist_traces: bool = False


class SamplingSpec(BaseModel):
    """LLM sampling parameters applied uniformly to every model in the matrix.

    The matrix-wide default sampling, applied to every model unless a per-model
    :class:`SamplingOverride` (``models.<alias>.sampling``) replaces specific
    keys for one model. Defaults are ``temperature=0.0``, ``top_p=1.0`` and
    native thinking ``on``. There is no ``seed``: Anthropic's SDK has no seed
    knob and the other providers honour it only best-effort, so the package does
    not claim bit-level reproducibility (see CONTEXT.md "Sampling regime").

    Provider-specific reasoning knobs (``reasoning_effort`` for OpenAI,
    ``thinking_level`` for Gemini) live here too — every model in the matrix
    that accepts the knob receives the same value. Each provider's
    :data:`gha_agents.env._PROVIDER_SAMPLING_KEYS` declares which subset of
    these keys it actually forwards to the model constructor;
    :meth:`gha_agents.model_factory.ModelFactory.sampling_for` strips the rest
    (and drops any ``None`` values so unset optional knobs don't surface as
    explicit no-op kwargs).
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning_effort: str | None = Field(default=None)
    thinking_level: str | None = Field(default=None)
    thinking: bool = Field(default=True)
    json_mode: bool = Field(default=True)


class ExperimentConfig(BaseModel):
    """Root schema for an experiment YAML."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    defaults: Defaults = Field(default_factory=Defaults)
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    matrix: MatrixSpec = Field(default_factory=MatrixSpec)
    runs: list[RunSpec] = Field(default_factory=list)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_model_specs(cls, value: object) -> object:
        """Accept legacy ``alias: id_string`` shorthand and coerce to ModelSpec.

        Bare strings are routed to ``ollama`` when prefixed ``ollama/`` and to
        ``openrouter`` otherwise — the two pre-existing dispatch paths. Native
        providers (anthropic / openai / gemini) require the explicit
        ``{id, provider}`` dict form.
        """
        if not isinstance(value, dict):
            return value
        coerced: dict[str, object] = {}
        for alias, entry in value.items():
            if isinstance(entry, str):
                provider = "ollama" if entry.startswith("ollama/") else "openrouter"
                coerced[alias] = {"id": entry, "provider": provider}
            else:
                coerced[alias] = entry
        return coerced

    @model_validator(mode="after")
    def _aliases_exist(self) -> ExperimentConfig:
        known = set(self.models)

        missing_matrix = [a for a in self.matrix.models if a not in known]
        if missing_matrix:
            raise ValueError(
                f"matrix.models reference undefined aliases: {missing_matrix}. "
                f"Defined: {sorted(known)}"
            )

        missing_runs = [r.model for r in self.runs if r.model not in known]
        if missing_runs:
            raise ValueError(
                f"runs reference undefined aliases: {missing_runs}. Defined: {sorted(known)}"
            )

        if self.matrix.is_empty() and not self.runs:
            raise ValueError("config must define at least one run (via 'matrix' or 'runs')")

        return self


# ─── Resolved run ─────────────────────────────────────────────────────────────


class ResolvedRun(BaseModel):
    """A fully-materialized run ready for dispatch.

    ``model_map`` carries the role → model_id pairs that override env at
    dispatch time (see :meth:`gha_agents.model_factory.ModelFactory.build`). ``provider`` is
    the alias's provider — applied to every role in the run since within-run
    sampling parity requires a single dispatch path. ``sampling`` is the
    effective sampling for this model (matrix default ← per-model override) and
    ``thinking`` the resolved native-thinking flag, both bound per cell.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    design: Design
    agents: int
    model_alias: str
    model_map: dict[str, str]
    provider: Provider = "openrouter"
    ground_truth_file: Path
    output_dir: Path
    limit: int | None
    repetition: int
    max_iterations: int
    verbose: bool
    sampling: dict = Field(default_factory=dict)
    thinking: bool = True
    json_mode: bool = True


# ─── Loader + expander ────────────────────────────────────────────────────────


def load_config(path: Path) -> ExperimentConfig:
    """Parse and validate an experiment YAML file."""
    if not path.exists():
        raise GhaConfigError(f"Config file not found: '{path}'")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GhaConfigError(f"Invalid YAML in '{path}': {exc}") from exc

    if raw is None:
        raise GhaConfigError(f"Config file '{path}' is empty")
    if not isinstance(raw, dict):
        raise GhaConfigError(
            f"Config root must be a mapping, got {type(raw).__name__} in '{path}'"
        )

    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as exc:
        raise GhaConfigError(f"Config validation failed for '{path}': {exc}") from exc


def _resolve_model_map(spec: ModelSpec) -> dict[str, str]:
    """Expand a :class:`ModelSpec` to a ``{role: model_id}`` dict.

    The same model id is applied to every agent role — per-role overrides
    are not supported (see :class:`ExperimentConfig.models`). The provider
    travels separately on :class:`ResolvedRun.provider` so dispatch can pick
    the right Agno model class without re-parsing the id.
    """
    return dict.fromkeys(AGENT_ROLES, spec.id)


def _merge_defaults(
    defaults: Defaults,
    overrides: dict,
    run_overrides: dict | None = None,
) -> dict:
    """Merge defaults ← matrix.overrides ← run-level overrides."""
    merged = defaults.model_dump()
    merged.update({k: v for k, v in overrides.items() if v is not None})
    if run_overrides:
        merged.update({k: v for k, v in run_overrides.items() if v is not None})
    return merged


def _resolve_sampling(
    global_spec: SamplingSpec, override: SamplingOverride | None
) -> tuple[dict, bool, bool]:
    """Merge a per-model sampling override onto the matrix-wide default.

    Returns ``(sampling, thinking, json_mode)`` where ``sampling`` carries the
    projectable params (temperature / top_p / reasoning_effort / thinking_level),
    ``thinking`` is the resolved native-thinking flag, and ``json_mode`` the
    resolved Agno JSON-mode flag. Only the keys the override sets (non-``None``)
    replace the global value; everything else inherits the global.
    """
    sampling = global_spec.model_dump()
    thinking = sampling.pop("thinking")
    json_mode = sampling.pop("json_mode")
    if override is not None:
        changes = override.model_dump(exclude_none=True)
        thinking = changes.pop("thinking", thinking)
        json_mode = changes.pop("json_mode", json_mode)
        sampling.update(changes)
    return sampling, thinking, json_mode


def _scenario_subdir(design: Design) -> str:
    """Per-design subdir name appended to ``output_dir`` (``D1`` → ``design_d1``)."""
    return f"design_{design.lower()}"


def _build_resolved(
    *,
    name: str,
    design: Design,
    model_alias: str,
    model_map: dict[str, str],
    provider: Provider,
    repetition: int,
    merged: dict,
    sampling: dict,
    thinking: bool,
    json_mode: bool,
) -> ResolvedRun:
    return ResolvedRun(
        name=name,
        design=design,
        agents=DESIGN_TO_AGENTS[design],
        model_alias=model_alias,
        model_map=model_map,
        provider=provider,
        ground_truth_file=Path(merged["ground_truth_file"]),
        output_dir=Path(merged["output_dir"]) / _scenario_subdir(design),
        limit=merged.get("limit"),
        repetition=repetition,
        max_iterations=merged["max_iterations"],
        verbose=merged["verbose"],
        sampling=sampling,
        thinking=thinking,
        json_mode=json_mode,
    )


def expand_runs(cfg: ExperimentConfig) -> list[ResolvedRun]:
    """Flatten matrix + named runs into a list of ResolvedRun.

    Matrix expansion comes first; named runs are appended in declaration order.
    """
    resolved: list[ResolvedRun] = []
    overrides = cfg.matrix.overrides

    if not cfg.matrix.is_empty():
        combos = product(cfg.matrix.designs, cfg.matrix.models, cfg.matrix.repetitions)
        for design, alias, rep in combos:
            merged = _merge_defaults(cfg.defaults, overrides)
            merged["repetition"] = rep
            name = f"matrix-{design}-{alias}-rep{rep}"
            spec = cfg.models[alias]
            sampling, thinking, json_mode = _resolve_sampling(cfg.sampling, spec.sampling)
            resolved.append(
                _build_resolved(
                    name=name,
                    design=design,
                    model_alias=alias,
                    model_map=_resolve_model_map(spec),
                    provider=spec.provider,
                    repetition=rep,
                    merged=merged,
                    sampling=sampling,
                    thinking=thinking,
                    json_mode=json_mode,
                )
            )

    for run in cfg.runs:
        run_overrides = {
            "limit": run.limit,
            "max_iterations": run.max_iterations,
            "ground_truth_file": run.ground_truth_file,
            "output_dir": run.output_dir,
        }
        merged = _merge_defaults(cfg.defaults, overrides, run_overrides)
        spec = cfg.models[run.model]
        sampling, thinking, json_mode = _resolve_sampling(cfg.sampling, spec.sampling)
        resolved.append(
            _build_resolved(
                name=run.name,
                design=run.design,
                model_alias=run.model,
                model_map=_resolve_model_map(spec),
                provider=spec.provider,
                repetition=run.repetition,
                merged=merged,
                sampling=sampling,
                thinking=thinking,
                json_mode=json_mode,
            )
        )

    return resolved
