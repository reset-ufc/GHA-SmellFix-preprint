"""
cli.py — ``gha-smellfix``: run one design of GHA-SmellFix on one workflow.

The developer gives the inputs of the approach: the workflow, the target smell
and, optionally, the repository context (owner/repo, default branch, path of the
workflow in the repository). The command runs design D1 or D2 with one of the
paper's four models, through the same agents, prompts and sampling as the
experiment runs of :mod:`gha_agents.experiment`, and needs no ground truth:

    uv run gha-smellfix .github/workflows/ci.yml --smell MissingJobTimeout \\
        --design D2 --model haiku45 --repository owner/repo \\
        --default-branch main --file-path .github/workflows/ci.yml

It prints the verdict, the justifications and the diff of the candidate patch.
``--out`` writes the workflow the run returns. ``--json`` writes the result; its
fields carry the names of the experiment's result JSON, and ``patched_workflow``
holds the returned workflow itself.
"""

from __future__ import annotations

import difflib
import hashlib
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from gha_agents.agents import ValidSmells
from gha_agents.config import (
    Design,
    ExperimentConfig,
    ModelSpec,
    ResolvedRun,
    RunSpec,
    SamplingOverride,
    SamplingSpec,
    expand_runs,
)
from gha_agents.errors import WorkflowValidationError
from gha_agents.main import _build_result_metadata, _prompt_hashes_for, _run_standard
from gha_agents.run_context import RunContext, RunScope
from gha_agents.teams import D1Workflow, D2Workflow
from gha_agents.user_turn import repository_context_from_entry, validate_workflow_input
from gha_agents.utils import save_result

# The four models of the paper, with the provider and sampling of its runs. The
# experiment configs (configs/*.yaml) hold the same entries; a test keeps them equal.
PAPER_MODELS: dict[str, ModelSpec] = {
    "gpt54-nano": ModelSpec(
        id="gpt-5.4-nano",
        provider="openai",
        sampling=SamplingOverride(reasoning_effort="medium"),
    ),
    "haiku45": ModelSpec(id="claude-haiku-4-5", provider="anthropic"),
    "gemini35-flash": ModelSpec(id="gemini-3.5-flash", provider="vertex"),
    "deepseek-v4-flash": ModelSpec(id="deepseek/deepseek-v4-flash", provider="openrouter"),
}

# The sampling block of the experiment configs.
PAPER_SAMPLING = SamplingSpec(json_mode=True)

# The HTTP timeout of each model call in the experiment configs, in seconds.
DEFAULT_TIMEOUT = 180.0

Smell = Enum("Smell", {smell.name: smell.name for smell in ValidSmells}, type=str)
Model = Enum("Model", {alias: alias for alias in PAPER_MODELS}, type=str)

_DESIGN_CLASSES = {"D1": D1Workflow, "D2": D2Workflow}


def resolve_run(design: Design, model: str) -> ResolvedRun:
    """The run of ``design`` with ``model``, resolved as the experiment resolves its runs."""
    cfg = ExperimentConfig(
        sampling=PAPER_SAMPLING,
        models={model: PAPER_MODELS[model]},
        runs=[RunSpec(name=f"{design}-{model}", design=design, model=model)],
    )
    (spec,) = expand_runs(cfg)
    return spec


def run_smellfix(
    yaml_content: str,
    *,
    smell: str,
    design: Design,
    model: str,
    repository: str | None = None,
    default_branch: str | None = None,
    file_path: str | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Run ``design`` with ``model`` on one workflow for one target smell.

    Raises :class:`WorkflowValidationError` when ``yaml_content`` is not a GitHub
    Actions workflow. Returns the inputs, the settings of the run and the fields of
    the experiment's result JSON (``smell_detected``, ``patched_workflow``, ...).
    """
    validate_workflow_input(file_path or "workflow", yaml_content)
    target = ValidSmells[smell]
    spec = resolve_run(design, model)
    repository_context = repository_context_from_entry(
        {"repository": repository, "default_branch": default_branch, "file_path": file_path}
    )
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    with RunScope(RunContext.for_run(spec, request_timeout=timeout)):
        result, state = _run_standard(
            yaml_content, _DESIGN_CLASSES[design], target, repository_context
        )
        metadata = _build_result_metadata(
            agents=spec.agents, repetition=spec.repetition, token_usage=state.token_usage
        )
    return {
        "repository": repository,
        "default_branch": default_branch,
        "file_path": file_path,
        "input_yaml_sha256": hashlib.sha256(yaml_content.encode("utf-8")).hexdigest(),
        "target_smell": target.name,
        "design": design,
        "model_alias": model,
        "model_ids": metadata["model_ids"],
        "sampling_settings": metadata["sampling_settings"],
        "prompt_hashes": _prompt_hashes_for(design, target.name),
        "run_started_at": started_at,
        "run_duration_seconds": round(time.monotonic() - started, 3),
        **result,
    }


# ─── Report ───────────────────────────────────────────────────────────────────


def _line_range(evidence: dict[str, int]) -> str:
    start, end = evidence["start_line"], evidence["end_line"]
    return str(start) if start == end else f"{start}-{end}"


def _diff_line_color(line: str) -> str | None:
    if line.startswith(("+++", "---")):
        return None
    if line.startswith("+"):
        return typer.colors.GREEN
    if line.startswith("-"):
        return typer.colors.RED
    if line.startswith("@@"):
        return typer.colors.CYAN
    return None


def _print_report(workflow: Path, original: str, record: dict[str, Any]) -> None:
    model_ids = ", ".join(sorted(set(record["model_ids"].values())))
    typer.echo(
        f"{record['design']} · {record['model_alias']} ({model_ids}) · "
        f"{record['target_smell']} · {workflow}\n"
    )

    details = []
    if record["smell_detected"]:
        details.append(f"{record['count']} instance(s)")
    if record["evidence_ranges"]:
        details.append("lines " + ", ".join(map(_line_range, record["evidence_ranges"])))
    if record["detection_confidence"]:
        details.append(f"confidence {record['detection_confidence']}")
    verdict = "present" if record["smell_detected"] else "absent"
    typer.echo(f"Verdict: {verdict}" + (f" ({'; '.join(details)})" if details else ""))
    typer.echo(f"Detection: {record['detection_justification']}")
    if record["refactor_justification"]:
        confidence = record["refactor_confidence"]
        label = f"Refactoring (confidence {confidence})" if confidence else "Refactoring"
        typer.echo(f"{label}: {record['refactor_justification']}")

    total = record["token_usage"]["total"]
    typer.echo(
        f"Effort: {record['agent_call_count']} model call(s), "
        f"{total['input_tokens']:,} input and {total['output_tokens']:,} output tokens, "
        f"{record['run_duration_seconds']:.1f} s"
    )

    patched = record["patched_workflow"]
    if patched.splitlines() == original.splitlines():
        typer.echo("\nThe workflow comes back unchanged.")
        return
    if record["patch_valid"]:
        typer.echo("Patch: valid against the GitHub Actions schema\n")
    else:
        error = (record["patch_validation_error"] or "").partition("\n")[0]
        typer.echo(f"Patch: invalid against the GitHub Actions schema ({error})\n")
    diff = difflib.unified_diff(
        original.splitlines(),
        patched.splitlines(),
        fromfile=f"{workflow.name} (original)",
        tofile=f"{workflow.name} (patched)",
        lineterm="",
    )
    for line in diff:
        typer.echo(typer.style(line, fg=_diff_line_color(line)))


# ─── CLI ──────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="gha-smellfix",
    help="Run design D1 or D2 of GHA-SmellFix on one workflow for one target smell.",
    add_completion=False,
)

_ARG_WORKFLOW = typer.Argument(
    ..., exists=True, dir_okay=False, readable=True, help="The workflow YAML file."
)
_OPT_SMELL = typer.Option(
    ...,
    "--smell",
    metavar="SMELL",
    help="The target smell: one of the 19 of docs/smell-catalog.md, e.g. MissingJobTimeout.",
)
_OPT_DESIGN = typer.Option(..., "--design", help="D1 (joint) or D2 (decomposed).")
_OPT_MODEL = typer.Option(
    ..., "--model", help="The model, with the provider and sampling of the paper's runs."
)
_OPT_REPOSITORY = typer.Option(
    None, "--repository", help="Repository context: the repository, as owner/repo."
)
_OPT_DEFAULT_BRANCH = typer.Option(
    None, "--default-branch", help="Repository context: the default branch."
)
_OPT_FILE_PATH = typer.Option(
    None,
    "--file-path",
    help="Repository context: the workflow's path in the repository, "
    "e.g. .github/workflows/ci.yml.",
)
_OPT_OUT = typer.Option(
    None, "--out", dir_okay=False, help="Write the workflow the run returns to this file."
)
_OPT_JSON = typer.Option(None, "--json", dir_okay=False, help="Write the result to this JSON file.")
_OPT_TIMEOUT = typer.Option(
    DEFAULT_TIMEOUT, "--timeout", min=1.0, help="HTTP timeout of each model call, in seconds."
)


@app.command()
def run(
    workflow: Path = _ARG_WORKFLOW,
    smell: Smell = _OPT_SMELL,
    design: Design = _OPT_DESIGN,
    model: Model = _OPT_MODEL,
    repository: str | None = _OPT_REPOSITORY,
    default_branch: str | None = _OPT_DEFAULT_BRANCH,
    file_path: str | None = _OPT_FILE_PATH,
    out: Path | None = _OPT_OUT,
    json_out: Path | None = _OPT_JSON,
    timeout: float = _OPT_TIMEOUT,
) -> None:
    """Detect the target smell in WORKFLOW and, when it is present, propose a patch."""
    yaml_content = workflow.read_text(encoding="utf-8")
    try:
        record = run_smellfix(
            yaml_content,
            smell=smell.value,
            design=design,
            model=model.value,
            repository=repository,
            default_branch=default_branch,
            file_path=file_path,
            timeout=timeout,
        )
    except WorkflowValidationError as exc:
        typer.echo(f"✖  {workflow} is not a GitHub Actions workflow.", err=True)
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        message = str(exc).strip().partition("\n")[0] or "no message"
        typer.echo(f"✖  The run failed ({type(exc).__name__}): {message}", err=True)
        raise typer.Exit(code=1) from exc

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(record["patched_workflow"], encoding="utf-8")
    if json_out is not None:
        save_result(json_out, {"workflow": str(workflow), **record})
    _print_report(workflow, yaml_content, record)


if __name__ == "__main__":
    app()
