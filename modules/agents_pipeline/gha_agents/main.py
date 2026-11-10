"""
main.py — the per-pair runner of the agents pipeline.

:func:`run_pipeline` runs one design on every (workflow, target_smell) pair of a
list of ground-truth entries; :mod:`gha_agents.experiment` calls it for each run of
a config. ``gha-smellfix`` (:mod:`gha_agents.cli`) runs one workflow through the
same :func:`_run_standard`.

Each ground_truth entry expands to one or more (workflow, target_smell) pairs — one per
present_smell (ground-truth-positive) and absent_smell (ground-truth-negative):
    results: <output>/results/<file_key_stem>@<design>@rep<k>@<smell_slug>@<model>.json
    patches: <output>/patched_workflows/<file_key_stem>@<design>@rep<k>@<smell_slug>@<model>.yml
Synthetic positives (file_key ending ##synth) get a trailing @synth tag on the slug,
so their results/patches are e.g. <stem>@<design>@rep<k>@<smell_slug>@<model>@synth.json —
distinct from the real instance and filterable in downstream metrics.
Failures are logged to: <output>/error.log
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from gha_agents.agents import ValidSmells
from gha_agents.credit import OutOfCreditError
from gha_agents.env import (
    get_all_model_ids,
    is_persist_traces_active,
)
from gha_agents.errors import (
    GhaConfigError,
    GhaError,
    GhaModelError,
    GhaRateLimitError,
    GhaStructuredOutputError,
    GhaTransientError,
    LLMOutputError,
    WorkflowValidationError,
    classify_agent_error,
)
from gha_agents.model_factory import ModelFactory
from gha_agents.progress import ProgressReporter
from gha_agents.prompts import PROMPT_HASHES
from gha_agents.record import RunRecord
from gha_agents.state import RoleTokenUsage, WorkflowState
from gha_agents.store import RunStore
from gha_agents.teams import D1Workflow, D2Workflow, SmellWorkflow
from gha_agents.user_turn import repository_context_from_entry, validate_workflow_input
from gha_agents.utils import (
    log_error,
    logger,
    persist_raw_response,
)

# The paper's design of a run, by its agent count: it names the slug, the ``design``
# field of the result JSON, the trace and the prompt-hash key.
_DESIGN = {1: "D1", 2: "D2"}


def _build_result_metadata(
    agents: int, repetition: int, token_usage: RoleTokenUsage
) -> dict[str, Any]:
    """Return experiment metadata fields shared by every result JSON.

    ``model_ids`` lists only the roles that actually ran (``calls > 0``), matching
    the pruned ``token_usage.by_role`` — emitting all five for a D1 run reads as if
    four idle agents had a model assigned. Cost is no longer baked here; analysts
    derive USD downstream from ``token_usage`` and ``model_ids``.
    """
    active_roles = {role for role, bucket in token_usage.by_role.items() if bucket.calls > 0}
    model_ids = {
        role: model_id
        for role, model_id in get_all_model_ids().items()
        if role in active_roles
    }
    return {
        "design": _DESIGN[agents],
        "model_ids": model_ids,
        "repetition": repetition,
        # Records the sampling actually applied (e.g. Anthropic thinking forces
        # temperature=1), not the pre-override request — see ModelFactory.
        "sampling_settings": ModelFactory().effective_sampling_for(),
    }


def _prompt_hashes_for(design: str, target_smell_pascal: str) -> dict[str, str]:
    """Filter PROMPT_HASHES to the (design, smell) pair active for this run."""
    hashes_by_role = PROMPT_HASHES.get(design, {})
    return {
        role: smell_hashes[target_smell_pascal]
        for role, smell_hashes in hashes_by_role.items()
        if target_smell_pascal in smell_hashes
    }


# ─── Ground truth loading ───────────────────────────────────────────────────────────


def load_ground_truth(file_path: Path) -> list[dict[str, Any]]:
    """Load an ground-truth-schema JSON array from disk.

    Raises GhaConfigError, which the experiment dispatcher handles like any
    other configuration error.
    """
    if not file_path.exists():
        raise GhaConfigError(f"Ground truth file not found: '{file_path}'")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GhaConfigError(f"Ground truth file is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise GhaConfigError("Ground truth file must contain a JSON array at the top level.")
    return data


# ─── Per-entry runners ────────────────────────────────────────────────────────


# Maps each typed the study exception to a stable, paper-friendly category label
# emitted into error.log under the ``category`` field. The label is also shown
# inline in the human-readable console message (`[STRUCTURED_OUTPUT]`, etc.)
# so failures are triaged at a glance without grepping the log.
_ERROR_CATEGORY_BY_TYPE: dict[type, str] = {
    GhaRateLimitError: "rate_limit",
    GhaTransientError: "transient",
    GhaModelError: "model_not_found",
    GhaConfigError: "auth",
    GhaStructuredOutputError: "structured_output",
    WorkflowValidationError: "workflow_validation",
}


def _error_category(typed: GhaError) -> str:
    if isinstance(typed, LLMOutputError):
        return "empty_response" if typed.cause is None else "malformed_json"
    return _ERROR_CATEGORY_BY_TYPE.get(type(typed), "unknown")


def _run_standard(
    yaml_content: str,
    workflow_cls: type[SmellWorkflow],
    target_smell: ValidSmells,
    repository_context: str | None = None,
) -> tuple[dict[str, Any], WorkflowState]:
    """Run a D1/D2 :class:`SmellWorkflow` for a single target smell.

    D1/D2 have no validation feedback loop. ``token_usage`` is read from the
    workflow's ``state``, which its ``run`` populates per-role as the agents
    run. The strategy returns a typed ``CombinedResult`` directly — the framework
    no longer round-trips it through a step-output string, so no JSON re-parse or
    step-skip sentinel handling is needed (the invoker's status guard raises a
    typed error on a failed call instead).
    """
    workflow = workflow_cls(target_smell)
    try:
        combined = workflow.run(yaml_content, repository_context)
    except Exception as exc:
        exc._study_state = workflow.state  # surface partial trace state to the caller
        raise
    state = workflow.state
    return RunRecord.from_standard(combined, state).dump(), state


def _run_for_smell(
    *,
    yaml_content: str,
    agents: int,
    target_smell: ValidSmells,
    cap: int,
    error_log: Path,
    file_key: str,
    repository_context: str | None = None,
) -> tuple[dict[str, Any], WorkflowState]:
    """Dispatch to the D1/D2 runner for a single (file, target_smell) pair."""
    if agents == 1:
        return _run_standard(yaml_content, D1Workflow, target_smell, repository_context)
    return _run_standard(yaml_content, D2Workflow, target_smell, repository_context)


# A synthetic positive's file_key (from the smell-synthesizer merge) carries this
# marker -- it must stay in sync with that project's records.SYNTH_KEY_MARKER. We
# strip it from the slug stem and re-add it as a trailing @synth tag (below), so a
# synthetic instance's results are cleanly named and distinguishable from the real
# one. Kept '@'-free so file_key still parses as owner@repo@filename via split('@').
_SYNTH_MARKER = "##synth"


def _is_synthetic(file_key: str) -> bool:
    return file_key.endswith(_SYNTH_MARKER)


def _stem_from_file_key(file_key: str) -> str:
    """Strip the synth marker (if any) then the trailing .yml/.yaml extension from a
    ground_truth file_key, keeping '@' separators.

    e.g. owner@repo@deploy.yml##synth -> owner@repo@deploy."""
    key = file_key[: -len(_SYNTH_MARKER)] if file_key.endswith(_SYNTH_MARKER) else file_key
    if key.endswith(".yaml"):
        return key[: -len(".yaml")]
    if key.endswith(".yml"):
        return key[: -len(".yml")]
    return key


def _build_slug(
    *,
    file_key: str,
    agents: int,
    repetition: int,
    target_smell: ValidSmells,
    model_alias: str | None,
) -> str:
    """Compose the per-entry output slug used for both patched YAML and result JSON.

    Without a model alias: ``{stem}@{D1|D2}@rep{k}@{smell_slug}``.
    Experiment runner appends ``@{model_alias}`` to disambiguate runs that share
    the same (file, design, rep, smell) tuple but use different models.
    """
    design = _DESIGN[agents]
    stem = _stem_from_file_key(file_key)
    smell_slug = target_smell.name.lower()
    slug = f"{stem}@{design}@rep{repetition}@{smell_slug}"
    if model_alias:
        slug = f"{slug}@{model_alias}"
    if _is_synthetic(file_key):
        slug = f"{slug}@synth"
    return slug


def _process_entry(
    *,
    entry: dict[str, Any],
    target_pascal: str,
    ground_truth_positive: bool,
    agents: int,
    repetition: int,
    max_iterations: int,
    error_log: Path,
    output_dir: Path,
    model_alias: str | None = None,
    resume: bool = False,
    echo_pairs: bool = True,
) -> dict[str, Any] | None:
    """Process one (file, target_smell) pair. Returns a summary row.

    ``echo_pairs`` gates the per-pair human echoes (start / ✔ / ✖ / ⏭). The live
    ``--progress`` dashboard sets it False so its bars own the terminal and fast
    resume-skips don't flood the redirected stdout and corrupt the live region;
    the same events are still logged to the run file and counted in the bars.
    """
    file_key = entry["file_key"]
    yaml_content = entry["yaml_content"]

    try:
        target_smell = ValidSmells[target_pascal]
    except KeyError:
        if echo_pairs:
            typer.echo(
                f"  ✖  {file_key}: unknown smell '{target_pascal}' (not in ValidSmells).",
                err=True,
            )
        log_error(error_log, file_key, f"Unknown smell: {target_pascal}")
        return {"file": file_key, "smells": "?", "solved": "?", "status": "INVALID"}

    try:
        validate_workflow_input(file_key, yaml_content)
    except WorkflowValidationError as exc:
        if echo_pairs:
            typer.echo(f"  ✖  {exc}", err=True)
        log_error(error_log, file_key, str(exc))
        return {"file": file_key, "smells": "?", "solved": "?", "status": "INVALID"}

    store = RunStore(output_dir)
    store.prepare()

    stem = _stem_from_file_key(file_key)
    slug = _build_slug(
        file_key=file_key,
        agents=agents,
        repetition=repetition,
        target_smell=target_smell,
        model_alias=model_alias,
    )
    patched_wf_path = store.patched_path(slug)
    result_json_path = store.result_path(slug)
    trace_required = is_persist_traces_active()

    if resume and store.exists(slug, trace_required=trace_required):
        logger.info("Resume: skipping '%s' — result already exists", result_json_path.name)
        if echo_pairs:
            typer.echo(f"  ⏭  {file_key} [{target_smell.value}] — skipped (resume)")
        return {
            "file": f"{file_key}#{target_smell.name.lower()}",
            "smells": "·",
            "solved": "·",
            "status": "SKIP",
        }
    if resume and result_json_path.exists():
        logger.info(
            "Resume: re-running '%s' — trace missing (persist_traces=true)",
            result_json_path.name,
        )

    run_started_at = datetime.now(UTC).isoformat(timespec="seconds")
    run_started_monotonic = time.monotonic()

    logger.info(
        "▶ %s [%s] design=%s rep=%d (yaml=%d chars)",
        file_key,
        target_smell.value,
        _DESIGN[agents],
        repetition,
        len(yaml_content),
    )

    persist_traces_active = trace_required
    try:
        result, state = _run_for_smell(
            yaml_content=yaml_content,
            agents=agents,
            target_smell=target_smell,
            cap=max_iterations,
            error_log=error_log,
            file_key=file_key,
            repository_context=repository_context_from_entry(entry),
        )
    except GhaConfigError:
        raise
    except OutOfCreditError:
        # Mid-run credit exhaustion aborts the whole experiment cleanly via the
        # dispatcher's handler — never a per-cell ERROR row that lets the run
        # keep burning calls against an empty balance.
        raise
    except Exception as exc:
        # Hybrid policy: a failed cell is always traced, even with traces off.
        partial_state = getattr(exc, "_study_state", None)
        store.save_trace(
            slug=slug,
            design=_DESIGN[agents],
            calls=partial_state.trace_calls if partial_state is not None else [],
            error=exc,
            persist_enabled=persist_traces_active,
        )
        typed = exc if isinstance(exc, GhaError) else classify_agent_error(exc)
        category = _error_category(typed)
        raw_response = getattr(typed, "raw_response", None)
        artifact_path: Path | None = None
        if raw_response:
            artifact_path = persist_raw_response(
                output_dir / "error_artifacts",
                slug=slug,
                raw_response=raw_response,
            )
        short = str(typed).splitlines()[0]
        if echo_pairs:
            typer.echo(
                f"  ✖  [{category.upper()}] {file_key} [{target_smell.value}] — {short}",
                err=True,
            )
        log_error(
            error_log,
            file_key,
            f"[{target_smell.value}] {typed}",
            category=category,
            exception_type=type(typed).__name__,
            target_smell=target_smell.value,
            model_ids=get_all_model_ids(),
            raw_response_path=str(artifact_path) if artifact_path else None,
        )
        return {"file": file_key, "smells": "?", "solved": "?", "status": "ERROR"}

    run_duration_seconds = round(time.monotonic() - run_started_monotonic, 3)
    logger.debug(
        "■ %s [%s] finished in %.2fs (smell_detected=%s, calls=%d)",
        file_key,
        target_smell.value,
        run_duration_seconds,
        result.get("smell_detected"),
        result.get("agent_call_count", 0),
    )

    patched_content = result.pop("patched_workflow")
    store.save_patched_workflow(slug, patched_content)

    # Persist any intermediate per-iteration patched YAML next to the winner so
    # downstream analysis can measure refactor quality progression across
    # iterations. D1/D2 carry no list (or an empty one) and never create the
    # iters/ subdirectory.
    patches_per_iteration = result.pop("patches_per_iteration", []) or []
    store.save_iter_patches(slug, patches_per_iteration)

    # Per-iteration loop entries, when present, are moved OUT of the result JSON
    # into their own self-describing files
    # (iteration_logs/{slug}@{detection|refactor}{n}.json) so the result stays
    # slim. D1/D2 (no loops) pop nothing and write no per-iteration files.
    detection_log = result.pop("detection_loop_iterations_log", None) or []
    refactor_log = result.pop("refactor_loop_iterations_log", None) or []

    # Hybrid policy: a successful cell is traced only when persist-traces is on.
    store.save_trace(
        slug=slug,
        design=_DESIGN[agents],
        calls=state.trace_calls,
        error=None,
        persist_enabled=persist_traces_active,
    )

    # Run-context metadata shared by the result JSON and every per-iteration file.
    result_header = {
        "workflow_id": stem,
        "file_key": file_key,
        "repository": entry.get("repository"),
        "file_path": entry.get("file_path"),
        # Fingerprint of the exact input workflow this result was computed from.
        # The resume slug is content-addressed by file_key+smell+config+rep+model
        # but NOT by yaml_content, so an in-place ground-truth edit that rewrites a
        # file's yaml under the same file_key is invisible to the skip rule. Baking
        # the input hash lets reconcile.py detect that drift and force a re-run
        # (see gha_agents.reconcile). sha256 over the UTF-8 bytes — the same
        # normalisation reconcile applies to the ground-truth entry.
        "input_yaml_sha256": hashlib.sha256(yaml_content.encode("utf-8")).hexdigest(),
        "target_smell": target_pascal,
        # Ground truth inherited from the pair's origin list (present → True,
        # absent → False); see run_pipeline's pair enumeration. Emitted under BOTH
        # keys: ``ground_truth`` is the historical name (asserted by older results
        # + test_ground_truth_schema); ``ground_truth_positive`` is the name every
        # downstream consumer reads — build_results_csv keys on it, so without
        # it the CSV confusion column was silently all-"NA". Keep them in lockstep.
        "ground_truth": ground_truth_positive,
        "ground_truth_positive": ground_truth_positive,
        "patched_workflow_path": str(patched_wf_path),
        "agent_count": agents,
        "model_alias": model_alias,
        **_build_result_metadata(
            agents=agents, repetition=repetition, token_usage=state.token_usage
        ),
        "prompt_hashes": _prompt_hashes_for(_DESIGN[agents], target_pascal),
        "run_started_at": run_started_at,
        "run_duration_seconds": run_duration_seconds,
    }
    store.save_iter_logs(
        slug, header=result_header, detection=detection_log, refactor=refactor_log
    )

    final_result = {**result_header, **result}
    store.save_result(slug, final_result)

    # D1/D2 run single-pass — there is no feedback loop to exhaust, so a written
    # cell is always OK.
    status = "OK"
    present_str = "1" if final_result.get("smell_detected") else "0"

    if echo_pairs:
        typer.echo(
            f"  ✔  {file_key} [{target_smell.value}] — "
            f"present={present_str}, status={status}"
        )
    logger.info("'%s' [%s] → %s", file_key, target_smell.value, status)

    return {
        "file": f"{file_key}#{target_smell.name.lower()}",
        "smells": present_str,
        "solved": "—",
        "status": status,
    }


# ─── Pipeline runner (reusable by experiment.py) ──────────────────────────────


def enumerate_pairs(
    entries: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str, bool]]:
    """Expand ground-truth entries into the evaluated (entry, smell, polarity) pairs.

    The evaluation set for a file is its curated present ∪ absent smells:
    ``present_smells`` → ground-truth-positive pairs, ``absent_smells`` → negative.
    A smell in neither list is not evaluated for that file (never the full
    catalog). The polarity assigned here IS the pair's ``ground_truth_positive``
    ground truth downstream — single source of truth, no re-derivation. A pair
    dropped from the ground truth simply stops appearing here, so resume never
    re-enumerates it (its stale result becomes a reconcile *orphan*, never a
    re-run); see :mod:`gha_agents.reconcile`.
    """
    pairs: list[tuple[dict[str, Any], str, bool]] = []
    for entry in entries:
        for smell in entry.get("present_smells", []):
            pairs.append((entry, smell["pascal"], True))
        for smell in entry.get("absent_smells", []):
            pairs.append((entry, smell["pascal"], False))
    return pairs


def run_pipeline(
    *,
    entries: list[dict[str, Any]],
    agents: int,
    repetition: int,
    max_iterations: int,
    output_dir: Path,
    error_log: Path,
    model_alias: str | None = None,
    label: str | None = None,
    resume: bool = False,
    reporter: ProgressReporter | None = None,
) -> list[dict[str, Any]]:
    """Run the pipeline over a list of ground_truth entries and return summary rows.

    The experiment dispatcher calls this. The caller is responsible for logging
    setup and for loading/truncating the ground_truth beforehand. ``model_alias`` is appended to output slugs so parallel
    runs with different models don't overwrite each other.
    """
    summary_rows: list[dict] = []
    prefix = f"[{label}] " if label else ""

    pairs = enumerate_pairs(entries)

    design = _DESIGN[agents]
    # Under the live dashboard the bars are the UI: silence the per-pair human
    # echoes so fast resume-skips don't flood the redirected stdout. Events still
    # log to the run file and advance the bars.
    echo_pairs = reporter is None
    for idx, (entry, target_pascal, ground_truth_positive) in enumerate(pairs, start=1):
        file_key = entry.get("file_key", f"<entry#{idx}>")
        if echo_pairs:
            typer.echo(f"  {prefix}[{idx}/{len(pairs)}] {file_key} [{target_pascal}] …")
        logger.info(
            "%sProcessing pair %d/%d '%s' [%s]",
            prefix,
            idx,
            len(pairs),
            file_key,
            target_pascal,
        )

        # Last-line containment: _process_entry guards the agent call, but its
        # setup (mkdir/stat) and result-save tail run outside that guard, and the
        # failure handler itself writes to disk. On a 1900-pair batch a stray
        # OSError (disk full, permissions) or unexpected bug there must fail ONE
        # pair, never abort the whole run. GhaConfigError / OutOfCreditError
        # stay fatal-by-design (halt the experiment; resume continues after a fix
        # or top-up).
        try:
            row = _process_entry(
                entry=entry,
                target_pascal=target_pascal,
                ground_truth_positive=ground_truth_positive,
                agents=agents,
                repetition=repetition,
                max_iterations=max_iterations,
                error_log=error_log,
                output_dir=output_dir,
                model_alias=model_alias,
                resume=resume,
                echo_pairs=echo_pairs,
            )
        except (GhaConfigError, OutOfCreditError):
            raise
        except Exception as exc:
            short = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            logger.exception(
                "Unhandled error processing pair '%s' [%s]", file_key, target_pascal
            )
            if echo_pairs:
                typer.echo(
                    f"  ✖  [UNHANDLED] {file_key} [{target_pascal}] — {short}", err=True
                )
            log_error(
                error_log,
                file_key,
                f"[{target_pascal}] unhandled: {exc}",
                category="unhandled",
                exception_type=type(exc).__name__,
                target_smell=target_pascal,
            )
            row = {"file": file_key, "smells": "?", "solved": "?", "status": "ERROR"}

        if row is not None:
            summary_rows.append(row)
        if reporter is not None:
            reporter.pair_done(design, model_alias or "?", row or {})

    return summary_rows
