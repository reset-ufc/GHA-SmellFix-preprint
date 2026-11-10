"""
experiment.py — Experiment orchestrator driven by a YAML config.

Usage:
    uv run python -m gha_agents.experiment config.yaml [--dry-run]
                                                          [--max-workers N]
                                                          [--verbose]

Loads an :class:`ExperimentConfig`, expands it into :class:`ResolvedRun`
specs (matrix cartesian product + named runs), and dispatches each run
through the shared pipeline in :mod:`gha_agents.main`. Parallelism is
controlled by ``execution.max_workers``; per-run model overrides are applied
via a contextvar so concurrent threads don't race on the global env.

Resume semantics: when ``execution.resume`` is true, entries whose result
JSON already exists on disk are skipped (per-entry skip; no whole-run skip).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

from gha_agents.config import ResolvedRun, expand_runs, load_config
from gha_agents.credit import (
    OutOfCreditError,
    is_out_of_credit_error,
    preflight_credit_check,
)
from gha_agents.errors import GhaConfigError
from gha_agents.main import load_ground_truth, run_pipeline
from gha_agents.preflight import PreflightError, assert_models_ready, format_report
from gha_agents.progress import (
    NullReporter,
    ProgressReporter,
    RichBarsReporter,
    count_pairs,
)
from gha_agents.reconcile import (
    GTIndex,
    ReconcilePlan,
    apply_plan,
    build_gt_index,
    build_plan,
)
from gha_agents.run_context import RunContext, RunScope
from gha_agents.utils import logger, print_summary, setup_logging

app = typer.Typer(
    name="study-experiment",
    help="Run the study experiments declaratively from a YAML config.",
    add_completion=False,
)

_ARG_CONFIG = typer.Argument(
    ...,
    exists=True,
    dir_okay=False,
    readable=True,
    help="Path to the experiment YAML config.",
)
_OPT_DRY_RUN = typer.Option(
    False,
    "--dry-run",
    help="Expand runs and print them, do not execute.",
)
_OPT_MAX_WORKERS = typer.Option(
    None,
    "--max-workers",
    min=1,
    help="Override execution.max_workers from the config.",
)
_OPT_VERBOSE = typer.Option(
    False,
    "--verbose",
    "-V",
    help="Show HTTP/telemetry logs on the console.",
)
_OPT_TIMEOUT = typer.Option(
    None,
    "--timeout",
    min=1.0,
    help=(
        "Per-LLM-call HTTP timeout in seconds. Overrides execution.request_timeout from the config."
    ),
)
_OPT_PROGRESS = typer.Option(
    False,
    "--progress/--no-progress",
    help=(
        "Show a live grouped-bar dashboard (overall + per-config + per-model bars "
        "+ counters). Routes logs to the run file only while it owns the terminal."
    ),
)
_OPT_SKIP_PREFLIGHT = typer.Option(
    False,
    "--skip-preflight",
    help=(
        "DANGER: skip the per-model pre-flight gate (key/thinking/temperature). "
        "Models start UNVERIFIED. The credit check still runs. Use only when you "
        "are certain the fleet is good."
    ),
)
_OPT_RECONCILE = typer.Option(
    False,
    "--reconcile",
    help=(
        "On ground-truth drift, auto-fix instead of aborting: quarantine orphaned "
        "results and supersede polarity-flipped / yaml-edited cells (in-matrix "
        "models only), then resume so exactly those cells re-run. Without it, drift "
        "aborts with the reconcile command to run."
    ),
)


# ─── Per-model credit circuit breaker ─────────────────────────────────────────


class _ModelCreditTracker:
    """Thread-safe set of model aliases that have hit out-of-credit mid-run.

    The matrix dispatches one cell per (design, model). When a model's
    provider runs out of credit, every remaining cell for that model would fail
    identically — so we mark the model here, skip its remaining cells, and let
    the other models keep running. When every planned model is exhausted the run
    has nothing left to do and stops cleanly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._exhausted: set[str] = set()

    def mark(self, model: str) -> None:
        with self._lock:
            self._exhausted.add(model)

    def is_exhausted(self, model: str) -> bool:
        with self._lock:
            return model in self._exhausted

    @property
    def exhausted(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._exhausted)


def _contain_out_of_credit(
    exc: OutOfCreditError,
    spec: ResolvedRun,
    credit_tracker: _ModelCreditTracker | None,
) -> None:
    """Mark the spec's model exhausted and report it — or re-raise if uncontainable.

    With no tracker (or no model alias to key on) out-of-credit keeps its old
    fast-fail semantics and propagates. Otherwise it is contained: the model is
    flagged so its remaining cells skip, and the other models keep running.
    """
    if credit_tracker is None or not spec.model_alias:
        raise exc
    credit_tracker.mark(spec.model_alias)
    logger.error(
        "Model '%s' out of credit — skipping its remaining cells (%s)",
        spec.model_alias,
        exc,
    )
    typer.echo(
        typer.style(
            f"  ⚠  model '{spec.model_alias}' out of credit — skipping its remaining cells",
            fg=typer.colors.YELLOW,
        ),
        err=True,
    )


# ─── Single-run runner ────────────────────────────────────────────────────────


def run_one_experiment(
    spec: ResolvedRun,
    *,
    resume: bool,
    request_timeout: float | None = None,
    persist_traces_flag: bool = False,
    reporter: ProgressReporter | None = None,
    credit_tracker: _ModelCreditTracker | None = None,
) -> list[dict]:
    """Execute a single ResolvedRun.

    Loads the ground_truth, truncates to ``spec.limit``, applies ``spec.model_map``,
    ``spec.sampling``, ``spec.thinking`` and ``spec.json_mode`` as contextvar
    overrides for the duration of the call, and delegates to
    :func:`gha_agents.main.run_pipeline`.

    If ``credit_tracker`` already flags this spec's model as out of credit, the
    run is skipped before any work — a sibling cell on the same model already hit
    an exhausted balance and the rest would fail identically.
    """
    if (
        credit_tracker is not None
        and spec.model_alias
        and credit_tracker.is_exhausted(spec.model_alias)
    ):
        logger.warning(
            "Skipping run '%s' — model '%s' is out of credit", spec.name, spec.model_alias
        )
        return []

    entries = load_ground_truth(spec.ground_truth_file)
    if spec.limit is not None:
        entries = entries[: spec.limit]
    if not entries:
        logger.warning("Run '%s': ground_truth has no entries, skipping", spec.name)
        return []

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    error_log = spec.output_dir / "error.log"

    logger.info(
        "Run '%s' start — %s, %d entries, model=%s, rep=%d",
        spec.name,
        spec.design,
        len(entries),
        spec.model_alias,
        spec.repetition,
    )

    run_context = RunContext.for_run(
        spec, request_timeout=request_timeout, persist_traces=persist_traces_flag
    )
    with RunScope(run_context):
        return run_pipeline(
            entries=entries,
            agents=spec.agents,
            repetition=spec.repetition,
            max_iterations=spec.max_iterations,
            output_dir=spec.output_dir,
            error_log=error_log,
            model_alias=spec.model_alias,
            label=spec.name,
            resume=resume,
            reporter=reporter,
        )


# ─── Dispatcher ───────────────────────────────────────────────────────────────


def _run_sequential(
    runs: list[ResolvedRun],
    *,
    resume: bool,
    stop_on_error: bool,
    request_timeout: float | None = None,
    persist_traces_flag: bool = False,
    reporter: ProgressReporter | None = None,
    credit_tracker: _ModelCreditTracker | None = None,
) -> list[dict]:
    all_rows: list[dict] = []
    for spec in runs:
        try:
            rows = run_one_experiment(
                spec,
                resume=resume,
                request_timeout=request_timeout,
                persist_traces_flag=persist_traces_flag,
                reporter=reporter,
                credit_tracker=credit_tracker,
            )
            all_rows.extend(rows)
        except GhaConfigError:
            raise
        except OutOfCreditError as exc:
            _contain_out_of_credit(exc, spec, credit_tracker)
        except Exception as exc:
            logger.exception("Run '%s' failed", spec.name)
            typer.echo(f"  ✖  run '{spec.name}' failed: {exc}", err=True)
            if stop_on_error:
                raise
    return all_rows


def _run_parallel(
    runs: list[ResolvedRun],
    *,
    resume: bool,
    stop_on_error: bool,
    max_workers: int,
    request_timeout: float | None = None,
    persist_traces_flag: bool = False,
    reporter: ProgressReporter | None = None,
    credit_tracker: _ModelCreditTracker | None = None,
) -> list[dict]:
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_one_experiment,
                s,
                resume=resume,
                request_timeout=request_timeout,
                persist_traces_flag=persist_traces_flag,
                reporter=reporter,
                credit_tracker=credit_tracker,
            ): s
            for s in runs
        }
        try:
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    rows = fut.result()
                    all_rows.extend(rows)
                except GhaConfigError:
                    raise
                except OutOfCreditError as exc:
                    _contain_out_of_credit(exc, spec, credit_tracker)
                except Exception as exc:
                    logger.exception("Run '%s' failed", spec.name)
                    typer.echo(f"  ✖  run '{spec.name}' failed: {exc}", err=True)
                    if stop_on_error:
                        raise
        except Exception:
            for f in futures:
                f.cancel()
            raise
    return all_rows


def dispatch(
    runs: list[ResolvedRun],
    *,
    max_workers: int,
    resume: bool,
    stop_on_error: bool,
    request_timeout: float | None = None,
    persist_traces_flag: bool = False,
    reporter: ProgressReporter | None = None,
    credit_tracker: _ModelCreditTracker | None = None,
) -> list[dict]:
    """Execute all expanded runs — sequentially or in parallel — and return summary rows."""
    if max_workers <= 1 or len(runs) <= 1:
        return _run_sequential(
            runs,
            resume=resume,
            stop_on_error=stop_on_error,
            request_timeout=request_timeout,
            persist_traces_flag=persist_traces_flag,
            reporter=reporter,
            credit_tracker=credit_tracker,
        )
    return _run_parallel(
        runs,
        resume=resume,
        stop_on_error=stop_on_error,
        max_workers=max_workers,
        request_timeout=request_timeout,
        persist_traces_flag=persist_traces_flag,
        reporter=reporter,
        credit_tracker=credit_tracker,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _describe_run(spec: ResolvedRun) -> str:
    limit = "all" if spec.limit is None else str(spec.limit)
    return (
        f"  {spec.name}: {spec.design} | model={spec.model_alias} | "
        f"rep={spec.repetition} | limit={limit} | max_iter={spec.max_iterations}"
    )


def _run_preflight_or_exit(runs: list[ResolvedRun], *, skip_models: bool = False) -> None:
    """Pre-flight gate: credit balance + per-model accessibility/thinking/temperature.

    Exits cleanly with code 2 on insufficient credit or any model-gate failure, so
    no cell starts against a bad fleet. ``skip_models`` bypasses ONLY the per-model
    probes (the credit check always runs); the caller is responsible for emitting
    the untested-gate warning first.
    """
    # The balance check is OpenRouter-specific (it queries OpenRouter's
    # /auth/key endpoint). Run it ONLY when the run set actually dispatches
    # through OpenRouter — otherwise an openai/anthropic/gemini/vertex-only
    # experiment would be gated by an unrelated OPENROUTER_API_KEY's balance, or
    # crash outright if that key is stale (raise_for_status → uncaught HTTPError).
    if any(run.provider == "openrouter" for run in runs):
        try:
            preflight_credit_check()
        except OutOfCreditError as exc:
            typer.echo(
                f"\n✖  OpenRouter credit exhausted before start: {exc}\n"
                "   Top up at https://openrouter.ai/credits and re-run.",
                err=True,
            )
            raise typer.Exit(code=2) from exc

    if skip_models:
        return
    try:
        checks = assert_models_ready(runs)
    except PreflightError as exc:
        typer.echo(
            f"\n✖  Pre-flight gate failed — run aborted, no cells started:\n\n{exc}\n\n"
            "   Fix the flagged models (key, reasoning/thinking, or temperature) and "
            "re-run.\n   Override at your own risk with --skip-preflight.",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    typer.echo(format_report(checks) + "\n")


def _report_credit_exhaustion(
    credit_tracker: _ModelCreditTracker, runs: list[ResolvedRun]
) -> None:
    """Echo a credit-exhaustion notice; exit 2 if EVERY planned model ran out.

    No-op when nothing ran out. When some models are exhausted but others
    completed, this is a warning (re-run after topping up to fill the gaps). When
    every planned model is exhausted the run has nothing left to do, so it stops
    with a non-zero exit so callers/CI see the failure.
    """
    exhausted = credit_tracker.exhausted
    if not exhausted:
        return
    planned = {s.model_alias for s in runs if s.model_alias}
    listed = ", ".join(sorted(exhausted))
    if planned and exhausted >= planned:
        typer.echo(
            typer.style(
                f"\n✖  All models out of credit ({listed}) — run stopped. "
                "Top up and re-run; resume skips already-completed pairs.",
                fg=typer.colors.RED,
                bold=True,
            ),
            err=True,
        )
        raise typer.Exit(code=2)
    typer.echo(
        typer.style(
            f"\n⚠  {len(exhausted)} model(s) out of credit, remaining cells skipped: "
            f"{listed}. Other models completed; top up and re-run to fill the gaps.",
            fg=typer.colors.YELLOW,
        ),
        err=True,
    )


# ─── Ground-truth drift gate ──────────────────────────────────────────────────


def _count_legacy_blind_spots(plan: ReconcilePlan) -> int:
    """Kept results with no input_yaml_sha256 — yaml edits on these can't be seen."""
    return sum(
        1
        for cell in plan.cells
        if cell.verdict.kind == "keep" and not cell.result.get("input_yaml_sha256")
    )


def reconcile_gate(runs: list[ResolvedRun], *, resume: bool, apply: bool) -> None:
    """Preflight gate: refuse to resume into results that pre-date a ground-truth edit.

    Resume is content-addressed by slug, so a ground-truth change that removed a
    pair (orphan), flipped its polarity, or rewrote a file's yaml under the same
    file_key leaves the kept results silently stale (see
    :mod:`gha_agents.reconcile`). For every model in this run we scan its
    on-disk results against the configured ground truth:

    * **cold start** (no results on disk at all) → return; nothing to reconcile.
    * **no drift** → return, after noting how many purely-additive cells resume
      will run fresh.
    * **drift + no --reconcile** → abort exit 2, naming each drifted (model, dir)
      with its orphan/flip/yaml counts and the reconcile command.
    * **drift + --reconcile** → quarantine orphans and supersede stale cells in
      place, then return so the run proceeds (resume re-runs exactly those).

    Only fires when ``resume`` is true (a fresh overwrite keeps nothing) and only
    inspects models in this run set (a gemma-only run is never blocked by
    deepseek's drift). yaml-edit detection needs the ``input_yaml_sha256`` baked
    into results going forward; older results can't self-report it, so we warn
    with a count rather than silently miss it.
    """
    if not resume:
        return

    gt_cache: dict[Path, GTIndex] = {}
    drifted: list[tuple[ResolvedRun, ReconcilePlan]] = []
    cells_scanned = 0
    added_total = 0
    legacy_total = 0

    for spec in runs:
        if not spec.output_dir.exists():
            continue
        gt = gt_cache.get(spec.ground_truth_file)
        if gt is None:
            gt = build_gt_index(load_ground_truth(spec.ground_truth_file))
            gt_cache[spec.ground_truth_file] = gt
        plan = build_plan(spec.output_dir, gt, model=spec.model_alias)
        cells_scanned += len(plan.cells)
        added_total += len(plan.added_pairs)
        legacy_total += _count_legacy_blind_spots(plan)
        if plan.needs_action:
            drifted.append((spec, plan))

    # Cold start (deleted/empty dirs, e.g. the fresh gemma re-run): nothing on
    # disk to keep, so resume just runs everything — no gate, no noisy "added".
    if cells_scanned == 0:
        return

    if added_total:
        typer.echo(
            typer.style(
                f"  ℹ  ground truth adds {added_total} result-cell(s) not yet on "
                "disk — resume will run them fresh.",
                fg=typer.colors.CYAN,
            )
        )
    if legacy_total:
        typer.echo(
            typer.style(
                f"  ⚠  {legacy_total} kept result(s) predate input_yaml_sha256 — an "
                "in-place yaml edit on those is NOT detectable here.",
                fg=typer.colors.YELLOW,
            )
        )

    if not drifted:
        return

    if not apply:
        lines = [
            "\n✖  Ground-truth drift — on-disk results don't match the configured "
            "ground truth:\n"
        ]
        for spec, plan in drifted:
            counts = plan.counts
            lines.append(
                f"   model '{spec.model_alias}' @ {spec.output_dir}: "
                f"{counts.get('orphan', 0)} orphan, {counts.get('rerun_flip', 0)} flip, "
                f"{counts.get('rerun_yaml', 0)} yaml"
            )
        lines.append(
            "\n   Re-run with --reconcile to quarantine orphans + supersede stale "
            "cells, then resume.\n   Or inspect first, e.g.:"
        )
        for model in sorted({spec.model_alias for spec, _ in drifted}):
            spec = next(s for s, _ in drifted if s.model_alias == model)
            lines.append(
                f"     uv run python -m gha_agents.reconcile "
                f"{spec.ground_truth_file} {spec.output_dir} --model {model}"
            )
        typer.echo("\n".join(lines), err=True)
        raise typer.Exit(code=2)

    for spec, plan in drifted:
        moved = apply_plan(plan)
        typer.echo(
            typer.style(
                f"  ✔ reconciled '{spec.model_alias}' @ {spec.output_dir} — "
                f"{moved.get('_orphaned', 0)} → _orphaned/, "
                f"{moved.get('_superseded', 0)} → _superseded/.",
                fg=typer.colors.GREEN,
            )
        )
    typer.echo(
        typer.style(
            "  → resume will now re-run the superseded + added cells, skip the rest.",
            fg=typer.colors.GREEN,
        )
    )


def run_experiment(
    config_path: Path,
    *,
    dry_run: bool = False,
    max_workers: int | None = None,
    verbose: bool = False,
    request_timeout: float | None = None,
    progress: bool = False,
    skip_preflight: bool = False,
    reconcile: bool = False,
) -> None:
    """Execute a YAML experiment — works for single-run smokes and N-run matrices.

    Loads ``config_path``, expands runs, runs the ground-truth drift gate + the
    preflight credit check (unless ``dry_run``), dispatches sequentially or in
    parallel, and prints a summary. Exits with code 2 on config errors, on
    ground-truth drift (unless ``reconcile`` auto-fixes it), or credit
    exhaustion.
    """
    try:
        cfg = load_config(config_path)
    except GhaConfigError as exc:
        typer.echo(f"✖  {exc}", err=True)
        raise typer.Exit(code=2) from exc

    runs = expand_runs(cfg)
    workers = max_workers if max_workers is not None else cfg.execution.max_workers
    timeout = request_timeout if request_timeout is not None else cfg.execution.request_timeout

    if dry_run:
        typer.echo(f"\n🧪 Expanded {len(runs)} run(s) (max_workers={workers}):\n")
        for spec in runs:
            typer.echo(_describe_run(spec))
        typer.echo("")
        return

    reporter: ProgressReporter = NullReporter()
    if progress and runs:
        # All matrix cells share the same ground_truth + limit, so one cell's pair count
        # sizes every bar. Load once and pre-size from the run plan.
        pair_entries = load_ground_truth(runs[0].ground_truth_file)
        if runs[0].limit is not None:
            pair_entries = pair_entries[: runs[0].limit]
        reporter = RichBarsReporter(
            [(s.design, s.model_alias or "?") for s in runs],
            count_pairs(pair_entries),
        )

    # Under the live dashboard, logs go to the run file only (the bars own stdout).
    setup_logging(
        verbose=verbose, log_dir=cfg.defaults.output_dir, console=not progress
    )

    timeout_label = f"{timeout:g}s" if timeout is not None else "off"
    typer.echo(
        f"\n🧪 the study Experiment — {len(runs)} run(s) | "
        f"max_workers={workers} | resume={cfg.execution.resume} | "
        f"timeout={timeout_label}\n"
    )
    logger.info(
        "Experiment start — %d runs, max_workers=%d, resume=%s, stop_on_error=%s, "
        "request_timeout=%s",
        len(runs),
        workers,
        cfg.execution.resume,
        cfg.execution.stop_on_error,
        timeout_label,
    )

    # Ground-truth drift gate — runs before the credit/model probes so a stale
    # output dir aborts (or self-heals via --reconcile) at zero network cost.
    reconcile_gate(runs, resume=cfg.execution.resume, apply=reconcile)

    if skip_preflight:
        typer.echo(
            "\n" + "═" * 72 + "\n"
            "⚠️  PRE-FLIGHT GATE SKIPPED  (--skip-preflight)\n"
            "    Models were NOT verified — key validity, thinking/reasoning, and\n"
            "    temperature are UNCHECKED. A misconfigured or unreachable model can\n"
            "    burn an entire run before it surfaces. Proceed only if you are\n"
            "    certain the fleet is good.\n" + "═" * 72 + "\n",
            err=True,
        )
    _run_preflight_or_exit(runs, skip_models=skip_preflight)

    credit_tracker = _ModelCreditTracker()
    try:
        with reporter:
            all_rows = dispatch(
                runs,
                max_workers=workers,
                resume=cfg.execution.resume,
                stop_on_error=cfg.execution.stop_on_error,
                request_timeout=timeout,
                persist_traces_flag=cfg.execution.persist_traces,
                reporter=reporter,
                credit_tracker=credit_tracker,
            )
    except GhaConfigError as exc:
        typer.echo(f"\n✖  Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except OutOfCreditError as exc:
        typer.echo(
            f"\n✖  OpenRouter credit exhausted: {exc}\n"
            "   Top up at https://openrouter.ai/credits and re-run — "
            "execution.resume will skip already-completed entries.",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        if is_out_of_credit_error(exc):
            typer.echo(
                f"\n✖  OpenRouter credit exhausted ({exc}).\n"
                "   Top up at https://openrouter.ai/credits and re-run — "
                "execution.resume will skip already-completed entries.",
                err=True,
            )
            raise typer.Exit(code=2) from exc
        raise

    print_summary(
        all_rows,
        verbose=verbose,
        retry_command=f"uv run python -m gha_agents.experiment {config_path}",
        resume_enabled=cfg.execution.resume,
    )
    _report_credit_exhaustion(credit_tracker, runs)


@app.command()
def run(
    config: Path = _ARG_CONFIG,
    dry_run: bool = _OPT_DRY_RUN,
    max_workers: int | None = _OPT_MAX_WORKERS,
    verbose: bool = _OPT_VERBOSE,
    timeout: float | None = _OPT_TIMEOUT,
    progress: bool = _OPT_PROGRESS,
    skip_preflight: bool = _OPT_SKIP_PREFLIGHT,
    reconcile: bool = _OPT_RECONCILE,
) -> None:
    """Run an experiment from a YAML config."""
    run_experiment(
        config,
        dry_run=dry_run,
        max_workers=max_workers,
        verbose=verbose,
        request_timeout=timeout,
        progress=progress,
        skip_preflight=skip_preflight,
        reconcile=reconcile,
    )


if __name__ == "__main__":
    app()
