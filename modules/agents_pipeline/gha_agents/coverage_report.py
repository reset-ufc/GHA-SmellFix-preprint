"""coverage_report.py — per-cell coverage + failure breakdown for an experiment run.

Surfaces what the detection metrics silently drop. A failed instance writes no
result JSON (only an ``error.log`` entry), so F1/precision/recall are computed
only over the pairs that *succeeded* — failures vanish from the denominator
rather than counting as wrong. That hides three things this report makes visible:

  * uneven coverage across models (an unreliable model is graded on a smaller,
    possibly non-random subset → its F1 is not comparable to a reliable one);
  * which failure categories dominate (``rate_limit`` / ``transient`` are infra
    and re-runnable; ``structured_output`` is model reliability);
  * exactly which cells are incomplete and by how much.

It reads the experiment YAML (the matrix source of truth for designs × models ×
reps and the alias→id map), the ground_truth (expected pairs per cell), and the
on-disk output (``design_d<n>/results/*.json`` + ``design_d<n>/error.log``).

Policy-neutral: it only reports holes. Whether to re-run them, configure a
later retry, or accept them as recorded failures is a separate decision.

Usage::

    uv run python -m gha_agents.coverage_report \
        modules/agents_pipeline/configs/full.yaml

    # override the locations the config points at
    uv run python -m gha_agents.coverage_report <config.yaml> \
        --output-dir output-full --ground-truth data/12b_final_ground_truth.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated

import typer

from gha_agents.config import _scenario_subdir, load_config

app = typer.Typer(add_completion=False, help=__doc__)


def _scenario_dir(design: str) -> str:
    """``D1`` → ``design_d1``: the RunStore layout, from config._scenario_subdir."""
    return _scenario_subdir(design)


def expected_pairs(ground_truth_file: Path, limit: int | None) -> tuple[int, int]:
    """Return ``(pairs, entries)`` expected per (design, model, rep) cell.

    A pair is one (file, smell) unit: ``present_smells`` ∪ ``absent_smells`` per
    entry — the same enumeration the runner uses. ``limit`` truncates the entry
    list in order (``entries[:limit]``), matching the runner's behaviour.
    """
    data = json.loads(ground_truth_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise typer.BadParameter(f"Ground truth must be a JSON array: {ground_truth_file}")
    if limit is not None:
        data = data[:limit]
    pairs = sum(
        len(e.get("present_smells") or []) + len(e.get("absent_smells") or [])
        for e in data
    )
    return pairs, len(data)


def _alias_of_slug(path: Path) -> str:
    """Model alias is the final ``@``-segment of the result filename stem. A
    synthetic positive's stem ends with a trailing ``@synth`` tag, so strip that
    first to read the model from the same position as a real instance's stem."""
    stem = path.stem
    if stem.endswith("@synth"):
        stem = stem[: -len("@synth")]
    return stem.rsplit("@", 1)[-1]


def _count_results(scenario_dir: Path) -> Counter:
    """``{model_alias: succeeded_count}`` from ``results/*.json`` in one scenario."""
    counts: Counter = Counter()
    results = scenario_dir / "results"
    if results.is_dir():
        for path in results.glob("*.json"):
            counts[_alias_of_slug(path)] += 1
    return counts


def _done_keys(scenario_dir: Path) -> set[tuple[str, str, str]]:
    """``{(file_key, target_smell, model_alias)}`` for every result JSON present.

    The reconciliation key for matching an ``error.log`` entry to a pair that has
    since SUCCEEDED. ``error.log`` is append-only, so a pair fixed on a later
    resume keeps its old failure line; without this, the report would count
    phantom holes. Mirrors the fields ``error.log`` records (``file`` /
    ``target_smell`` / model), read from the authoritative result JSONs.
    """
    keys: set[tuple[str, str, str]] = set()
    results = scenario_dir / "results"
    if not results.is_dir():
        return keys
    for path in results.glob("*.json"):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(r, dict):
            keys.add(
                (r.get("file_key", ""), r.get("target_smell", ""), r.get("model_alias", ""))
            )
    return keys


def _count_failures(
    scenario_dir: Path,
    alias_by_id: dict[str, str],
    done_keys: set[tuple[str, str, str]] | None = None,
) -> dict[str, Counter]:
    """``{model_alias: Counter(category -> n)}`` of CURRENT holes in one scenario.

    Each ``error.log`` line is a JSON failure record (see ``utils.log_error``);
    the model is resolved from ``model_ids`` via the config's alias→id table (an
    unresolved id falls into the ``"?"`` bucket so nothing is silently dropped).

    Because ``error.log`` is append-only, a pair that failed then succeeded on a
    later resume keeps its stale line. We reconcile against ``done_keys``: a pair
    is counted only if it has NO result JSON now, and each pair is counted once
    (its most recent failure category), so the totals equal the real remaining
    holes — not the cumulative history. ``done_keys=None`` disables reconciliation.
    """
    log = scenario_dir / "error.log"
    if not log.is_file():
        return {}
    done = done_keys or set()
    # Dedup append-only history to the LATEST failure per (file, smell, model).
    latest: dict[tuple[str, str, str], tuple[str, str]] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue  # a valid-JSON but non-object line — diagnostics never crash
        mids = entry.get("model_ids") or {}
        model_id = next(iter(mids.values()), "") if isinstance(mids, dict) else ""
        alias = alias_by_id.get(model_id, "?")
        key = (entry.get("file", ""), entry.get("target_smell", ""), alias)
        latest[key] = (alias, entry.get("category", "unknown"))

    # A model id that isn't in the config map yields alias "?", so its full key
    # can never match a result key — reconcile those on (file, smell) alone so a
    # pair that succeeded under its real alias isn't reported as a phantom hole.
    done_fs = {(f, s) for (f, s, _a) in done}
    failures: dict[str, Counter] = defaultdict(Counter)
    for key, (alias, category) in latest.items():
        f, s, a = key
        resolved = (f, s) in done_fs if a == "?" else key in done
        if resolved:
            continue  # pair has since succeeded — stale append-only line
        failures[alias][category] += 1
    return failures


@app.command()
def main(
    config_path: Annotated[Path, typer.Argument(help="Experiment YAML (matrix source of truth).")],
    output_dir: Annotated[
        Path | None, typer.Option(help="Override the config's defaults.output_dir.")
    ] = None,
    ground_truth: Annotated[
        Path | None, typer.Option(help="Override the config's defaults.ground_truth_file.")
    ] = None,
) -> None:
    """Print per-cell coverage and a failure-category breakdown for a run."""
    cfg = load_config(config_path)
    out_dir = output_dir or cfg.defaults.output_dir
    ground_truth_file = ground_truth or cfg.defaults.ground_truth_file
    if not out_dir.is_dir():
        raise typer.BadParameter(f"Output dir not found: {out_dir}")
    if not ground_truth_file.is_file():
        raise typer.BadParameter(f"Ground truth not found: {ground_truth_file}")

    pairs, entries = expected_pairs(ground_truth_file, cfg.defaults.limit)
    designs = list(cfg.matrix.designs)
    models: list[str] = cfg.matrix.models
    reps: list[int] = cfg.matrix.repetitions or [1]
    per_cell = pairs * len(reps)
    alias_by_id = {spec.id: alias for alias, spec in cfg.models.items()}

    # Gather succeeded + failed per (design, model).
    succeeded: dict[str, Counter] = {}
    failed: dict[str, dict[str, Counter]] = {}
    for design in designs:
        sdir = out_dir / _scenario_dir(design)
        succeeded[design] = _count_results(sdir)
        # Reconcile error.log (append-only) against current results so resumed/
        # fixed pairs don't show as phantom holes.
        failed[design] = _count_failures(sdir, alias_by_id, _done_keys(sdir))

    typer.echo(f"\nthe study coverage report — {out_dir}")
    typer.echo(
        f"expected per cell: {pairs} pairs × {len(reps)} rep(s) = {per_cell} "
        f"(ground_truth entries: {entries}{', limit ' + str(cfg.defaults.limit) if cfg.defaults.limit else ''})"
    )

    # ── Coverage table ───────────────────────────────────────────────────────
    typer.echo("\nCOVERAGE  (succeeded / expected)")
    header = f"{'model':<20}" + "".join(f"{design:>16}" for design in designs)
    typer.echo(header)
    typer.echo("─" * len(header))
    grand_ok = grand_exp = 0
    incomplete: list[str] = []
    cat_totals: Counter = Counter()
    for model in models:
        cells = ""
        for design in designs:
            ok = succeeded[design].get(model, 0)
            grand_ok += ok
            grand_exp += per_cell
            pct = (100 * ok / per_cell) if per_cell else 0.0
            mark = "" if ok >= per_cell else "*"
            if ok < per_cell:
                incomplete.append(f"{design}/{model} ({ok}/{per_cell})")
            cells += f"{f'{ok}/{per_cell} {pct:.0f}%{mark}':>16}"
        typer.echo(f"{model:<20}{cells}")

    # ── Failure breakdown ────────────────────────────────────────────────────
    # Split reconciled failures: mapped (attributable to a matrix model, these are
    # the real current holes + their category) vs "?" (model id not in the config
    # map — typically old-schema/stale error.log lines that can't be reconciled).
    for cfg_map in failed.values():
        for alias, cats in cfg_map.items():
            if alias != "?":
                cat_totals.update(cats)
    orphan = sum(
        sum(c.values()) for cfg_map in failed.values() for a, c in cfg_map.items() if a == "?"
    )

    typer.echo("\nFAILURES  (current holes — error.log reconciled vs results, by category)")
    any_fail = False
    for design in designs:
        for model in [*models, "?"]:
            cats = failed[design].get(model)
            if not cats:
                continue
            any_fail = True
            label = model if model != "?" else "?(stale/unreconciled)"
            detail = "  ".join(f"{k}={v}" for k, v in sorted(cats.items()))
            typer.echo(f"  {design:>3}/{label:<22} total={sum(cats.values()):<4} {detail}")
    if not any_fail:
        typer.echo("  (none)")

    # ── Totals ───────────────────────────────────────────────────────────────
    # The authoritative remaining-hole count is expected − succeeded (from result
    # JSONs), NOT the error.log line count: error.log is append-only and may carry
    # stale/old-schema lines for pairs that have since succeeded.
    remaining = grand_exp - grand_ok
    cov = (100 * grand_ok / grand_exp) if grand_exp else 0.0
    typer.echo("\nTOTALS")
    typer.echo(f"  coverage: {grand_ok}/{grand_exp} succeeded  ({cov:.1f}%)")
    typer.echo(f"  remaining holes (authoritative, from results): {remaining}")
    if cat_totals:
        typer.echo("  holes by category: " + "  ".join(f"{k}={v}" for k, v in cat_totals.most_common()))
    if orphan:
        typer.echo(
            f"  + {orphan} unreconciled error.log line(s) — old-schema/stale "
            "(no smell or model id); not counted as holes"
        )
    if incomplete:
        typer.echo(f"  incomplete cells ({len(incomplete)}): " + "; ".join(incomplete))
    else:
        typer.echo("  all cells complete ✓")
    typer.echo("")


if __name__ == "__main__":
    app()
