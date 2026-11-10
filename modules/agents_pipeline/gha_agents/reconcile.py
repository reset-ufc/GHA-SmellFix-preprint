"""reconcile.py — make resume coherent after an in-place ground-truth edit.

Resume (see :mod:`gha_agents.experiment`) is *content-addressed by slug*:
``{file_key_stem}@{config}@rep{k}@{smell}[@{model}]``. The skip rule keys on
``file_key + smell + config + rep + model`` — **not** on the pair's polarity nor
on the workflow's ``yaml_content``. So when the ground truth is edited under an
output directory that already holds results, three drifts slip past the slug-skip
rule and leave the kept results silently stale:

* **orphan** — a pair removed from the ground truth keeps its result on disk; it
  is no longer enumerated (never re-run) yet still globbed by
  ``scripts/build_results_csv.py`` (which does not cross-reference the GT).
* **polarity flip** — a (file_key, smell) whose polarity moved present↔absent
  keeps a result whose baked ``ground_truth`` is now wrong; the slug is
  unchanged, so resume skips it and the stale label survives into the metrics.
* **yaml edit** — a file whose ``yaml_content`` was rewritten under the same
  file_key keeps a result computed from the *old* workflow; detectable only via
  the ``input_yaml_sha256`` baked into results since that field was added (older
  results cannot self-report this, so it is reported as a soft skip).

This tool diffs the NEW ground truth against the result JSONs already on disk and
classifies every result **keep / rerun / orphan**, then (with ``--apply``)
quarantines orphans and supersedes rerun results — moving, never deleting, so the
operation is reversible — so a subsequent ``resume`` run redoes *exactly* the
drifted cells and nothing else. Added pairs need no action: resume runs them
fresh because their slug has no result yet.

    uv run python -m gha_agents.reconcile GROUND_TRUTH.json OUTPUT_ROOT \
        [--model deepseek-v4-flash] [--apply]

Dry-run by default — it only reports. Pass ``--apply`` to move files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import typer

from gha_agents.store import RunStore

# Buckets the --apply step moves drifted cells into, under the output root.
_ORPHAN_DIR = "_orphaned"
_SUPERSEDED_DIR = "_superseded"
_QUARANTINE_DIRS = frozenset({_ORPHAN_DIR, _SUPERSEDED_DIR})

VerdictKind = Literal["keep", "rerun_flip", "rerun_yaml", "orphan", "unreadable"]
_RERUN_KINDS = frozenset({"rerun_flip", "rerun_yaml"})


# ─── Ground-truth index ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class GTIndex:
    """Lookup structures derived once from the NEW ground-truth entries.

    ``polarity`` maps every evaluated (file_key, smell_pascal) pair to its
    ground truth (``True`` = present, ``False`` = absent). ``yaml_sha`` maps each
    file_key to the sha256 of its ``yaml_content`` — the same normalisation
    :func:`gha_agents.main` bakes into a result's ``input_yaml_sha256``.
    ``conflicts`` holds any (file_key, smell) listed as BOTH present and absent
    (a malformed GT); the last occurrence wins in ``polarity`` and the pair is
    reported so it is never silently classified.
    """

    polarity: dict[tuple[str, str], bool]
    yaml_sha: dict[str, str]
    pairs: frozenset[tuple[str, str, bool]]
    conflicts: frozenset[tuple[str, str]]


def yaml_fingerprint(yaml_content: str) -> str:
    """sha256 over the UTF-8 bytes — identical to main.py's ``input_yaml_sha256``."""
    return hashlib.sha256(yaml_content.encode("utf-8")).hexdigest()


def build_gt_index(entries: list[dict[str, Any]]) -> GTIndex:
    """Index the NEW ground-truth entries for O(1) per-result classification."""
    polarity: dict[tuple[str, str], bool] = {}
    yaml_sha: dict[str, str] = {}
    pairs: set[tuple[str, str, bool]] = set()
    seen_present: set[tuple[str, str]] = set()
    seen_absent: set[tuple[str, str]] = set()

    for entry in entries:
        file_key = entry["file_key"]
        yaml_sha[file_key] = yaml_fingerprint(entry.get("yaml_content", ""))
        for smell in entry.get("present_smells", []):
            key = (file_key, smell["pascal"])
            polarity[key] = True
            pairs.add((file_key, smell["pascal"], True))
            seen_present.add(key)
        for smell in entry.get("absent_smells", []):
            key = (file_key, smell["pascal"])
            polarity[key] = False
            pairs.add((file_key, smell["pascal"], False))
            seen_absent.add(key)

    conflicts = seen_present & seen_absent
    return GTIndex(
        polarity=polarity,
        yaml_sha=yaml_sha,
        pairs=frozenset(pairs),
        conflicts=frozenset(conflicts),
    )


# ─── Classification ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Verdict:
    kind: VerdictKind
    reason: str = ""

    @property
    def is_rerun(self) -> bool:
        return self.kind in _RERUN_KINDS


def _result_polarity(result: dict[str, Any]) -> bool | None:
    """Read the baked ground-truth polarity, tolerating the legacy key name."""
    pol = result.get("ground_truth")
    if pol is None:
        pol = result.get("ground_truth_positive")
    return None if pol is None else bool(pol)


def _result_smell(result: dict[str, Any]) -> str | None:
    return result.get("target_smell") or result.get("target_smell_pascal")


def classify(result: dict[str, Any], gt: GTIndex) -> Verdict:
    """Classify one on-disk result against the NEW ground truth.

    Order matters: a pair entirely absent from the GT is an ``orphan``; a pair
    still present but with the opposite polarity is a ``rerun_flip`` (the slug is
    unchanged, so resume would wrongly skip it); a surviving, same-polarity pair
    whose input workflow hash moved is a ``rerun_yaml``. Everything else is a
    ``keep``. A result missing its identity fields is ``unreadable`` and left
    untouched.
    """
    file_key = result.get("file_key")
    smell = _result_smell(result)
    if not file_key or not smell:
        return Verdict("unreadable", "result missing file_key or target_smell")

    key = (file_key, smell)
    if key not in gt.polarity:
        return Verdict("orphan", "pair no longer in ground truth")

    pol = _result_polarity(result)
    if pol is not None and pol != gt.polarity[key]:
        return Verdict(
            "rerun_flip",
            f"polarity flipped {_pol_word(pol)} → {_pol_word(gt.polarity[key])}",
        )

    baked = result.get("input_yaml_sha256")
    expected = gt.yaml_sha.get(file_key)
    if baked and expected and baked != expected:
        return Verdict("rerun_yaml", "yaml_content changed under same file_key")

    return Verdict("keep")


def _pol_word(positive: bool) -> str:
    return "present" if positive else "absent"


# ─── On-disk discovery ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResultCell:
    """One classified result and the store it lives under."""

    path: Path
    store_base: Path
    slug: str
    model: str | None
    result: dict[str, Any]
    verdict: Verdict


def _under_quarantine(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    return any(part in _QUARANTINE_DIRS for part in rel_parts)


def iter_result_files(root: Path) -> Iterator[Path]:
    """Yield every ``*/results/*.json`` under ``root``, skipping quarantine dirs.

    Handles the per-design layout (``design_d1/results``) and a flat top-level
    ``results`` alike — any directory
    literally named ``results`` is a store. Already-quarantined files (under
    ``_orphaned``/``_superseded``) are never re-collected.
    """
    for results_dir in sorted(root.rglob("results")):
        if not results_dir.is_dir() or _under_quarantine(results_dir, root):
            continue
        yield from sorted(results_dir.glob("*.json"))


def _slug_model(slug: str) -> str | None:
    """The trailing ``@model_alias`` of a slug, or None for the legacy CLI format.

    The model alias is the last ``@``-segment, but the legacy CLI slug has none
    (``stem@D1@rep1@smell``). We can't tell a model-less slug from a 4-segment
    one structurally, so callers that need exactness pass ``--model`` and we match
    on the suffix; this helper is best-effort for display only.
    """
    # A synthetic positive's slug carries a trailing @synth tag (added by
    # _build_slug); strip it so the model is read from the same position as a real
    # instance's slug.
    s = slug[: -len("@synth")] if slug.endswith("@synth") else slug
    return s.rsplit("@", 1)[-1] if s.count("@") >= 4 else None


def load_cell(path: Path, root: Path, gt: GTIndex) -> ResultCell:
    """Read + classify one result file into a :class:`ResultCell`."""
    slug = path.stem
    store_base = path.parent.parent  # …/<base>/results/<slug>.json
    result: dict[str, Any] = {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        verdict = Verdict("unreadable", f"could not read result: {exc}")
    else:
        if isinstance(loaded, dict):
            result = loaded
            verdict = classify(result, gt)
        else:
            verdict = Verdict("unreadable", "result JSON is not a JSON object")
    return ResultCell(
        path=path,
        store_base=store_base,
        slug=slug,
        model=_slug_model(slug),
        result=result,
        verdict=verdict,
    )


def cell_artifacts(store_base: Path, slug: str) -> list[Path]:
    """All existing on-disk artifacts for one slug, via the RunStore grammar.

    Result JSON, winning patched workflow, trace, every per-iteration patch
    and iteration log. Only paths that actually exist are returned, so the mover
    never touches absent files.
    """
    store = RunStore(store_base)
    candidates = [
        store.result_path(slug),
        store.patched_path(slug),
        store.trace_path(slug),
    ]
    candidates += sorted(store.iters_dir.glob(f"{slug}@iter*.yml"))
    candidates += sorted(store.iter_logs_dir.glob(f"{slug}@*.json"))
    return [p for p in candidates if p.exists()]


# ─── Planning ─────────────────────────────────────────────────────────────────


@dataclass
class ReconcilePlan:
    root: Path
    model: str | None
    cells: list[ResultCell] = field(default_factory=list)
    added_pairs: frozenset[tuple[str, str, bool]] = frozenset()
    gt_conflicts: frozenset[tuple[str, str]] = frozenset()

    def by_kind(self, kind: VerdictKind) -> list[ResultCell]:
        return [c for c in self.cells if c.verdict.kind == kind]

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for cell in self.cells:
            out[cell.verdict.kind] = out.get(cell.verdict.kind, 0) + 1
        out["added"] = len(self.added_pairs)
        return out

    @property
    def needs_action(self) -> bool:
        return any(
            c.verdict.kind == "orphan" or c.verdict.is_rerun for c in self.cells
        )


def build_plan(
    root: Path,
    gt: GTIndex,
    *,
    model: str | None = None,
) -> ReconcilePlan:
    """Classify every in-scope result on disk and compute the added-pair set.

    ``model`` restricts the scan to slugs ending in ``@{model}`` — use it to
    reconcile one model's results (e.g. ``deepseek-v4-flash``) without touching
    the others sharing the output directory. ``added_pairs`` are GT pairs with no
    on-disk result in scope; resume runs them fresh, so they need no action and
    are reported for transparency only.
    """
    cells: list[ResultCell] = []
    covered: set[tuple[str, str]] = set()
    suffix = f"@{model}" if model else None

    for path in iter_result_files(root):
        if suffix is not None:
            # Synthetic slugs end @{model}@synth, so match the model segment
            # whether or not a trailing @synth tag follows it -- else per-model
            # reconcile would silently skip every synthetic result.
            stem = path.stem
            base = stem[: -len("@synth")] if stem.endswith("@synth") else stem
            if not base.endswith(suffix):
                continue
        cell = load_cell(path, root, gt)
        cells.append(cell)
        file_key = cell.result.get("file_key")
        smell = _result_smell(cell.result)
        if file_key and smell:
            covered.add((file_key, smell))

    added = frozenset(p for p in gt.pairs if (p[0], p[1]) not in covered)
    return ReconcilePlan(
        root=root,
        model=model,
        cells=cells,
        added_pairs=added,
        gt_conflicts=gt.conflicts,
    )


# ─── Apply ────────────────────────────────────────────────────────────────────


def _quarantine_bucket(verdict: Verdict) -> str:
    return _ORPHAN_DIR if verdict.kind == "orphan" else _SUPERSEDED_DIR


def _move_into(path: Path, root: Path, bucket: str) -> Path:
    """Move ``path`` to ``root/bucket/<path-relative-to-root>``; return the dest."""
    dest = root / bucket / path.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    return dest


def apply_plan(plan: ReconcilePlan) -> dict[str, int]:
    """Quarantine orphans and supersede rerun cells; return per-bucket file counts.

    Orphans move to ``_orphaned/``; rerun cells (flip or yaml drift) move to
    ``_superseded/``. Both preserve the relative path under the root, so the move
    is a reversible quarantine — nothing is deleted. After a rerun cell's result
    is moved out of ``results/``, the next ``resume`` run re-runs exactly that
    slug (its result no longer satisfies the skip predicate). ``keep`` and
    ``unreadable`` cells are left untouched.
    """
    moved: dict[str, int] = {_ORPHAN_DIR: 0, _SUPERSEDED_DIR: 0}
    for cell in plan.cells:
        if cell.verdict.kind != "orphan" and not cell.verdict.is_rerun:
            continue
        bucket = _quarantine_bucket(cell.verdict)
        for artifact in cell_artifacts(cell.store_base, cell.slug):
            _move_into(artifact, plan.root, bucket)
            moved[bucket] += 1
    return moved


# ─── CLI ──────────────────────────────────────────────────────────────────────


app = typer.Typer(
    name="study-reconcile",
    help="Reconcile on-disk results against an edited ground truth before resume.",
    add_completion=False,
)


def _print_report(plan: ReconcilePlan) -> None:
    counts = plan.counts
    typer.echo("\n  Reconcile report")
    typer.echo(f"  root:  {plan.root}")
    typer.echo(f"  model: {plan.model or '(all)'}")
    typer.echo(
        "  ── keep={keep}  rerun_flip={rerun_flip}  rerun_yaml={rerun_yaml}  "
        "orphan={orphan}  added={added}  unreadable={unreadable}".format(
            keep=counts.get("keep", 0),
            rerun_flip=counts.get("rerun_flip", 0),
            rerun_yaml=counts.get("rerun_yaml", 0),
            orphan=counts.get("orphan", 0),
            added=counts.get("added", 0),
            unreadable=counts.get("unreadable", 0),
        )
    )
    if plan.gt_conflicts:
        typer.echo(
            typer.style(
                f"  ⚠  {len(plan.gt_conflicts)} (file_key, smell) listed as BOTH "
                "present and absent in the ground truth — fix the GT.",
                fg=typer.colors.YELLOW,
            )
        )
    for kind, label in (
        ("orphan", "ORPHAN (will quarantine)"),
        ("rerun_flip", "RERUN — polarity flip (will supersede → resume re-runs)"),
        ("rerun_yaml", "RERUN — yaml edit (will supersede → resume re-runs)"),
        ("unreadable", "UNREADABLE (left untouched)"),
    ):
        cells = plan.by_kind(kind)
        if not cells:
            continue
        typer.echo(f"\n  {label}: {len(cells)}")
        for cell in cells[:20]:
            typer.echo(f"    {cell.slug}  — {cell.verdict.reason}")
        if len(cells) > 20:
            typer.echo(f"    … and {len(cells) - 20} more")


_ARG_GT = typer.Argument(
    ..., exists=True, dir_okay=False, readable=True, help="NEW ground-truth JSON."
)
_ARG_ROOT = typer.Argument(
    ..., exists=True, file_okay=False, help="Output root holding results/ to reconcile."
)
_OPT_MODEL = typer.Option(
    None, "--model", help="Restrict to slugs ending in @MODEL (e.g. deepseek-v4-flash)."
)
_OPT_APPLY = typer.Option(
    False, "--apply", help="Actually move files. Without it this is a dry-run report."
)


@app.command()
def run(
    ground_truth: Path = _ARG_GT,
    output_root: Path = _ARG_ROOT,
    model: str | None = _OPT_MODEL,
    apply: bool = _OPT_APPLY,
) -> None:
    """Diff the NEW ground truth against on-disk results and report (or --apply)."""
    entries = json.loads(ground_truth.read_text(encoding="utf-8"))
    gt = build_gt_index(entries)
    plan = build_plan(output_root, gt, model=model)
    _print_report(plan)

    if not apply:
        if plan.needs_action:
            typer.echo(
                "\n  Dry-run. Re-run with --apply to quarantine orphans and "
                "supersede rerun cells, then resume the experiment."
            )
        else:
            typer.echo("\n  Nothing to do — every kept result is coherent.")
        return

    moved = apply_plan(plan)
    typer.echo(
        f"\n  ✔ applied — moved {moved[_ORPHAN_DIR]} file(s) → {_ORPHAN_DIR}/, "
        f"{moved[_SUPERSEDED_DIR]} file(s) → {_SUPERSEDED_DIR}/.\n"
        "  Now resume the experiment: the superseded slugs re-run, the rest skip."
    )


if __name__ == "__main__":
    app()
