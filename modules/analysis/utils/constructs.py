"""Construct loaders over the result corpus: walk patches, diff, write raw frames.

Walks the pipeline's result trees (``design_d*/results/*.json`` +
``patched_workflows/``) and the ground-truth JSON, and for every refactor patch
emits, via :mod:`utils.construct_diff` (schema-free, paper-catalogue based):

* the construct **tree** of the original and of the patched workflow,
* the added / removed / changed construct **diff**,
* the textual YAML diff + churn, and
* the schema-invalid-added count for SVR (the *only* schema touch, delegated to
  :mod:`utils.schema_validity`).

Restricted to the 19 paper smells. This module is the data layer: it produces
the raw per-patch frames + per-file construct trees and writes them to
``artifacts/constructs/``. The RQ2 *metrics* (footprint, SRR/SVR/CCR/NMR, the
oracle correct/attention rule) are defined in the notebooks (rq2_a / rq2_b),
which read these frames.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils import schema_validity
from utils.construct_diff import (
    churn,
    construct_trees,
    diff_records,
    evaluate_construct_patch,
    parse_workflow,
    unified_diff,
)
from utils.dataloader import DATA_ROOT
from utils.designs import DESIGNS, run_dir

# ─── The 19 paper smells (flat — no category grouping) ────────────────────────

PAPER_SMELLS: tuple[str, ...] = (
    "MissingPermExternalActions",
    "UnpinnedActionVersion",
    "MissingTokenPermissions",
    "MissingForkGuardPR",
    "MissingJobTimeout",
    "MissingConcurrencyPR",
    "MissingConcurrencyBranch",
    "MissingPathFilters",
    "MissingForkGuardSchedule",
    "UnsafeArtifactUploadFork",
    "MissingUploadCondition",
    "UnsafeDeployOnFork",
    "MissingConcurrencyGlobal",
    "MutableRunnerVersion",
    "SuboptimalCacheUsage",
    "MultiCommandStep",
    "SingleOSExecution",
    "UnpinnedDependencies",
    "SingleLangVersionCI",
)
_SMELL_ORDER = {smell: i for i, smell in enumerate(PAPER_SMELLS)}


def smell_sort_key(smell: str) -> tuple[int, str]:
    """Order smells by paper position; unknown smells sort last, alphabetically."""
    return (_SMELL_ORDER.get(smell, len(PAPER_SMELLS)), smell)


def sort_by_smell(
    df: pd.DataFrame, by: Sequence[str], *, ascending: bool | Sequence[bool] = True
) -> pd.DataFrame:
    """``df.sort_values(by)`` with any ``smell`` column ordered by paper position."""
    if df.empty:
        return df
    return df.sort_values(
        list(by),
        ascending=ascending,
        key=lambda col: col.map(smell_sort_key) if col.name == "smell" else col,
    ).reset_index(drop=True)


def write_csvs(items: Iterable[tuple[str, pd.DataFrame]], csv_dir: Path) -> dict[str, Path]:
    """Write each ``(filename, frame)`` under ``csv_dir`` (skipping empty frames)."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, frame in items:
        if frame is None or frame.empty:
            continue
        path = csv_dir / name
        frame.to_csv(path, index=False)
        written[name] = path
    return written


# ─── Detection ground truth ───────────────────────────────────────────────────

class GroundTruthSchemaError(Exception):
    """The ground truth file is missing, malformed, or yields zero pairs."""


@dataclass(frozen=True)
class GroundTruthIndex:
    """Membership index over the ground truth's present/absent smell sets."""

    positive: frozenset[tuple[str, str]]  # (file_key, smell) that are present
    negative: frozenset[tuple[str, str]]  # (file_key, smell) that are absent

    def is_positive(self, file_key: str, smell: str) -> bool | None:
        """True/False for a known pair, ``None`` for an unknown (orphan) pair."""
        key = (file_key, smell)
        if key in self.positive:
            return True
        if key in self.negative:
            return False
        return None

    @property
    def supported_smells(self) -> frozenset[str]:
        """Smells with at least one positive pair — the macro-average domain."""
        return frozenset(smell for _, smell in self.positive)

    @property
    def file_keys(self) -> frozenset[str]:
        return frozenset(fk for fk, _ in (self.positive | self.negative))


def load_ground_truth(gt_path: Path) -> GroundTruthIndex:
    """Parse the ground truth into a :class:`GroundTruthIndex`."""
    try:
        raw = json.loads(gt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GroundTruthSchemaError(f"ground truth not found at {gt_path}") from exc
    except json.JSONDecodeError as exc:
        raise GroundTruthSchemaError(
            f"ground truth at {gt_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, list) or not raw:
        raise GroundTruthSchemaError(
            f"ground truth at {gt_path} must be a non-empty list of entries"
        )

    positive: set[tuple[str, str]] = set()
    negative: set[tuple[str, str]] = set()
    for entry in raw:
        file_key = entry.get("file_key")
        if not file_key:
            raise GroundTruthSchemaError(f"ground-truth entry missing file_key: {entry!r:.120}")
        for s in entry.get("present_smells", []) or []:
            positive.add((file_key, s["pascal"]))
        for s in entry.get("absent_smells", []) or []:
            negative.add((file_key, s["pascal"]))

    if not positive and not negative:
        raise GroundTruthSchemaError(
            f"ground truth at {gt_path} yielded zero pairs — wrong schema?"
        )
    return GroundTruthIndex(positive=frozenset(positive), negative=frozenset(negative))


def load_originals(gt_path: Path) -> dict[str, str]:
    """Map each ``file_key`` to its original workflow YAML (the before-state)."""
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    return {e["file_key"]: e.get("yaml_content", "") for e in raw if e.get("file_key")}


# ─── Result-tree loaders (restricted to the 19 paper smells) ──────────────────

def _design_dirs(results_root: Path, designs: tuple[str, ...]) -> dict[str, Path]:
    """Map each design to its ``design_d<n>/results`` dir, if it exists."""
    dirs: dict[str, Path] = {}
    for design in designs:
        d = results_root / run_dir(design) / "results"
        if d.is_dir():
            dirs[design] = d
    return dirs


_NAN_COUNTS = {
    "parseable": False,
    "n_added": float("nan"),
    "n_removed": float("nan"),
    "n_changed": float("nan"),
    "n_invalid_added": float("nan"),
}


def _truth(gt: GroundTruthIndex | None, file_key: str, smell: str | None) -> object:
    """The ground-truth verdict for a pair, or ``pd.NA`` when unknown/unindexed."""
    if gt is None or smell is None:
        return pd.NA
    verdict = gt.is_positive(file_key, smell)
    return pd.NA if verdict is None else verdict


def _model_of_stem(stem: str) -> str:
    """The model alias is the last ``@``-separated field of a result/error name."""
    return stem.rsplit("@", 1)[-1]


def _stem_to_file_key(results_dir: Path) -> dict[str, str]:
    """Map each result name stem → its authoritative ``file_key`` from the JSON."""
    out: dict[str, str] = {}
    if not results_dir.is_dir():
        return out
    for f in results_dir.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))["file_key"]
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return out


def load_construct_frame(
    *,
    results_root: Path,
    originals: dict[str, str],
    gt: GroundTruthIndex | None = None,
    designs: tuple[str, ...] = DESIGNS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict], int]:
    """Construct-diff every final patch of a refactored pair (``smell_detected``).

    Returns ``(occ, per_patch, trees, skipped)``:

    * ``occ`` — one row per (patch, axis, construct): the long frequency frame,
      each construct tagged with its catalogue ``feature`` / ``level`` and a
      ``catalog`` flag (False = path not in the paper catalogue → ``other``).
    * ``per_patch`` — one row per patch, including unparseable ones, carrying the
      diff counts, the churn, and ``n_invalid_added`` (the schema-invalid-added
      count for SVR — the only schema touch).
    * ``trees`` — ``{stem: construct_trees(...)}`` for every parseable patch (the
      construct content of the original + patched workflow + the diff axes).
    * ``skipped`` — patches whose original or patch YAML did not parse.
    """

    occ_rows: list[dict] = []
    pp_rows: list[dict] = []
    trees: dict[str, dict] = {}
    skipped = 0
    for design, results_dir in _design_dirs(results_root, designs).items():
        patch_dir = results_dir.parent / "patched_workflows"
        for path in sorted(results_dir.glob("*.json")):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not d.get("smell_detected"):
                continue  # no intentional refactor → not a patch to score
            file_key = d.get("file_key", "")
            smell = d.get("target_smell")
            if smell not in PAPER_SMELLS:
                continue  # restrict to the 19 paper smells
            original = originals.get(file_key)
            patch_file = patch_dir / f"{path.stem}.yml"
            if original is None or not patch_file.is_file():
                skipped += 1
                continue
            ident = {
                "file_key": file_key,
                "smell": smell,
                "design": d.get("design", design),
                "model": d.get("model_alias"),
                "rep": int(d.get("repetition", 1)),
                "gt_positive": _truth(gt, file_key, smell),
            }
            patch_text = patch_file.read_text(encoding="utf-8")
            c_added, c_removed = churn(original, patch_text)
            diffmeta = {
                "stem": path.stem,
                "churn_added": c_added,
                "churn_removed": c_removed,
                "yaml_diff": unified_diff(original, patch_text),
                # model's own refactor rationale + self-rated confidence, carried into
                # the per-patch index for the validation app (never used by metrics).
                "refactor_justification": d.get("refactor_justification", ""),
                "refactor_confidence": d.get("refactor_confidence", ""),
                "refactor_confidence_justification": d.get("refactor_confidence_justification", ""),
            }
            diff = evaluate_construct_patch(original, patch_text)
            if diff is None:  # unparseable original or patch
                skipped += 1
                _, oerr = parse_workflow(original)
                _, perr = parse_workflow(patch_text)
                err = oerr or perr or ""
                pp_rows.append({
                    **ident, **_NAN_COUNTS, **diffmeta,
                    "fail_side": "original" if oerr else "patch",
                    "parser_error": err.splitlines()[0] if err else "",
                    "patch_path": f"{DATA_ROOT.name}/{results_root.name}/{patch_file.relative_to(results_root)}",
                })
                continue
            trees[path.stem] = construct_trees(original, patch_text)
            recs = diff_records(diff)
            occ_rows.extend({**ident, **r} for r in recs)
            # SVR's only schema touch: how many added constructs the GHA schema rejects.
            n_invalid = schema_validity.invalid_added(original, patch_text)
            pp_rows.append(
                {
                    **ident,
                    "parseable": True,
                    "n_added": len(diff.added),
                    "n_removed": len(diff.removed),
                    "n_changed": len(diff.changed),
                    "n_invalid_added": 0 if n_invalid is None else n_invalid,
                    **diffmeta,
                }
            )
    return pd.DataFrame(occ_rows), pd.DataFrame(pp_rows), trees, skipped


# ─── Build (raw frames + per-file trees) ──────────────────────────────────────


@dataclass
class ConstructFrames:
    per_patch: pd.DataFrame      # one row per patch (incl. unparseable): counts + churn + SVR signal
    occ: pd.DataFrame            # one row per (patch, axis, construct) with catalogue feature/level
    per_patch_index: pd.DataFrame  # per_patch minus the bulky yaml_diff, + diff_path
    unparseable: pd.DataFrame    # patches the diff could not build (fail_side + parser_error)
    skipped: int
    n_yaml_diffs: int            # textual YAML diff files written under yaml_diffs/
    n_trees: int                 # construct-tree JSON files written under construct_trees/


def _write_yaml_diffs(per_patch: pd.DataFrame, csv_dir: Path) -> tuple[pd.DataFrame, int]:
    """Write one unified-diff file per patch under ``yaml_diffs/``; return (index, n)."""
    if per_patch.empty or "yaml_diff" not in per_patch.columns:
        return per_patch, 0
    diff_dir = csv_dir / "yaml_diffs"
    paths: list[str] = []
    n_written = 0
    for text, stem in zip(per_patch["yaml_diff"], per_patch["stem"], strict=False):
        if text and isinstance(stem, str) and stem:
            diff_dir.mkdir(parents=True, exist_ok=True)
            (diff_dir / f"{stem}.diff").write_text(f"{text}\n", encoding="utf-8")
            paths.append(f"yaml_diffs/{stem}.diff")
            n_written += 1
        else:
            paths.append("")
    index = per_patch.drop(
        columns=["yaml_diff", "fail_side", "parser_error", "patch_path"], errors="ignore"
    ).copy()
    index["diff_path"] = paths
    return index, n_written


def _write_trees(trees: dict[str, dict], csv_dir: Path) -> int:
    """Write one construct-tree JSON per patch under ``construct_trees/``.

    Each file holds ``{original, patched, added, removed, changed}`` — the
    construct content of both sides of the patch plus the three diff axes.
    """
    if not trees:
        return 0
    tree_dir = csv_dir / "construct_trees"
    tree_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem, tree in trees.items():
        if tree is None:
            continue
        (tree_dir / f"{stem}.json").write_text(
            json.dumps(tree, indent=2, sort_keys=False), encoding="utf-8"
        )
        n += 1
    return n


def _unparseable_frame(per_patch: pd.DataFrame) -> pd.DataFrame:
    """Patches the construct diff could not build: which side failed + parser message."""
    if per_patch.empty or "parseable" not in per_patch.columns:
        return pd.DataFrame()
    cols = ["file_key", "smell", "design", "model", "rep", "gt_positive",
            "stem", "fail_side", "parser_error", "patch_path"]
    u = per_patch.loc[~per_patch["parseable"], [c for c in cols if c in per_patch.columns]].copy()
    if u.empty:
        return u
    u["diff_path"] = "yaml_diffs/" + u["stem"].astype(str) + ".diff"
    return u.sort_values(["fail_side", "model", "smell"]).reset_index(drop=True)


def build(
    *,
    results_roots: Sequence[Path],
    originals: dict[str, str],
    csv_dir: Path,
    gt: GroundTruthIndex | None = None,
) -> tuple[ConstructFrames, dict[str, Path]]:
    """Diff every final + intermediate patch across result roots; write raw frames.

    One root per model; the frames are pooled so the construct tables group across
    every model in one set of CSVs. Writes ``construct_per_patch.csv`` (occ),
    ``construct_per_patch_index.csv``, ``unparseable_patches.csv``, the per-patch
    ``yaml_diffs/`` and ``construct_trees/``. The RQ2 metrics are computed in the
    notebooks from these frames. The ground truth (present/absent) is carried into
    the raw frames as ``gt_positive``, never used as a filter.
    """
    occ_frames: list[pd.DataFrame] = []
    per_patch_frames: list[pd.DataFrame] = []
    all_trees: dict[str, dict] = {}
    skipped = 0
    for root in results_roots:
        occ, per_patch_frame, trees, n_skipped = load_construct_frame(
            results_root=root, originals=originals, gt=gt
        )
        skipped += n_skipped
        all_trees.update(trees)
        if not occ.empty:
            occ_frames.append(occ)
        if not per_patch_frame.empty:
            per_patch_frames.append(per_patch_frame)

    occ = pd.concat(occ_frames, ignore_index=True) if occ_frames else pd.DataFrame()
    per_patch = pd.concat(per_patch_frames, ignore_index=True) if per_patch_frames else pd.DataFrame()

    csv_dir.mkdir(parents=True, exist_ok=True)
    per_patch_index, n_yaml_diffs = _write_yaml_diffs(per_patch, csv_dir)
    n_trees = _write_trees(all_trees, csv_dir)

    frames = ConstructFrames(
        per_patch=per_patch,
        occ=occ,
        per_patch_index=per_patch_index,
        unparseable=_unparseable_frame(per_patch),
        skipped=skipped,
        n_yaml_diffs=n_yaml_diffs,
        n_trees=n_trees,
    )

    written = write_csvs(
        [
            ("construct_per_patch.csv", occ),
            ("construct_per_patch_index.csv", per_patch_index),
            ("unparseable_patches.csv", frames.unparseable),
        ],
        csv_dir,
    )
    if n_yaml_diffs:
        written["yaml_diffs/"] = csv_dir / "yaml_diffs"
    if n_trees:
        written["construct_trees/"] = csv_dir / "construct_trees"
    return frames, written
