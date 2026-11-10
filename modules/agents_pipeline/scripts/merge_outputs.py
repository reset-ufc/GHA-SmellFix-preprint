#!/usr/bin/env python3
"""Merge N agent-output folders into one merged folder for a single ground truth.

Use when a model's results were produced across several runs because the ground
truth drifted between them — e.g. a MissingPermExternalActions detector fix that
forced an MPE re-run, or an UnsafeDeployOnFork synthesis catch-up — and you want
the single-folder / single-GT layout the rest of the pipeline assumes
(``build_results_csv``, ``coverage_report``, ``reconcile``).

A run can be driven by the CLI flags below, or by a JSON manifest carrying the
same fields fed via ``--manifest``.

Examples
--------
    # n15: 3 sources + the 3 n15 GTs unioned (no merged 12e_n15 exists)
    uv run python scripts/merge_outputs.py \
        --dst output-gpt-n15 --on-collision last-wins \
        --gt data/12b_ground_truth_n15.json \
        --gt data/12b_ground_truth_n15_MissingPermExternalActions.json \
        --gt data/12f_ground_truth_n15fill_UnsafeDeployOnFork.json \
        --src output-stability-gpt-n15 \
        --src output-stability-gpt-n15-MissingPermExternalActions \
        --src output-unsafedeploy-gpt-n15fill

Semantics
---------
* ``--src`` order is priority ASCENDING: on a result-filename collision the LAST
  src wins (it supersedes). Default ``--on-collision=error`` aborts if any
  collision exists — pass ``last-wins`` only when a re-run is meant to override
  stale base slugs (e.g. an MPE re-run vs a pre-fix base). All of a kept result's
  sibling artifacts are taken from the SAME winning source.
* A result is KEPT iff its ``(stem, smell, synth)`` pair is in the union of the
  ``--gt`` worklists. Pairs outside the GT are dropped as orphans (GT-drift
  residue). GT pairs no source produced are reported as missing (and fail
  ``--strict``).
* Every sibling subdir is copied by OWNER SLUG: a file belongs to result slug S
  iff stripping its extension and any ``@detection{n}``/``@refactor{n}``/
  ``@iter{n}`` suffix yields exactly S. This copies results, patched_workflows,
  traces, error_artifacts — and any future subdir — only for kept results, with
  no strays and no plain-vs-synth (``S`` vs ``S@synth``) cross-assignment.
  Quarantine dirs (``_orphaned``/``_superseded``) are never read.
* ``patched_workflow_path`` inside each result JSON is re-rooted to ``--dst``
  (disable with ``--no-rewrite-paths``).
* ``--strict`` exits nonzero on any collision (error mode), missing GT pair, or
  empty/zero-design merge. Dropped orphans are BY DESIGN (stale-GT residue) — they
  are always reported but never fail the run.

Design layout: one ``design_d{n}`` directory per design, in every source and in the destination.

Slug grammar (mirrors ``gha_agents.main._build_slug`` — keep in sync):
    {stem}@{D1|D2}@rep{k}@{smell_lower}@{model_alias}[@synth]
    stem = file_key with ``##synth`` marker and ``.yml``/``.yaml`` stripped.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

# scripts/ -> agents_pipeline -> modules -> replication-package (repo root, where
# data/ and the output-* dirs live). Relative --src/--gt/--dst resolve against it,
# but a relative path that already exists under CWD wins.
REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTH_MARKER = "##synth"                                  # main._SYNTH_MARKER
QUARANTINE_DIRS = {"_orphaned", "_superseded"}            # reconcile._QUARANTINE_DIRS
DESIGN_RE = re.compile(r"^design_d(\d+)$", re.IGNORECASE)
ITER_SUFFIX_RE = re.compile(r"@(?:detection|refactor|iter)\d+$", re.IGNORECASE)


# ---------------------------------------------------------------- slug / GT keys
def stem_from_file_key(file_key: str) -> str:
    """Strip ``##synth`` then ``.yaml``/``.yml`` — mirror of main._stem_from_file_key."""
    key = file_key[: -len(SYNTH_MARKER)] if file_key.endswith(SYNTH_MARKER) else file_key
    if key.endswith(".yaml"):
        return key[: -len(".yaml")]
    if key.endswith(".yml"):
        return key[: -len(".yml")]
    return key


def owner_slug(filename: str) -> str:
    """The result slug that owns a sibling artifact file.

    Strips a trailing extension (``.json``/``.yml``/``.raw``/…) — but not the dots
    inside a stem like ``…checkmk.general@…`` since only the LAST segment is an
    ext — then a trailing iteration tag (``@detection1``/``@refactor2``/``@iter1``).
    ``S.json`` and ``S@iter1.yml`` and ``S@synth@iter1.yml`` map to ``S`` / ``S@synth``.
    """
    core, dot, ext = filename.rpartition(".")
    base = core if (dot and core and "@" not in ext and "." not in ext) else filename
    return ITER_SUFFIX_RE.sub("", base)


def pair_from_slug(slug: str) -> tuple[str, str, bool]:
    """result slug (no extension) -> (stem, smell_lower, synth). rep/design/model dropped."""
    parts = slug.split("@")
    synth = parts[-1] == "synth"
    if synth:
        parts = parts[:-1]
    smell = parts[-2].lower()
    stem = "@".join(parts[:-4])
    return stem, smell, synth


def gt_worklist(gt_paths: list[Path]) -> set[tuple[str, str, bool]]:
    """Union of (stem, smell_lower, synth) over every GT file's present+absent smells."""
    work: set[tuple[str, str, bool]] = set()
    for p in gt_paths:
        for inst in json.loads(p.read_text(encoding="utf-8")):
            fk = inst["file_key"]
            synth = fk.endswith(SYNTH_MARKER)              # same _is_synthetic rule
            stem = stem_from_file_key(fk)
            for bucket in ("present_smells", "absent_smells"):
                for s in inst.get(bucket, []) or []:
                    name = s["pascal"] if isinstance(s, dict) else s
                    work.add((stem, name.lower(), synth))
    return work


# ---------------------------------------------------------------- discovery
def design_dirs(src: Path) -> dict[str, Path]:
    """normalized 'd{n}' -> path for every design_d* dir under src."""
    out: dict[str, Path] = {}
    for d in sorted(src.iterdir()) if src.is_dir() else []:
        if not d.is_dir() or d.name in QUARANTINE_DIRS:
            continue
        m = DESIGN_RE.match(d.name)
        if m:
            out[f"d{m.group(1)}"] = d
    return out


def index_by_owner(design_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """owner_slug -> [(subdir_name, file_path)] across every non-quarantine subdir."""
    idx: dict[str, list[tuple[str, Path]]] = collections.defaultdict(list)
    for sub in sorted(design_dir.iterdir()):
        if not sub.is_dir() or sub.name in QUARANTINE_DIRS:
            continue
        for f in sub.iterdir():
            if f.is_file():
                idx[owner_slug(f.name)].append((sub.name, f))
    return idx


# ---------------------------------------------------------------- plan
class Plan:
    def __init__(self) -> None:
        self.copies: list[tuple[Path, Path, bool]] = []   # (src, dst, is_result_json)
        self.fatal = 0
        self.per_design: dict[str, dict] = {}
        self.global_missing: set = set()


def build_plan(srcs: list[Path], dst: Path, gt: set, on_collision: str) -> Plan:
    plan = Plan()
    all_designs = sorted({c for s in srcs for c in design_dirs(s)})
    covered_global: set = set()
    for dnum in all_designs:
        present = [(s, design_dirs(s)[dnum]) for s in srcs if dnum in design_dirs(s)]
        winner: dict[str, Path] = {}
        seen: dict[str, list[str]] = collections.defaultdict(list)
        for src_root, design_dir in present:
            rd = design_dir / "results"
            if not rd.is_dir():
                continue
            for f in sorted(rd.iterdir()):
                if f.name.endswith(".json"):
                    slug = f.name[:-5]
                    seen[slug].append(src_root.name)
                    winner[slug] = design_dir
        collisions = {s: who for s, who in seen.items() if len(who) > 1}
        if collisions and on_collision == "error":
            plan.fatal += len(collisions)

        kept: dict[str, Path] = {}
        orphans: list[str] = []
        reps: collections.Counter = collections.Counter()
        covered: set = set()
        for slug, design_dir in winner.items():
            pair = pair_from_slug(slug)
            if pair in gt:
                kept[slug] = design_dir
                covered.add(pair)
                m = re.search(r"@(rep\d+)@", slug)
                if m:
                    reps[m.group(1)] += 1
            else:
                orphans.append(slug)
        covered_global |= covered

        # schedule copies: every sibling owned by a kept slug, from its winning src
        idx_cache: dict[Path, dict] = {}
        for slug, W in kept.items():
            idx = idx_cache.setdefault(W, index_by_owner(W))
            for subname, path in idx.get(slug, []):
                is_res = subname == "results" and path.name == f"{slug}.json"
                plan.copies.append((path, dst / f"design_{dnum}" / subname / path.name, is_res))

        plan.per_design[dnum] = dict(
            sources=[s.name for s, _ in present], kept=len(kept),
            orphans=orphans, missing=gt - covered, collisions=collisions, reps=dict(reps),
        )
    plan.global_missing = gt - covered_global
    return plan


# ---------------------------------------------------------------- report / run
def report(plan: Plan, gt_size: int, on_collision: str, strict: bool) -> bool:
    ok = True
    print(f"GT worklist pairs (synth-aware, unioned) = {gt_size}\n")
    if not plan.per_design:
        print("  !! NO design dirs found under any --src (looked for design_d*).")
        print("     Are the sources nested one level too deep (e.g. tarball extracted to src/<name>/design_d1)?")
        return False
    for dnum, info in sorted(plan.per_design.items()):
        coll, miss, orph = len(info["collisions"]), len(info["missing"]), len(info["orphans"])
        # --strict fails on genuine incompleteness (missing GT pairs) or unresolved
        # collisions. Dropped orphans are BY DESIGN (stale-GT residue) -> reported, not fatal.
        bad = (coll and on_collision == "error") or (strict and miss)
        ok &= not bad
        print(f"  design_{dnum}: sources={info['sources']} kept={info['kept']} "
              f"reps={info['reps']} collisions={coll} dropped_orphans={orph} "
              f"missing_gt_pairs={miss}  {'FAIL' if bad else 'OK'}")
        if coll:
            print(f"      collision sample ({on_collision}): {list(info['collisions'].items())[:3]}")
        if orph:
            print(f"      dropped-orphan sample: {info['orphans'][:3]}")
        if miss:
            print(f"      missing-gt sample: {list(info['missing'])[:3]}")
    if plan.global_missing:
        print(f"\n  GLOBAL missing GT pairs (no source produced them, any design): {len(plan.global_missing)}")
        if strict:
            ok = False
    return ok


def execute(plan: Plan, srcs: list[Path], dst: Path, rewrite: bool) -> None:
    rewrites = 0
    rewrite_re = re.compile(r"^(" + "|".join(re.escape(s.name) for s in srcs) + r")/")
    for src_file, dst_file, is_res in plan.copies:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if is_res and rewrite:
            raw = json.loads(src_file.read_text(encoding="utf-8"))
            pwp = raw.get("patched_workflow_path")
            if pwp:
                new = rewrite_re.sub(dst.name + "/", pwp, count=1)
                if new != pwp:
                    raw["patched_workflow_path"] = new
                    rewrites += 1
            dst_file.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            shutil.copy2(src_file, dst_file)
    for dnum in plan.per_design:
        chunks = []
        for s in srcs:
            cd = design_dirs(s).get(dnum)
            elog = (cd / "error.log") if cd else None
            if elog and elog.is_file():
                chunks.append(f"# ===== {s.name} =====\n{elog.read_text(encoding='utf-8')}")
        if chunks:
            (dst / f"design_{dnum}").mkdir(parents=True, exist_ok=True)
            (dst / f"design_{dnum}" / "error.log").write_text("".join(chunks), encoding="utf-8")
    print(f"\npatched_workflow_path rewrites: {rewrites}")


def post_verify(dst: Path, gt: set) -> bool:
    ok = True
    found = sorted(d for d in dst.iterdir() if d.is_dir() and DESIGN_RE.match(d.name))
    if not found:
        print("  !! post_verify: no design dirs in dst"); return False
    for cd in found:
        res_slugs = {f.name[:-5] for f in (cd / "results").iterdir() if f.name.endswith(".json")}
        pairs = {pair_from_slug(s) for s in res_slugs}
        orphan = pairs - gt
        strays = 0
        sub_counts = {}
        for sub in sorted(cd.iterdir()):
            if not sub.is_dir():
                continue
            files = [f for f in sub.iterdir() if f.is_file()]
            sub_counts[sub.name] = len(files)
            strays += sum(1 for f in files if owner_slug(f.name) not in res_slugs)
        bad = bool(orphan) or strays > 0
        ok &= not bad
        print(f"  {cd.name}: results={len(res_slugs)} orphan_vs_gt={len(orphan)} "
              f"strays={strays} subdirs={sub_counts}  {'FAIL' if bad else 'OK'}")
    return ok


# ---------------------------------------------------------------- cli
def resolve(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return pp.resolve() if pp.exists() else (REPO_ROOT / pp).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge N agent-output folders into one (single GT).")
    ap.add_argument("--manifest")
    ap.add_argument("--dst")
    ap.add_argument("--src", action="append", default=[], help="source folder (repeat; LAST wins on collision)")
    ap.add_argument("--gt", action="append", default=[], help="ground-truth json (repeat; worklists unioned)")
    ap.add_argument("--on-collision", choices=("error", "last-wins"), default=None)
    ap.add_argument("--rewrite-paths", dest="rewrite", action="store_const", const=True, default=None)
    ap.add_argument("--no-rewrite-paths", dest="rewrite", action="store_const", const=False)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    m: dict = {}
    if a.manifest:
        m = json.loads(resolve(a.manifest).read_text(encoding="utf-8"))
    dst = a.dst or m.get("dst")
    src = a.src or m.get("sources", [])
    gt_in = a.gt or m.get("gt", [])
    # CLI flags win over manifest; manifest wins over hard default
    on_collision = a.on_collision or m.get("on_collision") or "error"
    rewrite = a.rewrite if a.rewrite is not None else m.get("rewrite_paths", True)

    if not dst or not src or not gt_in:
        ap.error("need --dst, at least one --src, and at least one --gt (or --manifest)")

    dst = resolve(dst)
    srcs = [resolve(s) for s in src]
    gt_paths = [resolve(g) for g in gt_in]
    for s in srcs:
        if not s.is_dir():
            ap.error(f"source not found: {s}")
    for g in gt_paths:
        if not g.is_file():
            ap.error(f"GT not found: {g}")

    gt = gt_worklist(gt_paths)
    print(f"DST={dst.name}  SRCS={[s.name for s in srcs]}  on_collision={on_collision} "
          f"rewrite={rewrite} dry_run={a.dry_run} strict={a.strict}\n")

    plan = build_plan(srcs, dst, gt, on_collision)
    ok = report(plan, len(gt), on_collision, a.strict)

    if not plan.per_design:
        print("\nABORT: nothing to merge (no design dirs discovered).")
        return 6
    if plan.fatal and on_collision == "error":
        print(f"\nABORT: {plan.fatal} filename collisions in error mode. "
              f"Re-run with --on-collision last-wins if a re-run is meant to supersede.")
        return 2
    if not ok:
        print("\nABORT: --strict and report has missing/orphan/collision FAILs.")
        return 3

    if a.dry_run:
        nres = sum(1 for _, _, r in plan.copies if r)
        print(f"\nDRY-RUN: would copy {len(plan.copies)} files ({nres} results) into {dst.name}. No writes.")
        return 0

    if dst.exists():
        print(f"\nABORT: dst {dst} already exists. Remove it or pick a new --dst.")
        return 4

    execute(plan, srcs, dst, rewrite)
    print(f"\nEXECUTED into {dst}")
    verified = post_verify(dst, gt)
    return 0 if (verified or not a.strict) else 5


if __name__ == "__main__":
    raise SystemExit(main())
