"""Shared ground-truth-sampling primitives for Step 10 and Step 10b.

Extracted verbatim from ``10_sample_ground_truth.ipynb`` so the full-run sampler
(Step 10) and the targeted re-sample top-up (Step 10b) draw candidates with
*identical* semantics: same smell catalog, same strict filter, same ranking,
and the same LOC-quartile balanced selection.

Nothing here is Step-10-specific. The full-run orchestration (``sample_ground_truth``,
LaTeX tables, report schema) stays in the Step 10 notebook; the targeted
re-sample orchestration stays in the Step 10b notebook. Both import these.
"""

import math
import re
from pathlib import Path

# -- Sampling configuration ---------------------------------------------------

N_PER_POLARITY = 50
"""Paper-stated target: confirmed-correct manual labels per polarity per
behavioral smell. Used by downstream analyses (per_smell coverage table,
LaTeX summaries) and by the manual-annotation tool as the stopping
threshold (see TODO.md in study-manualvalidation)."""

OVERSAMPLE_FACTOR = 2
"""Annotation-pool inflation factor. Step 10 delivers N_PER_POLARITY *
OVERSAMPLE_FACTOR candidates per polarity per smell so the manual-annotation
tool has reserves to absorb FP/FN. The first N_PER_POLARITY confirmed-correct
labels per (smell, polarity) bucket form the final ground_truth; remaining
priority>N_PER_POLARITY instances are reserves."""

N_DELIVERED_PER_POLARITY = N_PER_POLARITY * OVERSAMPLE_FACTOR
"""Candidates emitted per polarity per smell = 100. Total slots per smell
= 200; total across 19 behavioral smells = 3800 (file, smell) pairs."""

LOC_QUARTILE_POOL_MULTIPLIER = 4
"""Pool size multiplier for LOC-balanced selection. From the top-ranked
candidates we take a pool of (MULTIPLIER * N) files, split into four LOC
quartiles, and draw equally from each quartile."""

# -- Behavioral/presentational partition --------------------------------------

PRESENTATIONAL_SMELLS: frozenset[str] = frozenset({
    "AddCommentsToWorkflows",
    "NameRunSteps",
    "CorrectIndentation",
})
"""Smells affecting only YAML readability / documentation / log observability.
Recorded on each ground_truth entry under ``presentational_smells`` but NOT
evaluated against agent output."""

YAML_CACHE_DIR = Path("data/downloaded_workflows")

# -- Smell catalog (Khatami et al. SCAM 2024, Table III) ----------------------

# Keys match the PascalCase identifiers emitted by step 10.
#   value = (paper_id, family, latex_display_name)
SMELL_CATALOG: dict[str, tuple[int, str, str]] = {
    "MissingPermExternalActions":
        (1, "Security", "Define permissions for workflows with external actions"),
    "UnpinnedActionVersion":
        (2, "Security", "Use commit hash instead of tags for action versions"),
    "MissingTokenPermissions":
        (3, "Security", "Set permissions for GitHub Token"),
    "MissingForkGuardPR":
        (4, "Performance/Optimization", "Prevent running issue/PR actions on forks"),
    "MissingJobTimeout":
        (5, "Performance/Optimization", "Avoid jobs without timeouts"),
    "MissingConcurrencyPR":
        (6, "Performance/Optimization", "Stop running workflows when there is a newer commit in PR"),
    "MissingConcurrencyBranch":
        (7, "Performance/Optimization", "Stop running workflows when there is a newer commit in branch"),
    "MissingPathFilters":
        (8, "Performance/Optimization", "Avoid running CI actions when no source code has changed"),
    "MissingForkGuardSchedule":
        (9, "Performance/Optimization", "Avoid executing scheduled workflows on forks"),
    "UnsafeArtifactUploadFork":
        (10, "Performance/Optimization", "Avoid uploading artifacts on forks"),
    "MissingUploadCondition":
        (11, "Performance/Optimization", r"Use \texttt{if} for upload-artifact action"),
    "UnsafeDeployOnFork":
        (12, "Performance/Optimization", "Avoid deploying jobs on forks"),
    "MissingConcurrencyGlobal":
        (13, "Performance/Optimization", "Avoid starting new workflow while previous one is running"),
    "CorrectIndentation":
        (14, "Other CI/CD", "Correct indentation"),
    "MutableRunnerVersion":
        (15, "Other CI/CD", r"Use fixed version for \texttt{runs-on} argument"),
    "NameRunSteps":
        (16, "Other CI/CD", "Name run steps"),
    "SuboptimalCacheUsage":
        (17, "Other CI/CD", "Use cache parameter instead of cache option"),
    "MultiCommandStep":
        (18, "Other CI/CD", "Use single-command steps"),
    "SingleOSExecution":
        (19, "Other CI/CD", "Run tests on multiple OS's"),
    "UnpinnedDependencies":
        (20, "Other CI/CD", "Specify package versions"),
    "AddCommentsToWorkflows":
        (21, "Other CI/CD", "Add comments to workflows"),
    "SingleLangVersionCI":
        (22, "Other CI/CD", "Run CI on multiple language versions"),
}

SMELL_ORDER = sorted(SMELL_CATALOG.keys(), key=lambda s: SMELL_CATALOG[s][0])
BEHAVIORAL_SMELL_ORDER = [s for s in SMELL_ORDER if s not in PRESENTATIONAL_SMELLS]

_TEXTTT_RE = re.compile(r"\\texttt\{([^}]*)\}")

def _plain_description(latex_name: str) -> str:
    return _TEXTTT_RE.sub(r"\1", latex_name)

# -- Helpers -------------------------------------------------------------------

def read_yaml(file_key: str) -> str | None:
    path = YAML_CACHE_DIR / file_key
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None

def _passes_strict(smells: list[dict]) -> bool:
    """File passes strict if every behavioral smell has instance_count <= 1."""
    for s in smells:
        if s.get("pascal") in PRESENTATIONAL_SMELLS:
            continue
        if s.get("instance_count", 0) > 1:
            return False
    return True

def _build_sort_key(row: dict) -> tuple:
    """Ranking tuple: (total_unique_smells ASC, workflow_run_count DESC, file_key ASC)."""
    return (
        row["total_unique_smells"],
        -(row["workflow_run_count"] or 0),
        row["file_key"],
    )

def _select_loc_balanced(candidates: list[dict], n: int) -> list[dict]:
    """Select N candidates with LOC diversity via quartile sampling.

    Candidates must already be sorted by primary ranking (best first).
    Result is *always* returned in primary-rank order so callers can assign
    priority = 1..len(selected) by enumerate index.
    """
    primary_rank: dict[str, int] = {c["file_key"]: i for i, c in enumerate(candidates)}

    if len(candidates) <= n:
        return list(candidates)

    pool_size = min(LOC_QUARTILE_POOL_MULTIPLIER * n, len(candidates))
    pool = candidates[:pool_size]

    by_loc = sorted(pool, key=lambda c: c["file_line_count"] or 0)
    quartile_size = len(by_loc) // 4
    quartiles = [
        by_loc[:quartile_size],
        by_loc[quartile_size:2 * quartile_size],
        by_loc[2 * quartile_size:3 * quartile_size],
        by_loc[3 * quartile_size:],
    ]

    # Re-sort each quartile by primary ranking so we pick best within each band.
    for q in quartiles:
        q.sort(key=lambda c: primary_rank[c["file_key"]])

    per_quartile = math.ceil(n / 4)
    selected: list[dict] = []
    deficit = 0
    for quartile in quartiles:
        take = min(per_quartile + deficit, len(quartile))
        selected.extend(quartile[:take])
        deficit = (per_quartile + deficit) - take

    selected.sort(key=lambda c: primary_rank[c["file_key"]])
    if len(selected) > n:
        selected = selected[:n]

    return selected

def _partition_file_smells(smells: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split detected smells into (expected_labels, presentational_smells).

    Both lists sorted by paper_id.
    """
    behavioral: list[dict] = []
    presentational: list[dict] = []

    for profile in smells:
        pascal = profile.get("pascal")
        if pascal not in SMELL_CATALOG:
            continue
        paper_id, family, latex_name = SMELL_CATALOG[pascal]
        description = _plain_description(latex_name)
        instance_count = profile.get("instance_count", 0)

        if pascal in PRESENTATIONAL_SMELLS:
            presentational.append({
                "paper_id": paper_id,
                "pascal": pascal,
                "description": description,
                "category": family,
                "instance_count": instance_count,
            })
        else:
            occurrences = profile.get("occurrences") or []
            first = occurrences[0] if occurrences else {}
            behavioral.append({
                "paper_id": paper_id,
                "pascal": pascal,
                "description": description,
                "category": family,
                "instance_count": instance_count,
                "line": first.get("line", ""),
            })

    behavioral.sort(key=lambda d: d["paper_id"])
    presentational.sort(key=lambda d: d["paper_id"])
    return behavioral, presentational

def _build_entry(row: dict, yaml_content: str, smells: list[dict]) -> dict:
    expected_labels, presentational = _partition_file_smells(smells)
    return {
        "file_key": row["file_key"],
        "repository": row["repository"],
        "file_path": row["file_path"],
        "yaml_content": yaml_content,
        "file_line_count": row["file_line_count"],
        "workflow_run_count": row["workflow_run_count"],
        "repo_language": row["repo_language"],
        "total_unique_smells": row["total_unique_smells"],
        "total_behavioral_smells": len(expected_labels),
        "expected_labels": expected_labels,
        "presentational_smells": presentational,
        "evaluation_targets": [],
    }
