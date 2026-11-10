"""Semi-synthetic UnsafeDeployOnFork: GHA sanity check and exclusion sensitivity.

Robustness diagnostic behind the paper's *Semi-synthetic UnsafeDeployOnFork* paragraph
(section *Robustness and Diagnostic Analyses*). Natural positives of UnsafeDeployOnFork
were scarce, so the dataset pipeline had an LLM mutate real workflows that the
annotators had labeled negative for the smell (Step 12d), and the annotators then
labeled each mutated workflow (Step 12e). A semi-synthetic pair has ``##synth`` at the
end of its ``file_key``; its base workflow has the same key without the suffix.

Two checks:

- **GHA sanity check.** Whether GHA flags UnsafeDeployOnFork on each base workflow and
  on its mutated version. GHA is the repository's detector (``modules/gha_detector``),
  run in process with the calls of its batch CLI (``cli._analyze_workflow_file``), the
  path that produced the candidate pool's GHA output. The base YAML comes from the
  final ground truth, where every base appears as a real workflow. The check also looks
  up each base as a candidate for the smell in the manual labels
  (``utils.gha_retrieval.load_labels``): the paper says every base is, by selection, a
  negative of GHA's unflagged bucket.
- **Exclusion sensitivity.** The paper's values recomputed without the smell's pairs of
  the n15 ground truth (15 positive, 10 of them semi-synthetic, and 15 negative), with
  the loaders and rules of the tables they come from:

  * detection F1 per model and design (``utils.detection``, as in the overall
    detection table);
  * end-to-end Clean Removal per model and design: the scored positive runs of
    ``rq2_end_to_end_conditional``;
  * Unnecessary Edit per model and design: the negative rule of
    ``rq2_refactoring_outcomes_by_smell``, where an ``uncertain`` verdict is dropped;
  * the D1--D2 tests of ``rq3_paired_tests_across_designs``, per model and
    outcome, with the Holm family recomputed over the four models of the reduced set.
    The effect is the paper's (``paired_effects`` of that table): the D2 - D1 rate
    difference for the two binary outcomes, and the median of the paired D2 - D1
    differences for latency (s) and estimated cost (x1,000 USD). A binary outcome also
    gets its discordant pairs.

The paragraph reports the F1 and Clean Removal changes and every change of
significance. Unnecessary Edit (measured on negatives, all natural for this smell) and
the latency and cost effects (which do not depend on labels) stay in the CSVs only.

Run ``python -m utils.semi_synthetic`` for a self-check.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from gha_ci_detector import util as gha_util
from gha_ci_detector.cli import _validate_workflow_contract
from gha_ci_detector.Runner import Runner
from gha_ci_detector.Workflow import Workflow

from utils import detection
from utils.dataloader import ROOT
from utils.designs import DESIGNS
from utils.gha_retrieval import AGREED_LABELS, OVERTURNED_LABELS, load_labels
from utils.tables import rq3_paired_tests_across_designs as paired_tests
from utils.tables.rq2_end_to_end_conditional import _scored_positive_runs
from utils.tables.rq2_refactoring_outcomes_by_smell import (
    _BEHAVIOR_INDEX,
    _MANUAL_FINAL,
    _load_joined,
)

SMELL = "UnsafeDeployOnFork"
SYNTH_SUFFIX = "##synth"
MUTATED = ROOT / "data" / "12d_synth_UnsafeDeployOnFork_present.json"
MUTATED_LABELS = ROOT / "data" / "12e_sinth_manual_evaluation.csv"
BASES = ROOT / "data" / "13_final_ground_truth.json"

MODELS = [alias for _, alias in paired_tests._MODELS]
UNIT = paired_tests._UNIT
ALPHA = paired_tests._ALPHA
RATE_METRICS = ["f1", "clean_removal", "unnecessary_edit"]

GHA_CHECK_COLUMNS = [
    "file_key",
    "base_file_key",
    "label",
    "base_bucket",
    "base_label",
    "in_n15",
    "base_n15_negative",
    "base_flagged",
    "mutated_flagged",
]


def flags_smell(yaml_content: str, name: str, smell: str = SMELL) -> bool:
    """Whether GHA reports ``smell`` on a workflow: the calls of the batch CLI, without
    its retries. An invalid workflow raises instead of counting as unflagged."""
    workflow = Workflow(yaml_content, name)
    _validate_workflow_contract(workflow)
    smells = Runner(workflow).run_all()
    rows = gha_util.smells_to_csv_rows(workflow.name, workflow.file_content, smells)
    return smell in {row[2] for row in rows}


def _smell_pairs(entries: list[dict], field: str, smell: str = SMELL) -> set[str]:
    """``file_key`` of the entries that list ``smell`` under ``field``."""
    return {e["file_key"] for e in entries if any(s["pascal"] == smell for s in e[field])}


def gha_check(
    mutated: list[dict],
    labels: pd.DataFrame,
    bases: dict[str, str],
    n15: list[dict],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """One row per mutated workflow: its base, the annotators' label, the base's sampling
    bucket (``tool_output``) and manual label as a candidate for the smell in
    ``candidates`` (empty if it was no candidate), whether the pair is in the n15 ground
    truth (and its base is an n15 negative of the smell), and GHA's flag on the base and
    on the mutated workflow.

    ``ValueError`` if a key lacks the suffix or a base, if the labels do not cover the
    mutated workflows one to one, or if an n15 semi-synthetic pair is not an accepted
    mutated workflow.
    """
    keys = [m["file_key"] for m in mutated]
    if len(set(keys)) != len(keys) or not all(k.endswith(SYNTH_SUFFIX) for k in keys):
        raise ValueError(f"mutated keys must be unique and end with {SYNTH_SUFFIX!r}")
    if set(labels["target_smell"]) != {SMELL}:
        raise ValueError(f"labels must all target {SMELL}")
    if labels["file_key"].duplicated().any() or set(labels["file_key"]) != set(keys):
        raise ValueError("labels must cover the mutated workflows one to one")
    label = labels.set_index("file_key")["final_decision"]
    if not set(label).issubset({"present", "absent"}):
        raise ValueError(f"unexpected labels: {sorted(set(label))}")
    missing = sorted({k.removesuffix(SYNTH_SUFFIX) for k in keys} - set(bases))
    if missing:
        raise ValueError(f"no base YAML for {missing}")

    n15_positive = _smell_pairs(n15, "present_smells")
    n15_negative = _smell_pairs(n15, "absent_smells")
    n15_synth = {k for k in n15_positive if k.endswith(SYNTH_SUFFIX)}
    if not n15_synth.issubset({k for k in keys if label[k] == "present"}):
        raise ValueError("an n15 semi-synthetic pair is not an accepted mutated workflow")

    as_candidate = (candidates[candidates["target_smell"] == SMELL]
                    .set_index("file_key")[["tool_output", "final_decision"]])
    records = []
    for m in mutated:
        key = m["file_key"]
        base = key.removesuffix(SYNTH_SUFFIX)
        bucket, base_label = (as_candidate.loc[base].tolist() if base in as_candidate.index
                              else ("", ""))
        records.append({
            "file_key": key,
            "base_file_key": base,
            "label": label[key],
            "base_bucket": bucket,
            "base_label": base_label,
            "in_n15": key in n15_synth,
            "base_n15_negative": base in n15_negative,
            "base_flagged": flags_smell(bases[base], base),
            "mutated_flagged": flags_smell(m["yaml_content"], key),
        })
    return pd.DataFrame.from_records(records, columns=GHA_CHECK_COLUMNS)


def summarize_gha_check(check: pd.DataFrame) -> dict:
    """Counts of the GHA sanity check, over every mutated workflow and over the n15 ones.
    ``bases_unflagged_negatives`` counts the bases labeled ``absent`` in GHA's unflagged
    bucket."""
    accepted = check["label"] == "present"
    n15 = check["in_n15"]
    unflagged_negative = (check["base_bucket"] == "negative") & (check["base_label"] == "absent")
    return {
        "generated": len(check),
        "accepted": int(accepted.sum()),
        "rejected": int((~accepted).sum()),
        "bases_unflagged_negatives": int(unflagged_negative.sum()),
        "bases_flagged": int(check["base_flagged"].sum()),
        "mutated_flagged": int(check["mutated_flagged"].sum()),
        "accepted_mutated_flagged": int(check.loc[accepted, "mutated_flagged"].sum()),
        "rejected_mutated_flagged": int(check.loc[~accepted, "mutated_flagged"].sum()),
        "n15": {
            "semi_synthetic": int(n15.sum()),
            "bases_flagged": int(check.loc[n15, "base_flagged"].sum()),
            "mutated_flagged": int(check.loc[n15, "mutated_flagged"].sum()),
            "bases_are_n15_negatives": int(check.loc[n15, "base_n15_negative"].sum()),
        },
    }


def _rates(detection_runs: pd.DataFrame, positive_runs: pd.DataFrame,
           negative_runs: pd.DataFrame) -> pd.DataFrame:
    """F1, end-to-end Clean Removal and Unnecessary Edit, in %, per model and
    design, with the number of scored pairs or assessable outputs."""
    records = [
        {"metric": "f1", "model": r.model_alias, "design": r.design,
         "n": int(r.n), "pct": 100 * r.f1}
        for r in detection.metrics_by(detection_runs, "model_alias", "design").itertuples()
    ]
    for metric, runs, hit in [
        ("clean_removal", positive_runs, positive_runs["cat"] == "CR"),
        ("unnecessary_edit", negative_runs, negative_runs["resolved_verdict"] == "unnecessary"),
    ]:
        for (model, design), rows in runs.groupby(["model", "design"]):
            records.append({"metric": metric, "model": model, "design": design,
                            "n": len(rows), "pct": 100 * hit[rows.index].mean()})
    order = {"metric": RATE_METRICS, "model": MODELS, "design": list(DESIGNS)}
    return pd.DataFrame.from_records(records).sort_values(
        list(order), key=lambda column: column.map(order[column.name].index), ignore_index=True
    )


def _tests(outcomes: dict[str, tuple[pd.DataFrame, bool]]) -> pd.DataFrame:
    """The paired D1--D2 test of every outcome and model: pairs, effect and Holm p, plus
    the discordant pairs of a binary outcome (empty for the others)."""
    records = []
    for outcome, (values, binary) in outcomes.items():
        tests = paired_tests._outcome_tests(values, binary)
        effects = paired_tests.paired_effects(values, binary)
        for alias in MODELS:
            record = {"outcome": outcome, "model": alias,
                      "pairs": int(effects.at[alias, "pairs"]),
                      "effect": float(effects.at[alias, "effect"]),
                      "p_holm": float(tests.at[alias, "p_holm"])}
            if binary:
                record |= {column: int(effects.at[alias, column])
                           for column in ("d1_only", "d2_only")}
            records.append(record)
    return pd.DataFrame.from_records(records).astype({"d1_only": "Int64", "d2_only": "Int64"})


def _without_smell(frame: pd.DataFrame, column: str, smell: str) -> pd.DataFrame:
    return frame[frame[column] != smell]


def compare(full: pd.DataFrame, excluded: pd.DataFrame, keys: list[str],
            value: str) -> pd.DataFrame:
    """``full`` beside ``excluded`` on ``keys``, in the row order of ``full``, with
    ``delta`` = excluded - full of ``value``; ``ValueError`` if the two sides do not
    have the same keys."""
    if set(full[keys].itertuples(index=False)) != set(excluded[keys].itertuples(index=False)):
        raise ValueError("full and excluded values do not have the same keys")
    merged = full.merge(excluded, on=keys, how="left", suffixes=("_full", "_excluded"),
                        validate="one_to_one")
    merged["delta"] = merged[f"{value}_excluded"] - merged[f"{value}_full"]
    return merged


def exclusion_sensitivity(smell: str = SMELL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(rates, tests)``, each with the values on the full n15 set and without
    ``smell``: the per-model and per-design rates (``delta`` in percentage
    points), and the per-model paired tests (``delta`` of the effect, and each side's
    significance)."""
    detection_runs = detection.add_n15_label(detection.load_results())
    detection_runs = detection_runs[detection_runs["model_alias"].isin(MODELS)]
    positive_runs = _scored_positive_runs()
    joined = _load_joined()
    negative_runs = joined[(joined["population"] == "negative")
                           & (joined["resolved_verdict"] != "uncertain")]
    latency, cost = paired_tests._overhead_values()
    outcomes = {
        "detection": (paired_tests._detection_values(), True),
        "clean_removal": (paired_tests._clean_removal_values(), True),
        "latency_s": (latency, False),
        "cost_1k_usd": (cost, False),
    }

    rates = compare(
        _rates(detection_runs, positive_runs, negative_runs),
        _rates(_without_smell(detection_runs, "target_smell", smell),
               _without_smell(positive_runs, "smell", smell),
               _without_smell(negative_runs, "smell", smell)),
        ["metric", "model", "design"], "pct",
    )
    tests = compare(
        _tests(outcomes),
        _tests({name: (_without_smell(values, "smell", smell), binary)
                for name, (values, binary) in outcomes.items()}),
        ["outcome", "model"], "effect",
    )
    for side in ("full", "excluded"):
        tests[f"significant_{side}"] = tests[f"p_holm_{side}"] < ALPHA
    return rates, tests


def _largest(frame: pd.DataFrame, group: str, keys: list[str]) -> dict:
    """Per ``group`` value, the largest absolute ``delta`` and the ``keys`` where it is."""
    largest = {}
    for name, rows in frame.groupby(group, sort=False):
        top = rows.loc[rows["delta"].abs().idxmax()]
        largest[name] = {"max_abs_delta": round(abs(float(top["delta"])), 4)}
        largest[name] |= {key: top[key] for key in keys}
    return largest


def smell_pair_counts(n15: list[dict], smell: str = SMELL) -> dict:
    """The smell's pairs in the n15 ground truth: positive, semi-synthetic among them,
    and negative."""
    positive = _smell_pairs(n15, "present_smells", smell)
    return {
        "positive": len(positive),
        "semi_synthetic": sum(key.endswith(SYNTH_SUFFIX) for key in positive),
        "negative": len(_smell_pairs(n15, "absent_smells", smell)),
    }


def _significance_change(test) -> dict:
    """One test whose significance changes: its Holm p, effect and, for a binary
    outcome, discordant pairs on each side."""
    change = {"outcome": test.outcome, "model": test.model}
    for column in ("p_holm", "effect", "d1_only", "d2_only"):
        for side in ("full", "excluded"):
            value = getattr(test, f"{column}_{side}")
            if not pd.isna(value):
                change[f"{column}_{side}"] = round(float(value), 4) if column in (
                    "p_holm", "effect") else int(value)
    return change


def summarize_exclusion(rates: pd.DataFrame, tests: pd.DataFrame, smell_pairs: dict,
                        smell: str = SMELL) -> dict:
    """The pairs on each side and the smell's pairs, the largest change of each rate
    (percentage points) and of each test's effect (its own unit), and every test whose
    significance changes. ``ValueError`` if the pairs left out are not the smell's."""
    pairs = rates[rates["metric"] == "f1"]
    full, excluded = int(pairs["n_full"].max()), int(pairs["n_excluded"].max())
    if full - excluded != smell_pairs["positive"] + smell_pairs["negative"]:
        raise ValueError(f"{full - excluded} pairs left out, but {smell} has {smell_pairs}")
    changed = tests[tests["significant_full"] != tests["significant_excluded"]]
    return {
        "smell": smell,
        "pairs": {"full": full, "excluded": excluded},
        "smell_pairs": smell_pairs,
        "rates_pp": _largest(rates, "metric", ["model", "design"]),
        "effects": _largest(tests, "outcome", ["model"]),
        "significance_changes": [_significance_change(t) for t in changed.itertuples()],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def semi_synthetic_report(
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Write ``semi_synthetic_gha_check.csv``, ``semi_synthetic_exclusion_rates.csv``,
    ``semi_synthetic_exclusion_tests.csv`` and ``semi_synthetic.json`` to ``out_dir``.

    The JSON holds both summaries plus the path and SHA-256 of each input file; the
    runs themselves are read by the tables' loaders.
    """
    labels = pd.read_csv(MUTATED_LABELS, dtype=str, keep_default_na=False)
    bases = {e["file_key"]: e["yaml_content"] for e in _read_json(BASES)}
    n15 = _read_json(detection.N15_GT)
    check = gha_check(_read_json(MUTATED), labels, bases, n15, load_labels())
    rates, tests = exclusion_sensitivity()
    summary = {
        "gha_check": summarize_gha_check(check),
        "exclusion": summarize_exclusion(rates, tests, smell_pair_counts(n15)),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("gha_check", check), ("exclusion_rates", rates),
                        ("exclusion_tests", tests)]:
        frame.to_csv(out_dir / f"semi_synthetic_{name}.csv", index=False, lineterminator="\n")
    report = summary | {
        "inputs": [
            {"path": os.path.relpath(path, ROOT), "sha256": _sha256(path)}
            for path in (MUTATED, MUTATED_LABELS, BASES, AGREED_LABELS, OVERTURNED_LABELS,
                         detection.N15_GT, _BEHAVIOR_INDEX, _MANUAL_FINAL)
        ],
    }
    (out_dir / "semi_synthetic.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return check, rates, tests, summary


def _selfcheck() -> None:
    # GHA flags a job without a repository guard, and only that.
    guarded = "on: push\njobs:\n  a:\n    if: github.repository == 'o/r'\n" \
              "    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
    unguarded = guarded + "  b:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
    assert not flags_smell(guarded, "o@r@ci.yml")
    assert flags_smell(unguarded, "o@r@ci.yml")

    def entry(key, present=(), absent=()):
        return {"file_key": key, "present_smells": [{"pascal": s} for s in present],
                "absent_smells": [{"pascal": s} for s in absent]}

    mutated = [{"file_key": f"o@r@{n}.yml{SYNTH_SUFFIX}", "yaml_content": unguarded}
               for n in ("a", "b")]
    labels = pd.DataFrame({"file_key": [m["file_key"] for m in mutated],
                           "target_smell": SMELL, "final_decision": ["present", "absent"]})
    bases = {"o@r@a.yml": guarded, "o@r@b.yml": guarded}
    n15 = [entry(mutated[0]["file_key"], present=[SMELL]), entry("o@r@a.yml", absent=[SMELL])]
    # Base a is a GHA-unflagged negative of the smell; base b is no candidate for it.
    candidates = pd.DataFrame({"file_key": ["o@r@a.yml", "o@r@b.yml"],
                               "target_smell": [SMELL, "OtherSmell"],
                               "tool_output": ["negative", "negative"],
                               "final_decision": ["absent", "absent"]})
    check = gha_check(mutated, labels, bases, n15, candidates)
    assert check["base_bucket"].tolist() == ["negative", ""], check
    assert summarize_gha_check(check) == {
        "generated": 2, "accepted": 1, "rejected": 1, "bases_unflagged_negatives": 1,
        "bases_flagged": 0, "mutated_flagged": 2, "accepted_mutated_flagged": 1,
        "rejected_mutated_flagged": 1,
        "n15": {"semi_synthetic": 1, "bases_flagged": 0, "mutated_flagged": 1,
                "bases_are_n15_negatives": 1},
    }, check
    broken = [
        (mutated, labels.iloc[:1], bases, n15),  # a mutated workflow without a label
        (mutated, labels, {"o@r@a.yml": guarded}, n15),  # a base without YAML
        (mutated, labels, bases, [entry(mutated[1]["file_key"], present=[SMELL])]),  # rejected
    ]
    for args in broken:
        try:
            gha_check(*args, candidates)
        except ValueError:
            continue
        raise AssertionError("gha_check accepted a broken input")

    # Effects: D2 - D1 rate difference, or median paired difference; unpaired rows drop.
    rows = [("m", d, f"k{i}", "S", v) for i, (d1, d2) in enumerate([(0, 1), (0, 1), (1, 1), (1, 0)])
            for d, v in (("D1", d1), ("D2", d2))] + [("m", "D1", "k9", "S", 5.0)]
    values = pd.DataFrame(rows, columns=["model_alias", "design", *UNIT, "value"])
    # Pairs, effect, then the discordant pairs: 1 with a 1 only under D1, 2 only under D2.
    effects = paired_tests.paired_effects
    assert effects(values, binary=True).loc["m"].tolist() == [4, 0.25, 1, 2]
    values.loc[values["design"] == "D2", "value"] += [1, 2, 3, 9]
    assert effects(values, binary=False).loc["m"].tolist() == [4, 3.0]

    # The smell's pairs; the pairs left out must be exactly those.
    counts = smell_pair_counts(n15)
    assert counts == {"positive": 1, "semi_synthetic": 1, "negative": 1}, counts
    rates = pd.DataFrame({"metric": ["f1"], "n_full": [10], "n_excluded": [7]})
    try:
        summarize_exclusion(rates, pd.DataFrame(), counts)
    except ValueError:
        pass
    else:
        raise AssertionError("summarize_exclusion accepted pairs that are not the smell's")

    # Deltas are excluded - full; the two sides must share keys; the largest is absolute.
    full = pd.DataFrame({"metric": ["f1", "f1"], "model": ["a", "b"], "pct": [80.0, 90.0]})
    delta = compare(full, full.assign(pct=[80.5, 89.0]), ["metric", "model"], "pct")
    assert delta["delta"].tolist() == [0.5, -1.0], delta
    assert _largest(delta, "metric", ["model"]) == {"f1": {"max_abs_delta": 1.0, "model": "b"}}
    try:
        compare(full, full.iloc[:1], ["metric", "model"], "pct")
    except ValueError:
        pass
    else:
        raise AssertionError("compare accepted mismatched keys")
    print("semi_synthetic self-check: OK")


if __name__ == "__main__":
    _selfcheck()
