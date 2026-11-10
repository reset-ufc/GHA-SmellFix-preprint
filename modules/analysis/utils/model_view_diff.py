"""Diff-of-diffs audit: did the raters judge the diff the model actually produced?

agno 2.6.9 renders every agent message with ``string.Template.safe_substitute``
before sending it (``Agent.resolve_in_context`` defaults to True and the pipeline
never turns it off; see ``agno/agent/_messages.py::format_message_with_state_variables``).
``safe_substitute`` rewrites the ``$$`` escape to ``$``, so a workflow line such as
``TMPFILE=/tmp/tmp.$$`` reached the model as ``TMPFILE=/tmp/tmp.$``. The traces
record each user turn *before* that rendering, and the saved validation diffs were
taken against the ground-truth YAML. A model that echoed the ``$`` line it was shown
therefore appears, in the diff the raters judged, to have made an edit it never made.

The rendered message was never persisted: the traces hold the pre-render turn, the
run logs are INFO-level (agno logs messages only in debug mode), and OpenAI ran with
``store=False``. It has to be re-rendered here. A replay of every affected call
through agno's own ``Agent.run``, captured at the HTTP request, matched
:func:`agno_render` byte for byte.

For every output this module compares two diffs of the same patched workflow:

* **A** — the saved ``yaml_diffs/{stem}.diff``: patched vs the ground-truth original.
  This is the exact text the raters judged (its sha256 is the ``diff_sha`` stored
  with every verdict).
* **B** — patched vs the original *as the patch producer saw it*: the workflow
  block of the producer call's recorded user turn, rendered the way agno rendered it.

``A == B``: the verdict was given on the right diff. ``A != B``: the raters judged a
diff that misstates what the model did, so the output goes back for re-validation
against B.

The patch producer decides which prompt counts. D1: the combined agent, which always
emits a workflow. D2: the refactor agent when detection said present. When D2
detection said absent, the pipeline writes the ground-truth YAML through verbatim
(``CombinedResultAssembler.from_d2_absent``), so no model touched the patch and B
is taken against the ground truth, like A.

Run ``python -m utils.model_view_diff`` to write ``data/artifacts/revalidation/``.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from pathlib import Path

import pandas as pd

from utils.construct_diff import unified_diff
from utils.dataloader import DATA_ROOT, ROOT

GROUND_TRUTH = ROOT / "data" / "ground_truth_n15.json"
SAVED_DIFFS = ROOT / "data" / "artifacts" / "constructs" / "yaml_diffs"
VERDICTS = ROOT / "data" / "validation" / "manual_validation_final.csv"
OUT = ROOT / "data" / "artifacts" / "revalidation"

# The call whose output became the patched workflow, and the user-turn tag that
# carried the workflow it was given (gha_agents.teams: D1Workflow / D2Workflow).
PRODUCER = {"D1": ("combined", "workflow"), "D2": ("refactor", "original_workflow")}

# agno's only format variables in these runs: the per-run ids it injects into
# session_state (agno/agent/_init.py::_initialize_session_state). The pipeline
# passed no session_state, dependencies, metadata or user_id, and the id values
# were never recorded.
_RUN_VARS = "|".join(("current_session_id", "current_run_id"))
_RUN_VAR_REF = re.compile(rf"\$\{{?(?:{_RUN_VARS})\b|\{{(?:{_RUN_VARS})\}}")

_VERDICT_COLUMNS = [
    "stem",
    "gkey",
    "resolved_verdict",
    "resolved_potential_behavior_impact",
    "resolved_construct_ok",
    "resolution_source",
]


def agno_render(message: str) -> str:
    """``message`` as agno handed it to the model.

    Mirrors ``format_message_with_state_variables`` for these runs. A message that
    references one of agno's run variables cannot be re-rendered, because the
    values were never recorded, so it raises. Any other message renders exactly:
    ``safe_substitute`` leaves an unknown ``$name`` / ``${name}`` and every
    ``${{ }}`` expression alone and only rewrites the ``$$`` escape to ``$``.
    """
    if _RUN_VAR_REF.search(message):
        raise ValueError("message references an agno run variable; its value was not recorded")
    return string.Template(message).safe_substitute({})


def prompt_block(user_turn: str, tag: str) -> str:
    """Body of the ``<tag>`` block in a user turn.

    The pipeline wraps each input as ``<tag>\\n{body}\\n</tag>`` and joins the
    blocks with a blank line (``gha_agents.user_turn.UserTurnBuilder``).
    """
    m = re.search(rf"(?:\A|\n\n)<{tag}>\n(.*?)\n</{tag}>(?=\n\n<|\Z)", user_turn, re.S)
    if m is None:
        raise ValueError(f"no <{tag}> block in the user turn")
    return m.group(1)


def producer_turn(trace: dict) -> tuple[str, str | None]:
    """(producer, its recorded user turn); the turn is None for a D2 passthrough."""
    role, _ = PRODUCER[trace["design"]]
    turns = [c["user_prompt"] for c in trace["calls"] if c["role"] == role]
    if len(turns) > 1:
        raise ValueError(f"{trace['slug']}: {len(turns)} {role} calls")
    if turns:
        return role, turns[0]
    if trace["design"] == "D2":
        return "passthrough", None
    raise ValueError(f"{trace['slug']}: no {role} call")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_output(run_dir: Path, stem: str, originals: dict[str, str]) -> tuple[dict, str]:
    """Compare diff A and diff B for one output; return (row, diff B)."""
    result = json.loads((run_dir / "results" / f"{stem}.json").read_text(encoding="utf-8"))
    trace = json.loads((run_dir / "traces" / f"{stem}.json").read_text(encoding="utf-8"))
    patched = (run_dir / "patched_workflows" / f"{stem}.yml").read_text(encoding="utf-8")
    original = originals[result["file_key"]]

    saved_path = SAVED_DIFFS / f"{stem}.diff"
    saved = saved_path.read_text(encoding="utf-8") if saved_path.is_file() else None
    # The diff writers may append one newline; a unified diff never ends with one.
    diff_a = "" if saved is None else saved.removesuffix("\n")

    producer, turn = producer_turn(trace)
    if turn is None:
        shown = seen = original
    else:
        tag = PRODUCER[result["design"]][1]
        shown = prompt_block(turn, tag)
        seen = prompt_block(agno_render(turn), tag)
    diff_b = unified_diff(seen, patched)

    row = {
        "stem": stem,
        "file_key": result["file_key"],
        "smell": result["target_smell"],
        "design": result["design"],
        "model": result["model_alias"],
        "rep": int(result["repetition"]),
        "gt_positive": bool(result["ground_truth"]),
        "smell_detected": bool(result["smell_detected"]),
        "producer": producer,
        "prompt_matches_gt": shown == original,
        "render_changed_lines": sum(
            a != b for a, b in zip(shown.splitlines(), seen.splitlines(), strict=True)
        ),
        "saved_diff_exists": saved is not None,
        "saved_matches_gt": diff_a == unified_diff(original, patched),
        # Same convention as the validation app, so this joins on a verdict's diff_sha.
        "diff_sha": _sha(saved) if saved is not None else _sha(f"__missing__{stem}"),
        "model_view_diff_sha": _sha(f"{diff_b}\n") if diff_b else "",
        "model_view_diff_empty": not diff_b,
        "same_diff": diff_a == diff_b,
    }
    return row, diff_b


def build() -> tuple[pd.DataFrame, dict[str, str]]:
    """Check every output with a saved diff or a verdict.

    Returns one row per output plus diff B for each output where A != B.
    """
    raw = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    originals = {e["file_key"]: e["yaml_content"] for e in raw}
    verdicts = pd.read_csv(VERDICTS, usecols=_VERDICT_COLUMNS, dtype=str)
    run_dirs = {p.stem: p.parent.parent for p in DATA_ROOT.glob("*/design_d*/results/*.json")}

    stems = sorted({p.stem for p in SAVED_DIFFS.glob("*.diff")} | set(verdicts["stem"]))
    missing = [s for s in stems if s not in run_dirs]
    if missing:
        raise FileNotFoundError(f"{len(missing)} outputs have no result JSON, e.g. {missing[:3]}")

    rows, model_view_diffs = [], {}
    for stem in stems:
        row, diff_b = check_output(run_dirs[stem], stem, originals)
        rows.append(row)
        if not row["same_diff"]:
            model_view_diffs[stem] = diff_b

    checks = pd.DataFrame(rows).merge(verdicts, on="stem", how="left", validate="one_to_one")
    checks.insert(checks.columns.get_loc("same_diff"), "validated", checks["gkey"].notna())
    return checks, model_view_diffs


def summarize(checks: pd.DataFrame) -> pd.DataFrame:
    """Headline counts; the integrity rows are expected to be 0."""
    validated = checks[checks["validated"]]
    wrong = validated[~validated["same_diff"]]
    metrics = {
        "outputs_checked": len(checks),
        "outputs_validated": len(validated),
        "integrity_prompt_differs_from_gt": int((~checks["prompt_matches_gt"]).sum()),
        "integrity_saved_diff_differs_from_gt": int((~checks["saved_matches_gt"]).sum()),
        "outputs_rendered_differently": int((checks["render_changed_lines"] > 0).sum()),
        "diff_mismatch_all": int((~checks["same_diff"]).sum()),
        "diff_mismatch_validated": len(wrong),
        "diff_mismatch_validated_distinct_saved_diffs": wrong["diff_sha"].nunique(),
        "diff_mismatch_validated_distinct_groups": wrong["gkey"].nunique(),
        "diff_mismatch_validated_distinct_model_view_diffs": wrong["model_view_diff_sha"].nunique(),
        "diff_mismatch_validated_model_view_diff_empty": int(wrong["model_view_diff_empty"].sum()),
    }
    return pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})


def write(out: Path = OUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the per-output check, the summary and every non-empty diff B."""
    checks, model_view_diffs = build()
    summary = summarize(checks)
    diff_dir = out / "yaml_diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)
    for stale in diff_dir.glob("*.diff"):
        stale.unlink()
    for stem, text in model_view_diffs.items():
        if text:  # same layout as constructs._write_yaml_diffs; an empty diff gets no file
            (diff_dir / f"{stem}.diff").write_text(f"{text}\n", encoding="utf-8")
    checks.to_csv(out / "model_view_diff_check.csv", index=False)
    summary.to_csv(out / "model_view_diff_summary.csv", index=False)
    return checks, summary


def _selfcheck() -> None:
    # The $$ escape collapses; shell variables and GHA expressions pass through.
    assert agno_render("TMPFILE=/tmp/tmp.$$") == "TMPFILE=/tmp/tmp.$"
    kept = 'echo "$HOME ${VAR} ${{ github.sha }} $1 $"'
    assert agno_render(kept) == kept
    for ref in ("$current_run_id", "${current_session_id}", "{current_run_id}"):
        try:
            agno_render(ref)
        except ValueError:
            continue
        raise AssertionError(f"{ref!r} should be refused")
    assert agno_render("$current_run_ids") == "$current_run_ids"

    # The workflow block round-trips through the pipeline's wrapping.
    body = "on: push\njobs:\n  a:\n    steps:\n      - run: echo $$"
    turn = f"<workflow>\n{body}\n</workflow>\n\n<repository_context>\n- x: y\n</repository_context>"
    assert prompt_block(turn, "workflow") == body
    assert prompt_block(agno_render(turn), "workflow") == body.replace("$$", "$")

    # A model that echoes the $ line: A shows a phantom edit, B shows only the real one.
    original = "on: push\njobs:\n  a:\n    steps:\n      - run: echo $$\n"
    seen = agno_render(original)
    patched = seen.replace("on: push\n", "on: push\npermissions: {}\n")
    diff_a, diff_b = unified_diff(original, patched), unified_diff(seen, patched)
    changed_a = [ln for ln in diff_a.splitlines()[2:] if ln[:1] in "+-"]
    changed_b = [ln for ln in diff_b.splitlines()[2:] if ln[:1] in "+-"]
    phantom = ["-      - run: echo $$", "+      - run: echo $"]
    assert changed_a == ["+permissions: {}", *phantom], changed_a
    assert changed_b == ["+permissions: {}"], changed_b
    print("model_view_diff self-check: OK")


def main() -> None:
    _selfcheck()
    checks, summary = write()
    print(summary.to_string(index=False))
    wrong = checks[checks["validated"] & ~checks["same_diff"]]
    print(
        f"\nwrote {OUT}: {len(checks):,} outputs checked, {len(wrong):,} validated to re-validate"
    )


if __name__ == "__main__":
    main()
