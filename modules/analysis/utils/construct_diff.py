"""Construct derivation + diff for two GitHub Actions workflows — schema-free.

A *construct* is an abstracted GHA YAML path: user-chosen keys and list indices
are replaced by placeholders (``<job_id>`` / ``<parameter>`` / ``<matrix_var>`` /
``<env_var>`` / ``<output_id>`` / ``<key>`` / ``[*]``). Each construct is
identified against the **paper's construct catalogue**
(``constructors.csv`` — Talebzadeh et al., "On the GitHub Actions Language"),
which gives its ``feature`` and ``level``. A path not in the catalogue is kept
in the tree as ``catalog=False`` / ``feature='other'`` (transparency), and the
downstream oracle ignores it.

This module does **not** consult the GHA JSON Schema — matching the original
paper, which derived constructs structurally, not from a schema. Schema validity
is a separate concern handled only by :mod:`utils.schema_validity` (the SVR
metric). Given an original workflow and a refactored patch (YAML strings) this
produces: the construct *tree* of each side, the added/removed/changed construct
diff, each construct's catalogue feature/level, and the line-level textual churn.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from utils.dataloader import ROOT

_CATALOG_PATH = ROOT / "data" / "constructs_catalogue.csv"


# ─── Parse (YAML 1.2, so a top-level ``on:`` stays a string key, not ``True``) ─

class _WorkflowYamlLoader(yaml.SafeLoader):
    pass


_WorkflowYamlLoader.yaml_implicit_resolvers = {
    first_char: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first_char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_WorkflowYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def parse_workflow(yaml_content: str) -> tuple[object | None, str | None]:
    """Return (parsed_document, None) or (None, error_message)."""
    try:
        return yaml.load(yaml_content, Loader=_WorkflowYamlLoader), None  # noqa: S506
    except yaml.YAMLError as exc:
        return None, f"YAML syntax error: {exc}"


# ─── Placeholder tokens (by parent key) + path joining ───────────────────────

_PLACEHOLDER = {
    "jobs": "<id>",
    "services": "<id>",
    "with": "<param>",
    "env": "<var>",
    "matrix": "<var>",
    "outputs": "<out>",
}


def _placeholder_for(parent_key: str | None) -> str:
    """The abstraction token for a key whose *parent* map is ``parent_key``."""
    return _PLACEHOLDER.get(parent_key or "", "<key>")


_CONTAINER = object()  # value sentinel for dict/list nodes (no scalar)


def _value_of(value: object) -> object:
    return _CONTAINER if isinstance(value, (dict | list)) else value


def _join(prefix: str, segment: str) -> str:
    """Append a construct/path segment (``[*]`` attaches without a dot)."""
    if segment.startswith("["):
        return f"{prefix}{segment}"
    return f"{prefix}.{segment}" if prefix else segment


def _normalize_on(doc: object) -> object:
    """Normalize the workflow ``on:`` trigger to its mapping form.

    ``on: push`` and ``on: [push, pull_request]`` carry the same triggers as the
    mapping form ``on: {push: …}``; GHA event names are fixed keywords, not
    user-chosen values. Rewriting the scalar/list forms to a map makes the three
    spellings yield the same event constructs so a diff across spellings is not
    spurious churn. Non-mutating.
    """
    if not isinstance(doc, dict) or "on" not in doc:
        return doc
    trigger = doc["on"]
    if isinstance(trigger, str):
        return {**doc, "on": {trigger: None}}
    if isinstance(trigger, list):
        return {**doc, "on": {event: None for event in trigger if isinstance(event, str)}}
    return doc


# ─── Catalogue vocabulary: our placeholders → the paper's tokens ──────────────


def normalize(construct: str) -> str:
    """Rewrite our placeholder vocabulary into the paper catalogue's tokens.

    ``<id>``→``<job_id>``, ``<param>``→``<parameter>``; ``<var>`` splits into
    ``<env_var>`` under ``env`` and ``<matrix_var>`` elsewhere; a matrix
    dimension's list collapses (``matrix.<matrix_var>[*]`` → ``matrix.<matrix_var>``)
    so a dimension reads as one construct regardless of how many values it lists.
    """
    c = construct.replace("<id>", "<job_id>").replace("<param>", "<parameter>")
    c = c.replace("<out>", "<output_id>")
    c = c.replace("matrix.<var>", "matrix.<matrix_var>")
    c = c.replace("env.<var>", "env.<env_var>")
    c = c.replace("<var>", "<matrix_var>")
    c = re.sub(r"\.<matrix_var>\[\*\]", ".<matrix_var>", c)
    return c


# ─── Paper construct catalogue (constructors.csv) ─────────────────────────────

_catalog_cache: dict[str, tuple[str, str]] | None = None


def load_catalog() -> dict[str, tuple[str, str]]:
    """Map each catalogue ``abstract_path`` → (feature, level), parsed once."""
    global _catalog_cache
    if _catalog_cache is None:
        import csv

        out: dict[str, tuple[str, str]] = {}
        with _CATALOG_PATH.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[row["abstract_path"]] = (row["feature"], row["level"])
        _catalog_cache = out
    return _catalog_cache


# The catalogue's 14 features, in the order they appear in constructors.csv, plus
# the catch-all for paths the paper does not catalogue.
CATALOG_FEATURES: tuple[str, ...] = (
    "Action reuse",
    "Workflow reuse",
    "commands",
    "container",
    "context",
    "deployment",
    "environment_variables",
    "job orchestration",
    "matrix_strategy",
    "naming",
    "permissions",
    "services",
    "step orchestration",
    "triggers",
    "other",
)


def classify(construct: str) -> tuple[bool, str, str]:
    """(in_catalogue?, feature, level) for an already-normalized construct.

    A construct absent from the paper catalogue is ``(False, 'other', '')`` — kept
    in the tree for transparency but not part of the studied vocabulary.
    """
    hit = load_catalog().get(construct)
    if hit is None:
        return False, "other", ""
    return True, hit[0], hit[1]


# ─── Abstraction walk (structural — no schema) ────────────────────────────────

# (construct, scalar_value | _CONTAINER)
_Row = tuple[str, object]


def _flatten(doc: object) -> list[_Row]:
    """Walk ``doc`` structurally; one row per node, construct in catalogue vocab.

    A key takes a placeholder when its *parent* map is user-keyed (``jobs``,
    ``with``, ``env``, ``matrix``, ``outputs``, ``services``); otherwise it stays
    literal. List indices collapse to ``[*]``.
    """
    out: list[_Row] = []

    def walk(node: object, construct: str, parent_key: str | None) -> None:
        if isinstance(node, dict):
            for raw_key, value in node.items():
                key = str(raw_key)
                segment = _PLACEHOLDER[parent_key] if parent_key in _PLACEHOLDER else key
                construct_path = normalize(_join(construct, segment))
                out.append((construct_path, _value_of(value)))
                walk(value, _join(construct, segment), key)
        elif isinstance(node, list):
            for value in node:
                construct_path = normalize(_join(construct, "[*]"))
                out.append((construct_path, _value_of(value)))
                walk(value, _join(construct, "[*]"), parent_key)

    walk(_normalize_on(doc), "", None)
    return out


def _index(doc: object) -> dict[str, Counter]:
    """Group a document's rows by construct → scalar-value multiset.

    The pre-normalized construct path is the key; container nodes contribute no
    scalar. The multiset is what the ``changed`` axis compares.
    """
    by_construct: dict[str, Counter] = {}
    for construct, value in _flatten(doc):
        values = by_construct.setdefault(construct, Counter())
        if value is not _CONTAINER:
            values[value] += 1
    return by_construct


# ─── The diff (three axes) ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ConstructDiff:
    added: tuple[str, ...]    # present in patch, not original
    removed: tuple[str, ...]  # present in original, not patch
    changed: tuple[str, ...]  # present both sides, scalar-value multiset differs


def diff_constructs(original_doc: object, patch_doc: object) -> ConstructDiff:
    """The added / removed / changed constructs from ``original`` to ``patch``."""
    original_index = _index(original_doc)
    patch_index = _index(patch_doc)
    added = tuple(sorted(c for c in patch_index if c not in original_index))
    removed = tuple(sorted(c for c in original_index if c not in patch_index))
    changed = tuple(
        sorted(
            c
            for c in (original_index.keys() & patch_index.keys())
            if original_index[c] != patch_index[c]
        )
    )
    return ConstructDiff(added=added, removed=removed, changed=changed)


def evaluate_construct_patch(original_yaml: str, patch_yaml: str) -> ConstructDiff | None:
    """The construct diff for one (original, patch) pair, or ``None`` if either
    side fails to parse (the caller counts it as ``skipped_unparseable``)."""
    orig_doc, oerr = parse_workflow(original_yaml)
    patch_doc, perr = parse_workflow(patch_yaml)
    if oerr is not None or perr is not None:
        return None
    return diff_constructs(orig_doc, patch_doc)


def _record(construct: str) -> dict:
    """One construct with its catalogue membership / feature / level."""
    in_cat, feature, level = classify(construct)
    return {"construct": construct, "catalog": in_cat, "feature": feature, "level": level}


def diff_records(diff: ConstructDiff) -> list[dict]:
    """One record per (axis, construct) for a single patch, with catalogue info."""
    records: list[dict] = []
    for axis, constructs in (("added", diff.added), ("removed", diff.removed), ("changed", diff.changed)):
        for construct in constructs:
            records.append({"axis": axis, **_record(construct)})
    return records


# ─── Construct trees (the per-file construct content, both sides) ─────────────


def construct_trees(original_yaml: str, patch_yaml: str) -> dict | None:
    """The construct *tree* of each side plus the diff, or ``None`` if unparseable.

    Returns ``{original, patched, added, removed, changed}`` — ``original`` and
    ``patched`` are the full sorted construct lists of each workflow (the
    "content is the constructs"), each row carrying catalogue feature/level; the
    three diff axes carry the same per-construct records.
    """
    orig_doc, oerr = parse_workflow(original_yaml)
    patch_doc, perr = parse_workflow(patch_yaml)
    if oerr is not None or perr is not None:
        return None
    original_index = _index(orig_doc)
    patch_index = _index(patch_doc)
    diff = diff_constructs(orig_doc, patch_doc)
    return {
        "original": [_record(c) for c in sorted(original_index)],
        "patched": [_record(c) for c in sorted(patch_index)],
        "added": [_record(c) for c in diff.added],
        "removed": [_record(c) for c in diff.removed],
        "changed": [_record(c) for c in diff.changed],
    }


# ─── Textual diff (line level — companion to the abstracted construct diff) ────

def churn(original_yaml: str, patch_yaml: str) -> tuple[int, int]:
    """(lines added, lines removed) between original and patch."""
    added = removed = 0
    for line in difflib.ndiff(original_yaml.splitlines(), patch_yaml.splitlines()):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return added, removed


def unified_diff(original_yaml: str, patch_yaml: str) -> str:
    """Git-style unified textual diff original→patch (``''`` if identical)."""
    return "\n".join(
        difflib.unified_diff(
            original_yaml.splitlines(),
            patch_yaml.splitlines(),
            fromfile="original",
            tofile="patch",
            lineterm="",
        )
    )


def _selfcheck() -> None:
    # Adding a job-level timeout: one added construct, in the catalogue.
    base = "on: push\njobs:\n  b:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
    patched = (
        "on: push\njobs:\n  b:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n"
        "    steps:\n      - run: echo hi\n"
    )
    diff = evaluate_construct_patch(base, patched)
    assert diff.added == ("jobs.<job_id>.timeout-minutes",), diff.added
    assert diff.removed == () and diff.changed == (), diff

    # The added construct resolves to the catalogue; an unknown key does not.
    assert classify("jobs.<job_id>.timeout-minutes")[0] is True
    assert classify("jobs.<job_id>.bogus_thing") == (False, "other", "")

    # on: spellings collapse to the same trigger constructs.
    a, _ = parse_workflow("on: push\njobs: {}")
    b, _ = parse_workflow("on: [push]\njobs: {}")
    assert set(_index(a)) == set(_index(b)), (set(_index(a)), set(_index(b)))

    # Trees carry both sides; container parents stay out of the catalogue.
    trees = construct_trees(base, patched)
    assert any(r["construct"] == "jobs.<job_id>.timeout-minutes" for r in trees["added"])
    assert {r["construct"] for r in trees["original"]} <= {r["construct"] for r in trees["patched"]}
    print("construct_diff self-check: OK")


if __name__ == "__main__":
    _selfcheck()
