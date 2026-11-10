"""GHA-schema validity of a workflow document — the SVR instrument ONLY.

This is the *one* place the project still consults the official GitHub Actions
JSON Schema (``data/github-workflow.json``). It answers a single question:
does a patched workflow introduce a key the schema rejects? — which feeds the
RQ2 **SVR** (Schema Validity Rate) safety metric.

It is deliberately decoupled from construct *derivation*: the construct tree /
diff (``construct_diff.py``) abstracts YAML structurally and identifies
constructs against the paper's catalogue (``constructors.csv``), with NO schema
— matching the original paper's method. The schema lives here, and only here, so
"the YAML is valid against the GHA schema" stays a real metric while the
construct vocabulary is the paper's, not the schema's.

Validity is read off the schema by a parallel walk of (document × schema). It is
a *lenient* (union-over-branches) verdict — a key is valid if accepted by ANY
``oneOf``/``anyOf``/``allOf`` branch; cross-branch combinations are not flagged.
On the n15 corpus this never produced a false-valid (audited), so the leniency is
immaterial here; if strictness ever matters, evaluate ``allOf`` as intersection.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from utils.construct_diff import (
    _join,
    _normalize_on,
    _placeholder_for,
    parse_workflow,
)
from utils.dataloader import ROOT

_SCHEMA_PATH = ROOT / "data" / "github-workflow.json"

_schema_root_cache: dict | None = None


def schema_root() -> dict:
    """The vendored GHA JSON Schema as a plain dict (parsed once)."""
    global _schema_root_cache
    if _schema_root_cache is None:
        _schema_root_cache = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _schema_root_cache


# ─── Schema position: the keys/patterns/items allowed at a node ──────────────


@dataclass(frozen=True)
class Allowed:
    """What a schema node permits at the next document level."""

    properties: dict[str, dict]
    patterns: tuple[tuple[str, dict], ...]
    additional: dict | bool
    items: dict | None


_EMPTY_ALLOWED = Allowed({}, (), True, None)


def _resolve_ref(node: dict, root: dict) -> dict:
    """Follow local ``#/definitions/<name>`` refs to their node (cycle-guarded)."""
    seen: set[str] = set()
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            return {}
        seen.add(ref)
        target: object = root
        for part in ref[2:].split("/"):
            target = target.get(part, {}) if isinstance(target, dict) else {}
        node = target  # type: ignore[assignment]
    return node if isinstance(node, dict) else {}


def _branches(node: dict, root: dict, seen: set[int]) -> Iterator[dict]:
    """Yield ``node`` and every schema reachable through its
    ``oneOf``/``anyOf``/``allOf`` branches (refs resolved, cycle-guarded)."""
    node = _resolve_ref(node, root)
    node_id = id(node)
    if not node or node_id in seen:
        return
    seen.add(node_id)
    yield node
    for combinator in ("oneOf", "anyOf", "allOf"):
        for branch in node.get(combinator, ()):
            yield from _branches(branch, root, seen)


def _resolve_additional(declarations: list[object]) -> dict | bool:
    """Union ``additionalProperties`` declarations across branches."""
    open_schema: dict | None = None
    has_false = False
    for declaration in declarations:
        if declaration is True:
            return True
        if declaration is False:
            has_false = True
        elif isinstance(declaration, dict):
            open_schema = open_schema or declaration
    if open_schema is not None:
        return open_schema
    return not has_false


_allowed_cache: dict[int, Allowed] = {}


def _allowed_at(node: dict, root: dict) -> Allowed:
    """The :class:`Allowed` for a schema node, unioned over its branches."""
    node = _resolve_ref(node, root)
    cached = _allowed_cache.get(id(node))
    if cached is not None:
        return cached

    properties: dict[str, dict] = {}
    patterns: list[tuple[str, dict]] = []
    declarations: list[object] = []
    items: dict | None = None
    for branch in _branches(node, root, set()):
        for key, subschema in branch.get("properties", {}).items():
            properties.setdefault(key, subschema)
        for pattern, subschema in branch.get("patternProperties", {}).items():
            patterns.append((pattern, subschema))
        if "additionalProperties" in branch:
            declarations.append(branch["additionalProperties"])
        if items is None and "items" in branch:
            items = branch["items"]
    allowed = Allowed(properties, tuple(patterns), _resolve_additional(declarations), items)
    _allowed_cache[id(node)] = allowed
    return allowed


def _match_pattern(patterns: tuple[tuple[str, dict], ...], key: str) -> dict | None:
    """The first ``patternProperties`` subschema whose regex matches ``key``."""
    for pattern, subschema in patterns:
        try:
            if re.search(pattern, key):
                return subschema
        except re.error:
            continue
    return None


# ─── The validity walk (abstracted construct -> schema-valid?) ────────────────

# (abstracted construct, valid)
_ValidRow = tuple[str, bool]


def _flatten_valid(doc: object, root: dict) -> list[_ValidRow]:
    """Walk ``doc`` against the schema ``root``; one (construct, valid) per node.

    ``valid`` is the AND of "every segment up to this node matched the schema".
    Descendants of a schema-rejected key are noise and are not emitted. The
    abstraction mirrors ``construct_diff``'s placeholders so the construct strings
    line up across the two modules.
    """
    out: list[_ValidRow] = []
    _walk(_normalize_on(doc), _allowed_at(root, root), "", True, None, root, out)
    return out


def _walk(
    node: object,
    allowed: Allowed,
    construct: str,
    valid: bool,
    parent_key: str | None,
    root: dict,
    out: list[_ValidRow],
) -> None:
    if isinstance(node, dict):
        for raw_key, value in node.items():
            key = str(raw_key)
            segment, child_schema, child_valid, recurse = _classify(key, allowed, valid, parent_key)
            construct_path = _join(construct, segment)
            out.append((construct_path, child_valid))
            if recurse:
                child_allowed = _allowed_at(child_schema, root) if child_schema else _EMPTY_ALLOWED
                _walk(value, child_allowed, construct_path, child_valid, key, root, out)
    elif isinstance(node, list):
        item_allowed = _allowed_at(allowed.items, root) if allowed.items else _EMPTY_ALLOWED
        for value in node:
            construct_path = _join(construct, "[*]")
            out.append((construct_path, valid))
            _walk(value, item_allowed, construct_path, valid, parent_key, root, out)


def _classify(
    key: str, allowed: Allowed, valid: bool, parent_key: str | None
) -> tuple[str, dict | None, bool, bool]:
    """Map a document key to (segment, child_schema, child_valid, recurse?)."""
    if key in allowed.properties:
        return key, allowed.properties[key], valid, True
    subschema = _match_pattern(allowed.patterns, key)
    if subschema is not None:
        return _placeholder_for(parent_key), subschema, valid, True
    if allowed.additional is True:
        return key, None, valid, True  # schema-open, no child schema (eventObject keywords)
    if isinstance(allowed.additional, dict):
        return _placeholder_for(parent_key), allowed.additional, valid, True
    return key, None, False, False  # additionalProperties:false, no match → invalid


def _valid_index(doc: object, root: dict) -> dict[str, bool]:
    """Construct -> schema-validity (AND-reduced over all its occurrences)."""
    index: dict[str, bool] = {}
    for construct, valid in _flatten_valid(doc, root):
        index[construct] = index.get(construct, True) and valid
    return index


def invalid_added(original_yaml: str, patch_yaml: str) -> int | None:
    """How many constructs the patch ADDS that the GHA schema rejects.

    ``None`` if either side fails to parse (the caller counts it unparseable).
    This is the SVR signal: ``svr = parseable and invalid_added == 0``. Only
    *added* constructs are scored, so a pre-existing invalid key the patch leaves
    untouched is not charged to the model.
    """
    orig_doc, oerr = parse_workflow(original_yaml)
    patch_doc, perr = parse_workflow(patch_yaml)
    if oerr is not None or perr is not None:
        return None
    root = schema_root()
    original = _valid_index(orig_doc, root)
    patched = _valid_index(patch_doc, root)
    return sum(1 for c, ok in patched.items() if c not in original and not ok)


def _selfcheck() -> None:
    # A patch adding a schema-invalid key under a closed node is counted; a valid add is not.
    base = "on: push\njobs:\n  b:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
    good = base + "    timeout-minutes: 5\n"  # valid add
    bad = (
        "on: push\njobs:\n  b:\n    runs-on: ubuntu-latest\n    bogus_key: 1\n"
        "    steps:\n      - run: echo hi\n"
    )
    assert invalid_added(base, good) == 0, invalid_added(base, good)
    assert invalid_added(base, bad) == 1, invalid_added(base, bad)
    assert invalid_added("a: [", base) is None
    print("schema_validity self-check: OK")


if __name__ == "__main__":
    _selfcheck()
