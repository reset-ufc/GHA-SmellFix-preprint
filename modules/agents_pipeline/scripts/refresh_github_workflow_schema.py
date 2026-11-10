#!/usr/bin/env python3
"""Refresh the vendored GitHub Actions workflow JSON Schema.

Downloads the official schema from SchemaStore and writes it to
``gha_agents/schemas/github-workflow.json``. The runtime tool
(:func:`gha_agents.tools.validate_github_actions_syntax`) loads only this
vendored copy and never reaches the network; this script is the one place that
fetches it. The schema is fully self-contained (no remote ``$ref``s), so the
single file is enough to validate workflows offline.

Usage:
    uv run python modules/agents_pipeline/scripts/refresh_github_workflow_schema.py
"""

from __future__ import annotations

import json
import os
import urllib.request

SCHEMA_URL = "https://www.schemastore.org/github-workflow.json"

_DEST = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "gha_agents",
        "schemas",
        "github-workflow.json",
    )
)


def main() -> None:
    request = urllib.request.Request(  # noqa: S310
        SCHEMA_URL, headers={"User-Agent": "study-refresh"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        raw = response.read()

    # Parse to confirm we fetched valid JSON before overwriting the vendored copy,
    # and to surface any remote $ref that would break offline validation.
    schema = json.loads(raw)
    remote_refs = sorted(_iter_remote_refs(schema))
    if remote_refs:
        print(f"WARNING: schema declares remote $ref(s) not vendored: {remote_refs}")

    os.makedirs(os.path.dirname(_DEST), exist_ok=True)
    with open(_DEST, "wb") as fh:
        fh.write(raw)
    print(f"wrote {os.path.relpath(_DEST)} ({len(raw)} bytes from {SCHEMA_URL})")


def _iter_remote_refs(node: object) -> set[str]:
    refs: set[str] = set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(("http://", "https://")):
            refs.add(ref)
        for value in node.values():
            refs |= _iter_remote_refs(value)
    elif isinstance(node, list):
        for value in node:
            refs |= _iter_remote_refs(value)
    return refs


if __name__ == "__main__":
    main()
