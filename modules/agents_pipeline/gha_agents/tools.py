"""
tools.py — Agno-compatible tools for Agents.

These are plain Python functions passed directly to Agent(tools=[...]).
Agno automatically generates JSON schemas from the signatures and docstrings.

Tools:
  enumerate_workflow_lines  — prefix every line with its line number so agents
                              can cite exact locations in ConfigSmell fields.
  validate_yaml_syntax      — parse the YAML produced by the refactor agent and
                              return a human-readable validation report.
  validate_github_actions_syntax
                            — one level above validate_yaml_syntax: after the
                              YAML parses, validate the document against the
                              vendored official GitHub Actions JSON Schema
                              (required keys, job/step/trigger structure) and
                              report the offending path on a violation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema
import jsonschema.validators
import yaml
from jsonschema.exceptions import best_match
from jsonschema.protocols import Validator


def enumerate_workflow_lines(yaml_content: str) -> str:
    """
    Return the GitHub Actions workflow with every line prefixed by its line number.

    Use this tool BEFORE analyzing a workflow so you can report precise
    start_line / end_line values in each ConfigSmell.

    Args:
        yaml_content: Raw YAML content of a GitHub Actions workflow file.

    Returns:
        A string where each line is prefixed by a zero-padded line number,
        e.g. "L001: name: CI\\nL002: on: push\\n…".
        This lets you reference exact locations when reporting smells.
    """
    lines = yaml_content.splitlines()
    width = len(str(len(lines)))
    numbered = "\n".join(f"L{str(i + 1).zfill(width)}: {line}" for i, line in enumerate(lines))
    return numbered


def validate_yaml_syntax(yaml_content: str) -> str:
    """
    Validate that the given string is syntactically correct YAML.

    Use this tool AFTER generating the patched workflow to confirm it is
    parseable before returning it as the final output.

    Args:
        yaml_content: The YAML string to validate (typically the patched workflow).

    Returns:
        "OK" if the YAML is valid.
        A descriptive error string (including line number) if parsing fails.
    """
    try:
        yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        return f"YAML syntax error: {exc}"
    else:
        return "OK"


# ─── GitHub Actions YAML loader ────────────────────────────────────────────────
# PyYAML's SafeLoader follows YAML 1.1, which resolves `on`, `off`, `yes`, and
# `no` to booleans. A workflow's top-level `on:` trigger key would then parse as
# the boolean True instead of the string "on", so every real workflow would
# spuriously fail the schema's `required: [on, jobs]` check. GitHub Actions reads
# workflows as YAML 1.2, where only `true`/`false` are booleans. This loader
# restores that behavior: `on:` stays a string key, while `true`/`false` values
# still parse as booleans for keys such as `cancel-in-progress`.
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


# ─── Vendored GitHub Actions workflow schema ────────────────────────────────────
# Resolved relative to this file so it is robust to the caller's working
# directory (the same pattern prompts.py uses for its bundled data). The file is
# the official SchemaStore schema; refresh it with
# scripts/refresh_github_workflow_schema.py. It is self-contained (no remote
# $refs), so validation runs fully offline from this single file.
GITHUB_WORKFLOW_SCHEMA_PATH = Path(__file__).parent / "schemas" / "github-workflow.json"

_workflow_validator_cache: Validator | None = None


def _workflow_validator() -> Validator:
    """Build and cache the compiled GitHub Actions schema validator once.

    The schema is read and compiled on the first call and reused thereafter — the
    tool is invoked many times in a run, so the file is never re-read per call.
    """
    global _workflow_validator_cache
    if _workflow_validator_cache is None:
        with open(GITHUB_WORKFLOW_SCHEMA_PATH, encoding="utf-8") as fh:
            schema = json.load(fh)
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        _workflow_validator_cache = validator_cls(schema)
    return _workflow_validator_cache


def _json_location(path) -> str:
    """Render a jsonschema error path as a readable location.

    e.g. ``deque(['jobs', 'build', 'steps', 0])`` -> ``'jobs.build.steps[0]'``.
    Returns ``'(root)'`` for a top-level (empty-path) error.
    """
    rendered = ""
    for token in path:
        if isinstance(token, int):
            rendered += f"[{token}]"
        else:
            rendered += f".{token}" if rendered else str(token)
    return rendered or "(root)"


def validate_github_actions_syntax(yaml_content: str) -> str:
    """
    Validate that the content is a structurally valid GitHub Actions workflow.

    This is one level above validate_yaml_syntax: after confirming the content
    parses as YAML, it validates the parsed document against the official GitHub
    Actions workflow JSON Schema — required top-level keys (`on`, `jobs`), the
    job and step structure, trigger shape, and so on. Use it AFTER generating the
    patched workflow to confirm it is a well-formed workflow, not merely
    well-formed YAML.

    Args:
        yaml_content: The workflow YAML string to validate (typically the patch).

    Returns:
        "OK" if the workflow parses and satisfies the GitHub Actions schema.
        "YAML syntax error: ..." if the content is not parseable YAML.
        "GitHub Actions schema error at <path>: <problem>" naming the offending
        location (e.g. "jobs.build.steps[0]: 'uses' is a required property") so
        the workflow can be fixed directly from the message.
    """
    try:
        # _WorkflowYamlLoader is a yaml.SafeLoader subclass (S506 false positive).
        workflow = yaml.load(yaml_content, Loader=_WorkflowYamlLoader)  # noqa: S506
    except yaml.YAMLError as exc:
        return f"YAML syntax error: {exc}"

    try:
        validator = _workflow_validator()
    except FileNotFoundError:
        return (
            "GitHub Actions schema unavailable: the vendored schema was not found at "
            f"{GITHUB_WORKFLOW_SCHEMA_PATH}. Run "
            "scripts/refresh_github_workflow_schema.py to fetch it."
        )

    errors = list(validator.iter_errors(workflow))
    if not errors:
        return "OK"

    primary = best_match(iter(errors))
    location = _json_location(primary.absolute_path)
    report = f"GitHub Actions schema error at {location}: {primary.message}"
    remaining = len(errors) - 1
    if remaining > 0:
        report += f" (+{remaining} more schema violation{'s' if remaining > 1 else ''})"
    return report
