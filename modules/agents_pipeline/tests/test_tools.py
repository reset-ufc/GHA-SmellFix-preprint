"""Tests for gha_agents.tools.

Covers the GitHub Actions schema validator (``validate_github_actions_syntax``)
layered above the raw YAML check, plus the line-numbering helper. The schema is
vendored under ``gha_agents/schemas/github-workflow.json``; the schema-backed
tests skip cleanly when that file is absent (e.g. a checkout where the refresh
script has not run) so the suite never hard-fails on a missing data file.
"""

from __future__ import annotations

import json
import os

import pytest
from gha_agents import tools
from gha_agents.tools import (
    GITHUB_WORKFLOW_SCHEMA_PATH,
    enumerate_workflow_lines,
    validate_github_actions_syntax,
    validate_yaml_syntax,
)

_SCHEMA_PRESENT = os.path.isfile(GITHUB_WORKFLOW_SCHEMA_PATH)

requires_schema = pytest.mark.skipif(
    not _SCHEMA_PRESENT,
    reason=(
        "vendored github-workflow.json not present; run "
        "scripts/refresh_github_workflow_schema.py to fetch it"
    ),
)

_VALID_MINIMAL_WORKFLOW = """\
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"""


# ─── enumerate_workflow_lines (existing tool, light coverage) ──────────────────


def test_enumerate_workflow_lines_prefixes_each_line():
    out = enumerate_workflow_lines("name: CI\non: push")
    assert out == "L1: name: CI\nL2: on: push"


# ─── validate_yaml_syntax (existing tool, unchanged) ───────────────────────────


def test_validate_yaml_syntax_accepts_valid_yaml():
    assert validate_yaml_syntax(_VALID_MINIMAL_WORKFLOW) == "OK"


def test_validate_yaml_syntax_reports_malformed_yaml():
    assert validate_yaml_syntax("jobs: [unclosed").startswith("YAML syntax error:")


# ─── validate_github_actions_syntax ────────────────────────────────────────────


@requires_schema
def test_valid_minimal_workflow_returns_ok():
    """on: push + one job with a step satisfies the GitHub Actions schema."""
    assert validate_github_actions_syntax(_VALID_MINIMAL_WORKFLOW) == "OK"


@requires_schema
def test_on_key_is_not_coerced_to_boolean():
    """The YAML-1.1 `on -> True` coercion is suppressed, so a workflow whose only
    trigger is `on:` validates instead of failing the schema's required `on`."""
    workflow = (
        "on:\n"
        "  push:\n"
        "    branches: [main]\n"
        "jobs:\n"
        "  t:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo hi\n"
    )
    assert validate_github_actions_syntax(workflow) == "OK"


def test_malformed_yaml_returns_yaml_error():
    """A YAML parse failure short-circuits before schema validation (no schema
    file needed), mirroring validate_yaml_syntax's error shape."""
    result = validate_github_actions_syntax("jobs: [unclosed")
    assert result.startswith("YAML syntax error:")


@requires_schema
def test_missing_on_trigger_is_a_schema_error():
    workflow = "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
    result = validate_github_actions_syntax(workflow)
    assert result.startswith("GitHub Actions schema error")
    assert "'on'" in result
    assert "required" in result


@requires_schema
def test_jobs_as_list_names_the_jobs_path():
    workflow = "on: push\njobs:\n  - build\n"
    result = validate_github_actions_syntax(workflow)
    assert result.startswith("GitHub Actions schema error at jobs")
    assert "object" in result


@requires_schema
def test_step_without_uses_or_run_names_the_step_path():
    workflow = (
        "on: push\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - name: nothing\n"
    )
    result = validate_github_actions_syntax(workflow)
    assert result.startswith("GitHub Actions schema error")
    assert "jobs.build.steps[0]" in result


# ─── vendored-schema wiring (loaded from disk, never the network) ──────────────


def test_schema_path_is_inside_the_package_schemas_dir():
    package_dir = os.path.dirname(tools.__file__)
    expected = os.path.join(package_dir, "schemas", "github-workflow.json")
    assert os.path.normpath(GITHUB_WORKFLOW_SCHEMA_PATH) == os.path.normpath(expected)


@requires_schema
def test_vendored_schema_file_is_the_github_workflow_schema():
    with open(GITHUB_WORKFLOW_SCHEMA_PATH, encoding="utf-8") as fh:
        schema = json.load(fh)
    assert "github-workflow" in schema.get("$id", "")
    assert "jobs" in schema.get("properties", {})


@requires_schema
def test_validator_is_compiled_once_and_cached():
    """The schema is loaded/compiled a single time, not per call."""
    first = tools._workflow_validator()
    second = tools._workflow_validator()
    assert first is second
