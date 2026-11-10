"""Tests for gha_agents.utils — xml_wrap demarcation and the CLI summary table."""

from __future__ import annotations

from gha_agents.utils import print_summary, xml_wrap


def test_xml_wrap_wraps_raw_content():
    """xml_wrap puts the raw content between the tag pair (locks the docstring example)."""
    assert xml_wrap("workflow", "on: push") == "<workflow>\non: push\n</workflow>"


def test_print_summary_lists_errored_pair_in_full(capsys):
    """An errored pair's full identifier appears in the default (terse) summary —
    no truncation to 5/29 chars."""
    rows = [
        {
            "file": "my_long_workflow.yml#MissingJobTimeout",
            "smells": "?",
            "solved": "?",
            "status": "ERROR",
        }
    ]
    print_summary(rows)
    out = capsys.readouterr().out
    # The whole identifier survives — the old [:5]/[:29] truncation is gone.
    assert "my_long_workflow.yml#MissingJobTimeout" in out
    assert "my_lo " not in out


def test_print_summary_is_terse_by_default_and_full_under_verbose(capsys):
    """A clean OK run does not dump every pair to the console; the full per-pair
    table appears only with verbose=True."""
    rows = [{"file": "ci.yml#X", "smells": "0", "solved": "—", "status": "OK"}]

    print_summary(rows)
    terse = capsys.readouterr().out
    assert "ci.yml#X" not in terse  # OK rows are not listed by default
    assert "1 pairs" in terse and "1 ok" in terse

    print_summary(rows, verbose=True)
    full = capsys.readouterr().out
    assert "ci.yml#X" in full  # verbose echoes the full table
    assert "…" not in full


def test_print_summary_emits_retry_hint_for_error_instances(capsys):
    """With error rows and a retry_command, the summary prints the re-run hint
    counting only the resume-retryable ERROR pairs (not OK/FAILED)."""
    rows = [
        {"file": "a.yml#S", "smells": "?", "solved": "?", "status": "ERROR"},
        {"file": "b.yml#S", "smells": "1", "solved": "1", "status": "OK"},
        {"file": "c.yml#S", "smells": "0", "solved": "0", "status": "FAILED"},
    ]
    print_summary(
        rows,
        retry_command="uv run python -m gha_agents.experiment cfg.yaml",
        resume_enabled=True,
    )
    out = capsys.readouterr().out
    assert "retry the 1 error instance(s)" in out
    assert "uv run python -m gha_agents.experiment cfg.yaml" in out


def test_print_summary_no_retry_hint_when_no_errors(capsys):
    """A clean run prints no retry hint even when a retry_command is supplied."""
    rows = [{"file": "ci.yml#X", "smells": "1", "solved": "1", "status": "OK"}]
    print_summary(rows, retry_command="uv run python -m gha_agents.experiment cfg.yaml")
    out = capsys.readouterr().out
    assert "re-run" not in out
