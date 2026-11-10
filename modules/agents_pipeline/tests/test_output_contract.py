"""Tests for OutputContract — the single declarative source of each role's
``<output_format>`` block, round-tripped against the hand-written Pydantic
schema at boot (#35).

The contract declares the field order + the prose class-name once; the rendered
``<output_format>`` text is generated from that declaration (so the prose can
never silently drift from the field list), and a boot-time assertion proves the
declared field order equals the bound schema's ``model_fields``. PROMPT_HASHES
must stay byte-identical: the rendered bytes are unchanged, only their source
moved.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from gha_agents.agents import (
    CombinedResult,
    DetectionContract,
    DetectionValidationResult,
    RefactorResult,
    RefactorValidationResult,
)
from gha_agents.output_contract import (
    OUTPUT_CONTRACTS,
    OutputContract,
    assert_output_contracts_consistent,
)
from gha_agents.prompts import PROMPT_HASHES

# Captured from the frozen prompts BEFORE the contract refactor (see #35). The
# rendered <output_format> bytes — and therefore every prompt hash — must be
# byte-identical after sourcing the blocks from OutputContract. A later slice
# that intentionally changes a prompt (e.g. #41 dropping the D2 YAML echo) is
# expected to update this fingerprint deliberately.
# Updated by #41: the D2 detection prompt dropped the original_workflow echo, so
# its 'detection' role hash changed for every smell. D1, validators, and refactor
# are untouched.
# Updated 2026-06-07: the `pascal` join-key was renamed verbose -> short paper
# identifier (DefinePermissionsForWorkflowsWithExternalActions ->
# MissingPermExternalActions, etc.). PROMPT_HASHES is keyed by pascal, so its
# keys changed; the hash VALUES are unchanged (pascal is not part of the
# rendered prompt — only name/description are), so the fingerprint is re-pinned.
# Updated 2026-06-09: every smell `description` was replaced with column v4 of
# smell-definitions.csv (the richer multi-condition definitions). description is
# interpolated into <target_smell>, so all 19 updated smells' hashes changed
# across every role; the 3 presentational smells (v4 empty) are unchanged.
# Updated 2026-06-09: every role's <inputs> block gained a `<repository_context>`
# line (repo metadata now wired from the baseline ground_truth into the user turn), so
# every system-prompt hash shifted for every smell and role.
# Updated 2026-06-09: SMELL_CATALOG pruned to the 19 evaluated behavioral smells;
# the 3 presentational smells (Correct indentation / Name run steps / Add comments)
# were dropped, so their per-role hashes no longer appear in PROMPT_HASHES.
# Updated 2026-06-15: DETECTION_EVIDENCE_RULES dropped the stale
# `len(evidence_ranges) <= count` rule and now permits evidence_ranges on an
# absent verdict (count=0), matching the relaxed DetectionContract validator. The
# rule is part of DETECTION_CORE_RULES, so every detection-bearing role's
# system-prompt hash shifted for every smell.
# Updated: the four-agent configuration variant was removed, so PROMPT_HASHES now
# holds only the D1 and D2 scenario keys. The D1/D2 hash VALUES are unchanged
# (their templates were untouched); only the dropped top-level scenario key
# re-pins the fingerprint.
# Updated 2026-10-17: the scenario keys were renamed to D1/D2, the paper's design
# names. Every per-role hash VALUE is unchanged (no template text changed); only the
# renamed top-level keys re-pin the fingerprint.
_FROZEN_PROMPT_HASHES_FINGERPRINT = (
    "ad0ce378930076a4cc0c98dbe5fb94ddd9fe59e12bcd0051f1129eefc50d3e93"
)

# The schema bound to each contract key — the actual output_schema the agent of
# that (scenario, role) emits. After #41 D2 detection emits the single
# DetectionContract.
_SCHEMA_BY_KEY = {
    "combined": CombinedResult,
    "detection": DetectionContract,
    "detection_validation": DetectionValidationResult,
    "refactor": RefactorResult,
    "refactor_validation": RefactorValidationResult,
}


def test_prompt_hashes_fingerprint_is_byte_identical():
    """Sourcing <output_format> from OutputContract changes no rendered byte."""
    blob = json.dumps(PROMPT_HASHES, sort_keys=True).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == _FROZEN_PROMPT_HASHES_FINGERPRINT


def test_every_contract_has_a_bound_schema():
    assert set(OUTPUT_CONTRACTS) == set(_SCHEMA_BY_KEY)


def test_output_format_field_order_matches_schema_for_every_role():
    for key, contract in OUTPUT_CONTRACTS.items():
        schema = _SCHEMA_BY_KEY[key]
        assert contract.fields == tuple(schema.model_fields), key
        assert contract.fields[0] == "reasoning", key


def test_render_round_trips_the_declared_field_list():
    """The rendered prose parses back to exactly the declared field tuple."""
    for key, contract in OUTPUT_CONTRACTS.items():
        rendered = contract.render()
        assert rendered.startswith("<output_format>Emit ")
        assert rendered.endswith(". No extra fields.</output_format>")
        body = rendered.split("exactly: ", 1)[1].split(". No extra fields.", 1)[0]
        assert tuple(body.split(", ")) == contract.fields, key


def test_assert_consistent_passes_for_the_real_schemas():
    # The production wiring — must not raise.
    assert_output_contracts_consistent(_SCHEMA_BY_KEY)


def test_assert_consistent_raises_on_field_drift():
    class _Drifted(CombinedResult):
        extra_unexpected: str = ""

    drifted = dict(_SCHEMA_BY_KEY)
    drifted["combined"] = _Drifted
    with pytest.raises(ValueError, match="combined"):
        assert_output_contracts_consistent(drifted)


def test_assert_consistent_raises_when_reasoning_not_leading():
    bad = OutputContract(prose_name="X", fields=("smell_detected", "reasoning"))
    with pytest.raises(ValueError, match="reasoning"):
        bad.assert_reasoning_leads()
