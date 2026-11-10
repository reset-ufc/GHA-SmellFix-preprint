"""Unit tests for the pure ground-truth balancer (cap-then-min) and the
synthesis-target selector (Step 12c, cap-only / no-min)."""

import pytest

from dataset_utils.ground_truth_balancing import (
    balance,
    build_target_ground_truth,
    select_targets,
)

SMELL = "MissingJobTimeout"
OTHER = "UnpinnedActionVersion"


def _entries(present=0, absent=0, smell=SMELL, start=0):
    """`present` files each carrying one present `smell`, then `absent` files
    each carrying one absent `smell`. Unique file_keys via `start` offset."""
    out = []
    i = start
    for _ in range(present):
        out.append({"file_key": f"f{i}", "present_smells": [{"pascal": smell, "count": 1}], "absent_smells": []})
        i += 1
    for _ in range(absent):
        out.append({"file_key": f"f{i}", "present_smells": [], "absent_smells": [{"pascal": smell}]})
        i += 1
    return out


def _after(report, smell=SMELL):
    return report["per_smell_after"].get(smell, {"present": 0, "absent": 0})


def test_min_balances_to_smaller_polarity():
    # 10 present / 5 absent -> 5 / 5 (the headline example).
    gt = _entries(present=10, absent=5)
    out, report = balance(gt, n=50, balance_to_min=True)
    assert _after(report) == {"present": 5, "absent": 5}
    assert report["n_entries_out"] == 10  # 5 present + 5 absent survive; rest dropped empty
    assert len(out) == 10


def test_cap_bites_before_min():
    # 60 present / 70 absent, cap 50 -> 50 / 50 after min.
    gt = _entries(present=60, absent=70)
    _, report = balance(gt, n=50, balance_to_min=True)
    assert _after(report) == {"present": 50, "absent": 50}


def test_cap_then_min_asymmetric():
    # 60 present / 40 absent, cap 50 -> 50 / 40 -> min -> 40 / 40.
    gt = _entries(present=60, absent=40)
    _, report = balance(gt, n=50, balance_to_min=True)
    assert _after(report) == {"present": 40, "absent": 40}


def test_balance_to_min_off_keeps_cap_only():
    # 60 present / 40 absent, cap 50, NO min -> 50 / 40 (asymmetric).
    gt = _entries(present=60, absent=40)
    _, report = balance(gt, n=50, balance_to_min=False)
    assert _after(report) == {"present": 50, "absent": 40}


def test_parametric_n15():
    # 20 present / 18 absent, N=15 -> 15 / 15.
    gt = _entries(present=20, absent=18)
    _, report = balance(gt, n=15, balance_to_min=True)
    assert _after(report) == {"present": 15, "absent": 15}
    assert report["n_per_polarity_cap"] == 15


def test_deficit_recorded_vs_cap():
    # 3 present / 5 absent, N=50 -> 3 / 3, deficits = 47 / 47.
    gt = _entries(present=3, absent=5)
    _, report = balance(gt, n=50, balance_to_min=True)
    assert _after(report) == {"present": 3, "absent": 3}
    assert report["deficits_vs_cap"][SMELL] == {"present_deficit": 47, "absent_deficit": 47}


def test_priority_tiebreak_keeps_lowest_rank():
    # 3 present, cap to 2: keep the two LOWEST-priority file_keys.
    gt = _entries(present=3, absent=0)  # f0, f1, f2
    pri = {
        ("f0", SMELL, "present"): 30,
        ("f1", SMELL, "present"): 10,
        ("f2", SMELL, "present"): 20,
    }
    out, _ = balance(gt, pri, n=2, balance_to_min=False)
    kept = {e["file_key"] for e in out}
    assert kept == {"f1", "f2"}  # priorities 10 and 20 win over 30


def test_independent_smells_balanced_separately():
    gt = _entries(present=8, absent=4, smell=SMELL, start=0)
    gt += _entries(present=2, absent=6, smell=OTHER, start=100)
    _, report = balance(gt, n=50, balance_to_min=True)
    assert _after(report, SMELL) == {"present": 4, "absent": 4}
    assert _after(report, OTHER) == {"present": 2, "absent": 2}


def test_empty_entries_dropped():
    # An entry whose only smell is trimmed away must not appear in the output.
    gt = _entries(present=5, absent=1)  # absent bucket has 1 -> min = 1
    out, _ = balance(gt, n=50, balance_to_min=True)
    # 1 present + 1 absent survive => 2 entries; 4 present entries dropped.
    assert len(out) == 2
    for e in out:
        assert e["present_smells"] or e["absent_smells"]


# --- Synthesis-target selector (Step 12c) -----------------------------------

def _balanced_keys(out, smell, polarity):
    field = f"{polarity}_smells"
    return {e["file_key"] for e in out if any(s["pascal"] == smell for s in e.get(field, []))}


def test_select_targets_caps_to_n_no_min():
    # 60 present / 40 absent: select is per-bucket, cap-only, NO min coupling.
    gt = _entries(present=60, absent=40)
    assert len(select_targets(gt, None, SMELL, "present", n=50)) == 50  # capped
    assert len(select_targets(gt, None, SMELL, "absent", n=50)) == 40   # all (below cap)


def test_select_targets_priority_order():
    # Same (priority, file_key) sort as balance; rank 1 = lowest priority.
    gt = _entries(present=3, absent=0)  # f0, f1, f2
    pri = {
        ("f0", SMELL, "present"): 30,
        ("f1", SMELL, "present"): 10,
        ("f2", SMELL, "present"): 20,
    }
    assert select_targets(gt, pri, SMELL, "present", n=2) == ["f1", "f2"]
    assert select_targets(gt, pri, SMELL, "present", n=3) == ["f1", "f2", "f0"]


def test_selection_is_superset_of_balanced():
    # THE integrity property: every (smell, polarity) instance the balanced
    # cap-then-min balancer keeps must also be in the cap-only selection.
    gt = _entries(present=10, absent=5)
    pri = {(f"f{i}", SMELL, "present"): (10 - i) for i in range(10)}  # f9 best .. f0 worst
    out, _ = balance(gt, pri, n=50, balance_to_min=True)  # -> 5 present / 5 absent
    for polarity in ("present", "absent"):
        selected = set(select_targets(gt, pri, SMELL, polarity, n=50))
        kept = _balanced_keys(out, SMELL, polarity)
        assert kept <= selected, (polarity, kept - selected)


def test_build_target_records_projected_to_target_only():
    # Output records keep ONLY the target (smell, polarity) instance; other
    # smells on the same file are stripped (so the result merges cleanly).
    gt = [{
        "file_key": "f0",
        "present_smells": [],
        "absent_smells": [{"pascal": SMELL}, {"pascal": OTHER}],
    }]
    records, report = build_target_ground_truth(gt, None, SMELL, "absent", n=50)
    assert len(records) == 1
    assert records[0]["absent_smells"] == [{"pascal": SMELL}]
    assert records[0]["present_smells"] == []
    assert report["target_smell"] == SMELL
    assert report["n_selected"] == 1
    assert report["capped"] is False


def test_polarity_alias_and_validation():
    gt = _entries(present=0, absent=2)
    assert (
        select_targets(gt, None, SMELL, "negative", n=50)
        == select_targets(gt, None, SMELL, "absent", n=50)
    )
    with pytest.raises(ValueError):
        select_targets(gt, None, SMELL, "sideways", n=50)
