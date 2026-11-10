"""Paired statistics across designs (D1/D2) for a per-instance metric.

For a per-instance metric (detection correctness, clean removal, latency, tokens,
...) we ask, per model, whether the design changes the outcome. Each ordered
design pair is compared with:

* **Cliff's delta** -- non-parametric effect size in [-1, 1]; ``delta(A, B) > 0``
  means A tends to be larger. Magnitude bins follow Romano et al. (2006).
* a **paired Wilcoxon signed-rank** test (same instance under every design, matched
  on ``unit_cols``); ``binary=True`` switches to **McNemar's exact test** for a 0/1
  metric, and ``paired=False`` to Mann-Whitney U.
* **Holm-Bonferroni** correction across the pairs handed to one
  :func:`pairwise_design_stats` call.

scipy.stats only -- Holm, McNemar and Cliff's delta are implemented here. Run
``python -m utils.stats`` (or execute the file) for a self-check.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon, mannwhitneyu

from utils.designs import DESIGNS


def cliffs_delta(a, b) -> float:
    """Cliff's delta of ``a`` vs ``b``: (#(a>b) - #(a<b)) / (|a|*|b|).

    O((n+m)*log m) via ``searchsorted`` on a sorted ``b``."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float)
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    bs = np.sort(b)
    less = np.searchsorted(bs, a, side="left").sum()                 # #(b < a_i)
    greater = (b.size - np.searchsorted(bs, a, side="right")).sum()  # #(b > a_i)
    return float((less - greater) / (a.size * b.size))


def cliff_magnitude(d: float) -> str:
    """Romano et al. (2006) magnitude label for a Cliff's delta."""
    if d != d:  # NaN
        return "--"
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.330:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved.

    NaN inputs (a degenerate comparison) pass through as NaN and do not consume a
    rank in the family."""
    idx = [i for i, p in enumerate(pvals) if p == p]  # non-NaN positions
    m = len(idx)
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvals[j])):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def mcnemar_exact(a, b) -> float:
    """Two-sided exact McNemar p for paired 0/1 outcomes: a binomial test (p = 0.5) on
    the discordant pairs, where exactly one of ``a`` and ``b`` is 1. NaN when no pair
    is discordant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (np.isin(a, (0, 1)).all() and np.isin(b, (0, 1)).all()):
        raise ValueError("McNemar's test needs 0/1 outcomes")
    only_a = int(((a == 1) & (b == 0)).sum())
    discordant = only_a + int(((a == 0) & (b == 1)).sum())
    if discordant == 0:
        return float("nan")
    return float(binomtest(only_a, discordant).pvalue)


def _test(a: np.ndarray, b: np.ndarray, paired: bool, binary: bool = False) -> float:
    """p-value for ``a`` vs ``b``; NaN when the comparison is degenerate (no data,
    or every paired difference is zero so there is nothing to rank)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if paired:
        if a.size == 0 or np.all(a == b):
            return float("nan")
        if binary:
            return mcnemar_exact(a, b)
        try:
            return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:
            return float("nan")  # all-zero differences after dropping ties
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def pairwise_design_stats(df: pd.DataFrame, value_col: str, unit_cols: list[str],
                          design_col: str = "design",
                          designs: list[str] = None, paired: bool = True,
                          binary: bool = False, alpha: float = 0.05) -> pd.DataFrame:
    """Every ordered design pair compared on ``value_col``: Cliff's delta + a paired
    Wilcoxon (or Mann-Whitney) test, Holm-corrected across the pairs in this call.

    ``unit_cols`` identify a matched instance; for ``paired=True`` the two designs are
    inner-joined on them so only instances present in both sides are compared.
    ``binary=True`` (paired only) tests a 0/1 metric with McNemar's exact test."""
    if binary and not paired:
        raise ValueError("binary=True needs paired=True")
    designs = designs or [d for d in DESIGNS if d in set(df[design_col])]
    recs, pvals = [], []
    for ca, cb in combinations(designs, 2):
        ga = df[df[design_col] == ca]
        gb = df[df[design_col] == cb]
        if paired:
            merged = ga.merge(gb, on=unit_cols, suffixes=("_a", "_b"))
            va = merged[f"{value_col}_a"].to_numpy(dtype=float)
            vb = merged[f"{value_col}_b"].to_numpy(dtype=float)
        else:
            va = ga[value_col].to_numpy(dtype=float)
            vb = gb[value_col].to_numpy(dtype=float)
        d = cliffs_delta(va, vb)
        p = _test(va, vb, paired, binary)
        recs.append({"pair": f"{ca}-{cb}", "a": ca, "b": cb,
                     "n_a": int(va.size), "n_b": int(vb.size),
                     "median_a": float(np.nanmedian(va)) if va.size else float("nan"),
                     "median_b": float(np.nanmedian(vb)) if vb.size else float("nan"),
                     "cliffs_delta": d, "magnitude": cliff_magnitude(d),
                     "p_raw": p})
        pvals.append(p)
    out = pd.DataFrame(recs)
    out["p_holm"] = holm(pvals)
    out["significant"] = out["p_holm"] < alpha
    return out


def by_model(df: pd.DataFrame, value_col: str, unit_cols: list[str],
             model_col: str = "model_alias", **kw) -> pd.DataFrame:
    """:func:`pairwise_design_stats` per model (Holm corrected within each model's
    pairs), concatenated with a leading ``model`` column."""
    parts = []
    for m, g in df.groupby(model_col):
        r = pairwise_design_stats(g, value_col, unit_cols, **kw)
        r.insert(0, "model", m)
        parts.append(r)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _selfcheck() -> None:
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert cliff_magnitude(0.10) == "negligible" and cliff_magnitude(0.5) == "large"

    # Holm step-down on p=[.01,.04,.03] (m=3) -> [.03, .06, .06].
    h = holm([0.01, 0.04, 0.03])
    assert abs(h[0] - 0.03) < 1e-12, h
    assert abs(h[1] - 0.06) < 1e-12, h
    assert abs(h[2] - 0.06) < 1e-12, h
    assert holm([float("nan"), 0.02])[1] == 0.02  # lone non-NaN: m=1, unchanged

    # McNemar: 1 pair only A, 5 only B, 4 ties -> 2 * P(X <= 1 | n=6) = 14/64.
    assert abs(mcnemar_exact([1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 0, 0]) - 14 / 64) < 1e-12
    assert np.isnan(mcnemar_exact([1, 0], [1, 0]))  # no discordant pair

    # End-to-end: D1 > D2 on a paired metric over 30 matched units.
    rows = []
    for u in range(30):
        for design, base in (("D1", 9), ("D2", 5)):
            rows.append({"file_key": u, "smell": "X", "design": design,
                         "val": base + (u % 3)})
    res = pairwise_design_stats(pd.DataFrame(rows), "val", ["file_key", "smell"])
    d1d2 = res[res["pair"] == "D1-D2"].iloc[0]
    assert d1d2["cliffs_delta"] > 0.4 and d1d2["significant"], res
    assert (res["p_holm"] >= res["p_raw"] - 1e-12).all(), res  # Holm never shrinks p
    print("stats self-check: OK")


if __name__ == "__main__":
    _selfcheck()
