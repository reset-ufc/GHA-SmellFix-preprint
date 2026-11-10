"""progress.py — optional live progress dashboard for experiment runs.

Default is a no-op (:class:`NullReporter`), which preserves the plain per-pair
echo path. ``--progress`` swaps in :class:`RichBarsReporter`: a live, grouped
bar view — one overall bar, one bar per design (D1/D2), one bar per
model — plus a counters footer (detected / solved / failed / skipped).

The reporter is shared across the dispatcher's worker threads (one cell per
thread), so every mutating call takes a lock. Bar denominators are pre-sized
from the run plan, so they never grow mid-run. Resume-skipped pairs advance the
bars too: they are processed work, just fast.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Protocol


def count_pairs(entries: list[dict[str, Any]]) -> int:
    """Eval pairs a single cell processes: present ∪ absent smells per file.

    Mirrors the pair enumeration in ``main.run_pipeline`` (present_smells →
    positive, absent_smells → negative); used to pre-size the progress bars.
    """
    return sum(
        len(e.get("present_smells", [])) + len(e.get("absent_smells", []))
        for e in entries
    )


class ProgressReporter(Protocol):
    """Per-pair progress sink, usable as a context manager around dispatch."""

    def pair_done(self, design: str, model: str, row: dict[str, Any]) -> None: ...
    def __enter__(self) -> ProgressReporter: ...
    def __exit__(self, *exc: object) -> None: ...


class NullReporter:
    """No-op reporter — the default; the runner keeps its plain per-pair echo."""

    def pair_done(self, design: str, model: str, row: dict[str, Any]) -> None:
        return None

    def __enter__(self) -> NullReporter:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _LiveFooter:
    """Counters line that rebuilds from the reporter's live state on every Live
    refresh.

    Holding this in the render tree (instead of a snapshot ``Text``) lets the
    footer track the run off the single auto-refresh tick, so ``pair_done`` never
    has to call ``Live.update``. That matters: ``update`` fired from every one of
    N worker threads, on top of the auto-refresh thread, raced the cursor and
    tore partial frames into the scrollback under high concurrency.
    """

    def __init__(self, reporter: RichBarsReporter) -> None:
        self._reporter = reporter

    def __rich__(self) -> Any:
        return self._reporter._footer()


class RichBarsReporter:
    """Live grouped progress bars: overall + per-design + per-model.

    ``plan`` is ``(design, model_alias)`` for every cell that will run;
    ``pairs_per_cell`` is the pair count each cell processes (uniform across a
    matrix). Totals are derived once so the bars are sized before the first pair
    completes. ``rich`` is imported lazily so importing this module never costs
    anything on the default (no-progress) path.
    """

    def __init__(self, plan: list[tuple[str, str]], pairs_per_cell: int) -> None:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()

        design_cells = Counter(d for d, _ in plan)
        model_cells = Counter(m for _, m in plan)
        overall_total = max(1, len(plan) * pairs_per_cell)

        self._overall = Progress(
            TextColumn("[bold]Overall"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(compact=True),
            expand=True,
        )
        self._design = Progress(
            TextColumn("  {task.fields[label]:<5}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            expand=True,
        )
        self._model = Progress(
            TextColumn("  {task.fields[label]:<20}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            expand=True,
        )

        self._overall_id = self._overall.add_task("", total=overall_total)
        self._design_ids = {
            design: self._design.add_task("", total=n * pairs_per_cell, label=design)
            for design, n in sorted(design_cells.items())
        }
        self._model_ids = {
            m: self._model.add_task("", total=n * pairs_per_cell, label=m)
            for m, n in sorted(model_cells.items())
        }
        self._live: Any = None

    # ── rendering ─────────────────────────────────────────────────────────────

    def _footer(self) -> Any:
        from rich.text import Text

        c = self._counts
        return Text.assemble(
            ("detected ", "dim"),
            (str(c["detected"]), "bold green"),
            ("  ·  solved ", "dim"),
            (str(c["solved"]), "bold green"),
            ("  ·  failed ", "dim"),
            (str(c["failed"]), "bold red"),
            ("  ·  skipped ", "dim"),
            (str(c["skipped"]), "bold yellow"),
        )

    def _render(self) -> Any:
        from rich.console import Group
        from rich.text import Text

        return Group(
            self._overall,
            Text(""),
            Text("By design", style="dim"),
            self._design,
            Text(""),
            Text("By model", style="dim"),
            self._model,
            Text(""),
            _LiveFooter(self),
        )

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> RichBarsReporter:
        from rich.live import Live

        # Per-pair echoes are silenced under the dashboard (echo_pairs=False) and
        # Agno logs are routed to the file, so the live region normally owns the
        # terminal alone. redirect_stdout/stderr stays as a safety net: any stray
        # third-party write is buffered and scrolled cleanly ABOVE the pinned bars
        # instead of corrupting the live region.
        self._live = Live(
            self._render(),
            refresh_per_second=8,
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(*exc)
            self._live = None

    # ── per-pair update (called from worker threads) ──────────────────────────

    def pair_done(self, design: str, model: str, row: dict[str, Any]) -> None:
        with self._lock:
            self._overall.advance(self._overall_id)
            if design in self._design_ids:
                self._design.advance(self._design_ids[design])
            if model in self._model_ids:
                self._model.advance(self._model_ids[model])

            status = row.get("status")
            if row.get("smells") == "1":
                self._counts["detected"] += 1
            if row.get("solved") == "1":
                self._counts["solved"] += 1
            if status in ("FAILED", "ERROR"):
                self._counts["failed"] += 1
            elif status == "SKIP":
                self._counts["skipped"] += 1
            # No Live.update() here: the bars are live references inside the render
            # tree and the footer is an auto-rebuilding _LiveFooter, so the single
            # auto-refresh tick redraws everything. Calling update() from every
            # worker thread (on top of auto-refresh) is what tore partial frames
            # into the scrollback under high --max-workers concurrency.
