"""RunStore — the per-cell artifact Repository: layout, resume, save (#45).

A *cell* is one (workflow, smell, config, rep, model) unit, identified by its
slug (see ``main._build_slug``). RunStore is the single owner of the on-disk
layout under ``output_dir`` and of the trace-aware resume predicate, so the
pipeline runner never hand-builds these paths or re-derives the skip rule.

Layout (all relative to ``output_dir``):

    results/{slug}.json                       result JSON
    patched_workflows/{slug}.yml              winning patched workflow
    patched_workflows_iters/{slug}@iter{n}.yml  per-iteration patches
    iteration_logs/{slug}@{detection|refactor}{n}.json  per-iteration loop entry
    traces/{slug}.json                        audit trail (opt-in / on failure)

The store owns both the trace path grammar and the hybrid write policy
(:meth:`save_trace`): a failed cell is always written; a successful cell only
when persist-traces is enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .trace import AgentCall, TraceError, TraceFile
from .utils import save_result as _write_result_json


class RunStore:
    """Repository over the per-cell output layout, resume predicate and saves."""

    _RESULTS = "results"
    _PATCHED = "patched_workflows"
    _PATCHED_ITERS = "patched_workflows_iters"
    _ITER_LOGS = "iteration_logs"
    _TRACES = "traces"

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    # ─── path grammar ─────────────────────────────────────────────────────────

    @property
    def results_dir(self) -> Path:
        return self.output_dir / self._RESULTS

    @property
    def patched_dir(self) -> Path:
        return self.output_dir / self._PATCHED

    @property
    def iters_dir(self) -> Path:
        return self.output_dir / self._PATCHED_ITERS

    @property
    def iter_logs_dir(self) -> Path:
        return self.output_dir / self._ITER_LOGS

    @property
    def traces_dir(self) -> Path:
        return self.output_dir / self._TRACES

    def result_path(self, slug: str) -> Path:
        return self.results_dir / f"{slug}.json"

    def patched_path(self, slug: str) -> Path:
        return self.patched_dir / f"{slug}.yml"

    def trace_path(self, slug: str) -> Path:
        return self.traces_dir / f"{slug}.json"

    def iter_patch_path(self, slug: str, index: int) -> Path:
        return self.iters_dir / f"{slug}@iter{index}.yml"

    def iter_log_path(self, slug: str, loop: str, index: int) -> Path:
        return self.iter_logs_dir / f"{slug}@{loop}{index}.json"

    def prepare(self) -> None:
        """Create the always-present result + patched-workflow directories."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.patched_dir.mkdir(parents=True, exist_ok=True)

    # ─── resume ───────────────────────────────────────────────────────────────

    def exists(self, slug: str, *, trace_required: bool) -> bool:
        """True when the cell is complete and resume should skip it.

        Complete means the result JSON is present and — when traces are being
        persisted — its trace JSON is present too. A missing result, or a
        missing trace under persist-traces, forces a re-run. A trace without a
        result never satisfies skip.
        """
        if not self.result_path(slug).exists():
            return False
        return not trace_required or self.trace_path(slug).exists()

    # ─── persistence ──────────────────────────────────────────────────────────

    def save_patched_workflow(self, slug: str, content: str) -> Path:
        """Write the winning patched workflow; returns its path."""
        path = self.patched_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def save_iter_patches(self, slug: str, patches: list[str]) -> list[Path]:
        """Write one ``{slug}@iter{n}.yml`` per patched workflow, in order.

        Empty ``patches`` is a no-op and the iters directory is left
        uncreated (so D1/D2 runs never materialise it). Returns the paths
        written, in iteration order.
        """
        if not patches:
            return []
        self.iters_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for index, patched in enumerate(patches, start=1):
            path = self.iter_patch_path(slug, index)
            path.write_text(patched, encoding="utf-8")
            written.append(path)
        return written

    def save_iter_logs(
        self,
        slug: str,
        *,
        header: dict[str, Any],
        detection: list[dict[str, Any]],
        refactor: list[dict[str, Any]],
    ) -> list[Path]:
        """Write one self-describing JSON per loop iteration; returns the paths.

        ``iteration_logs/{slug}@detection{n}.json`` for each A1↔A2 detection
        iteration and ``…@refactor{n}.json`` for each A3↔A4 refactor iteration,
        where ``n`` is the entry's own 1-based ``iteration`` index (the two
        loops are independent sequences). Each file is the shared per-cell
        ``header`` (run-context metadata) merged with that iteration's log
        entry, so a file stands alone for analysis. Both loops empty is a no-op
        and the directory is left uncreated (D1/D2 runs never materialise it).
        Detection files are written first, then refactor, each in iteration
        order.
        """
        written: list[Path] = []
        for loop, entries in (("detection", detection), ("refactor", refactor)):
            for entry in entries:
                path = self.iter_log_path(slug, loop, entry["iteration"])
                _write_result_json(path, {**header, "loop": loop, **entry})
                written.append(path)
        return written

    def save_result(self, slug: str, data: dict[str, Any]) -> Path:
        """Persist the result dict as indented JSON; returns its path."""
        path = self.result_path(slug)
        _write_result_json(path, data)
        return path

    def save_trace(
        self,
        slug: str,
        *,
        design: Literal["D1", "D2"],
        calls: list[AgentCall],
        error: BaseException | None,
        persist_enabled: bool,
    ) -> Path | None:
        """Write ``traces/{slug}.json`` per the hybrid policy; returns its path.

        Returns ``None`` when the trail is intentionally skipped (a successful
        cell with persist-traces off). A failure (``error`` set) is always
        written, regardless of the flag, so a failed cell stays diagnosable; a
        success is written only when ``persist_enabled``. On failure the captured
        ``error`` is recorded as a :class:`~gha_agents.trace.TraceError`
        pointing just past the calls captured before it was raised.
        """
        if error is None and not persist_enabled:
            return None

        trace_error: TraceError | None = None
        if error is not None:
            message = str(error)
            trace_error = TraceError(
                type=type(error).__name__,
                message=message.splitlines()[0] if message else "",
                raised_at_call_index=len(calls),
            )
        trace = TraceFile(
            slug=slug,
            design=design,
            calls=list(calls),
            error=trace_error,
        )
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_path(slug)
        path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        return path
