"""
utils.py — Shared utilities for Agents.

Provides:
  logger          — pre-configured stdlib logger for the whole project
  setup_logging() — configure console + file handlers based on verbosity
  xml_wrap()      — wrap content in an XML tag pair for safe LLM data demarcation
  log_error()     — append a structured JSON failure entry to output/error.log
  save_result()   — write a per-file result dict as formatted JSON
  print_summary() — print a CLI summary table after processing all files
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import typer

# ─── Logger ───────────────────────────────────────────────────────────────────

# Module-level logger — always available; handlers attached by setup_logging().
logger = logging.getLogger("study")

# Serializes concurrent appends to a shared per-config error.log: same-config
# repetitions run on parallel workers and a multi-KB JSON entry can split across
# write() syscalls, interleaving two threads into a corrupt JSONL line. A
# process-wide lock keeps each entry atomic relative to its in-process siblings.
_ERROR_LOG_LOCK = threading.Lock()

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

# Verbose flag readable by other modules (e.g. agent factories that want to
# turn on Agno's own debug output). Set by ``setup_logging``.
_VERBOSE = False


def is_verbose() -> bool:
    """Return True iff ``setup_logging`` was last called with ``verbose=True``."""
    return _VERBOSE


class _ConsoleAllowlistFilter(logging.Filter):
    """Allow only loggers whose name starts with one of the given prefixes."""

    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__()
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefixes)


class _AgnoNoiseFilter(logging.Filter):
    """Collapse Agno's failed-tool-call noise to a single clean line.

    Agno emits two records when a tool call raises (e.g. a model passing the
    wrong kwargs to ``validate_yaml_syntax``): a ``"Could not run function …"``
    warning whose message is a multi-line pydantic dump, and a standalone
    full-traceback via ``log_exception()``. Both are non-fatal self-heal noise.
    This filter drops the traceback record entirely and truncates the warning
    to its first line, leaving one readable marker per failed tool call.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Drop the standalone traceback (log_exception always sets exc_info).
        if record.levelno >= logging.ERROR and record.exc_info is not None:
            return False
        # Collapse the multi-line "Could not run function …" dump to one line.
        message = record.getMessage()
        if "Could not run function" in message:
            record.msg = message.split("\n", 1)[0]
            record.args = ()
        return True


def setup_logging(verbose: bool, log_dir: Path, *, console: bool = True) -> None:
    """
    Configure logging for the the study pipeline.

    Console handler (stdout):
      - verbose=False : only ``study.*`` messages at INFO level. Third-party
                        libs (httpx, agno, httpcore, …) are suppressed.
      - verbose=True  : ``study.*`` and ``agno.*`` messages at DEBUG level —
                        per-agent invocations, prompts, token deltas, loop
                        iterations, Agno-internal step traces. HTTP/network
                        chatter (httpx, httpcore, openai) is still suppressed
                        because it adds noise without research signal.

    File handler (log_dir/<timestamp>.log):
      - Always enabled. Writes everything at DEBUG when verbose, INFO otherwise.
    """
    global _VERBOSE
    _VERBOSE = verbose

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    study_logger = logging.getLogger("study")
    study_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # ── Console handler ──────────────────────────────────────────────────────
    # Skipped when a live progress dashboard owns the terminal (logs → file only).
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(formatter)

        if verbose:
            console_handler.addFilter(_ConsoleAllowlistFilter(("study", "agno")))
        else:
            console_handler.addFilter(logging.Filter("study"))

        root_logger.addHandler(console_handler)

    # ── File handler — always logs everything ────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_name = f"{timestamp}.log"

    file_handler = logging.FileHandler(log_dir / log_name, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # ── Tame Agno's tool-execution noise ─────────────────────────────────────
    # When a model mis-calls a tool (e.g. an open-weight model passing the whole
    # output object to validate_yaml_syntax), Agno logs TWO records: a
    # "Could not run function …" warning whose message is the multi-line pydantic
    # dump, and a standalone full-traceback via log_exception() (always exc_info).
    # The failure is non-fatal — Agno feeds it back to the model and the pair
    # usually still completes. The 'agno' logger has propagate=False and its own
    # handler, so it bypasses the handlers above and must be filtered on itself.
    # Drop the traceback record and collapse the warning to a single clean line.
    agno_logger = logging.getLogger("agno")
    if not any(isinstance(f, _AgnoNoiseFilter) for f in agno_logger.filters):
        agno_logger.addFilter(_AgnoNoiseFilter())

    # Under the live dashboard (console=False) Agno's OWN handler would still
    # spill provider errors — Claude API 400s, "Error in Agent run", retry
    # classifications — straight to the terminal and shred the pinned bars. Drop
    # its private handlers and let it propagate to the root file handler so those
    # records land in the log only. Verbose runs keep console=True, so Agno errors
    # still surface there; the _AgnoNoiseFilter stays attached either way.
    if not console:
        agno_logger.handlers.clear()
        agno_logger.propagate = True


# ─── LLM Data Demarcation ────────────────────────────────────────────────────


def xml_wrap(tag: str, content: str) -> str:
    """
    Wrap content in a pair of XML tags to clearly separate data from instructions.

    This is the industry-standard approach (recommended by Anthropic and OpenAI)
    to prevent prompt injection and improve LLM accuracy when passing raw file
    content or structured data inside a user message.

    Example:
        xml_wrap("workflow", "on: push")
        # → "<workflow>\non: push\n</workflow>"
    """
    return f"<{tag}>\n{content}\n</{tag}>"


# ─── Output Helpers ───────────────────────────────────────────────────────────


def log_error(
    log_path: Path,
    file_name: str,
    detail: str,
    *,
    category: str | None = None,
    exception_type: str | None = None,
    target_smell: str | None = None,
    model_ids: dict[str, str] | None = None,
    raw_response_path: str | None = None,
) -> None:
    """
    Append a structured JSON failure entry to the error log (one line per event).

    Backward compatible: the original 3-positional-arg form still works and
    produces the original ``{timestamp, file, error}`` schema. The optional
    keyword args enrich the entry for forensic auditing — see paper §3 Step 6
    (failure taxonomy) and Step 8 (per-run debugging).
    """
    entry: dict[str, object] = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "file": file_name,
        "error": detail,
    }
    optional = {
        "category": category,
        "exception_type": exception_type,
        "target_smell": target_smell,
        "model_ids": model_ids,
        "raw_response_path": raw_response_path,
    }
    for key, value in optional.items():
        if value is not None:
            entry[key] = value

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _ERROR_LOG_LOCK, log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    logger.warning("Failure logged for '%s': %s", file_name, detail)
    typer.echo(f"  ⚠  Failure logged → {log_path}", err=True)


def persist_raw_response(artifacts_dir: Path, *, slug: str, raw_response: str) -> Path:
    """Write a raw LLM response to ``<artifacts_dir>/<slug>.raw`` and return the path.

    The forensic file lets the user audit what the model actually returned when
    parsing into the agent's ``output_schema`` failed.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target = artifacts_dir / f"{slug}.raw"
    target.write_text(raw_response, encoding="utf-8")
    return target


def save_result(out_file: Path, data: dict) -> None:
    """
    Persist a result dict to the specified JSON file.
    Creates parent directories if they do not exist.

    Written atomically: the payload is fully written to a sibling ``.tmp`` file,
    flushed + fsync'd, then ``os.replace``d onto the target. A rename within one
    directory is atomic, so an interruption mid-write leaves the temp file (cleaned
    on rerun) rather than a torn JSON — which resume's existence check would
    otherwise lock in as "complete" and never regenerate. Readers and the resume
    predicate therefore only ever observe a whole file.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    tmp_file = out_file.with_name(f"{out_file.name}.tmp")
    with tmp_file.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_file, out_file)
    logger.debug("Result saved → %s", out_file)


# ─── CLI Summary ──────────────────────────────────────────────────────────────

_COL_WIDTH = 68
_MAX_LISTED_PROBLEMS = 15
# Errored/invalid pairs are listed; the retry count is ERROR only. ERROR pairs
# left no result on disk so a resume re-run picks them up; FAILED (cap-exhausted)
# pairs DID save a result, so resume skips them — not retry candidates.
_PROBLEM_STATUS = {"ERROR", "INVALID"}


def _summary_counts(rows: list[dict]) -> dict[str, int]:
    """Tally rows by status plus the detected/solved positives."""
    from collections import Counter

    status = Counter(r.get("status", "?") for r in rows)
    return {
        "total": len(rows),
        "ok": status.get("OK", 0),
        "failed": status.get("FAILED", 0),
        "error": status.get("ERROR", 0),
        "skipped": status.get("SKIP", 0),
        "invalid": status.get("INVALID", 0),
        "detected": sum(1 for r in rows if r.get("smells") == "1"),
        "solved": sum(1 for r in rows if r.get("solved") == "1"),
    }


def _log_full_table(rows: list[dict]) -> None:
    """Persist the full per-pair table to the run log so the console can stay terse."""
    lines = [f"{'FILE':<44} {'SMELLS':>6} {'SOLVED':>6} {'STATUS':>8}"]
    for r in rows:
        lines.append(
            f"{r.get('file', '?'):<44} {r.get('smells', '?'):>6} "
            f"{r.get('solved', '?'):>6} {r.get('status', '?'):>8}"
        )
    logger.info("Result summary (%d pairs):\n%s", len(rows), "\n".join(lines))


def print_summary(
    rows: list[dict],
    *,
    verbose: bool = False,
    retry_command: str | None = None,
    resume_enabled: bool = True,
) -> None:
    """Print a compact, colored result summary; the full table goes to the log.

    Default console output is terse: a one-line colored tally, only the
    errored/invalid pairs (capped at :data:`_MAX_LISTED_PROBLEMS`), and a
    copy-paste re-run hint to retry the error instances. The full per-pair table
    is always written to the run log via :func:`_log_full_table`, and is also
    echoed to the console when ``verbose`` is set — so nothing is lost, the
    default just stops flooding the terminal on big batches.
    """
    dim = typer.colors.BRIGHT_BLACK
    c = _summary_counts(rows)
    _log_full_table(rows)

    sep = typer.style("═" * _COL_WIDTH, fg=dim)
    thin = typer.style("─" * _COL_WIDTH, fg=dim)

    typer.echo("\n" + sep)
    typer.echo(typer.style("  Run summary", bold=True))

    tally = [
        typer.style(f"{c['total']} pairs", bold=True),
        typer.style(f"✔ {c['ok']} ok", fg=typer.colors.GREEN),
    ]
    if c["failed"]:
        tally.append(typer.style(f"✖ {c['failed']} failed", fg=typer.colors.RED))
    if c["error"]:
        tally.append(typer.style(f"⚠ {c['error']} error", fg=typer.colors.BRIGHT_RED, bold=True))
    if c["skipped"]:
        tally.append(typer.style(f"⏭ {c['skipped']} skipped", fg=typer.colors.YELLOW))
    if c["invalid"]:
        tally.append(typer.style(f"? {c['invalid']} invalid", fg=typer.colors.MAGENTA))
    typer.echo("  " + "   ".join(tally))
    typer.echo(
        "  "
        + typer.style("detected ", fg=dim)
        + typer.style(str(c["detected"]), fg=typer.colors.CYAN)
        + typer.style("  ·  solved ", fg=dim)
        + typer.style(str(c["solved"]), fg=typer.colors.CYAN)
    )

    problems = [r for r in rows if r.get("status") in _PROBLEM_STATUS]
    if problems:
        typer.echo(thin)
        for r in problems[:_MAX_LISTED_PROBLEMS]:
            st = r.get("status", "?")
            color = typer.colors.BRIGHT_RED if st == "ERROR" else typer.colors.MAGENTA
            icon = "⚠" if st == "ERROR" else "?"
            typer.echo(
                "  " + typer.style(f"{icon} {st:<7}", fg=color) + " " + r.get("file", "?")
            )
        if len(problems) > _MAX_LISTED_PROBLEMS:
            extra = len(problems) - _MAX_LISTED_PROBLEMS
            typer.echo(typer.style(f"  … +{extra} more (see error.log)", fg=dim))

    if c["error"] and retry_command:
        typer.echo(thin)
        typer.echo(
            "  re-run "
            + typer.style(f'"{retry_command}"', fg=typer.colors.CYAN, bold=True)
            + f" to retry the {c['error']} error instance(s)"
        )
        if resume_enabled:
            typer.echo(
                typer.style(f"  resume is on — the {c['ok']} completed pairs are skipped.", fg=dim)
            )
        else:
            typer.echo(
                typer.style(
                    "  enable execution.resume so completed pairs are skipped on retry.",
                    fg=typer.colors.YELLOW,
                )
            )

    if verbose:
        typer.echo(thin)
        typer.echo(typer.style(f"  {'FILE':<44} {'SMELLS':>6} {'SOLVED':>6} {'STATUS':>8}", bold=True))
        for r in rows:
            typer.echo(
                f"  {r.get('file', '?'):<44} {r.get('smells', '?'):>6} "
                f"{r.get('solved', '?'):>6} {r.get('status', '?'):>8}"
            )

    typer.echo(sep + "\n")
