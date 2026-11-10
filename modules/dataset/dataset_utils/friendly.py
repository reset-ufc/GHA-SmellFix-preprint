"""Notebook-friendly exception messages.

Install via ``lib.friendly.install()`` at the top of a notebook (the bootstrap
cell already does this) to suppress full tracebacks for *predictable* pipeline
errors and replace them with a single-line explanation. Examples covered:

* ``FileNotFoundError`` — missing required input
* ``PipelineIntegrityError`` — input SHA does not match upstream report
* ``BootstrapError`` — corrupt or missing shipped artifact archive

To see the full traceback (debugging), set ``study_VERBOSE=1`` in the
environment, or pass ``verbose=True`` to ``install()``.

Outside IPython (plain ``python``) this module is a no-op — the script you ran
gets its normal traceback, which is what scripts/CIs expect.
"""

from __future__ import annotations

import os
import re
import sys

from dataset_utils.integrity import PipelineIntegrityError

try:
    from dataset_utils.bootstrap import BootstrapError as _BootstrapError
except ImportError:
    _BootstrapError = None  # type: ignore[assignment]


_HANDLED: tuple[type[BaseException], ...] = (FileNotFoundError, PipelineIntegrityError)
if _BootstrapError is not None:
    _HANDLED = (*_HANDLED, _BootstrapError)


_HINT = "(set study_VERBOSE=1 to see the full traceback)"


def _is_verbose() -> bool:
    return bool(os.environ.get("study_VERBOSE"))


# Strip "step N <label>: " prefix from PipelineIntegrityError headlines.
_STEP_PREFIX_RE = re.compile(r"^step\s+\S+\s+\S+:\s+", flags=re.IGNORECASE)


def _first_line(text: str) -> str:
    return text.splitlines()[0].rstrip(".:") if text else ""


def _capitalize(line: str) -> str:
    return line[:1].upper() + line[1:] if line else line


def _friendly_message(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        path = getattr(exc, "filename", None) or str(exc)
        return f"Input file missing: {path}"
    if isinstance(exc, PipelineIntegrityError):
        line = _STEP_PREFIX_RE.sub("", _first_line(str(exc)))
        return _capitalize(line) or "Input integrity check failed"
    if _BootstrapError is not None and isinstance(exc, _BootstrapError):
        return _capitalize(_first_line(str(exc))) or "Shipped artifact bootstrap failed"
    return f"{type(exc).__name__}: {exc}"


def install(verbose: bool | None = None) -> None:
    """Register an IPython custom exception handler. No-op outside IPython."""
    if verbose is None:
        verbose = _is_verbose()
    if verbose:
        return
    try:
        from IPython import get_ipython
    except ImportError:
        return
    ip = get_ipython()
    if ip is None:
        return

    def handler(shell, etype, evalue, tb, tb_offset=None):
        print(f"✗ {_friendly_message(evalue)}", file=sys.stderr)
        print(_HINT, file=sys.stderr)
        return None

    ip.set_custom_exc(_HANDLED, handler)


__all__ = ["install"]
