"""preflight.py — model accessibility / thinking / temperature gate.

Runs after the credit check and before dispatch. Every distinct model in the run
set is probed once with a shared trivial prompt; the run is aborted unless ALL
models pass three checks:

  * **accessibility** — the provider key authenticates and the model id resolves
    (no auth / not-found / 4xx). A build or call failure fails this leg.
  * **thinking** — the probe returns reasoning: ``reasoning_content`` non-empty
    OR ``reasoning_tokens > 0``. OpenAI hides the reasoning text, so the token
    count is its only signal; every other provider surfaces text (and tokens).
  * **temperature** — a static build inspection (no API call): every provider pins
    ``temperature == 0.0`` EXCEPT the two reasoning families whose API rejects any
    temperature ``!= 1`` once reasoning is on, where the factory drops it to the
    model default: **anthropic with thinking enabled** (also pins ``top_p`` to the
    0.95 floor — with thinking *off* it follows the ordinary 0.0 rule) and **openai
    reasoning models** (those given a ``reasoning_effort``). Verified live
    2026-06-18: gpt-5.4-nano returns HTTP 400 ``unsupported_value`` on
    ``temperature=0`` once a ``reasoning_effort`` is set.

Transient provider hiccups (HTTP 502/503/504, rate limits) are retried in place
via :class:`~gha_agents.invoker.AgentInvoker`, so a capacity blip cannot
false-block the run. The gate is a hard requirement — there is no per-config
opt-out; the only escape is the explicit ``--skip-preflight`` CLI flag, which the
caller is responsible for guarding with a loud untested-gate warning.

The probe reuses the production call path (a bare Agno agent driven through
``AgentInvoker``) so ``reasoning_tokens`` is captured exactly as a real cell
would capture it — essential for OpenAI, whose only thinking signal is the count.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .config import Provider, ResolvedRun
from .errors import GhaError
from .invoker import AgentCallResult, AgentInvoker
from .model_factory import _ANTHROPIC_THINKING_TOP_P, ModelFactory
from .run_context import RunContext, RunScope

# One fixed, trivial reasoning prompt shared by every model — enough to elicit a
# reasoning stream and a completion, without exercising the agent's system prompt
# or output schema (that is what the real run tests).
_PROBE_PROMPT = "Is 17 prime? Think step by step, then answer in one sentence."
_PROBE_INSTRUCTIONS = "You are a terse assistant. Answer briefly."


class PreflightError(GhaError):
    """Raised when one or more models fail the pre-flight gate.

    Carries the formatted per-model report as its message so the CLI can echo it
    verbatim before exiting non-zero.
    """


@dataclass(frozen=True, slots=True)
class ModelCheck:
    """The outcome of probing a single model across the three gate legs."""

    alias: str
    provider: Provider
    accessible: bool
    temperature_ok: bool
    thinking_ok: bool
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.accessible and self.temperature_ok and self.thinking_ok


def _temperature_ok(provider: Provider, model: Any, thinking_on: bool) -> tuple[bool, str]:
    """Static check: the built model carries the sampling rule for its provider.

    Most providers must pin ``temperature == 0.0``. Two reasoning families are
    exceptions, because their API rejects any temperature other than the default 1
    once reasoning is on, so the factory drops it to use the model default:

      * **anthropic with thinking enabled** — extended thinking; also pins ``top_p``
        to the 0.95 floor. With thinking **off** the factory leaves the matrix
        sampling untouched (``temperature=0.0``), so anthropic then follows the
        ordinary 0.0 rule like every other provider — hence the ``thinking_on``
        gate, which mirrors :meth:`ModelFactory._apply_thinking_overrides` applying
        the override only when thinking is on. (Without this gate the probe
        false-fails a thinking-off anthropic run, whose valid build carries
        ``temperature=0.0``/``top_p=1.0``.)
      * **openai reasoning models** (those given a ``reasoning_effort``) — the
        effort being set IS the reasoning signal, independent of the native-thinking
        toggle, exactly as the factory keys its temperature drop on
        ``reasoning_effort`` presence (not on ``thinking_on``).
    """
    temperature = getattr(model, "temperature", None)
    top_p = getattr(model, "top_p", None)
    if provider == "anthropic" and thinking_on:
        if temperature is not None:
            return False, (
                f"anthropic (thinking) temperature must be unset (model default), got {temperature!r}"
            )
        if top_p != _ANTHROPIC_THINKING_TOP_P:
            return False, f"anthropic (thinking) top_p must be {_ANTHROPIC_THINKING_TOP_P}, got {top_p!r}"
        return True, ""
    if provider == "openai" and getattr(model, "reasoning_effort", None):
        if temperature is not None:
            return False, (
                f"openai reasoning model temperature must be unset (model default), got {temperature!r}"
            )
        return True, ""
    if temperature != 0.0:
        return False, f"{provider} temperature must be 0.0, got {temperature!r}"
    return True, ""


def _thinking_ok(result: AgentCallResult) -> bool:
    """The unified thinking signal: reasoning text OR a positive reasoning-token
    count. OpenAI never returns the text, so its pass rides solely on the count;
    the other providers return both.
    """
    if result.reasoning_content:
        return True
    return (getattr(result.metrics, "reasoning_tokens", 0) or 0) > 0


def _probe_model(invoker: AgentInvoker, model: Any) -> AgentCallResult:
    """Drive one shared prompt through the production call path (bare agent →
    ``AgentInvoker``) so transient retries and ``reasoning_tokens`` capture match
    a real cell. Imported lazily so this module stays framework-light until used.
    """
    from agno.agent import Agent

    agent = Agent(model=model, instructions=_PROBE_INSTRUCTIONS, resolve_in_context=False)
    return invoker.invoke("preflight", agent, _PROBE_PROMPT)


def run_model_preflight(
    runs: Iterable[ResolvedRun],
    *,
    invoker: AgentInvoker | None = None,
    factory: ModelFactory | None = None,
    probe: Callable[[AgentInvoker, Any], AgentCallResult] | None = None,
) -> list[ModelCheck]:
    """Probe each distinct model in ``runs`` once and return a check per model.

    Models are de-duplicated by alias (the matrix is design × model, but a
    model's key/thinking/temperature are identical across designs, so one
    probe suffices). Each model is built and called under the same
    :class:`RunScope` a real cell would use, so the probe sees the exact sampling,
    provider and thinking the run will. Build/call failures are captured as a
    failed leg rather than raised, so the report covers every model in one pass.

    ``invoker`` / ``factory`` / ``probe`` are injectable for tests (a fake probe
    exercises every branch with no LLM).
    """
    invoker = invoker or AgentInvoker()
    factory = factory or ModelFactory()
    probe = probe or _probe_model

    checks: list[ModelCheck] = []
    seen: set[str] = set()
    for run in runs:
        if run.model_alias in seen:
            continue
        seen.add(run.model_alias)

        ctx = RunContext(
            provider=run.provider,
            model_overrides=run.model_map,
            sampling_overrides=run.sampling or None,
            thinking=run.thinking,
            json_mode=False,  # the probe wants free text, not the agent's schema
        )
        with RunScope(ctx):
            try:
                model = factory.build("detection")
            except Exception as exc:
                checks.append(
                    ModelCheck(run.model_alias, run.provider, False, False, False, f"build failed: {exc}")
                )
                continue

            temperature_ok, temp_detail = _temperature_ok(run.provider, model, run.thinking)

            try:
                result = probe(invoker, model)
            except Exception as exc:
                checks.append(
                    ModelCheck(
                        run.model_alias,
                        run.provider,
                        accessible=False,
                        temperature_ok=temperature_ok,
                        thinking_ok=False,
                        detail=temp_detail or f"call failed: {type(exc).__name__}: {str(exc)[:160]}",
                    )
                )
                continue

            thinking_ok = _thinking_ok(result)
            detail = temp_detail
            if not thinking_ok and not detail:
                detail = "no reasoning_content and reasoning_tokens == 0 (thinking not observed)"
            checks.append(
                ModelCheck(run.model_alias, run.provider, True, temperature_ok, thinking_ok, detail)
            )

    return checks


def format_report(checks: list[ModelCheck]) -> str:
    """A compact per-model pass/fail table, failures annotated with the reason."""
    width = max((len(c.alias) for c in checks), default=12)
    lines = [f"Pre-flight gate — {len(checks)} model(s):"]
    for c in checks:
        mark = "✔" if c.passed else "✖"
        # vertex authenticates via ADC (project + region), not an API key, so label
        # its accessibility leg "proj" to avoid implying a GOOGLE_API_KEY check.
        auth_leg = "proj" if c.provider == "vertex" else "key"
        legs = (
            f"{auth_leg}{'✔' if c.accessible else '✖'} "
            f"temp{'✔' if c.temperature_ok else '✖'} "
            f"thinking{'✔' if c.thinking_ok else '✖'}"
        )
        suffix = f"  — {c.detail}" if (not c.passed and c.detail) else ""
        lines.append(f"  {mark} {c.alias:<{width}}  {c.provider:<10} {legs}{suffix}")
    return "\n".join(lines)


def assert_models_ready(
    runs: Iterable[ResolvedRun],
    **kwargs: Any,
) -> list[ModelCheck]:
    """Run the gate; raise :class:`PreflightError` (carrying the report) if any
    model fails. Returns the checks on success so the caller can log the table.
    """
    checks = run_model_preflight(runs, **kwargs)
    if not checks:
        return checks
    if any(not c.passed for c in checks):
        raise PreflightError(format_report(checks))
    return checks
