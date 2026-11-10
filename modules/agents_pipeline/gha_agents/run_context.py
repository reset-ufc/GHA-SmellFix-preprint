"""Parameter Object + atomic scope for the seven run-scoped settings (#44).

A matrix cell is configured by seven values that the env readers consult via
contextvars: the active provider, the role → model-id map, the sampling
overrides, an optional request timeout, the audit-trail toggle, the
native-thinking toggle, and the JSON-mode toggle. They used to be threaded as
loose parameters and
entered as separate context managers at the dispatch site.

:class:`RunContext` bundles them into one frozen object; :class:`RunScope` binds
the bundle atomically and tears it down in one shot. Binding goes through the
existing per-setting context managers in :mod:`gha_agents.env` (still the
binding primitives, and still the unit-test seam), composed under a single
:class:`contextlib.ExitStack` so a parallel worker can never half-configure a
cell: if any bind raises, the ones already entered are unwound before the error
propagates.

**Ordering is load-bearing.** Provider is bound *before* sampling so that
``ModelFactory.sampling_for`` — which projects the merged sampling dict onto the
keys the active provider accepts (only OpenAI takes ``reasoning_effort``; only
Gemini ``thinking_level``; Anthropic takes neither) — reads the provider that
is already live. Teardown is the reverse order, handled by the ExitStack.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass

from .config import Provider, ResolvedRun
from .env import (
    json_mode_overrides,
    model_overrides,
    persist_traces,
    provider_overrides,
    request_timeout_overrides,
    sampling_overrides,
    thinking_overrides,
)


@dataclass(frozen=True)
class RunContext:
    """The seven run-scoped settings bound per matrix cell, as one value.

    All fields default to the inert value of a scope that pins nothing: no
    provider pin (infer from the model id), no model/sampling override, no
    client-side timeout, traces off, thinking on, json-mode on. ``RunScope``
    reads them in the order the read-order hazard requires.
    """

    provider: Provider | None = None
    model_overrides: dict[str, str] | None = None
    sampling_overrides: dict[str, float | int] | None = None
    request_timeout: float | None = None
    persist_traces: bool = False
    thinking: bool = True
    json_mode: bool = True

    @classmethod
    def for_run(
        cls,
        spec: ResolvedRun,
        *,
        request_timeout: float | None = None,
        persist_traces: bool = False,
    ) -> RunContext:
        """The settings of one resolved run: its model, provider and sampling."""
        return cls(
            provider=spec.provider,
            model_overrides=spec.model_map,
            sampling_overrides=spec.sampling or None,
            request_timeout=request_timeout,
            persist_traces=persist_traces,
            thinking=spec.thinking,
            json_mode=spec.json_mode,
        )


class RunScope:
    """Bind a :class:`RunContext` for the duration of a ``with`` block, atomically.

    Enters the per-setting context managers under one ExitStack — provider
    first (so sampling projection sees it), then model, sampling, timeout, the
    trace toggle, the thinking toggle and the json-mode toggle. Any failure
    mid-bind unwinds what was already entered; exit tears them all down in reverse.
    """

    def __init__(self, context: RunContext) -> None:
        self._context = context
        self._stack = ExitStack()

    def __enter__(self) -> RunContext:
        ctx = self._context
        try:
            self._stack.enter_context(provider_overrides(ctx.provider))
            self._stack.enter_context(model_overrides(ctx.model_overrides))
            self._stack.enter_context(sampling_overrides(ctx.sampling_overrides))
            self._stack.enter_context(request_timeout_overrides(ctx.request_timeout))
            self._stack.enter_context(persist_traces(ctx.persist_traces))
            self._stack.enter_context(thinking_overrides(ctx.thinking))
            self._stack.enter_context(json_mode_overrides(ctx.json_mode))
        except BaseException:
            self._stack.close()
            raise
        return ctx

    def __exit__(self, *exc_info: object) -> None:
        self._stack.close()
