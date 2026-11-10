"""ModelFactory — the only module that constructs framework model objects (#47).

All Agno model imports live here so ``env`` ends framework-free: ``env`` owns the
run-scoped configuration (contextvars, API keys, model-id / provider resolution),
ModelFactory turns that configuration into a concrete provider model.

Two projections are applied per provider. (1) Sampling-key projection narrows the
merged sampling dict (defaults ← YAML overrides) to the keys each provider's SDK
accepts — ``seed`` is dropped everywhere, OpenAI adds ``reasoning_effort``, Gemini
adds ``thinking_level`` — since a stray key would raise ``TypeError``. (2) Native
thinking is on by default (user decision 2026-06-02), toggleable per model via
``models.<alias>.sampling.thinking``: Anthropic gets ``thinking={"type":
"enabled", "budget_tokens": …}`` and, because extended thinking rejects
``temperature != 1`` and ``top_p < 0.95`` at the API, drops ``temperature`` (so
the model uses its default) and pins ``top_p=0.95`` — it does NOT run off the
matrix's ``temperature=0`` (a documented threat to validity, see CONTEXT.md
"Sampling regime"); Gemini gets ``include_thoughts=True`` plus a family-aware
depth knob — ``thinking_budget`` (2.5) or ``thinking_level`` (3.x replaced the
token budget with an enum), see ``_gemini_thinking_kwargs`` (``vertex`` is the
same Gemini class on Vertex AI — identical thinking wiring, but ``vertexai=True``
+ ADC auth instead of an API key); OpenRouter gets ``extra_body={"reasoning": {"max_tokens":
…}}`` (off → ``{"exclude": True}``) because hybrid reasoners such as gemma-4 emit
NO reasoning channel by default — without the opt-in their chain-of-thought never
reaches ``reasoning_content`` and is silently lost (deepseek-v4 reasons by
default, so it was unaffected; OpenRouter ignores the field for non-reasoning
models). Native ollama / openai need nothing wired beyond ``reasoning_effort``,
as OpenAI exposes no token budget (only effort) and its o-series never streams
``reasoning_content``. A single ``_REASONING_BUDGET_TOKENS`` pins the same native
reasoning budget across Anthropic / OpenRouter / Gemini / Vertex so depth is
comparable; tune it alone to trade reasoning depth for cost across the fleet.
"""

from __future__ import annotations

import os
import re
from typing import Any

from .config import Provider
from .env import (
    _DEFAULT_SAMPLING,
    _OLLAMA_DEFAULT_BASE_URL,
    _OLLAMA_PREFIX,
    _get_anthropic_api_key,
    _get_api_key,
    _get_gemini_api_key,
    _get_openai_api_key,
    _get_vertex_config,
    _provider_override_var,
    _resolve_provider,
    _sampling_override_var,
    _timeout_override_var,
    get_model_id,
    is_thinking_active,
)
from .errors import GhaConfigError

# Sentinel so ``sampling_for()`` with no argument resolves the provider from the
# active pin (matching the legacy ``env._get_sampling()`` no-arg behaviour),
# while ``sampling_for(None)`` still means "no provider — return unprojected".
_RESOLVE_FROM_CONTEXT: Any = object()

# Sampling keys each provider's SDK accepts. ``seed`` is intentionally absent for
# every provider: Anthropic's SDK has no seed knob, extended thinking makes its
# sampling non-deterministic regardless, and the other providers honour seed only
# best-effort — so the package does not claim bit-reproducibility it cannot
# deliver. OpenAI adds ``reasoning_effort`` and Gemini ``thinking_level`` — the
# optional reasoning knobs the YAML may set.
_PROVIDER_SAMPLING_KEYS: dict[Provider, frozenset[str]] = {
    "openrouter": frozenset({"temperature", "top_p"}),
    "ollama": frozenset({"temperature", "top_p"}),
    "openai": frozenset({"temperature", "top_p", "reasoning_effort"}),
    "gemini": frozenset({"temperature", "top_p", "thinking_level"}),
    "vertex": frozenset({"temperature", "top_p", "thinking_level"}),
    "anthropic": frozenset({"temperature", "top_p"}),
}

# Uniform native reasoning budget (tokens), applied to every provider that exposes
# a token knob so reasoning depth is comparable across models (user decision
# 2026-06-18): Anthropic ``thinking.budget_tokens``, OpenRouter
# ``reasoning.max_tokens``, Gemini / Vertex ``thinking_budget``. OpenAI is the sole
# exception — it has no token knob, only ``reasoning_effort`` (set per-model in the
# YAML). Tune this one value to trade reasoning depth for cost across the fleet.
_REASONING_BUDGET_TOKENS = 2048
# Back-compat alias: Anthropic's extended-thinking budget is the same uniform value.
_ANTHROPIC_THINKING_BUDGET_TOKENS = _REASONING_BUDGET_TOKENS

# OpenRouter provider routing: sort candidate endpoints by throughput ("nitro"),
# exactly equivalent to the ``:nitro`` model-slug shortcut
# (https://openrouter.ai/docs/guides/routing/provider-selection#provider-sorting).
# Hardcoded as the fleet default for every OpenRouter model rather than set per
# config so the experiment slugs/YAML stay routing-agnostic and the slower
# default (price-sorted) routing never silently re-applies. Trades cost for speed
# — accepted: OpenRouter throughput is the bottleneck, not per-token price.
_OPENROUTER_PROVIDER_SORT = "throughput"

# Gemini 3.x replaced the 2.5-era token knob ``thinking_budget`` with the
# ``thinking_level`` enum (MINIMAL/LOW/MEDIUM/HIGH, case-insensitive). Sending
# ``thinking_budget`` to a 3.x id is silently ignored — reasoning reverts to the
# model default and the uniform-depth parity above is lost — or rejected outright.
# "medium" mirrors the OpenAI ``reasoning_effort: medium`` the fleet already uses,
# keeping cross-model reasoning depth as comparable as the enum allows. UNVERIFIED
# against a live 3.x call — confirm ``usage_metadata.thoughts_token_count > 0``
# before trusting parity (see docs/model_capabilities.md).
_GEMINI_3X_THINKING_LEVEL = "medium"

# Major version parsed from a Gemini model id, e.g. "gemini-3.5-flash" → 3.
_GEMINI_MAJOR_RE = re.compile(r"gemini-(\d+)")


def _gemini_thinking_kwargs(model_id: str, thinking_on: bool) -> dict[str, Any]:
    """Native-thinking kwargs for the Agno Gemini class, selected by model family.

    With thinking off, only ``include_thoughts=False`` is sent (no budget/level).
    With it on, a Gemini 3.x id gets ``thinking_level`` (the 2.5 ``thinking_budget``
    is a no-op / rejected there) while 2.5-and-earlier keep ``thinking_budget`` at
    the uniform fleet budget. Shared by the ``gemini`` and ``vertex`` branches
    (same Agno class). The caller drops ``thinking_level`` here when the YAML
    sampling already set one, to avoid a duplicate-kwarg ``TypeError``.
    """
    if not thinking_on:
        return {"include_thoughts": False}
    match = _GEMINI_MAJOR_RE.search(model_id)
    major = int(match.group(1)) if match else 0
    if major >= 3:
        return {"include_thoughts": True, "thinking_level": _GEMINI_3X_THINKING_LEVEL}
    return {"include_thoughts": True, "thinking_budget": _REASONING_BUDGET_TOKENS}

# Anthropic extended thinking rejects top_p < 0.95 and any temperature != 1 at the
# API. We pin top_p to this floor and leave temperature unset (model default)
# rather than forcing temperature=1 — user decision 2026-06-18.
_ANTHROPIC_THINKING_TOP_P = 0.95


class ModelFactory:
    """Builds a configured Agno model for an agent role from the active run state."""

    def sampling_for(
        self, provider: Provider | None = _RESOLVE_FROM_CONTEXT
    ) -> dict[str, float | int]:
        """Merge defaults ← overrides, then project onto ``provider``'s keys.

        With no argument the provider is read from the active pin; with an
        explicit ``provider=None`` (no pin) the merged dict is
        returned unprojected. ``None`` values are dropped so optional knobs left
        unset never reach the constructor.
        """
        if provider is _RESOLVE_FROM_CONTEXT:
            provider = _provider_override_var.get()
        override = _sampling_override_var.get()
        merged: dict[str, float | int] = (
            dict(_DEFAULT_SAMPLING) if not override else {**_DEFAULT_SAMPLING, **override}
        )
        if provider is None:
            return {k: v for k, v in merged.items() if v is not None}
        allowed = _PROVIDER_SAMPLING_KEYS.get(provider, frozenset(merged))
        return {k: v for k, v in merged.items() if k in allowed and v is not None}

    @staticmethod
    def _apply_thinking_overrides(
        provider: Provider | None,
        sampling: dict[str, float | int],
        thinking_on: bool,
    ) -> dict[str, float | int]:
        """Fold in the sampling overrides ``build`` applies for native thinking.

        Anthropic extended thinking constrains sampling at the API: ``temperature``
        may only be ``1`` (any other value 400s) and ``top_p`` is rejected unless
        it is ``>= 0.95``. Rather than pin ``temperature`` we drop it so the model
        uses its own default, and set ``top_p`` to the ``0.95`` floor — a user
        decision (2026-06-18). OpenAI **reasoning** models (those given a
        ``reasoning_effort``) impose the same rule: ``temperature`` must be the
        default 1, so ``temperature=0`` 400s — we drop it there too (``top_p=1.0``
        already equals the API default, so it stays). With thinking off / no
        reasoning effort the request passes through untouched, so a ``temperature``
        override is honoured. Centralised here so the model construction and the
        recorded provenance share one source of truth.
        """
        if thinking_on and provider == "anthropic":
            overridden = {**sampling, "top_p": _ANTHROPIC_THINKING_TOP_P}
            overridden.pop("temperature", None)
            return overridden
        # OpenAI reasoning models reject BOTH temperature != 1 and top_p; the effort
        # being set IS the reasoning signal (independent of the native-thinking
        # toggle, which the OpenAI path does not otherwise consult). They route
        # through the Responses API (see build), which 400s on either sampling knob
        # ("Unsupported parameter: 'top_p'"), so drop both → model defaults.
        if provider == "openai" and sampling.get("reasoning_effort"):
            overridden = dict(sampling)
            overridden.pop("temperature", None)
            overridden.pop("top_p", None)
            return overridden
        return sampling

    def effective_sampling_for(
        self,
        provider: Provider | None = _RESOLVE_FROM_CONTEXT,
        thinking_on: bool | None = None,
    ) -> dict[str, float | int]:
        """The sampling kwargs ``build`` actually passes to the provider SDK.

        Like :meth:`sampling_for` but with the per-provider thinking overrides
        folded in, so result metadata records what was *applied* (e.g. Anthropic's
        forced ``temperature=1``), never the pre-override request. ``thinking_on``
        defaults to the active context toggle; pass it explicitly to skip a second
        contextvar read when the caller already has it.
        """
        if provider is _RESOLVE_FROM_CONTEXT:
            provider = _provider_override_var.get()
        if thinking_on is None:
            thinking_on = is_thinking_active()
        return self._apply_thinking_overrides(provider, self.sampling_for(provider), thinking_on)

    def build(self, role: str) -> Any:
        """Return a configured Agno model object for ``role``.

        Resolves the model id (YAML override) and provider, projects sampling,
        enables native thinking by default, and dispatches to the provider SDK.
        """
        model_id = get_model_id(role)
        provider = _resolve_provider(model_id)
        thinking_on = is_thinking_active()
        sampling = self.effective_sampling_for(provider, thinking_on=thinking_on)
        timeout = _timeout_override_var.get()

        # A client-side timeout also disables OpenAI SDK retries: the SDK retries
        # httpx.TimeoutException up to max_retries (default 2), multiplying the
        # wall-clock cap by (max_retries+1) and defeating the cutoff. Anthropic /
        # Gemini expose ``timeout`` but no ``max_retries`` knob on their Agno
        # wrappers — passing it would raise TypeError.
        openai_timeout_kwargs: dict[str, float | int] = (
            {"timeout": timeout, "max_retries": 0} if timeout is not None else {}
        )
        native_sdk_timeout_kwargs: dict[str, float] = (
            {"timeout": timeout} if timeout is not None else {}
        )

        if provider == "anthropic":
            from agno.models.anthropic import Claude

            # ``sampling`` already carries the thinking overrides (top_p=0.95, no
            # temperature) via effective_sampling_for when thinking is on; with it
            # off the request passes through and the thinking kwarg is omitted.
            thinking_kwargs: dict[str, Any] = {}
            if thinking_on:
                thinking_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": _ANTHROPIC_THINKING_BUDGET_TOKENS,
                }
            return Claude(
                id=model_id,
                api_key=_get_anthropic_api_key(),
                # Prompt-cache the per-role system message (smell definition + rules):
                # it is byte-stable across pairs of the same role/smell while the
                # workflow YAML rides in the user prompt (see teams.py), so Anthropic
                # prefix-caches it after the first call within the 5-min TTL. The first
                # call of each distinct system prompt writes the cache (cache_write
                # tokens, billed 1.25x); every reuse reads it (cache_read, 0.1x).
                # extended_cache_time (1h TTL) left off; enable cache_tools=True too if
                # the tool block ever grows large enough to dominate the static prefix.
                cache_system_prompt=True,
                **thinking_kwargs,
                **sampling,
                **native_sdk_timeout_kwargs,
            )

        if provider == "gemini":
            from agno.models.google import Gemini

            # include_thoughts surfaces the reasoning summary; the depth knob is
            # selected by model family (thinking_level for 3.x, thinking_budget for
            # 2.5) — see _gemini_thinking_kwargs. Both ride the thinking toggle:
            # off → no summary, no depth knob.
            thinking_kwargs = _gemini_thinking_kwargs(model_id, thinking_on)
            # An explicit YAML ``sampling.thinking_level`` wins and must not collide
            # with the factory default as a duplicate kwarg.
            if "thinking_level" in sampling:
                thinking_kwargs.pop("thinking_level", None)
            return Gemini(
                id=model_id,
                api_key=_get_gemini_api_key(),
                **thinking_kwargs,
                **sampling,
                **native_sdk_timeout_kwargs,
            )

        if provider == "vertex":
            from agno.models.google import Gemini

            # Vertex AI = the same Agno Gemini class as the ``gemini`` branch, but
            # billed through a Google Cloud project (aiplatform.googleapis.com)
            # rather than AI Studio. ``vertexai=True`` selects the Vertex transport
            # and the google-genai client authenticates via Application Default
            # Credentials (``gcloud auth application-default login``) — NO
            # GOOGLE_API_KEY is read in this mode (agno/models/google/gemini.py
            # skips the api_key branch when vertexai is on). include_thoughts /
            # thinking_budget are transport-agnostic, so reasoning capture and the
            # uniform _REASONING_BUDGET_TOKENS match the native gemini branch
            # exactly; only the credential + endpoint differ.
            project_id, location = _get_vertex_config()
            # Same Agno Gemini class as the ``gemini`` branch → same family-aware
            # depth knob (thinking_level for 3.x, thinking_budget for 2.5).
            thinking_kwargs = _gemini_thinking_kwargs(model_id, thinking_on)
            if "thinking_level" in sampling:
                thinking_kwargs.pop("thinking_level", None)
            return Gemini(
                id=model_id,
                vertexai=True,
                project_id=project_id,
                location=location,
                **thinking_kwargs,
                **sampling,
                **native_sdk_timeout_kwargs,
            )

        from agno.models.openai import OpenAIChat

        if provider == "openai":
            # Native OpenAI: same Agno class as OpenRouter, but no base_url so the
            # SDK uses its default (api.openai.com). reasoning_content is never
            # returned over chat, so there is no thinking kwarg to set here.
            #
            # BUT a reasoning model (reasoning_effort set) cannot use Agno's
            # structured output on /v1/chat/completions: output_schema is sent as a
            # function tool, and OpenAI rejects "function tools with reasoning_effort"
            # on chat/completions ("use /v1/responses instead"). So reasoning OpenAI
            # models route through the Responses API (OpenAIResponses), which supports
            # reasoning + tools; non-reasoning OpenAI stays on chat/completions.
            if sampling.get("reasoning_effort"):
                from agno.models.openai import OpenAIResponses

                # store=False: no server-side retention of the response, and fully
                # STATELESS calls. The Responses API defaults store=True for reasoning
                # models (Agno forces it unless told otherwise) and would then chain
                # turns via previous_response_id off OpenAI-held state. We opt out so
                # nothing is persisted on OpenAI and no opaque external state can leak
                # in: reasoning continuity within a single run is carried by Agno's
                # encrypted in-request payload (reasoning.encrypted_content) instead.
                # This mirrors the prior chat/completions behaviour — reproducible,
                # isolated per pair.
                return OpenAIResponses(
                    id=model_id,
                    api_key=_get_openai_api_key(),
                    store=False,
                    **sampling,
                    **openai_timeout_kwargs,
                )
            return OpenAIChat(
                id=model_id,
                api_key=_get_openai_api_key(),
                **sampling,
                **openai_timeout_kwargs,
            )

        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", _OLLAMA_DEFAULT_BASE_URL).rstrip("/")
            return OpenAIChat(
                id=model_id.removeprefix(_OLLAMA_PREFIX),
                api_key="ollama",
                base_url=base_url,
                **sampling,
                **openai_timeout_kwargs,
            )

        # Every modelled provider has its own explicit branch above; the tail is
        # OpenRouter ONLY. Guard it so a future Provider value added to the Literal
        # without a branch fails loud here instead of silently misrouting to
        # OpenRouter with the wrong base_url + OPENROUTER_API_KEY (the trap that
        # made this an unconditional fall-through dangerous for `vertex`).
        if provider != "openrouter":
            raise GhaConfigError(
                f"unhandled provider {provider!r} in ModelFactory.build — add an "
                "explicit dispatch branch (no silent OpenRouter fall-through)."
            )

        # provider == "openrouter"
        # OpenRouter surfaces a provider's native reasoning channel only when the
        # request opts in. Hybrid reasoners (e.g. gemma-4) return NO reasoning
        # tokens by default, so without this their chain-of-thought never reaches
        # ``reasoning_content`` and is silently dropped (deepseek-v4 reasons by
        # default and was unaffected). Mirrors the native-thinking toggle the other
        # providers honour: on → request ``reasoning.max_tokens`` = the uniform
        # budget so the reasoning streams back at the same depth as the native
        # providers and Agno maps the response's ``reasoning`` field onto
        # ``reasoning_content``; off → ``exclude`` lets the model still reason
        # internally but return no tokens. OpenRouter ignores the field for models
        # without reasoning support, so it is safe to send unconditionally.
        reasoning_body = (
            {"max_tokens": _REASONING_BUDGET_TOKENS} if thinking_on else {"exclude": True}
        )
        return OpenAIChat(
            id=model_id,
            api_key=_get_api_key(),
            base_url="https://openrouter.ai/api/v1",
            # ``provider.sort: throughput`` = the "nitro" routing default for the
            # whole OpenRouter fleet (see _OPENROUTER_PROVIDER_SORT); rides next to
            # ``reasoning`` in the same provider-preferences body.
            extra_body={
                "reasoning": reasoning_body,
                "provider": {"sort": _OPENROUTER_PROVIDER_SORT},
            },
            **sampling,
            **openai_timeout_kwargs,
        )
