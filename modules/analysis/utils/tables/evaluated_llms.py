"""LaTeX table for the evaluated LLMs (tab:evaluated_llms).

One row per model, in the paper's order. The In and Out prices come from
``utils.cost_latency.PRICING``, the rates of the RQ3 cost, so the table shows exactly
what the cost uses. Prices print with two decimals, or three when the rate has them
(DeepSeek V4 Flash's weighted-average effective rate).

The other cells are fixed text, checked on 2026-09-28 against the runs and the
pipeline configuration (``modules/agents_pipeline``):
  * Access   = the provider in ``configs/*.yaml``.
  * Temp.    = the ``sampling_settings`` recorded in the result JSONs: 0 for Gemini
               3.5 Flash and DeepSeek V4 Flash. GPT-5.4-nano and Claude Haiku 4.5
               send no temperature in their thinking modes, so the provider default
               of 1 applies.
  * Thinking = ``reasoning_effort`` / ``thinking_level`` medium, or the 2,048-token
               reasoning budget of ``gha_agents.model_factory``.
  * Params.  = the DeepSeek V4 Flash model card (284B total, 13B active).

The output is only the paper's ``table`` environment.
"""

from pathlib import Path

from utils import cost_latency

# Paper row order: display name, alias, then the Access, Type, Params., Temp. and
# Thinking cells.
_MODELS = [
    ("GPT-5.4-nano", "gpt54-nano",
     "OpenAI API", "Proprietary", "--", "1", "Medium"),
    ("Claude Haiku 4.5", "haiku45",
     "Anthropic API", "Proprietary", "--", "1", "2,048 tokens"),
    ("Gemini 3.5 Flash", "gemini35-flash",
     "Gemini API", "Proprietary", "--", "0", "Medium"),
    ("DeepSeek V4 Flash", "deepseek-v4-flash",
     "OpenRouter API", "Open-weight", "284B (13B active)", "0", "2,048 tokens"),
]

_HEADER = r"""
\begin{table}[!bp]
\centering
\scriptsize
\caption{Overview of the selected LLMs. Costs are reported in USD per 1M tokens (June 2026).}
\label{tab:evaluated_llms}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{l l l r c c r r}
\toprule
\textbf{Model}
& \textbf{Access}
& \textbf{Type}
& \textbf{Params.}
& \textbf{Temp.}
& \textbf{Thinking}
& \textbf{In (1M/USD)}
& \textbf{Out (1M/USD)} \\
\midrule"""

_FOOTER = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\begin{minipage}{\columnwidth}
\centering
\tiny
GPT-5.4-nano and Gemini 3.5 Flash expose reasoning effort as a qualitative level rather than an explicit token budget; their token allocation is provider-managed.
\end{minipage}
\end{table}"""


def _price(usd: float) -> str:
    """A USD rate with two decimals, or three when the rate has a third."""
    text = f"{usd:.3f}"
    return text[:-1] if text.endswith("0") else text


def _render(pricing: dict) -> str:
    """The table environment, with each model's prices taken from ``pricing``."""
    models = []
    for name, alias, *cells in _MODELS:
        prices = [_price(pricing[alias]["in"]), _price(pricing[alias]["out"])]
        models.append("\n".join([name, *(f"& {cell}" for cell in [*cells, *prices])]) + r" \\")
    return "\n".join([_HEADER, "\n\n".join(models), _FOOTER]).strip() + "\n"


def evaluated_llms_table(
    output_path: Path | str = Path("data/artifacts") / "evaluated_llms.tex",
) -> str:
    """Write and return the EVALUATED LLMS LaTeX table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = _render(cost_latency.PRICING)
    output_path.write_text(latex, encoding="utf-8")
    return latex


if __name__ == "__main__":
    print(evaluated_llms_table())
