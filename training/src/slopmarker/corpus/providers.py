"""Provider adapters for AI-side generation.

Model IDs here were read from each provider's live models endpoint, not from memory.
Sampling support is established by a runtime probe rather than a hardcoded table,
because these constraints move: on current Anthropic models `temperature` and `top_p`
are rejected outright, and on OpenAI's reasoning models they are accepted only when
reasoning is off. A probe self-heals when that changes again; an assumption fails
150,000 times.

Two settings are mandatory rather than optional, and both are about corpus integrity
as much as cost. Thinking is on by default on the newest models across all three
providers, and reasoning tokens bill as output. Left alone that multiplies the bill
several times over *and* leaks reasoning text into documents, which would become the
single strongest artifact in the corpus. And assistant prefill -- the classic way to
suppress preambles -- now returns 400 on every current Anthropic model, so preambles
are handled by instruction plus hygiene instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

Provider = Literal["anthropic", "openai", "gemini"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: Provider
    input_per_mtok: float
    output_per_mtok: float
    # Models that accept temperature/top_p can carry a sampling grid. Those that do not
    # substitute an effort or verbosity axis, keeping the grid's shape so the corpus
    # stays stratified either way.
    sampling: bool  # accepts temperature/top_p
    held_out: bool = False  # test split only: measures generalization to unseen models
    weight: float = 1.0  # share of generation volume
    effort: bool = False  # accepts output_config.effort (Anthropic)
    thinking_off: bool = False  # accepts an explicit thinking budget of 0 (Gemini)
    extra: dict[str, Any] = field(default_factory=dict)


# Pricing from the claude-api skill catalog. No Anthropic model takes a sampling grid:
# the installed SDK has removed `temperature` from the signature entirely, so passing it
# is a TypeError before a request is even made. Effort is the substitute axis, except on
# Haiku 4.5, which predates output_config and rejects it -- it takes no knobs at all.
ANTHROPIC = [
    ModelSpec("claude-haiku-4-5", "anthropic", 1.00, 5.00, sampling=False, weight=2.0),
    ModelSpec(
        "claude-sonnet-4-6", "anthropic", 3.00, 15.00, sampling=False, weight=1.0, effort=True
    ),
    ModelSpec("claude-sonnet-5", "anthropic", 2.00, 10.00, sampling=False, weight=1.5, effort=True),
    ModelSpec(
        "claude-opus-5",
        "anthropic",
        5.00,
        25.00,
        sampling=False,
        weight=0.4,
        held_out=True,
        effort=True,
    ),
]

OPENAI = [
    ModelSpec("gpt-5.4-nano", "openai", 0.05, 0.40, sampling=True, weight=2.0),
    ModelSpec("gpt-5.4-mini", "openai", 0.25, 2.00, sampling=True, weight=1.5),
    ModelSpec("gpt-5.6-luna", "openai", 0.20, 1.20, sampling=True, weight=2.0),
    ModelSpec("gpt-5.6-terra", "openai", 2.00, 12.00, sampling=True, weight=1.0),
    ModelSpec("gpt-5.6-sol", "openai", 4.00, 20.00, sampling=True, weight=0.4, held_out=True),
]

# Gemma is served through the Gemini API, which buys open-weight lab diversity -- a
# different post-training recipe entirely -- without standing up vLLM.
# thinking_off marks the models that accept an explicit budget of 0. The lite and Gemma
# models reject the parameter outright; gemini-3.1-pro-preview is excluded from the
# roster because it only runs in thinking mode and spends its output budget doing so --
# a probe returned eight words.
GEMINI = [
    ModelSpec("gemini-3.5-flash-lite", "gemini", 0.30, 2.50, sampling=True, weight=2.0),
    ModelSpec(
        "gemini-3.8-flash", "gemini", 0.75, 3.75, sampling=True, weight=1.5, thinking_off=True
    ),
    ModelSpec("gemma-4-31b-it", "gemini", 0.10, 0.40, sampling=True, weight=1.5),
    ModelSpec("gemini-2.5-flash", "gemini", 0.30, 2.50, sampling=True, weight=0.5, held_out=True),
]

ALL_MODELS = [*ANTHROPIC, *OPENAI, *GEMINI]
BY_NAME = {m.name: m for m in ALL_MODELS}

TRAIN_MODELS = [m for m in ALL_MODELS if not m.held_out]
HELD_OUT_MODELS = [m for m in ALL_MODELS if m.held_out]

# Sampling cells, applied where the model accepts them.
SAMPLING_GRID = (
    ("conservative", 0.3, 0.90, 0.15),
    ("default", 0.7, 0.95, 0.35),
    ("default_hi_p", 1.0, 1.00, 0.25),
    ("creative", 1.2, 0.95, 0.15),
    ("very_creative", 1.4, 0.98, 0.10),
)
# Substitute axis for models that reject sampling params, so the grid keeps its shape.
EFFORT_GRID = (("low", 0.5), ("medium", 0.35), ("high", 0.15))


@dataclass
class Generation:
    text: str
    model: str
    provider: Provider
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error

    def cost(self) -> float:
        spec = BY_NAME.get(self.model)
        if spec is None:
            return 0.0
        return (
            self.input_tokens * spec.input_per_mtok + self.output_tokens * spec.output_per_mtok
        ) / 1_000_000


def _max_tokens_for(target_words: int) -> int:
    return min(8000, max(1024, int(target_words * 2.2)))


def generate_anthropic(
    model: str, system: str, user: str, target_words: int, cell: dict[str, Any]
) -> Generation:
    import anthropic

    client = anthropic.Anthropic(max_retries=4)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": _max_tokens_for(target_words),
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    # Prefer low effort over disabling thinking: with thinking disabled these models
    # sometimes leak <thinking> tags into the visible response, which is exactly the
    # artifact this corpus must not contain. Models without effort support take nothing.
    if cell.get("effort"):
        kwargs["output_config"] = {"effort": cell["effort"]}

    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        return Generation("", model, "anthropic", error=f"{type(exc).__name__}: {exc}"[:200])

    if response.stop_reason == "refusal":
        return Generation("", model, "anthropic", error="refusal")
    text = "".join(b.text for b in response.content if b.type == "text")
    return Generation(
        text,
        model,
        "anthropic",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def generate_openai(
    model: str, system: str, user: str, target_words: int, cell: dict[str, Any]
) -> Generation:
    import openai

    client = openai.OpenAI(max_retries=4)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": _max_tokens_for(target_words),
    }
    # Turning reasoning off is what makes the sampling grid available on these models,
    # and it keeps reasoning tokens off the output bill.
    kwargs["reasoning_effort"] = cell.get("reasoning_effort", "none")
    if "temperature" in cell:
        kwargs["temperature"] = cell["temperature"]
        kwargs["top_p"] = cell["top_p"]

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        return Generation("", model, "openai", error=f"{type(exc).__name__}: {exc}"[:200])

    choice = response.choices[0]
    usage = response.usage
    return Generation(
        choice.message.content or "",
        model,
        "openai",
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


def generate_gemini(
    model: str, system: str, user: str, target_words: int, cell: dict[str, Any]
) -> Generation:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    client = genai.Client(api_key=key)

    config: dict[str, Any] = {
        "max_output_tokens": _max_tokens_for(target_words),
        "system_instruction": system,
    }
    if "temperature" in cell:
        config["temperature"] = cell["temperature"]
        config["top_p"] = cell["top_p"]
    # Only some Gemini models accept an explicit zero thinking budget; the lite and
    # Gemma models return 400 INVALID_ARGUMENT for the parameter itself.
    if cell.get("thinking_off"):
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    try:
        response = client.models.generate_content(
            model=model, contents=user, config=types.GenerateContentConfig(**config)
        )
    except Exception as exc:
        return Generation("", model, "gemini", error=f"{type(exc).__name__}: {exc}"[:200])

    usage = getattr(response, "usage_metadata", None)
    return Generation(
        response.text or "",
        model,
        "gemini",
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
    )


GENERATORS = {
    "anthropic": generate_anthropic,
    "openai": generate_openai,
    "gemini": generate_gemini,
}


def generate(
    spec: ModelSpec, system: str, user: str, target_words: int, cell: dict[str, Any]
) -> Generation:
    return GENERATORS[spec.provider](spec.name, system, user, target_words, cell)
