"""Targeted probe for the models the main probe rejected."""

from __future__ import annotations

import modal

app = modal.App("slopmarker-probe-fix")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "anthropic", "openai", "google-genai"
)
SECRETS = [
    modal.Secret.from_name("anthropic-api-key"),
    modal.Secret.from_name("gemini-api-secret"),
]

SYSTEM = "You are writing a short paragraph for a website. Output only the paragraph."
USER = "Write about the history of the wheelbarrow."


@app.function(image=image, secrets=SECRETS, timeout=1200)
def check() -> list[str]:
    import os

    import anthropic
    from google import genai
    from google.genai import types

    out: list[str] = []

    out.append(f"anthropic sdk version: {anthropic.__version__}")
    client = anthropic.Anthropic(max_retries=2)
    for model in ("claude-haiku-4-5", "claude-haiku-4-5-20251001"):
        try:
            r = client.messages.create(
                model=model,
                max_tokens=200,
                system=SYSTEM,
                messages=[{"role": "user", "content": USER}],
            )
            out.append(f"  {model}: OK out={r.usage.output_tokens}")
        except Exception as exc:
            out.append(f"  {model}: {type(exc).__name__}: {str(exc)[:160]}")

    key = os.environ.get("GEMINI_API_KEY") or ""
    gclient = genai.Client(api_key=key)
    variants = [
        ("gemini-3.5-flash-lite", True),
        ("gemini-3.5-flash-lite", False),
        ("gemini-flash-lite-latest", False),
        ("gemini-3.1-flash-lite", False),
        ("gemini-3.1-pro-preview", False),
        ("gemini-2.5-flash-lite", False),
    ]
    for model, with_budget in variants:
        cfg: dict = {"max_output_tokens": 400, "system_instruction": SYSTEM, "temperature": 0.7}
        if with_budget:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        try:
            r = gclient.models.generate_content(
                model=model, contents=USER, config=types.GenerateContentConfig(**cfg)
            )
            words = len((r.text or "").split())
            out.append(f"  {model} budget0={with_budget}: OK words={words}")
        except Exception as exc:
            out.append(f"  {model} budget0={with_budget}: {type(exc).__name__}: {str(exc)[:150]}")
    return out


@app.local_entrypoint()
def main() -> None:
    for line in check.remote():
        print(line)
