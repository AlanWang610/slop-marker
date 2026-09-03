"""Ask each provider which models it actually serves.

Model IDs and their constraints drift faster than any table can be maintained, and a
wrong id fails 150k times rather than once. The API keys live in Modal secrets, so this
runs there and prints what comes back.

    uv run modal run scripts/discover_models.py
"""

from __future__ import annotations

import modal

app = modal.App("slopmarker-discover")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "anthropic", "openai", "google-genai"
)

SECRETS = [
    modal.Secret.from_name("anthropic-api-key"),
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("gemini-api-secret"),
]


@app.function(image=image, secrets=SECRETS, timeout=600)
def discover() -> dict[str, list[str]]:
    import os

    out: dict[str, list[str]] = {}

    try:
        import anthropic

        client = anthropic.Anthropic()
        out["anthropic"] = [m.id for m in client.models.list(limit=40)]
    except Exception as exc:
        out["anthropic"] = [f"ERROR {type(exc).__name__}: {exc}"[:300]]

    try:
        import openai

        client = openai.OpenAI()
        ids = sorted(m.id for m in client.models.list())
        out["openai"] = [m for m in ids if m.startswith("gpt") or m.startswith("o")]
    except Exception as exc:
        out["openai"] = [f"ERROR {type(exc).__name__}: {exc}"[:300]]

    try:
        from google import genai

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        client = genai.Client(api_key=key)
        names = []
        for model in client.models.list():
            actions = getattr(model, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(model.name.replace("models/", ""))
        out["gemini"] = sorted(names)
    except Exception as exc:
        out["gemini"] = [f"ERROR {type(exc).__name__}: {exc}"[:300]]

    out["env_keys"] = sorted(
        k
        for k in os.environ
        if any(t in k.upper() for t in ("ANTHROPIC", "OPENAI", "GEMINI", "GOOGLE"))
    )
    return out


@app.local_entrypoint()
def main() -> None:
    result = discover.remote()
    for provider, models in result.items():
        print(f"\n=== {provider} ({len(models)}) ===")
        for name in models:
            print(f"   {name}")
