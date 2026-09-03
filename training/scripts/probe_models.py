"""Send one tiny request per (model, sampling mode) and report what actually works.

Worth the few cents: a model id or parameter that is wrong fails once here instead of
150,000 times mid-run, and provider constraints on sampling and reasoning change often
enough that a hardcoded table goes stale.

    uv run modal run scripts/probe_models.py
"""

from __future__ import annotations

import modal

app = modal.App("slopmarker-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("anthropic", "openai", "google-genai")
    .add_local_python_source("slopmarker")
)

SECRETS = [
    modal.Secret.from_name("anthropic-api-key"),
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("gemini-api-secret"),
]

SYSTEM = "You are writing a short paragraph for a website. Output only the paragraph."
USER = "Write about the history of the wheelbarrow."


@app.function(image=image, secrets=SECRETS, timeout=1800)
def probe() -> list[dict]:
    from slopmarker.corpus.providers import ALL_MODELS, generate

    results = []
    for spec in ALL_MODELS:
        cells = [("plain", {})]
        if spec.sampling:
            cells.append(("sampling", {"temperature": 0.7, "top_p": 0.95}))
        if spec.effort:
            cells.append(("effort", {"effort": "low"}))
        if spec.thinking_off:
            cells.append(
                ("thinking_off", {"thinking_off": True, "temperature": 0.9, "top_p": 0.95})
            )
        for label, cell in cells:
            gen = generate(spec, SYSTEM, USER, 60, cell)
            results.append(
                {
                    "model": spec.name,
                    "provider": spec.provider,
                    "mode": label,
                    "ok": gen.ok,
                    "error": gen.error,
                    "words": len(gen.text.split()),
                    "in": gen.input_tokens,
                    "out": gen.output_tokens,
                    "cost": round(gen.cost(), 6),
                    "sample": gen.text[:100].replace("\n", " "),
                }
            )
    return results


@app.local_entrypoint()
def main() -> None:
    rows = probe.remote()
    print(f"{'model':26} {'mode':9} {'ok':3} {'words':6} {'out':6} {'cost':9} note")
    total = 0.0
    for r in rows:
        total += r["cost"]
        note = r["error"] if not r["ok"] else r["sample"][:44]
        print(
            f"{r['model']:26} {r['mode']:9} {r['ok']!s:3} {r['words']:<6} "
            f"{r['out']:<6} {r['cost']:<9.6f} {note}"
        )
    print(f"\nprobe cost: ${total:.4f}")

    working = {}
    for r in rows:
        if r["ok"]:
            working.setdefault(r["model"], []).append(r["mode"])
    print("\nusable models:")
    for model, modes in sorted(working.items()):
        print(f"   {model:26} {modes}")
    broken = sorted({r["model"] for r in rows} - set(working))
    if broken:
        print(f"\nUNUSABLE: {broken}")
