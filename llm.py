"""Turn a plain-text workload description into a validated Workload using Claude."""
import importlib, json, os, re
from dataclasses import asdict, replace
from .core import Workload

MODEL = os.environ.get("AUDIT_MODEL", "claude-sonnet-5-5")


def validate(data: dict) -> Workload:
    """Keep known keys with correct types; clamp ranges; leave the rest at defaults."""
    base = Workload()
    keys = asdict(base).keys()
    clean = {}
    for k, v in data.items():
        if k in keys and isinstance(v, (int, float, bool)):
            clean[k] = type(getattr(base, k))(v)
    w = replace(base, **clean)
    w.util = min(max(w.util, 0.1), 1.0)
    w.trial_frac = min(max(w.trial_frac, 0.01), 1.0)
    w.gpus = max(w.gpus, 1)
    w.pue = max(w.pue, 1.0)
    return w


def extract_workload(text: str) -> Workload:
    anthropic = importlib.import_module("anthropic")  # needs: pip install "carbon-audit[llm]"
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    keys = list(asdict(Workload()).keys())
    msg = client.messages.create(
        model=MODEL, max_tokens=500,
        system="Extract ML workload details. Reply with only a JSON object. "
               f"Allowed keys: {keys}. util and trial_frac are fractions (0-1). "
               "grid is gCO2e/kWh. Omit any key not stated or inferable. Never guess.",
        messages=[{"role": "user", "content": text}],
    )
    raw = re.search(r"\{.*\}", msg.content[0].text, re.S)
    if not raw:
        raise ValueError("Model did not return JSON")
    return validate(json.loads(raw.group(0)))
