#!/usr/bin/env python3
"""ML carbon audit: estimate, apply fixes, optionally parse a plain-text description with an LLM.

Setup:   pip install anthropic codecarbon
Run:     python carbon_audit.py --demo
         python carbon_audit.py "8 A100s in Mumbai for 5 days, grid search 100 trials"
Measure: from carbon_audit import tracked   (see bottom of file)
"""
import argparse, json, os, re
from dataclasses import dataclass, asdict, replace

MODEL = os.environ.get("AUDIT_MODEL", "claude-sonnet-5-5")

# Approximate annual grid averages (gCO2e/kWh). Check Electricity Maps / Ember for current values.
GRID = {"mumbai": 700, "us-east-1": 380, "iowa": 430, "oregon": 120,
        "finland": 90, "montreal": 25}


@dataclass
class Workload:
    gpu_tdp: float = 400        # W per GPU
    gpus: int = 8
    hours: float = 120          # one full training run
    util: float = 0.75          # average draw as a fraction of TDP
    host_w: float = 500         # CPU + RAM
    pue: float = 1.10
    grid: float = 700           # gCO2e/kWh
    trials: int = 100
    trial_frac: float = 1.0     # trial length as a fraction of a full run
    bf16: bool = False
    inference_kwh: float = 0.0  # total inference energy, if known


def run_kwh(w: Workload) -> float:
    """Energy (kWh) = (TDP x GPUs x util + host W) x hours x PUE / 1000"""
    return (w.gpu_tdp * w.gpus * w.util + w.host_w) * w.hours * w.pue / 1000


def phases(w: Workload, pre=0.03, val=0.05) -> dict:
    r = run_kwh(w)
    return {"preprocessing": r * pre, "training": r, "hpo": r * w.trials * w.trial_frac,
            "validation": r * val, "inference": w.inference_kwh}


def emissions_t(kwh: float, grid: float) -> float:
    """Emissions (gCO2e) = kWh x grid intensity; returned in tonnes."""
    return kwh * grid / 1e6


def apply_fix(w: Workload, region="montreal", max_trials=30, pruned_frac=0.25,
              bf16_speedup=2.0) -> tuple[Workload, dict]:
    """Bayesian search + pruning, BF16, cached preprocessing, cleaner region."""
    fixed = replace(
        w,
        trials=min(w.trials, max_trials),
        trial_frac=pruned_frac,
        hours=w.hours / (1 if w.bf16 else bf16_speedup),
        bf16=True,
        grid=GRID.get(region, w.grid),
    )
    return fixed, phases(fixed, pre=0.005, val=0.01)


def report(w: Workload) -> None:
    base = phases(w)
    fixed_w, fixed = apply_fix(w)
    tb, tf = sum(base.values()), sum(fixed.values())
    top = max(base, key=base.get)
    print(f"{'phase':<15}{'baseline kWh':>14}{'after fix kWh':>16}")
    for k in base:
        print(f"{k:<15}{base[k]:>14,.0f}{fixed[k]:>16,.0f}")
    eb, ef = emissions_t(tb, w.grid), emissions_t(tf, fixed_w.grid)
    print(f"\nHighest phase: {top} ({base[top] / tb:.0%} of energy)")
    print(f"Baseline: {tb:,.0f} kWh -> {eb:,.2f} tCO2e @ {w.grid:.0f} g/kWh")
    print(f"After:    {tf:,.0f} kWh -> {ef:,.2f} tCO2e @ {fixed_w.grid:.0f} g/kWh "
          f"({1 - ef / eb:.1%} lower)" if eb else "")


def extract_workload(text: str) -> Workload:
    """Ask Claude to turn a plain-text description into Workload fields."""
    try:
        anthropic = __import__("anthropic")  # imported lazily so --demo works without it
    except ModuleNotFoundError as exc:
        if exc.name == "anthropic":
            raise RuntimeError("Install the optional Anthropic dependency with: pip install anthropic") from exc
        raise
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    keys = list(asdict(Workload()).keys())
    msg = client.messages.create(
        model=MODEL, max_tokens=500,
        system="Extract ML workload details. Reply with only a JSON object. "
               f"Allowed keys: {keys}. util and trial_frac are fractions (0-1). "
               "grid is gCO2e/kWh. Omit any key not stated or inferable. Never guess.",
        messages=[{"role": "user", "content": text}],
    )
    raw = re.search(r"\{.*\}", msg.content[0].text, re.S).group(0)
    data = json.loads(raw)
    base, clean = Workload(), {}
    for k, v in data.items():  # validate: known keys, right types, sane ranges
        if k in keys and isinstance(v, (int, float, bool)):
            clean[k] = type(getattr(base, k))(v)
    w = replace(base, **clean)
    w.util = min(max(w.util, 0.1), 1.0)
    w.trial_frac = min(max(w.trial_frac, 0.01), 1.0)
    return w


# --- Measure real energy instead of estimating -------------------------------
def tracked(train_fn, project="ml-run"):
    """Run train_fn() under CodeCarbon and return measured kgCO2e."""
    codecarbon = __import__("codecarbon")  # optional dependency, loaded only when tracking
    EmissionsTracker = codecarbon.EmissionsTracker
    tracker = EmissionsTracker(project_name=project, measure_power_secs=15)
    tracker.start()
    try:
        train_fn()
    finally:
        kg = tracker.stop()
    print(f"Measured: {kg:.3f} kgCO2e")
    return kg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("description", nargs="?", help="plain-text workload description")
    ap.add_argument("--demo", action="store_true", help="use the example workload, no API call")
    args = ap.parse_args()
    if args.demo or not args.description:
        w = Workload()
    else:
        w = extract_workload(args.description)
        print("Parsed workload:", asdict(w), "\n")
    report(w)
