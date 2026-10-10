"""Energy and emissions model.

Energy (kWh)      = (GPU TDP x GPUs x util + host W) x hours x PUE / 1000
Emissions (gCO2e) = Energy (kWh) x grid intensity (gCO2e/kWh)
"""
from dataclasses import dataclass, replace

# Approximate annual grid averages (gCO2e/kWh). Verify against Electricity Maps / Ember.
GRID = {"mumbai": 700, "us-east-1": 380, "iowa": 430, "oregon": 120,
        "finland": 90, "montreal": 25}


@dataclass
class Workload:
    gpu_tdp: float = 400        # W per accelerator
    gpus: int = 8
    hours: float = 120          # one full training run
    util: float = 0.75          # average draw as a fraction of TDP
    host_w: float = 500         # CPU + RAM
    pue: float = 1.10
    grid: float = 700           # gCO2e/kWh
    trials: int = 100           # hyperparameter trials
    trial_frac: float = 1.0     # trial length as a fraction of a full run
    bf16: bool = False          # already using mixed precision
    inference_kwh: float = 0.0  # total inference energy, if known


def run_kwh(w: Workload) -> float:
    return (w.gpu_tdp * w.gpus * w.util + w.host_w) * w.hours * w.pue / 1000


def phases(w: Workload, pre: float = 0.03, val: float = 0.05) -> dict:
    """kWh per phase. Preprocessing and validation are fractions of one run (assumptions)."""
    r = run_kwh(w)
    return {"preprocessing": r * pre, "training": r,
            "hpo": r * w.trials * w.trial_frac,
            "validation": r * val, "inference": w.inference_kwh}


def emissions_t(kwh: float, grid: float) -> float:
    """Tonnes CO2e."""
    return kwh * grid / 1e6


def apply_fix(w: Workload, region: str = "montreal", max_trials: int = 30,
              pruned_frac: float = 0.25, bf16_speedup: float = 2.0):
    """Bayesian search + pruning, BF16, cached preprocessing, cleaner region.
    Returns (fixed_workload, fixed_phases)."""
    fixed = replace(
        w,
        trials=min(w.trials, max_trials),
        trial_frac=pruned_frac,
        hours=w.hours / (1 if w.bf16 else bf16_speedup),
        bf16=True,
        grid=GRID.get(region, w.grid),
    )
    return fixed, phases(fixed, pre=0.005, val=0.01)


def summarize(w: Workload, **fix_kwargs) -> dict:
    base = phases(w)
    fixed_w, fixed = apply_fix(w, **fix_kwargs)
    tb, tf = sum(base.values()), sum(fixed.values())
    eb, ef = emissions_t(tb, w.grid), emissions_t(tf, fixed_w.grid)
    top = max(base, key=base.get)
    return {
        "baseline_kwh": tb, "baseline_t": eb, "after_kwh": tf, "after_t": ef,
        "reduction": (1 - ef / eb) if eb else 0.0,
        "top_phase": top, "top_share": base[top] / tb if tb else 0.0,
        "phases_baseline": base, "phases_after": fixed,
    }
