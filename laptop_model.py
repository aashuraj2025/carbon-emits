"""Machine-learning model that predicts the carbon emissions of YOUR laptop or PC.

Pipeline: laptop configuration + usage habits -> predicted electricity use (kWh/year, with an 80%
interval) -> emissions using your grid's carbon intensity -> plus manufacturing carbon spread over
the years you keep it -> total kgCO2e/year, a budget alert and a component breakdown.

Train on synthetic data (baseline) or on your own measurements:
    python -m carbon_audit.laptop_model train
    python -m carbon_audit.laptop_model train --csv my_laptops.csv
CSV columns: cpu_tdp,ram_gb,gpu_w,screen_in,brightness,hours_day,days_week,cpu_load,gpu_load,charger_eff,measured_kwh_year
"""
import argparse, csv, os
from dataclasses import dataclass, asdict
import numpy as np  # type: ignore[import-not-found]

FEATURES = ["cpu_tdp", "ram_gb", "gpu_w", "screen_in", "brightness", "hours_day",
            "days_week", "cpu_load", "gpu_load", "charger_eff"]
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "laptop_model.joblib")


@dataclass
class Laptop:
    cpu_tdp: float = 28        # W power class of the CPU
    ram_gb: float = 16
    gpu_w: float = 0           # discrete GPU power limit, 0 if none
    screen_in: float = 14
    brightness: float = 0.6    # 0-1
    hours_day: float = 8
    days_week: float = 5
    cpu_load: float = 0.35     # average CPU load 0-1
    gpu_load: float = 0.0      # average GPU load 0-1
    charger_eff: float = 0.85
    grid: float = 700          # gCO2e/kWh
    years: float = 4           # years you keep the laptop
    embodied_kg: float = 300   # manufacturing footprint, kgCO2e


def components_w(l: Laptop) -> dict:
    """Average component power (W) while in use: the physical model used to generate training data."""
    gpu = l.gpu_w * (0.05 + 0.8 * l.gpu_load) if l.gpu_w > 0 else 0.0
    parts = {"CPU": l.cpu_tdp * (0.1 + 0.7 * l.cpu_load), "GPU": gpu, "RAM": l.ram_gb * 0.375,
             "Screen": 0.9 * l.screen_in * (0.3 + 0.7 * l.brightness), "Board & other": 4.0}
    return {k: v / max(l.charger_eff, 0.5) for k, v in parts.items()}  # wall power incl. charger loss


def physical_kwh_year(l: Laptop) -> float:
    return sum(components_w(l).values()) * l.hours_day * l.days_week * 52 / 1000


def _x(rows):
    return np.array([[float(r[f]) for f in FEATURES] for r in rows])


def synthetic(n=6000, seed=0):
    """Random laptops/PCs; target = physical-model kWh/year with ~10% log-normal measurement noise."""
    rng = np.random.default_rng(seed)
    rows, y = [], []
    for _ in range(n):
        gpu_w = float(rng.choice([0, 0, 0, 60, 80, 100, 170, 250]))
        l = Laptop(cpu_tdp=float(rng.choice([15, 20, 28, 40, 45, 55, 65, 95, 125])),
                   ram_gb=float(rng.choice([8, 16, 32, 64])), gpu_w=gpu_w,
                   screen_in=float(rng.choice([13, 14, 15.6, 16, 17.3])),
                   brightness=float(rng.uniform(0.2, 1)), hours_day=float(rng.uniform(0.5, 16)),
                   days_week=float(rng.integers(1, 8)), cpu_load=float(rng.uniform(0.05, 0.9)),
                   gpu_load=float(rng.uniform(0, 0.8)) if gpu_w else 0.0,
                   charger_eff=float(rng.uniform(0.75, 0.95)))
        rows.append(asdict(l))
        y.append(np.log(physical_kwh_year(l) * rng.lognormal(0, 0.10)))
    return rows, np.array(y)


def from_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, np.log(np.array([float(r["measured_kwh_year"]) for r in rows]))


def train(rows, y, path=DEFAULT_PATH, seed=0):
    from importlib import import_module
    try:
        joblib = import_module("joblib")
    except ModuleNotFoundError as exc:
        raise ImportError("Install joblib to train or load the laptop model (pip install joblib).") from exc
    sklearn_ensemble = import_module("sklearn.ensemble")
    sklearn_model_selection = import_module("sklearn.model_selection")
    GradientBoostingRegressor = sklearn_ensemble.GradientBoostingRegressor
    train_test_split = sklearn_model_selection.train_test_split
    X = _x(rows)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    kw = dict(n_estimators=250, max_depth=3, learning_rate=0.06, subsample=0.8, random_state=seed)
    mid = GradientBoostingRegressor(loss="squared_error", **kw).fit(Xtr, ytr)
    lo = GradientBoostingRegressor(loss="quantile", alpha=0.1, **kw).fit(Xtr, ytr)
    hi = GradientBoostingRegressor(loss="quantile", alpha=0.9, **kw).fit(Xtr, ytr)
    p = mid.predict(Xte)
    mape = float(np.mean(np.abs(np.exp(p) - np.exp(yte)) / np.exp(yte)))
    r2 = float(1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2))
    cover = float(np.mean((yte >= lo.predict(Xte)) & (yte <= hi.predict(Xte))))
    joblib.dump({"mid": mid, "lo": lo, "hi": hi}, path)
    return {"r2_log": r2, "mape": mape, "interval_coverage": cover, "n": len(y)}


class LaptopCarbonModel:
    def __init__(self, path=DEFAULT_PATH):
        from importlib import import_module
        try:
            joblib = import_module("joblib")
        except ModuleNotFoundError as exc:
            raise ImportError("Install joblib to train or load the laptop model (pip install joblib).") from exc
        self.m = joblib.load(path)

    def predict(self, l: Laptop, budget_kg: float = None) -> dict:
        X = _x([asdict(l)])
        kwh, lo, hi = (float(np.exp(self.m[k].predict(X)[0])) for k in ("mid", "lo", "hi"))
        lo, hi = min(lo, kwh), max(hi, kwh)
        man = l.embodied_kg / max(l.years, 1)
        f = lambda k: k * l.grid / 1000
        out = {"kwh_year": kwh, "kwh_low": lo, "kwh_high": hi,
               "elec_kg": f(kwh), "embodied_kg_year": man,
               "total_kg_year": f(kwh) + man, "total_low": f(lo) + man, "total_high": f(hi) + man,
               "lifetime_kg": (f(kwh)) * l.years + l.embodied_kg}
        out["km_car_equiv"] = out["total_kg_year"] / 0.17
        c = components_w(l); s = sum(c.values()) or 1
        out["breakdown"] = {k: v / s for k, v in c.items()}
        if budget_kg is not None:
            out["level"] = ("critical" if out["total_low"] > budget_kg else
                            "warning" if (out["total_kg_year"] > budget_kg or out["total_high"] > budget_kg)
                            else "ok")
            out["budget_kg"] = budget_kg
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows, y = from_csv(a.csv) if a.csv else synthetic()
    print(train(rows, y))
