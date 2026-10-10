"""Machine-learning model that predicts a training project's energy before you run it.

Predicts total kWh (all phases) from workload settings, with an 80% prediction interval,
then converts to emissions with the grid factor and raises an alert if the budget is at risk.

Train on synthetic data (default, a baseline built from the physical model) or on YOUR logs:
    python -m carbon_audit.model train                      # synthetic baseline
    python -m carbon_audit.model train --csv my_runs.csv    # real measurements
CSV columns: gpu_tdp,gpus,hours,util,host_w,pue,trials,trial_frac,bf16,measured_kwh
"""
import argparse, csv, os
import importlib

np = importlib.import_module("numpy")
from dataclasses import asdict
from .core import Workload, phases, emissions_t

FEATURES = ["gpu_tdp", "gpus", "hours", "util", "host_w", "pue", "trials", "trial_frac", "bf16"]
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "carbon_model.joblib")


def _x(rows):
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    X[:, [0, 1, 2, 4, 6]] = np.log1p(X[:, [0, 1, 2, 4, 6]])  # scale-spanning features
    return X


def synthetic(n=6000, seed=0):
    """Random workloads; target = physical-model kWh with ~8% lognormal measurement noise."""
    rng = np.random.default_rng(seed)
    rows, y = [], []
    for _ in range(n):
        w = Workload(
            gpu_tdp=float(rng.choice([70, 250, 300, 400, 700])), gpus=int(rng.integers(1, 65)),
            hours=float(rng.uniform(0.5, 400)), util=float(rng.uniform(0.3, 1.0)),
            host_w=float(rng.uniform(100, 1500)), pue=float(rng.uniform(1.05, 1.6)),
            trials=int(rng.integers(0, 200)), trial_frac=float(rng.uniform(0.05, 1.0)),
            bf16=bool(rng.integers(0, 2)))
        d = asdict(w); d["bf16"] = float(w.bf16)
        rows.append(d)
        y.append(np.log(sum(phases(w).values()) * rng.lognormal(0, 0.08)))
    return rows, np.array(y)


def from_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, np.log(np.array([float(r["measured_kwh"]) for r in rows]))


def train(rows, y, path=DEFAULT_PATH, seed=0):
    import importlib
    joblib = importlib.import_module("joblib")
    GradientBoostingRegressor = importlib.import_module("sklearn.ensemble").GradientBoostingRegressor
    train_test_split = importlib.import_module("sklearn.model_selection").train_test_split
    X = _x(rows)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    kw = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=seed)
    mid = GradientBoostingRegressor(loss="squared_error", **kw).fit(Xtr, ytr)
    lo = GradientBoostingRegressor(loss="quantile", alpha=0.1, **kw).fit(Xtr, ytr)
    hi = GradientBoostingRegressor(loss="quantile", alpha=0.9, **kw).fit(Xtr, ytr)
    pred = mid.predict(Xte)
    mape = float(np.mean(np.abs(np.exp(pred) - np.exp(yte)) / np.exp(yte)))
    r2 = float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))
    cover = float(np.mean((yte >= lo.predict(Xte)) & (yte <= hi.predict(Xte))))
    joblib.dump({"mid": mid, "lo": lo, "hi": hi}, path)
    return {"r2_log": r2, "mape": mape, "interval_coverage": cover, "n": len(y), "path": path}


class CarbonModel:
    def __init__(self, path=DEFAULT_PATH):
        import importlib
        joblib = importlib.import_module("joblib")
        self.m = joblib.load(path)

    def predict(self, w: Workload) -> dict:
        d = asdict(w); d["bf16"] = float(w.bf16)
        X = _x([d])
        kwh, lo, hi = (float(np.exp(self.m[k].predict(X)[0])) for k in ("mid", "lo", "hi"))
        lo, hi = min(lo, kwh), max(hi, kwh)
        return {"kwh": kwh, "kwh_low": lo, "kwh_high": hi,
                "tco2e": emissions_t(kwh, w.grid), "tco2e_low": emissions_t(lo, w.grid),
                "tco2e_high": emissions_t(hi, w.grid)}

    def alert(self, w: Workload, budget_t: float) -> dict:
        p = self.predict(w)
        if p["tco2e_low"] > budget_t:
            level = "critical"      # even the optimistic estimate is over budget
        elif p["tco2e"] > budget_t or p["tco2e_high"] > budget_t:
            level = "warning"       # likely or possibly over budget
        else:
            level = "ok"
        return {**p, "budget_t": budget_t, "level": level}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows, y = from_csv(a.csv) if a.csv else synthetic()
    print(train(rows, y))
