import math
from carbon_audit import Workload, run_kwh, phases, emissions_t, apply_fix
from carbon_audit import summarize
from carbon_audit import validate


def test_run_kwh_matches_formula():
    w = Workload()  # (400*8*0.75 + 500) * 120 * 1.10 / 1000
    assert math.isclose(run_kwh(w), 382.8, rel_tol=1e-4)


def test_hpo_dominates_grid_search():
    p = phases(Workload())
    assert max(p, key=p.get) == "hpo"


def test_emissions_units():
    assert math.isclose(emissions_t(1000, 700), 0.7, rel_tol=1e-6)


def test_fix_reduces_energy_and_emissions():
    w = Workload()
    s = summarize(w)
    assert s["after_kwh"] < s["baseline_kwh"] * 0.1
    assert s["reduction"] > 0.99


def test_bf16_not_double_counted():
    w = Workload(bf16=True)
    fixed, _ = apply_fix(w)
    assert fixed.hours == w.hours


def test_validate_drops_bad_values():
    w = validate({"gpus": 4, "util": 5, "bogus": 1, "hours": "x"})
    assert w.gpus == 4 and w.util == 1.0 and w.hours == Workload().hours


def test_device_watts_and_cpu_guess():
    from carbon_audit import Device, device_watts, guess_cpu_tdp, workload_from_device
    assert guess_cpu_tdp("Intel(R) Core(TM) i7-1355U") == 15
    assert guess_cpu_tdp("12th Gen Intel(R) Core(TM) i7-12700H") == 45
    assert guess_cpu_tdp("Apple M2") == 20
    d = Device(cpu_tdp=28, ram_gb=16)  # (28*0.5 + 0 + 16*0.375 + 8) / 0.85
    assert math.isclose(device_watts(d), 32.94, rel_tol=1e-3)
    w = workload_from_device(d, 10)
    assert w.trials == 0 and w.pue == 1.0
