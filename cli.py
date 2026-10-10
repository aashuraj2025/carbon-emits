import argparse, json
from dataclasses import asdict
from .core import Workload, summarize


def main(argv=None):
    ap = argparse.ArgumentParser(prog="carbon-audit", description="ML carbon & energy audit")
    ap.add_argument("description", nargs="?", help="plain-text workload description (needs the llm extra)")
    ap.add_argument("--demo", action="store_true", help="use the example workload")
    ap.add_argument("--config", help="JSON file with Workload fields")
    ap.add_argument("--region", default="montreal", help="target region for the fix scenario")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--predict", action="store_true", help="predict emissions with the trained ML model and alert against --budget-t")
    ap.add_argument("--budget-t", type=float, default=5.0, help="carbon budget in tonnes CO2e (with --predict)")
    ap.add_argument("--my-device", action="store_true", help="detect this PC/laptop and audit a run on it")
    ap.add_argument("--hours", type=float, default=10, help="run length in hours (with --my-device)")
    ap.add_argument("--grid", type=float, default=700, help="grid gCO2e/kWh (with --my-device)")
    ap.add_argument("--cpu-tdp", type=float, help="override detected CPU power class in watts")
    ap.add_argument("--laptop", action="store_true", help="predict YOUR laptop's yearly emissions with the ML model")
    ap.add_argument("--hours-day", type=float, default=8)
    ap.add_argument("--days-week", type=float, default=5)
    ap.add_argument("--cpu-load", type=float, default=0.35, help="average CPU load 0-1")
    ap.add_argument("--screen", type=float, default=14, help="screen size in inches")
    ap.add_argument("--years", type=float, default=4, help="years you keep the laptop")
    ap.add_argument("--embodied-kg", type=float, default=300, help="manufacturing footprint kgCO2e")
    ap.add_argument("--budget-kg", type=float, default=150, help="yearly budget in kgCO2e (with --laptop)")
    a = ap.parse_args(argv)

    if a.laptop:
        from .device import detect, describe
        from .laptop_model import Laptop, LaptopCarbonModel
        d = detect()
        if a.cpu_tdp:
            d.cpu_tdp = a.cpu_tdp
        print("This device:", describe(d))
        l = Laptop(cpu_tdp=d.cpu_tdp, ram_gb=d.ram_gb or 16, gpu_w=d.gpu_w, screen_in=a.screen,
                   hours_day=a.hours_day, days_week=a.days_week, cpu_load=a.cpu_load,
                   grid=a.grid, years=a.years, embodied_kg=a.embodied_kg)
        r = LaptopCarbonModel().predict(l, a.budget_kg)
        print(f"\nElectricity: {r['kwh_year']:.0f} kWh/year ({r['kwh_low']:.0f} to {r['kwh_high']:.0f})")
        print(f"Emissions:   {r['elec_kg']:.0f} kg from electricity + {r['embodied_kg_year']:.0f} kg manufacturing"
              f" = {r['total_kg_year']:.0f} kgCO2e/year ({r['total_low']:.0f} to {r['total_high']:.0f})")
        print(f"Lifetime:    {r['lifetime_kg']:.0f} kgCO2e over {a.years:g} years (about {r['km_car_equiv']:.0f} km by car per year)")
        print("Breakdown:   " + ", ".join(f"{k} {v:.0%}" for k, v in r["breakdown"].items()))
        print(f"ALERT [{r['level'].upper()}] yearly budget {a.budget_kg:g} kgCO2e")
        return
    if a.my_device:
        from .device import detect, describe, workload_from_device, device_watts
        d = detect()
        if a.cpu_tdp:
            d.cpu_tdp = a.cpu_tdp
        print("This device:", describe(d))
        print(f"Estimated draw while training: {device_watts(d):.0f} W (CPU at 50%, charger 85%)\n")
        w = workload_from_device(d, a.hours, a.grid)
    elif a.config:
        with open(a.config) as f:
            w = Workload(**json.load(f))
    elif a.description and not a.demo:
        from .llm import extract_workload
        w = extract_workload(a.description)
        print("Parsed workload:", asdict(w), "\n")
    else:
        w = Workload()

    if a.predict:
        from .model import CarbonModel
        r = CarbonModel().alert(w, a.budget_t)
        print(f"Predicted: {r['kwh']:,.0f} kWh ({r['kwh_low']:,.0f} to {r['kwh_high']:,.0f}) -> "
              f"{r['tco2e']:.2f} tCO2e ({r['tco2e_low']:.2f} to {r['tco2e_high']:.2f}), 80% interval")
        print(f"ALERT [{r['level'].upper()}] budget {r['budget_t']} tCO2e")
        return
    s = summarize(w, region=a.region)
    if a.json:
        print(json.dumps(s, indent=2))
        return
    d = 0 if s["baseline_kwh"] >= 100 else 3  # show decimals for small (laptop-scale) runs
    print(f"{'phase':<15}{'baseline kWh':>14}{'after fix kWh':>16}")
    for k, v in s["phases_baseline"].items():
        print(f"{k:<15}{v:>14,.{d}f}{s['phases_after'][k]:>16,.{d}f}")
    print(f"\nHighest phase: {s['top_phase']} ({s['top_share']:.0%} of energy)")
    print(f"Baseline: {s['baseline_kwh']:,.{d}f} kWh -> {s['baseline_t']*1000 if d else s['baseline_t']:,.{2 if not d else 3}f} {'kgCO2e' if d else 'tCO2e'}")
    print(f"After:    {s['after_kwh']:,.{d}f} kWh -> {s['after_t']*1000 if d else s['after_t']:,.{2 if not d else 3}f} {'kgCO2e' if d else 'tCO2e'} ({s['reduction']:.1%} lower)")


if __name__ == "__main__":
    main()
