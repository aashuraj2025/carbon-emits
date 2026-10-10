# carbon-audit

Estimate, reduce and measure the energy and carbon footprint of ML workloads.

```
Energy (kWh)      = (GPU TDP x GPUs x util + host W) x hours x PUE / 1000
Emissions (gCO2e) = Energy (kWh) x grid intensity (gCO2e/kWh)
```

## Layout

| Path | Purpose |
|---|---|
| `src/carbon_audit/core.py` | Energy model, phase breakdown, fix scenario |
| `src/carbon_audit/llm.py` | Plain text to workload fields with Claude |
| `src/carbon_audit/model.py` | ML model that predicts kWh/tCO2e (with 80% interval) and raises budget alerts |
| `src/carbon_audit/device.py` | Detect your own PC/laptop and estimate its power draw |
| `src/carbon_audit/laptop_model.py` | ML model predicting your own laptop/PC yearly carbon emissions |
| `src/carbon_audit/tracking.py` | CodeCarbon measurement wrapper |
| `src/carbon_audit/cli.py` | `carbon-audit` command |
| `src/carbon_audit/api.py` | Optional FastAPI service |
| `web/index.html` | Interactive calculator (open in a browser) |
| `tests/`, `examples/` | Tests and a tracking example |

## Install

```bash
pip install -e ".[dev]"            # core + tests
pip install -e ".[llm,track,api]"  # optional extras
```

## Use

```bash
carbon-audit --demo
carbon-audit --config workload.json --region finland
export ANTHROPIC_API_KEY=...
carbon-audit "8 A100s in Mumbai for 5 days, grid search 100 trials"
uvicorn carbon_audit.api:app --reload      # POST /audit, POST /parse
pytest
```

`workload.json` takes the fields of `Workload`, for example `{"gpus": 4, "hours": 48, "grid": 380}`.

## Assumptions to replace with measurements

- Preprocessing is 3% and validation 5% of one training run.
- The fix scenario assumes a 2x BF16 speedup and pruned trials at 25% length.
- Grid factors in `core.GRID` are rough annual averages. Check Electricity Maps or Ember.
- TDP overstates real draw; set `util` from measured values, or use `tracking.tracked`.

## Prediction model (alerts before you run)

`model.py` trains gradient-boosted regressors (mean + 10th/90th percentile) on workload settings and
predicts total project energy with an 80% interval. `CarbonModel.alert(w, budget_t)` returns
`ok`, `warning` (possibly over budget) or `critical` (even the optimistic estimate is over budget).

```bash
pip install -e ".[ml]"
python -m carbon_audit.model train                      # synthetic baseline
python -m carbon_audit.model train --csv my_runs.csv    # your measured runs (recommended)
carbon-audit --demo --predict --budget-t 5
```

CSV columns: `gpu_tdp,gpus,hours,util,host_w,pue,trials,trial_frac,bf16,measured_kwh`
(log real runs with CodeCarbon). The bundled model is trained on synthetic data generated from the physical
model with 8% noise, so it only reproduces the formula. Retrain on your own measurements for it to be useful.
On the synthetic holdout: R² (log scale) 0.99, MAPE about 15%, and the "80%" interval actually covers about 70%.

## Your own PC or laptop

```bash
carbon-audit --my-device --hours 10 --grid 700
carbon-audit --my-device --hours 10 --cpu-tdp 28    # if the detected power class is wrong
```

`device.py` reads the CPU model, thread count, RAM and (if present) the NVIDIA GPU power limit, guesses the CPU power
class from the model name, and estimates wall power as (CPU x load + GPU x load + RAM + screen/board) / charger
efficiency. Verify the result with a plug-in watt meter or CodeCarbon; the CPU power class is a guess from the name.

## Laptop carbon model (your own machine)

`laptop_model.py` predicts the yearly emissions of a laptop or PC from its configuration and your habits:
CPU power class, RAM, GPU, screen size and brightness, hours per day, days per week, CPU/GPU load and charger
efficiency. Gradient-boosted regressors predict electricity (kWh/year, 80% interval); the code then applies your
grid intensity and adds manufacturing carbon spread over the years you keep the laptop. It also returns a
component breakdown (CPU, GPU, RAM, screen, board) and a budget alert.

```bash
python -m carbon_audit.laptop_model train                    # baseline on synthetic data
python -m carbon_audit.laptop_model train --csv laptops.csv  # your measured data (recommended)
carbon-audit --laptop --hours-day 8 --days-week 5 --grid 700 --years 4 --budget-kg 150
```

CSV columns: `cpu_tdp,ram_gb,gpu_w,screen_in,brightness,hours_day,days_week,cpu_load,gpu_load,charger_eff,measured_kwh_year`
(measure with a plug-in watt meter and scale to a year). The bundled model is trained on synthetic laptops generated
from a component power model with 10% noise, so it reproduces that model; real measurements make it useful.
Synthetic hold-out: R2 (log) 0.986, MAPE about 11%, and the nominal 80% interval covers about 74%.
