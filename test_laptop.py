import pytest  # type: ignore[import-not-found]
pytest.importorskip("sklearn")
import sys
from pathlib import Path

# Make the repository root importable when pytest is run from another directory.
for directory in Path(__file__).resolve().parents:
    if (directory / "carbon_audit" / "laptop_model.py").is_file():
        sys.path.insert(0, str(directory))
        break

from carbon_audit.laptop_model import Laptop, LaptopCarbonModel, synthetic, train, physical_kwh_year  # type: ignore[import-not-found]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("l") / "l.joblib")
    rows, y = synthetic(2000)
    return LaptopCarbonModel(path) if train(rows, y, path) else None


def test_close_to_physics(model):
    l = Laptop()
    p = model.predict(l)
    assert abs(p["kwh_year"] - physical_kwh_year(l)) / physical_kwh_year(l) < 0.25


def test_breakdown_sums_to_one(model):
    assert sum(model.predict(Laptop())["breakdown"].values()) == pytest.approx(1.0)


def test_gaming_laptop_emits_more(model):
    assert model.predict(Laptop(gpu_w=100, gpu_load=0.6, cpu_tdp=55))["elec_kg"] > model.predict(Laptop())["elec_kg"]


def test_alert_levels(model):
    assert model.predict(Laptop(), budget_kg=10000)["level"] == "ok"
    assert model.predict(Laptop(), budget_kg=1)["level"] == "critical"
