"""Measure real energy with CodeCarbon instead of estimating."""
from contextlib import contextmanager
from importlib import import_module


@contextmanager
def tracked(project: str = "ml-run", every_s: int = 15):
    EmissionsTracker = import_module("codecarbon").EmissionsTracker
    tracker = EmissionsTracker(project_name=project, measure_power_secs=every_s)
    result = {"kg_co2e": None}
    tracker.start()
    try:
        yield result
    finally:
        result["kg_co2e"] = tracker.stop()
