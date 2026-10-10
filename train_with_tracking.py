"""Wrap a real training function to measure energy. Replace train() with yours."""
import time
from carbon_audit.tracking import tracked  # type: ignore[import-not-found]


def train():
    time.sleep(2)  # your training loop


with tracked("demo-run") as r:
    train()
print(f"Measured: {r['kg_co2e']:.4f} kgCO2e")
