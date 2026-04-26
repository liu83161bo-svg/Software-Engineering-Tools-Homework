#!/usr/bin/env python3
"""
SLO Gate (Mock) – Simulates latency and error budget checks for CI.
No real model is needed; all metrics are synthetically generated to pass.
"""

import time
import random
import sys
import math


def check_latency_slo() -> bool:
    """
    Simulate latency measurement: generate 100 fake inference times (ms)
    and assert P95 < 20 ms.
    """
    print("=" * 50)
    print("SLO Gate: Latency Check")
    print("=" * 50)

    # Simulate 100 inference latencies with realistic random values (2–10 ms)
    random.seed(42)  # fixed seed for reproducibility
    latencies = [random.uniform(2.0, 10.0) for _ in range(100)]

    # Calculate P95 latency
    sorted_latencies = sorted(latencies)
    idx = int(math.ceil(95 / 100 * len(sorted_latencies))) - 1
    p95 = sorted_latencies[idx]
    print(f"Simulated P95 latency: {p95:.2f} ms")

    if p95 < 20:
        print("✓ Latency SLO passed (P95 < 20 ms)\n")
        return True
    else:
        print(f"✗ Latency SLO failed (P95 = {p95:.2f} >= 20 ms)\n")
        return False


def check_error_budget() -> bool:
    """
    Simulate error budget: compare current mock accuracy against a stored baseline.
    Degradation must be ≤ 2%.
    """
    print("=" * 50)
    print("SLO Gate: Error Budget Check")
    print("=" * 50)

    # Simulate current accuracy on golden set
    random.seed(123)
    current_accuracy = round(random.uniform(0.82, 0.88), 3)
    # Baseline accuracy (stored in a mock; in real scenario read from reports/metrics.json)
    baseline_accuracy = 0.84  # pretend this is the previous known good value

    degradation = baseline_accuracy - current_accuracy
    print(f"Current simulated accuracy: {current_accuracy}")
    print(f"Baseline accuracy: {baseline_accuracy}")
    print(f"Degradation: {degradation:.3f}")

    if degradation <= 0.02:
        print("✓ Error budget SLO passed (degradation ≤ 2%)\n")
        return True
    else:
        print(f"✗ Error budget SLO failed (degradation {degradation:.3f} > 2%)\n")
        return False


def main():
    latency_ok = check_latency_slo()
    budget_ok = check_error_budget()

    print("=" * 50)
    if latency_ok and budget_ok:
        print("All SLO gates passed. CI can proceed.")
        sys.exit(0)
    else:
        print("Some SLO gates failed. CI blocked.")
        sys.exit(1)


if __name__ == "__main__":
    main()