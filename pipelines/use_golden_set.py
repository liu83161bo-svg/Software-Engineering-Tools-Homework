#!/usr/bin/env python3
"""
Test script to verify golden set usage
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def test_golden_set():
    """Test that the golden set can be loaded and used"""
    golden_path = "../data/golden_set/golden_samples.jsonl"

    if not Path(golden_path).exists():
        print(f"ERROR: Golden set not found at {golden_path}")
        return False

    # Load and validate
    with open(golden_path, 'r') as f:
        samples = [json.loads(line.strip()) for line in f]

    print(f"Loaded {len(samples)} samples from golden set")

    # Basic validation
    errors = []

    for i, sample in enumerate(samples):
        # Check required fields
        if 'age' not in sample:
            errors.append(f"Sample {i}: Missing 'age' field")

        if 'signal' not in sample:
            errors.append(f"Sample {i}: Missing 'signal' field")
        else:
            signal = sample['signal']
            if not isinstance(signal, list):
                errors.append(f"Sample {i}: Signal is not a list")
            elif len(signal) != 1000:
                errors.append(f"Sample {i}: Signal length is {len(signal)}, expected 1000")

            # Check signal values
            if signal and isinstance(signal, list):
                try:
                    arr = np.array(signal, dtype=float)
                    if not np.all(np.isfinite(arr)):
                        errors.append(f"Sample {i}: Signal contains non-finite values")
                except:
                    errors.append(f"Sample {i}: Signal cannot be converted to float array")

        # Check age is integer
        if 'age' in sample and not isinstance(sample['age'], int):
            errors.append(f"Sample {i}: Age is not integer: {sample['age']}")

    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
        return False

    # Statistics
    ages = [s['age'] for s in samples]
    unique_ages = set(ages)

    print(f"\nGolden Set Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Unique ages: {len(unique_ages)}")
    print(f"  Age range: {min(ages)} - {max(ages)}")

    # Age distribution
    print(f"\nAge distribution:")
    age_counts = {}
    for age in ages:
        age_counts[age] = age_counts.get(age, 0) + 1

    for age in sorted(age_counts.keys()):
        print(f"  Age {age}: {age_counts[age]} samples")

    # Signal statistics
    signals = [np.array(s['signal']) for s in samples]
    if signals:
        signal_means = [s.mean() for s in signals]
        signal_stds = [s.std() for s in signals]

        print(f"\nSignal statistics:")
        print(f"  Mean of means: {np.mean(signal_means):.6f}")
        print(f"  Std of means: {np.std(signal_means):.6f}")
        print(f"  Mean of stds: {np.mean(signal_stds):.6f}")

    print(f"\n✓ Golden set validation passed!")
    return True


def create_dvc_tracking():
    """Create DVC tracking for golden set"""
    import subprocess
    import os

    golden_dir = "./data/golden_set"

    # Add golden set to DVC if not already tracked
    files_to_track = [
        "golden_samples.jsonl",
        "golden_samples.csv",
        "metadata.json",
        "golden_set_report.txt"
    ]

    for filename in files_to_track:
        filepath = os.path.join(golden_dir, filename)
        if os.path.exists(filepath):
            try:
                result = subprocess.run(
                    ["dvc", "add", filepath],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"✓ Added {filename} to DVC")
                else:
                    print(f"⚠ Could not add {filename} to DVC: {result.stderr}")
            except Exception as e:
                print(f"⚠ DVC command failed for {filename}: {e}")
        else:
            print(f"⚠ File not found: {filepath}")


if __name__ == "__main__":
    print("Testing Golden Set Usage")
    print("=" * 50)

    success = test_golden_set()

    if success:
        print("\n" + "=" * 50)
        print("Golden set is ready for use!")
        print("\nUsage examples:")
        print("  1. For evaluation: python scripts/eval_gate.py")
        print("  2. For DVC tracking: python scripts/use_golden_set.py --dvc")
        print("  3. For CI testing: python scripts/create_golden_set.py --ci")
    else:
        print("\n✗ Golden set validation failed!")
        print("Try regenerating: python scripts/create_golden_set.py")

    # Check for DVC flag
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--dvc":
        create_dvc_tracking()