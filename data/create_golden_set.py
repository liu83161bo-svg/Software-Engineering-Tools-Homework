#!/usr/bin/env python3
"""
Create Golden Set for HW4 - EEG Age Classification
Using our simulated data from HW3
"""

import numpy as np
import pandas as pd
import json
import os
import random
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def load_simulated_data():
    """Load our simulated EEG data from HW3"""
    csv_path = "./data/sample_eeg_data.csv"
    jsonl_path = "./data/sample_eeg_data.jsonl"

    print("Loading simulated data from HW3...")

    if os.path.exists(csv_path):
        # Load CSV data (metadata only)
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV: {len(df)} samples")

        # Load JSONL for full signals
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                jsonl_data = [json.loads(line.strip()) for line in f]

            # Match signals with metadata
            samples = []
            for i, row in df.iterrows():
                if i < len(jsonl_data):
                    json_record = jsonl_data[i]

                    # Verify age matches
                    if int(row['age']) == json_record.get('age', row['age']):
                        samples.append({
                            'sample_id': f"sim_{row['trial_id']}",
                            'file_name': row.get('file_name', f"sim_{row['trial_id']}.mat"),
                            'trial_index': int(row.get('trial_index', 0)),
                            'age': int(row['age']),
                            'subject_hash': row.get('subject_hash', f"hash_{row['trial_id']}"),
                            'recording_date': row.get('recording_date', '2023-01-01'),
                            'signal': json_record.get('signal', []),
                            'quality_score': 0.95,  # Simulated data is high quality
                            'is_synthetic': False,
                            'notes': 'Generated from HW3 simulated data'
                        })
                else:
                    # Create signal from CSV signal columns
                    signal = []
                    for col in row.index:
                        if 'signal_' in col:
                            signal.append(float(row[col]))

                    # Fill to 1000 points if needed
                    if len(signal) < 1000:
                        signal = signal + [0.0] * (1000 - len(signal))

                    samples.append({
                        'sample_id': f"sim_{row['trial_id']}",
                        'file_name': row.get('file_name', f"sim_{row['trial_id']}.mat"),
                        'trial_index': int(row.get('trial_index', 0)),
                        'age': int(row['age']),
                        'subject_hash': row.get('subject_hash', f"hash_{row['trial_id']}"),
                        'recording_date': row.get('recording_date', '2023-01-01'),
                        'signal': signal[:1000],  # Ensure exactly 1000 points
                        'quality_score': 0.90,
                        'is_synthetic': False,
                        'notes': 'Generated from CSV columns'
                    })

            print(f"Successfully loaded {len(samples)} samples from simulated data")
            return samples
        else:
            print(f"JSONL file not found at {jsonl_path}")
    else:
        print(f"CSV file not found at {csv_path}")

    # Fallback to synthetic generation
    return create_fallback_synthetic_set()

def create_fallback_synthetic_set():
    """Create synthetic data if simulated data is not available"""
    print("Creating synthetic golden set as fallback...")

    samples = []
    # Same age distribution as our simulated data
    ages = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 26, 47]

    # Create 2-3 samples per age (total 40)
    for age in ages:
        num_samples = random.choice([2, 3])

        for i in range(num_samples):
            # Generate synthetic EEG signal similar to our simulated data
            t = np.linspace(0, 1, 1000)

            # Age-specific patterns (consistent with HW3 simulation)
            base_freq = 10 + (age % 5)  # Vary frequency slightly by age
            alpha = np.sin(2 * np.pi * base_freq * t)
            beta = 0.5 * np.sin(2 * np.pi * (base_freq * 2) * t)
            noise = 0.2 * np.random.randn(1000)
            signal = alpha + beta + noise

            # Normalize (consistent with preprocessing)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            samples.append({
                'sample_id': f"synth_{age}_{i}",
                'file_name': f"synth_{age}_{i}.mat",
                'trial_index': i,
                'age': age,
                'subject_hash': f"hash_synth_{age}_{i}",
                'recording_date': f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                'signal': signal.tolist(),
                'quality_score': 0.85,
                'is_synthetic': True,
                'notes': f'Synthetic sample for age {age} (fallback)'
            })

    return samples

def select_golden_samples(samples, target_size=40):
    """Select representative samples for golden set"""
    print(f"Selecting {target_size} samples from {len(samples)} available...")

    # Group by age
    age_groups = {}
    for sample in samples:
        age = sample['age']
        if age not in age_groups:
            age_groups[age] = []
        age_groups[age].append(sample)

    # Calculate how many samples per age
    num_ages = len(age_groups)
    samples_per_age = max(2, target_size // num_ages)

    # Select samples
    golden_samples = []
    remaining_target = target_size

    # First pass: select from each age group
    for age, group_samples in age_groups.items():
        if remaining_target <= 0:
            break

        # Select min(samples_per_age, len(group), ceil(remaining_target/remaining_groups))
        num_to_select = min(samples_per_age, len(group_samples))
        num_to_select = min(num_to_select, remaining_target)

        if num_to_select > 0:
            # Prioritize high quality and non-synthetic
            group_samples.sort(key=lambda x: (-x['quality_score'], not x.get('is_synthetic', False)))
            selected = group_samples[:num_to_select]
            golden_samples.extend(selected)
            remaining_target -= num_to_select

    # Second pass: if we still need more, add from largest groups
    if remaining_target > 0:
        for age, group_samples in age_groups.items():
            if remaining_target <= 0:
                break

            # Check if we already selected from this group
            already_selected = sum(1 for s in golden_samples if s['age'] == age)
            available = len(group_samples) - already_selected

            if available > 0:
                num_to_select = min(1, remaining_target)
                # Get next best samples from this group
                group_samples.sort(key=lambda x: (-x['quality_score'], not x.get('is_synthetic', False)))
                remaining = [s for s in group_samples if s not in golden_samples]

                if remaining:
                    golden_samples.append(remaining[0])
                    remaining_target -= 1

    print(f"Selected {len(golden_samples)} samples for golden set")
    return golden_samples

def analyze_golden_set(samples):
    """Analyze and print statistics about golden set"""
    print("\nGolden Set Analysis:")
    print("-" * 40)

    # Age distribution
    ages = [s['age'] for s in samples]
    unique_ages = set(ages)

    print(f"Total samples: {len(samples)}")
    print(f"Unique ages: {len(unique_ages)}")
    print(f"Age range: {min(ages)} - {max(ages)}")

    # Count by age
    age_counts = {}
    for age in ages:
        age_counts[age] = age_counts.get(age, 0) + 1

    print("\nAge distribution:")
    for age in sorted(age_counts.keys()):
        print(f"  Age {age}: {age_counts[age]} samples")

    # Quality analysis
    quality_scores = [s.get('quality_score', 0) for s in samples]
    print(f"\nQuality scores: min={min(quality_scores):.2f}, "
          f"avg={np.mean(quality_scores):.2f}, max={max(quality_scores):.2f}")

    # Synthetic vs real
    synthetic_count = sum(1 for s in samples if s.get('is_synthetic', False))
    print(f"Synthetic samples: {synthetic_count} ({synthetic_count/len(samples)*100:.1f}%)")

    # Signal length analysis
    signal_lengths = [len(s['signal']) for s in samples]
    if signal_lengths:
        print(f"Signal length: all={len(set(signal_lengths)) == 1}, "
              f"length={signal_lengths[0] if len(set(signal_lengths)) == 1 else 'variable'}")

    return age_counts

def save_golden_set(samples, output_dir="./data/golden_set"):
    """Save golden set in multiple formats"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. JSONL format (primary)
    jsonl_path = os.path.join(output_dir, "golden_samples.jsonl")
    with open(jsonl_path, 'w') as f:
        for sample in samples:
            # Clean sample for JSON serialization
            clean_sample = {
                'sample_id': sample['sample_id'],
                'age': sample['age'],
                'signal': sample['signal'],
                'metadata': {
                    'file_name': sample.get('file_name', ''),
                    'trial_index': sample.get('trial_index', 0),
                    'subject_hash': sample.get('subject_hash', ''),
                    'recording_date': sample.get('recording_date', ''),
                    'quality_score': sample.get('quality_score', 0),
                    'is_synthetic': sample.get('is_synthetic', False),
                    'notes': sample.get('notes', '')
                }
            }
            f.write(json.dumps(clean_sample) + '\n')

    print(f"✓ Saved JSONL to {jsonl_path}")

    # 2. CSV format (for easy viewing)
    csv_data = []
    for sample in samples:
        # Extract first 5 signal points for CSV
        signal = sample['signal']
        signal_preview = signal[:5] if len(signal) >= 5 else signal

        csv_data.append({
            'sample_id': sample['sample_id'],
            'age': sample['age'],
            'file_name': sample.get('file_name', ''),
            'subject_hash': sample.get('subject_hash', ''),
            'quality_score': sample.get('quality_score', 0),
            'is_synthetic': sample.get('is_synthetic', False),
            'signal_length': len(signal),
            'signal_mean': np.mean(signal) if signal else 0,
            'signal_std': np.std(signal) if signal else 0,
            'signal_preview': str(signal_preview),
            'notes': sample.get('notes', '')[:100]  # Truncate long notes
        })

    csv_path = os.path.join(output_dir, "golden_samples.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"✓ Saved CSV to {csv_path}")

    # 3. Metadata summary
    metadata = {
        'creation_date': pd.Timestamp.now().isoformat(),
        'total_samples': len(samples),
        'age_distribution': {age: sum(1 for s in samples if s['age'] == age)
                            for age in set(s['age'] for s in samples)},
        'source': 'HW3 simulated data' if not any(s.get('is_synthetic', False) for s in samples)
                 else 'Mixed (simulated + synthetic)',
        'description': 'Golden set for HW4 evaluation gate testing',
        'signal_length': len(samples[0]['signal']) if samples else 0,
        'quality_stats': {
            'min': min(s.get('quality_score', 0) for s in samples),
            'mean': np.mean([s.get('quality_score', 0) for s in samples]),
            'max': max(s.get('quality_score', 0) for s in samples)
        }
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")

    return jsonl_path, csv_path, metadata_path

def create_golden_set_report(samples, output_dir="./data/golden_set"):
    """Create a comprehensive report of the golden set"""
    report_path = os.path.join(output_dir, "golden_set_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GOLDEN SET REPORT - HW4 EEG Age Classification\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Creation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(samples)}\n\n")

        # Age distribution table
        f.write("AGE DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        ages = [s['age'] for s in samples]
        age_counts = {age: ages.count(age) for age in sorted(set(ages))}

        for age, count in age_counts.items():
            f.write(f"Age {age:2d}: {count:2d} samples ({count/len(samples)*100:5.1f}%)\n")

        f.write("\n" + "-" * 30 + "\n\n")

        # Sample quality
        quality_scores = [s.get('quality_score', 0) for s in samples]
        f.write("QUALITY STATISTICS:\n")
        f.write(f"  Average quality: {np.mean(quality_scores):.3f}\n")
        f.write(f"  Minimum quality: {min(quality_scores):.3f}\n")
        f.write(f"  Maximum quality: {max(quality_scores):.3f}\n\n")

        # Signal statistics
        if samples:
            signal_lengths = [len(s['signal']) for s in samples]
            signal_means = [np.mean(s['signal']) for s in samples if s['signal']]
            signal_stds = [np.std(s['signal']) for s in samples if s['signal']]

            f.write("SIGNAL STATISTICS:\n")
            f.write(f"  Signal length: {signal_lengths[0]} (consistent: {len(set(signal_lengths)) == 1})\n")
            f.write(f"  Mean of means: {np.mean(signal_means):.6f}\n")
            f.write(f"  Mean of stds:  {np.mean(signal_stds):.6f}\n\n")

        # Sample details
        f.write("SAMPLE DETAILS (first 5 samples):\n")
        f.write("-" * 60 + "\n")
        for i, sample in enumerate(samples[:5]):
            f.write(f"\nSample {i+1}:\n")
            f.write(f"  ID: {sample['sample_id']}\n")
            f.write(f"  Age: {sample['age']}\n")
            f.write(f"  Quality: {sample.get('quality_score', 0):.2f}\n")
            f.write(f"  Synthetic: {sample.get('is_synthetic', False)}\n")
            if sample.get('file_name'):
                f.write(f"  File: {sample['file_name']}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")

    print(f"✓ Created report at {report_path}")
    return report_path

def main():
    """Main function to create golden set from simulated data"""
    print("=" * 60)
    print("HW4 GOLDEN SET CREATION")
    print("Using HW3 Simulated EEG Data")
    print("=" * 60)

    # Step 1: Load simulated data
    all_samples = load_simulated_data()

    if not all_samples:
        print("ERROR: No samples loaded!")
        return None

    print(f"\nLoaded {len(all_samples)} total samples")

    # Step 2: Select golden set (target 40 samples)
    golden_samples = select_golden_samples(all_samples, target_size=40)

    if len(golden_samples) < 20:
        print(f"WARNING: Only {len(golden_samples)} samples selected, minimum 20 required")
        # Add more samples if needed
        needed = 20 - len(golden_samples)
        available = [s for s in all_samples if s not in golden_samples]
        if available:
            golden_samples.extend(available[:needed])
            print(f"Added {min(needed, len(available))} more samples")

    # Step 3: Analyze the golden set
    age_counts = analyze_golden_set(golden_samples)

    # Step 4: Save in multiple formats
    print("\nSaving golden set...")
    jsonl_path, csv_path, metadata_path = save_golden_set(golden_samples)

    # Step 5: Create report
    report_path = create_golden_set_report(golden_samples)

    # Step 6: Print summary
    print("\n" + "=" * 60)
    print("GOLDEN SET CREATION COMPLETE")
    print("=" * 60)
    print(f"✓ Created golden set with {len(golden_samples)} samples")
    print(f"✓ Age distribution: {len(age_counts)} unique ages")
    print(f"✓ Files saved:")
    print(f"    - {jsonl_path}")
    print(f"    - {csv_path}")
    print(f"    - {metadata_path}")
    print(f"    - {report_path}")
    print("\nNext steps:")
    print("  1. Review the CSV file for sample details")
    print("  2. Use golden set with eval_gate.py")
    print("  3. Commit golden set to DVC for versioning")

    return golden_samples

def create_minimal_golden_set():
    """Create a minimal golden set for CI/testing"""
    print("Creating minimal golden set for CI...")

    # Create 20 samples quickly
    samples = []
    ages = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 26, 47]

    for i, age in enumerate(ages):
        # Very simple signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        samples.append({
            'sample_id': f"ci_{i}",
            'age': age,
            'signal': signal.tolist(),
            'quality_score': 0.9,
            'is_synthetic': True,
            'notes': 'Minimal set for CI testing'
        })

    # Save
    output_dir = "golden_set"
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "golden_samples_ci.jsonl")
    with open(jsonl_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Created minimal golden set with {len(samples)} samples at {jsonl_path}")
    return samples

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Create golden set for HW4")
    parser.add_argument("--ci", action="store_true", help="Create minimal set for CI")
    parser.add_argument("--size", type=int, default=40, help="Target size of golden set")

    args = parser.parse_args()

    if args.ci:
        create_minimal_golden_set()
    else:
        golden_samples = main()