import numpy as np
import pandas as pd
import os
import json
from pathlib import Path


def generate_sample_data():
    """
    Generate simulated EEG dataset with fixed age distribution.
    Each age has exactly 20 samples.
    """
    np.random.seed(42)  # Fixed seed for reproducibility

    # Fixed age distribution (as per your requirement)
    ages = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 26, 47]
    samples_per_age = 20
    total_samples = len(ages) * samples_per_age

    print(f"Generating {total_samples} samples ({len(ages)} ages x {samples_per_age} samples each)")

    # Create data records
    data = []
    sample_id = 0

    for age in ages:
        # Generate 20 samples for each age
        for age_sample_idx in range(samples_per_age):
            # Generate random EEG signal (1000 time points)
            t = np.linspace(0, 1, 1000)
            alpha = np.sin(2 * np.pi * 10 * t)  # 10Hz alpha
            beta = 0.5 * np.sin(2 * np.pi * 20 * t)  # 20Hz beta
            noise = 0.2 * np.random.randn(1000)  # Gaussian noise
            signal = alpha + beta + noise

            # Create unique subject ID for each sample
            # In real scenario, one subject could have multiple samples
            # But for simplicity, each sample is from a different subject
            subject_id = f"sub_{age:03d}_{age_sample_idx:03d}"
            file_name = f"{subject_id}_eeg.mat"

            record = {
                'trial_id': sample_id + 1000,
                'file_name': file_name,
                'trial_index': 0,  # All samples have index 0 (single trial per subject)
                'age': age,
                'subject_hash': f"hash_{hash(subject_id) % 100000:05d}",
                'recording_date': f"2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            }

            # Add first 10 signal points to CSV
            for j in range(min(10, len(signal))):
                record[f'signal_{j}'] = signal[j]

            data.append(record)
            sample_id += 1

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    csv_path = 'sample_eeg_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV data to {csv_path}")

    # Create full signals version (JSONL format)
    jsonl_data = []
    sample_id = 0

    for age in ages:
        for age_sample_idx in range(samples_per_age):
            t = np.linspace(0, 1, 1000)
            alpha = np.sin(2 * np.pi * 10 * t)
            beta = 0.5 * np.sin(2 * np.pi * 20 * t)
            noise = 0.2 * np.random.randn(1000)
            signal = (alpha + beta + noise).tolist()

            jsonl_data.append({
                'trial_id': sample_id + 1000,
                'signal': signal,
                'age': age
            })
            sample_id += 1

    # Save to JSONL
    jsonl_path = 'sample_eeg_data.jsonl'
    with open(jsonl_path, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved JSONL data to {jsonl_path}")

    # Verify age distribution
    age_counts = df['age'].value_counts().sort_index()
    print("\nAge distribution verification:")
    for age, count in age_counts.items():
        print(f"  Age {age}: {count} samples")

    return df


if __name__ == "__main__":
    df = generate_sample_data()
    print(f"\nTotal samples generated: {len(df)}")
    print("\nFirst 3 records:")
    print(df[['trial_id', 'age', 'subject_hash']].head(3))