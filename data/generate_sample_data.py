"""
EEG Sample Data Generator for AI System Course
Generates synthetic EEG data with fixed age distribution
Each age: 20 trials, total 320 trials (16 ages × 20)
"""

import numpy as np
import pandas as pd
import json
import hashlib
from pathlib import Path
import os


class EEGDataGenerator:
    """Generates synthetic EEG data with realistic characteristics"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.age_distribution = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 26, 47]
        self.trials_per_age = 20
        self.signal_length = 1000
        self.total_subjects = 160  # 10 subjects per age (each has 2 trials)

    def _generate_eeg_signal(self, age):
        """Generate synthetic EEG signal based on age"""
        t = np.linspace(0, 1, self.signal_length)

        # Age-dependent frequency characteristics
        if age <= 1:  # Infants
            dominant_freq = np.random.uniform(3, 8)  # Delta/Theta
            amplitude = np.random.uniform(0.5, 1.0)
        elif age <= 12:  # Children
            dominant_freq = np.random.uniform(8, 12)  # Alpha
            amplitude = np.random.uniform(1.0, 1.5)
        else:  # Adults
            dominant_freq = np.random.uniform(12, 30)  # Beta
            amplitude = np.random.uniform(0.8, 1.2)

        # Generate signal components
        alpha = amplitude * np.sin(2 * np.pi * dominant_freq * t)
        beta = 0.3 * amplitude * np.sin(2 * np.pi * (dominant_freq * 1.5) * t)
        gamma = 0.1 * amplitude * np.sin(2 * np.pi * (dominant_freq * 2) * t)
        noise = 0.15 * np.random.randn(self.signal_length)

        signal = alpha + beta + gamma + noise
        return signal

    def _create_subject_hash(self, subject_id):
        """Create deterministic hash for subject ID"""
        return hashlib.md5(subject_id.encode()).hexdigest()[:10]

    def generate_samples(self):
        """Generate dataset with fixed age distribution"""
        records = []
        full_signals = []
        trial_id = 1000

        # Create 10 subjects per age, each with 2 trials = 20 trials per age
        for age_idx, age in enumerate(self.age_distribution):
            for subject_idx in range(10):  # 10 subjects per age
                subject_id = f"sub_{age:02d}_{subject_idx:03d}"
                subject_hash = self._create_subject_hash(subject_id)

                # Generate 2 trials per subject
                for trial_num in range(2):
                    # Generate EEG signal
                    signal = self._generate_eeg_signal(age)

                    # Create record for CSV
                    record = {
                        'trial_id': trial_id,
                        'subject_id': subject_id,
                        'subject_hash': subject_hash,
                        'file_name': f"{subject_id}_ses{np.random.randint(1, 4)}.mat",
                        'trial_index': trial_num,
                        'age': age,
                        'session_id': np.random.randint(1, 4),
                        'recording_date': f"2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
                        'channel_count': 64,
                        'sampling_rate': 256,
                        'recording_duration': 1.0  # seconds
                    }

                    # Add first 10 signal points to CSV
                    for j in range(min(10, len(signal))):
                        record[f'signal_point_{j:03d}'] = round(signal[j], 4)

                    records.append(record)

                    # Create full signal for JSONL
                    full_signals.append({
                        'trial_id': trial_id,
                        'subject_hash': subject_hash,
                        'age': age,
                        'signal': [round(float(x), 6) for x in signal],
                        'metadata': {
                            'subject_id': subject_id,
                            'trial_index': trial_num,
                            'sampling_rate': 256,
                            'age_group': 'infant' if age <= 1 else 'child' if age <= 12 else 'adult'
                        }
                    })

                    trial_id += 1

        # Create DataFrames
        df = pd.DataFrame(records)
        print(f"Generated {len(df)} records ({len(self.age_distribution)} ages × {self.trials_per_age} trials)")
        print(f"Age distribution: {self.age_distribution}")
        print(f"Total unique subjects: {df['subject_hash'].nunique()}")

        return df, full_signals

    def save_data(self, df, full_signals):
        """Save data to CSV and JSONL formats"""
        # Ensure directories exist
        Path("raw").mkdir(parents=True, exist_ok=True)
        Path("processed").mkdir(parents=True, exist_ok=True)

        # Save to CSV
        csv_path = "raw/sample_eeg_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")

        # Save to JSONL
        jsonl_path = "processed/sample_eeg_data.jsonl"
        with open(jsonl_path, 'w') as f:
            for item in full_signals:
                f.write(json.dumps(item) + '\n')
        print(f"Saved JSONL to: {jsonl_path}")

        return csv_path, jsonl_path


def main():
    """Main function to generate and save sample data"""
    print("=" * 60)
    print("Generating EEG Sample Data")
    print("=" * 60)

    generator = EEGDataGenerator(seed=42)
    df, full_signals = generator.generate_samples()
    csv_path, jsonl_path = generator.save_data(df, full_signals)

    print("\nData Summary:")
    print("-" * 40)
    print(f"Total trials: {len(df)}")
    print(f"Unique subjects: {df['subject_hash'].nunique()}")
    print(f"Age distribution: {sorted(df['age'].unique())}")
    print(f"Trials per age: {df['age'].value_counts().min()}")

    print("\nFirst 3 records:")
    print(df[['trial_id', 'subject_hash', 'age', 'trial_index']].head(3))

    return df


if __name__ == "__main__":
    df = main()