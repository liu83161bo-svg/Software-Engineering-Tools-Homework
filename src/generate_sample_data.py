# scripts/generate_sample_data.py
"""
EEG Age Classification - Sample Data Generator
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict
import itertools


class UniformEEGDataGenerator:

    FIXED_AGES = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 26, 47]
    SAMPLES_PER_AGE = 20
    SIGNAL_LENGTH = 1000
    NUM_CHANNELS = 64

    def __init__(self, seed: int = 42):

        np.random.seed(seed)

    def generate_eeg_signal(self, age: int, trial_idx: int) -> np.ndarray:

        t = np.linspace(0, 1, self.SIGNAL_LENGTH)


        if age < 13:
            delta = 0.5 * np.sin(2 * np.pi * 2 * t)  # 2Hz delta波（睡眠）
            theta = 0.6 * np.sin(2 * np.pi * 6 * t)  # 6Hz theta波（困倦）
            alpha = 0.4 * np.sin(2 * np.pi * 10 * t)  # 10Hz alpha波（放松）
            beta = 0.3 * np.sin(2 * np.pi * 20 * t)  # 20Hz beta波（思考）

        elif age < 20:
            delta = 0.3 * np.sin(2 * np.pi * 2 * t)
            theta = 0.4 * np.sin(2 * np.pi * 6 * t)
            alpha = 0.7 * np.sin(2 * np.pi * 10 * t)  # alpha波更强
            beta = 0.5 * np.sin(2 * np.pi * 20 * t)

        else:
            delta = 0.2 * np.sin(2 * np.pi * 2 * t)
            theta = 0.3 * np.sin(2 * np.pi * 6 * t)
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t)
            beta = 0.8 * np.sin(2 * np.pi * 20 * t)  # beta波最强


        channel_noise = 0.1 * np.random.randn(self.SIGNAL_LENGTH)


        signal = delta + theta + alpha + beta + channel_noise

        signal += 0.05 * np.random.randn(self.SIGNAL_LENGTH)

        return signal

    def generate_subjects_for_age(self, age: int, age_idx: int) -> List[Dict]:

        records = []


        n_subjects_per_age = 4
        trials_per_subject = 5

        for subject_idx in range(n_subjects_per_age):
            subject_id = f"sub_age{age_idx:02d}_{subject_idx:02d}"
            subject_hash = f"hash_{hash(subject_id + str(age)) % 1000000:06d}"

            for trial_idx in range(trials_per_subject):

                signal = self.generate_eeg_signal(age, trial_idx)

                global_trial_idx = age_idx * self.SAMPLES_PER_AGE + subject_idx * trials_per_subject + trial_idx


                record = {
                    'trial_id': 1000 + global_trial_idx,
                    'subject_id': subject_id,
                    'subject_hash': subject_hash,
                    'file_name': f"{subject_id}_ses{trial_idx + 1:02d}.mat",
                    'trial_index': trial_idx,
                    'age': age,
                    'recording_date': f"2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
                    'session': trial_idx + 1,
                    'n_channels': self.NUM_CHANNELS,
                    'sampling_rate': 250,
                    'age_group': self.get_age_group(age),
                }

                for j in range(min(10, len(signal))):
                    record[f'signal_{j:02d}'] = signal[j]

                records.append(record)

        return records

    def get_age_group(self, age: int) -> str:
        if age < 2:
            return "infant"
        elif age < 13:
            return "child"
        elif age < 20:
            return "teen"
        elif age < 30:
            return "young_adult"
        else:
            return "adult"

    def generate_data(self) -> pd.DataFrame:
        """"""
        all_records = []

        print(f"start...")
        print(f"age groups list: {self.FIXED_AGES}")
        print(f"SAMPLES_PER_AGE: {self.SAMPLES_PER_AGE}")
        print(f"SAMPLES_NUM: {len(self.FIXED_AGES) * self.SAMPLES_PER_AGE}")

        for age_idx, age in enumerate(self.FIXED_AGES):
            age_records = self.generate_subjects_for_age(age, age_idx)
            all_records.extend(age_records)

            print(f"  age {age}: generate {len(age_records)} 条记录")

        df = pd.DataFrame(all_records)

        print("\nage dist:")
        age_counts = df['age'].value_counts().sort_index()

        total_samples = 0
        for age, count in age_counts.items():
            print(f"  age {age}: {count} samples")
            total_samples += count

        print(f"\ngenerate:")
        print(f"  samples_count: {len(df)}")
        print(f"  subjects: {df['subject_id'].nunique()}")
        print(f"  age: {df['age'].nunique()}")
        print(f"  age_group: {df['age_group'].unique()}")

        for age in self.FIXED_AGES:
            count = (df['age'] == age).sum()
            if count != self.SAMPLES_PER_AGE:
                print(f"warn: age {age} has {count} samples，should has {self.SAMPLES_PER_AGE}")

        return df


def save_formats(df: pd.DataFrame, output_dir: Path = Path('data')):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sample_eeg_data.csv"
    df.to_csv(csv_path, index=False)

    jsonl_path = output_dir / "sample_eeg_data.jsonl"
    jsonl_records = []

    for _, row in df.iterrows():
        t = np.linspace(0, 1, 100)

        if row['age'] < 13:
            signal = np.sin(2 * np.pi * 6 * t)
        elif row['age'] < 20:
            signal = np.sin(2 * np.pi * 10 * t)
        else:
            signal = np.sin(2 * np.pi * 20 * t)

        signal += 0.1 * np.random.randn(100)

        jsonl_record = {
            'trial_id': int(row['trial_id']),
            'subject_id': row['subject_id'],
            'subject_hash': row['subject_hash'],
            'age': int(row['age']),
            'age_group': row['age_group'],
            'signal': signal.tolist(),
            'metadata': {
                'session': int(row['session']),
                'n_channels': int(row['n_channels']),
                'sampling_rate': int(row['sampling_rate']),
                'recording_date': row['recording_date']
            }
        }
        jsonl_records.append(jsonl_record)

    with open(jsonl_path, 'w') as f:
        for record in jsonl_records:
            f.write(json.dumps(record) + '\n')

    stats = {
        'total_samples': len(df),
        'total_subjects': df['subject_id'].nunique(),
        'age_distribution': df['age'].value_counts().to_dict(),
        'age_group_distribution': df['age_group'].value_counts().to_dict(),
        'generated_at': pd.Timestamp.now().isoformat()
    }

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return csv_path, jsonl_path, stats_path


def main():
    parser = argparse.ArgumentParser(description='generate EEG Age Classification')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='output dir')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')

    args = parser.parse_args()

    print(f"config:")
    print(f"  seed: {args.seed}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  FIXED_AGES: {len(UniformEEGDataGenerator.FIXED_AGES)}")
    print(f"  SAMPLES_PER_AGE: {UniformEEGDataGenerator.SAMPLES_PER_AGE}")
    print(f"  SAMPLES_COUNT: {len(UniformEEGDataGenerator.FIXED_AGES) * UniformEEGDataGenerator.SAMPLES_PER_AGE}")

    generator = UniformEEGDataGenerator(seed=args.seed)
    df = generator.generate_data()

    output_dir = Path(args.output_dir)
    csv_path, jsonl_path, stats_path = save_formats(df, output_dir)

    print(f"\nSAVED:")
    print(f"  CSV: {csv_path} ({df.shape[0]}ROWS, {df.shape[1]}COLUMNS)")
    print(f"  JSONL: {jsonl_path}")
    print(f"  stats_path: {stats_path}")

    # 显示数据预览
    print(f"\nVIEW（5）:")
    print(df[['trial_id', 'subject_id', 'age', 'age_group', 'session']].head())

    print(f"\ngroup by AGE:")
    age_groups = df.groupby(['age', 'age_group'])['subject_id'].nunique().reset_index()
    age_groups.columns = ['age', 'age_group', 'subject_count']
    print(age_groups.to_string(index=False))

    return df


if __name__ == "__main__":
    df = main()