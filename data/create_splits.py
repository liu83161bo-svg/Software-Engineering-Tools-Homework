"""
Subject-based Data Splitting for EEG Dataset
Ensures no data leakage by splitting at subject level (not trial level)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


class SubjectSplitter:
    """Creates subject-based splits to prevent data leakage"""

    def __init__(self, csv_path="raw/sample_eeg_data.csv"):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path)
        self.split_dir = Path("splits")
        self.split_dir.mkdir(exist_ok=True)

    def validate_subject_integrity(self):
        """Check that all trials from same subject have same age"""
        subject_ages = self.df.groupby('subject_hash')['age'].nunique()
        invalid_subjects = subject_ages[subject_ages > 1]

        if len(invalid_subjects) > 0:
            raise ValueError(f"Subjects with multiple ages: {invalid_subjects.index.tolist()}")

        print(f"✓ Subject integrity check passed: {len(self.df['subject_hash'].unique())} unique subjects")

    def create_stratified_splits(self, ratios=(0.7, 0.15, 0.15), seed=42):
        """Create stratified splits preserving age distribution"""
        np.random.seed(seed)

        # Get unique subjects with their age
        subject_info = self.df[['subject_hash', 'age']].drop_duplicates()
        subjects_by_age = subject_info.groupby('age')['subject_hash'].apply(list).to_dict()

        # Initialize split lists
        train_subjects = []
        val_subjects = []
        test_subjects = []

        # Split subjects by age to maintain distribution
        for age, subjects in subjects_by_age.items():
            np.random.shuffle(subjects)
            n = len(subjects)

            n_train = int(n * ratios[0])
            n_val = int(n * ratios[1])

            train_subjects.extend(subjects[:n_train])
            val_subjects.extend(subjects[n_train:n_train + n_val])
            test_subjects.extend(subjects[n_train + n_val:])

        # Shuffle final lists
        np.random.shuffle(train_subjects)
        np.random.shuffle(val_subjects)
        np.random.shuffle(test_subjects)

        return train_subjects, val_subjects, test_subjects

    def get_trials_by_subjects(self, subject_list):
        """Get all trial_ids for given subjects"""
        mask = self.df['subject_hash'].isin(subject_list)
        return self.df.loc[mask, 'trial_id'].tolist()

    def save_splits(self, train_subjects, val_subjects, test_subjects):
        """Save split files in multiple formats"""

        # Get trial IDs for each split
        train_trials = self.get_trials_by_subjects(train_subjects)
        val_trials = self.get_trials_by_subjects(val_subjects)
        test_trials = self.get_trials_by_subjects(test_subjects)

        # Save as .txt files (simple lists)
        splits = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }

        for split_name, subjects in splits.items():
            with open(self.split_dir / f"{split_name}_subjects.txt", 'w') as f:
                for subject in subjects:
                    f.write(f"{subject}\n")

        # Save as JSON for metadata
        split_metadata = {
            'split_strategy': 'subject_stratified',
            'ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
            'statistics': {
                'total_subjects': len(self.df['subject_hash'].unique()),
                'total_trials': len(self.df),
                'train_subjects': len(train_subjects),
                'val_subjects': len(val_subjects),
                'test_subjects': len(test_subjects),
                'train_trials': len(train_trials),
                'val_trials': len(val_trials),
                'test_trials': len(test_trials)
            },
            'age_distribution': {
                'train': self.df[self.df['subject_hash'].isin(train_subjects)]['age'].value_counts().to_dict(),
                'val': self.df[self.df['subject_hash'].isin(val_subjects)]['age'].value_counts().to_dict(),
                'test': self.df[self.df['subject_hash'].isin(test_subjects)]['age'].value_counts().to_dict()
            }
        }

        with open(self.split_dir / 'split_metadata.json', 'w') as f:
            json.dump(split_metadata, f, indent=2)

        # Save trial-level splits
        trial_splits = {
            'train_trials.txt': train_trials,
            'val_trials.txt': val_trials,
            'test_trials.txt': test_trials
        }

        for filename, trials in trial_splits.items():
            with open(self.split_dir / filename, 'w') as f:
                for trial_id in trials:
                    f.write(f"{trial_id}\n")

        return split_metadata

    def verify_no_leakage(self, train_subjects, val_subjects, test_subjects):
        """Verify no subject appears in multiple splits"""
        train_set = set(train_subjects)
        val_set = set(val_subjects)
        test_set = set(test_subjects)

        # Check intersections
        assert len(train_set & val_set) == 0, "Leakage between train and val!"
        assert len(train_set & test_set) == 0, "Leakage between train and test!"
        assert len(val_set & test_set) == 0, "Leakage between val and test!"

        # Check coverage
        all_subjects = train_set | val_set | test_set
        unique_subjects = set(self.df['subject_hash'].unique())
        assert all_subjects == unique_subjects, "Not all subjects are split!"

        print("✓ No data leakage detected")
        print(f"  Train: {len(train_set)} subjects")
        print(f"  Val: {len(val_set)} subjects")
        print(f"  Test: {len(test_set)} subjects")

    def run(self):
        """Execute complete splitting pipeline"""
        print("=" * 60)
        print("Creating Subject-Based Data Splits")
        print("=" * 60)

        # Validate data
        self.validate_subject_integrity()

        # Create splits
        train_subjects, val_subjects, test_subjects = self.create_stratified_splits()

        # Verify no leakage
        self.verify_no_leakage(train_subjects, val_subjects, test_subjects)

        # Save splits
        metadata = self.save_splits(train_subjects, val_subjects, test_subjects)

        print("\nSplit Statistics:")
        print("-" * 40)
        stats = metadata['statistics']
        print(f"Total Subjects: {stats['total_subjects']}")
        print(f"Total Trials: {stats['total_trials']}")
        print(f"Train: {stats['train_subjects']} subjects, {stats['train_trials']} trials")
        print(f"Validation: {stats['val_subjects']} subjects, {stats['val_trials']} trials")
        print(f"Test: {stats['test_subjects']} subjects, {stats['test_trials']} trials")

        print(f"\n✓ Splits saved to: {self.split_dir}")

        return metadata


def main():
    """Main function to create splits"""
    splitter = SubjectSplitter()
    metadata = splitter.run()
    return metadata


if __name__ == "__main__":
    metadata = main()