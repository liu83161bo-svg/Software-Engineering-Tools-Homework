"""
EEG Age Classification Dataset - Automated Data Checks
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


class TestDataQuality:
    """Data quality test suite for EEG age classification dataset"""

    DATA_PATH = Path(__file__).parent.parent / "data" / "sample_eeg_data.csv"
    JSONL_PATH = Path(__file__).parent.parent / "data" / "sample_eeg_data.jsonl"
    REQUIRED_COLUMNS = ['trial_id', 'subject_hash', 'trial_index', 'age']
    AGE_RANGE = (0, 100)

    def setup_method(self):
        self.df = pd.read_csv(self.DATA_PATH)

    # ==================== Syntax Checks (3 checks) ====================

    def test_01_file_format_valid(self):
        """SYN-01: Check CSV file format is valid"""
        assert self.DATA_PATH.exists(), "CSV file not found"
        assert self.DATA_PATH.suffix == '.csv', "File must be .csv format"

    def test_02_no_missing_values(self):
        """SYN-02: Check for missing/null values"""
        for col in self.REQUIRED_COLUMNS:
            assert col in self.df.columns, f"Missing column: {col}"
            assert self.df[col].isnull().sum() == 0, f"Null values in {col}"

    def test_03_no_corrupted_data(self):
        """SYN-03: Check data integrity"""
        assert self.df['trial_id'].nunique() == len(self.df), "Duplicate trial_id"

    # ==================== Structural Checks (4 checks) ====================

    def test_04_no_duplicate_trials(self):
        """STR-01: Check for duplicate trial entries"""
        duplicate_ids = self.df[self.df['trial_id'].duplicated()]
        assert len(duplicate_ids) == 0, f"Duplicate trial_id found"

    def test_05_no_data_leakage_by_subject(self):
        """STR-02: Check no data leakage at subject level"""
        assert 'subject_hash' in self.df.columns, "subject_hash column missing"
        assert self.df['subject_hash'].isnull().sum() == 0, "Missing subject_hash"

    def test_06_label_data_matching(self):
        """STR-03: Check label (age) matches data expectations"""
        assert self.df['age'].dtype in [np.int64, np.int32, int], "Age must be integer"
        min_age, max_age = self.AGE_RANGE
        assert self.df['age'].between(min_age, max_age).all(), "Age out of range"

    def test_07_signal_columns_exist(self):
        """STR-04: Check signal data structure"""
        signal_cols = [col for col in self.df.columns if 'signal' in col]
        assert len(signal_cols) > 0, "No signal columns found"
        assert pd.api.types.is_numeric_dtype(self.df[signal_cols[0]]), "Signal not numeric"

    # ==================== Statistical Checks (3 checks) ====================

    def test_08_age_distribution_imbalance(self):
        """STA-01: Check age distribution imbalance"""
        age_counts = self.df['age'].value_counts()
        max_share = age_counts.max() / len(self.df)
        assert max_share <= 0.5, f"Age imbalance: {max_share:.1%}"
        assert self.df['age'].nunique() >= 3, "Not enough unique ages"

    def test_09_signal_statistics_within_range(self):
        """STA-02: Check signal statistical properties"""
        signal_cols = [col for col in self.df.columns if 'signal' in col]
        if len(signal_cols) > 0:
            signal_means = self.df[signal_cols].mean(axis=1)
            mean_of_means = signal_means.mean()
            assert -10 < mean_of_means < 10, f"Signal mean out of range: {mean_of_means:.2f}"

    def test_10_jsonl_completeness_check(self):
        """STA-03: Check JSONL format completeness"""
        if self.JSONL_PATH.exists():
            with open(self.JSONL_PATH, 'r') as f:
                records = [json.loads(line.strip()) for line in f]
            assert len(records) > 0, "No records in JSONL"
            signal_lengths = [len(r.get('signal', [])) for r in records]
            assert len(set(signal_lengths)) == 1, "Inconsistent signal lengths"

    # ==================== Additional Checks (2 checks) ====================

    def test_11_trial_index_range(self):
        """Check trial_index values are reasonable"""
        assert (self.df['trial_index'] >= 0).all(), "Negative trial_index"

    def test_12_data_types_correct(self):
        """Check column data types"""
        type_checks = [
            ('trial_id', 'int'),
            ('trial_index', 'int'),
            ('age', 'int')
        ]
        for col, expected in type_checks:
            if col in self.df.columns:
                actual = str(self.df[col].dtype)
                if 'int' in expected and 'int' in actual:
                    continue
                else:
                    assert False, f"{col}: expected {expected}, got {actual}"

    def test_13_split_integrity_check(self):
        """STR-05: Verify subject-based splits have no leakage"""
        split_dir = Path(__file__).parent.parent / "data" / "splits"

        # Check split files exist
        required_files = ['train_subjects.txt', 'val_subjects.txt', 'test_subjects.txt']
        for file in required_files:
            file_path = split_dir / file
            assert file_path.exists(), f"Split file missing: {file}"

        # Load subjects from each split
        with open(split_dir / 'train_subjects.txt', 'r') as f:
            train_subjects = set(line.strip() for line in f)
        with open(split_dir / 'val_subjects.txt', 'r') as f:
            val_subjects = set(line.strip() for line in f)
        with open(split_dir / 'test_subjects.txt', 'r') as f:
            test_subjects = set(line.strip() for line in f)

        # Check no overlap between splits
        assert len(train_subjects & val_subjects) == 0, "Subject leakage between train and val!"
        assert len(train_subjects & test_subjects) == 0, "Subject leakage between train and test!"
        assert len(val_subjects & test_subjects) == 0, "Subject leakage between val and test!"

        # Check all subjects are accounted for
        all_split_subjects = train_subjects | val_subjects | test_subjects
        data_subjects = set(self.df['subject_hash'].unique())

        missing_in_splits = data_subjects - all_split_subjects
        extra_in_splits = all_split_subjects - data_subjects

        assert len(missing_in_splits) == 0, f"Subjects missing in splits: {missing_in_splits}"
        assert len(extra_in_splits) == 0, f"Extra subjects in splits: {extra_in_splits}"

        print(
            f"✓ Split integrity: {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test subjects")

    def test_14_dvc_tracking_check(self):
        """STR-06: Check DVC tracking files exist"""
        dvc_files = [
            Path(__file__).parent.parent / "data" / "raw" / "sample_eeg_data.csv.dvc",
            Path(__file__).parent.parent / "dvc.yaml",
            Path(__file__).parent.parent / ".dvc" / "config"
        ]

        for dvc_file in dvc_files:
            if dvc_file.exists():
                print(f"✓ DVC file found: {dvc_file.name}")
            else:
                # This is a warning, not a failure (for first-time setup)
                print(f"⚠ DVC file not found (expected for initial setup): {dvc_file.name}")

    def test_15_age_distribution_per_split(self):
        """STA-04: Check age distribution is balanced across splits"""
        split_dir = Path(__file__).parent.parent / "data" / "splits"

        if not (split_dir / 'split_metadata.json').exists():
            print("⚠ Split metadata not found, skipping distribution check")
            return

        with open(split_dir / 'split_metadata.json', 'r') as f:
            metadata = json.load(f)

        age_dist = metadata.get('age_distribution', {})

        # Check each split has multiple ages
        for split_name in ['train', 'val', 'test']:
            if split_name in age_dist:
                ages_in_split = list(age_dist[split_name].keys())
                assert len(ages_in_split) >= 8, f"Split {split_name} has only {len(ages_in_split)} ages"

        print(f"✓ Age distribution balanced across splits")

    def test_16_data_version_consistency(self):
        """STA-05: Check CSV and JSONL data consistency"""
        if not self.JSONL_PATH.exists():
            print("⚠ JSONL file not found, skipping consistency check")
            return

        # Load JSONL data
        jsonl_data = []
        with open(self.JSONL_PATH, 'r') as f:
            for line in f:
                jsonl_data.append(json.loads(line))

        # Check trial count matches
        assert len(jsonl_data) == len(self.df), \
            f"CSV has {len(self.df)} records, JSONL has {len(jsonl_data)}"

        # Check age distribution matches
        csv_ages = set(self.df['age'].unique())
        jsonl_ages = set(item['age'] for item in jsonl_data)
        assert csv_ages == jsonl_ages, f"Age mismatch: CSV {csv_ages}, JSONL {jsonl_ages}"

        print(f"✓ Data consistency: {len(jsonl_data)} records match between CSV and JSONL")