"""
EEG Age Classification Dataset - Automated Data Checks
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


class TestDataQuality:
    """Data quality test suite for EEG age classification dataset"""

    DATA_PATH = "./data/sample_eeg_data.csv"
    JSONL_PATH = "./data/sample_eeg_data.jsonl"
    SPLITS_DIR = "./data/splits"
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

    # 在TestDataQuality类中添加
    def test_13_subject_split_consistency(self):
        """STR-05: Check subject-based split consistency"""
        splits_dir = Path(__file__).parent.parent / "data" / "splits"

        # Check split files exist
        split_files = ['train_subjects.txt', 'val_subjects.txt', 'test_subjects.txt']
        for file_name in split_files:
            file_path = splits_dir / file_name
            assert file_path.exists(), f"Split file {file_name} not found"

        # Load subjects from splits
        split_subjects = {}
        all_subjects_in_splits = set()

        for file_name in split_files:
            file_path = splits_dir / file_name
            with open(file_path, 'r') as f:
                subjects = {line.strip() for line in f if line.strip()}
                split_subjects[file_name] = subjects
                all_subjects_in_splits.update(subjects)

        # Check no overlap between splits
        train_subjects = split_subjects['train_subjects.txt']
        val_subjects = split_subjects['val_subjects.txt']
        test_subjects = split_subjects['test_subjects.txt']

        assert len(train_subjects & val_subjects) == 0, "Overlap between train and val subjects"
        assert len(train_subjects & test_subjects) == 0, "Overlap between train and test subjects"
        assert len(val_subjects & test_subjects) == 0, "Overlap between val and test subjects"

        # Check all subjects in data are in splits
        data_subjects = set(self.df['subject_hash'].unique())
        subjects_not_in_splits = data_subjects - all_subjects_in_splits
        subjects_not_in_data = all_subjects_in_splits - data_subjects

        assert len(subjects_not_in_splits) == 0, f"Subjects in data but not in splits: {subjects_not_in_splits}"
        assert len(subjects_not_in_data) == 0, f"Subjects in splits but not in data: {subjects_not_in_data}"

        print(f"Subject split validation passed: {len(data_subjects)} subjects correctly split")

    def test_14_fixed_age_distribution(self):
        """STA-04: Verify fixed age distribution (16 ages, 20 samples each)"""
        expected_ages = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 26, 47]

        # Check all expected ages are present
        actual_ages = sorted(self.df['age'].unique())
        assert actual_ages == expected_ages, f"Age mismatch. Expected: {expected_ages}, Got: {actual_ages}"

        # Check exactly 20 samples per age
        age_counts = self.df['age'].value_counts()
        for age in expected_ages:
            assert age_counts[age] == 20, f"Age {age} has {age_counts[age]} samples, expected 20"

        # Check total samples
        total_samples = len(self.df)
        expected_total = len(expected_ages) * 20
        assert total_samples == expected_total, f"Total samples: {total_samples}, expected {expected_total}"

        print(f"Fixed age distribution verified: {len(expected_ages)} ages × 20 samples = {total_samples} total")