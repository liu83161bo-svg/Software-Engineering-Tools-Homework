"""
EEG Age Classification Dataset - Automated Data Checks
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


class TestDataQuality:
    """Data quality test suite for EEG dataset"""

    DATA_PATH = Path(__file__).parent.parent / "data" / "sample_eeg_data.csv"
    JSONL_PATH = Path(__file__).parent.parent / "data" / "sample_eeg_data.jsonl"
    SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"

    REQUIRED_COLS = ['trial_id', 'subject_id', 'subject_hash', 'age']
    AGE_RANGE = (0, 100)

    def setup_method(self):
        """Load test data"""
        self.df = pd.read_csv(self.DATA_PATH)

    # ==================== Syntax Checks ====================

    def test_csv_format_valid(self):
        """SYN-01: Check CSV format"""
        assert self.DATA_PATH.exists()
        assert self.DATA_PATH.suffix == '.csv'

    def test_no_missing_values(self):
        """SYN-02: Check for missing values"""
        for col in self.REQUIRED_COLS:
            assert col in self.df.columns
            assert self.df[col].isnull().sum() == 0

    def test_no_corrupted_data(self):
        """SYN-03: Check data integrity"""
        assert self.df['trial_id'].nunique() == len(self.df)
        assert (self.df['subject_hash'].str.startswith('hash_')).all()

    # ==================== Structural Checks ====================

    def test_no_duplicate_trials(self):
        """STR-01: Check for duplicate trials"""
        assert not self.df['trial_id'].duplicated().any()

    def test_subject_integrity(self):
        """STR-02: Check subject data consistency"""
        if 'subject_id' in self.df.columns and 'subject_hash' in self.df.columns:
            assert self.df.groupby('subject_id')['subject_hash'].nunique().max() == 1
            assert self.df.groupby('subject_hash')['subject_id'].nunique().max() == 1

    def test_age_data_matching(self):
        """STR-03: Check age data quality"""
        assert self.df['age'].dtype in [np.int64, np.int32, int]
        assert self.df['age'].between(*self.AGE_RANGE).all()
        assert (self.df['age'] >= 0).all()

    def test_signal_data_structure(self):
        """STR-04: Check signal data"""
        signal_cols = [c for c in self.df.columns if c.startswith('signal_')]
        assert len(signal_cols) >= 10
        for col in signal_cols[:5]:
            assert pd.api.types.is_numeric_dtype(self.df[col])

    # ==================== Statistical Checks ====================

    def test_age_distribution(self):
        """STA-01: Check age distribution"""
        age_counts = self.df['age'].value_counts()
        assert age_counts.max() / len(self.df) <= 0.5
        assert self.df['age'].nunique() >= 3

    def test_signal_statistics(self):
        """STA-02: Check signal statistics"""
        signal_cols = [c for c in self.df.columns if c.startswith('signal_')]
        if signal_cols:
            signal_means = self.df[signal_cols].mean(axis=1)
            mean_of_means = signal_means.mean()
            assert -10 < mean_of_means < 10

            signal_stds = self.df[signal_cols].std(axis=1).mean()
            assert 0.1 < signal_stds < 100

    def test_jsonl_completeness(self):
        """STA-03: Check JSONL format"""
        if self.JSONL_PATH.exists():
            with open(self.JSONL_PATH, 'r') as f:
                records = [json.loads(line) for line in f]

            assert len(records) > 0
            signal_lengths = [len(r.get('signal', [])) for r in records]
            assert len(set(signal_lengths)) == 1

    # ==================== Split Validation ====================

    def test_split_files_exist(self):
        """SPLIT-01: Check split files exist"""
        assert self.SPLITS_DIR.exists()
        required_files = ['train_ids.txt', 'val_ids.txt', 'test_ids.txt']
        for fname in required_files:
            assert (self.SPLITS_DIR / fname).exists()

    def test_split_integrity(self):
        """SPLIT-02: Validate split assignments"""
        splits = {}
        for split_file in self.SPLITS_DIR.glob("*_ids.txt"):
            split_name = split_file.stem.replace('_ids', '')
            with open(split_file, 'r') as f:
                splits[split_name] = {int(line.strip()) for line in f if line.strip()}

        # Check no overlap between splits
        for s1 in splits:
            for s2 in splits:
                if s1 != s2:
                    assert len(splits[s1] & splits[s2]) == 0

        # Check coverage
        all_split_trials = set()
        for s in splits.values():
            all_split_trials.update(s)

        all_data_trials = set(self.df['trial_id'])
        assert all_data_trials == all_split_trials


def run_all_checks():
    """Run all data checks and print summary"""
    tester = TestDataQuality()
    tester.setup_method()

    results = []
    test_methods = [m for m in dir(tester) if m.startswith('test_')]

    print("=" * 60)
    print("Data Quality Checks")
    print("=" * 60)

    for method_name in sorted(test_methods):
        method = getattr(tester, method_name)
        try:
            method()
            results.append((method_name, "PASS"))
            print(f"✓ {method_name}")
        except AssertionError as e:
            results.append((method_name, f"FAIL: {str(e)}"))
            print(f"✗ {method_name}: {str(e)}")

    print("\n" + "=" * 60)
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    print(f"Result: {passed}/{total} passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_checks()
    exit(0 if success else 1)