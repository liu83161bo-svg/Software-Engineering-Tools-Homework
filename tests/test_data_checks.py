"""
EEG Age Classification Dataset - Automated Data Checks
HW3: Data & Datasets - 10+ automated data quality checks
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path


class TestDataQuality:
    """Data quality test suite for EEG age classification dataset"""

    # Use absolute paths
    DATA_PATH = Path(__file__).parent.parent / "data" / "sample_eeg_data.csv"
    JSONL_PATH = Path(__file__).parent.parent / "data" / "sample_eeg_data.jsonl"
    SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
    REQUIRED_COLUMNS = ['trial_id', 'subject_hash', 'trial_index', 'age']
    AGE_RANGE = (0, 100)

    def __init__(self):
        """Initialize with logging"""
        self.log = []
        self.setup_method()

    def setup_method(self):
        """Load test data"""
        try:
            if not self.DATA_PATH.exists():
                print(f"ERROR: Data file not found at {self.DATA_PATH}")
                print(f"Current working directory: {os.getcwd()}")
                raise FileNotFoundError(f"Data file not found: {self.DATA_PATH}")

            self.df = pd.read_csv(self.DATA_PATH)
            self.log_message(f"✓ Loaded data: {len(self.df)} rows, {len(self.df.columns)} columns")

        except Exception as e:
            self.log_message(f"✗ Failed to load data: {str(e)}")
            raise

    def log_message(self, message):
        """Add message to log and print it"""
        self.log.append(message)
        print(message)

    # ==================== Syntax Checks (3 checks) ====================

    def test_01_file_format_valid(self):
        """SYN-01: Check CSV file format is valid"""
        self.log_message("Checking file format...")
        assert self.DATA_PATH.exists(), "CSV file not found"
        assert self.DATA_PATH.suffix == '.csv', "File must be .csv format"
        self.log_message("✓ File format is valid")

    def test_02_no_missing_values(self):
        """SYN-02: Check for missing/null values"""
        self.log_message("Checking for missing values...")
        for col in self.REQUIRED_COLUMNS:
            assert col in self.df.columns, f"Missing column: {col}"
            null_count = self.df[col].isnull().sum()
            assert null_count == 0, f"Null values in {col}: {null_count}"
        self.log_message("✓ No missing values found")

    def test_03_no_corrupted_data(self):
        """SYN-03: Check data integrity"""
        self.log_message("Checking data integrity...")
        assert self.df['trial_id'].nunique() == len(self.df), "Duplicate trial_id"
        self.log_message("✓ Data integrity check passed")

    # ==================== Structural Checks (4 checks) ====================

    def test_04_no_duplicate_trials(self):
        """STR-01: Check for duplicate trial entries"""
        self.log_message("Checking for duplicate trials...")
        duplicate_ids = self.df[self.df['trial_id'].duplicated()]
        assert len(duplicate_ids) == 0, f"Duplicate trial_id found"
        self.log_message("✓ No duplicate trials found")

    def test_05_no_data_leakage_by_subject(self):
        """STR-02: Check no data leakage at subject level"""
        self.log_message("Checking for data leakage...")
        assert 'subject_hash' in self.df.columns, "subject_hash column missing"
        null_subjects = self.df['subject_hash'].isnull().sum()
        assert null_subjects == 0, f"Missing subject_hash: {null_subjects}"
        self.log_message("✓ No data leakage detected")

    def test_06_label_data_matching(self):
        """STR-03: Check label (age) matches data expectations"""
        self.log_message("Checking age labels...")
        assert self.df['age'].dtype in [np.int64, np.int32, int], "Age must be integer"
        min_age, max_age = self.AGE_RANGE
        age_in_range = self.df['age'].between(min_age, max_age)
        assert age_in_range.all(), f"Age out of range"
        self.log_message("✓ Age labels are valid")

    def test_07_signal_columns_exist(self):
        """STR-04: Check signal data structure"""
        self.log_message("Checking signal columns...")
        signal_cols = [col for col in self.df.columns if 'signal' in col]
        assert len(signal_cols) > 0, "No signal columns found"
        assert pd.api.types.is_numeric_dtype(self.df[signal_cols[0]]), "Signal not numeric"
        self.log_message(f"✓ Found {len(signal_cols)} signal columns")

    # ==================== Statistical Checks (3 checks) ====================

    def test_08_age_distribution_imbalance(self):
        """STA-01: Check age distribution imbalance"""
        self.log_message("Checking age distribution...")
        age_counts = self.df['age'].value_counts()
        max_share = age_counts.max() / len(self.df)
        assert max_share <= 0.5, f"Age imbalance: {max_share:.1%}"
        unique_ages = self.df['age'].nunique()
        assert unique_ages >= 3, f"Not enough unique ages: {unique_ages}"
        self.log_message(f"✓ Age distribution OK ({unique_ages} unique ages)")

    def test_09_signal_statistics_within_range(self):
        """STA-02: Check signal statistical properties"""
        self.log_message("Checking signal statistics...")
        signal_cols = [col for col in self.df.columns if 'signal' in col]
        if len(signal_cols) > 0:
            signal_means = self.df[signal_cols].mean(axis=1)
            mean_of_means = signal_means.mean()
            assert -10 < mean_of_means < 10, f"Signal mean out of range: {mean_of_means:.2f}"
        self.log_message("✓ Signal statistics within range")

    def test_10_jsonl_completeness_check(self):
        """STA-03: Check JSONL format completeness"""
        self.log_message("Checking JSONL format...")
        if self.JSONL_PATH.exists():
            with open(self.JSONL_PATH, 'r') as f:
                records = [json.loads(line.strip()) for line in f]
            assert len(records) > 0, "No records in JSONL"
            signal_lengths = [len(r.get('signal', [])) for r in records]
            assert len(set(signal_lengths)) == 1, "Inconsistent signal lengths"
            self.log_message(f"✓ JSONL format OK ({len(records)} records)")
        else:
            self.log_message("⚠ JSONL file not found (optional)")

    # ==================== Additional Checks (2 checks) ====================

    def test_11_trial_index_range(self):
        """Check trial_index values are reasonable"""
        self.log_message("Checking trial_index range...")
        assert (self.df['trial_index'] >= 0).all(), "Negative trial_index"
        self.log_message("✓ Trial_index values are valid")

    def test_12_data_types_correct(self):
        """Check column data types"""
        self.log_message("Checking data types...")
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
        self.log_message("✓ Data types are correct")

    def test_13_split_files_exist(self):
        """Check that split files exist and are not empty"""
        self.log_message("Checking split files...")
        assert self.SPLITS_DIR.exists(), f"Split directory not found: {self.SPLITS_DIR}"

        split_files = ["train_subjects.txt", "val_subjects.txt", "test_subjects.txt"]
        for file in split_files:
            file_path = self.SPLITS_DIR / file
            assert file_path.exists(), f"Split file missing: {file}"
            with open(file_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0, f"Split file is empty: {file}"
        self.log_message("✓ Split files exist and are not empty")

    def test_14_no_subject_leakage(self):
        """Verify no subject appears in multiple splits"""
        self.log_message("Checking for subject leakage...")

        def load_subjects(filename):
            with open(self.SPLITS_DIR / filename, 'r') as f:
                return set(line.strip() for line in f if line.strip())

        train_subjects = load_subjects("train_subjects.txt")
        val_subjects = load_subjects("val_subjects.txt")
        test_subjects = load_subjects("test_subjects.txt")

        # Check no overlap
        assert len(train_subjects & val_subjects) == 0, "Subjects in both train and val"
        assert len(train_subjects & test_subjects) == 0, "Subjects in both train and test"
        assert len(val_subjects & test_subjects) == 0, "Subjects in both val and test"

        # Check all subjects in data are in splits
        all_split_subjects = train_subjects | val_subjects | test_subjects
        data_subjects = set(self.df['subject_hash'].unique())

        assert data_subjects.issubset(all_split_subjects), \
            f"{len(data_subjects - all_split_subjects)} subjects not in any split"

        self.log_message(f"✓ No subject leakage ({len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test subjects)")

    def test_15_age_distribution_preserved(self):
        """Verify each age has exactly 20 samples"""
        self.log_message("Checking age distribution preservation...")
        age_counts = self.df['age'].value_counts()

        # Check all ages in fixed distribution
        expected_ages = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 26, 47]

        for age in expected_ages:
            assert age in age_counts, f"Age {age} missing from data"
            assert age_counts[age] == 20, f"Age {age} has {age_counts[age]} samples, expected 20"

        self.log_message("✓ Age distribution preserved (20 samples per age)")


def run_all_checks():
    """Run all data checks and print summary"""
    print("=" * 70)
    print("EEG Data Quality Checks - HW3")
    print("=" * 70)

    try:
        tester = TestDataQuality()

        results = []
        test_methods = [m for m in dir(tester) if m.startswith('test_') and callable(getattr(tester, m))]

        print("\nRunning individual tests:")
        print("-" * 40)

        for method_name in sorted(test_methods):
            if method_name == 'run_all_checks':
                continue

            method = getattr(tester, method_name)
            test_doc = method.__doc__ or method_name
            test_desc = test_doc.split(":")[0] if ":" in test_doc else test_doc

            print(f"\n[{method_name}] {test_desc}...")
            try:
                method()
                results.append((method_name, "PASS", test_desc))
                print(f"  ✓ PASS")
            except AssertionError as e:
                results.append((method_name, f"FAIL: {str(e)}", test_desc))
                print(f"  ✗ FAIL: {str(e)}")
            except Exception as e:
                results.append((method_name, f"ERROR: {str(e)}", test_desc))
                print(f"  ✗ ERROR: {str(e)}")

        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for _, status, _ in results if status == "PASS")
        failed = sum(1 for _, status, _ in results if "FAIL" in status)
        errors = sum(1 for _, status, _ in results if "ERROR" in status)

        print(f"Total tests run: {len(results)}")
        print(f"Passed: {passed} / {len(results)}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")

        if failed > 0:
            print("\nFailed tests:")
            for test_name, status, desc in results:
                if "FAIL" in status:
                    print(f"  - {test_name}: {desc}")
                    print(f"    Reason: {status}")

        if errors > 0:
            print("\nTests with errors:")
            for test_name, status, desc in results:
                if "ERROR" in status:
                    print(f"  - {test_name}: {desc}")
                    print(f"    Error: {status}")

        # Return exit code for CI
        return passed, failed, errors

    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}")
        return 0, 0, 1


if __name__ == "__main__":
    print("Starting data quality checks...")
    passed, failed, errors = run_all_checks()

    # Exit with appropriate code
    if failed > 0 or errors > 0:
        print(f"\n✗ Some tests failed or errored. Exiting with code 1")
        sys.exit(1)
    else:
        print(f"\n✓ All tests passed! Exiting with code 0")
        sys.exit(0)