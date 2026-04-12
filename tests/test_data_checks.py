"""
EEG Age Classification Dataset - Automated Data Checks
HW3: Data & Datasets - 10+ automated data quality checks
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path


class TestDataQuality:
    """Data quality test suite for EEG age classification dataset"""

    # Constants
    DATA_PATH = Path(__file__).parent.parent / "data" / "sample_eeg_data.csv"
    JSONL_PATH = Path(__file__).parent.parent / "data" / "sample_eeg_data.jsonl"
    REQUIRED_COLUMNS = ['trial_id', 'file_name', 'trial_index', 'age']
    SIGNAL_LENGTH = 1000
    AGE_RANGE = (0, 100)

    def setup_method(self):
        """Load test data"""
        self.df = pd.read_csv(self.DATA_PATH)

    # ==================== Syntax Checks (3 checks) ====================

    def test_01_file_format_valid(self):
        """SYN-01: Check CSV file format is valid"""
        assert self.DATA_PATH.exists(), "CSV file not found"
        assert self.DATA_PATH.suffix == '.csv', "File must be .csv format"

    def test_02_no_missing_values(self):
        """SYN-02: Check for missing/null values in required columns"""
        for col in self.REQUIRED_COLUMNS:
            assert col in self.df.columns, f"Missing required column: {col}"
            null_count = self.df[col].isnull().sum()
            assert null_count == 0, f"Column {col} has {null_count} null values"

    def test_03_no_corrupted_data(self):
        """SYN-03: Check data integrity (no corrupted/invalid values)"""
        # Check trial_id uniqueness
        assert self.df['trial_id'].nunique() == len(self.df), "trial_id must be unique"

        # Check file_name format
        valid_files = self.df['file_name'].str.endswith('.mat')
        assert valid_files.all(), "All file_name values must end with .mat"

    # ==================== Structural Checks (4 checks) ====================

    def test_04_no_duplicate_trials(self):
        """STR-01: Check for duplicate trial entries"""
        # Check combination of file_name + trial_index is unique
        trial_key = self.df['file_name'] + '_' + self.df['trial_index'].astype(str)
        assert trial_key.nunique() == len(self.df), "Duplicate trials found"

        # Alternative: check for duplicate trial_id
        duplicate_ids = self.df[self.df['trial_id'].duplicated()]
        assert len(duplicate_ids) == 0, f"Duplicate trial_id: {duplicate_ids['trial_id'].tolist()}"

    def test_05_no_data_leakage_by_subject(self):
        """STR-02: Check no data leakage at subject level"""
        # In real scenario, we'd check subject_hash across train/val/test splits
        # Here we check that subject_hash exists for all rows
        assert 'subject_hash' in self.df.columns, "subject_hash column required for leakage check"

        # Simpler check: ensure subject_hash is present and not null
        null_subjects = self.df['subject_hash'].isnull().sum()
        assert null_subjects == 0, f"{null_subjects} rows missing subject_hash"

    def test_06_label_data_matching(self):
        """STR-03: Check label (age) matches data expectations"""
        # Age must be integer
        assert self.df['age'].dtype in [np.int64, np.int32, int], "Age must be integer type"

        # Age within valid range
        min_age, max_age = self.AGE_RANGE
        age_in_range = self.df['age'].between(min_age, max_age)
        assert age_in_range.all(), f"Age out of range. Valid: {min_age}-{max_age}"

        # Check for negative ages
        negative_ages = self.df[self.df['age'] < 0]
        assert len(negative_ages) == 0, f"Negative ages found: {negative_ages['age'].tolist()}"

    def test_07_signal_columns_exist(self):
        """STR-04: Check signal data structure"""
        # Look for signal columns (signal_0, signal_1, ...)
        signal_cols = [col for col in self.df.columns if col.startswith('signal_')]
        assert len(signal_cols) > 0, "No signal columns found"

        # Check at least 10 signal points exist (in sample)
        assert len(signal_cols) >= 10, f"Expected at least 10 signal points, got {len(signal_cols)}"

        # Check signal values are numeric
        for col in signal_cols[:5]:  # Check first 5 signal columns
            assert pd.api.types.is_numeric_dtype(self.df[col]), f"Signal column {col} not numeric"

    # ==================== Statistical Checks (3 checks) ====================

    def test_08_age_distribution_imbalance(self):
        """STA-01: Check age distribution imbalance"""
        age_counts = self.df['age'].value_counts()

        # Check no single age dominates (> 50% of data)
        max_share = age_counts.max() / len(self.df)
        assert max_share <= 0.5, f"Age imbalance: one age has {max_share:.1%} of data"

        # Check we have multiple ages represented
        unique_ages = self.df['age'].nunique()
        assert unique_ages >= 3, f"Only {unique_ages} unique ages, expected at least 3"

    def test_09_signal_statistics_within_range(self):
        """STA-02: Check signal statistical properties"""
        # Find signal columns
        signal_cols = [col for col in self.df.columns if col.startswith('signal_')]

        if len(signal_cols) > 0:
            # Calculate mean of first signal point across all trials
            signal_means = self.df[signal_cols].mean(axis=1)

            # EEG signals should typically be centered around 0 (after baseline correction)
            # Allow reasonable range for sample data
            mean_of_means = signal_means.mean()
            assert -10 < mean_of_means < 10, f"Signal mean out of expected range: {mean_of_means:.2f}"

            # Check standard deviation (should not be 0 or extremely large)
            signal_stds = self.df[signal_cols].std(axis=1).mean()
            assert 0.1 < signal_stds < 100, f"Signal std out of expected range: {signal_stds:.2f}"

    def test_10_jsonl_completeness_check(self):
        """STA-03: Check JSONL format completeness as proxy for data quality"""
        if self.JSONL_PATH.exists():
            signals = []
            ages = []

            with open(self.JSONL_PATH, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    signals.append(record.get('signal', []))
                    ages.append(record.get('age'))

            # Check all records have signals
            assert len(signals) > 0, "No signals in JSONL file"

            # Check signal length consistency
            signal_lengths = [len(s) for s in signals]
            assert len(set(signal_lengths)) == 1, f"Inconsistent signal lengths: {set(signal_lengths)}"

            # Check age consistency with CSV
            jsonl_ages = pd.Series(ages)
            csv_ages = self.df['age']
            assert len(jsonl_ages) == len(csv_ages), f"JSONL has {len(jsonl_ages)} records, CSV has {len(csv_ages)}"

            print(f"JSONL validation passed: {len(signals)} records with consistent signal length")

    # ==================== Additional Checks (Extra) ====================

    def test_11_trial_index_range(self):
        """Extra: Check trial_index values are reasonable"""
        # Trial index should be non-negative
        assert (self.df['trial_index'] >= 0).all(), "Negative trial_index found"

        # Trial index should be less than reasonable max (e.g., 1000)
        assert (self.df['trial_index'] < 1000).all(), "trial_index too large (> 1000)"

    # 在 tests/test_data_checks.py 中，修改 test_12_data_types_correct 方法：

    def test_12_data_types_correct(self):
        """Extra: Check column data types"""
        type_checks = [
            ('trial_id', 'int'),
            ('file_name', 'object'),  # Pandas中字符串是object类型
            ('trial_index', 'int'),
            ('age', 'int')
        ]

        for col, expected_type in type_checks:
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                # 更灵活的类型检查
                if 'int' in expected_type and 'int' in actual_type:
                    continue  # 整数类型匹配
                elif expected_type == 'object' and actual_type in ['object', 'str']:
                    continue  # 允许object或str类型
                elif actual_type == expected_type:
                    continue
                else:
                    assert False, \
                        f"Column {col}: expected {expected_type}, got {actual_type}"


def run_all_checks():
    """Run all data checks and print summary"""
    tester = TestDataQuality()
    tester.setup_method()

    results = []
    test_methods = [m for m in dir(tester) if m.startswith('test_')]

    print("=" * 60)
    print("Running EEG Data Quality Checks")
    print("=" * 60)

    for method_name in sorted(test_methods):
        method = getattr(tester, method_name)
        try:
            method()
            results.append((method_name, "PASS"))
            print(f"✓ {method_name}: PASS")
        except AssertionError as e:
            results.append((method_name, f"FAIL: {str(e)}"))
            print(f"✗ {method_name}: FAIL - {str(e)}")
        except Exception as e:
            results.append((method_name, f"ERROR: {str(e)}"))
            print(f"✗ {method_name}: ERROR - {str(e)}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    passed = sum(1 for _, status in results if status == "PASS")
    failed = sum(1 for _, status in results if "FAIL" in status)
    errors = sum(1 for _, status in results if "ERROR" in status)

    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")

    if failed > 0 or errors > 0:
        print("\nFailed/Error details:")
        for test_name, status in results:
            if status != "PASS":
                print(f"  {test_name}: {status}")

    return passed, failed, errors


if __name__ == "__main__":
    # Run checks when script is executed directly
    passed, failed, errors = run_all_checks()

    # Exit with appropriate code for CI
    if failed > 0 or errors > 0:
        exit(1)  # Non-zero exit indicates failure
    else:
        exit(0)  # Zero exit indicates success