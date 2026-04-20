import pandas as pd
import numpy as np
import os
from pathlib import Path


def create_subject_based_splits():
    """
    Create train/val/test splits based on subject_hash.
    Ensures all trials from the same subject are in the same split.

    Split ratio: 70% train, 15% validation, 15% test
    """
    # Load the data
    data_path = Path("sample_eeg_data.csv")
    if not data_path.exists():
        print("Error: Data file not found. Run generate_sample_data.py first.")
        return

    df = pd.read_csv(data_path)

    # Get unique subjects
    subjects = df['subject_hash'].unique()
    n_subjects = len(subjects)
    print(f"Found {n_subjects} unique subjects")

    # Shuffle subjects randomly (with fixed seed)
    np.random.seed(42)
    shuffled_subjects = np.random.permutation(subjects)

    # Calculate split sizes
    n_train = int(0.7 * n_subjects)
    n_val = int(0.15 * n_subjects)

    # Create splits
    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train:n_train + n_val]
    test_subjects = shuffled_subjects[n_train + n_val:]

    # Create splits directory
    splits_dir = Path(__file__).parent / "splits"
    splits_dir.mkdir(exist_ok=True)

    # Save subject lists to files
    def save_subject_list(filename, subject_list):
        with open(splits_dir / filename, 'w') as f:
            for subject in subject_list:
                f.write(f"{subject}\n")

    save_subject_list("train_subjects.txt", train_subjects)
    save_subject_list("val_subjects.txt", val_subjects)
    save_subject_list("test_subjects.txt", test_subjects)

    print(f"Split sizes:")
    print(f"  Train: {len(train_subjects)} subjects")
    print(f"  Validation: {len(val_subjects)} subjects")
    print(f"  Test: {len(test_subjects)} subjects")

    # Verify no overlap
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    if train_set & val_set:
        print("ERROR: Overlap between train and validation sets!")
    if train_set & test_set:
        print("ERROR: Overlap between train and test sets!")
    if val_set & test_set:
        print("ERROR: Overlap between validation and test sets!")

    # Count samples per split
    def count_samples_in_split(subject_list):
        return len(df[df['subject_hash'].isin(subject_list)])

    train_samples = count_samples_in_split(train_subjects)
    val_samples = count_samples_in_split(val_subjects)
    test_samples = count_samples_in_split(test_subjects)

    print(f"\nSample counts per split:")
    print(f"  Train: {train_samples} samples")
    print(f"  Validation: {val_samples} samples")
    print(f"  Test: {test_samples} samples")
    print(f"  Total: {train_samples + val_samples + test_samples} samples")

    # Verify all subjects are assigned
    all_assigned = len(train_set | val_set | test_set)
    if all_assigned == n_subjects:
        print("\n✓ All subjects assigned to splits successfully")
    else:
        print(f"\nERROR: {n_subjects - all_assigned} subjects not assigned!")

    return {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects
    }


if __name__ == "__main__":
    splits = create_subject_based_splits()