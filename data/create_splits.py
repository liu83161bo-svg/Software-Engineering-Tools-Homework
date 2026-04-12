"""
Create data splits for EEG dataset based on subject_hash
Strategy: Split by subject to prevent data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict


def split_by_subject(df, test_ratio=0.15, val_ratio=0.15, random_seed=42):
    """
    Split dataset by subject to prevent data leakage

    Args:
        df: DataFrame with 'subject_hash' column
        test_ratio: Proportion of subjects for test set
        val_ratio: Proportion of subjects for validation set
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with split assignments
    """
    np.random.seed(random_seed)

    # Get unique subjects
    subjects = df['subject_hash'].unique()
    n_subjects = len(subjects)

    print(f"Total unique subjects: {n_subjects}")
    print(f"Total trials: {len(df)}")

    # Shuffle subjects
    shuffled_subjects = np.random.permutation(subjects)

    # Calculate split sizes
    n_test = int(n_subjects * test_ratio)
    n_val = int(n_subjects * val_ratio)

    # Assign subjects to splits
    test_subjects = shuffled_subjects[:n_test]
    val_subjects = shuffled_subjects[n_test:n_test + n_val]
    train_subjects = shuffled_subjects[n_test + n_val:]

    # Create split dictionary
    splits = defaultdict(list)
    for _, row in df.iterrows():
        subject = row['subject_hash']
        trial_id = row['trial_id']

        if subject in test_subjects:
            splits['test'].append(trial_id)
        elif subject in val_subjects:
            splits['val'].append(trial_id)
        else:
            splits['train'].append(trial_id)

    # Print statistics
    print(f"\nSplit Statistics:")
    print(f"Train: {len(splits['train'])} trials ({len(train_subjects)} subjects)")
    print(f"Validation: {len(splits['val'])} trials ({len(val_subjects)} subjects)")
    print(f"Test: {len(splits['test'])} trials ({len(test_subjects)} subjects)")

    # Check for data leakage
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])

    assert len(train_set & val_set) == 0, "Data leakage: Train/Val overlap!"
    assert len(train_set & test_set) == 0, "Data leakage: Train/Test overlap!"
    assert len(val_set & test_set) == 0, "Data leakage: Val/Test overlap!"

    return splits


def create_split_files(df, output_dir="data/splits"):
    """
    Create split files and save to disk

    Args:
        df: Input DataFrame
        output_dir: Directory to save split files
    """
    # Create splits
    splits = split_by_subject(df)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save each split as text file
    for split_name, trial_ids in splits.items():
        file_path = Path(output_dir) / f"{split_name}_ids.txt"
        with open(file_path, 'w') as f:
            for trial_id in trial_ids:
                f.write(f"{trial_id}\n")
        print(f"Saved {len(trial_ids)} trial IDs to {file_path}")

    # Also save as JSON for easy reference
    json_path = Path(output_dir) / "splits_metadata.json"
    metadata = {
        "split_strategy": "subject_based",
        "description": "Splits created by subject_hash to prevent data leakage",
        "total_trials": len(df),
        "unique_subjects": df['subject_hash'].nunique(),
        "split_counts": {k: len(v) for k, v in splits.items()}
    }

    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSplit metadata saved to {json_path}")

    return splits


def validate_splits(df, split_dir="data/splits"):
    """
    Validate that splits are correctly implemented

    Args:
        df: Original DataFrame
        split_dir: Directory containing split files
    """
    split_dir = Path(split_dir)

    # Load split IDs
    splits = {}
    for split_file in split_dir.glob("*_ids.txt"):
        split_name = split_file.stem.replace("_ids", "")
        with open(split_file, 'r') as f:
            trial_ids = [int(line.strip()) for line in f if line.strip()]
        splits[split_name] = set(trial_ids)

    # Check all trial IDs are accounted for
    all_split_ids = set()
    for split_set in splits.values():
        all_split_ids.update(split_set)

    all_trial_ids = set(df['trial_id'])

    print(f"\nSplit Validation:")
    print(f"Total trials in dataset: {len(all_trial_ids)}")
    print(f"Total trials in splits: {len(all_split_ids)}")

    # Check for missing or extra trials
    missing = all_trial_ids - all_split_ids
    extra = all_split_ids - all_trial_ids

    if missing:
        print(f"WARNING: {len(missing)} trials missing from splits")
    if extra:
        print(f"WARNING: {len(extra)} extra trials in splits")

    # Check for overlaps between splits
    for split1_name, split1_set in splits.items():
        for split2_name, split2_set in splits.items():
            if split1_name != split2_name:
                overlap = split1_set & split2_set
                if overlap:
                    print(f"ERROR: {len(overlap)} overlapping trials between {split1_name} and {split2_name}")

    return len(missing) == 0 and len(extra) == 0


if __name__ == "__main__":
    # Load the dataset
    data_path = Path(__file__).parent.parent / "data" / "sample_eeg_data.csv"
    df = pd.read_csv(data_path)

    print("Creating subject-based splits...")
    splits = create_split_files(df)

    print("\nValidating splits...")
    is_valid = validate_splits(df)

    if is_valid:
        print("✓ All splits validated successfully!")
    else:
        print("✗ Split validation failed!")
        exit(1)