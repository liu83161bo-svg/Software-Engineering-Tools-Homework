#!/bin/bash

# Setup script for AI System Course HW3
echo "Setting up data pipeline for EEG Age Classification..."

# Step 1: Install dependencies
echo "Installing Python dependencies..."
pip install pandas numpy dvc

# Step 2: Initialize DVC (if not already)
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    dvc remote add -d mylocalremote .dvc/cache
fi

# Step 3: Generate sample data
echo "Generating sample EEG data..."
python data/generate_data.py

# Step 4: Create subject-based splits
echo "Creating subject-based splits..."
python data/create_splits.py

# Step 5: Add data to DVC tracking
echo "Adding data to DVC version control..."
dvc add data/raw/sample_eeg_data.csv
dvc add data/processed/sample_eeg_data.jsonl
dvc add data/splits/

# Step 6: Commit DVC files
echo "Committing DVC files to Git..."
git add data/*.dvc .dvc .dvcignore dvc.yaml
git commit -m "HW3: Add EEG dataset with DVC version control"

echo "Setup complete! Run 'dvc push' to sync data to remote storage."