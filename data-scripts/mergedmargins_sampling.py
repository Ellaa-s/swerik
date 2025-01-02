import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

# Load your sampled dataset
data = pd.read_csv(f'{args.data_folder}/data_annotated.csv')

# the target column containing merged-margin classification
target_column = 'merged'

# Stratified Shuffle Split to create train, validation, and test sets
# Step 1: Split data into train and temporary sets (temp contains validation + test)
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in split1.split(data, data[target_column]):
    train_merged_data = data.iloc[train_idx]
    temp_merged_data = data.iloc[temp_idx]

# Step 2: Split temporary set into validation and test sets
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in split2.split(temp_merged_data, temp_merged_data[target_column]):
    val_merged_data = temp_merged_data.iloc[val_idx]
    test_merged_data = temp_merged_data.iloc[test_idx]

# Reset index for all sets
train_merged_data = train_merged_data.reset_index(drop=True)
val_merged_data = val_merged_data.reset_index(drop=True)
test_merged_data = test_merged_data.reset_index(drop=True)

# # Save to CSV files
train_merged_data.to_csv(f'{args.data_folder}/train_merged_stratified.csv', index=False)
val_merged_data.to_csv(f'{args.data_folder}/val_merged_stratified.csv', index=False)
test_merged_data.to_csv(f'{args.data_folder}/test_merged_stratified.csv', index=False)

# Check class distribution to ensure stratification
print("Class Distribution in merged margin Train Set:")
print(train_merged_data["merged"].value_counts())
print("\nClass Distribution in merged margin Validation Set:")
print(val_merged_data["merged"].value_counts())
print("\nClass Distribution in merged margin Test Set:")
print(test_merged_data["merged"].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", type=str, default='./data/data_stratified_sampling')
