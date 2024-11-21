import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Load your sampled dataset
data = pd.read_csv("/home/nikitha/swerik/data/data_annotated.csv")

# Assume that 'label' is the column that contains the classes
target_column = 'marginal_text'

# Stratified Shuffle Split to create train, validation, and test sets
# Step 1: Split data into train and temporary sets (temp contains validation + test)
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in split1.split(data, data[target_column]):
    train_data = data.iloc[train_idx]
    temp_data = data.iloc[temp_idx]

# Step 2: Split temporary set into validation and test sets
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in split2.split(temp_data, temp_data[target_column]):
    val_data = temp_data.iloc[val_idx]
    test_data = temp_data.iloc[test_idx]

# Reset index for all sets
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# # Save to CSV files
train_data.to_csv("/home/nikitha/swerik/data_stratified_sampling/train_data_stratified.csv", index=False)
val_data.to_csv("/home/nikitha/swerik/data_stratified_sampling/val_set_stratified.csv", index=False)
test_data.to_csv("/home/nikitha/swerik/data_stratified_sampling/test_set_stratified.csv", index=False)

# Check class distribution to ensure stratification
print("Class Distribution in Train Set:")
print(train_data["marginal_text"].value_counts())
print("\nClass Distribution in Validation Set:")
print(val_data["marginal_text"].value_counts())
print("\nClass Distribution in Test Set:")
print(test_data["marginal_text"].value_counts())
