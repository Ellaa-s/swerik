import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Load your sampled dataset
train_data = pd.read_csv("/home/nikitha/swerik/data/train_set.csv")
val_data = pd.read_csv("/home/nikitha/swerik/data/val_set.csv")
test_data = pd.read_csv("/home/nikitha/swerik/data/test_set.csv")

# Assume that 'label' is the column that contains the classes
target_column = 'marginal_text'

# Check class distribution to ensure stratification
print("Class Distribution in Train Set:")
print(train_data["marginal_text"].value_counts())
print("\nClass Distribution in Validation Set:")
print(val_data["marginal_text"].value_counts())
print("\nClass Distribution in Test Set:")
print(test_data["marginal_text"].value_counts())
