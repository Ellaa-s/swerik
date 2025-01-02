import pandas as pd
import argparse

def main(args):
    # Load your sampled dataset
    train_data = pd.read_csv(f'{args.data_folder}/train_set.csv')
    val_data = pd.read_csv(f'{args.data_folder}/val_set.csv')
    test_data = pd.read_csv(f'{args.data_folder}/test_set.csv')

    # Assume that 'marginal_text' is the column that contains the classes
    target_column = 'marginal_text'

    # Check class distribution for the unsampled dataset
    print("Class Distribution in Train Set:")
    print(train_data["marginal_text"].value_counts())
    print("\nClass Distribution in Validation Set:")
    print(val_data["marginal_text"].value_counts())
    print("\nClass Distribution in Test Set:")
    print(test_data["marginal_text"].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", type=str, default="../data")
    args = parser.parse_args()
    main(args)
