import pandas as pd
import random

# Define a function to extract the decade
def extract_decade(file_path):
    year_match = file_path.split("\\")[2]  
    year = int(year_match[:4])
    return (year // 10) * 10 

# Load your sampled dataset
data = pd.read_csv("./data/data_annotated.csv")
data['decade'] = data['file'].apply(extract_decade)

# Get a dict of all files in the dataset by decade
files_by_decade = {}

# Loop through each row in the DataFrame
for _, row in data.iterrows():
    decade = row['decade']
    file_name = row['file']
    
    # Initialize the list for the decade if not already present
    if decade not in files_by_decade:
        files_by_decade[decade] = []
    
    # Add the file name if it's not already in the list for that decade
    if file_name not in files_by_decade[decade]:
        files_by_decade[decade].append(file_name)

# Generate test_data
test_files = []
for i in range(1,4):
    test_files.extend(files[-i] for files in files_by_decade.values())
test_data = data[data['file'].isin(test_files)].reset_index(drop =True).drop(data.columns[-1], axis=1)

# Generate val_data
val_files = []
for i in range(1,4):
    val_files.extend(files[-i-3] for files in files_by_decade.values())
val_data = data[data['file'].isin(val_files)].reset_index(drop =True).drop(data.columns[-1], axis=1)

# Generate train data
train_data = data[~data["file"].isin(test_data["file"]) & ~data["file"].isin(val_data["file"])].reset_index(drop =True).drop(data.columns[-1], axis=1)


# Save to CSV files
test_data.to_csv("./data/test_set.csv", index=False)
val_data.to_csv("./data/val_set.csv", index=False)
train_data.to_csv("./data/train_set.csv", index=False)