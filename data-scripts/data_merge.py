## Ella

import pandas as pd
import os

# Specify the folder where the partitioned CSV files are stored
input_folder = "./data_partitioned"
output_file = "./data_annotated.csv"

# List all CSV files in the input folder
csv_files = sorted([file for file in os.listdir(input_folder) if file.endswith('.csv')])

# Read and concatenate all CSV files
data_combined = pd.concat([pd.read_csv(os.path.join(input_folder, file)) for file in csv_files])

# Save the combined DataFrame to a new CSV file
data_combined.to_csv(output_file, index=False)

# Verify the length of the combined DataFrame
print(f"Total rows in combined file: {len(data_combined)}")
