import pandas as pd
import os
import numpy as np

#loading the data 
data_main = pd.read_csv("./data.csv")

#splitting the data into 3 parts (change the value of number of splits based on the requirement)
split_value = np.array_split(data_main,3)

output_data = "./data_partitioned"
os.makedirs(output_data,exist_ok=True)
for i,split in enumerate(split_value):
    split.to_csv(f"{output_data}/part_{i+1}.csv", index=False)

#to verify length of files
for i, split in enumerate(split_value):
    print(f"Rows in part {i+1}: {len(split)}")


