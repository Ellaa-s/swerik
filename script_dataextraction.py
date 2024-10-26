import os
import random
import shutil
import argparse


def extract_files(base_folder, start_year, end_year, output_folder, files_per_folder=5):
    list_years = set(list(range(start_year , start_year+10)))

    available_year = []

    for folder in os.listdir(base_folder):
        if int(folder[:4]) in list_years:
            available_year.append(folder)

    Unique_files = set()

    # Loop through each year folder in the specified range
    for folder_name in available_year:
        year_folder = os.path.join(base_folder, folder_name)

        # Check if the year folder exists
        if not os.path.exists(year_folder):
            print(f"Folder for the year {folder_name} does not exist. Skipping.")
            continue

        # List all XML files in the year folder
        xml_files = [file for file in os.listdir(year_folder) if file.endswith('.xml')]

        #filter files that has already been selected 
        filter_files = [file for file in xml_files if file not in Unique_files]

        # If there are fewer XML files than requested, use all of them
        if len(filter_files) < files_per_folder:
            selected_files = filter_files[:files_per_folder]
        else:
            # Randomly select the specified number of XML files
            selected_files = random.sample(filter_files, files_per_folder)
        
        Unique_files.update(selected_files)
        
        # Create a subfolder in the output folder for the specific year
        year_output_folder = os.path.join(output_folder, folder_name)
        if not os.path.exists(year_output_folder):
            os.makedirs(year_output_folder)

        # Copy each selected file to the year-specific output folder
        for file_name in selected_files:
            source_path = os.path.join(year_folder, file_name)
            dest_path = os.path.join(year_output_folder, file_name)
            shutil.copy2(source_path, dest_path)

        print(f"Copied {len(selected_files)} files from {year_folder} to {year_output_folder}")

    
# Set parameters
#base_folder = "/home/nikitha/riksdagen-records/data"  # Path to the main dataset folder               # Set for a 10-year span
#output_folder = "/home/nikitha/random_folder"  # Main output folder for selected XML files
  

# Set up command-line arguments
parser = argparse.ArgumentParser(description="Extract XML files from specified year range.")
parser.add_argument("start_year", type=int, help="The start year for the range (e.g., 1867)")
#parser.add_argument("end_year", type=int, help="The end year for the range (e.g., 1876)")
parser.add_argument("--base_folder", type=str, default="/", help="/home/nikitha/riksdagen-records/data")
parser.add_argument("--output_folder", type=str, default="/home/nikitha/random_folder", help="Main output folder for selected XML files")
parser.add_argument("--files_per_folder", type=int, default=5, help="Number of files to randomly extract per folder")

# Parse arguments
args = parser.parse_args()
end_year = args.start_year + 9


# Run the extraction function
extract_files(args.base_folder, args.start_year, end_year, args.output_folder, args.files_per_folder)

