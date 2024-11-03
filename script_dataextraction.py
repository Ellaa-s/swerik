import os
import random
import shutil
import argparse
from lxml import etree
import re
import pandas as pd

# Define namespaces and XML parser
tei_ns = "{http://www.tei-c.org/ns/1.0}"
xml_ns = "{http://www.w3.org/XML/1998/namespace}"
xml_parser = etree.XMLParser(remove_blank_text=True)

def extract_text_from_pages(file_path, num_pages=1):
    """
    Parses an XML file, extracts a random sample of pages based on <pb> tags, and retrieves text content line by line.

    Args:
    - file_path (str): Path to the XML file.
    - num_pages (int): Number of random pages to sample.

    Returns:
    - List of dictionaries with page link, each line of text, and file path.
    """
    results = []
    root = etree.parse(file_path, xml_parser)
    pages = []

    # Initialize variables to keep track of the current page content
    current_page_link = None
    page_lines = []

    # Traverse XML and find 'pb' elements (page breaks) and text nodes
    for elem in root.iter():
        # If element is a page marker (<pb>), save the previous page's content
        if elem.tag == f"{tei_ns}pb":
            # Append the current page to the list if it has a link
            if current_page_link and page_lines:
                pages.append({"page_link": current_page_link, "lines": page_lines})

            # Start a new page
            current_page_link = elem.attrib.get("facs")  # Page identifier
            page_lines = []  # Reset lines for the new page

        # If the element is a paragraph or segment, add each line to the current page
        elif elem.tag in {f"{tei_ns}u", f"{tei_ns}seg", f"{tei_ns}note"}:
            # text = .strip()
            text = re.sub(r'\s+', ' ',(elem.text or "").replace("\n", "")).strip()  # Remove line breaks
            if text:

                page_lines.append(text)

    # Add the last page if it exists
    if current_page_link and page_lines:
        pages.append({"page_link": current_page_link, "lines": page_lines})

    # Randomly sample pages
    if len(pages) <= num_pages:
        sampled_pages = pages
    else:
        sampled_pages = random.sample(pages, num_pages)

    # Store each line of sampled pages with file information
    for page in sampled_pages:
        for line in page["lines"]:
            results.append({
                "file": file_path,
                "page_link": page["page_link"],
                "text_line": line,
                "marginal_text": 0.0
            })

    return results


def extract_files(base_folder, start_year, end_year, files_per_folder=5):
    list_years = set(list(range(start_year , end_year)))

    results = []
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
            # dest_path = os.path.join(year_output_folder, file_name)
            # shutil.copy2(source_path, dest_path)

            result = extract_text_from_pages(source_path)
            print(f'File extracted from {folder_name} and name of the {file_name}')
            results.extend(result)

        df = pd.DataFrame(results)
        df.to_csv('output.csv', index=False)






        #print(f"Copied {len(selected_files)} files from {year_folder} to {year_output_folder}")


# Set parameters
#base_folder = "/home/nikitha/riksdagen-records/data"  # Path to the main dataset folder               # Set for a 10-year span
#output_folder = "/home/nikitha/random_folder"  # Main output folder for selected XML files


# Set up command-line arguments
parser = argparse.ArgumentParser(description="Extract XML files from specified year range.")
#group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument("--start_year",type=int, help="The start year for the range (e.g., 1867)")
#parser.add_argument("end_year", type=int, help="The end year for the range (e.g., 1876)")
parser.add_argument("--base_folder", type=str, default="file path", help="/home/nikitha/riksdagen-records/data")
#parser.add_argument("--output_folder", type=str, default="file path", help="Main output folder for selected XML files")
parser.add_argument("--files_per_folder", type=int, default=5, help="Number of files to randomly extract per folder")
parser.add_argument('--full_data', action='store_true', help="Get the full data set without further input.")

# Parse arguments
args = parser.parse_args()

if args.full_data:
    extract_files(args.base_folder, 1867, 2022, args.output_folder, args.files_per_folder)

elif args.start_year is not None:
    end_year = args.start_year + 9
    extract_files(args.base_folder, args.start_year, end_year, args.files_per_folder)

# Run the extraction function
