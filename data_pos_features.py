import os
import requests
from pyriksdagen.download import fetch_files, dl_kb_blocks, LazyArchive, count_pages, convert_alto
from pyriksdagen.utils import infer_metadata, XML_NS
import argparse
import progressbar
from get_positional_features import get_page_position_information, get_protocol_information
import alto
from alto import parse_file
from lxml import etree
import pandas as pd


tei_ns = "{http://www.tei-c.org/ns/1.0}"
xml_ns = "{http://www.w3.org/XML/1998/namespace}"
parser = etree.XMLParser(remove_blank_text=True)

    
# get filenames from file_paths (remove leading zeros in the filename)
def get_filename_from_filepath(file_path):
    filename = file_path.rsplit("\\", 1)[-1].rsplit(".xml", 1)[0]
    parts = filename.split("--")
    parts[-1] = str(int(parts[-1]))
    cleaned_filename = "--".join(parts)
    return cleaned_filename

def get_page_number(our_data, file_path):
    page_number = our_data.loc[our_data['file'] == file_path, 'page_number']-1
    return page_number.values[0]

def create_raw_path(package_id, page_number):
    formatted_path = package_id.replace("-", "_").replace("--", "__")
    raw_path = f"{formatted_path}-{page_number:03}.xml"
    return raw_path

def main(args):
    df = count_pages(args.start, args.end)
    # gives a list of all file names in the given years
    package_ids = list(df["protocol_id"])
    archive = LazyArchive()
    
    # read in our data set we want to get positional features for
    our_data = pd.read_csv("./swerik/data/test_set.csv")
    our_data = our_data[115:135]
    
    # get all the file paths in our annotated data sets
    file_paths = []
    for row in range(our_data.shape[0]):
        if our_data.iloc[row]["file"] not in file_paths:
            file_paths.append(our_data.iloc[row]["file"])
    print(f"this is the file_path{file_paths}")
    
    # key we want to add to our csv later
    keys_to_add = ["posLeft", "posUpper","posRight", "posLower"]

    # I only want to do that for the package ids, which correlate to my file paths:
    for file_path in progressbar.progressbar(file_paths):

        print(file_path)
        package_id = get_filename_from_filepath(file_path)
        print(package_id)
        pkg = archive.get(package_id)        
        
        page_number = get_page_number(our_data,file_path)
        raw_file_path = create_raw_path(package_id,page_number)
        print(raw_file_path)
        alto = parse_file(pkg.get_raw(raw_file_path)) 
            
        # Parse an XML string or file
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.parse(file_path, parser).getroot()
        
        # iterate over all elements on one page
        for elem in root.iter():
            elem_id = elem.get(f'{XML_NS}id')
            # If element is seg or note (these contain text)
            if elem.tag in {f"{tei_ns}seg", f"{tei_ns}note"}:
                #get the positional features
                out_pos = get_page_position_information(elem, alto, elem_type = None)
                #print(out_pos)
                for key in keys_to_add:
                    #our_data[key] = out_pos[key]
                    our_data.loc[our_data['id'] == elem_id, key] = out_pos[key]
                    
        our_data.to_csv("./swerik/data/test_pos_set.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1867)
    parser.add_argument("--end", type=int, default=1990)
    # parser.add_argument("--authority", type=str, default="SWERIK Project, 2023-2027")
    # parser.add_argument("--protocol_ids", type=str, nargs="+", default=None)
    # parser.add_argument("--local-alto", type=str, nargs="+", default=None, help="Locally stored alto package (folder=protocol name, contents=pages.")
    # parser.add_argument("--alto-path", type=str, help="Path to `--local-alto` directories")
    args = parser.parse_args()
    main(args)