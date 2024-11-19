from pyriksdagen.download import LazyArchive
from pyriksdagen.utils import XML_NS
import argparse
import progressbar
from get_positional_features import get_page_position_information, get_protocol_information
from alto import parse_file
from lxml import etree
import pandas as pd


tei_ns = "{http://www.tei-c.org/ns/1.0}"
xml_ns = "{http://www.w3.org/XML/1998/namespace}"
parser = etree.XMLParser(remove_blank_text=True)

    
# get filenames from file_paths (remove leading zeros in the filename)
def get_filename_from_filepath(file_path):
    #filename = file_path.rsplit("\\", 1)[-1].rsplit(".xml", 1)[0]
    #parts = filename.split("--")
    #parts[-1] = str(int(parts[-1]))
    #cleaned_filename = "--".join(parts)
    cleaned_filename = file_path.split('/')[-2].replace('.jp2', '')
    print(cleaned_filename)
    return cleaned_filename

def get_package_id(page_name):
    package_id = page_name.replace("__", "--")
    package_id = page_name.replace("_", "-")
    package_id = package_id.rsplit('-', 1)[0]
    return package_id

def get_page_number(our_data, file_path):
    page_number = our_data.loc[our_data['file'] == file_path, 'page_number']-1
    return page_number.values[0]

def create_raw_path(package_id, page_number):
    year = int(package_id.split('-')[1])
    if 1860 <= year < 1870:
        parts = package_id.split('--')
        last_number = parts[-1]
        formatted_last_number = last_number.zfill(4)  # "0116"
        new_package_id = f"{parts[0]}--{parts[1]}--{formatted_last_number}"
        formatted_path = new_package_id.replace("-", "_").replace("--", "__")    
        raw_path = f"{formatted_path}-{page_number:03}.xml"
    else:
        formatted_path = package_id.replace("-", "_").replace("--", "__")
        raw_path = f"{formatted_path}-{page_number:03}.xml"
    return raw_path

def main(args):
    archive = LazyArchive()

    # read in our data set we want to get positional features for
    our_data = pd.read_csv(args.input_file) #pd.read_csv("./swerik/data/test_set.csv")
    # for easier debugging
    #our_data = our_data[0:50]
    
    # get all the kb_paths and file_paths in our annotated data sets
    kb_paths = []
    file_paths = []
    for row in range(our_data.shape[0]):
        if our_data.iloc[row]["page_link"] not in kb_paths and our_data.iloc[row]["file"] not in file_paths:
            kb_paths.append(our_data.iloc[row]["page_link"])
            file_paths.append(our_data.iloc[row]["file"])
    print(f"all file_paths: {file_paths}")
    print(f"all kb_paths: {kb_paths}")

    
    # key we want to add to our csv later
    keys_to_add = ["posLeft", "posUpper","posRight", "posLower"]
    prot_keys_to_add = ["year", "relative_page_number","even_page", "second_chamber", "unicameral"]


    # loop over the files in the input data
    for file_path, kb_path in progressbar.progressbar(zip(file_paths, kb_paths)):
        # get page_name to reach the page_name.xml alto file 
        page_name = get_filename_from_filepath(kb_path)
        
        # get package_id (formatted differently then page_name)
        # format is necessary to get the alto file from the archive
        package_id = get_package_id(page_name)
        print(f"package id: {package_id}")
        pkg = archive.get(package_id)
        
        #save pkg to this file (for easier debugging)
        with open('./swerik/package.txt', 'w') as f:
            print(pkg, file=f)

        # Access alto file
        raw_file_path = page_name + '.xml'
        print(f"Raw file path: {raw_file_path}")
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
                prot_inf = get_protocol_information(elem)
                #print(out_pos)
                for key in keys_to_add:
                    our_data.loc[our_data['id'] == elem_id, key] = out_pos[key]
                for prot_key in prot_keys_to_add:
                    our_data.loc[our_data['id'] == elem_id, prot_key] = prot_inf[prot_key]
        our_data.to_csv(args.output_file, index=False) #"./swerik/data/test_pos_set.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1867)
    parser.add_argument("--end", type=int, default=1990)
    parser.add_argument("--input_file", type=str, default="file path", help="Input file to which positional features should be added")
    parser.add_argument("--output_file", type=str, default="file path", help="Main output folder for selected XML files")
    args = parser.parse_args()
    main(args)