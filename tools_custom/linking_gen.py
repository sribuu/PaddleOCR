import json
import argparse
import os

# Membuat objek parser
parser = argparse.ArgumentParser()

# Menambahkan argumen yang diharapkan
parser.add_argument("--labelFile", type=str, default="Label.txt", help="Input label as Label.txt")
parser.add_argument("--linkingFile", type=str, default="key_linking.txt", help="Key-Value pair as linking, e.g key_linking.txt")
parser.add_argument("--labelOutputFile", type=str, default="Label-linked.txt", help="Generated Label with id and linking. e.g Label-output.txt")

# Parse argumen dari terminal
args = parser.parse_args()

# Mendapatkan direktori dari full path file
directory = os.path.dirname(args.labelFile)

# Mendapatkan current directory
current_directory = os.getcwd()

rootPath = directory

if(directory=="") :
    rootPath = current_directory

rootPath = f"{rootPath}/"

filename = f"{rootPath}{args.labelFile}"  # file to read
linking_file = f"{rootPath}{args.linkingFile}"

_dict = {}  # text will be store as dictionary
_links = []  # text will be store as array of key_linking

with open(linking_file, "r") as file:
    file_contents = file.read()  # read file
    _links = eval(file_contents)  # store into key_linking list

# creating dictionary
with open(filename) as fh:
    for line in fh:
        # reads each line and turn into key value pair dict

        command, description = line.split("\t", 1)

        _dict[command] = description.strip()

        for key, values in _dict.items():
            temp = json.loads(values)  # temporary store array values from dict

            for i in range(len(temp)):
                temp_data = temp[i]
                _id = i + 1  # starting from 1
                temp_data["id"] = _id  # assign id
                temp_data["linking"] = []  # default value is empty list
                if "key_cls" in temp_data: #rename key_cls to label if needed
                    temp_data["label"] = (temp_data.pop("key_cls")).upper()
                else:
                    temp_data["label"] = (temp_data["label"]).upper()
                temp[i] = temp_data  # modify current data on temp


            for i in range(len(temp)):
                temp_data = temp[i]

                label = temp_data["label"]  # get label

                value_link = next((item for item in _links if item["value"] == label), None)
                
                if(value_link is not None) : 
                    index = next((index for index, item in enumerate(temp) if item["label"] == value_link["key"]), -1)
                    key_data = temp[index]
                    
                    if(index > -1 and key_data["label"] == value_link["key"]):
                        temp[index]["linking"].append([key_data["id"], temp_data["id"]])
                        temp_data["linking"].append([key_data["id"], temp_data["id"]])

                temp[i] = temp_data  # modify current data on temp

            _dict[key] = json.dumps(temp)  # putting back into _dict

# convert dictionary into text separated by new line
text = "\n".join([f"{key}\t{value}" for key, value in _dict.items()])

# write the text to a file
with open(f"{rootPath}{args.labelOutputFile}", "w") as file:
    file.write(text+"\n")