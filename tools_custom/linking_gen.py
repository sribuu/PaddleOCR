import json
import argparse
import os

# Membuat objek parser
parser = argparse.ArgumentParser()

# Menambahkan argumen yang diharapkan
parser.add_argument(
    "--labelFile", type=str, default="Label.txt", help="Input label as Label.txt"
)
parser.add_argument(
    "--labelOutputFile",
    type=str,
    default="Label-linked.txt",
    help="Generated Label with id and linking. e.g Label-output.txt",
)

# Parse argumen dari terminal
args = parser.parse_args()

# Mendapatkan direktori dari full path file
directory = os.path.dirname(args.labelFile)

# Mendapatkan current directory
current_directory = os.getcwd()

rootPath = directory

if directory == "":
    rootPath = current_directory

rootPath = f"{rootPath}/"
filename = f"{rootPath}{args.labelFile}"  # file to read
_dict = {}  # text will be store as dictionary

# creating dictionary
with open(filename) as fh:
    for line in fh:
        # reads each line and turn into key value pair dict
        command, description = line.split("\t", 1)
        _dict[command] = description.strip()

        for key, values in _dict.items():
            temp = json.loads(values)  # temporary store array values from dict

            c = 1  # while counting starts from 1 to allow us safely remove elements from the temp
            i = c - 1 # real index from array

            while c <= len(temp):
                temp_data = temp[i]

                if len(temp_data["linking"]) == 0:
                    del temp[i]  # remove annotation data if it has no linking
                    continue
                else:
                    if "key_cls" in temp_data:  # rename key_cls to label if needed
                        temp_data["label"] = temp_data.pop("key_cls")

                    if temp_data["label"] == "None":
                        temp_data["label"] = "IGNORE"

                    temp[i] = temp_data  # modify current data in temp

                c += 1

            _dict[key] = json.dumps(temp)  # putting back into _dict

# convert dictionary into text separated by new line
text = "\n".join([f"{key}\t{value}" for key, value in _dict.items()])

# write the text to a file
with open(f"{rootPath}{args.labelOutputFile}", "w") as file:
    file.write(text + "\n")
