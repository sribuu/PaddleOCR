import os
import re
import uuid

def get_base_name(file_name):
    match = re.search(r'^(.+?)_\d+$', file_name)
    if match:
        return match.group(1)
    return file_name

def rename_files(data_file):
    with open(data_file, 'r') as file:
        lines = file.readlines()

    prev_base_name = ""
    random_string = str(uuid.uuid4().hex)[:5]  # Generate initial random string

    new_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        file_path, data = line.split('\t')
        file_dir, file_name = os.path.split(file_path)
        file_name, file_ext = os.path.splitext(file_name)

        base_name = get_base_name(file_name)

        if base_name != prev_base_name:
            random_string = str(uuid.uuid4().hex)[:10]  # Generate new random string

        prev_base_name = base_name

        # Extract the index from the file name using regex
        match = re.search(r'_(\d+)$', file_name)
        if match:
            index = match.group(1)
            new_file_name = f"{random_string}_{index}{file_ext}"
        else:
            new_file_name = f"{random_string}{file_ext}"
    
        new_file_path = os.path.join(file_dir, new_file_name)
        new_line = f"{new_file_path}\t{data}\n"
        if os.path.exists(file_path):
            os.rename(file_path, new_file_path)
            new_lines.append(new_line)

    with open(data_file, 'w') as file:
        file.writelines(new_lines)

# Contoh penggunaan:
data_file = "Label.txt"
rename_files(data_file)
