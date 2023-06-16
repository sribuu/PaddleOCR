import os
import cv2
import numpy as np
import json
from tools.infer.utility import get_rotate_crop_image
import argparse

# Mendapatkan current directory
current_directory = os.getcwd()

# Membuat objek parser
parser = argparse.ArgumentParser()

# Menambahkan argumen yang diharapkan
parser.add_argument("--labelFile", type=str, default="Label.txt", help="Input label as Label.txt")
parser.add_argument("--outputFileGT", type=str, default="rec_gt.txt", help="Out list in txt format")
parser.add_argument("--outputFileDir", type=str, default="crop_img/", help="Ouput File Directory for cropped image")

# Parse argumen dari terminal
args = parser.parse_args()

label_file = os.path.join(current_directory,args.labelFile)
rec_gt_dir = os.path.join(current_directory,args.outputFileGT)
crop_img_dir = os.path.join(current_directory,args.outputFileDir)

ques_img = []
if not os.path.exists(crop_img_dir):
    os.mkdir(crop_img_dir)

with open(rec_gt_dir, 'w', encoding='utf-8') as f:
    with open(label_file) as fh:
        for line in fh:
            key, annotation = line.split("\t", 1)
            try:
                img_path = os.path.join(current_directory,key)
                img = cv2.imread(img_path)
                for i, label in enumerate(json.loads(annotation.strip())):
                    if label['difficult']:
                        continue
                    img_crop = get_rotate_crop_image(img, np.array(label['points'], np.float32))
                    img_name = os.path.splitext(os.path.basename(img_path))[0] + '_crop_' + str(i) + '.jpg'
                    cv2.imwrite(crop_img_dir + img_name, img_crop)
                    f.write('crop_img/' + img_name + '\t')
                    f.write(label['transcription'] + '\n')
            except Exception as e:
                ques_img.append(key)
                print("Can not read image ", e)