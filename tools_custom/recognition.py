from paddleocr import PaddleOCR
import os
import cv2
import numpy as np
import json
from tools.infer.utility import get_rotate_crop_image
import argparse
import ast


# Membuat objek ArgumentParser
parser = argparse.ArgumentParser()

parser.add_argument('--shapePoints', type=ast.literal_eval, help='Shape Coordinates')
parser.add_argument('--filePath', type=str, help='Full File Path')

# Mem-parsing argumen dari baris perintah
args = parser.parse_args()

ocrService = PaddleOCR(use_pdserving=False,
                             use_angle_cls=True,
                             det=True,
                             cls=True,
                             use_gpu=False,
                             lang="en",
                             show_log=False)

class CustomError(Exception):
    def __init__(self, message):
        self.message = message

def singleRerecognition(filePath,shapePoints):
        img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),1)
        box = [[int(p[0]), int(p[1])] for p in shapePoints]
        img_crop = get_rotate_crop_image(img, np.array(box, np.float32))
        if img_crop is None:
            return[None,0]
        result = ocrService.ocr(img_crop, cls=True, det=False)[0]
        if result[0][0] != '':
            return result[0]
        else:
            return[None,0]

res = singleRerecognition(args.filePath,args.shapePoints)
if res[0] is not None:
    print(f"[\"{res[0]}\",{res[1]}]")
else:
    print(f"[null]")
