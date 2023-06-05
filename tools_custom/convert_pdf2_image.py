import os
import PIL
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import argparse

# Membuat objek parser
parser = argparse.ArgumentParser()

# Menambahkan argumen yang diharapkan
parser.add_argument("--sourceDir", type=str, required=True, help="Input directory which contain .pdf files to be converted")
parser.add_argument("--destDir", type=str, required=True, help="Destination directory for converted files")

# Parse argumen dari terminal
args = parser.parse_args()

# Path to folder containing PDF files
pdf_folder = args.sourceDir

# Path to output folder for PNG images
png_folder = args.destDir

PIL.Image.MAX_IMAGE_PIXELS = None
# Initialize PaddleOCR
ocr = PaddleOCR(lang='id',max_batch_size = 20,
    total_process_num = os.cpu_count() * 2 - 1)

# Loop through all PDF files in folder
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith('.pdf'):
        # Convert PDF to list of image pages using pdf2image
        pages = convert_from_path(os.path.join(pdf_folder, pdf_file), 500)
        
        # Loop through all image pages and process with PaddleOCR
        for i, page in enumerate(pages):
            # Convert image page to PNG format and save to output folder
            png_file = f'{pdf_file[:-4]}_{i}.png'

            print(f'Start extraction of {png_file}:\n')

            page.save(os.path.join(png_folder, png_file), 'PNG')
            
            # Process image page with PaddleOCR and extract text
            result = ocr.ocr(os.path.join(png_folder, png_file))
            
            text = result
            # text = '\n'.join([line[1][0] for line in result])
            
            # Print extracted text
            print(f'Text extracted from {png_file}:\n{text}\n')
