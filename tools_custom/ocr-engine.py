import argparse
import os
import subprocess
import re
import sys

# Create argument parser
parser = argparse.ArgumentParser(description="Script to simplify OCR Engine")
parser.add_argument("--config", help="OCR engine config file.yml")
parser.add_argument("--train", choices=["SER", "RE", "ALL"], help="Start Training. [SER, RE, ALL] or ALL to start training both sequentially")
parser.add_argument("--trainResume", choices=["SER", "RE"], help="Resume Training. [SER, RE]. If you have any changes on dataset, please retraining instead.")
parser.add_argument("--predict", help="Start Prediction.")
parser.add_argument("--useCPU", action="store_true", help="Use this param to disable GPU and use CPU Instead.")
parser.add_argument("--predictInfer", help="Start Prediction using Raw/Checkpoint Model.")
args = parser.parse_args()

file = args.config if args.config else "ocr-engine-config.yml"
train = args.train
train_resume = args.trainResume
predict = "true" if args.predict or args.predictInfer else None
predict_infer = args.predictInfer if args.predictInfer else "false"
predict_file = args.predict if args.predict else args.predictInfer
use_gpu = not args.useCPU

# Get required path and files
linking_gen_script = ""
dataset_divider_script = ""
trainer_script = ""
predict_script = ""
predict_ser_script = ""
infer_script = ""
infer_ser_script = ""
linking_file = ""
export_model_script = ""
rec_gt_gen_script = ""
visual_font = ""
algorithm_ser = "algorithm_ser.yml"
algorithm_re = "algorithm_re.yml"

model_compiled_re = "./model_compiled/re"
model_compiled_ser = "./model_compiled/ser"
model_checkpoint_re = "./model_checkpoint/re/best_accuracy"
model_checkpoint_ser = "./model_checkpoint/ser/best_accuracy"

# Read the YAML file and retrieve the key-value pairs
with open(file, 'r') as f:
    lines = f.readlines()

for line in lines:
    key, value = map(str.strip, line.split(':', 1))
    if key == "linking_gen_script":
        linking_gen_script = value
    elif key == "dataset_divider_script":
        dataset_divider_script = value
    elif key == "trainer_script":
        trainer_script = value
    elif key == "predict_script":
        predict_script = value
    elif key == "predict_ser_script":
        predict_ser_script = value
    elif key == "infer_script":
        infer_script = value
    elif key == "infer_ser_script":
        infer_ser_script = value
    elif key == "linking_file":
        linking_file = value
    elif key == "export_model_script":
        export_model_script = value
    # elif key == "algorithm_ser":
    #     algorithm_ser = value
    # elif key == "algorithm_re":
    #     algorithm_re = value
    elif key == "rec_gt_gen_script":
        rec_gt_gen_script = value
    elif key == "visual_font":
        visual_font = value

try:
    # Prepare anything and do training
    if train:
        print("== TRAINING ==")

        #Update label list
        with open("label-key-list.txt", "r") as file:
            lines = file.readlines()

        result = [line.strip().upper() for line in lines]

        with open("label-key-list.txt", "w") as file:
            file.write("\n".join(result))

        print(f"== Generating id and linking for dataset label based on {linking_file}")
        subprocess.run(["python3", linking_gen_script, "--linkingFile", "label-key-list-pair.txt", "--labelFile", "Label.txt", "--labelOutputFile", "Label-linked.txt"])

        print("== Generate Rec Cropped Img")
        subprocess.run(["python3", rec_gt_gen_script, "--outputFileGT", "rec_gt.txt", "--labelFile", "Label-linked.txt", "--outputFileDir", "crop_img/"])

        print("== Splitting dataset with ratio 6:2:2")
        subprocess.run(["python3", dataset_divider_script, "--trainValTestRatio", "6:2:2", "--datasetRootPath", "", "--detLabelFileName", "Label-linked.txt", "--recLabelFileName", "rec_gt.txt", "--recImageDirName", "crop_img", "--detRootPath", "./train_data/det", "--recRootPath", "./train_data/rec"])

        # Update Entity labels for RE
        data_dict = {}
        num_count = 0

        with open("label-key-list.txt", 'r') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                line = line.strip()
                if line:
                    data_dict[line] = index
                    num_count += 1

        num_classes = (2 * num_count) - 1
        print(f"== Update num_classes to {num_classes}")

        with open(algorithm_ser, 'r+') as f:
            content = f.read()
            new_content = re.sub(r'&num_classes.*', f'&num_classes {num_classes}', content)
            f.seek(0)
            f.write(new_content)
            f.truncate()
        
        with open(algorithm_re, 'r+') as f:
            content = f.read()
            new_content = re.sub(r'&entities_labels.*', f'&entities_labels {data_dict}', content)
            f.seek(0)
            f.write(new_content)
            f.truncate()

            

        if train == "SER" or train == "ALL":
            print("== Training SER Model")
            try:
                subprocess.run(["python3", trainer_script, "-c", algorithm_ser, "-o", f"Global.use_gpu={use_gpu}"], check=True)
            except subprocess.CalledProcessError as e:
                print("Error Training SER Model:", e)
                exit(1)

            print("== Export SER Model to Inference")
            subprocess.run(["python3", export_model_script, "-c", algorithm_ser, "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_ser}", f"Global.use_gpu={use_gpu}"])

        if train == "RE" or train == "ALL":
            print("== Training RE Model")
            try:
                subprocess.run(["python3", trainer_script, "-c", algorithm_re, "-o", f"Global.use_gpu={use_gpu}"], check=True)
            except subprocess.CalledProcessError as e:
                print("Error Training RE Model:", e)
                exit(1)

            print("== Export RE Model to Inference")
            subprocess.run(["python3", export_model_script, "-c", algorithm_re, "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_re}", f"Global.use_gpu={use_gpu}"])

    elif predict == "true":
        if not predict_file:
            print("Error: The file to predict is required")
            exit(1)
        
        output_predict_dir = os.path.join("output",os.path.basename(predict_file))

        if predict_infer == "false":
            if os.path.isdir(model_compiled_re) and os.path.isdir(model_compiled_ser):
                subprocess.run([
                    "python3",
                    predict_script,
                    "--kie_algorithm=LayoutXLM",
                    f"--re_model_dir={model_compiled_re}",
                    f"--ser_model_dir={model_compiled_ser}",
                    "--use_visual_backbone=False",
                    f"--image_dir={predict_file}",
                    f"--output={output_predict_dir}",
                    "--ser_dict_path=label-key-list.txt",
                    f"--vis_font_path={visual_font}",
                    "--ocr_order_method=tb-yx"
                ])
            else:
                subprocess.run([
                    "python3",
                    predict_ser_script,
                    "--kie_algorithm=LayoutXLM",
                    f"--ser_model_dir={model_compiled_ser}",
                    "--use_visual_backbone=False",
                    f"--image_dir={predict_file}",
                    f"--output={output_predict_dir}",
                    "--ser_dict_path=label-key-list.txt",
                    f"--vis_font_path={visual_font}",
                    "--ocr_order_method=tb-yx"
                ])
        else:
            if os.path.isdir(model_checkpoint_re) and os.path.isdir(model_checkpoint_ser):
                subprocess.run([
                    "python3",
                    infer_script,
                    "-c", algorithm_re,
                    "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_re}",
                    f"Global.infer_img={predict_file}",
                    "Global.infer_mode=True",
                    f"Global.use_gpu={use_gpu}",
                    "-c_ser", algorithm_ser,
                    "-o_ser", f"Architecture.Backbone.checkpoints={model_checkpoint_ser}"
                ])
            else:
                subprocess.run([
                    "python3",
                    infer_ser_script,
                    "-c", algorithm_ser,
                    "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_ser}",
                    f"Global.infer_img={predict_file}",
                    "Global.infer_mode=True",
                    f"Global.use_gpu={use_gpu}"
                ])
    elif train_resume:
        if train_resume == "SER":
            print("== Training SER Model")
            subprocess.run(["python3", trainer_script, "-c", algorithm_ser, "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_ser}", f"Global.use_gpu={use_gpu}"])
        elif train_resume == "RE":
            print("== Training RE Model")
            subprocess.run(["python3", trainer_script, "-c", algorithm_re, "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_re}", f"Global.use_gpu={use_gpu}"])
        else:
            print("Error: Wrong argument, Choose SER or RE.")
            exit(1)
    else:
        print("Error: Argument is missing.")
        exit(1)
except subprocess.CalledProcessError as e:
    # Menangani kesalahan yang terjadi pada subprocess
    print("Error executing subprocess:", e)
    sys.exit(1)
    