#!/bin/bash
usage="$(basename "$0") [[--help]] [[--train "ALL"]] [[--trainResume "SER"]] [[--predict "file_path.png"]] [[--predictInfer "file_path.png"]] [[--config "path.yml"]] [[--useCPU]] Script to simplify OCR Engine

where:
    --config ocr engine config file.yml
    --help show this help text
    --train Start Training. [SER, RE] or ALL to start training both sequentially
    --trainResume Resume Training. [SER, RE]
    --predict Start Prediction. 
    --useCPU Use this param to disable GPU and use CPU Instead. 
    --predictInfer Start Prediction using Raw/Checkpoint Model." 

# Get required path and files
file="ocr-engine-config.yml"
train="ALL"
train_resume=""
predict="false"
predict_infer="false"
predict_file=""
use_gpu="True"

# Parse the options
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --config) file="$2" shift ;;
    --train) train="$2" predict="false" train_resume="";;
    --useCPU) use_gpu="False";;
    --trainResume) train_resume="$2" train="" predict="false";;
    --predict) predict="true" train=""  train_resume="" predict_file="$2";;
    --predictInfer) predict="true" train="" predict_infer="true" train_resume="" predict_file="$2";;
    --help) echo "$usage"
          exit 0 ;;
    \?)  echo "Invalid option: $key" >&2; exit 1;;
  esac
  shift
done

# echo "== Prepare script configuration from $file"

linking_gen_script=""
dataset_divider_script=""
trainer_script=""
predict_script=""
predict_ser_script=""
infer_script=""
infer_ser_script=""
linking_file=""
export_model_script=""
rec_gt_gen_script=""
visual_font=""

# Membaca file YAML dan mengambil key dan value
while IFS=: read -r key value; do
  # Menghapus whitespace pada key dan value
  key=$(echo $key | tr -d ' ')
  value=$(echo $value | tr -d ' ')

  if [ "$key" == "linking_gen_script" ]; then
    linking_gen_script=$value
  elif [ "$key" == "dataset_divider_script" ]; then
    dataset_divider_script=$value
  elif [ "$key" == "trainer_script" ]; then
    trainer_script=$value
  elif [ "$key" == "predict_script" ]; then
    predict_script=$value
  elif [ "$key" == "predict_ser_script" ]; then
    predict_ser_script=$value
  elif [ "$key" == "infer_script" ]; then
    infer_script=$value
  elif [ "$key" == "infer_ser_script" ]; then
    infer_ser_script=$value
  elif [ "$key" == "linking_file" ]; then
    linking_file=$alue
  elif [ "$key" == "export_model_script" ]; then
    export_model_script=$value
  elif [ "$key" == "algorithm_ser" ]; then
    algorithm_ser=$value
  elif [ "$key" == "algorithm_re" ]; then
    algorithm_re=$value
  elif [ "$key" == "rec_gt_gen_script" ]; then
    rec_gt_gen_script=$value
  elif [ "$key" == "visual_font" ]; then
    visual_font=$value
  # else
  #   echo "String 1 tidak sama dengan String 2"
  fi
done < "$file"


# Prepare anything and do training
if [[ -n "$train" ]] 
then
    echo "== TRAINING =="
    # echo "== Navigating to directory $(dirname "$0")"

    # Navigate to current model id director
    # cd "$(dirname "$0")"
    
    echo "== Generating id and linking for dataset label based on linking_file"
    # Generate id and linking for dataset label based on linking_file
    python3 "$linking_gen_script" --linkingFile "label-key-list-pair.txt" --labelFile "Label.txt" --labelOutputFile "Label-linked.txt"
    
    echo "== Generate Rec Cropped Img"
    # Generate Rec Cropped Img
    python3 "$rec_gt_gen_script" --outputFileGT "rec_gt.txt" --labelFile "Label.txt" --outputFileDir "crop_img/"

    echo "== Splitting dataset with ratio 6:2:2"
    # Split dataset with ratio 6:2:2
    python3 "$dataset_divider_script" --trainValTestRatio 6:2:2 --datasetRootPath "" --detLabelFileName "Label-linked.txt" --recLabelFileName "rec_gt.txt" --recImageDirName "crop_img" --detRootPath "./train_data/det" --recRootPath "./train_data/rec"
    
    #Count num classes
    num_count=$(grep -vc '^$' "label-key-list.txt")
    num_classes=$(((2 * $num_count) - 1))

    echo "== Update num_classes to $num_classes"

    old_value="\&num_classes"
    new_value="\&num_classes $num_classes"
    # Update num_classes
    sed -i '' -E "s/$old_value.*/$new_value/g" "$algorithm_ser"

    echo "== Training SER Model"

    if [[ $train = "SER" || $train = "ALL" ]]
    then
        # Training SER Model
        python3 "$trainer_script" \
            -c "$algorithm_ser" \
            -o Global.save_model_dir=./model_checkpoint/ser/ \
            Global.use_gpu=$use_gpu 

        echo "== Export SER Model to Inference"
        # Export SER Model to Inference
        python3 "$export_model_script" -c "$algorithm_ser" -o Architecture.Backbone.checkpoints=./model_checkpoint/ser/best_accuracy Global.use_gpu=$use_gpu 
    fi

    if [[ $train = "RE" || $train = "ALL" ]]
    then
        echo "== Training RE Model"
        # Training RE Model
        python3 "$trainer_script" \
            -c "$algorithm_re" \
            -o Global.save_model_dir=./model_checkpoint/re/ \
            Global.use_gpu=$use_gpu 

        echo "== Export RE Model to Inference"
        # Export RE Model to Inference
        python3 "$export_model_script" -c "$algorithm_re" -o Architecture.Backbone.checkpoints=./model_checkpoint/re/best_accuracy Global.use_gpu=$use_gpu 
    fi
elif [[ $predict = "true" ]]
then
    if [ -z "$predict_file" ]; then
        echo "Error: The file to predict is required"
        exit 1
    fi

    # Check compiled model and decie to use infer or predict, 
    if [ $predict_infer = "false" ]; then
        if [ -d "./model_compiled/re" ] && [ -d "./model_compiled/ser" ]; then
          python3 "$predict_script" \
              --kie_algorithm=LayoutXLM \
              --re_model_dir=./model_compiled/re \
              --ser_model_dir=./model_compiled/ser \
              --use_visual_backbone=False \
              --image_dir="$predict_file" \
              --ser_dict_path="label-key-list.txt" \
              --vis_font_path="visual_font" \
              --ocr_order_method="tb-yx"  
        else
          python3 "$predict_ser_script" \
            --kie_algorithm=LayoutXLM \
            --ser_model_dir=./model_compiled/ser \
            --use_visual_backbone=False \
            --image_dir="$predict_file" \
            --ser_dict_path="label-key-list.txt" \
            --vis_font_path="visual_font" \
            --ocr_order_method="tb-yx"  
        fi
    else
        if [ -d "./model_checkpoint/re/best_accuracy" ] && [ -d "./model_checkpoint/ser/best_accuracy" ]; then
            python3 "$infer_script" \
            -c algorithm_re.yml \
            -o Architecture.Backbone.checkpoints=./model_checkpoint/re/best_accuracy/ \
            Global.infer_img="$predict_file" \
            Global.infer_mode=True \
            Global.use_gpu=$use_gpu \
            -c_ser algorithm_ser.yml \
            -o_ser Architecture.Backbone.checkpoints=./model_checkpoint/ser/best_accuracy/
        else
            python3 "$infer_ser_script" \
            -c algorithm_ser.yml \
            -o Architecture.Backbone.checkpoints=./model_checkpoint/ser/best_accuracy/ \
            Global.infer_img="$predict_file" \
            Global.infer_mode=True \
            Global.use_gpu=$use_gpu 
        fi
    fi
elif [[ -n "$train_resume" ]]
then 
    if [[ $train_resume = "SER" ]]
    then
      python3 "$trainer_script" -c algorithm_ser.yml -o Architecture.Backbone.checkpoints=./model_checkpoint/ser/best_accuracy  Global.use_gpu=$use_gpu
    elif [[ $train_resume = "RE" ]]
    then
      python3 "$trainer_script" -c algorithm_re.yml -o Architecture.Backbone.checkpoints=./model_checkpoint/re/best_accuracy  Global.use_gpu=$use_gpu
    else
      echo "Error: Wrong argument, Choose SER or RE."
      exit 1
    fi
else
    echo "Error: Argument is missing."
    exit 1 
fi
