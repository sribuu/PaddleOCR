'''
FIXME:
    Dissect this humongous script into 4 classes:
        > Model (model building, training, prediction, model saving)
        > Trainer (calling Model, instantiate model, train, metric output)
        > Data (Clean , compartmentalise data)
        > HyperParameterOptimisation(Optimisation of hyperparams based on the trainer metric)

        vqa_token_relation.py --> limitation in RE only use 2 labels
'''
import sys
import os
#Start importing modules
import argparse
import subprocess
import re
from tqdm import tqdm
import json

import cv2
import numpy as np
from tools import program, train, export_model
from tools.infer.utility import get_rotate_crop_image

import pandas as pd
from ppocr.utils.utility import set_seed

from gen_ocr_train_val_test import genDetRecTrainVal

#Importing paddle

import paddle.distributed as dist

import time

#Import hyperparameters class
from hyperparameters import HyperParameters    

#Import Bayesian Optimiser
import functools
from bayes_opt import BayesianOptimization as BO

from numba import cuda
# import cupy as cp


dist.get_world_size()
#Get the directory of this file which should be located in tools_custom
__dir__ = os.path.dirname(os.path.abspath(__file__))

#Get the abs directory of the OCR repo
__parent__ = os.path.dirname(__dir__)

#Append it to sys s.t. Python can find the modules
sys.path.insert(0,__parent__)
sys.path.insert(0,__dir__)



class SribuuOCRTrainer(object):
    '''
        Class abstraction for Trainer to train the OCR model
        Params:
            model_dir, str, path to directory where the model we want to train is hosted (invoice, e-statements etc)
    '''
    def __init__(self, model_dir:str, trainResume:str, predict=False, useCPU=True, predictInfer=False,isPrepared=False):
        #Config YML file
        #self.fn_config = fn_config

        #Resume training or nah?
        self.trainResume = trainResume

        #Predict or nah
        self.predict = predict

        #GPU or CPU or just U <3
        self.useCPU = useCPU

        #Predict infer
        self.predicInfer = predictInfer

        #Model dir
        self.model_dir = model_dir

        #avoid regenerate existing data
        self.is_prepared = isPrepared

        self.label_file = os.path.join(self.model_dir,"Label.txt")

        #Create crop_img folder
        if not self.is_prepared:
            self.crop_img_dir = "%s/train_data/ocr_crop/"%(self.model_dir)
            if not os.path.exists(self.crop_img_dir):
                os.makedirs(
                    self.crop_img_dir
                )
            #Construct the pathway of rec_gt.txt
            self.rec_gt_fn = "%s/train_data/ocr_crop.txt"%(self.model_dir)

    
    def unpack_utilities_file(self):
        '''
            Method to unpack the *.yml file which contains absolute path to the utilities file.
            Put the keys, values in dictionary 
        '''
        pass
    
    def reformat_label_list(self):
        #Update label list
        with open("%s/label-key-list.txt"%(self.model_dir), 'r') as file:
            lines = file.readlines()

        result = [line.strip().upper() for line in lines]

        with open("%s/label-key-list.txt"%(self.model_dir), "w") as file:
            file.write("\n".join(result))

    def gen_cropped_img(self):
        ques_img = []
        #Start populating the self.rec_gt.txt
        with open(self.rec_gt_fn, 'w', encoding='utf-8') as f:
            with open(self.label_file) as fh:
                for line in tqdm(fh):
                    key, annotation = line.split("\t", 1)

                    #Read the image contained in img_path
                    try:
                        img_path = os.path.join(self.model_dir,key)
                        img = cv2.imread(img_path)

                        #Read the label
                        for i, label in enumerate(json.loads(annotation.strip())):
                            # if label['difficult']:
                            #     continue
                            img_crop = get_rotate_crop_image(img, np.array(label['points'], np.float32))
                            img_name = os.path.splitext(os.path.basename(img_path))[0] + '_crop_' + str(i) + '.jpg'

                            #Writing the cropped images to self.crop_img_dir
                            cv2.imwrite(self.crop_img_dir + img_name, img_crop)
                            f.write('train_data/ocr_crop/' + img_name + '\t'+label['transcription'].replace('\n','\\n').strip() + '\n')
                    except Exception as e:
                        ques_img.append(key)
                        print("Can not read image ", e)
    
    def split_data(self, train_fraction, validation_fraction, test_fraction):
        self.datasetRootPath = self.model_dir
        self.detLabelFileName = self.label_file
        self.recLabelFileName = self.rec_gt_fn
        self.recImageDirName = self.crop_img_dir
        self.detRootPath = "%s/train_data/kie"%(self.model_dir)
        self.recRootPath = "%s/train_data/ocr"%(self.model_dir) 
        self.overwriteLabelFile = False

        #All added has to be one innit?
        whole = test_fraction + train_fraction + validation_fraction

        try:
            assert(
                whole <= 1
            )
        except Exception:
            raise ValueError(
                "Train fraction + validation fraction + test fraction > 1."
            )
        
        self.trainValTestRatio = "%s:%s:%s"%(
            int(train_fraction*10), int(validation_fraction*10), int(test_fraction*10)
        )

        #Split the data -- for some reason the Chinese parse args instead of object. 
        genDetRecTrainVal(self)

    def read_hyperparameter(self, hyperparams:dict):
        '''
        What are the important hyperparameters that have to be contained in hp??
            1. algorithm: default is 'LayoutXLM'
            2. optimizer_name: defalut is 'AdamW'
            3. loss_name: default is 'VQASerTokenLayoutLMLoss'
            4. beta1: default is 0.9
            5. beta2: default is 0.999
            6. lr_name: default is "Linear"
            7. learning_rate: default is 5e-5
            8. regularizer_name: default is 'L2'
            9. regularizer_factor: default is 0.0

            Discuss with Suzie and Arief what else to add
        '''
        try:
            profiler_options = hyperparams["profiler_options"]
        except KeyError:
            profiler_options = None

        #Instatiate the Hyperparameters class with args taken from a dictionary -- this dictionary contains decision variables for the hyperparameter optimisation
        hp = HyperParameters(
            model_dir = self.model_dir,
            epoch_num = hyperparams["epoch_num"],
            architecture_name = hyperparams['architecture_name'],
            global_model = hyperparams["global_model"],
            use_gpu = not self.useCPU,
            optimizer_name = hyperparams["optimizer_name"],
            loss_name = hyperparams["loss_name"],
            beta1 = hyperparams['beta1'],
            beta2 = hyperparams['beta2'],
            lr_name = hyperparams["lr_name"],
            learning_rate = hyperparams["learning_rate"],
            regularizer_name = hyperparams["regularizer_name"],
            regularizer_factor = hyperparams["regularizer_factor"],
            num_classes = hyperparams['num_classes'],
            profiler_options = profiler_options
        )

        return hp
    
    def export(self,hp):
        '''Method to export the trained model'''
        export_model.main(hp.config)
        
    def fit(
            self, hyperparams:dict, model:str, 
            train_fraction, validation_fraction, test_fraction, #static hyperparams
            beta1, beta2, learning_rate, regularizer_factor #optimised hyperparams
        ):
        '''
        Train the model. This is where the magic happens mate!
            args:
                model, str, what models you want to train. Choices are: "SER","RE","ALL"
                train_fraction, float, fraction of train data
                validation_fraction, float, fraction of validation data
        '''
        #What model do you want to train
        self.model = model
        
        allocate_gpu_memory()  # Alokasikan memori GPU sebelum pelatihan
        
        print("== TRAINING ==")

        if not self.is_prepared:
            # print("== Generating id and linking the dataset label based on the linking file")

            #Unpack utilities
            self.unpack_utilities_file()

            self.reformat_label_list()

            #Generating rec cropped image
            print("== Generate Rec Cropped Img")
            self.gen_cropped_img()

            #Splitting data set
            print("== Splitting dataset")
            self.split_data(train_fraction, validation_fraction, test_fraction)
            self.is_prepared = True

        #Count number of classes
        with open("%s/label-key-list.txt"%(self.model_dir), 'r') as f:
            num_count = len([line for line in f if line.strip()])

        #Updating the number classes
        self.num_classes = (2 * num_count) - 1
        print("== Update num_classes to %s"%(self.num_classes))
    
        #Instantiate hyperparameters dict
        hyperparams["num_classes"] = self.num_classes
        
        #Assign key, values of the optimised hyperparams to the hyperparams dict
        hyperparams["beta1"] = beta1
        hyperparams["beta2"] = beta2
        hyperparams["learning_rate"] = learning_rate
        hyperparams["regularizer_factor"] = regularizer_factor

        '''
        Start the training for a specific model as self.model
        '''
        if self.model == "SER" or self.model == "ALL":
            '''
            So this part is a bit messy. Calling a script full of functions where one of them instantiate an argparse. But hey, when it works, it works. Therefore, I wrap this part of the code with a wrapper to make it more Pythonic (I guess)
            '''
            #Tricky thing is that if user wants to do "ALL" training, then this part of the code is only for SER, that's why I hardcoded SER here
            if self.model == "ALL":
                self.model = "SER"

            print("== Training %s Model"%(self.model))

            #Instantiate hyperparameter object
            hyperparams["global_model"] = self.model
            hp = self.read_hyperparameter(hyperparams)
            
            #Calling program method and traing SER model
            config, device, logger, vdl_writer = program.preprocess(is_train = not self.predict, flags_ = hp)
            seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
            set_seed(seed)

            now = time.time()
            best_metric = train.main(config, device, logger, vdl_writer,seed)

            print(
                "==================================== Training %s model took %.2f seconds. Metric = %s with score %.4f"%(
                    "SER",
                    time.time() - now,
                    hp.config["Metric"]["name"],
                    best_metric
                )
            )

            #Write optimisation variables and result of the ith iteration
            fn_optim = "%s/optimisation_model_%s.csv"%(self.model_dir, self.model)
            with open(
                fn_optim,"a"
            ) as f:
                f.write(
                    "%s,%s,%s,%s,%s\n"%(
                    beta1, beta2, learning_rate, regularizer_factor, best_metric
                )
            )
                
            print(fn_optim)
            
            #Export model            
            df__ = pd.read_csv(
                fn_optim
            )

            past_best_metric = df__["metric"]

            #You actually have done the optimisaiton
            if len(past_best_metric)>0:
                if best_metric >= past_best_metric.max():
                    self.export(hp)

        if self.model == "RE" or self.model == "ALL":
            #Tricky thing is that if user wants to do "ALL" training, then this part of the code is only for SER, that's why I hardcoded SER here
            if self.model == "ALL":
                self.model = "RE"

            #RE model training
            print("== Training %s Model"%(self.model))

            #Instantiate hyperparameter object
            hyperparams["global_model"] = self.model
            hp = self.read_hyperparameter(hyperparams)

            print(
                hp.config["Train"]
            )

            #TESTING YAML
            #fn = "%s/algorithm_re.yml"%(self.model_dir)
            #with open(fn, 'r') as f:
            #    valuesYaml = yaml.load(f, Loader=yaml.FullLoader)
            #
            ##Global
            #for key,val in valuesYaml.items():
            #    try:
            #        tmp__ = hp.config[key]
            #    except:
            #        raise KeyError(
            #            "No %s in hyperparams"%(key)
            #        )
            #    for k,v in val.items():
            #        if key not in ["Train","Eval"]:
            #            try:
            #                print(
            #                    "%s,%s,%s"%(
            #                        k, hp.config[key][k], v
            #                    )
            #                )
            #            except Exception as e:
            #                print(str(e))
            #        
            #t_yml, e_yml = valuesYaml["Train"], valuesYaml["Eval"]
            #t_cfg, e_cfg = hp.config["Train"], hp.config["Eval"]

            #print(
            #    t_yml["dataset"]["name"], t_cfg["dataset"]["name"],"\n",
            #    t_yml["dataset"]["data_dir"], t_cfg["dataset"]["data_dir"],"\n",
            #    t_yml["dataset"]["label_file_list"], t_cfg["dataset"]["label_file_list"],"\n",
            #    t_yml["dataset"]["ratio_list"], t_cfg["dataset"]["ratio_list"],"\n",

            #    e_yml["dataset"]["name"], e_cfg["dataset"]["name"],"\n",
            #    e_yml["dataset"]["data_dir"],e_cfg["dataset"]["data_dir"],"\n",
            #    e_yml["dataset"]["label_file_list"], e_cfg["dataset"]["label_file_list"],"\n"
            #)

            #t_trans_yml, e_trans_yml = valuesYaml["Train"]['dataset']['transforms'], valuesYaml["Eval"]['dataset']['transforms']
            #t_trans_cfg, e_trans_cfg = hp.config["Train"]['dataset']['transforms'], hp.config["Eval"]['dataset']['transforms']   
            #
            #print(
            #    "Transform train yml \n\n\n\n\n",t_trans_yml, "Transform train python \n\n\n\n\n", t_trans_cfg
            #)
            #print(
            #    "Transform eval yml \n\n\n\n\n",e_trans_yml, "Transform eval python \n\n\n\n\n", e_trans_cfg
            #)     

            #Calling program method and train RE model
            config, device, logger, vdl_writer = program.preprocess(is_train = not self.predict, flags_ = hp)
            seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
            set_seed(seed)

            now = time.time()
            best_metric = train.main(config, device, logger, vdl_writer,seed)

            print(
                "==================================== Training %s model took %.2f seconds. Metric = %s with score %.4f"%(
                    "RE",
                    time.time() - now,
                    hp.config["Metric"]["name"],
                    best_metric
                )
            )
                            
            #Write optimisation variables and result of the ith iteration
            fn_optim = "%s/optimisation_model_%s.csv"%(self.model_dir, self.model)
            with open(
                fn_optim,"a"
            ) as f:
                f.write(
                    "%s,%s,%s,%s,%s\n"%(
                    beta1, beta2, learning_rate, regularizer_factor, best_metric
                )
            )                

            #print(fn_optim)

            #Export model by first reading the hyperparameter optimisation log, and see if the current metric is better than the ones recorded. If that's the case, then save the model.          
            df__ = pd.read_csv(
                fn_optim
            )

            past_best_metric = df__["metric"]

            #You actually have done the optimisaiton
            if len(past_best_metric)>0:
                if best_metric >= past_best_metric.max():
                    self.export(hp)

        #Return best_metric for the optimisation's objective function
        deallocate_gpu_memory()  # Dealokasikan memori GPU setelah pelatihan

        return best_metric    

def create_log_optimisation(model_dir, model):
    #File to store hyperparameter optimisation
    with open(
        "%s/optimisation_model_%s.csv"%(model_dir, model),"w"
    ) as f:
        f.write(
            "beta1,beta2,learning_rate,regularizer_factor,metric\n"
        )

def allocate_gpu_memory():
    print("Allocating Memory do Nothing...")
  

def deallocate_gpu_memory():
    # Setelah membersihkan cache
    print("Deallocate Memory do Nothing...")

def free_GPU():
    #Free-ing GPU resources
    print("Releasing GPU resources.......\n\n")
    #Method to kill all processes running on GPU and free GPU resources
    cuda.select_device(0)
    cuda.close()
    print("Done releasing GPU.........")

def predict(
        absolute_path_script_folder, 
        model_dir,
        sdk_path_dir,
        predict_file,
        predict_file_output,
        use_gpu=True
    ):
    '''
    Still using subprocess because dissecting the PPOCR-native predict script risks changing the package :(
        path_to_predict_folder: path to the folder in which predict_kie_token_ser_re.py and predict_kie_token_ser.py are hosted
        model_dir: path to the directory where the trained models are hosted
    '''
    #Text formatting
    predict_script = "%s/predict_kie_token_ser_re.py"%absolute_path_script_folder
    predict_ser_script = "%s/predict_kie_token_ser.py"%absolute_path_script_folder
    model_compiled_re = "%s/model_compiled/re"%(model_dir)
    model_compiled_ser = "%s/model_compiled/ser"%(model_dir)
    visual_font = f"{sdk_path_dir}tools_custom/fonts/simfang.ttf"
    
    output_predict_dir = os.path.join(model_dir,"output",os.path.basename(predict_file if (predict_file_output is None) else predict_file_output))

    #If both SER and RE model are found -- predict using both
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
            "--ser_dict_path=%s/label-key-list.txt"%(model_dir),
            f"--vis_font_path={visual_font}",
            "--ocr_order_method=tb-yx",
            f"--use_gpu={use_gpu}"
        ])
    #If not: SER prediction only
    else:
        subprocess.run([
            "python3",
            predict_ser_script,
            "--kie_algorithm=LayoutXLM",
            f"--ser_model_dir={model_compiled_ser}",
            "--use_visual_backbone=False",
            f"--image_dir={predict_file}",
            f"--output={output_predict_dir}",
            "--ser_dict_path=%s/label-key-list.txt"%(model_dir),
            f"--vis_font_path={visual_font}",
            "--ocr_order_method=tb-yx",
            f"--use_gpu={use_gpu}"
        ])

def predict_checkpoint():
    #What does it do? Why predict using checkpoint model, not waiting until we finish training? For sure we won't use the "partly-trained" model
    pass

def data_downloader(s3,file_name_in_bucket,file_name_local,force_download=False):
    bucket_name = 'business-ocr-image'

    parent_path = os.path.dirname(file_name_local)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    # Memeriksa apakah file sudah ada di lokal sebelum mendownload
    if not os.path.exists(file_name_local) or force_download:
        # Mendownload file dari bucket S3
        try:
            s3.download_file(bucket_name, file_name_in_bucket, file_name_local)
            return file_name_local
        except Exception as e:
            print(f"Gagal mendownload {file_name_in_bucket}. Terjadi kesalahan: {e}")  

def fetch_dataset(model_dir,model_id):
    import boto3
    #Import s3 bucket
    from dotenv import load_dotenv

    #Fetch dataset from S3 bucket based on list of data in Label.txt for selected model_id
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')  # Ganti dengan path sesuai lokasi file .env

    # Memuat variabel dari file .env sesuai dengan path yang ditentukan
    load_dotenv(dotenv_path=dotenv_path)

    # Menggunakan os.getenv untuk mengambil nilai variabel dari file .env
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY')
    aws_secret_access_key = os.environ.get('AWS_SECRET_KEY')

    # Membuat koneksi ke bucket S3 menggunakan kredensial
    s3 = boto3.client('s3',
                    region_name='ap-southeast-3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key)
    label_file_bucket = f"{model_id}/Label.txt"
    label_file_local = f"{model_dir}{model_id}/Label.txt"
    
    label_file = data_downloader(s3=s3,file_name_in_bucket=label_file_bucket,file_name_local=label_file_local,force_download=True)

    label_key_file_bucket = f"{model_id}/label-key-list.txt"
    label_key_file_local = f"{model_dir}{model_id}/label-key-list.txt"
    data_downloader(s3=s3,file_name_in_bucket=label_key_file_bucket,file_name_local=label_key_file_local,force_download=True)
    
    label_dict = {}
    with open(label_file) as f:
        for line in f:
            #Read each line and turn into key value pair dict    
            splitted = line.split("\t", 1)

            if(len(splitted)<2):
                splitted = line.split("g [",1)
                filename_bucket = f"{splitted[0]}g"
                label_dict[filename_bucket] = f"[{splitted[1]}"
            else:
                filename_bucket = splitted[0]
                label_dict[filename_bucket] = splitted[1]
            
            filename_local = f"{model_dir}{filename_bucket}"
            data_downloader(s3=s3,file_name_in_bucket=filename_bucket,file_name_local=filename_local)
    
    text = "\n".join([f"{key}\t{value}" for key, value in label_dict.items()])

    print(f"Dataset total to be trained: {len(label_dict.items())}")

    # write the text to a file
    with open(label_file_local, "w") as f:
        f.write(text+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action='store_const', const=True, default=False,help="Use this to do prediction")
    parser.add_argument("--train", action='store_const', const=True, default=False,help="Use this to do training")
    parser.add_argument("--mode",choices=["ALL","SER","RE"],default="SER", help="ALL=SER+RE ,SER, RE")
    parser.add_argument("--predict_file", type=str, help="Absolute path for Predict File")
    parser.add_argument("--predict_file_output", type=str, help="Absolute path for Output Predict File")
    parser.add_argument("--sdk_path", type=str, default="/home/ubuntu", help="SDK Path")
    parser.add_argument("--model_id", type=str,required=True,  help="Model id for training or prediction\n\n"+
                        """
                            # INVOICE : 7a81e532-af43-4e8c-af67-dcdedb778e96
                            # RECEIPT : a0c1e53d-5bec-4e0d-aaee-71b28936181a
                            # ESTATEMENT : 509b4e5d-d470-4eec-bbdf-59daf50af631
                            # PURCHASE_ORDER : 20bb2d54-661f-440d-9dc8-80b1ed743435
                            # DELIVERY_NOTES : e9603dc4-f03a-40de-849b-4ce18273fbfe
                            # BKPB : f9e974af-81aa-4e2a-a6d4-8f1583cc7f9a
                        """)
    parser.add_argument("--use_cpu", type=bool, default=False, help="Enable to use cpu instead gpu")
    parser.add_argument("--skip_preparation", action='store_const', const=True, default=False,help="Use existing data instead of generate new set")
    parser.add_argument("--annotation_file", type=str,help="Annotation File E.g Label.txt")

    args = parser.parse_args()

    sdk_path = args.sdk_path

    script_path = os.path.join(sdk_path,"ocr-engine/paddle-ocr/PaddleOCR/ppstructure/kie")
    #PaddleFile only for Local
    model_dir_path = os.path.join(sdk_path,"models")
    useCPU = args.use_cpu

    if args.predict:
        predict(absolute_path_script_folder=script_path,
                model_dir= os.path.join(model_dir_path,args.model_id),
                sdk_path_dir= os.path.join(sdk_path,"PaddleOCR"),
                predict_file=args.predict_file,
                predict_file_output=args.predict_file_output,
                use_gpu=not useCPU,)

    else: #train
        try:

            model_id = args.model_id #"509b4e5d-d470-4eec-bbdf-59daf50af631"
            model_dir_only = model_dir_path #"/Users/ariefwijaya/Documents/ARIEFW/Project/PaddleFile/models/"
            model_dir = os.path.join(model_dir_only,model_id)

            train_fraction = 0.8
            validation_fraction = 0.2
            test_fraction = 0.0

            #What model do you train?ALL
            model = args.mode

            #Disable this temporary to avoid replace existing dataset
            # fetch_dataset(model_dir=model_dir_only,model_id=model_id)

            trainer = SribuuOCRTrainer(
                model_dir = model_dir,
                trainResume = None,
                useCPU = useCPU,
                isPrepared=args.skip_preparation
            )

            if args.annotation_file:
                trainer.label_file = os.path.join(trainer.model_dir,args.annotation_file)
# 
            
            # only to split dataset
            # ============
            # trainer.reformat_label_list()
            # #Splitting data set
            # print("== Splitting dataset")
            # trainer.split_data(train_fraction, validation_fraction, test_fraction)
            # exit( "Prepare only")
            # # =================

            if model == "ALL":
                create_log_optimisation(
                    model_dir = model_dir,
                    model = "SER"
                )

                create_log_optimisation(
                    model_dir = model_dir,
                    model = "RE"
                )
            else:
                create_log_optimisation(
                    model_dir = model_dir,
                    model = model
                )



            #Instantiate dictionary that contains hyperparameters
            hyperparams = {}

            '''
            Differentiate static hyperparams e.g. epoch from the optimised params
                static: epoch_num, algorithm, optimizer_name, loss_name, regularize_name

                optimised: beat1, beta2, learning_rate, regularizer_factor
            '''
            hyperparams["epoch_num"] = 200
            hyperparams["algorithm"] = "LayoutXLM"
            hyperparams["optimizer_name"] = "AdamW"

            if model != "RE":
                hyperparams["loss_name"] = "VQASerTokenLayoutLMLoss"
                hyperparams['architecture_name'] = "LayoutXLMForSer"
            else:
                hyperparams["loss_name"] = "LossFromOutput"
                hyperparams['architecture_name'] = "LayoutXLMForRe"

            hyperparams["regularizer_name"] = "L2"
            hyperparams["lr_name"] = "Linear"

            #Instantiate partial obj function, where hyperparams is left off for optimisation routine
            objfunc = functools.partial(
                trainer.fit, hyperparams = hyperparams, model=model, 
                train_fraction=train_fraction, validation_fraction=validation_fraction, test_fraction=test_fraction
            )

            #Instantiate the hyperparameters bound
            parameterbounds = {
                'beta1':(0.8,0.95),
                'beta2':(0.9,0.95),
                'learning_rate':(5e-7,5e-4),
                'regularizer_factor':(0,0.499)
            }

            #Instantiate the optimisation object
            opt = BO(
                f=objfunc,
                pbounds=parameterbounds,
                verbose=2,
                random_state=42 #should be 42
            )

            #Start bayesian optimiser
            start_time = time.time()
            
            #should be [opt.maximize(init_points=5,n_iter=25)]
            #We limit to 3 iteration,1+2
            # opt.maximize(init_points=1,n_iter=2)
            opt.maximize(init_points=5,n_iter=25)
            
            delta = time.time() - start_time

            print("Total time for bayesian optimisation: %s s"%delta)

            print("Output best metric = %s"%(opt.max))
            
            # if not useCPU:
            #     free_GPU()
        except Exception as error:
            print("An exception occurred:", error) 
            # if not useCPU:
            #     free_GPU()
