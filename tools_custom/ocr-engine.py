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

#Get the directory of this file which should be located in tools_custom
__dir__ = os.path.dirname(os.path.abspath(__file__))

#Get the abs directory of the OCR repo
__parent__ = os.path.dirname(__dir__)

#Append it to sys s.t. Python can find the modules
sys.path.insert(0,__parent__)
sys.path.insert(0,__dir__)

#Start importing modules
import argparse
import subprocess
import re
import json

from paddleocr import PaddleOCR, PPStructure
import cv2
import numpy as np
from tools import program, train, export_model
from tools.infer.utility import get_rotate_crop_image
import argparse

import pandas as pd

from gen_ocr_train_val_test import *

#Importing paddle
import yaml
import paddle
import paddle.distributed as dist

from ppocr.data import build_dataloader
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
from ppocr.modeling.architectures import apply_to_static
from ppocr.modeling.architectures import build_model

import time

dist.get_world_size()

#Import hyperparameters class
from hyperparameters import HyperParameters    

#Import Bayesian Optimiser
import functools
from bayes_opt import BayesianOptimization as BO

from numba import cuda

class SribuuOCRTrainer(object):
    '''
        Class abstraction for Trainer to train the OCR model
        Params:
            model_dir, str, path to directory where the model we want to train is hosted (invoice, e-statements etc)
    '''
    def __init__(self, model_dir:str, trainResume:str, predict=False, useCPU=True, predictInfer=False):
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
    
    def unpack_utilities_file(self):
        '''
            Method to unpack the *.yml file which contains absolute path to the utilities file.
            Put the keys, values in dictionary 
        '''
        pass

    def link_file(self):
        #Join the self.model_dir with *.txt
        self.linking_file = "%s/label-key-list-pair.txt"%(self.model_dir)
        self.label_file = "%s/Label.txt"%(self.model_dir)

        #Instantiate dictionary and lists
        self.linking_dict = {}
        self.linking_links = []
        
        '''
        Read the linked files, store it as a dict
        '''
        with open(self.linking_file, "r") as file:
            #Read self.linking_file
            file_contents = file.read()

            #Store the link into key_linking list
            self.linking_links = eval(file_contents)  
        
        # creating dictionary
        with open(self.label_file) as f:
            for line in f:
                #Read each line and turn into key value pair dict    
                command, description = line.split("\t", 1)
    
                self.linking_dict[command] = description.strip()
    
                for key, values in self.linking_dict.items():
                    temp = json.loads(values)  # temporary store array values from dict
    
                    for i in range(len(temp)):
                        temp_data = temp[i]
                        _id = i + 1  # starting from 1
                        temp_data["id"] = _id  # assign id
                        temp_data["linking"] = []  # default value is empty list

                        if "key_cls" in temp_data: #rename key_cls to label if needed
                            temp_data["label"] = temp_data.pop("key_cls")
                        else:
                            temp_data["label"] = (temp_data["label"])
                                                
                        if temp_data["label"] == "None":
                            temp_data["label"] = "IGNORE"

                        temp[i] = temp_data  # modify current data on temp
    
                    self.linking_dict[key] = json.dumps(temp)  # putting back into _dict

        # convert dictionary into text separated by new line
        text = "\n".join([f"{key}\t{value}" for key, value in self.linking_dict.items()])

        # write the text to a file
        with open("%s/Label-linked.txt"%(self.model_dir), "w") as f:
            f.write(text+"\n")

    def gen_cropped_img(self):
        #Create crop_img folder
        self.crop_img_dir = "%s/crop_img/"%(self.model_dir)
        if not os.path.exists(self.crop_img_dir):
            os.mkdir(
                self.crop_img_dir
            )

        #Construct the pathway of rec_gt.txt
        self.rec_gt_fn = "%s/rec_gt.txt"%(self.model_dir)

        #Start populating the self.rec_gt.txt
        with open(self.rec_gt_fn, 'w', encoding='utf-8') as f:
            with open(self.label_file) as fh:
                for line in fh:
                    key, annotation = line.split("\t", 1)

                    #Read the image contained in img_path
                    try:
                        img_path = os.path.join(self.model_dir,key)
                        img = cv2.imread(img_path)

                        #Read the label
                        for i, label in enumerate(json.loads(annotation.strip())):
                            if label['difficult']:
                                continue
                            img_crop = get_rotate_crop_image(img, np.array(label['points'], np.float32))
                            img_name = os.path.splitext(os.path.basename(img_path))[0] + '_crop_' + str(i) + '.jpg'

                            #Writing the cropped images to self.crop_img_dir
                            cv2.imwrite(self.crop_img_dir + img_name, img_crop)
                            f.write('crop_img/' + img_name + '\t')
                            f.write(label['transcription'] + '\n')
                    except Exception as e:
                        ques_img.append(key)
                        print("Can not read image ", e)
    
    def split_data(self, train_fraction, validation_fraction, test_fraction):
        self.datasetRootPath = ""
        self.detLabelFileName = "%s/Label-linked.txt"%(self.model_dir)
        self.recLabelFileName = self.rec_gt_fn
        self.recImageDirName = self.crop_img_dir
        self.detRootPath = "%s/train_data/det"%(self.model_dir)
        self.recRootPath = "%s/train_data/rec"%(self.model_dir) 

        #All added has to be one innit?
        whole = test_fraction + train_fraction + validation_fraction

        try:
            assert(
                whole <= 1
            )
        except Exception as e:
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

        print("== TRAINING ==")
        print("== Generating id and linking the dataset label based on the linking file")

        #Unpack utilities
        self.unpack_utilities_file()

        #Linking files
        self.link_file()

        #Generating rec cropped image
        print("== Generate Rec Cropped Img")
        self.gen_cropped_img()

        #Splitting data set
        print("== Splitting dataset")
        self.split_data(train_fraction, validation_fraction, test_fraction)

        #Count number of classes
        with open("%s/label-key-list.txt"%(self.model_dir), 'r') as f:
            num_count = len([line for line in f if line.strip()])

        num_classes = (2 * num_count) - 1

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
            best_metric = train.main(config, device, logger, vdl_writer)

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
            fn = "%s/algorithm_re.yml"%(self.model_dir)
            with open(fn, 'r') as f:
                valuesYaml = yaml.load(f, Loader=yaml.FullLoader)
            
            #Global
            for key,val in valuesYaml.items():
                try:
                    tmp__ = hp.config[key]
                except:
                    raise KeyError(
                        "No %s in hyperparams"%(key)
                    )
                for k,v in val.items():
                    if key not in ["Train","Eval"]:
                        try:
                            print(
                                "%s,%s,%s"%(
                                    k, hp.config[key][k], v
                                )
                            )
                        except Exception as e:
                            print(str(e))
                    
            t_yml, e_yml = valuesYaml["Train"], valuesYaml["Eval"]
            t_cfg, e_cfg = hp.config["Train"], hp.config["Eval"]

            print(
                t_yml["dataset"]["name"], t_cfg["dataset"]["name"],"\n",
                t_yml["dataset"]["data_dir"], t_cfg["dataset"]["data_dir"],"\n",
                t_yml["dataset"]["label_file_list"], t_cfg["dataset"]["label_file_list"],"\n",
                t_yml["dataset"]["ratio_list"], t_cfg["dataset"]["ratio_list"],"\n",

                e_yml["dataset"]["name"], e_cfg["dataset"]["name"],"\n",
                e_yml["dataset"]["data_dir"],e_cfg["dataset"]["data_dir"],"\n",
                e_yml["dataset"]["label_file_list"], e_cfg["dataset"]["label_file_list"],"\n"
            )

            t_trans_yml, e_trans_yml = valuesYaml["Train"]['dataset']['transforms'], valuesYaml["Eval"]['dataset']['transforms']
            t_trans_cfg, e_trans_cfg = hp.config["Train"]['dataset']['transforms'], hp.config["Eval"]['dataset']['transforms']   

            print(
                "\n\n\n\n\n",t_trans_yml, "\n\n\n\n\n", t_trans_cfg
            )
            print(
                "\n\n\n\n\n",e_trans_yml, "\n\n\n\n\n", e_trans_cfg
            )     

            
            #Calling program method and train RE model
            config, device, logger, vdl_writer = program.preprocess(is_train = not self.predict, flags_ = hp)
            seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
            set_seed(seed)

            now = time.time()
            best_metric = train.main(config, device, logger, vdl_writer)

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

            print(fn_optim)

            #Export model by first reading the hyperparameter optimisation log, and see if the current metric is better than the ones recorded. If that's the case, then save the model.          
            df__ = pd.read_csv(
                fn_optim
            )

            past_best_metric = df__["metric"]

            #You actually have done the optimisaiton
            if len(past_best_metric)>0:
                if best_metric >= past_best_metric.max():
                    self.export(hp)

        #Return best_metric for the optimisaiton's objective function
        return best_metric
    
    def predict(self):
        '''Predict from the final trained-model'''
        pass

    def infer(self):
        '''Predict from the model checkpoints'''
        pass
        

def create_log_optimisation(model_dir, model):
    #File to store hyperparameter optimisation
    with open(
        "%s/optimisation_model_%s.csv"%(model_dir, model),"w"
    ) as f:
        f.write(
            "beta1,beta2,learning_rate,regularizer_factor,metric\n"
        )

def free_GPU():
    #Method to kill all processes running on GPU and free GPU resources
    cuda.select_device(0)
    cuda.close()


if __name__ == "__main__":
    #FIXME: Add hyperparams for RE
    # for _ in 20000 maximise training metric by changing hyperparams
    model_dir = '/home/models/20bb2d54-661f-440d-9dc8-80b1ed743435' #'/home/philgun/Documents/sribuu/ocr/models/re'
    useCPU = False

    train_fraction = 0.6
    validation_fraction = 0.2
    test_fraction = 0.2

    #What model do you train?ALL
    model = "RE"

    trainer = SribuuOCRTrainer(
        model_dir = model_dir,
        trainResume = None,
        useCPU = useCPU
    )

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
    hyperparams["epoch_num"] = 2
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

    '''
    hyperparams["beta1"] = 0.9
    hyperparams["beta2"] = 0.999
    hyperparams["learning_rate"] = 5e-5
    hyperparams["regularizer_factor"] = 0.0
    '''

    #Instantiate partial obj function, where hyperparams is left off for optimisation routine
    objfunc = functools.partial(
        trainer.fit, hyperparams = hyperparams, model=model, 
        train_fraction=train_fraction, validation_fraction=validation_fraction, test_fraction=test_fraction
    )

    #Instantiate the training parameters
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
        random_state=42
    )

    #Star bayesian optimiser
    start_time = time.time()
    
    opt.maximize(init_points=1,n_iter=1)
    
    delta = time.time() - start_time

    print("Total time for bayesian optimisation: %s s"%delta)

    '''
    #Catch training metric
    best_metric = trainer.fit(
        model=model, hyperparams = hyperparams
    )
    '''

    print(
        "Output best metric = %s"%(opt.max)
    )

    #Free-ing GPU resources
    print(
        "Releasing GPU resources.......\n\n"
    )
    free_GPU()
    print(
        "Done releasing GPU.........\n\n"
    )


'''
        #Link the linking file
/home/philgun/Documents/sribuu/ocr/paddle-ocr-sribuu/tools_custom
        subprocess.run(["python3", linking_gen_script, "--linkingFile", "label-key-list-pair.txt", "--labelFile", "Label.txt", "--labelOutputFile", "Label-linked.txt"])

        print("== Generate Rec Cropped Img")
        subprocess.run(["python3", rec_gt_gen_script, "--outputFileGT", "rec_gt.txt", "--labelFile", "Label.txt", "--outputFileDir", "crop_img/"])

        print("== Splitting dataset with ratio 6:2:2")
        subprocess.run(["python3", dataset_divider_script, "--trainValTestRatio", "6:2:2", "--datasetRootPath", "", "--detLabelFileName", "Label-linked.txt", "--recLabelFileName", "rec_gt.txt", "--recImageDirName", "crop_img", "--detRootPath", "./train_data/det", "--recRootPath", "./train_data/rec"])

        # Count num classes
        with open("label-key-list.txt", 'r') as f:
            num_count = len([line for line in f if line.strip()])

        num_classes = (2 * num_count) - 1
        print(f"== Update num_classes to {num_classes}")

        with open(algorithm_ser, 'r+') as f:
            content = f.read()
            new_content = re.sub(r'&num_classes.*', f'&num_classes {num_classes}', content)
            f.seek(0)
            f.write(new_content)
            f.truncate()

        if train == "SER" or train == "ALL":
            print("== Training SER Model")
            try:
                os.system("python3 %s -c %s -o Global.save_model_dir=./model_checkpoint/ser/ Global.use_gpu=%s"%(trainer_script,algorithm_ser,use_gpu))
                #subprocess.run(["python3", trainer_script, "-c", algorithm_ser, "-o", "Global.save_model_dir=./model_checkpoint/ser/", f"Global.use_gpu={use_gpu}"], check=True)
            #except subprocess.CalledProcessError as e:
            except Exception as e:
                #print("Error running predict script:", str(e))
                print("Error jing:", str(e))
                exit(1)

            print("== Export SER Model to Inference")
            subprocess.run(["python3", export_model_script, "-c", algorithm_ser, "-o", "Architecture.Backbone.checkpoints=./model_checkpoint/ser/best_accuracy", f"Global.use_gpu={use_gpu}"])

        if train == "RE" or train == "ALL":
            print("== Training RE Model")
            subprocess.run(["python3", trainer_script, "-c", algorithm_re, "-o", "Global.save_model_dir=./model_checkpoint/re/", f"Global.use_gpu={use_gpu}"])

            print("== Export RE Model to Inference")
            subprocess.run(["python3", export_model_script, "-c", algorithm_re, "-o", "Architecture.Backbone.checkpoints=./model_checkpoint/re/best_accuracy", f"Global.use_gpu={use_gpu}"])

        elif predict == "true":
            if not predict_file:
                print("Error: The file to predict is required")
                exit(1)

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
                        "--ser_dict_path=label-key-list.txt",
                        "--vis_font_path=visual_font",
                        "--ocr_order_method=tb-yx"
                    ])
                else:
                    subprocess.run([
                        "python3",
                        predict_ser_script,
                        "--kie_algorithm=LayoutXLM",
                        f"--ser_model_dir={model_compiled_re}",
                        "--use_visual_backbone=False",
                        f"--image_dir={predict_file}",
                        "--ser_dict_path=label-key-list.txt",
                        "--vis_font_path=visual_font",
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
                subprocess.run(["python3", trainer_script, "-c", "algorithm_ser.yml", "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_ser}", f"Global.use_gpu={use_gpu}"])
            elif train_resume == "RE":
                print("== Training RE Model")
                subprocess.run(["python3", trainer_script, "-c", "algorithm_re.yml", "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_re}", f"Global.use_gpu={use_gpu}"])
            else:
                print("Error: Wrong argument, Choose SER or RE.")
                exit(1)
        else:
            print("Error: Argument is missing.")
            exit(1)

    def predict(self):
        pass
        
    

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
    elif key == "algorithm_ser":
        algorithm_ser = value
    elif key == "algorithm_re":
        algorithm_re = value
    elif key == "rec_gt_gen_script":
        rec_gt_gen_script = value
    elif key == "visual_font":
        visual_font = value

try:
    # Prepare anything and do training
    if train:
        print("== TRAINING ==")
        print(f"== Generating id and linking for dataset label based on {linking_file}")
        subprocess.run(["python3", linking_gen_script, "--linkingFile", "label-key-list-pair.txt", "--labelFile", "Label.txt", "--labelOutputFile", "Label-linked.txt"])

        print("== Generate Rec Cropped Img")
        subprocess.run(["python3", rec_gt_gen_script, "--outputFileGT", "rec_gt.txt", "--labelFile", "Label.txt", "--outputFileDir", "crop_img/"])

        print("== Splitting dataset with ratio 6:2:2")
        subprocess.run(["python3", dataset_divider_script, "--trainValTestRatio", "6:2:2", "--datasetRootPath", "", "--detLabelFileName", "Label-linked.txt", "--recLabelFileName", "rec_gt.txt", "--recImageDirName", "crop_img", "--detRootPath", "./train_data/det", "--recRootPath", "./train_data/rec"])

        # Count num classes
        with open("label-key-list.txt", 'r') as f:
            num_count = len([line for line in f if line.strip()])

        num_classes = (2 * num_count) - 1
        print(f"== Update num_classes to {num_classes}")

        with open(algorithm_ser, 'r+') as f:
            content = f.read()
            new_content = re.sub(r'&num_classes.*', f'&num_classes {num_classes}', content)
            f.seek(0)
            f.write(new_content)
            f.truncate()
            
        if train == "SER" or train == "ALL":
            print("== Training SER Model")
            try:
                os.system("python3 %s -c %s -o Global.save_model_dir=./model_checkpoint/ser/ Global.use_gpu=%s"%(trainer_script,algorithm_ser,use_gpu))
                #subprocess.run(["python3", trainer_script, "-c", algorithm_ser, "-o", "Global.save_model_dir=./model_checkpoint/ser/", f"Global.use_gpu={use_gpu}"], check=True)
            #except subprocess.CalledProcessError as e:
            except Exception as e:
                #print("Error running predict script:", str(e))
                print("Error jing:", str(e))
                exit(1)

            print("== Export SER Model to Inference")
            subprocess.run(["python3", export_model_script, "-c", algorithm_ser, "-o", "Architecture.Backbone.checkpoints=./model_checkpoint/ser/best_accuracy", f"Global.use_gpu={use_gpu}"])

        if train == "RE" or train == "ALL":
            print("== Training RE Model")
            subprocess.run(["python3", trainer_script, "-c", algorithm_re, "-o", "Global.save_model_dir=./model_checkpoint/re/", f"Global.use_gpu={use_gpu}"])

            print("== Export RE Model to Inference")
            subprocess.run(["python3", export_model_script, "-c", algorithm_re, "-o", "Architecture.Backbone.checkpoints=./model_checkpoint/re/best_accuracy", f"Global.use_gpu={use_gpu}"])

    elif predict == "true":
        if not predict_file:
            print("Error: The file to predict is required")
            exit(1)

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
                    "--ser_dict_path=label-key-list.txt",
                    "--vis_font_path=visual_font",
                    "--ocr_order_method=tb-yx"
                ])
            else:
                subprocess.run([
                    "python3",
                    predict_ser_script,
                    "--kie_algorithm=LayoutXLM",
                    f"--ser_model_dir={model_compiled_re}",
                    "--use_visual_backbone=False",
                    f"--image_dir={predict_file}",
                    "--ser_dict_path=label-key-list.txt",
                    "--vis_font_path=visual_font",
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
            subprocess.run(["python3", trainer_script, "-c", "algorithm_ser.yml", "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_ser}", f"Global.use_gpu={use_gpu}"])
        elif train_resume == "RE":
            print("== Training RE Model")
            subprocess.run(["python3", trainer_script, "-c", "algorithm_re.yml", "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_re}", f"Global.use_gpu={use_gpu}"])
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
try:
    # Prepare anything and do training
    if train:
        print("== TRAINING ==")
        print(f"== Generating id and linking for dataset label based on {linking_file}")
        subprocess.run(["python3", linking_gen_script, "--linkingFile", "label-key-list-pair.txt", "--labelFile", "Label.txt", "--labelOutputFile", "Label-linked.txt"])

        print("== Generate Rec Cropped Img")
        subprocess.run(["python3", rec_gt_gen_script, "--outputFileGT", "rec_gt.txt", "--labelFile", "Label.txt", "--outputFileDir", "crop_img/"])

        print("== Splitting dataset with ratio 6:2:2")
        subprocess.run(["python3", dataset_divider_script, "--trainValTestRatio", "6:2:2", "--datasetRootPath", "", "--detLabelFileName", "Label-linked.txt", "--recLabelFileName", "rec_gt.txt", "--recImageDirName", "crop_img", "--detRootPath", "./train_data/det", "--recRootPath", "./train_data/rec"])

        # Count num classes
        with open("label-key-list.txt", 'r') as f:
            num_count = len([line for line in f if line.strip()])

        num_classes = (2 * num_count) - 1
        print(f"== Update num_classes to {num_classes}")

        with open(algorithm_ser, 'r+') as f:
            content = f.read()
            new_content = re.sub(r'&num_classes.*', f'&num_classes {num_classes}', content)
            f.seek(0)
            f.write(new_content)
            f.truncate()
            
        if train == "SER" or train == "ALL":
            print("== Training SER Model")
            try:
                os.system("python3 %s -c %s -o Global.save_model_dir=./model_checkpoint/ser/ Global.use_gpu=%s"%(trainer_script,algorithm_ser,use_gpu))
                #subprocess.run(["python3", trainer_script, "-c", algorithm_ser, "-o", "Global.save_model_dir=./model_checkpoint/ser/", f"Global.use_gpu={use_gpu}"], check=True)
            #except subprocess.CalledProcessError as e:
            except Exception as e:
                #print("Error running predict script:", str(e))
                print("Error jing:", str(e))
                exit(1)

            print("== Export SER Model to Inference")
            subprocess.run(["python3", export_model_script, "-c", algorithm_ser, "-o", "Architecture.Backbone.checkpoints=./model_checkpoint/ser/best_accuracy", f"Global.use_gpu={use_gpu}"])

        if train == "RE" or train == "ALL":
            print("== Training RE Model")
            subprocess.run(["python3", trainer_script, "-c", algorithm_re, "-o", "Global.save_model_dir=./model_checkpoint/re/", f"Global.use_gpu={use_gpu}"])

            print("== Export RE Model to Inference")
            subprocess.run(["python3", export_model_script, "-c", algorithm_re, "-o", "Architecture.Backbone.checkpoints=./model_checkpoint/re/best_accuracy", f"Global.use_gpu={use_gpu}"])

    elif predict == "true":
        if not predict_file:
            print("Error: The file to predict is required")
            exit(1)

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
                    "--ser_dict_path=label-key-list.txt",
                    "--vis_font_path=visual_font",
                    "--ocr_order_method=tb-yx"
                ])
            else:
                subprocess.run([
                    "python3",
                    predict_ser_script,
                    "--kie_algorithm=LayoutXLM",
                    f"--ser_model_dir={model_compiled_re}",
                    "--use_visual_backbone=False",
                    f"--image_dir={predict_file}",
                    "--ser_dict_path=label-key-list.txt",
                    "--vis_font_path=visual_font",
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
            subprocess.run(["python3", trainer_script, "-c", "algorithm_ser.yml", "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_ser}", f"Global.use_gpu={use_gpu}"])
        elif train_resume == "RE":
            print("== Training RE Model")
            subprocess.run(["python3", trainer_script, "-c", "algorithm_re.yml", "-o", f"Architecture.Backbone.checkpoints={model_checkpoint_re}", f"Global.use_gpu={use_gpu}"])
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
'''