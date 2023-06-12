class HyperParameters(object):
    '''
    Class to contain all hyperparameters. This is needed to completely get rid off the *.yml file. Don't wanna write bunch of yml files when doing optimisation since it can get super messy.
        args:
            hyperparams, dict, key, values are bunch of stuff I can't remember. Check the class instantiation
    '''
    def __init__(
            self, model_dir, global_model = "SER",

            profiler_options=None,
            
            use_gpu=False, epoch_num=200, log_smooth_window=10, print_batch_step=10, save_model_dir='./model_checkpoint/ser',
            save_epoch_step=2000, eval_batch_step = [0,19], cal_metric_during_train=False, save_inference_dir='./model_compiled/ser',
            use_visualdl=False, seed=143, infer_img='./train_data/det/test.txt', infer_mode=False, save_res_path='./output/ser',
            kie_rec_model_dir=None, kie_det_model_dir=None, 
            
            model_type='kie',algorithm='LayoutXLM', Transform=None, architecture_name='LayoutXLMForSer', pretrained=True, checkpoints=None, mode='vi', num_classes=105, 
            
            loss_name='VQASerTokenLayoutLMLoss', key='backbone_out', optimizer_name='AdamW',beta1=0.9,beta2=0.999,lr_name="Linear",learning_rate=5e-5, warmup_epoch=2, regularizer_name="L2", regularizer_factor=0.0,

            postprocess_name='VQASerTokenLayoutLMPostProcess', postprocess_class_path='label-key-list.txt',

            metric_name='VQASerTokenMetric', metric_main_indicator="hmean"
        ):
        #FIXME: Add hyperparams for global_model "SER"


        #Instantiate self.config attribute
        self.config = {}
        self.config["profiler_options"] = profiler_options

        #Populate the Global key
        self.config["Global"] = {}
        self.config["Global"]['use_gpu'] = use_gpu
        self.config["Global"]['epoch_num'] = epoch_num
        self.config["Global"]['log_smooth_window'] = log_smooth_window
        self.config["Global"]['print_batch_step'] = print_batch_step
        self.config["Global"]['save_model_dir'] = save_model_dir
        self.config["Global"]['save_epoch_step'] = save_epoch_step
        self.config["Global"]['eval_batch_step'] = eval_batch_step
        self.config["Global"]['cal_metric_during_train'] = cal_metric_during_train
        self.config["Global"]['save_inference_dir'] = save_inference_dir
        self.config["Global"]['use_visualdl'] = use_visualdl
        self.config["Global"]['seed'] = seed
        self.config["Global"]['infer_img'] = infer_img
        self.config["Global"]['infer_mode'] = infer_mode
        self.config["Global"]['save_res_path'] = save_res_path
        self.config["Global"]['kie_rec_model_dir'] = kie_rec_model_dir
        self.config["Global"]['kie_det_model_dir'] = kie_det_model_dir

        #Populate the Architecture key
        self.config["Architecture"] = {}
        self.config["Architecture"]["model_type"] = model_type
        self.config["Architecture"]["algorithm"] = algorithm
        self.config["Architecture"]["Transform"] = Transform
        self.config["Architecture"]["Backbone"] = {}
        self.config["Architecture"]["Backbone"]["name"] = architecture_name
        self.config["Architecture"]["Backbone"]["pretrained"] = pretrained
        self.config["Architecture"]["Backbone"]["checkpoints"] = checkpoints
        self.config["Architecture"]["Backbone"]["mode"] = mode
        self.config["Architecture"]["Backbone"]["num_classes"] = num_classes

        #Populate the Loss key
        self.config["Loss"] = {}
        self.config["Loss"]["name"] = loss_name
        self.config["Loss"]["num_classes"] = num_classes
        self.config["Loss"]["key"] = key

        #Populate Optimizer key
        self.config["Optimizer"] = {}
        self.config["Optimizer"]["name"] = optimizer_name
        self.config["Optimizer"]["beta1"] = beta1
        self.config["Optimizer"]["beta2"] = beta2
        self.config["Optimizer"]["lr"] = {}
        self.config["Optimizer"]["lr"]["name"] = lr_name
        self.config["Optimizer"]["lr"]["learning_rate"] = learning_rate
        self.config["Optimizer"]["lr"]["epochs"] = epoch_num
        self.config["Optimizer"]["lr"]["warmup_epoch"] = warmup_epoch
        self.config["Optimizer"]["regularizer"] = {}
        self.config["Optimizer"]["regularizer"]["name"] = regularizer_name
        self.config["Optimizer"]["regularizer"]["factor"] = regularizer_factor

        #Populate PostProcess key
        self.config["PostProcess"] = {}
        self.config["PostProcess"]["name"] = postprocess_name
        self.config["PostProcess"]["class_path"] = "%s/%s"%(model_dir,postprocess_class_path)

        #Populate Metric key
        self.config["Metric"] = {}
        self.config["Metric"]["name"] = metric_name
        self.config["Metric"]["main_indicator"] = metric_main_indicator

        #Populate Train key
        self.config["Train"] = {'dataset': {'name': 'SimpleDataSet', 'data_dir': '%s/train_data/det/train'%(model_dir), 'label_file_list': ['%s/train_data/det/train.txt'%(model_dir)], 'ratio_list': [1.0], 'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': algorithm, 'class_path': '%s/label-key-list.txt'%(model_dir), 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}}, {'NormalizeImage': {'scale': 1, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]}, 'loader': {'shuffle': True, 'drop_last': False, 'batch_size_per_card': 8, 'num_workers': 4}}

        #Populate Eval key
        self.config["Eval"] = {'dataset': {'name': 'SimpleDataSet', 'data_dir': '%s/train_data/det/val'%(model_dir), 'label_file_list': ['%s/train_data/det/val.txt'%(model_dir)], 'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': algorithm, 'class_path': '%s/label-key-list.txt'%(model_dir), 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}}, {'NormalizeImage': {'scale': 1, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]}, 'loader': {'shuffle': False, 'drop_last': False, 'batch_size_per_card': 8, 'num_workers': 4}}

    def load_config(self):
        return self.config
    
if __name__ == "__main__":
    import os
    cwd = os.getcwd()
    c = HyperParameters(
        model_dir=cwd
    )
    d = c.load_config()

    for k,v in d.items():
        print(
            k,v
        )