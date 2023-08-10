import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
from numba import cuda
import argparse
import os
from PIL import Image

class DocClassifier:

    def __init__(self, path_to_model, path_to_processor):
        self.processor = DonutProcessor.from_pretrained(path_to_processor)
        self.model = VisionEncoderDecoderModel.from_pretrained(path_to_model)
        self.device = "cuda" if cuda.is_available() else "cpu"

    '''
    input: a PIL image/pixel values
    output: a label
    '''
    def predict_class(self, img):
        # converting the image data to pixel
        pixel_values = self.processor(img.convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # specifying task prompt, <s_rvlcdip> for classification
        task_prompt = "<s_rvlcdip>"
        
        # creating decoder input_ids
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        # autoregressively generating output sequences based on the pixel_values
        outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        
        # turning the sequence into JSON
        seq = self.processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = self.processor.token2json(seq)
        
        # the final output is seq['class']
        return(seq['class'])
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--model_path", type=str)

    args, left_argv = parser.parse_known_args()
    image = Image.open(args.file_name)
    path_parent = args.model_path
    path_to_model = os.path.join(path_parent,"latest-model-073123")
    path_to_processor = os.path.join(path_parent,"latest-processor-073123")

    # initializing the model class
    doc_classifier = DocClassifier(path_to_model, path_to_processor)

    # calling predict on image
    print(doc_classifier.predict_class(image))