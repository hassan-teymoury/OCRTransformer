import cv2
import pytesseract
import pandas as pd
from typing import Union, Optional, List, Dict
import gdown
import os
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM





class Detector(object):
    
    def __init__(self, img_path) -> None:
        
        # Loading images and extracting text from them
        img = cv2.imread(img_path)
        refined_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        text = pytesseract.image_to_string(refined_img)
        text = text.replace("\n", " ")
        text_sections = [t.strip() for t in text.split(" ")]
        self.text = " ".join(text_sections)
        self.text = "summarize: " + self.text
        
        
        
        # Downloading Pretrained weights
        self.model_dir = os.path.join(os.getcwd(), "checkpoints")
        
        pretrained_path = "https://drive.google.com/drive/folders/11ytj6ERnKxEd1-2RC2KkLtlngnqjsNUP?usp=drive_link"
        
        print("Downloading t5 Seq2Seq pretrained model ....")
        gdown.download_folder(pretrained_path, quiet=True, use_cookies=False,
                              output=f"{self.model_dir}")
        print("Finished")
        
    
    def forward_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        inputs = tokenizer(self.text, return_tensors="pt").input_ids
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
        self.results_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def postprocess(self):
        results_splits = self.results_txt.split("and")
        date_script = results_splits[0]
        total_script = results_splits[1]
        date_ = date_script.split(" ")[-1]
        total_ = total_script.split(" ")[-1]
        self.final_res = {"date":date_, "total":total_}

        