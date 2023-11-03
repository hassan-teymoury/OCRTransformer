import pandas as pd
import random
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from datasets import load_dataset, Dataset
import os
import gdown
import pytesseract
import cv2
import json
from tqdm import tqdm
import time


class Train(object):
    
    def __init__(self, imgs_path, anno_path, augment=True, epochs=4, batch_size=4, output_dir="summarizaton_model"):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            self.output_dir = os.path.join(os.getcwd(),self.output_dir)
            os.mkdir(self.output_dir)
            
        self.model_dir = os.path.join(os.getcwd(), "checkpoints")
        self.augment = augment
        with open(anno_path, "r") as f:
            annos = json.load(f)
            f.close()

        imgs_list = sorted(os.listdir(imgs_path))

        data = []

        columns = ["ocr_text", "gt_date", "gt_total"]

        print("Loading Started ...")
        for img_name in tqdm(imgs_list):
            img_path = os.path.join(imgs_path, img_name)
            img_id = img_name.split(".")[0]
            img_info = annos[img_id]
            gt_date = img_info["date"]
            gt_total = img_info["total"]
            
            img = cv2.imread(img_path)
            refined_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

            text = pytesseract.image_to_string(refined_img)
            text = text.replace("\n", " ")
            text_sections = [t.strip() for t in text.split(" ")]
            text = " ".join(text_sections)
            
            final_row = [text, gt_date, gt_total]
            data.append(final_row)
            time.sleep(1)
        
        
        self.df = pd.DataFrame(data=data, columns=columns, index=[i for i in range(len(data))])
        
        print("Loading finished")
        
        if self.augment:
            print("Augmenting data for training ...")
            
            augmented_data = []

            for d in self.df.values.tolist():
                try:
                    ocr_text = d[0]
                    gt_date = d[1]
                    gt_total = d[2]
                    ocr_text_splits = ocr_text.split(" ")
                    for i in range(200):
                        random.shuffle(ocr_text_splits)
                        ocr_text = ", ".join(ocr_text_splits)
                        summary = f"date is {gt_date}" + " and " + f"total is {gt_total}"
                        row_aug = [ocr_text, gt_date, gt_total, summary]
                        augmented_data.append(row_aug)

                except:
                    pass

            self.df = pd.DataFrame(data=augmented_data, columns=["text", "gt_date", "gt_total", "summary"],
                            index=[i for i in range(len(augmented_data))])
            print("Augmentation Finished")
    
    
    
    
    
    
    def train(self):
        
        if not os.path.isdir(self.model_dir):
            pretrained_path = "https://drive.google.com/drive/folders/12rpAG5JopyFcL0E2wGiEhhBfm05KzXmw?usp=sharing"
        
            print("Downloading t5 Seq2Seq pretrained model ....")
            gdown.download_folder(pretrained_path, quiet=True, use_cookies=False,
                                output=f"{self.model_dir}")
            print("Finished")

        rouge = evaluate.load("rouge")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        data = Dataset.from_pandas(self.df)
        data = data["train"].train_test_split(test_size=0.2)
        
        prefix = "summarize: "
        
        def preprocess_function(examples):
            inputs = [prefix + doc for doc in examples["text"]]
            model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

            labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_data = data.map(preprocess_function, batched=True)

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=self.model_dir)
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)

            return {k: round(v, 4) for k, v in result.items()}
        
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.epochs,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
            save_strategy = "epoch"
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        print("Training started ...")

        trainer.train()

        print("Finished Training")