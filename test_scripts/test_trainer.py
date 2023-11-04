import unittest
from ocrtransform.model import trainer
import cv2
import pytesseract
import os




class TestDetector(unittest.TestCase):
    def setUp(self) -> None:
        
        imgs_path = "/content/OCRTransformer/dataset/images"
        labels_path = "/content/OCRTransformer/dataset/key_information/key_information.json"
        batch_size = 4
        epochs = 1
        augment = True
        out_dir = "myawesome_detector"
        
        self.trainer_engine = trainer.Trainer(
            imgs_path=imgs_path, anno_path=labels_path,
            batch_size=batch_size, augment=augment,
            epochs=epochs, output_dir=out_dir
        )

        # Loading data
        self.trainer_engine.load_data()
        self.prim_data = self.trainer_engine.df
        
        # Preprocessing data
        self.trainer_engine.preprocess_texts()
        self.processed_df = self.trainer_engine.df
        
        # Download pretrained model
        self.trainer_engine.download_pretrained_model()
        self.model_dir = self.trainer_engine.model_dir
        
        # Tokenize text data
        self.trainer_engine.tokenize_text_data()
        
        # Train
        self.trainer_engine.train()
    
    def test_trainer_engine(self):
        condition = (self.trainer_engine != None) and (type(self.trainer_engine) == trainer.Trainer)
        self.assertTrue(condition,
                        msg="Trainer object was not built successfully")
    
    def test_data_loaded(self):
        condition = (self.prim_data != None) and (len(self.prim_data)>0)
        self.assertTrue(condition, msg="Data was not loaded successfully")
    
    
    def test_preprocess_texts(self):
        condition = (self.processed_df != None) and (len(self.processed_df)>0)
        self.assertTrue(condition, msg="Data was not preprocessed successfully")
    
    
    
    def test_model_loading(self):
        condition = os.path.isdir(self.model_dir)
        self.assertTrue(condition,
                        msg="Pretrained model not loaded successfuly")
    

    def test_tokenizing(self):
        train_tokenized_data = self.trainer_engine.tokenized_data_train
        test_tokenized_data = self.trainer_engine.tokenized_data_test
        condition = (train_tokenized_data != None) and (test_tokenized_data != None)
        self.assertTrue(condition,
                        msg="Tokenizing failed, make sure pretrained model downloaded successfully")
    
    def test_training_process(self):
        self.assertTrue(self.trainer_engine.train_state, 
                        msg="Training process failed")

if __name__ == "__main__":
    unittest.main()
