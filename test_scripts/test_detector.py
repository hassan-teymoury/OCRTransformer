import unittest
from ocrtransform.model import detector
import cv2
import pytesseract
import os




class TestDetector(unittest.TestCase):
    def setUp(self) -> None:
        img_path = "dataset/test_data/02.jpg"
        self.img = cv2.imread(img_path)
        self.denoised_img = cv2.fastNlMeansDenoisingColored(self.img, None, 10, 10, 7, 15)
        self.text = pytesseract.image_to_string(self.denoised_img)
        text = self.text.replace("\n", " ")
        text_sections = [t.strip() for t in text.split(" ")]
        text = ", ".join(text_sections)
        self.text_processed = "summarize: " + text
        self.inferencer = detector.Detector()
        self.model_dir = os.path.join(os.getcwd(), "checkpoints")
        self.txt_res = self.inferencer.forward_model(self.text_processed)
        self.json_res = self.inferencer.postprocess(self.txt_res)
    
    
    
    def test_img_loaded(self):
        self.assertIsNotNone(self.img, msg="Wrong image path")
    
    
    def test_preprocess_img(self):
        self.assertNotEqual(self.img, self.denoised_img, msg="Your image didn't change after denoising")
    
    
    def test_ocr(self):
        self.assertIsNotNone(self.text,
                             msg="The text content has not been extracerd. This issue could be happened when your image doesn't have any text content or bad calling of pytesseract")
    
    
    def test_model_loading(self):
        cond_1 = self.inferencer != None
        cond_2 = os.path.isdir(self.model_dir)
        condition = cond_1==True and cond_2==True
        self.assertTrue(condition,
                        msg="Pretrained model not loaded successfuly")
    
    def test_text_processing(self):
        self.assertNotEqual(self.text, self.text_processed,
                            msg="The extracted text didn't change after preprocessing")
    
    def test_text_res(self):
        condition = self.txt_res not in (None, "", ' ')
        self.assertTrue(condition,
                        msg="The model could not generate an output using the pretrained seq2seq model")
    
    
    def test_postprocess(self):
        condition = (self.json_res != None) and (type(self.json_res) == dict)
        self.assertTrue(condition, 
                        msg="The result after posprocessing didn't convert to dictionary or key-value format")
        


if __name__ == "__main__":
    unittest.main()
