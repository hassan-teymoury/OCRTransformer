# OCRTransformer

A Python package to extract key information from images of the purchase receipts using OCR and Transformers.
This package is created to extract texts from some purchase receipts and detect key informations such as __date__ and __total__ price value. For this purpose, I have used __`pytesseract`__ to extract texts from images at the first stage. At the second stage, this extracted text passes through a __Seq2Seq__ model that learns the mapping between the input text and its true value (label). I have used a [__summarization__](https://huggingface.co/docs/transformers/tasks/summarization) architecture. I used 80% of images for training and 20% of them for final validation. Model does not see the validation data during training process.
You can find train and test data in __`OCRTransformer/dataset`__.

Overal architecture of this model is illusterated in the folling picture:

### Model architecture
![Alt text](relative%20path/to/img.jpg?raw=true "Title")



## Get started

### installation of the package




