import subprocess
from setuptools import setup



subprocess.run(f"apt install tesseract-ocr", shell=True)
subprocess.run(f"apt install libtesseract-dev", shell=True)

def parse_packages(packages_txt):

    with open(packages_txt, "r") as f:
        lines = f.readlines()
        f.close()

    packages_list = []
    for line in lines:
        packages_list.append(str(line.strip()))

    return packages_list



if __name__ == "__main__":
    
    setup(
        name='ocrtransform',
        version='0.1.0',
        description='A Python package to extract key information from images of the purchase receipts using OCR and Transformers',
        author='Hassan Teymouri',
        author_email='hassan.teymoury@gmail.com',
        url="https://github.com/hassan-teymoury/OCRTransformer.git",
        license='Apache License 2.0',
        packages=["ocrtransform.model"],
        include_package_data=True,
        install_requires=parse_packages("requirements.txt"),
    )
    