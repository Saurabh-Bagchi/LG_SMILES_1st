# DACON Smiles competition Code - PBRH

Main approach uses a RCNN type model to directly predict individual atom and bond locations, and constructs molecules from those.


## Installation

Note: Miniconda3 must be installed to install RDkit. 
The code runs on linux (tested on Ubuntu 16.04).
Code tested on cuda10.1+pytorch1.6.
```bash
pip install requirements.txt
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
conda install -c conda-forge rdkit=2020.03.1 
```

## Usage

- **main.py** : Main entry to code, constructs trainer object to train model, and predicts smiles for images in test folder.
- **trainer.py** : Trainer class, preprocesses data, and trains model.
- **labels_generation.py** : Functions to preprocess data and generate annotations for RCNN.
- **inference.py** : Constructs mol object from predicted atom and bond bounding boxes .
- **utils.py** : Other general helper functions.

#### How to train
Set train_new_model to True in main.py and run main.py.

#### How to predict
Set test_data_path in main.py to the directory containing test images and run main.py.
Really important to set the parameter **'input_format'** to **"RGB" or "GBR"** depending
on the particular set of images!

#### Data
**No external data was used for any training or prediction**, all was calculated directly
from RDKit. Even the training images are generated in case they are not present in 
the folder **"./data/images/train"**.

#### Pretrained model
A pretrain model is available to obtain predictions similar to the competition submission.
**"./trained_models/final_submission.pth"**.

#### Final remarks
Due to random initialization difference between different operating systems, compute environment and GPU types, 
results can slightly differ between machines.  