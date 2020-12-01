# A-Modern-Approach-To-Image-Captioning
This repository contains the source code for my course project on Image Captioning in EC-353, Computer Vision.

## Table of Contents
  1. [Motivation](#Motivation)
  2. [The Dataset](#Data-Acquisition)
  3. [Prerequisites and Dependencies](#Prerequisites-and-Dependencies)
  4. [Methodology](#Methodology)
  5. [License](https://github.com/ramanshgrover/A-Modern-Approach-To-Image-Captioning/blob/master/LICENSE)

## Motivation
In the past few years, recent developments in Image Captioning Systems have been inspired by advancements in object detection and machine translation. The task of image captioning involves two main aspects: (1) resolving the object detection problem in computer vision and (2) creating a language model that can accurately generate a sentence describing the detected objects. Seeing the success of encoder-decoder models with "soft" attention, I use soft alignment ([https://arxiv.org/pdf/1409.0473.pdf](https://arxiv.org/pdf/1409.0473.pdf)) and modern approaches in object detection ([https://arxiv.org/pdf/1412.7755.pdf](https://arxiv.org/pdf/1412.7755.pdf)) as a baseline model. To extend this work, I investigate the effect of pre-trained embeddings by integrating GloVe embeddings ([https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf)) and contextualized BERT vectors ([https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf)) to enhance the models performance and reduce training time.

## Data Acquisition
I opted for Microsoft's Common Objects in Context (COCO) 2014 Dataset ([https://arxiv.org/pdf/1405.0312.pdf](https://arxiv.org/pdf/1405.0312.pdf)) for both training and validation readily utilizing the pyCOCO API for cleaning and structuring scripts to parse the captions, extracting the vocabulary, and in order to batch the images to optimize the training process for the models. This dataset roughly consists of 83K training and 41K validation samples.

## Prerequisites, Downloadables and Dependencies
Ensure that you have [Python 3.5+](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py) installed on your machine.

Clone the repository and create a virtual environment.
```shell
git clone https://github.com/ramanshgrover/A-Modern-Approach-To-Image-Captioning/
python -m venv .env
source .env/bin/activate
```

Install [dependencies](https://github.com/ramanshgrover/A-Modern-Approach-To-Image-Captioning/tree/master/requirements.txt) directly by
```shell
cd A-Modern-Approach-To-Image-Captioning/
pip install -r requirements.txt
``` 
**Disclaimer:** _This may take a while._

For downloadables, simply execute `download.sh` by inputting the following in your terminal.
```shell
./download.sh
```
**Disclaimer:** _Make sure you have 25+ GB of storage space on your system (where the repository is cloned)._

After this step, your directory structure should look like this:
```
A-Modern-Approach-To-Image-Captioning
├── checkpoints
├── data
|   ├── annotations
|   |   ├── captions_train2014.json
|   |   ├── captions_val2014.json
|   |   ├── captions_train2014.json
|   |   ├── instances_train2014.json
|   |   ├── instances_val2014.json
|   |   ├── person_keypoints_train2014.json
|   |   └── person_keypoints_val2014.json
|   ├── train2014
|   |   └── * (83K training images)
|   └── val2014
|       └── * (41K validation images)
├── glove.6B
|   ├── glove.6B.50d.txt
|   ├── glove.6B.100d.txt
|   ├── glove.6B.200d.txt
|   └── glove.6B.300d.txt
├── LICENCE
├── README.md
├── download.sh
├── embeddings.py
├── load_data.py
├── model.py
├── preprocess.py
├── requirements.txt
└── validate.ipynb
```
## Training the models
1. Open model.py and scroll to 'START Parameters' (Pre-Trained Models: Baseline, GloVe, BERT)
2. Edit the parameters to train/test the particular model you want
3. Run model.py with `python3`

It is recommended to validate the models via the [jupyter notebook](https://github.com/ramanshgrover/A-Modern-Approach-To-Image-Captioning/blob/main/validate.ipynb) provided.

## Pre-Trained Models
1. Baseline Soft Attention Model ([Show, Attend and Tell: Neural Image Caption Generation with Visual Attention])(https://arxiv.org/pdf/1502.03044.pdf)
2. GloVe Soft Attention Model
3. BERT Soft Attention Model

Due github memory limitations, I wasn't able to upload my trained models. If you wish to validate Pre-Trained Models better, it's much simpler to use the Jupyter Notebook in this repository. Open the notebook and find the Load model section and pick the model you wish to validate and execute. If you would like to compare all the models against each other, try the compare_all() function.
