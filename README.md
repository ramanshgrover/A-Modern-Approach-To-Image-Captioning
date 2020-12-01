# A-Modern-Approach-To-Image-Captioning
This repository contains the source code for my course project on Image Captioning in EC-353, Computer Vision.

## Table of Contents
  1. [Motivation](#Motivation)
  2. [The Dataset](#Data-Acquisition)
  3. [Prerequisites, Downloadables and Dependencies](#Prerequisites,-Downloadables-and-Dependencies)
  4. [Data Preprocessing](#Setup)
  4. [Model Training](#Training-the-models)
  5. [Model Evaluation](#Evaluation)
  6. [License](https://github.com/ramanshgrover/A-Modern-Approach-To-Image-Captioning/blob/master/LICENSE)

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

## Setup
Uncomment the last line and run `preprocess.py` – this will re-size and normalize all the images to 224x224px, extract and tokenize the captions to build a vocabulary with all the training and validation set words (which came to be a list of 8,856 words) and generate train2014_resized, val2014_resized, and vocab.pkl respectively in the data folder.

Now run `embeddings.py`. This will assign a weight (from the Pre-Trained GloVe Embeddings) to each word from the vocabulary generated in the previous step and generate glove_words.pkl in the glove.6B folder.

## Training the models
1. Open model.py and scroll to '[START Parameters](https://github.com/ramanshgrover/A-Modern-Approach-To-Image-Captioning/blob/bc34fd97a35790007ddbf6afd27e3013729aa1b9/model.py#L39-L57)' (Pre-Trained Models: Baseline, GloVe, BERT)
2. Edit the parameters to train/test the particular model you want
3. Run model.py with `python3`

It is recommended to validate the models via the [Jupyter Notebook](https://github.com/ramanshgrover/A-Modern-Approach-To-Image-Captioning/blob/main/validate.ipynb) provided.

## Evaluation
1. Baseline Soft Attention Model ([Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf))
2. GloVe Soft Attention Model
3. BERT Soft Attention Model

Due github memory limitations, I wasn't able to upload my trained models. If you wish to validate and evaluate the Pre-Trained Models better, it's much simpler to use the Jupyter Notebook in this repository. Open the notebook and find the Load model section and pick the model you wish to validate and execute. If you would like to compare all the models against each other, try the `compare_all()` function. Below are the results from my experiments.

|   Model  | Validation Loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|:--------:|:---------------:|:------:|:------:|:------:|:------:|
| Baseline |      3.452      |  48.51 |  18.84 |  7.406 |  3.097 |
|   GloVe  |      3.325      |  49.70 |  20.07 |  8.214 |  3.552 |
|   BERT   |      **1.901**      |  **78.27** |  **59.53** |  **46.22** |  **36.53** |

You can find a detailed walkthrough and explanations of this project within the [Project Report](https://github.com/ramanshgrover/A-Modern-Approach-To-Image-Captioning/blob/master/Report.pdf) attached within.
