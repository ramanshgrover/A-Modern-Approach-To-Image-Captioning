# A-Modern-Approach-To-Image-Captioning
This repository contains the source code for my course project on Image Captioning in EC-353, Computer Vision.

## Table of Contents
  1. [Motivation](#Motivation)
  2. [The Dataset](#Data-Acquisition)
  3. [Prerequisites and Dependencies](#Prerequisites-and-Dependencies)
  4. [Methodology](#Methodology)
  5. [License](https://github.com/rachitsaksena/Multilingual-Agression-Classification/blob/master/LICENSE)

## Motivation
In the past few years, recent developments in Image Captioning Systems have been inspired by advancements in object detection and machine translation. The task of image captioning involves two main aspects: (1) resolving the object detection problem in computer vision and (2) creating a language model that can accurately generate a sentence describing the detected objects. Seeing the success of encoder-decoder models with "soft" attention, I use soft alignment [https://arxiv.org/pdf/1409.0473.pdf](https://arxiv.org/pdf/1409.0473.pdf) and modern approaches in object detection [https://arxiv.org/pdf/1412.7755.pdf](https://arxiv.org/pdf/1412.7755.pdf) as a baseline model. To extend this work, I investigate the effect of pre-trained embeddings by integrating GloVe embeddings [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf) and contextualized BERT vectors [https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf) to enhance the models performance and reduce training time.

## Data Acquisition
I opted for Microsoft's Common Objects in Context (COCO) 2014 Dataset [https://arxiv.org/pdf/1405.0312.pdf](https://arxiv.org/pdf/1405.0312.pdf) for both training and validation readily utilizing the pyCOCO API for cleaning and structuring scripts to parse the captions, extracting the vocabulary, and in order to batch the images to optimize the training process for the models. This dataset roughly consists of 83K training and 41K validation samples.

## Prerequisites and Dependencies
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

Enter the Python console using `python` , input the following and exit the shell
```python
>>> import nltk
>>> nltk.download(‘punkt’)
```

## Execution
