#!/usr/bin/env bash

if [ ! -x /usr/bin/curl ] ; then                      
    command -v curl >/dev/null 2>&1 || { echo >&2 "Please install curl or set it in your path. Aborting."; exit 1; }
fi

cd "${BASH_SOURCE%/*}"

printf "Downloading NLTK Tokenizer...\n"
python -c 'import nltk; nltk.download("punkt")'

printf "\nDownloading Pre-Trained GloVe Word Vectors...\n"
mkdir glove.6B/                                                            && \
cd glove.6B/                                                               && \
curl http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip > glove.6B.zip && \
unzip glove.6B.zip                                                         && \
rm glove.6B.zip                                                            && \
cd ../

printf "\nDownloading MS COCO 2014...\n"
mkdir data/                                                           && \
cd data/                                                              && \
curl http://images.cocodataset.org/zips/train2014.zip > train2014.zip && \
unzip train2014.zip                                                   && \
rm train2014.zip                                                      && \
curl http://images.cocodataset.org/zips/val2014.zip > val2014.zip     && \
unzip val2014.zip                                                     && \
rm val2014.zip                                                        && \
mkdir annotations/                                                    && \
cd annotations/                                                       && \
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip > annotations.zip     && \
unzip annotations.zip                                                 && \
rm annotations.zip                                                    && \
cd ../../

mkdir checkpoints/