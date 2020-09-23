#!/usr/bin/python3

from argparse import ArgumentParser

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np

import os

import pickle

from sentence_transformers import SentenceTransformer

import string

GLOVE_DIR = "data/glove.840B.300d.txt"


def load_glove_vectors():
    '''Load GloVe embeddings from .txt file'''
    glove_emb = {}
    f = open(GLOVE_DIR, 'rb')
    cont = f.readlines()
    for i, line in enumerate(cont):
        values = line.rstrip().split()
        word = values[0].decode("utf-8")
        coef = np.asarray(values[1:], dtype="float32")
        glove_emb[word] = coef
        if i % 100 == 0:
            print("{} out of {} words processed".format(i, len(cont)))
    f.close()

    return glove_emb


def preprocess_text(text):
    '''Preprocess text to encode it with GloVe embeddings'''

    lemmatizer = WordNetLemmatizer()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    return lemmas


def embed_glove(id_texts, emb_glove):
    """Encode node texts with Glove embeddings"""
    nodes_glove = {}
    for node_id, node_text in id_texts.items():
        words = preprocess_text(node_text)
        vs = []
        if words:
            for word in words:
                if word in emb_glove:
                    v = emb_glove[word]
                else:
                    v = np.zeros((300, ), dtype="float32")
                vs.append(v)
            vs = np.vstack(vs)
            sent_emb = np.average(vs, axis=0)
        else:
            sent_emb = np.zeros((300, ), dtype="float32")
        nodes_glove[node_id] = sent_emb

    return nodes_glove


def embed_nodes_bert(id_texts):
    """Encdde node texts with sentence BERT embeddings"""
    nodes_bert_sent = {}
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    for node_id, node_text in id_texts.items():
        sent_emb = model.encode(node_text)
        nodes_bert_sent[node_id] = sent_emb

    return nodes_bert_sent

if __name__ == '__main__':

    parser = ArgumentParser(
        description="Encode node texts with GloVe and BERT sentence embeddings")
    parser.add_argument(
        "dataset", help="Dataset with sentences to encode")
    parser.add_argument(
        "--glove", default="data/glove.pickle",
        help="Path to the file with .pkl glove embeddings")
    args = parser.parse_args()

    if os.path.isfile(args.glove):
        with open(args.glove, "rb") as f:
            glove_vectors = pickle.load(f)
    else:
        glove_vectors = load_glove_vectors()
        with open("data/glove.pickle", "wb") as output:
            pickle.dump(glove_vectors, output)

    with open(args.dataset, 'rb') as fd:
        dataset = pickle.load(fd)
    node_texts = dataset[0]

    nodes_glove = embed_glove(node_texts, glove_vectors)
    with open("data/nodes_glove.pickle", "wb") as output:
        pickle.dump(nodes_glove, output)

    nodes_bert = embed_nodes_bert(node_texts)
    with open("data/nodes_bert.pickle", "wb") as output:
        pickle.dump(nodes_bert, output)
