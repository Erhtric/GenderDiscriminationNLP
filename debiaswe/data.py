import json
import os
import gensim.downloader as api
import itertools
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

"""
Tools for data operations

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

def load_professions():
    professions_file = os.path.join(PKG_DIR, '../data', 'professions.json')
    with open(professions_file, 'r') as f:
        professions = json.load(f)
    print('Loaded professions\n' +
          'Format:\n' +
          'word,\n' +
          'definitional female -1.0 -> definitional male 1.0\n' +
          'stereotypical female -1.0 -> stereotypical male 1.0')
    return professions

def load_text8():
    """
    Loads the text8 dataset and save it in the data folder as a .pkl file.
    """
    name = "text8"
    print(api.info(name)["description"])
    filepath = os.path.join(PKG_DIR, '../data/english_data', 'corpus.pkl')

    if not os.path.isfile(filepath):
        print(api.info(name)["description"])
        dataset = api.load("text8")
        corpus = list(itertools.chain.from_iterable(dataset))
        with open(filepath, "wb+") as f:
            pickle.dump(corpus, f)
    else:
        with open(filepath, 'rb') as f:
            corpus = pickle.load(f)

    return corpus

def load_embeddings_pandas():
    path  = os.path.join(PKG_DIR, '../embeddings', 'vectors300.txt')

    dt = pd.read_csv(path, sep=" ", header=None)
    dt.rename(columns={0: 'word'}, inplace=True)

    return dt

def load_glove_embeddings_pandas():
    path  = os.path.join(PKG_DIR, '../embeddings', 'glove.6B.300d.txt')

    dt = pd.read_csv(path, sep=" ", header=None)
    dt.rename(columns={0: 'word'}, inplace=True)

    return dt