from importlib.resources import path
import numpy as np
import argparse
import sys
import os
from pkg_resources import parse_requirements
import scipy
from scipy.spatial import distance
#for testing purpose

sys.path.append('../')
import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_professions


def normalized_attribute_set(attribute_set, embedding_dim):
    sum_att_vector = np.zeros(embedding_dim,dtype=object)
    for a in attribute_set:
        normalized_a = a/np.linalg.norm(a)
        sum_att_vector += normalized_a
    print(len(attribute_set))
    return sum_att_vector/len(attribute_set)

def pairwise_bias(first_attribute_set,second_attribute_set,embedded_word):
    normalized_first_set = normalized_attribute_set(first_attribute_set,128)
    normalized_second_set = normalized_attribute_set(second_attribute_set,128)
    
    diff_set = normalized_first_set-normalized_second_set
    print(type(embedded_word))
    print(type(diff_set))
    bias_value = np.dot(embedded_word,diff_set) / (np.linalg.norm(embedded_word)*np.linalg.norm(diff_set))

    return bias_value

def bias_score(embedded_word_set,first_attribute_set,second_attribute_set):
    bias_score = 0
    for word in embedded_word_set:
        bias_score += np.abs(pairwise_bias(first_attribute_set,second_attribute_set,word))
    
    return (bias_score/len(embedded_word_set))

def skew_bias_score(embedded_word_set,first_attribute_set,second_attribute_set):
    skew_score = 0
    for word in embedded_word_set:
        skew_score += pairwise_bias(first_attribute_set,second_attribute_set,word)
    
    return (skew_score/len(embedded_word_set))

def stereotype_bias_score(embedded_word_set,first_attribute_set,second_attribute_set):
    skew_score = skew_bias_score(embedded_word_set,first_attribute_set,second_attribute_set)
    stereotype_score = 0
    for word in embedded_word_set:
        stereotype_score  += (pairwise_bias(first_attribute_set,second_attribute_set,word) - skew_score)**2

    return (np.sqrt(stereotype_score)/len(embedded_word_set))

if __name__ == "__main__":
    print(sys.path)
    E = WordEmbedding('../embeddings/filtered_itwac128.tsv')
    female_words = ["lei", "donna", "madre", "moglie", "sorella", "femmina"]
    
    male_words = ["lui", "uomo", "padre", "marito", "fratello", "maschio"]


    female_set = [ E.vecs[E.index[w]] for w in female_words ]
    male_set = [ E.vecs[E.index[w]] for w in male_words ]
    words = ["ingegnere","statista","pallacanestro"]
    emb_words = [ E.vecs[E.index[word]] for word in words]
    value = bias_score(emb_words,male_set,female_set)
    value_skew = skew_bias_score(emb_words,male_set,female_set)
    value_single = pairwise_bias(male_set,female_set,emb_words[0])
    print(value_single)
    print(value)
    print(value_skew)
    
