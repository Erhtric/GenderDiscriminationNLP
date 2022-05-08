from importlib.resources import path
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))


def normalized_attribute_set(attribute_set, embedding_dim):
    sum_att_vector = np.zeros(embedding_dim,dtype=object)
    for a in attribute_set:
        normalized_a = a/np.linalg.norm(a)
        sum_att_vector += normalized_a
    return sum_att_vector/len(attribute_set)

def pairwise_bias(first_attribute_set, second_attribute_set, embedded_word):
    normalized_first_set = normalized_attribute_set(first_attribute_set,128)
    normalized_second_set = normalized_attribute_set(second_attribute_set,128)
    
    diff_set = normalized_first_set-normalized_second_set
    bias_value = np.dot(embedded_word,diff_set) / (np.linalg.norm(embedded_word)*np.linalg.norm(diff_set))

    return bias_value

def bias_score(embedded_word_set, first_attribute_set, second_attribute_set):
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

