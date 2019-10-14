"""
From https://github.com/dmesquita/understanding_tensorflow_nn
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from  UsefulTools.UtilsFunctions import pt
text = "Sufro mucho estrés ,"
vocab = Counter()

for word in text.split(' '):
    word_lowercase = word.lower()
    vocab[word_lowercase] += 1

def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i
    return word2index

word2index = get_word_2_index(vocab)

total_words = len(vocab)
matrix = np.zeros((total_words), dtype=float)

for word in text.split():
    matrix[word2index[word.lower()]] += 1

pt(text, matrix)