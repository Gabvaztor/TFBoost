"""
Code from:
https://ewanlee.github.io/2017/04/26/LSTM-by-Example-using-Tensorflow-Text-Generate/
"""

import collections
from  UsefulTools.UtilsFunctions import *
import nltk


text = "long ago , the mice had a general council to consider what measures they could take to outwit their common enemy" \
       " , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to " \
       "make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in " \
       "the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her" \
       " approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured ," \
       " and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , " \
       "and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an" \
       " old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another" \
       "and nobody spoke . then the old mouse said it is easy to propose impossible remedies."


def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    values = list(dictionary.values())
    keys = list(dictionary.keys())
    reverse_dictionary = {key: value for (value, key) in zip(keys, values)}
    pt("reverse_dictionary", reverse_dictionary)
    return dictionary, reverse_dictionary

words = set(nltk.word_tokenize(text))
pt("words", words)
fd = nltk.FreqDist(word.lower() for word in words)
fdf= fd.most_common()
dictionary, reverse_dictionary = build_dataset(words)

pt("dictionary",dictionary)
pt("reverse_dictionary",reverse_dictionary)

# Target log path
logs_path = '../rnn_words'
writer = tf.summary.FileWriter(logs_path)

# TODO finish