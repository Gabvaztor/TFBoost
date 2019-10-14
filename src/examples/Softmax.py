"""
This class contains an example of Softmax algorithm
"""
import numpy as np

#TODO Docs

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x),axis=0)

scores = [1.0, 2.0]
print (softmax(scores))
