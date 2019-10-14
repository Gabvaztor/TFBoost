"This file contains all algorithms about API functionality."
import numpy as np

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x))

l = [2,4,6,8,10]
s = softmax(l)
[print(x) for x in s]
print(str(softmax(l)))