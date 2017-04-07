""" To understand axis
"""
from UsefulTools.UtilsFunctions import *
import numpy as np

p = [[1,2,3]]
p_array = np.asarray(p)
s = [[4,1,1],[0,2,1]]
s_array = np.asarray(s)

pt('p',p_array.shape)
pt('p0',np.argmax(p_array,axis=0))
pt('p1',np.argmax(p_array,axis=1))
pt('s',s_array.shape)
pt('s0',np.argmax(s_array,axis=0))
pt('s1',np.argmax(s_array,axis=1))

print = 'p: (1, 3)' \
        'p0: [0 0 0]' \
        'p1: [2] ' \
        's: (2, 3) ' \
        's0:  [0 1 0]' \
        's1: [0 1]'