"""
Author: @gabvaztor
StartDate: 04/03/2017

With this class you can import a lot of labeled data like Kaggle problems.

- This class not preprocessed de data reducing noise.
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

# --------------------------------------------------------------------------
'''

To install pandas: pip3 install pandas
'''
import pandas as pd
# --------------------------------------------------------------------------

class Reader(object):
    types = set()
    def __init__(self):
        self.types.add('Lineal')
        self.types.add('CNN')

if __name__ == '__main__':
    print ("Creating Reader")