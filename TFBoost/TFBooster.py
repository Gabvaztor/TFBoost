"""
Author: @gabvaztor
StartDate: 04/03/2017

This file contains the next information:
    - Libraries to import with installation comment and reason.
    - Data Mining Algorithm.
    - Sets (train,validation and test) information.
    - ANN Arquitectures.
    - A lot of utils methods which you'll get useful advantage


The code's structure is:
    - Imports
    - Global Variables
    - Interface
    - Reading data algorithms
    - Data Mining
    - Training and test
    - Show final conclusions

Notes:
    * This file use TensorFlow version 1.0.
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

'''LOCAL IMPORTS
* UtilsFunctions is a library that contains a lot of functions which will help us
to code expressively, clearly and efficiently.
* TensorFlowGUI's library contains all GUI's methods. Contains EasyGUI.
Here you can download the library: https://pypi.python.org/pypi/easygui#downloads
It had been used the version: 0.98.1
'''

import UsefulTools.UtilsFunctions as uf
import TFBoost.TFEasyGui as eg
import TFBoost.TFReader as tfr
import TFBoost.TFDataMining as tfd
from TFBoost.TFEncoder import Dictionary


''' TensorFlow: https://www.tensorflow.org/
To upgrade TensorFlow to last version:
*CPU: pip3 install --upgrade tensorflow
*GPU: pip3 install --upgrade tensorflow-gpu
'''
import tensorflow as tf
print("TensorFlow: " + tf.__version__)


''' Numpy is an extension to the Python programming language, adding support for large,
multi-dimensional arrays and matrices, along with a large library of high-level
mathematical functions to operate on these arrays.
It is mandatory to install 'Numpy+MKL' before scipy.
Install 'Numpy+MKL' from here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
http://www.numpy.org/
https://en.wikipedia.org/wiki/NumPy '''
import numpy as np

'''
# You need to install the 64bit version of Scipy, at least on Windows.
# It is mandatory to install 'Numpy+MKL' before scipy.
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
# We can find scipi in the url: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy'''
import scipy.io as sio

''' Matlab URL: http://matplotlib.org/users/installing.html'''
import matplotlib.pyplot as plt



"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- GLOBAL VARIABLES ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

isAnUniqueCSV = True  # If this variable is true, then only one CSV file will be passed and it will be treated like trainSet, validationSet and testSet
knownDataType = ''  # Contains the type of data if the data file contains an unique type of data. Examples: Number or Chars.

trainSetCSV = ''
validationSetCSV = ''
testSetCSV = ''

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- USER INTERFACE ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

'''Creating user interface'''
#properties = eg.EasyGui()
#uf.pt("Typos GUI",properties.types)

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- READING DATA ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

"""
Creating Reader Features
"""
csvList = []
csvList.append(Dictionary.path_Breast_Cancer_Wisconsin)  # Example
percentagesSets = [0.5,0.3,0.2]  # Example
labelsSet = [Dictionary.label_column_name_Breast_Cancer_Wisconsin]

tfReaderFeatures = tfr.ReaderFeatures(set_data_files = csvList,labels_set = labelsSet,
                                      is_unique_csv = isAnUniqueCSV,known_data_type = knownDataType,
                                      percentages_sets = percentagesSets)

"""
Creating Reader from ReaderFeatures
"""
tfReader = tfr.Reader(reader_features = tfReaderFeatures)  # Reader Object with all information

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- DATA MINING ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

"""
Manipulate Reader with DataMining and update it.
"""
chooses = None  # This variable contains the data mining options. None do nothing
# TODO Define this
tfReader = tfd.DataMining(tfReader,chooses)

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- TENSORFLOW SECTION ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

"""
Getting train, validation (if necessary) and test set.
"""

trainSet = tfReader.trainSet  # Train Set
validationSet = tfReader.validationSet  # Validation Set
testSet = tfReader.testSet  # Test Set

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)