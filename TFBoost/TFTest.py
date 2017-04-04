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

import TFBoost.TFEasyGui as eg
import TFBoost.TFReader as tfr
import TFBoost.TFDataMining as tfd
from TFBoost.TFEncoder import Dictionary
from UsefulTools.UtilsFunctions import *
import TFBoost.TFModels as models
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

''' TFLearn library. License MIT.
Git Clone : https://github.com/tflearn/tflearn.git
To install: pip install tflearn'''
import tflearn

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- GLOBAL VARIABLES ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

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
pathTrainAndTestImages = [Dictionary.path_signals_university_train,Dictionary.path_signals_university_test]
number_of_classes = 58 # Start in 0
percentagesSets = None  # Example
labelsSet = [Dictionary.labels_type_option_hierarchy]
isAnUniqueCSV = False  # If this variable is true, then only one CSV file will be passed and it will be treated like trainSet, validationSet and testSet
knownDataType = ''  # Contains the type of data if the data file contains an unique type of data. Examples: Number or Chars.

tfReaderFeatures = tfr.ReaderFeatures(set_data_files = pathTrainAndTestImages,number_of_classes = number_of_classes,
                                      labels_set = labelsSet,
                                      is_unique_csv = isAnUniqueCSV,known_data_type = knownDataType,
                                      percentages_sets = percentagesSets)

"""
Creating Reader from ReaderFeatures
"""
tfReader = tfr.Reader(reader_features = tfReaderFeatures)  # Reader Object with all information

"""
Getting train, validation (if necessary) and test set.
"""

trainSet = tfReader.trainSet  # Train Set
testSet = tfReader.testSet  # Test Set

pt('testSet',trainSet[0][0])
pt('trainSet:',trainSet[1][0])


import PIL.Image

#img = PIL.Image.open(trainSet[0][0])

models.convolution_model(input=trainSet,test=testSet[0],
                         input_labels=trainSet[1],test_labels=testSet[1],
                         number_of_classes=number_of_classes)

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- TENSORFLOW SECTION ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""




init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# TODO Make TFModels heritable and with capability to return section of tensorflow code