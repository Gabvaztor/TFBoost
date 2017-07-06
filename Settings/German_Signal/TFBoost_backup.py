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

Style: "Google Python Style Guide" 
https://google.github.io/styleguide/pyguide.html

Notes:
    * This file use TensorFlow version >1.0.
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

import TFBoost.TFReader as tfr
import TFBoost.TFDataMining as tfd
from TFBoost.TFEncoder import Dictionary
from UsefulTools.UtilsFunctions import *
import TFBoost.TFModels as models
import SettingsObject


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

''' Matlab URL: http://matplotlib.org/users/installing.html
python -m pip install matplotlib'''
import matplotlib.pyplot as plt

''' TFLearn library. License MIT.
Git Clone : https://github.com/tflearn/tflearn.git
    To install: pip install tflearn'''
import tflearn
'''
 Sklearn(scikit-learn): Simple and efficient tools for data mining and data analysis
 To install: pip install -U scikit-learn
'''
from sklearn.model_selection import train_test_split
"""
To install pandas: pip3 install pandas
"""
import pandas as pd
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- GLOBAL VARIABLES ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- USER INTERFACE ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""Creating user interface
#properties = eg.EasyGui()
#uf.pt("Typos GUI",properties.types)

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- READING DATA ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

"""
Creating Reader Features
"""
setting_object = SettingsObject.Settings(Dictionary.string_settings_german_signal_path)

path_train_and_test_images = [setting_object.train_path,setting_object.test_path]
number_of_classes = 59 # Start in 0
percentages_sets = None  # Example
labels_set = [Dictionary.string_labels_type_option_hierarchy]
is_an_unique_csv = False  # If this variable is true, then only one CSV file will be passed and it will be treated like
# trainSet, validationSet(if necessary) and testSet
known_data_type = ''  # Contains the type of data if the data file contains an unique type of data. Examples: # Number
# or Chars.
"""
Creating Reader Features
"""
reader_features = tfr.ReaderFeatures(set_data_files = path_train_and_test_images,number_of_classes = number_of_classes,
                                      labels_set = labels_set,
                                      is_unique_csv = is_an_unique_csv,known_data_type = known_data_type,
                                      percentages_sets = percentages_sets)
"""
Creating Reader from ReaderFeatures
"""
"""
Creating Reader from ReaderFeatures
"""
tf_reader = tfr.Reader(reader_features = reader_features)  # Reader Object with all information

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

"""
Getting train, validation (if necessary) and test set.
"""
test_set = tf_reader.test_set  # Test Set
train_set = tf_reader.train_set  # Train Set
del reader_features
del tf_reader

option_problem = Dictionary.string_option_signals_images_problem

models = models.TFModels(input=train_set[0],test=test_set[0],
                         input_labels=train_set[1],test_labels=test_set[1],
                         number_of_classes=number_of_classes, setting_object=setting_object,
                         option_problem=option_problem, load_model_configuration=True)
models.convolution_model_image()



