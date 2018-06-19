# -*- coding: utf-8 -*-
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
#import tensorflow as tf
#print("TensorFlow: " + tf.__version__)


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
python -m pip3 install matplotlib'''
import matplotlib.pyplot as plt

''' TFLearn library. License MIT.
Git Clone : https://github.com/tflearn/tflearn.git
    To install: pip3 install tflearn'''
import tflearn
'''
 Sklearn(scikit-learn): Simple and efficient tools for data mining and data analysis
 To install: pip3 install -U scikit-learn
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
option_problem = Dictionary.string_option_retinopathy_k_problem
setting_object = SettingsObject.Settings(Dictionary.string_settings_retinopathy_k)
options = [option_problem, 1, 720, 1280]
path_train_and_test_images = [setting_object.train_path, setting_object.test_path]
number_of_classes = 5 # Start in 0
percentages_sets = None  # Example
labels_set = [Dictionary.string_labels_type_option_hierarchy]
is_an_unique_csv = False  # If this variable is true, then only one CSV file will be passed and it will be treated like
# trainSet, validationSet(if necessary) and testSet
known_data_type = ''  # Contains the type of data if the data file contains an unique type of data. Examples: # Number
# or Chars.

# TODO (@gabvaztor) Check if file exist automatically
load_dataset = True
if load_dataset:
    path_to_load = setting_object.saved_dataset_path
    x_train_string = "x_train.npy"
    y_train_string = "y_train.npy"
    x_test_string = "x_test.npy"
    y_test_string = "y_test.npy"

    x_train = np.load(file=path_to_load + x_train_string)
    y_train = np.load(file=path_to_load + y_train_string)
    x_test = np.load(file=path_to_load + x_test_string)
    y_test = np.load(file=path_to_load + y_test_string)

else:
    """
    Creating Reader Features
    """
    reader_features = tfr.ReaderFeatures(set_data_files = path_train_and_test_images, number_of_classes = number_of_classes,
                                          labels_set = labels_set,
                                          is_unique_csv = is_an_unique_csv, known_data_type = known_data_type,
                                          percentages_sets = percentages_sets)

    """
    Creating Reader from ReaderFeatures
    """

    tf_reader = tfr.Reader(type_problem=option_problem, reader_features=reader_features,
                           settings=setting_object)  # Reader Object with all information

    x_train = tf_reader.x_train
    y_train = tf_reader.y_train
    x_test = tf_reader.x_test
    y_test = tf_reader.y_test

pt("x_train", x_train.shape)
pt("y_train", y_train.shape)
pt("x_test", x_test.shape)
pt("y_test", y_test.shape)

models = models.TFModels(setting_object=setting_object, option_problem=options,
                         input_data=x_train,test=x_test,
                         input_labels=y_train,test_labels=y_test,
                         number_of_classes=number_of_classes, type=None,
                         validation=None, validation_labels=None,
                         load_model_configuration=False)
#with tf.device('/cpu:0'):  # CPU
with tf.device('/gpu:0'):  # GPU
    models.convolution_model_image()