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

from UsefulTools.UtilsFunctions import *
import TFBoost.TFModels as models
import SettingsObject
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

""" To get via parameter"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_fullpath", required=False,
                help="Image FullPath to predict")
ap.add_argument("-l", "--image_label_real", required=False,
                help="Image Real Label")

args = vars(ap.parse_args())
image_fullpath_to_predict = args["image_fullpath"]
image_label_real = args["image_label_real"]
input_data = []
input_labels = []
predict_flag = False
if image_fullpath_to_predict:
    input_data = [image_fullpath_to_predict]
    predict_flag = True
if image_label_real:
    input_labels = [image_label_real]
train_set = [image_fullpath_to_predict]
setting_object = SettingsObject.Settings(Dictionary.string_settings_retinopathy_k)
option_problem = Dictionary.string_option_retinopathy_k_problem
options = [option_problem, 1, 720, 1280]
number_of_classes = 5 # Start in 0


models = models.TFModels(setting_object=setting_object, option_problem=options,
                         input_data=input_data,test=None,
                         input_labels=input_labels,test_labels=None,
                         number_of_classes=number_of_classes, type=None,
                         validation=None, validation_labels=None, predict_flag=True)
with tf.device('/cpu:0'):  # CPU
    models.convolution_model_image()

