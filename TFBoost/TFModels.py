#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: @gabvaztor
StartDate: 04/03/2017

This file contains samples and overrides deep learning algorithms.

Style: "Google Python Style Guide"
https://google.github.io/styleguide/pyguide.html

"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

'''LOCAL IMPORTS'''
from UsefulTools.UtilsFunctions import *
from UsefulTools.TensorFlowUtils import *
from TFBoost.TFEncoder import Dictionary as dict
from TFBoost.TFEncoder import Constant as const
from UsefulTools.Prediction import *
import SettingsObject


''' TensorFlow: https://www.tensorflow.org/
To upgrade TensorFlow to last version:
*CPU: pip3 install --upgrade tensorflow
*GPU: pip3 install --upgrade tensorflow-gpu
'''
import tensorflow as tf

# noinspection PyUnresolvedReferences
print("TensorFlow: " + tf.__version__)

''' Numpy is an extension to the Python programming language, adding support for large,
multi-dimensional arrays and matrices, along with a large library of high-level
mathematical functions to operate on these arrays.
It is mandatory to install 'Numpy+MKL' before scipy.
Install 'Numpy+MKL' from here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
http://www.numpy.org/
https://en.wikipedia.org/wiki/NumPy '''
import numpy as np

''' Matlab URL: http://matplotlib.org/users/installing.html'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

''' Pillow URL: https://pillow.readthedocs.io/en/5.1.x/
Problem with OpenCV on Raspbian. Installed Pillow. '''

import PIL.Image as Image
import PIL

''' TFLearn library. License MIT.
Git Clone : https://github.com/tflearn/tflearn.git
To install: pip3 install tflearn'''
import tflearn

'''"Best image library"
pip3 install opencv-python
Imported by petition (variable flag) beacuse of problem on raspberry. Used PILLOW instead'''
# import cv2

"""Python libraries"""
""" Random to shuffle lists """
import random

""" Time """
import time
""" Datetime"""
import datetime

""" To serialize object"""
import json

""" To print stacktrace"""
import traceback

""" To work with numbers"""
import numbers

""" To work with types"""
import types

""" To recollect python rash"""
import gc

global global_function
global global_metadata

class TFModels():
    """
    Long Docs ...
    """
    # TODO (@gabvaztor) Docs
    def __init__(self, setting_object, option_problem, input_data=None, test=None, input_labels=None, test_labels=None,
                 number_of_classes=None , type=None, validation=None, validation_labels=None,predict_flag=False):
        # TODO (@gabvaztor) Show and save graphs during all training asking before
        # TODO (@gabvaztor) Run some operations in other python execution or multiprocessing
        # NOTE: IF YOU LOAD_MODEL_CONFIGURATION AND CHANGE SOME TENSORFLOW ATTRIBUTE AS NEURONS, THE TRAIN WILL START
        # AGAIN
        self._input = input_data
        self._validation = validation
        self._test = test
        self._input_labels = input_labels
        self._validation_labels = validation_labels
        self._test_labels = test_labels
        self._number_of_classes = number_of_classes
        self._settings_object = setting_object  # Setting object represent a kaggle configuration
        self._input_batch = None
        self._label_batch = None
        # Parallel processes
        self._processes = []
        # CONFIGURATION VARIABLES
        self._debug_level = 0 # TODO (@gabvaztor) Explain debug levels
        self._restore_model = False # Labels and logits info. Load only to continue training.
        self._restore_model_configuration = self.restore_model  # By defect, use restore_model value. This, load variables from configuration file.
        self._restore_to_predict = predict_flag  # Load pretrained model to do a prediction. Restrictive
        self._save_model_information = True  # If must to save model or not
        self._ask_to_save_model_information = False  # If True and 'save_model' is true, ask to save model each time
        # 'should_save'
        self._show_when_save_information = False  # If True then you will see printed in console when during training
        # the information.json has been saved.
        self._ask_to_continue_creating_model_without_exist = False  # If True and 'restore_model' is True,
        # ask to continues save model at first if there isn't a model to restore
        self._show_advanced_info = False  # Labels and logits info.
        self._show_images = False  # If True show images when show_info is True
        self._save_model_configuration = True  # If True, then all attributes will be saved in a settings_object path.
        self._shuffle_data = True  # If True, then the train and validation data will be shuffled separately.
        self._generate_predictions = False  # If true, it tries to generate a prediction
        self._save_graphs_images = False  # If True, then save graphs images from statistical values. NOTE that this will
        # decrease the performance during training. Although this is true or false, for each time an epoch has finished,
        # the framework will save a graph
        # TRAIN MODEL VARIABLES
        self._input_rows_numbers = option_problem[2] # For example, in german problem, number of row pixels
        self._input_columns_numbers = option_problem[3]  # For example, in german problem, number of column pixels
        self._kernel_size = [7, 7]  # Kernel patch size
        self._epoch_numbers = 5  # Epochs number
        self._batch_size = 9  # Batch size
        if self.input is not None and not self.restore_to_predict:  # Change if necessary
            self._input_size = self.input.shape[0]  # Change if necessary
            self._trains = int(self.input_size / self.batch_size) + 1  # Total number of trains for epoch
        else:
            self._input_size = None  # Change if necessary
            self._trains = None  # Total number of trains for epoch
        if self.validation is not None:
            self._validation_size = validation.shape[0] # Change if necessary
        else:
            self._validation_size = None
        if self.test is not None:
            self._test_size = len(test) # Change if necessary
        else:
            self._test_size = None
        self._train_dropout = 0.5  # Keep probably to dropout to avoid overfitting
        self._first_label_neurons = 8
        self._second_label_neurons = 16
        self._third_label_neurons = 16
        self._fourth_label_neurons = 32
        self._learning_rate = 1e-3  # Learning rate
        self._number_epoch_to_change_learning_rate = 60  #You can choose a number to change the learning rate. Number
        # represent the number of epochs before be changed.
        self._print_information = 100  # How many trains are needed to print information
        # INFORMATION VARIABLES
        self._index_buffer_data = 0  # The index for mini_batches during training. Start at zero.
        self._num_trains_count = 1  # Start at one
        self._num_epochs_count = 1  # Start at one
        self._train_accuracy = None
        self._validation_accuracy = None
        self._test_accuracy = None
        self._train_loss = None
        self._validation_loss = None
        self._test_loss = None
        self._problem_information = "Accuracy represent error. Low is better"
        # OPTIONS
        # Options represent a list with this structure:
        #               - First position: "string_option" --> unique string to represent problem in question
        #               - Others positions: all variables you need to process each input and label elements
        self._options = option_problem
        # RESTART TRAINING
        self._save_and_restart = False  # All history and metadata will be saved in a different folder and the execution
        # will be restarted
        if self.save_and_restart and not self.restore_to_predict:
            save_and_restart(self.settings_object.model_path)
        # SAVE AND LOAD MODEL
        # If load_model_configuration is True, then it will load a configuration from settings_object method
        if self.restore_model_configuration and not self.restore_to_predict:
            # And restore time too.
            if self.restore_model:
                # input("You will load model configuration but no restore the tensorflow model, do you want to continue?")
                pt("Loading model configuration", self.settings_object.configuration_path)
                self._load_model_configuration(self.settings_object.load_actual_configuration())
        if self.save_model_configuration and not self.restore_to_predict:
            # Save model configuration in a json file
            self._save_json_configuration(Constant.attributes_to_delete_configuration)

    @property
    def problem_information(self):
        return self._problem_information

    @problem_information.setter
    def problem_information(self, value):
        self._problem_information = value

    @property
    def validation_size(self):
        return self._validation_size

    @validation_size.setter
    def validation_size(self, value):
        self._validation_size = value

    @property
    def print_information(self):
        return self._print_information

    @print_information.setter
    def print_information(self, value):
        self._print_information = value

    @property
    def validation(self):
        return self._validation

    @validation.setter
    def validation(self, value):
        self._validation = value

    @property
    def validation_labels(self):
        return self._validation_labels

    @validation_labels.setter
    def validation_labels(self, value):
        self._validation_labels = value

    @property
    def show_when_save_information(self):
        return self._show_when_save_information

    @show_when_save_information.setter
    def show_when_save_information(self, value):
        self._show_when_save_information = value

    @property
    def ask_to_continue_creating_model_without_exist(self):
        return self._ask_to_continue_creating_model_without_exist

    @ask_to_continue_creating_model_without_exist.setter
    def ask_to_continue_creating_model_without_exist(self, value):
        self._ask_to_continue_creating_model_without_exist = value

    @property
    def number_epoch_to_change_learning_rate(self):
        return self._number_epoch_to_change_learning_rate

    @number_epoch_to_change_learning_rate.setter
    def number_epoch_to_change_learning_rate(self, value):
        self._number_epoch_to_change_learning_rate = value

    @property
    def save_and_restart(self):
        return self._save_and_restart

    @save_and_restart.setter
    def save_and_restart(self, value):
        self._save_and_restart = value

    @property
    def num_epochs_count(self):
        return self._num_epochs_count

    @num_epochs_count.setter
    def num_epochs_count(self, value):
        self._num_epochs_count = value

    @property
    def save_graphs_images(self):
        return self._save_graphs_images

    @save_graphs_images.setter
    def save_graphs_images(self, value):
        self._save_graphs_images = value

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, value):
        self._options = value

    @property
    def input_batch(self):
        return self._input_batch

    @input_batch.setter
    def input_batch(self, value):
        self._input_batch = value

    @property
    def label_batch(self):
        return self._label_batch

    @label_batch.setter
    def label_batch(self, value):
        self._label_batch = value

    @property
    def show_advanced_info(self):
        return self._show_advanced_info

    @show_advanced_info.setter
    def show_advanced_info(self, value):
        self._show_advanced_info = value

    @property
    def save_model_information(self):
        return self._save_model_information

    @save_model_information.setter
    def save_model_information(self, value):
        self._save_model_information = value

    @property
    def save_model_configuration(self):
        return self._save_model_configuration

    @save_model_configuration.setter
    def save_model_configuration(self, value):
        self._save_model_configuration = value

    @property
    def ask_to_save_model_information(self):
        if self._save_model_information:
            return self._ask_to_save_model_information
        else:
            return False

    @ask_to_save_model_information.setter
    def ask_to_save_model_information(self, value):
        self._ask_to_save_model_information = value

    @property
    def restore_model(self):
        return self._restore_model

    @restore_model.setter
    def restore_model(self, value):
        self._restore_model = value

    @property
    def train_accuracy(self):
        return self._train_accuracy

    @train_accuracy.setter
    def train_accuracy(self, value):
        self._train_accuracy = value

    @property
    def test_accuracy(self):
        return self._test_accuracy

    @test_accuracy.setter
    def test_accuracy(self, value):
        self._test_accuracy = value

    @property
    def validation_accuracy(self):
        return self._validation_accuracy

    @validation_accuracy.setter
    def validation_accuracy(self, value):
        self._validation_accuracy = value

    @property
    def settings_object(self):
        return self._settings_object

    @settings_object.setter
    def settings_object(self, value):
        self._settings_object = value

    @property
    def learning_rate(self):
        return float("{0:.64f}".format(self._learning_rate))

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def show_images(self):
        return self._show_images

    @show_images.setter
    def show_images(self, value):
        self._show_images = value

    @property
    def shuffle_data(self):
        return self._shuffle_data

    @shuffle_data.setter
    def shuffle_data(self, value):
        self._shuffle_data = value

    @property
    def input_rows_numbers(self):
        return self._input_rows_numbers

    @input_rows_numbers.setter
    def input_rows_numbers(self, value):
        self._input_rows_numbers = value

    @property
    def input_columns_numbers(self):
        return self._input_columns_numbers

    @input_columns_numbers.setter
    def input_columns_numbers(self, value):
        self._input_columns_numbers = value

    @property
    def input_columns_after_reshape(self):
        return self.input_rows_numbers * self.input_columns_numbers

    @input_columns_after_reshape.setter
    def input_columns_after_reshape(self, value):
        self.input_columns_after_reshape = value

    @property
    def input_rows_columns_array(self):
        return [self.input_rows_numbers, self.input_columns_numbers]

    @input_rows_columns_array.setter
    def input_rows_columns_array(self, value):
        self.input_rows_columns_array = value

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        self._kernel_size = value

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        self._test_size = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def train_dropout(self):
        return self._train_dropout

    @train_dropout.setter
    def train_dropout(self, value):
        self._train_dropout = value

    @property
    def index_buffer_data(self):
        return self._index_buffer_data

    @index_buffer_data.setter
    def index_buffer_data(self, value):
        self._index_buffer_data = value

    @property
    def first_label_neurons(self):
        return self._first_label_neurons

    @first_label_neurons.setter
    def first_label_neurons(self, value):
        self._first_label_neurons = value

    @property
    def second_label_neurons(self):
        return self._second_label_neurons

    @second_label_neurons.setter
    def second_label_neurons(self, value):
        self._second_label_neurons = value

    @property
    def third_label_neurons(self):
        return self._third_label_neurons

    @third_label_neurons.setter
    def third_label_neurons(self, value):
        self._third_label_neurons = value

    @property
    def fourth_label_neurons(self):
        return self._fourth_label_neurons

    @fourth_label_neurons.setter
    def fourth_label_neurons(self, value):
        self._fourth_label_neurons = value

    @property
    def trains(self):
        return self._trains

    @trains.setter
    def trains(self, value):
        self._trains = value

    @property
    def num_trains_count(self):
        return self._num_trains_count

    @num_trains_count.setter
    def num_trains_count(self, value):
        self._num_trains_count = value

    @property
    def number_of_classes(self):
        return self._number_of_classes

    @number_of_classes.setter
    def number_of_classes(self, value):
        self._number_of_classes = value

    @property
    def input_labels(self):
        return self._input_labels

    @input_labels.setter
    def input_labels(self, value):
        self._input_labels = value

    @property
    def test_labels(self):
        return self._test_labels

    @test_labels.setter
    def test_labels(self, value):
        self._test_labels = value

    @property
    def epoch_numbers(self):
        return self._epoch_numbers

    @epoch_numbers.setter
    def epoch_numbers(self, value):
        self._epoch_numbers = value

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    @property
    def restore_to_predict(self):
        return self._restore_to_predict

    @restore_to_predict.setter
    def restore_to_predict(self, value):
        self._restore_to_predict = value

    @property
    def debug_level(self):
        return self._debug_level

    @debug_level.setter
    def debug_level(self, value):
        self._debug_level = value

    @property
    def train_loss(self):
        return self._train_loss

    @train_loss.setter
    def train_loss(self, value):
        self._train_loss = value

    @property
    def validation_loss(self):
        return self._validation_loss

    @validation_loss.setter
    def validation_loss(self, value):
        self._validation_loss = value

    @property
    def test_loss(self):
        return self._test_loss

    @test_loss.setter
    def test_loss(self, value):
        self._test_loss = value

    @property
    def restore_model_configuration(self):
        return self._restore_model_configuration

    @restore_model_configuration.setter
    def restore_model_configuration(self, value):
        self._restore_model_configuration = value

    def _save_json_configuration(self, attributes_to_delete_configuration):
        try:
            self._save_model_to_json(self.settings_object.configuration_path,
                                     attributes_to_delete_configuration,
                                     type_file="Configuration")
            """
            f = self._save_model_to_json(self.settings_object.configuration_path,
                                                   attributes_to_delete_configuration,
                                                   type_file="Configuration")
                                                   """
            """
            p = multiprocessing.Process(target=self._save_model_to_json, args=(self.settings_object.configuration_path,
                                                                           attributes_to_delete_configuration,
                                                                           "Configuration"))
            
            #import Asynchronous
            #Asynchronous.execte_asynchronous_process(functions=f, arguments=None)

            global global_function
            global global_metadata
            global_function = self._save_model_to_json
            global_metadata = (self.settings_object.configuration_path, attributes_to_delete_configuration, "Configuration")
            global_metadata = (self)

            import Asynchronous
            pass
            """
        except Exception as e:
            pt(Errors.error, e)
            traceback.print_exc()
            pass

    @timed
    def convolution_model_image(self):
        """
        Generic convolutional model
        """
        # Print actual configuration
        self.print_actual_configuration()
        # TODO Try python EVAL method to do multiple variable neurons
        #with tf.device('/cpu:0'):  # CPU
        with tf.device('/gpu:0'):  # GPU
            # Placeholders
            x_input, y_labels, keep_probably = self.placeholders(args=None, kwargs=None)
            # Reshape x placeholder into a specific tensor
            x_reshape = tf.reshape(x_input, [-1, self.input_rows_numbers, self.input_columns_numbers, 1])
            # Network structure
            y_prediction = self.network_structure(x_reshape, args=None, keep_probably=keep_probably)
            cross_entropy, train_step, correct_prediction, accuracy = self.model_evaluation(y_labels=y_labels,
                                                                                            y_prediction=y_prediction)
            # Session
            sess = initialize_session(self.debug_level)
            # Saver session
            saver = tf.train.Saver()  # Saver
            # Batching values and labels from input and labels (with batch size)
            if not self.restore_to_predict:
                self.update_batch(create_dataset_flag=False)
                # To restore model
                if self.restore_model:
                    self.load_and_restore_model(sess)
                # TODO (@gabvaztor) When problem requires cross validation with train and test, do it during training.
                # Besides this, when test/validation set requires check its accuracy but its size is very long to save
                # in memory, it has to update all files during training to get the exact precision.
                self.train_model(args=None, kwargs=locals())
            else:
                self.prediction(x_input=x_input, y_prediction=y_prediction, keep_probably=keep_probably, sess=sess)

    def prediction(self, x_input, y_prediction, keep_probably, sess):
        try:
            input_path = self.input[0]
            label = 99
            if self.input_labels is not None and self.input_labels:
                label = self.input_labels[0]
            x_input_pred, real_label = process_input_unity_generic(input_path, label, self.options)
            if label == 99:
                real_label = None
            else:
                real_label = int(np.argmax(real_label))
            self.test_prediction(sess=sess, x_input_tensor=x_input, y_prediction=y_prediction,
                                                  x_input_pred=x_input_pred, keep_probably=keep_probably,
                                                  real_label=real_label, input_path=input_path)
        except Exception as err:
            self.write_log_error(err)

    def write_log_error(self, err):
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()  # this is to get error line number and description.
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]  # to get File Name.
        error_string = "ERROR : Error Msg:{},File Name : {}, Line no : {}\n".format(err, file_name,
                                                                                    exc_tb.tb_lineno)
        pt(error_string)
        file_log = open("python_prediction_error_log.log", "a")
        file_log.write(error_string + "\n\n" + str(err))
        file_log.close()

    def test_prediction(self, sess, x_input_tensor, y_prediction, x_input_pred, keep_probably, real_label=None,
                        input_path=None):
        start_time_load_model = datetime.datetime.now()
        # Restore model
        self.load_and_restore_model(session=sess)
        pt('Time to load model', (datetime.datetime.now()- start_time_load_model).total_seconds())
        if self.debug_level > 0:
            pt("x_input", x_input_tensor)
            pt("x_input.shape", x_input_tensor.shape)
            pt("x_input_pred", x_input_pred)
            pt("x_input_pred", x_input_pred.shape)
        x_input_pred = np.asarray([x_input_pred])
        feed_dict_prediction = {x_input_tensor: x_input_pred, keep_probably: 1.0}
        i = 0
        if x_input_pred is not None:
            while i < 15:
                start_datetime_ = datetime.datetime.now()
                i += 1
                prediction = y_prediction.eval(feed_dict=feed_dict_prediction)
                pt("Prediction " + str(i), np.argmax(prediction))
                if real_label:
                    pt("Real Label", real_label)
                delta = datetime.datetime.now() - start_datetime_
                pt("Time to do inference " + str(i), delta.total_seconds())
            path_saved = None
            information = "German Signal prediction"
            try:
                prediction_class = GermanSignal(information=information, real_label=real_label,
                                                image_fullpath=input_path,
                                                prediction_label=int(np.argmax(prediction)))
                prediction_class.save_json(save_fullpath=self.settings_object.submission_path)
            except Exception as e:
                pt(Errors.error, e)
                raise ValueError("Can not save prediction json")
        else:
            raise ValueError("Can not predict the test")
        return path_saved

    def update_inputs_and_labels_shuffling(self, inputs, inputs_labels):
        """
        Update inputs_processed and labels_processed variables with an inputs and inputs_labels shuffled
        :param inputs: Represent input data
        :param inputs_labels:  Represent labels data
        """
        c = list(zip(inputs, inputs_labels))
        random.shuffle(c)
        self.inputs_processed, self.labels_processed = zip(*c)

    def data_buffer_generic_class(self, inputs, inputs_labels, shuffle_data=False, batch_size=None, is_test=False,
                                  options=None, create_dataset_flag=False):
        """
        Create a data buffer having necessaries class attributes (inputs,labels,...)
        :param inputs: Inputs
        :param inputs_labels: Inputs labels
        :param shuffle_data: If it is necessary shuffle data.
        :param batch_size: The batch size.
        :param is_test: if the inputs are the test set.
        :param options: options
        :return: Two numpy arrays (x_batch and y_batch) with input data and input labels data batch_size like shape.
        """
        x_batch = []
        y_batch = []
        if is_test:
            # TODO (@gabvaztor) Create process set to create new datasets
            x_batch, y_batch = process_test_set(inputs, inputs_labels, options, create_dataset_flag=create_dataset_flag)
        else:
            if shuffle_data and self.index_buffer_data == 0:
                self.input, self.input_labels = get_inputs_and_labels_shuffled(self.input, self.input_labels)
            else:
                self.input, self.input_labels = self.input, self.input_labels  # To modify if is out class
            batch_size, out_range = self.get_out_range_and_batch()  # out_range will be True if
            # next batch is out of range
            for _ in range(batch_size):
                x, y = process_input_unity_generic(self.input[self.index_buffer_data],
                                                   self.input_labels[self.index_buffer_data],
                                                   options)
                x_batch.append(x)
                y_batch.append(y)
                self.index_buffer_data += 1
            x_batch = np.asarray(x_batch)
            y_batch = np.asarray(y_batch)
            if out_range:  # Reset index_buffer_data
                self.index_buffer_data = 0
        return x_batch, y_batch

    def get_out_range_and_batch(self):
        """
        Return out_range flag and new batch_size if necessary. It is necessary when batch is bigger than input rest of
        self.index_buffer_data
        :return: out_range (True or False), batch_size (int)
        """
        out_range = False
        batch_size = self.batch_size
        if self.input_size - self.index_buffer_data == 0:  # When is all inputs
            out_range = True
        elif self.input_size - self.index_buffer_data < self.batch_size:
            batch_size = self.input_size - self.index_buffer_data
            out_range = True
        return batch_size, out_range

    def should_save(self, saves_information_list=None, check_loss_train=False, if_is_equal=True):
        """
        Check if must save from validation/test accuracy/error

        :return: if should save
        """
        # TODO (@gabvaztor) Detect when stop learning. From 60% to 10% validation/test
        should_save = False
        save_for_information = True
        if saves_information_list:
            if saves_information_list.count(1) / 2 >= 25:
                save_for_information = False
            if len(saves_information_list) >= 50:
                del saves_information_list[0]
        if self.save_model_information and save_for_information:
            actual_information = self.settings_object.load_actual_information()
            if actual_information:
                last_train_accuracy = actual_information._train_accuracy
                last_test_accuracy = actual_information._test_accuracy
                last_validation_accuracy = actual_information._validation_accuracy
                if last_train_accuracy and last_validation_accuracy and not self.ask_to_save_model_information:
                    # TODO(@gabvaztor) Check when, randomly, gradient descent obtain high accuracy
                    if self.validation_accuracy and last_validation_accuracy:
                        if if_is_equal:
                            if self.validation_accuracy >= last_validation_accuracy:  # Save checking validation
                                #  accuracies in this moment
                                should_save = True
                        elif self.validation_accuracy > last_validation_accuracy:
                            should_save = True
                elif last_train_accuracy and last_test_accuracy and not self.ask_to_save_model_information:
                    # TODO(@gabvaztor) Check when, randomly, gradient descent obtain high accuracy
                    if self.test_accuracy and last_test_accuracy:
                        if if_is_equal:
                            # TODO (@gabvaztor) Sometimes, gradient break and always obtain same test. Fix it. (restart
                            # learning)
                            if self.test_accuracy >= last_test_accuracy:  # Save checking test
                                #  accuracies in this moment
                                should_save = True
                        elif self.test_accuracy > last_test_accuracy:
                                should_save = True
                elif check_loss_train:
                    if self.num_trains_count % 50 == 0:
                        should_save = True
                else:
                    if self.ask_to_save_model_information:
                        pt("last_train_accuracy", last_train_accuracy)
                        pt("last_test_accuracy", last_test_accuracy)
                        pt("last_validation_accuracy", last_validation_accuracy)
                        pt("actual_train_accuracy", self.train_accuracy)
                        pt("actual_test_accuracy", self.test_accuracy)
                        pt("actual_validation_accuracy", self.validation_accuracy)
                        option_choosed = recurrent_ask_to_save_model()
                    else:
                        option_choosed = True
                    if option_choosed:
                        should_save = True
            else:
                should_save = True
        if should_save and saves_information_list:
            saves_information_list.append(1)
        elif not should_save and saves_information_list:
            saves_information_list.append(0)
        return should_save

    def _load_model_configuration(self, configuration):
        """
        Load previous configuration to class Model (self).

        This will update all class' attributes with the configuration in a json file.

        If configuration is None, the file will be created after this method if save_configuration attribute is True
        :param configuration: the json class
        """
        if configuration:
            # TODO Add to docs WHEN it is necessary to add more attributes = Do documentation
            if not self.restore_model:
                self.restore_model = configuration._restore_model
            self.save_model = configuration._save_model_information
            self.ask_to_save_model = configuration._ask_to_save_model_information
            self.show_info = configuration._show_advanced_info
            self.show_images = configuration._show_images
            self.save_model_configuration = configuration._save_model_configuration
            self.save_model_information = configuration._save_model_information
            self.shuffle_data = configuration._shuffle_data
            self.input_rows_numbers = configuration._input_rows_numbers
            self.input_columns_numbers = configuration._input_columns_numbers
            self.kernel_size = configuration._kernel_size
            self.epoch_numbers = configuration._epoch_numbers
            self.batch_size = configuration._batch_size
            self.input_size = configuration._input_size
            self.test_size = configuration._test_size
            self.train_dropout = configuration._train_dropout
            self.first_label_neurons = configuration._first_label_neurons
            self.second_label_neurons = configuration._second_label_neurons
            self.third_label_neurons = configuration._third_label_neurons
            self.learning_rate = configuration._learning_rate
            self.trains = configuration._trains
            self.number_epoch_to_change_learning_rate = configuration._number_epoch_to_change_learning_rate
            self.save_graphs_images = configuration._save_graphs_images
            self.ask_to_continue_creating_model_without_exist = \
                configuration._ask_to_continue_creating_model_without_exist
            self.ask_to_save_model_information = configuration._ask_to_save_model_information
            self.show_when_save_information = configuration._show_when_save_information
            self.print_information = configuration._print_information
            self.validation_size = configuration._validation_size
            self.problem_information = configuration._problem_information
            self.restore_to_predict = configuration._restore_to_predict
            self.debug_level = configuration._debug_level
            self.fourth_label_neurons = configuration._fourth_label_neurons
            self.restore_model_configuration = configuration._restore_model_configuration
            self.train_loss = configuration._train_loss
            self.test_loss = configuration._test_loss
            self.validation_loss = configuration._validation_loss
            # If you don't restore model then you won't load train number and epochs number
            if self.restore_model:
                self.num_trains_count = configuration._num_trains_count
                self.num_epochs_count = configuration._num_epochs_count
                self.index_buffer_data = configuration._index_buffer_data
            pt("Loaded model configuration")

    def _save_model_to_json(self, fullpath, attributes_to_delete=None, *args, **kwargs):
        """
        Save actual model configuration (with some attributes) in a json file.
        :param attributes_to_delete: represent witch attributes set must not be save in json file.
        """
        type_file = kwargs["type_file"]
        accuracy = ""
        if "accuracy" in kwargs:
            accuracy = kwargs["accuracy"]
        filepath = ""
        try:
            pt("Saving model" + type_file + " ... DO NOT STOP PYTHON PROCESS")
            json = object_to_json(object=self, attributes_to_delete=attributes_to_delete)
            write_string_to_pathfile(json, fullpath)
            filepath = create_historic_folder(fullpath, type_file, accuracy)
            write_string_to_pathfile(json, filepath)
            pt("Model " + type_file + " has been saved")
        except Exception as e:
            pt("Can not get json from class to save " + type_file + " file.")
            pt("Do you have float32? (Probably you need numpy float64 or int) Be careful with data types.")
            pt(Errors.error, e)
            traceback.print_exc()
        return filepath

    def load_and_restore_model(self, session):
        """
        Restore a tensorflow model from a model_path checking if model_path exists and create if not.
        :param session: Tensorflow session
        """
        if self.settings_object.model_path:
            pt("Restoring model...", self.settings_object.model_path)
            try:
                # TODO (@gabvaztor) Do Generic possibles models
                model_possible_1 = self.settings_object.model_path + Dictionary.string_ckpt_extension
                model_possible_2 = model_possible_1 + Dictionary.string_meta_extension
                model_possible_3 = model_possible_1 + Dictionary.string_ckpt_extension
                model_possible_4 = model_possible_3 + Dictionary.string_meta_extension
                possibles_models = [model_possible_1, model_possible_2, model_possible_3, model_possible_4]
                model = [x for x in possibles_models if file_exists_in_path_or_create_path(x)]
                if model:
                    saver = tf.train.import_meta_graph(model[0])
                    # Restore variables from disk.
                    saver.restore(session, model_possible_1)
                    pt("Model restored without problems")
                else:
                    if self.ask_to_continue_creating_model_without_exist:
                        response = recurrent_ask_to_continue_without_load_model()
                        if not response:
                            raise Exception()
                    else:
                        pt("The model won't load because it doesn't exist",
                           "You chose 'continue_creating_model_without_exist")
            except Exception as e:
                pt(Errors.error, e)
                raise Exception(Errors.error + " " + Errors.can_not_restore_model)

    def placeholders(self, *args, **kwargs):
        """
        This method will contains all TensorFlow code about placeholders (variables which will be modified during
        process)
        :return: Inputs, labels and others placeholders
        """
        # Placeholders
        #x = tf.placeholder(tf.float32, shape=[None, self.input_columns_after_reshape])  # All images will be 24*24 = 574
        x = tf.placeholder(tf.float32, shape=[None, self.input_rows_numbers, self.input_columns_numbers, 3])  # All images will be 24*24 = 574
        y_ = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])  # Number of labels
        keep_probably = tf.placeholder(tf.float32)  # Value of dropout. With this you can set a value for each data set
        return x, y_, keep_probably

    def network_structure(self, input, *args, **kwargs):
        """
        This method will contains all TensorFlow code about your network structure.
        :param input: inputs
        :return: The prediction (network output)
        """
        keep_dropout = kwargs['keep_probably']
        # First Convolutional Layer
        convolution_1 = tf.layers.conv2d(
            inputs=input,
            filters=self.first_label_neurons,
            kernel_size=self.kernel_size,
            padding="same")
        # Pool Layer 1 and reshape images by 2
        pool1 = tf.layers.max_pooling2d(inputs=convolution_1,
                                        pool_size=[2, 2],
                                        strides=2,
                                        padding="same")
        dropout1 = tf.nn.dropout(pool1, keep_dropout)
        # Second Convolutional Layer
        convolution_2 = tf.layers.conv2d(
            inputs=dropout1,
            filters=self.second_label_neurons,
            kernel_size=self.kernel_size,
            padding="same",
            activation=tf.nn.relu)
        # # Pool Layer 2 nd reshape images by 2
        pool2 = tf.layers.max_pooling2d(inputs=convolution_2,
                                        pool_size=[2, 2],
                                        strides=2,
                                        padding="same")
        dropout2 = tf.nn.dropout(pool2, keep_dropout)

        """
        convolution_3 = tf.layers.conv2d(
            inputs=dropout2,
            filters=self.third_label_neurons,
            kernel_size=self.kernel_size,
            padding="same")

        dropout3 = tf.nn.dropout(convolution_3, keep_dropout)
        """
        # Dense Layer
        # TODO Checks max pools numbers
        pool2_flat = tf.reshape(dropout2, [-1, int(int(self._input_rows_numbers / 1) * int(self._input_columns_numbers / 1) * 3)])
        dense = tf.layers.dense(inputs=pool2_flat, units=self.fourth_label_neurons)
        #dropout = tf.nn.dropout(dense, keep_dropout)
        # Readout Layer
        w_fc2 = weight_variable([self.fourth_label_neurons, self.number_of_classes])
        b_fc2 = bias_variable([self.number_of_classes])
        y_convolution = (tf.matmul(dense, w_fc2) + b_fc2)
        return y_convolution

    def model_evaluation(self, y_labels, y_prediction, *args, **kwargs):
        """
        This methods will contains all TensorFlow about model evaluation.
        :param y_labels: Labels
        :param y_prediction: The prediction
        :return: The output must contains all necessaries variables that it used during training
        """
        # Evaluate model
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_labels,
                                                    logits=y_prediction))

        #train_step = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(cross_entropy)  # Adadelta Optimizer
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)  # Adam Optimizer
        #train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)  # Adam Optimizer

        # Sure is axis = 1
        correct_prediction = tf.equal(tf.argmax(y_prediction, axis=1),
                                      tf.argmax(y_labels, axis=1))  # Get Number of right values in tensor
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Get accuracy in float

        return cross_entropy, train_step, correct_prediction, accuracy

    def show_advanced_information(self, y_labels, y_prediction, feed_dict):
        y__ = y_labels.eval(feed_dict)
        pt("y_pred", y__[0])
        #argmax_labels_y_ = [np.argmax(m) for m in y__]
        #pt('y_labels_shape', y__.shape)
        #pt('argmax_labels_y__', argmax_labels_y_)
        #pt('y__[-1]', y__[-1])
        #pt("y_labels",y__)
        y__prediction = y_prediction.eval(feed_dict)
        #argmax_labels_y_convolutional = [np.argmax(m) for m in y__prediction]
        #pt('argmax_y_conv', argmax_labels_y_convolutional)
        #pt('y_pred_shape', y__prediction.shape)
        #pt("y_pred", y__prediction[0])
        pt("y_pred", y__prediction[0])
       #pt('index_buffer_data', self.index_buffer_data)
        #pt("SMAPE", smape(y__, y__prediction).eval(feed_dict))

    def save(self, saver, session):
        # Save variables to disk.
        if self.settings_object.model_path:
            try:
                pt("Saving model... DO NOT STOP PYTHON PROCESS")
                saver.save(session, self.settings_object.model_path + Dictionary.string_ckpt_extension)
                pt("Model saved without problem")
                if self.show_when_save_information:
                    pt("Saving model information...")
                if self.save_model_information:
                    accuracy = None
                    if self.validation_accuracy:
                        accuracy = self.validation_accuracy
                    elif self.test_accuracy:
                        accuracy = self.test_accuracy
                    filepath = self._save_model_to_json(
                        fullpath=self.settings_object.information_path,
                        attributes_to_delete=Constant.attributes_to_delete_information,
                        type_file="Information", accuracy=accuracy)
                else:
                    filepath = self.settings_object.history_information_path
                if self.show_when_save_information:
                    pt("Model information has been saved")
                return filepath
            except Exception as e:
                pt(Errors.error, e)
        else:
            pt(Errors.error, Errors.model_path_bad_configuration)

    def show_save_statistics(self, accuracies_train, accuracies_validation=None, accuracies_test=None,
                             loss_train=None, loss_validation=None, loss_test=None,
                             folder_to_save=None, show_graphs=None, is_new_epoch_flag=False):
        """
        Show all necessary visual and text information.
        """
        if is_new_epoch_flag:
            accuracies_train, accuracies_validation, accuracies_test, \
            loss_train, loss_validation, loss_test = preprocess_lists([accuracies_train, accuracies_validation,
                                                                       accuracies_test, loss_train, loss_validation,
                                                                       loss_test], index_to_eliminate=2)

        accuracy_plot = plt.figure(0)
        plt.title(str(self.options[0]))
        plt.xlabel("ITERATIONS | Batch Size=" + str(self.batch_size) + " | Trains for epoch: " + str(self.trains))
        plt.ylabel("ACCURACY (BLUE = Train | RED = Validation | GREEN = Test)")
        plt.plot(accuracies_train, 'b')
        if accuracies_validation:
            plt.plot(accuracies_validation, 'r')
        if accuracies_test:
            plt.plot(accuracies_test, 'g')
        if folder_to_save:
            folder = get_directory_from_filepath(folder_to_save)
            complete_name = folder + "\\graph_accuracy" + Dictionary.string_extension_png
            if self.save_graphs_images or is_new_epoch_flag:
                plt.savefig(complete_name)
        if (accuracies_train or accuracies_validation or accuracies_test) and show_graphs:
            accuracy_plot.show()
        loss_plot = plt.figure(1)
        plt.title("LOSS")
        plt.xlabel("ITERATIONS | Batch Size=" + str(self.batch_size) + " | Trains for epoch: " + str(self.trains))
        plt.ylabel("LOSS (BLUE = Train | RED = Validation | GREEN = Test)")
        plt.plot(loss_train, 'b')
        if loss_validation:
            plt.plot(loss_validation, 'r')
        if loss_test:
            plt.plot(loss_test, 'g')
        if (loss_train or loss_validation or loss_test) and show_graphs:
            loss_plot.show()
        if folder_to_save:
            folder = get_directory_from_filepath(folder_to_save)
            complete_name = folder + "\\graph_loss" + Dictionary.string_extension_png
            if self.save_graphs_images or is_new_epoch_flag:
                plt.savefig(complete_name)

    def print_actual_configuration(self):
        """
        Print all attributes to console
        """
        pt('first_label_neurons', self.first_label_neurons)
        pt('second_label_neurons', self.second_label_neurons)
        pt('third_label_neurons', self.third_label_neurons)
        pt('fourth_label_neurons',self._fourth_label_neurons)
        pt('input_size', self.input_size)
        pt('batch_size', self.batch_size)

    def update_batch(self, is_test=False, create_dataset_flag=False):
        if not is_test:
            self.input_batch, self.label_batch = self.data_buffer_generic_class(inputs=self.input,
                                                                                inputs_labels=self.input_labels,
                                                                                shuffle_data=self.shuffle_data,
                                                                                batch_size=self.batch_size,
                                                                                is_test=False,
                                                                                options=self.options,
                                                                                create_dataset_flag=create_dataset_flag)
        elif is_test:
            x_test_feed, y_test_feed = self.data_buffer_generic_class(inputs=self.test,
                                                                      inputs_labels=self.test_labels,
                                                                      shuffle_data=self.shuffle_data,
                                                                      batch_size=None,
                                                                      is_test=True,
                                                                      options=self.options,
                                                                      create_dataset_flag=create_dataset_flag)
            return x_test_feed, y_test_feed

    def train_model(self, *args, **kwargs):

        x = kwargs['kwargs']['x_input']
        y_labels = kwargs['kwargs']['y_labels']
        keep_probably = kwargs['kwargs']['keep_probably']
        accuracy = kwargs['kwargs']['accuracy']
        train_step = kwargs['kwargs']['train_step']
        cross_entropy = kwargs['kwargs']['cross_entropy']
        saver = kwargs['kwargs']['saver']
        sess = kwargs['kwargs']['sess']
        y_prediction = kwargs['kwargs']['y_prediction']

        x_test_feed, y_test_feed = self.update_batch(is_test=True)

        # TRAIN VARIABLES
        start_time = time.time()  # Start time

        # TO STATISTICS
        # To load accuracies and losses
        accuracies_train, accuracies_test, loss_train, loss_test = load_accuracies_and_losses(
            self.settings_object.accuracies_losses_path, self.restore_model)

        # Folders and file where information and configuration files will be saved.
        filepath_save = None

        # Update test feeds ( will be not modified during training)
        feed_dict_test_100 = {x: x_test_feed, y_labels: y_test_feed, keep_probably: 1}
        # Update real num_train:
        num_train_start = int(self.num_trains_count % self.trains)
        if num_train_start == self.trains:
            num_train_start = 0
        is_new_epoch_flag = False  # Represent if training come into a new epoch. With this, a graph will be saved each
        # new epoch
        saves_information = []  # Represent
        # START  TRAINING
        for epoch in range(self.num_epochs_count, self.epoch_numbers):  # Start with load value or 0
            for num_train in range(num_train_start, self.trains):  # Start with load value or 0
                # Update feeds
                feed_dict_train_100 = {x: self.input_batch, y_labels: self.label_batch, keep_probably: 1}
                feed_dict_train_dropout = {x: self.input_batch, y_labels: self.label_batch,
                                           keep_probably: self.train_dropout}

                # Setting values
                train_step.run(feed_dict_train_dropout)
                # TODO(@gabvaztor) Add validation_accuracy to training
                self.train_accuracy = accuracy.eval(feed_dict_train_100) * 100
                self.test_accuracy = accuracy.eval(feed_dict_test_100) * 100
                # Mandatory to save as numpy float64, not float32
                self.train_loss = np.float64(cross_entropy.eval(feed_dict_train_100))
                self.test_loss = np.float64(cross_entropy.eval(feed_dict_test_100))

                # To generate statistics
                accuracies_train.append(self.train_accuracy)
                accuracies_test.append(self.test_accuracy)
                loss_train.append(self.train_loss)
                loss_test.append(self.test_loss)

                with tf.device('/cpu:0'):
                    numpy_arrays = [accuracies_train, accuracies_test, loss_train, loss_test]
                    numpy_names = ["accuracies_train", "accuracies_test", "loss_train", "loss_test"]
                    save_numpy_arrays_generic(folder_to_save=self.settings_object.accuracies_losses_path,
                                              numpy_files=numpy_arrays,
                                              names=numpy_names)
                y_pre = y_prediction.eval(feed_dict_train_100)
                prediction_ = np.argmax(y_pre, axis=1)
                p = tf.argmax(y_prediction, axis=1).eval(feed_dict_train_100)
                pt("y_pre", y_pre)
                pt("y_pre_sum", y_pre.sum())
                pt("prediction_", prediction_)
                pt("p", p)
                pt("saves_information", saves_information)
                if num_train % 2 == 0:
                    percent_advance = "{0:.3f}".format(float(num_train * 100 / self.trains))
                    pt('Time', str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))
                    pt('TRAIN NUMBER: ' + str(self.num_trains_count) + ' | Percent Epoch ' +
                       str(epoch) + ": " + percent_advance + '%')
                    pt('train_accuracy', self.train_accuracy)
                    pt('cross_entropy_train', self.train_loss)
                    pt('test_accuracy', self.test_accuracy)
                    pt('self.index_buffer_data', self.index_buffer_data)
                    # DEBUG MODE
                    #y_pre = y_prediction.eval(feed_dict_train_100)
                    #pt("y_pre", y_pre)

                # Update indexes
                # Update num_epochs_counts
                if num_train + 1 == self.trains:  # +1 because start in 0
                    self.num_epochs_count += 1
                    is_new_epoch_flag = True
                # To decrement learning rate during training
                if self.num_epochs_count % self.number_epoch_to_change_learning_rate == 0 \
                        and self.num_epochs_count != 1 and self.index_buffer_data == 0:
                    self.learning_rate = float(self.learning_rate / 10.)
                if self.should_save(saves_information_list=saves_information, check_loss_train=True, if_is_equal=False):
                    filepath_save = self.save(saver=saver, session=sess)
                if self.show_advanced_info:
                    self.show_advanced_information(y_labels=y_labels, y_prediction=y_prediction,
                                                   feed_dict=feed_dict_train_100)
                with tf.device('/cpu:0'):
                    if (self.save_graphs_images and filepath_save) or (is_new_epoch_flag and filepath_save):
                        self.show_save_statistics(accuracies_train=accuracies_train, accuracies_test=accuracies_test,
                                                  loss_train=loss_train, loss_test=loss_test,
                                                  folder_to_save=filepath_save, show_graphs=False,
                                                  is_new_epoch_flag=is_new_epoch_flag)
                        is_new_epoch_flag = False

                # Update num_trains_count and num_epoch_count
                self.num_trains_count += 1
                # Collect trash
                if self.num_trains_count % 100 == 0:
                    gc.collect()
                # Update batches values
                self.update_batch()
                if self.save_model_configuration:
                    # Save configuration to that results
                    self._save_json_configuration(Constant.attributes_to_delete_configuration)
        pt('END TRAINING ')
        self.show_save_statistics(accuracies_train=accuracies_train, accuracies_test=accuracies_test,
                                  loss_train=loss_train, loss_test=loss_test, folder_to_save=filepath_save)
        self.make_predictions()

    def make_predictions(self):
        # TODO (@gabvaztor) Finish method
        pass



"""
STATIC METHODS: Not need "self" :argument
"""


def get_inputs_and_labels_shuffled(inputs, inputs_labels):
    """
    Get inputs_processed and labels_processed variables with an inputs and inputs_labels shuffled
    :param inputs: Represent input data
    :param inputs_labels:  Represent labels data
    :returns inputs_processed, labels_processed
    """
    c = list(zip(inputs, inputs_labels))
    random.shuffle(c)
    inputs_processed, labels_processed = zip(*c)
    inputs_processed, labels_processed = np.asarray(inputs_processed), np.asarray(labels_processed)
    return inputs_processed, labels_processed


def image_process_retinopathy(image, image_type, height, width, is_test=False, cv2_flag=False, debug_mode=False,
                              to_save=False):
    fullpath_image = image
    if not cv2_flag and not debug_mode:
        if image_type == 0:  # GrayScale
            image = Image.open(image).convert('L')
        else:
            image = Image.open(image)
        #pil_image_resized_antialias = np.array(image.resize((height, width), PIL.Image.ANTIALIAS))
        # Save resized
        if to_save:
            # Resize image and modify
            image_array = np.asarray(image)[:, 140:-127, :]
            image = Image.fromarray(image_array)
            width2, height2 = image.size
            pt("width2", width2)
            pt("height2", height2)
            image = np.array(image.resize((width, height)))
            pt("image", image.shape)

            # TODO (@gabvaztor) Create new place in SETTINGS to save new datasets
            # Image path
            path_to_save = os.path.dirname(fullpath_image)
            filename = os.path.basename(fullpath_image)[:-5]
            if is_test:
                folder = "\\test\\"
                pass  # We have already save test images
            else:
                folder = "\\train\\"
            fullpath_to_save = path_to_save + folder + filename
            create_directory_from_fullpath(fullpath=fullpath_to_save)
            Image.fromarray(image).save(fullpath_to_save + ".jpeg")
        else:
            image = np.asarray(image)
        return image


def process_input_unity_generic(x_input, y_label, options=None, is_test=False, to_save=False):
    """
    Generic method that process input and label across a if else statement witch contains a string that represent
    the option (option = how process data)
    :param x_input: A single input
    :param y_label: A single input label
    :param options: All attributes to process data. First position must to be the option.
    :param is_test: Sometimes you don't want to do some operation to test set.
    :return: x_input and y_label processed
    """
    if options:
        option = options[0]  # Option selected
        if option == Dictionary.string_option_signals_images_problem:
            x_input = process_image_signals_problem(x_input, options[1], options[2],
                                                    options[3], is_test=is_test)
        if option == Dictionary.string_option_german_prizes_problem:
            x_input = process_german_prizes_csv(x_input, is_test=is_test)
        if option == Dictionary.string_option_retinopathy_k_problem:
            x_input = image_process_retinopathy(image=x_input, image_type=options[1], height=options[2],
                                                width=options[3], is_test=is_test, to_save=to_save,
                                                cv2_flag=False, debug_mode=False)
    return x_input, y_label


# noinspection PyUnresolvedReferences
def process_image_signals_problem(image, image_type, height, width, is_test=False, cv2_flag=False, debug_mode=False):
    """
    Process signal image
    :param image: The image to change
    :param image_type: Gray Scale, RGB, HSV
    :param height: image height
    :param width: image width
    :param is_test: flag with True if image is in test set
    :return:
    """
    # TODO (@gabvaztor) Doc method
    if not cv2_flag and not debug_mode:
        image = Image.open(image).convert('L')
        image = np.array(image.resize((height, width)))
        pt("image", image)
        pt("image", image.size)
        #pil_image_resized_antialias = np.array(image.resize((height, width), PIL.Image.ANTIALIAS))

    elif not cv2_flag and debug_mode:
        import cv2
        image_ = cv2.imread(image, image_type)
        image_2 = cv2.resize(image_, (height, width))

        cv_image = np.array(image_)
        cv_image_resized = np.array(image_2)

        image = Image.open(image).convert('L')

        pil_image = np.array(image)
        pil_image_resized = np.array(image.resize((height, width)))
        pil_image_resized_antialias = np.array(image.resize((height, width), PIL.Image.ANTIALIAS))

        pt("image_shape", pil_image.shape)
        pt("image_shape", pil_image_resized.shape)
        pt("image", pil_image_resized_antialias.shape)

        cv_image_sum = np.sum(cv_image_resized, axis=1)
        pil_image_sum = np.sum(pil_image, axis=1)
        pil_image_resized_sum = np.sum(pil_image_resized, axis=1)
        pil_image_resized_antialias_sum = np.sum(pil_image_resized_antialias, axis=1)

        pt("cv_image_sum", cv_image_sum)
        pt("pil_image_sum", pil_image_sum)
        pt("pil_image_resized_sum", pil_image_resized_sum)
        pt("pil_image_resized_antialias_sum", pil_image_resized_antialias_sum)

        pt("cv_image_resized", np.sum(cv_image_resized))
        pt("pil_image_resized_sum", np.sum(pil_image_resized_sum))
        pt("pil_image_resized_antialias_sum", np.sum(pil_image_resized_antialias_sum))

        if np.array_equal(cv_image, pil_image):
            pt("YES")
        if np.array_equal(cv_image_resized, pil_image_resized):
            pt("YES")
        if np.array_equal(cv_image_resized, pil_image_resized_antialias):
            pt("YES")

        pt("image", pil_image)
        pt("image", pil_image_resized)
        pt("image", pil_image_resized_antialias)

    else:
        # 1- Get image in GrayScale
        # 2- Modify intensity and contrast
        # 3- Transform to gray scale
        # 4- Return image
        import cv2
        image = cv2.imread(image, image_type)
        image = cv2.resize(image, (height, width))
        #image = cv2.equalizeHist(image)

        if not is_test:
            random_percentage = random.randint(3, 20)
            to_crop_height = int((random_percentage * height) / 100)
            to_crop_width = int((random_percentage * width) / 100)
            image = image[to_crop_height:height - to_crop_height, to_crop_width:width - to_crop_width]
            image = cv2.copyMakeBorder(image, top=to_crop_height,
                                       bottom=to_crop_height,
                                       left=to_crop_width,
                                       right=to_crop_width,
                                       borderType=cv2.BORDER_CONSTANT)

        #image = image.reshape(-1)
        #cv2.imshow('image', image)
        #cv2.waitKey(0)  # Wait until press key to destroy image
    return image

def process_test_set(test, test_labels, options, create_dataset_flag=False):
    """
    Process test set and return it
    :param test: Test set
    :param test_labels: Test labels set
    :param options: All attributes to process data. First position must to be the option.
    :return: x_test and y_test
    """
    x_test = []
    y_test = []
    for i in range(len(test)):
        if i % 350 == 0:
            x, y = process_input_unity_generic(test[i], test_labels[i], options, is_test=True, to_save=create_dataset_flag)
            if not create_dataset_flag:
                x_test.append(x)
                y_test.append(y)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_test, y_test

def process_german_prizes_csv(x_input, is_test=False):
    return x_input

def call_method(method):
    method()