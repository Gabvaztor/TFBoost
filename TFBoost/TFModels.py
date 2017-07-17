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
import SettingsObject

''' TensorFlow: https://www.tensorflow.org/
To upgrade TensorFlow to last version:
*CPU: pip3 install --upgrade tensorflow
*GPU: pip3 install --upgrade tensorflow-gpu
'''
import tensorflow as tf
from tensorflow.contrib import rnn

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

''' TFLearn library. License MIT.
Git Clone : https://github.com/tflearn/tflearn.git
To install: pip3 install tflearn'''
import tflearn

'''"Best image library"
pip install opencv-python'''
import cv2

"""Python libraries"""
""" Random to shuffle lists """
import random

""" Time """
import time

""" To serialize object"""
import json

class TFModels():
    """
    Long Docs ...
    """

    # TODO Docs
    def __init__(self, setting_object, option_problem, input=None, test=None, input_labels=None, test_labels=None,
                 number_of_classes=None , type=None, validation=None, validation_labels=None,
                 load_model_configuration=False, *args , **kwargs):
        # TODO(@gabvaztor) Show and save graphs during all training asking before
        # NOTE: IF YOU LOAD_MODEL_CONFIGURATION AND CHANGE SOME TENSORFLOW ATTRIBUTE AS NEURONS, THE TRAIN WILL START
        # AGAIN
        self._input = input
        self._validation = validation
        self._test = test
        self._input_labels = input_labels
        self._validation_labels = validation_labels
        self._test_labels = test_labels
        self._number_of_classes = number_of_classes
        self._settings_object = setting_object  # Setting object represent a kaggle configuration
        self._input_batch = None
        self._label_batch = None
        # CONFIGURATION VARIABLES
        self._restore_model = False  # Labels and logits info.
        self._save_model_information = False  # If must to save model or not
        self._ask_to_save_model_information = False  # If True and 'save_model' is true, ask to save model each time
        # 'should_save'
        self._show_when_save_information = False  # If True then you will see printed in console when during training
        # the information.json has been saved.
        self._ask_to_continue_creating_model_without_exist = False  # If True and 'restore_model' is True,
        # ask to continues save model at first if there isn't a model to restore
        self._show_advanced_info = False  # Labels and logits info.
        self._show_images = False  # If True show images when show_info is True
        self._save_model_configuration = False  # If True, then all attributes will be saved in a settings_object path.
        self._shuffle_data = False  # If True, then the train and validation data will be shuffled separately.
        self._save_graphs_images = False  # If True, then save graphs images from statistical values. NOTE that this will
        # decrease the performance during training. Although this is true or false, for each time an epoch has finished,
        # the framework will save a graph
        # TRAIN MODEL VARIABLES
        self._input_rows_numbers = 60
        self._input_columns_numbers = 60
        self._kernel_size = [5, 5]  # Kernel patch size
        self._epoch_numbers = 130  # Epochs number
        self._batch_size = 100  # Batch size
        if input:
            self._input_size = 145062  # Change if necessary
            self._trains = int(self.input_size / self.batch_size) + 1  # Total number of trains for epoch
        if test:
            self._test_size = len(test)  # Change if necessary
        self._train_dropout = 0.5  # Keep probably to dropout to avoid overfitting
        self._first_label_neurons = 50
        self._second_label_neurons = 55
        self._third_label_neurons = 50
        self._learning_rate = 1e-3  # Learning rate
        self._number_epoch_to_change_learning_rate = 15  #You can choose a number to change the learning rate. Number
        # represent the number of epochs before be changed.
        # INFORMATION VARIABLES
        self._index_buffer_data = 0  # The index for mini_batches during training. Start at zero.
        self._num_trains_count = 1  # Start at one
        self._num_epochs_count = 1  # Start at one
        self._train_accuracy = None
        self._validation_accuracy = None
        self._test_accuracy = None
        # RESTART TRAINING
        self._save_and_restart = False  # All history and metadata will be saved in a different folder and the execution
        # will be restarted
        if self.save_and_restart:
            save_and_restart(self.settings_object.model_path)
        # OPTIONS
        # Options represent a list with this structure:
        #               - First position: "string_option" --> unique string to represent problem in question
        #               - Others positions: all variables you need to process each input and label elements
        # noinspection PyUnresolvedReferences
        self._options = [option_problem]
        # SAVE AND LOAD MODEL
        # If load_model_configuration is True, then it will load a configuration from settings_object method
        if load_model_configuration:
            # And restore time too.
            if self.restore_model:
                # input("You will load model configuration but no restore the tensorflow model, do you want to continue?")
                pt("Loading model configuration", self.settings_object.configuration_path)
                self._load_model_configuration(self.settings_object.load_actual_configuration())
        if self._save_model_configuration:
            # Save model configuration in a json file
            pt("Saving model configuration...")
            self._save_json_configuration(Constant.attributes_to_delete_configuration)
            pt("Model configuration has been saved")

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

    def _save_json_configuration(self, attributes_to_delete_configuration):
        self._save_model_configuration_to_json(self.settings_object.configuration_path,
                                               attributes_to_delete_configuration,
                                               type_file="Configuration")

    def properties(self, attributes_to_delete=None):
        """
        Return a string with actual features without not necessaries
        :param attributes_to_delete: represent witch attributes set must be deleted.
        :return: A copy of class.__dic__ without deleted attributes
        """
        dict_copy = self.__dict__.copy()  # Need to be a copy to not get original class' attributes.
        # Remove all not necessaries values
        if attributes_to_delete:
            for x in attributes_to_delete:
                del dict_copy[x]
        return dict_copy

    def to_json(self, attributes_to_delete=None):
        """
        Convert TFModel class to json with properties method.
        :param attributes_to_delete: String set with all attributes' names to delete from properties method
        :return: sort json from class properties.
        """
        return json.dumps(self, default=lambda o: self.properties(attributes_to_delete),
                          sort_keys=True, indent=4)


    @timed
    def rnn_lstm_web_traffic_time(self, *args, **kwargs):
        """
        LSTM solve to WEB_TRAFFIC_TIME problem
        """

        self.input_size = 145062  # Change if necessary
        self.trains = int(self.input_size / self.batch_size) + 1  # Total number of trains for epoch
        names_of_data = ["input_data", "validation_data", "inputs_labels", "validation_labels"]
        """
        self.create_input_and_label_data()
        save_accuracies_and_losses_training(folder_to_save=self.settings_object.accuracies_losses_path,
                                            numpy_file_1=self.input,
                                            numpy_file_2=self.validation,
                                            numpy_file_3=self.input_labels,
                                            numpy_file_4=self.validation_labels,
                                            names=names_of_data)
        """
        self.input, self.validation, self.input_labels, self.validation_labels = \
            load_4_numpy_files(path_to_load=self.settings_object.accuracies_losses_path, names_to_load_4=names_of_data)
        to_convert_to_numpy_array = [self.input, self.validation, self.input_labels, self.validation_labels]
        self.input, self.validation, self.input_labels, self.validation_labels = \
            convert_to_numpy_array(to_convert_to_numpy_array_list=to_convert_to_numpy_array)
        self.update_batch(is_test=False)
        # TODO After that, create lstm network and feed with batches.
        pt("input_shape", self.input.shape)
        # Network Parameters
        n_input = 28  # MNIST data input (img shape: 28*28)
        n_steps = 28  # timesteps
        n_hidden = 128  # hidden layer num of features
        n_classes = 1

        # TENSORFLOW --------------------------------------
        tf.reset_default_graph()
        # TODO (@gabvaztor) Continue creating x placeholder to "be a sequence"
        # tf Graph input
        x = tf.placeholder(tf.string, shape=[None, 1, 1])
        # x = tf.decode_raw(x, tf.float32)
        y_labels = tf.placeholder(tf.float32, [None, 1])
        keep_probably = tf.placeholder(tf.float32)

        # Define weights
        weights = tf.Variable(tf.random_normal([self.first_label_neurons, n_classes]))
        biases =  tf.Variable(tf.random_normal([n_classes]))

        # y_prediction = self.RNN(x, keep_probably, weights, biases)
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        #x = tf.unstack(input, n_steps, 1)
        # x = tf.unstack(x, 1)
        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.first_label_neurons, forget_bias=1)
        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.string)
        dropout = tf.nn.dropout(outputs[-1], keep_probably)
        # Linear activation, using rnn inner loop last output
        y_prediction = tf.matmul(dropout, weights) + biases

        # Define loss and optimizer
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_prediction, labels=y_labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)

        # Evaluate model
        correct_prediction = smape(y_labels, y_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initializing the variables
        sess = initialize_session()
        # Saver session
        saver = tf.train.Saver()  # Saver
        # To restore model
        if self.restore_model:
            self.load_and_restore_model(sess)

        # TRAIN ----------------------------------------------

        pt("input.shape", self.input.shape)
        pt("input_labels.shape", self.input_labels.shape)
        pt("validation.shape", self.validation.shape)
        pt("validation_labels.shape", self.validation_labels.shape)
        x_validation_feed, y_validation_feed = self.validation, self.validation_labels

        # TRAIN VARIABLES
        start_time = time.time()  # Start time

        # TO STATISTICS
        # To load accuracies and losses
        accuracies_train, accuracies_validation, loss_train, loss_validation = load_accuracies_and_losses(
            self.settings_object.accuracies_losses_path, self.restore_model)

        # Folders and file where information and configuration files will be saved.
        filepath_save = None

        # Update test feeds ( will be not modified during training)
        feed_dict_validation_100 = {x: x_validation_feed, y_labels: y_validation_feed, keep_probably: 1.}
        # Update real num_train:
        num_train_start = int(self.num_trains_count % self.trains)
        if num_train_start == self.trains:
            num_train_start = 0
        is_new_epoch_flag = False  # Represent if training come into a new epoch. With this, a graph will be saved each
        # new epoch
        # START  TRAINING
        for epoch in range(self.num_epochs_count, self.epoch_numbers):  # Start with load value or 0
            for num_train in range(num_train_start, self.trains):  # Start with load value or 0
                # Update feeds
                feed_dict_train_100 = {x: self.input_batch, y_labels: self.label_batch, keep_probably: 1}
                feed_dict_train_dropout = {x: self.input_batch, y_labels: self.label_batch,
                                           keep_probably: self.train_dropout}
                # Setting values
                # TODO(@gabvaztor) Add validation_accuracy to training
                self.validation_accuracy = accuracy.eval(feed_dict_train_100) * 100
                optimizer.run(feed_dict_train_dropout)
                self.validation_accuracy = accuracy.eval(feed_dict_validation_100) * 100
                cross_entropy_train = cross_entropy.eval(feed_dict_train_100)
                cross_entropy_validation = cross_entropy.eval(feed_dict_validation_100)

                # To generate statistics
                accuracies_train.append(self.train_accuracy)
                accuracies_validation.append(self.test_accuracy)
                loss_train.append(cross_entropy_train)
                loss_validation.append(cross_entropy_validation)
                with tf.device('/cpu:1'):
                    save_accuracies_and_losses_training(folder_to_save=self.settings_object.accuracies_losses_path,
                                                        numpy_file_1=accuracies_train,
                                                        numpy_file_2=accuracies_validation,
                                                        numpy_file_3=loss_train,
                                                        numpy_file_4=loss_validation)

                if num_train % 10 == 0:
                    percent_advance = str(num_train * 100 / self.trains)
                    pt('Time', str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))
                    pt('TRAIN NUMBER: ' + str(self.num_trains_count) + ' | Percent Epoch ' +
                       str(epoch) + ": " + percent_advance + '%')
                    pt('train_accuracy', self.train_accuracy)
                    pt('cross_entropy_train', cross_entropy_train)
                    pt('test_accuracy', self.test_accuracy)
                    pt('self.index_buffer_data', self.index_buffer_data)

                # Update indexes
                # Update num_epochs_counts
                if num_train + 1 == self.trains:  # +1 because start in 0
                    self.num_epochs_count += 1
                    is_new_epoch_flag = True
                # To decrement learning rate during training
                if self.num_epochs_count % self.number_epoch_to_change_learning_rate == 0 \
                        and self.num_epochs_count != 1 and self.index_buffer_data == 0:
                    self.learning_rate = float(self.learning_rate / 10.)
                if self.should_save():
                    filepath_save = self.save(saver=saver, session=sess)
                if self.show_advanced_info:
                    self.show_advanced_information(y_labels=y_labels, y_prediction=y_prediction,
                                                   feed_dict=feed_dict_train_100)
                with tf.device('/cpu:0'):
                    if (self.save_graphs_images and filepath_save) or (is_new_epoch_flag and filepath_save):
                        self.show_save_statistics(accuracies_train=accuracies_train,
                                                  accuracies_test=accuracies_validation,
                                                  loss_train=loss_train, loss_test=loss_validation,
                                                  folder_to_save=filepath_save, show_graphs=False,
                                                  is_new_epoch_flag=is_new_epoch_flag)
                        is_new_epoch_flag = False

                # Update num_trains_count and num_epoch_count
                self.num_trains_count += 1
                # Update batches values
                self.update_batch()
                if self.save_model_configuration:
                    # Save configuration to that results
                    self._save_json_configuration(Constant.attributes_to_delete_configuration)


    def RNN(self, input, keep_probably, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        #x = tf.unstack(input, n_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(self.first_label_neurons, forget_bias=1.0)
        input = tf.unstack(input, 1, 1)
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, input, dtype=tf.string)
        dropout = tf.nn.dropout(outputs[-1], keep_probably)
        # Linear activation, using rnn inner loop last output
        y_prediction = tf.matmul(dropout, weights) + biases
        return y_prediction

    @timed
    def convolution_model_image(self):
        """
        Generic convolutional model
        """
        # Print actual configuration
        self.print_actual_configuration()
        # TODO Try python EVAL method to do multiple variable neurons
        # Placeholders
        x_input, y_labels, keep_probably = self.placeholders(args=None, kwargs=None)
        # Reshape x placeholder into a specific tensor
        x_reshape = tf.reshape(x_input, [-1, self.input_rows_numbers, self.input_columns_numbers, 1])
        # Network structure
        y_prediction = self.network_structure(x_reshape, args=None, keep_probably=keep_probably)
        cross_entropy, train_step, correct_prediction, accuracy = self.model_evaluation(y_labels=y_labels,
                                                                                        y_prediction=y_prediction)
        # Batching values and labels from input and labels (with batch size)
        self.update_batch()
        # Session
        sess = initialize_session()
        # Saver session
        saver = tf.train.Saver()  # Saver
        # To restore model
        if self.restore_model:
            self.load_and_restore_model(sess)
        self.train_model(args=None, kwargs=locals())

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
                                  options=None):
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
            x_batch, y_batch = process_test_set(inputs, inputs_labels, options)
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

    def should_save(self):
        """
        Get last configuration from path
        
        :return: if should save 
        """
        should_save = False
        if self.save_model_information:
            actual_information = self.settings_object.load_actual_information()
            if actual_information:
                last_train_accuracy = actual_information._train_accuracy
                last_test_accuracy = actual_information._test_accuracy
                if last_train_accuracy and last_test_accuracy:
                    # TODO(@gabvaztor) Check when, randomly, gradient descent obtain high accuracy
                    if self.test_accuracy > last_test_accuracy:  # Save checking tests accuracies in this moment
                        should_save = True
                else:
                    if self.ask_to_save_model_information:
                        pt("last_train_accuracy", last_train_accuracy)
                        pt("last_test_accuracy", last_test_accuracy)
                        pt("actual_train_accuracy", self.train_accuracy)
                        pt("actual_test_accuracy", self.test_accuracy)
                        option_choosed = recurrent_ask_to_save_model()
                    else:
                        option_choosed = True
                    if option_choosed:
                        should_save = True
            else:
                should_save = True
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
            self._restore_model = configuration._restore_model
            self._save_model = configuration._save_model_information
            self._ask_to_save_model = configuration._ask_to_save_model_information
            self._show_info = configuration._show_advanced_info
            self._show_images = configuration._show_images
            self._save_model_configuration = configuration._save_model_configuration
            self._save_model_information = configuration._save_model_information
            self._shuffle_data = configuration._shuffle_data
            self._input_rows_numbers = configuration._input_rows_numbers
            self._input_columns_numbers = configuration._input_columns_numbers
            self._kernel_size = configuration._kernel_size
            self._epoch_numbers = configuration._epoch_numbers
            self._batch_size = configuration._batch_size
            self._input_size = configuration._input_size
            self._test_size = configuration._test_size
            self._train_dropout = configuration._train_dropout
            self._first_label_neurons = configuration._first_label_neurons
            self._second_label_neurons = configuration._second_label_neurons
            self._third_label_neurons = configuration._third_label_neurons
            self._learning_rate = configuration._learning_rate
            self._trains = configuration._trains
            self._number_epoch_to_change_learning_rate = configuration._number_epoch_to_change_learning_rate
            self._save_graphs_images = configuration._save_graphs_images
            self._ask_to_continue_creating_model_without_exist = \
                configuration._ask_to_continue_creating_model_without_exist
            self._ask_to_save_model_information = configuration._ask_to_save_model_information
            self._show_when_save_information = configuration._show_when_save_information
            # If you don't restore model then you won't load train number and epochs number
            if self.restore_model:
                self._num_trains_count = configuration._num_trains_count
                self._num_epochs_count = configuration._num_epochs_count
                self._index_buffer_data = configuration._index_buffer_data
            pt("Loaded model configuration")

    def _save_model_configuration_to_json(self, fullpath, attributes_to_delete=None, *args, **kwargs):
        """
        Save actual model configuration (with some attributes) in a json file.
        :param attributes_to_delete: represent witch attributes set must not be save in json file.
        """
        type_file = kwargs["type_file"]
        test_accuracy = ""
        if "test_accuracy" in kwargs:
            test_accuracy = kwargs["test_accuracy"]
        write_string_to_pathfile(self.to_json(attributes_to_delete),
                                 fullpath)
        # TODO (gabvaztor) Using new SettingObject path
        filepath = create_historic_folder(fullpath, type_file, test_accuracy)
        write_string_to_pathfile(self.to_json(attributes_to_delete),
                                 filepath)
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
        x = tf.placeholder(tf.float32, shape=[None, self.input_columns_after_reshape])  # All images will be 24*24 = 574
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
            filters=self.third_label_neurons,
            kernel_size=self.kernel_size,
            padding="same",
            activation=tf.nn.relu)
        # Pool Layer 1 and reshape images by 2
        pool1 = tf.layers.max_pooling2d(inputs=convolution_1, pool_size=[2, 2], strides=2)
        # Second Convolutional Layer
        convolution_2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.third_label_neurons,
            kernel_size=self.kernel_size,
            padding="same",
            activation=tf.nn.relu)
        # # Pool Layer 2 nd reshape images by 2
        pool2 = tf.layers.max_pooling2d(inputs=convolution_2, pool_size=[2, 2], strides=2)
        # Dense Layer
        # TODO Checks max pools numbers
        pool2_flat = tf.reshape(pool2, [-1, int(self._input_rows_numbers / 4) * int(self._input_columns_numbers / 4)
                                        * self.third_label_neurons])
        dense = tf.layers.dense(inputs=pool2_flat, units=self.third_label_neurons, activation=tf.nn.relu)
        dropout = tf.nn.dropout(dense, keep_dropout)
        # Readout Layer
        w_fc2 = weight_variable([self.third_label_neurons, self.number_of_classes])
        b_fc2 = bias_variable([self.number_of_classes])
        y_convolution = (tf.matmul(dropout, w_fc2) + b_fc2)
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
            tf.nn.softmax_cross_entropy_with_logits(labels=y_labels,
                                                    logits=y_prediction))  # Cross entropy between y_ and y_conv

        # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)  # Adadelta Optimizer
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)  # Adam Optimizer
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)  # Adam Optimizer

        # Sure is axis = 1
        correct_prediction = tf.equal(tf.argmax(y_prediction, axis=1),
                                      tf.argmax(y_labels, axis=1))  # Get Number of right values in tensor
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Get accuracy in float

        return cross_entropy, train_step, correct_prediction, accuracy

    def show_advanced_information(self, y_labels, y_prediction, feed_dict):
        y__ = y_labels.eval(feed_dict)
        argmax_labels_y_ = [np.argmax(m) for m in y__]
        pt('y__shape', y__.shape)
        pt('argmax_labels_y__', argmax_labels_y_)
        pt('y__[-1]', y__[-1])
        y__conv = y_prediction.eval(feed_dict)
        argmax_labels_y_convolutional = [np.argmax(m) for m in y__conv]
        pt('argmax_y_conv', argmax_labels_y_convolutional)
        pt('y_conv_shape', y__conv.shape)
        pt('index_buffer_data', self.index_buffer_data)

    def save(self, saver, session):
        # Save variables to disk.
        if self.settings_object.model_path:
            try:
                saver.save(session, self.settings_object.model_path + Dictionary.string_ckpt_extension)
                if self.show_when_save_information:
                    pt("Saving model information...")
                if self.save_model_information:
                    filepath = self._save_model_configuration_to_json(
                        fullpath=self.settings_object.information_path,
                        attributes_to_delete=Constant.attributes_to_delete_information,
                        type_file="Information", test_accuracy=self.test_accuracy)
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
        pt('input_size', self.input_size)
        pt('batch_size', self.batch_size)

    def update_batch(self, is_test=False):
        if not is_test:
            self.input_batch, self.label_batch = self.data_buffer_generic_class(inputs=self.input,
                                                                                inputs_labels=self.input_labels,
                                                                                shuffle_data=self.shuffle_data,
                                                                                batch_size=self.batch_size,
                                                                                is_test=False,
                                                                                options=self.options)
        elif is_test:
            x_test_feed, y_test_feed = self.data_buffer_generic_class(inputs=self.test,
                                                                      inputs_labels=self.test_labels,
                                                                      shuffle_data=self.shuffle_data,
                                                                      batch_size=None,
                                                                      is_test=True,
                                                                      options=self.options)
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
        # START  TRAINING
        for epoch in range(self.num_epochs_count, self.epoch_numbers):  # Start with load value or 0
            for num_train in range(num_train_start, self.trains):  # Start with load value or 0
                # Update feeds
                feed_dict_train_100 = {x: self.input_batch, y_labels: self.label_batch, keep_probably: 1}
                feed_dict_train_dropout = {x: self.input_batch, y_labels: self.label_batch,
                                           keep_probably: self.train_dropout}
                # Setting values
                # TODO(@gabvaztor) Add validation_accuracy to training
                self.train_accuracy = accuracy.eval(feed_dict_train_100) * 100
                train_step.run(feed_dict_train_dropout)
                self.test_accuracy = accuracy.eval(feed_dict_test_100) * 100
                cross_entropy_train = cross_entropy.eval(feed_dict_train_100)
                cross_entropy_test = cross_entropy.eval(feed_dict_test_100)

                # To generate statistics
                accuracies_train.append(self.train_accuracy)
                accuracies_test.append(self.test_accuracy)
                loss_train.append(cross_entropy_train)
                loss_test.append(cross_entropy_test)
                with tf.device('/cpu:1'):
                    save_accuracies_and_losses_training(folder_to_save=self.settings_object.accuracies_losses_path,
                                                        train_accuracies=accuracies_train,
                                                        validation_accuracies=accuracies_test,
                                                        train_losses=loss_train,
                                                        validation_losses=loss_test)

                if num_train % 10 == 0:
                    percent_advance = str(num_train * 100 / self.trains)
                    pt('Time', str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))
                    pt('TRAIN NUMBER: ' + str(self.num_trains_count) + ' | Percent Epoch ' +
                       str(epoch) + ": " + percent_advance + '%')
                    pt('train_accuracy', self.train_accuracy)
                    pt('cross_entropy_train', cross_entropy_train)
                    pt('test_accuracy', self.test_accuracy)
                    pt('self.index_buffer_data', self.index_buffer_data)

                # Update indexes
                # Update num_epochs_counts
                if num_train + 1 == self.trains:  # +1 because start in 0
                    self.num_epochs_count += 1
                    is_new_epoch_flag = True
                # To decrement learning rate during training
                if self.num_epochs_count % self.number_epoch_to_change_learning_rate == 0 \
                        and self.num_epochs_count != 1 and self.index_buffer_data == 0:
                    self.learning_rate = float(self.learning_rate / 10.)
                if self.should_save():
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
        pass

    def create_input_and_label_data(self):
        """
        Create input and label data by problem
        """
        if self.options[0] == Dictionary.string_option_web_traffic_problem:
            path_to_read_trains = self.settings_object.train_path
            # Read csv trains
            csv = Dictionary.string_csv_extension
            csv_files_list = []
            for number_file in range(1, 10):
                path_to_read = path_to_read_trains + str(number_file) + csv
                csv_files_list.append(path_to_read)
            filename_queue = tf.train.string_input_producer(csv_files_list)
            line_reader = tf.TextLineReader(skip_header_lines=1)
            _, csv_row = line_reader.read(filename_queue)
            # Default values, in case of empty columns. Also specifies the type of the
            # decoded result.
            record_defaults = [[""], [3.0]]
            page_date, visits = tf.decode_csv(csv_row, record_defaults=record_defaults)
            features = tf.stack(page_date)
            sess = initialize_session()
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            self.input, self.validation = np.array([[]]), np.array([])
            self.input_labels, self.validation_labels = np.array([]), np.array([])
            self.input_size = 145062  # Change if necessary
            self.trains = int(self.input_size / self.batch_size) + 1  # Total number of trains for epoch
            percent_80_input_size = int(self.input_size * 0.8)
            pt("Generating input and validation data...")
            is_first = True
            is_validation = False
            is_first_validation = True
            for i in range(self.input_size):
                # Retrieve a single instance:
                input, label = sess.run([features, visits])
                input_resized = np.asarray([[input.decode("utf-8")]])
                label_resized = np.asarray([[label]])
                if is_first and not is_validation:
                    self.input = input_resized
                    self.input_labels = label_resized
                    is_first = False
                    pt("input_shape", self.input.shape)
                    pt("input_labels_shape", self.input_labels.shape)
                    pt("input_resized_shape", input_resized.shape)
                    pt("label_resized_shape", label_resized.shape)

                if i < percent_80_input_size:
                    self.input = np.concatenate((self.input, input_resized))
                    self.input_labels = np.concatenate((self.input_labels, label_resized))
                else:
                    is_validation = True
                    if is_first_validation:
                        self.validation = input_resized
                        self.validation_labels = label_resized
                        is_first_validation = False
                    else:
                        self.validation = np.concatenate((self.validation, input_resized))
                        self.validation_labels = np.concatenate((self.validation_labels, label_resized))
            coord.request_stop()
            coord.join(threads)


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
    return inputs_processed, labels_processed


def process_input_unity_generic(x_input, y_label, options=None, is_test=False):
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
    return x_input, y_label


# noinspection PyUnresolvedReferences
def process_image_signals_problem(image, image_type, height, width, is_test=False):
    """
    Process signal image
    :param image: The image to change
    :param image_type: Gray Scale, RGB, HSV
    :param height: image height
    :param width: image width
    :param is_test: flag with True if image is in test set
    :return: 
    """
    # 1- Get image in GrayScale
    # 2- Modify intensity and contrast
    # 3- Transform to gray scale
    # 4- Return image
    image = cv2.imread(image, image_type)
    image = cv2.resize(image, (height, width))
    image = cv2.equalizeHist(image)
    image = cv2.equalizeHist(image)
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
    image = image.reshape(-1)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)  # Wait until press key to destroy image
    return image


def process_test_set(test, test_labels, options):
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
        x, y = process_input_unity_generic(test[i], test_labels[i], options, is_test=True)
        x_test.append(x)
        y_test.append(y)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_test, y_test


def process_german_prizes_csv(x_input, is_test=False):
    return x_input
