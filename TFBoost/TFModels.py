"""
Author: @gabvaztor
StartDate: 04/03/2017

This file contains samples and overrides deep learning algorithms.

Style: "Google Python Style Guide" 
https://google.github.io/styleguide/pyguide.html

"""
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
from TFBoost.TFEncoder import Dictionary as dict
from TFBoost.TFEncoder import Constant as const
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

''' Matlab URL: http://matplotlib.org/users/installing.html'''
import matplotlib.pyplot as plt
from pylab import *

''' TFLearn library. License MIT.
Git Clone : https://github.com/tflearn/tflearn.git
To install: pip install tflearn'''
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
    def __init__(self,input, test, input_labels, test_labels, number_of_classes, setting_object,
                 type=None, validation=None, validation_labels=None, load_model_configuration=False):
        # TODO(@gabvaztor) Load configuration by problem from json file in Settings folder
        # TODO (@gabvaztor) Add all methods using 'self' like class methods
        self._input = input
        self._test = test
        self._input_labels = input_labels
        self._test_labels = test_labels
        self._number_of_classes = number_of_classes
        self._settings_object = setting_object  # Setting object represent a kaggle configuration
        # CONFIGURATION VARIABLES
        self._restore_model = True  # Labels and logits info.
        self._save_model = True  # If must to save model or not
        self._ask_to_save_model = False  # If True and 'save_model' is true, ask to save model each time 'should_save'
        self._show_info = False  # Labels and logits info.
        self._show_images = False  # If True show images when show_info is True
        self._save_model_configuration = False  # If True, then all attributes will be saved in a settings_object path
        # TRAIN MODEL VARIABLES
        self._shuffle_data = True
        self._input_rows_numbers = 60
        self._input_columns_numbers = 60
        self._kernel_size = [5, 5]  # Kernel patch size
        self._epoch_numbers = 100  # Epochs number
        self._batch_size = 64  # Batch size
        self._input_size = len(input)  # Change if necessary
        self._test_size = len(test)  # Change if necessary
        self._train_dropout = 0.5  # Keep probably to dropout to avoid overfitting
        self._first_label_neurons = 50
        self._second_label_neurons = 55
        self._third_label_neurons = 50
        self._learning_rate = 1e-3  # Learning rate
        self._trains = int(self.input_size / self.batch_size) + 1
        # INFORMATION VARIABLES
        self._index_buffer_data = 0  # The index for batches during training
        self._num_trains_count = 0
        # TODO(@gabvaztor) add validation_accuracy
        self._train_accuracy = None
        self._test_accuracy = None
        # SAVE AND LOAD MODEL
        # TODO(@gabvaztor) Finish load_model_configuration function
        # If load_model_configuration is True, then it will load a configuration from settings_object method
        if load_model_configuration:
            pt("Loading model", self.settings_object.configuration_path)
            self._load_model_configuration(
                self.settings_object.load_model_configuration(self.settings_object.configuration_path))
        # TODO(@gabvaztor) Finish _save_model_configuration function
        if self._save_model_configuration:
            # Save model configuration in a json file
            self._save_model_configuration_to_json(self.settings_object.configuration_path,
                                                   Constant.attributes_to_delete_save_all)

    @property
    def show_info(self): return self._show_info

    @show_info.setter
    def show_info(self, value): self._show_info = value

    @property
    def save_model(self): return self._save_model

    @save_model.setter
    def save_model(self, value): self._save_model = value

    @property
    def save_model_configuration(self): return self._save_model_configuration

    @save_model_configuration.setter
    def save_model_configuration(self, value):  self._save_model_configuration = value

    @property
    def ask_to_save_model(self):
        if self.save_model:
            return self._ask_to_save_model
        else:
            return False

    @ask_to_save_model.setter
    def ask_to_save_model(self, value): self._ask_to_save_model = value

    @property
    def restore_model(self): return self._restore_model

    @restore_model.setter
    def restore_model(self, value): self._restore_model = value

    @property
    def train_accuracy(self): return self._train_accuracy

    @train_accuracy.setter
    def train_accuracy(self, value): self._train_accuracy = value

    @property
    def test_accuracy(self): return self._test_accuracy

    @test_accuracy.setter
    def test_accuracy(self, value): self._test_accuracy = value

    @property
    def settings_object(self): return self._settings_object

    @settings_object.setter
    def settings_object(self, value): self._settings_object = value

    @property
    def learning_rate(self): return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value): self._learning_rate = value

    @property
    def show_images(self): return self._show_images

    @show_images.setter
    def show_images(self, value): self._show_images = value

    @property
    def shuffle_data(self): return self._shuffle_data

    @shuffle_data.setter
    def shuffle_data(self, value): self._shuffle_data = value

    @property
    def input_rows_numbers(self): return self._input_rows_numbers

    @input_rows_numbers.setter
    def input_rows_numbers(self, value): self._input_rows_numbers = value

    @property
    def input_columns_numbers(self): return self._input_columns_numbers

    @input_columns_numbers.setter
    def input_columns_numbers(self, value): self._input_columns_numbers = value

    @property
    def input_columns_after_reshape(self): return self.input_rows_numbers * self.input_columns_numbers

    @input_columns_after_reshape.setter
    def input_columns_after_reshape(self, value): self.input_columns_after_reshape = value

    @property
    def input_rows_columns_array(self): return [self.input_rows_numbers, self.input_columns_numbers]

    @input_rows_columns_array.setter
    def input_rows_columns_array(self, value): self.input_rows_columns_array = value

    @property
    def kernel_size(self): return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value): self._kernel_size = value

    @property
    def input_size(self): return self._input_size

    @input_size.setter
    def input_size(self, value): self._input_size = value

    @property
    def test_size(self): return self._test_size

    @test_size.setter
    def test_size(self, value): self._test_size = value

    @property
    def batch_size(self): return self._batch_size

    @batch_size.setter
    def batch_size(self, value): self._batch_size = value

    @property
    def train_dropout(self): return self._train_dropout

    @train_dropout.setter
    def train_dropout(self, value): self._train_dropout = value

    @property
    def index_buffer_data(self): return self._index_buffer_data

    @index_buffer_data.setter
    def index_buffer_data(self, value): self._index_buffer_data = value

    @property
    def first_label_neurons(self): return self._first_label_neurons

    @first_label_neurons.setter
    def first_label_neurons(self, value): self._first_label_neurons = value

    @property
    def second_label_neurons(self): return self._second_label_neurons

    @second_label_neurons.setter
    def second_label_neurons(self, value): self._second_label_neurons = value

    @property
    def third_label_neurons(self): return self._third_label_neurons

    @third_label_neurons.setter
    def third_label_neurons(self, value): self._third_label_neurons = value

    @property
    def trains(self): return self._trains

    @trains.setter
    def trains(self, value): self._trains = value

    @property
    def num_trains_count(self): return self._num_trains_count

    @num_trains_count.setter
    def num_trains_count(self, value): self._num_trains_count = value

    @property
    def number_of_classes(self): return self._number_of_classes

    @number_of_classes.setter
    def number_of_classes(self, value): self._number_of_classes = value

    @property
    def input_labels(self): return self._input_labels

    @input_labels.setter
    def input_labels(self, value): self._input_labels = value

    @property
    def test_labels(self): return self._test_labels

    @test_labels.setter
    def test_labels(self, value): self._test_labels = value

    @property
    def epoch_numbers(self): return self._epoch_numbers

    @epoch_numbers.setter
    def epoch_numbers(self, value): self._epoch_numbers = value

    @property
    def input(self): return self._input

    @input.setter
    def input(self, value): self._input = value

    @property
    def test(self): return self._test

    @test.setter
    def test(self, value): self._test = value

    def properties(self, attributes_to_delete=None):
        """
        Return a string with actual features without not necessaries
        :param attributes_to_delete: represent witch attributes set must be deleted.
        :return:
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
    def convolution_model_image(self):
        """
        Generic convolutional model
        """
        # Print actual configuration
        pt('first_label_neurons', self.first_label_neurons)
        pt('second_label_neurons', self.second_label_neurons)
        pt('third_label_neurons', self.third_label_neurons)
        pt('input_size', self.input_size)
        pt('batch_size', self.batch_size)
        # TODO Try python EVAL method to do multiple variable neurons

        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, self.input_columns_after_reshape])  # All images will be 24*24 = 574
        y_ = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])  # Number of labels
        keep_probably = tf.placeholder(tf.float32)  # Value of dropout. With this you can set a value for each data set

        # Reshape x placeholder into a specific tensor
        x_reshape = tf.reshape(x, [-1, self.input_rows_numbers, self.input_columns_numbers, 1])

        # First Convolutional Layer
        convolution_1 = tf.layers.conv2d(
            inputs=x_reshape,
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
        dropout = tf.nn.dropout(dense, keep_probably)
        # Readout Layer
        w_fc2 = weight_variable([self.third_label_neurons, self.number_of_classes])
        b_fc2 = bias_variable([self.number_of_classes])
        y_convolution = (tf.matmul(dropout, w_fc2) + b_fc2)

        # Evaluate model
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                    logits=y_convolution))  # Cross entropy between y_ and y_conv

        # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)  # Adadelta Optimizer
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)  # Adam Optimizer
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)  # Adam Optimizer

        # Sure is axis = 1
        correct_prediction = tf.equal(tf.argmax(y_convolution, axis=1),
                                      tf.argmax(y_, axis=1))  # Get Number of right values in tensor
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Get accuracy in float
        # TODO(@gabvaztor) define options
        options = [Dictionary.string_option_signals_images_problem, cv2.IMREAD_GRAYSCALE,
                   self.input_rows_columns_array[0], self.input_rows_columns_array[1]]
        # Batching values and labels from input and labels (with batch size)
        x_batch_feed, label_batch_feed = self.data_buffer_generic_class(inputs=self.input,
                                                                        inputs_labels=self.input_labels,
                                                                        shuffle_data=self.shuffle_data,
                                                                        batch_size=self.batch_size,
                                                                        is_test=False,
                                                                        options=options)
        x_test_feed, y_test_feed = self.data_buffer_generic_class(inputs=self.test,
                                                                  inputs_labels=self.test_labels,
                                                                  shuffle_data=self.shuffle_data,
                                                                  batch_size=None,
                                                                  is_test=True,
                                                                  options=options)
        # Session
        sess = tf.InteractiveSession()
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # Saver session
        saver = tf.train.Saver()  # Saver

        # TRAIN VARIABLES
        start_time = time.time()  # Start time
        is_first_time = True  # Check if is first train
        feed_dict_train_100 = {x: x_batch_feed, y_: label_batch_feed, keep_probably: 1}
        feed_dict_test_100 = {x: x_test_feed, y_: y_test_feed, keep_probably: 1}

        # To restore model
        if self.restore_model:
            self.create_path_and_restore_model(sess)

        # START TRAINING
        for epoch in range(self.epoch_numbers):
            for i in range(self.trains):
                # Setting values
                feed_dict_train_50 = {x: x_batch_feed, y_: label_batch_feed, keep_probably: self.train_dropout}
                self.train_accuracy = accuracy.eval(feed_dict_train_100) * 100
                train_step.run(feed_dict_train_50)
                self.test_accuracy = accuracy.eval(feed_dict_test_100) * 100
                cross_entropy_train = cross_entropy.eval(feed_dict_train_100)
                if self.should_save():
                    # Save variables to disk.
                    if self.settings_object.model_path:
                        try:
                            saver.save(sess, self.settings_object.model_path+Dictionary.string_ckpt_extension)
                            self._save_model_configuration_to_json(
                                fullpath=self.settings_object.information_path,
                                attributes_to_delete=Constant.attributes_to_delete_save_all)
                        except Exception as e:
                            pt(Errors.error,e)
                    else:
                        pt(Errors.error, Errors.model_path_bad_configuration)
                # TODO Use validation set
                if self.show_info:
                    y__ = y_.eval(feed_dict_train_100)
                    argmax_labels_y_ = [np.argmax(m) for m in y__]
                    pt('y__shape', y__.shape)
                    pt('argmax_labels_y__', argmax_labels_y_)
                    pt('y__[-1]', y__[-1])
                    y__conv = y_convolution.eval(feed_dict_train_100)
                    argmax_labels_y_convolutional = [np.argmax(m) for m in y__conv]
                    pt('argmax_y_conv', argmax_labels_y_convolutional)
                    pt('y_conv_shape', y__conv.shape)
                    pt('index_buffer_data', self.index_buffer_data)
                if i % 10 == 0:
                    percent_advance = str(i * 100 / self.trains)
                    pt('Time', str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))
                    pt('TRAIN NUMBER: ' + str(self.num_trains_count + 1) + ' | Percent Epoch ' +
                       str(epoch + 1) + ": " + percent_advance + '%')
                    pt('train_accuracy', self.train_accuracy)
                    pt('cross_entropy_train', cross_entropy_train)
                    pt('test_accuracy', self.test_accuracy)
                    pt('self.index_buffer_data', self.index_buffer_data)
                self.num_trains_count += 1
                # Update batches values
                x_batch_feed, label_batch_feed = self.data_buffer_generic_class(inputs=self.input,
                                                                                inputs_labels=self.input_labels,
                                                                                shuffle_data=self.shuffle_data,
                                                                                batch_size=self.batch_size,
                                                                                is_test=False,
                                                                                options=options)
        pt('END TRAINING ')

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
        :param batch_size: The batch size .
        :param is_test: if the inputs are the test set.
        :param options: options       
        :return: Two numpy arrays (x_batch and y_batch) with input data and input labels data batch_size like shape.
        """
        # TODO Add this method like class method
        x_batch = []
        y_batch = []
        if is_test:
            x_batch, y_batch = process_test_set(inputs,inputs_labels,options)
        else:
            if shuffle_data and self.index_buffer_data == 0:
                self.input, self.input_labels = get_inputs_and_labels_shuffled(self.input,self.input_labels)
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
        if self.save_model:
            actual_information = self.settings_object.load_actual_information()
            if actual_information:
                last_train_accuracy = actual_information._train_accuracy
                last_test_accuracy = actual_information._test_accuracy
                if last_train_accuracy and last_test_accuracy:
                    # TODO(@gabvaztor) Check when, randomly, gradient descent obtain aprox 100% accuracy
                    if self.test_accuracy > last_test_accuracy:  # Save checking tests accuracies in this moment
                        should_save = True
                else:
                    if self.ask_to_save_model:
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
        :param configuration: the json class 
        """
        # TODO Add to docs WHEN it is necessary to add more attributes
        self._restore_model = configuration._restore_model
        self._save_model = configuration._save_model
        self._ask_to_save_model = configuration._ask_to_save_model
        self._show_info = configuration._show_info
        self._show_images = configuration._show_images
        self._save_model_configuration = configuration._save_model_configuration
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

    def _save_model_configuration_to_json(self, fullpath, attributes_to_delete=None):
        """
        Save actual model configuration (with some attributes) in a json file.
        :param attributes_to_delete: represent witch attributes set must not be save in json file.
        """
        pt("Saving model...")
        write_string_to_pathfile(self.to_json(attributes_to_delete),
                                 fullpath)
        pt("Model has been saved")

    def create_path_and_restore_model(self, session):
        """
        Restore a model from a model_path checking if model_path exists and create if not.
        :param session: Tensorflow session
        """
        # TODO Show comments when restoring model operation start or finish
        if self.settings_object.model_path:
            pt("Restoring model...", self.settings_object.model_path)
            try:
                # TODO (@gabvaztor) Fix this 'if' statement
                if file_exists_in_path_or_create_path(
                                self.settings_object.model_path + Dictionary.string_ckpt_extension) or \
                        file_exists_in_path_or_create_path(
                                            self.settings_object.model_path + Dictionary.string_ckpt_extension + Dictionary.string_meta_extension):
                    saver = tf.train.import_meta_graph(
                        self.settings_object.model_path + Dictionary.string_ckpt_extension + Dictionary.string_meta_extension)
                    # Restore variables from disk.
                    saver.restore(session, self.settings_object.model_path + Dictionary.string_ckpt_extension)
                    pt("Model restored without problems")
                else:
                    pt(Errors.error, Errors.can_not_restore_model_because_path_not_exists)
                    input("Press enter to continue")
            except Exception as e:
                pt(Errors.error, e)
                input(Errors.error + " " + Errors.can_not_restore_model + " Press enter to continue")


"""
STATIC METHODS
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
            x_input = process_image_signals_problem(x_input,options[1],options[2],options[3],is_test=is_test)
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
        x, y = process_input_unity_generic(test[i],test_labels[i],options,is_test=True)
        x_test.append(x)
        y_test.append(y)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_test, y_test

def process_german_prizes_csv(x_input, is_test=False):
    return x_input

def weighted_mape_tf(y_true, y_pred):
    tot = tf.reduce_sum(y_true)
    # tot = tf.clip_by_value(tot, clip_value_min=-550, clip_value_max=550)
	# wmape = tf.realdiv(tf.reduce_sum(tf.abs(tf.subtract(y_true, y_pred))), tot)  # /tot
    wmape = tf.truediv(tf.reduce_sum(tf.abs(tf.subtract(y_true, y_pred))),tot)# /tot
    return wmape
def root_mean_squared_logarithmic_error(y_true, y_pred):
    """
    Calculate the Root Mean Squared Logarithmic Error
    :param y_true: 
    :param y_pred: 
    :return: Root Mean Squared Logarithmic Error
    """
    # TODO (@gabvaztor) Do Root Mean Squared Logarithmic Error
    pass

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    # initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')