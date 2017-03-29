"""
Author: @gabvaztor
StartDate: 04/03/2017

This file contains samples and overrides deep learning algorithms.
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

'''LOCAL IMPORTS
'''

import UsefulTools.UtilsFunctions as uf
from TFBoost.TFEncoder import Dictionary as dict

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

class Models():
    # TODO Docs
    """
    This class
    """
    # TODO Implement deep learning algorithms
    # TODO  Use tflearn to use basics algorithms

    def lineal_model_basic_with_gradient_descent(self, input, test, input_labels, test_labels,number_of_inputs,number_of_classes,
                                      learning_rate = 0.001,trains = 100, type = None ,validation = None,
                                      validation_labels = None, deviation = None):
        """
        This method doesn't do softmax.
        :param input: Input data
        :param validation: Validation data
        :param test: Test data
        :param type: Type of data (float32, float16, ...)
        :param trains: Number of trains for epoch
        :param number_of_inputs: Represents the number of records in input data
        :param number_of_classes: Represents the number of labels in data
        :param deviation: Number of the deviation for the weights and bias
        :return:
        """
        # TODO Do general
        x = tf.placeholder(shape=[None,number_of_classes])
        y_ = tf.placeholder([None, number_of_classes])

        W = tf.Variable(tf.zeros([number_of_inputs, number_of_classes]))
        b = tf.Variable(tf.zeros([number_of_classes]))
        y = tf.matmul(x, W) + b

        cross_entropy_lineal = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = cross_entropy_lineal.minimize(cross_entropy_lineal)

        # TODO Error
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # TODO Train for epoch and training number

        for i in range(trains):
            pass

    def convolution_model(self, input, test, input_labels, test_labels,number_of_inputs,number_of_classes,
                                      learning_rate = 0.001,trains = 100, type = None ,validation = None,
                                      validation_labels = None, deviation = None):
        """

        :return:
        """

        x = tf.placeholder(shape=[None,number_of_classes])
        y_ = tf.placeholder([None, number_of_classes])



        # TODO Create an simple but generic convolutional model to analyce sets.

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

