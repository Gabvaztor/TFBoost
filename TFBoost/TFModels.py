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

from UsefulTools.UtilsFunctions import *
from TFBoost.TFEncoder import Dictionary as dict
from TFBoost.TFEncoder import Constant as const
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
from pylab import *
''' TFLearn library. License MIT.
Git Clone : https://github.com/tflearn/tflearn.git
To install: pip install tflearn'''
import tflearn
'''PILOU for show images'''
from PIL import Image


"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- GLOBAL VARIABLES ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""


    # TODO Implement deep learning algorithms
    # TODO  Use tflearn to use basics algorithms

def lineal_model_basic_with_gradient_descent(self, input, test, input_labels, test_labels, number_of_inputs,
                                             number_of_classes,
                                             learning_rate=0.001, trains=100, type=None, validation=None,
                                             validation_labels=None, deviation=None):
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
    """
    # TODO Do general
    x = tf.placeholder(shape=[None, number_of_classes])
    y_ = tf.placeholder([None, number_of_classes])

    W = tf.Variable(tf.zeros([number_of_inputs, number_of_classes]))
    b = tf.Variable(tf.zeros([number_of_classes]))
    y = tf.matmul(x, W) + b

    cross_entropy_lineal = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = cross_entropy_lineal.minimize(cross_entropy_lineal)

    # TODO Error
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # TODO Train for epoch and training number
    for i in range(trains):
        pass

# TODO Do class object with all attributes of neuronal network (x,y,y_,accuracy,...) to, after that, create a generic
# TODO train class or method.
def convolution_model(input, test, input_labels, test_labels, number_of_classes, number_of_inputs=None,
                      learning_rate=1e-4, trains=100, type=None, validation=None,
                      validation_labels=None, deviation=None):
    """
    Generic convolutional model
    """

    # TODO Create an simple but generic convolutional model to analyse sets.
    # TODO Define firstLabelNeurons
    first_label_neurons = const.first_label_neurons  # Weight first label neurons
    second_label_neurons = const.second_label_neurons  # Weight first label neurons
    third_label_neurons = const.third_label_neurons  # Weight first label neurons

    first_patch = const.w_first_patch  # Weight first patch
    second_patch = const.w_second_patch  # Weight second patch
    number_inputs = const.w_number_inputs  # Weight number of inputs

    x1_rows_number = 24
    x1_column_number = 24
    x_columns = x1_rows_number*x1_column_number
    x_rows_column = [50,50]
    kernel_size = [2, 2]  # Kernel patch size
    num_epoch = 30  # Epochs number
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    min_after_dequeue = 1000
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.
    capacity = min_after_dequeue + 3 * num_epoch
    batch_size = 30
    input_size = len(input[0])
    pt('input_size',input_size)
    # TODO Try python EVAL method to do multiple variable neurons

    # Placeholders
    x = tf.placeholder(tf.float32,shape=[None, x_columns]) #  All images will be 25*25 = 625
    y_ = tf.placeholder(tf.float32,shape=[None, number_of_classes])  # Number of labels
    keep_probably = tf.placeholder(tf.float32)  # Value of dropout. With this you can set a value for each data set

    x_reshape = tf.reshape(x, [-1, x1_rows_number, x1_column_number, 1])  # Reshape x placeholder into a specific tensor
    # TODO Define shape and stddev in methods

    '''
    w_layer_1 = weight_variable([first_patch, second_patch, number_inputs, first_label_neurons])  # Weights and bias
    b_layer_1 = bias_variable([first_label_neurons])
    h_conv1 = tf.nn.relu(conv2d(x_reshape, w_layer_1) + b_layer_1)  # 1. RELU with Convolution 2D
    '''
    # TODO Define multiple layers
    # First Convolutional Layer
    conv1 = tf.layers.conv2d(
        inputs=x_reshape,
        filters=first_label_neurons,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)
    # Pool Layer 1 and reshape images into 12x12 with pool 2x2 and strides 2x2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Second Convolutional Layer
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=second_label_neurons,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)
    # Pool Layer 2 and reshape images into 6x6 with pool 2x2 and strides 2x2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 6 * 6 * second_label_neurons])
    dense = tf.layers.dense(inputs=pool2_flat, units=third_label_neurons, activation=tf.nn.relu)
    dropout = tf.nn.dropout(dense, keep_probably)
    # Readout Layer
    W_fc2 = weight_variable([third_label_neurons, number_of_classes])
    b_fc2 = bias_variable([number_of_classes])
    y_conv = (tf.matmul(dropout, W_fc2) + b_fc2)

    # Evaluate model
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) # Cross entropy between y_ and y_conv

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy) # Adam Optimizer (gradient descent)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # Get Number of right values in tensor
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Get accuracy in float
    filename_queue = tf.train.string_input_producer(input[0],
                                                    shuffle=False)  # List of files to read with extension '.png'
    #input_processed, label_processed = read_from_reader_signal_competition(filename_queue,x_rows_column)
    reader = tf.WholeFileReader()  # Reader with queue
    key, value = reader.read(filename_queue)
    my_img = tf.image.decode_png(value,channels=1)  # Use png decode and output gray scale
    resized_image = tf.image.resize_images(my_img, x_rows_column) # Resize image to dimension
    # Session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    path = key.eval()
    label = getSetsFromFullPathSignals(path)  # Must obtain path from file
    asd
    input_processed = resized_image
    label_processed = label

    x_batch, label_batch = tf.train.shuffle_batch(
        [input_processed, label_processed], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)



    # TRAIN
    # TODO TRAIN

    for i in range(3):
        ''' Show image '''

        image = x_batch.eval()
        image_array = image[:,:,-1] # Get gray scale dimension without channels (to show it)
        Image.fromarray(np.asarray(image_array)).show()
        imshow(np.asarray(image_array))

        train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: label_batch, keep_probably: 1.0}) * 100
        train_step.run(feed_dict={x: x_batch, y_: label_batch, keep_probably: 0.5})
        #validation_accuracy = accuracy.eval(feed_dict={x: validationPlaceholder, y_: validationLabels, keep_probably: 1.0}) * 100

        crossEntropyTrain = cross_entropy.eval(feed_dict={x: x_batch, y_: label_batch, keep_probably: 1.0})
       # crossEntropyValidation = cross_entropy.eval( feed_dict={x: validationPlaceholder, y_: validationLabels, keep_probably: 1.0})

        pt('train_accuracy',train_accuracy)
        pt('crossEntropyTrain',crossEntropyTrain)

    # When finish coord
    coord.request_stop()
    coord.join(threads)

def read_from_reader_signal_competition(filename_queue,x_rows_column):
    reader = tf.WholeFileReader()  # Reader with queue
    key, value = reader.read(filename_queue)
    my_img = tf.image.decode_png(value,channels=1)  # Use png decode and output gray scale
    resized_image = tf.image.resize_images(my_img, x_rows_column) # Resize image to dimension
    #image_transform = tf.image.rgb_to_grayscale(resized_image)  # Gray Scale because mathematically data has the same info
    path = key.eval()
    sda
    label = getSetsFromFullPathSignals(path)  # Must obtain path from file
    return resized_image, label
def weight_variable(shape):
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

