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
                      learning_rate=1e-3, trains=None, type=None, validation=None,
                      validation_labels=None, deviation=None):
    """
    Generic convolutional model
    """

    # TODO Create an simple but generic convolutional model to analyse sets.
    show_info = True # Show images, labels and logits info.
    x1_rows_number = 24
    x1_column_number = 24
    x_columns = x1_rows_number*x1_column_number
    x_rows_column = [24,24]
    kernel_size = [2, 2]  # Kernel patch size
    num_epoch = 5  # Epochs number
    batch_size = 5  # Batch size
    input_size = len(input)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    min_after_dequeue = 1000
    # capacity must be larger than min_after_dequeue and the amount larger
    # determines the maximum we will prefetch.
    capacity = int(input_size / 4)
    first_label_neurons = number_neurons(input_size, batch_size, number_of_classes)  # Weight first label neurons
    second_label_neurons = number_neurons(input_size, first_label_neurons, number_of_classes)  # Weight second label neurons
    third_label_neurons = number_neurons(input_size, second_label_neurons, number_of_classes)  # Weight third label neurons
    if not trains:
        trains = int(input_size/batch_size)+1

    pt('first_label_neurons', first_label_neurons)
    pt('second_label_neurons', second_label_neurons)
    pt('third_label_neurons', third_label_neurons)
    pt('input_size', input_size)
    # TODO Try python EVAL method to do multiple variable neurons

    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, x_columns])  # All images will be 24*24 = 574
    y_ = tf.placeholder(tf.float32, shape=[None, number_of_classes])  # Number of labels
    keep_probably = tf.placeholder(tf.float32)  # Value of dropout. With this you can set a value for each data set

    x_reshape = tf.reshape(x, [-1, x1_rows_number, x1_column_number, 1])  # Reshape x placeholder into a specific tensor
    # TODO Define shape and stddev in methods
    # First Convolutional Layer
    convolution_1 = tf.layers.conv2d(
        inputs=x_reshape,
        filters=first_label_neurons,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)
    # Pool Layer 1 and reshape images into 12x12 with pool 2x2 and strides 2x2
    pool1 = tf.layers.max_pooling2d(inputs=convolution_1, pool_size=[2, 2], strides=2)
    # Second Convolutional Layer
    convolution_2 = tf.layers.conv2d(
        inputs=pool1,
        filters=second_label_neurons,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)
    # Pool Layer 2 and reshape images into 6x6 with pool 2x2 and strides 2x2
    pool2 = tf.layers.max_pooling2d(inputs=convolution_2, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 6 * 6 * second_label_neurons])
    dense = tf.layers.dense(inputs=pool2_flat, units=third_label_neurons, activation=tf.nn.relu)
    dropout = tf.nn.dropout(dense, keep_probably)
    # Readout Layer
    w_fc2 = weight_variable([third_label_neurons, number_of_classes])
    b_fc2 = bias_variable([number_of_classes])
    y_convolution = (tf.matmul(dropout, w_fc2) + b_fc2)

    # Evaluate model
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_convolution))  # Cross entropy between y_ and y_conv

    #train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)  # Adadelta Optimizer (gradient descent)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)  # Adam Optimizer (gradient descent)
    correct_prediction = tf.equal(tf.argmax(y_convolution, axis=0), tf.argmax(y_, axis=0))  # Get Number of right values in tensor
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Get accuracy in float

    # Creating BATCH
    inputs_array = np.array(input)
    labels_array = np.array(input_labels)
    inputs_tensor = tf.convert_to_tensor(inputs_array, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(labels_array, dtype=tf.float32)
    inputs_and_labels = [inputs_tensor, labels_tensor]

    # TODO BUG when try to put labels with shape (x,)
    # Slice inputs and labels into one example
    train_input_queue = tf.train.slice_input_producer(inputs_and_labels,
                                                    shuffle=True)  # List of files to read with extension '.png'
    # Obtain real value and real label in tensor
    input_processed, label_processed =\
        read_from_reader_signal_competition(train_input_queue,x_rows_column)
    pt('input_processed', input_processed)
    # Batching values and labels from input_processed (with batch size)
    x_batch, label_batch = tf.train.batch(
        [input_processed, label_processed],
        batch_size=batch_size, capacity=capacity)
    # Session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # TRAIN
    for epoch in range (num_epoch):
        for i in range(trains):
            '''
            Test
            '''

            '''
            #No change the positions
            pt('inputs_tensor', inputs_tensor.eval()[0])
            pt('labels_tensor', labels_tensor.eval()[0])
            pt('train_input_queue[0]', train_input_queue[0].eval())
            pt('train_input_queue[1]', train_input_queue[1].eval())
            pt('input_processed', input_processed.eval().shape)
            pt('label_processed', label_processed.eval())
            '''

            ''' Show image '''
            x_train_feed = x_batch.eval()
            label_train_feed = label_batch.eval()
            feed_dict_train_50 = {x: x_train_feed, y_: label_train_feed, keep_probably: 0.5}
            feed_dict_train_100 = {x: x_train_feed, y_: label_train_feed, keep_probably: 1}
            #pt('the resized_imag', resize_image.eval().shape)
            # image_array = resized_imag[:,:,-1] # Get gray scale dimension without channels (to show it)
            # pt('the image', image_array)

            train_step.run(feed_dict_train_50)
            train_accuracy = accuracy.eval(feed_dict_train_100) * 100
            # validation_accuracy = accuracy.eval(feed_dict={x: validationPlaceholder, y_: validationLabels, keep_probably: 1.0}) * 100

            crossEntropyTrain = cross_entropy.eval(feed_dict_train_100)
            # crossEntropyValidation = cross_entropy.eval( feed_dict={x: validationPlaceholder, y_: validationLabels, keep_probably: 1.0})

            if show_info and epoch == 0 and i == 0:
                pt('the image', x_train_feed.shape)

                argmax_labels = [np.argmax(c) for c in label_train_feed]
                pt('label_train_feed', argmax_labels)
                pt('label_train_feed_shape', label_train_feed.shape)

                y__ = y_.eval(feed_dict_train_100)
                argmax_labels_y_ = [np.argmax(m) for m in y__]
                pt('y__', argmax_labels_y_)
                pt('y__shape', y__.shape)

                y__conv = y_convolution.eval(feed_dict_train_100)
                argmax_labels_y_convolutional = [np.argmax(m) for m in y__conv]
                pt('y_conv', argmax_labels_y_convolutional)
                pt('y_conv_shape', y__conv.shape)

                show_image_from_tensor(x_train_feed, 24) # shows images

            pt('train_accuracy',train_accuracy)
            pt('crossEntropyTrain',crossEntropyTrain)
            # pt('correct_prediction',correct_prediction.eval())
            #pt('y_conv',y_convolution.eval(feed_dict={x: x_train_feed, y_: label_train_feed, keep_probably: 1.0}) * 100)
            percent_avance = str(i*100/trains)
            pt('Percent Epoch ' + str(epoch+1), percent_avance + '%')
    # When finish coord
    coord.request_stop()
    coord.join(threads)
    pt('END TRAINING ')


def read_from_reader_signal_competition(filename_queue,x_rows_column):
    # reader = tf.WholeFileReader()  # Reader with queue
    # key, value = reader.read(filename_queue[0])
    unique_input =  tf.read_file(filename_queue[0])
    input_label = filename_queue[1]
    my_img = tf.image.decode_png(unique_input,channels=1)  # Use png decode and output gray scale
    resized_image = tf.image.resize_images(my_img, x_rows_column) # Resize image to dimension
    flatten_image = tf.reshape(resized_image, [-1])
    # image_transform = tf.image.rgb_to_grayscale(resized_image)  # Gray Scale because mathematically data has the same info
    pt('resized_image',resized_image)
    pt('flatten_image',flatten_image)
    return flatten_image, input_label

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

def number_neurons(total_input_size,input_sample_size,output_size,alpha=2):
    """
    :param total_input_size:
    :param input_sample_size:
    :param output_size:
    :param alpha:
    :return: number of neurons for layer
    """
    return int(total_input_size/(alpha*(input_sample_size+output_size)))

def show_image_from_tensor(tensors,image_dimension):
    """Shows images from tensors"""
    for img_index in range(5):
        resize_image = np.reshape(tensors[img_index], [-1, image_dimension])
        Image.fromarray(resize_image).show()