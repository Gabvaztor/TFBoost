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

from pandas.core.indexing import _iAtIndexer

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
'''"Best image library"
pip install opencv-python'''
import cv2

""" Random to suffle lists """
import random
""" Time """
import time
"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- GLOBAL VARIABLES ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""
# TODO Must be in class attribute
index_buffer_data = 0  # The index for batches during training
inputs_processed, labels_processed = [],[]  # New inputs and labels processed for training. (Change during shuffle)

    # TODO Implement deep learning algorithms
    # TODO  Use tflearn to use basics algorithms

def lineal_model_basic_with_gradient_descent(self, input, test, input_labels, test_labels, number_of_inputs,
                                             number_of_classes,
                                             learning_rate=0.015, trains=100, type=None, validation=None,
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
    # TODO Create placeholders with variable shapes
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
@timed
def convolution_model_image(input, test, input_labels, test_labels, number_of_classes, number_of_inputs=None,
                      learning_rate=1e-3, trains=None, type=None, validation=None,
                      validation_labels=None, deviation=None):
    """
    Generic convolutional model
    """

    # TODO Create an simple but generic convolutional model to analyse sets.
    show_info = 0 # Labels and logits info.
    show_images = 0 # if True show images when show_info is True
    shuffle_data = True
    to_array = True  # If the images must be reshaped into an array
    x1_rows_number = 60
    x1_column_number = 60
    x_columns = x1_rows_number*x1_column_number
    x_rows_column = [x1_rows_number,x1_column_number]
    kernel_size = [5, 5]  # Kernel patch size
    input_size = len(input)
    num_epoch = 100  # Epochs number
    #batch_size = int(input_size/10)+1  # Batch size
    batch_size = 100  # Batch size
    # capacity must be larger than min_after_dequeue and the amount larger
    # determines the maximum we will prefetch.
    # capacity = int(input_size / 4)
    '''
    first_label_neurons = number_neurons(input_size, batch_size, number_of_classes)  # Weight first label neurons
    second_label_neurons = number_neurons(input_size, first_label_neurons, number_of_classes)  # Weight second label neurons
    third_label_neurons = number_neurons(input_size, second_label_neurons, number_of_classes)  # Weight third label neurons
    '''
    first_label_neurons = 50
    second_label_neurons = 55
    third_label_neurons = 50
    if not trains:
        trains = int(input_size/batch_size)+1

    pt('first_label_neurons', first_label_neurons)
    pt('second_label_neurons', second_label_neurons)
    pt('third_label_neurons', third_label_neurons)
    pt('input_size', input_size)
    pt('batch_size', batch_size)
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
        filters=third_label_neurons,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)
    # Pool Layer 1 and reshape images by 2
    pool1 = tf.layers.max_pooling2d(inputs=convolution_1, pool_size=[2, 2], strides=2)
    # Second Convolutional Layer

    convolution_2 = tf.layers.conv2d(
        inputs=pool1,
        filters=third_label_neurons,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)

    # # Pool Layer 2 nd reshape images by 2
    pool2 = tf.layers.max_pooling2d(inputs=convolution_2, pool_size=[2, 2], strides=2)
    # Dense Layer
    # TODO Checks max pools numbers
    pool2_flat = tf.reshape(pool2, [-1, int(x1_rows_number/4) * int(x1_column_number/4) * third_label_neurons])
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
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)  # Adam Optimizer (gradient descent)  # 97-98 test
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)  # Adam Optimizer (gradient descent)
    # Sure is axis = 1
    correct_prediction = tf.equal(tf.argmax(y_convolution, axis=1), tf.argmax(y_, axis=1))  # Get Number of right values in tensor
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Get accuracy in float
    
    '''
    # Creating BATCH
    inputs_array = np.array(input)
    labels_array = np.array(input_labels)
    # pt([np.argmax(f,axis=0) for f in input_labels[40:60]])
    inputs_tensor = tf.convert_to_tensor(inputs_array, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(labels_array, dtype=tf.float32)
    inputs_and_labels = [inputs_tensor, labels_tensor]
    '''
    # TODO BUG when try to put labels with shape (x,)
    # TODO BUG: With Signal Data: When you put a batch of 15 or higher, the output from
    # TODO tf.train.batch randomize input with label so create a big problem for supervised model
    '''
    # Slice inputs and labels into one example
    train_input_queue = tf.train.slice_input_producer(inputs_and_labels,
                                                    shuffle=False,num_epochs=num_epoch)  # List of files to read with extension '.png'
    # Obtain real value and real label in tensor
    input_processed, label_processed =\
        read_from_reader_signal_competition(train_input_queue,x_rows_column)
    pt('input_processed', input_processed)
    # Batching values and labels from input_processed (with batch size)
    x_batch, label_batch = tf.train.batch(
        [input_processed, label_processed],
        batch_size=batch_size, capacity=32, allow_smaller_final_batch=True)
    '''
    # Batching values and labels from input and labels (with batch size)
    x_batch_feed, label_batch_feed = get_data_buffer_images(input, input_labels,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_data, to_type=Dictionary.gray_scale,
                                                  x_rows_column=x_rows_column, to_array=to_array,is_test=False)
    x_test_feed, y_test_feed = get_data_buffer_images(test, test_labels,
                                                  batch_size=len(test),
                                                  shuffle=False, to_type=Dictionary.gray_scale,
                                                  x_rows_column=x_rows_column, to_array=to_array, is_test=True)
    # Session
    sess = tf.InteractiveSession()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # initialize the queue threads to start to shovel data
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(coord=coord)
    num_trains_acum = 0
    start_time = time.time()
    # TRAIN
    for epoch in range (num_epoch):
        for i in range(trains):
            '''
            Test
            '''
            
            # Setting values
            feed_dict_train_50 = {x: x_batch_feed, y_: label_batch_feed, keep_probably: 0.5}
            feed_dict_train_100 = {x: x_batch_feed, y_: label_batch_feed, keep_probably: 1}
            feed_dict_test_100 = {x: x_test_feed, y_: y_test_feed, keep_probably: 1}
            train_accuracy = accuracy.eval(feed_dict_train_100) * 100
            train_step.run(feed_dict_train_50)
            test_accuracy = accuracy.eval(feed_dict_test_100)*100

            # validation_accuracy = accuracy.eval(feed_dict={x: validationPlaceholder, y_: validationLabels, keep_probably: 1.0}) * 100

            crossEntropyTrain = cross_entropy.eval(feed_dict_train_100)
            # crossEntropyValidation = cross_entropy.eval( feed_dict={x: validationPlaceholder, y_: validationLabels, keep_probably: 1.0})
            #pt('conv1',tf.argmax(y_convolution, axis=1).eval(feed_dict_train_100))
            y_pt = tf.argmax(y_, axis=1).eval(feed_dict_train_100)
            #first_image = convolution_1.eval(feed_dict_train_100)[0]
            # pt('convolution_1[shape]',first_image.shape)

            # TODO Use validation set
            # if show_info and epoch == 0 and i == 0:
            if show_info:
                #pt('the image', x_batch_feed.shape)

                y__ = y_.eval(feed_dict_train_100)
                argmax_labels_y_ = [np.argmax(m) for m in y__]
                #pt('y__', argmax_labels_y_)
                pt('y__shape', y__.shape)
                pt('argmax_labels_y__', argmax_labels_y_)
                pt('y__[-1]', y__[-1])

                y__conv = y_convolution.eval(feed_dict_train_100)
                argmax_labels_y_convolutional = [np.argmax(m) for m in y__conv]
                pt('argmax_y_conv', argmax_labels_y_convolutional)
                pt('y_conv_shape', y__conv.shape)
                #pt('y__conv', y__conv)
                pt('index_buffer_data', index_buffer_data)
                if show_images and i == 0:
                    show_image_from_tensor(x_batch_feed,x_rows_column) # shows images
            if i == 0 or i % 10 == 0:
                percent_avance = str(i*100/trains)
                pt('Seconds', (time.time() - start_time))
                pt('Time', str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))
                pt('TRAIN NUMBER: '+str(num_trains_acum+1) + ' | Percent Epoch ' + str(epoch+1) + ": " + percent_avance + '%')
                pt('train_accuracy',train_accuracy)
                pt('crossEntropyTrain',crossEntropyTrain)
                pt('test_accuracy',test_accuracy)
            num_trains_acum += 1
            # pt('correct_prediction',correct_prediction.eval())
            #pt('y_conv',y_convolution.eval(feed_dict={x: x_batch_feed, y_: label_batch_feed, keep_probably: 1.0}) * 100)
            # TODO Change to get without arguments (DO CLASS)
            # Update batchs values
            x_batch_feed, label_batch_feed = get_data_buffer_images(input, input_labels,
                                                                batch_size=batch_size,
                                                                shuffle=shuffle_data, to_type=Dictionary.gray_scale,
                                                                x_rows_column=x_rows_column, to_array=to_array)

    # When finish coord
    #coord.request_stop()
    #coord.join(threads)
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
    initial = tf.truncated_normal(shape, stddev=0.01)
    #initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def number_neurons(total_input_size,input_sample_size,output_size,alpha=1):
    """
    :param total_input_size:
    :param input_sample_size:
    :param output_size:
    :param alpha:
    :return: number of neurons for layer
    """
    return int(total_input_size/(alpha*(input_sample_size+output_size)))

def show_image_from_tensor(images,x_rows_column):
    """Shows images from tensors"""
    for img_index in range(images.shape[0]):
        img = images[img_index].reshape([x_rows_column[0],x_rows_column[1]])
        pt("images", img.shape)
        cv2.imshow('image', img)
        cv2.waitKey(0)  # Wait until press key to destroy image
        #cv2.destroyAllWindows()  # Destroy all windows

# Redoit when this file be a class
def get_data_buffer_images(inputs, inputs_labels, batch_size, shuffle=False,to_type=None,x_rows_column=None,
                           to_array=False,is_test=False):
    """
    Return a x_input and y_labels with a batch_size and the same order.
    Inputs and inputs_label must have same shape.
    If next batch is out of range then takes until last element.
    :param input:
    :param input_labels:
    :param batch_size:
    :param index: # TODO make global in class attribute
    :param shuffle:
    :param x_rows_column: A list [row,column] if want to reshape the images in that shape
    :param to_array: If result must be and array and not a matrix
    :return:
    """
    # TODO DOCS
    # TODO, change when class
    # TODO, Another method when test
    global index_buffer_data
    global inputs_processed, labels_processed
    x_inputs = []
    y_labels = []
    out_range = False  # True if next batch is out of range
    image_type = None # Represent the type of image (to_type)
    # TODO Change checks when class
    if to_type is not None:
        if to_type == Dictionary.gray_scale:
            image_type = cv2.IMREAD_GRAYSCALE
    if not is_test:
        if shuffle and index_buffer_data == 0:
            c = list(zip(inputs, inputs_labels))
            random.shuffle(c)
            inputs_processed, labels_processed = zip(*c)
        elif index_buffer_data == 0:
            inputs_processed, labels_processed = inputs, inputs_labels
        # TODO Change if array
        if len(inputs) - index_buffer_data == 0: # When is all inputs
            out_range = True
        elif len(inputs) - index_buffer_data < batch_size:
            batch_size = len(inputs) - index_buffer_data
            out_range = True
    for _ in range(batch_size):
        # Reshape
        if x_rows_column is not None:
            img = preprocess_image(inputs_processed[index_buffer_data],image_type,x_rows_column[0],x_rows_column[1])
        if to_array:
            img = img.reshape(-1)
        x_inputs.append(img)
        y_labels.append(labels_processed[index_buffer_data])
        index_buffer_data+=1
    x_inputs = np.asarray(x_inputs)
    y_labels = np.asarray(y_labels)
    if out_range and not is_test:  # Reset index_buffer_data
        index_buffer_data = 0
    #TODO Check errors
    return x_inputs, y_labels

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def preprocess_image(image, image_type, height, width):
    """

    :param image: The image to change
    :param image_type: Gray Scale, RGB, HSV
    :return:
    """
    # TODO Realize this with ALL inputs and use the returned sets to train
    # TODO Normalize image
    # 1- Get image in GrayScale
    # 2- Modify intensity and contrast
    # 3- Transform to gray scale
    # 4- Return image
    image = cv2.imread(image,0)
    image = cv2.resize(image, (height, width))
    image = cv2.equalizeHist(image)
    image = cv2.equalizeHist(image)
    random_percentage = random.randint(3,20)
    to_crop_height = int((random_percentage*height)/100)
    to_crop_width = int((random_percentage*width)/100)
    image = image[to_crop_height:height-to_crop_height, to_crop_width:width-to_crop_width]
    #image = np.array(image, dtype = np.float64)
    #random_bright = .5+np.random.uniform()
    #image[:,:,2] = image[:,:,2]*random_bright
    #image[:,:,2][image[:,:,2]>255]  = 255
    #image = cv2.normalize(image,image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.copyMakeBorder(image, top = to_crop_height,
                               bottom = to_crop_height,
                               left = to_crop_width,
                               right = to_crop_width,
                               borderType = cv2.BORDER_CONSTANT)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)  # Wait until press key to destroy image
    return image
    # TODO Scale image to center into figure
# Load an color image in grayscale (0 is gray scale)


class TFModels():
    """
    Long Docs ...
    """
    # TODO Docs
    def __init__(self,input, test, input_labels, test_labels, number_of_classes, number_of_inputs=None,
                      learning_rate=1e-3, trains=None, type=None, validation=None,
                      validation_labels=None, deviation=None):
        pass

