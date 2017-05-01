"""
Author: @gabvaztor
StartDate: 04/03/2017

With this class you can import a lot of labeled data like Kaggle problems.

- This class not preprocessed de data reducing noise.

To select the csv reader we have followed the following benchmark:
http://softwarerecs.stackexchange.com/questions/7463/fastest-python-library-to-read-a-csv-file

For read data in clusters, we will use "ParaText": http://www.wise.io/tech/paratext

Style: "Google Python Style Guide" 
https://google.github.io/styleguide/pyguide.html
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

# --------------------------------------------------------------------------
'''LOCAL IMPORTS
* UtilsFunctions is a library that contains a lot of functions which will help us
to code expressively, clearly and efficiently.
* TensorFlowGUI's library contains all GUI's methods. Contains EasyGUI.
Here you can download the library: https://pypi.python.org/pypi/easygui#downloads
It had been used the version: 0.98.1
'''

from UsefulTools.UtilsFunctions import *
from TFBoost.TFEncoder import *
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
'''
To install pandas: pip3 install pandas
'''
import pandas as pd
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
'''
Time
'''
import time
# --------------------------------------------------------------------------
'''
Traceback and Os to search
'''
import traceback
import os
# --------------------------------------------------------------------------
import numpy as np
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import tensorflow as tf
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
'''
 Sklearn(scikit-learn): Simple and efficient tools for data mining and data analysis
'''
from sklearn.model_selection import train_test_split
# --------------------------------------------------------------------------

class Reader(object):
    """
    Docs
    """
    # TODO
    types = set()
    data = []
    train_set_csv = ''
    validation_set_csv = ''
    test_set_csv = ''
    train_validation_set = []
    train_set = []
    validation_set = []
    test_set = []

    x_train = []  # Train inputs without labels
    y_train = []  # Train labels without inputs
    x_validation = []  # Validation inputs without labels
    y_validation = []  # Validation labels without inputs
    x_test = []  # Test inputs without labels
    y_test = []  # Test labels without inputs

    reader_features = None  # ReaderFeatures Object

    def __init__(self,reader_features):
        """

        :param reader_features:
        """
        # TODO: Check if knownDataType is empty, number or char.

        self.reader_features = reader_features
        if self.reader_features.is_unique_csv:
            self.uniqueDataFile()
        else:
            self.multipleDataFiles()

    @timed
    def uniqueDataFile(self):
        """
        This method will be used only when one data file was passed.
        Return train, validation and test sets from an unique file.

        """

        # TODO When the csv has only a type is much better use numpy. Use known_data_type
        # self.data = np.fromfile(dataFile,dtype = np.float64)
        # Time to execute Breast_Cancer_Wisconsin Data.csv with np.fromfile:  0.0s

        # TODO Parametrizable delimiter
        self.data = pd.read_csv(self.reader_features.set_data_files[0], delimiter=',')
        # Time to execute Breast_Cancer_Wisconsin Data.csv with pd.read_csv:  0.007000446319580078s
        pt("DataTest Shape",self.data.shape)

        # TODO Create labelData Variable from a list of strings
        # TODO For each pop we have a class
        # TODO Fix this with advanced for <--
        label_data = [self.data.pop(self.reader_features.labels_sets[index]) for index in self.reader_features.labels_sets]  # Data's labels
        pt('label_data', label_data)
        input_data = self.data  # Input data

        trainSize = self.reader_features.train_validation_test_percentages[0]  # first value contains trainSize
        test_size = self.reader_features.train_validation_test_percentages[-1]  # last value contains testSize
        validationSize = None

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input_data,label_data,test_size = test_size )  # Divide set into train and test sets (if it has validation set, into train and validation set for the first part and test set for the second part)

        if self.reader_features.there_is_validation:  # If it has validation percentage

            validationSize = self.reader_features.train_validation_test_percentages[1]  # Get validation percentage
            totalLen = self.data.shape[0]  # All data rows
            # TODO If the data is in columns, we have to take the shape[1] value.
            trainValidationLen = self.x_train.shape[0]  # All train validation rows
            valueValidationPercentage = validationSize * totalLen  # Value of validation percentage in x_train (train and validation)
            validationSize =  valueValidationPercentage / trainValidationLen  # Update validation percentage

            pt("ValidationSize: ",validationSize)
            # TODO Convert sets into Tensors
            self.x_train, self.x_validation, self.y_train, self.y_validation = tf.convert_to_tensor(train_test_split(self.x_train,
                                                                                                self.y_train,
                                                                                                test_size=validationSize))  # Divide train and validation sets into two separate sets.
            # TODO If there is not train and test set with optional validation then Reader will do nothing

    def multipleDataFiles(self):
        """
        Start: 04/04/17 19:30
        :return: train and test sets
        """
        #TODO check nulls
        #TODO lowletters in methods
        features = self.reader_features
        tfSearch = Searcher(features=features)
        tfSearch.findTrainAndTestSetFromPathSignals()

        self.train_set.append(self.x_train)
        self.train_set.append(self.y_train)
        self.test_set.append(self.x_test)
        self.test_set.append(self.y_test)

class ReaderFeatures():
    """ ReaderFeatures Class

    To access Reader class you have to create this object with some parameters.

    Attributes:
    setDataFiles (str): Description of `attr1`.
    isUniqueCSV (:obj:`int`, optional): Description of `attr2`.
    knownDataType
    labels_sets (list: 'str'): Contains all labels values of data.
    train_validation_test_percentages (list:'float',optional): Must contains 2 or 3 percentages values:
        If 3: First is train set, second is validation set and third is test set.
        If 2: First is train set and second test set.
        TODO If none must be randomized

    """
    set_data_files = []
    is_unique_csv = False
    known_data_type = ''
    labels_sets = []
    train_validation_test_percentages = []
    there_is_validation = False
    number_of_classes = None # Number of labels of the input

    def __init__(self, set_data_files,number_of_classes,labels_set = '',
                 is_unique_csv = False,known_data_type = '',
                 percentages_sets = None):

        self.set_data_files = set_data_files
        self.number_of_classes = number_of_classes
        self.is_unique_csv =  is_unique_csv
        self.known_data_type = known_data_type
        self.labels_sets =  labels_set

        # TODO Fix this
        if percentages_sets :  # If it is not None
            if type(percentages_sets) is type([]) \
                    and (len(percentages_sets) is 2 or len(percentages_sets) is 3)\
                    and all(isinstance(x, float) for x in percentages_sets)\
                    and sum(percentages_sets) == 1. \
                    and len([x for x in percentages_sets if x > 0]):  # Must be float list, all values must be float and all values must be positives
                if len(percentages_sets) is 3:
                    self.there_is_validation = True
                    if percentages_sets[1] <= percentages_sets[0]:
                        self.train_validation_test_percentages = percentages_sets
                    else:
                        raise RuntimeError (Errors.validation_error)
            else:
                raise RuntimeError(Errors.percentages_sets)


class Searcher(Reader):
    path_to_read = ''

    def __init__(self,features):
        super(Reader, self).__init__()
        self.path_to_read = features.set_data_files
        self.features = features
    def findTrainAndTestSetFromPathSignals(self):
        """

        :return: Path list from train and test path
        """
        # TODO check nulls
        for path in self.path_to_read:
            for root, dirs, files in os.walk(path):
                for file_name in files:
                    if (file_name.endswith(Dictionary.extension_png)):
                        full_path = os.path.join(root, file_name)
                        self.__getSetsFromFullPathSignals(full_path)



    def __getSetsFromFullPathSignals(self,path):
        """
        If path contains 'train', y_label is two dir up. Else if path contains 'test', y_label is one dir up.
        :param path: the full path
        """
        labels = np.zeros(self.features.number_of_classes, dtype=np.float32)
        if Dictionary.string_train in path: # If 'train' in path
            y_label_dir = os.path.dirname(os.path.dirname(path))  # Directory of directory of file
            y_label = os.path.basename(y_label_dir)
            labels[int(y_label)] = 1
            self.y_train.append(list(labels))
            self.x_train.append(path)
        elif Dictionary.string_test in path: # If 'test' in path
            y_label_dir = os.path.dirname(path)  # Directory of file
            y_label = os.path.basename(y_label_dir)
            labels[int(y_label)] = 1
            self.y_test.append(list(labels))
            self.x_test.append(path)