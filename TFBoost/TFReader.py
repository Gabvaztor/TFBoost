"""
Author: @gabvaztor
StartDate: 04/03/2017

With this class you can import a lot of labeled data like Kaggle problems.

- This class not preprocessed de data reducing noise.

To select the csv reader we have followed the following benchmark:
http://softwarerecs.stackexchange.com/questions/7463/fastest-python-library-to-read-a-csv-file

For read data in clusters, we will use "ParaText": http://www.wise.io/tech/paratext
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

# --------------------------------------------------------------------------
import numpy as np
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
'''
 Sklearn(scikit-learn): Simple and efficient tools for data mining and data analysis
'''
from sklearn.model_selection import train_test_split
# --------------------------------------------------------------------------

class Reader(object):
    types = set()
    data = []
    trainSetCSV = ''
    validationSetCSV = ''
    testSetCSV = ''
    trainValidationSet = []
    trainSet = []
    validationSet = []
    testSet = []

    x_train = []  # Train inputs without labels
    y_train = []  # Train labels without inputs
    x_validation = []  # Validation inputs without labels
    y_validation = []  # Validation labels without inputs
    x_test = []  # Test inputs without labels
    y_test = []  # Test labels without inputs

    rFeatures = None  # ReaderFeatures Object

    def __init__(self,reader_features):

        # TODO: Check if knownDataType is empty, number or char.

        self.rFeatures = reader_features
        if self.rFeatures.isUniqueCSV:
            self.__uniqueDataFile()
        else:
            self.__multipleDataFiles()

    @timed
    def __uniqueDataFile(self):
        """
        This method will be used only when one data file was passed.
        Return train, validation and test sets from an unique file.
        :return: trainSet, validationSet, testSet
        """

        # TODO When the csv has only a type is much better use numpy
        # self.data = np.fromfile(dataFile,dtype = np.float64)
        # Time to execute Breast_Cancer_Wisconsin Data.csv with np.fromfile:  0.0s

        self.data = pd.read_csv(self.rFeatures.setDataFiles[0],delimiter = ',')
        # Time to execute Breast_Cancer_Wisconsin Data.csv with pd.read_csv:  0.007000446319580078s
        pt("DataTest Shape",self.data.shape)

        # TODO Create labelData Variable from a list of strings
        labelData = self.data.pop(self.rFeatures.labelsSet)  # Data's labels
        inputData = self.data  # Input data

        pt("type", type(labelData))
        # TODO SPLIT well the columns and rows
        x_train_validation, y_train_validation, x_test, y_test = train_test_split(inputData.index,labelData,test_size = 0.2)
        #self.trainSet,self.testSet = train_test_split(self.data, test_size = 0.2)

        pt("labelData", labelData.shape)
        pt("inputData", inputData.shape)
        pt("x_test", x_test.shape)
        pt("y_test", y_test.shape)
        pt("labelData", labelData.shape)


    def __multipleDataFiles(self,setDataFiles):
        pass

class ReaderFeatures():
    """
    Reader features

    To access Reader class you have to create this object with some parameters
    """
    setDataFiles = []
    isUniqueCSV = False
    knownDataType = ''
    labelsSet = []

    def __init__(self,set_data_files,labels_set = '',
                 is_unique_csv = False,known_data_type = ''):

        self.setDataFiles = set_data_files
        self.isUniqueCSV =  is_unique_csv
        self.knownDataType = known_data_type
        self.labelsSet =  labels_set

if __name__ == '__main__':
    print ("Creating Reader")