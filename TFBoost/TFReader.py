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
    """
    Docs
    """
    # TODO
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
        """

        :param reader_features:
        """
        # TODO: Check if knownDataType is empty, number or char.

        self.rFeatures = reader_features
        if self.rFeatures.isUniqueCSV:
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

        self.data = pd.read_csv(self.rFeatures.setDataFiles[0],delimiter = ',')
        # Time to execute Breast_Cancer_Wisconsin Data.csv with pd.read_csv:  0.007000446319580078s
        pt("DataTest Shape",self.data.shape)

        # TODO Create labelData Variable from a list of strings
        # TODO For each pop we have a class
        label_data = self.data.pop(self.rFeatures.labelsSet)  # Data's labels
        inputData = self.data  # Input data

        trainSize = self.rFeatures.trainValidationTestPercentage[0]  # 0 contains trainSize
        testSize = self.rFeatures.trainValidationTestPercentage[-1]  # last contains testSize
        validationSize = None

        self.x_train, self.x_test, self.y_train, self.y_test  = train_test_split(inputData,label_data,test_size = testSize )  # Divide set into train and test sets (if it has validation set, into train and validation set for the first part and test set for the second part)

        if self.rFeatures.thereIsValidation:  # If it has validation percentage

            validationSize = self.rFeatures.trainValidationTestPercentage[1]  # Get validation percentage
            totalLen = self.data.shape[0]  # All data rows
            # TODO If the data is in columns, we have to take the shape[1] value.
            trainValidationLen = self.x_train.shape[0]  # All train validation rows
            valueValidationPercentage = validationSize * totalLen  # Value of validation percentage in x_train (train and validation)
            validationSize =  valueValidationPercentage / trainValidationLen  # Update validation percentage

            pt("ValidationSize: ",validationSize)
            # TODO Convert sets into Tensors
            self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(self.x_train,
                                                                                                self.y_train,
                                                                                                test_size=validationSize)  # Divide train and validation sets into two separate sets.
            # TODO If there is not train and test set with optional validation then Reader will do nothing

    def multipleDataFiles(self):
        pass

class ReaderFeatures():
    """ ReaderFeatures Class

    To access Reader class you have to create this object with some parameters.

    Attributes:
    setDataFiles (str): Description of `attr1`.
    isUniqueCSV (:obj:`int`, optional): Description of `attr2`.
    knownDataType
    labelsSet (list: 'str'): Contains all labels values of data.
    trainValidationTestPercentage (list:'float',optional): Must contains 2 or 3 percentages values:
        If 3: First is train set, second is validation set and third is test set.
        If 2: First is train set and second test set.
        TODO If none must be randomized

    """
    setDataFiles = []
    isUniqueCSV = False
    knownDataType = ''
    labelsSet = []
    trainValidationTestPercentage = []
    thereIsValidation = False
    number_of_classes = None # Number of labels of the input

    def __init__(self,set_data_files,number_of_classes,labels_set = '',
                 is_unique_csv = False,known_data_type = '',
                 percentages_sets = None,):

        self.setDataFiles = set_data_files
        self.number_of_classes = number_of_classes
        self.isUniqueCSV =  is_unique_csv
        self.knownDataType = known_data_type
        self.labelsSet =  labels_set

        if percentages_sets :  # If it is not None
            if type(percentages_sets) is type([]) and (len(percentages_sets) is 2 or len(percentages_sets) is 3)\
                    and all(isinstance(x, float) for x in percentages_sets)\
                    and sum(percentages_sets) is 1 \
                    and len([x for x in percentages_sets if x > 0]):  # Must be float list, all values must be float and all values must be positives
                if len(percentages_sets) is 3:
                    self.thereIsValidation = True
                    if percentages_sets[1] <= percentages_sets[0]:
                        self.trainValidationTestPercentage = percentages_sets
                    else:
                        raise RuntimeError (Errors.validation_error)
            else:
                raise RuntimeError(Errors.percentages_sets)

if __name__ == '__main__':
    print ("Creating Reader")