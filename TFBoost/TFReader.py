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

class Reader(object):
    types = set()
    data = []
    trainSetCSV = ''
    validationSetCSV = ''
    testSetCSV = ''
    def __init__(self,setDataFiles,isAnUniqueCSV,knownDataType=''):

        # TODO: Check if knownDataType is empty, number or char.

        if isAnUniqueCSV:
            self.__uniqueDataFile(setDataFiles[0])
        else:
            self.__multipleDataFiles(setDataFiles)

    @timed
    def __uniqueDataFile(self,dataFile):
        """
        This method will be used only when one data file was passed.
        Return train, validation and test sets from an unique file.
        :return: trainSet, validationSet, testSet
        """

        # TODO When the csv has only a type is much better use numpy
        # self.data = np.fromfile(dataFile,dtype = np.float64)
        # Time to execute Breast_Cancer_Wisconsin Data.csv with np.fromfile:  0.0

        self.data = pd.read_csv(dataFile)
        # Time to execute Breast_Cancer_Wisconsin Data.csv with pd.read_csv:  0.007000446319580078
        pt("DataTest Shape",self.data.shape)


    def __multipleDataFiles(self,setDataFiles):
        pass

if __name__ == '__main__':
    print ("Creating Reader")