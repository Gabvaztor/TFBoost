# -*- coding: utf-8 -*-
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

from TFBoost.TFEncoder import Dictionary as Dictionary
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
import collections
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
'''
 Sklearn(scikit-learn): Simple and efficient tools for data mining and data analysis
'''
from sklearn.model_selection import train_test_split
# --------------------------------------------------------------------------




class Reader(object):
    """
    DOCS...
    """
    # TODO
    train_set = []
    validation_set = []
    test_set = []

    x_train = []  # Train inputs without labels
    y_train = []  # Train labels without inputs
    x_validation = []  # Validation inputs without labels
    y_validation = []  # Validation labels without inputs
    x_test = []  # Test inputs without labels
    y_test = []  # Test labels without inputs

    number_classes = None  # Represent number of columns in csv without labels
    reader_features = None  # ReaderFeatures Object

    def __init__(self, type_problem, reader_features=None, settings=None, paths_to_read=None, number_of_classes=None, delimiter=";",
                 labels_set=None, is_unique_file=None, known_data_type=None,
                 percentages_sets=None):
        """
        :param reader_features: 
        :param paths_to_read: 
        :param number_of_classes: 
        :param delimiter: 
        :param type_problem: Represent the id to difference one problem from another
        :param labels_set: 
        :param is_unique_file: 
        :param known_data_type: 
        :param percentages_sets: 
        :param settings:
        """
        # TODO (@gabvaztor) DOCs
        self.paths_to_read = paths_to_read
        self.number_of_classes = number_of_classes
        self.is_unique_file = is_unique_file
        self.known_data_type = known_data_type
        self.labels_sets = labels_set
        self.there_is_validation, self.train_validation_test_percentages = self.calculate_percentages(percentages_sets)
        self.reader_features = reader_features
        self.delimiter = delimiter
        self.settings = settings
        if reader_features:
            if self.reader_features.is_unique_csv:
                self.unique_data_file(type_problem)
            else:
                self.multiple_data_files(type_problem)
        elif self.is_unique_file:
            self.unique_data_file(type_problem)
        else:
            self.multiple_data_files(type_problem)

    @timed
    def unique_data_file(self, type_problem):
        """
        This method will be used only when one data file was passed.
        Return train, validation and test sets from an unique file.		
        """
        if type_problem == Dictionary.string_breast_cancer_wisconsin_problem:
            self.read_generic_problem()

    def multiple_data_files(self, type_problem):
        """ 
        Start: 04/04/17 19:30
        
        :return: train and test sets
        """
        # TODO check nulls
        # TODO low letters in methods

        # TODO (@gabvaztor) Create new path in setting with "DATASET_PATH"
        # By defect, saves in model path (without "model") string
        path_to_save = self.settings.model_path  # path\\model --> path\\
        path_to_save = path_to_save[0:-5]

        if type_problem == Dictionary.string_option_signals_images_problem:
            # TODO(@gabvaztor) Change this to use new structure
            features = self.reader_features
            tf_search = Searcher(features=features, reader=self)
            tf_search.find_train_and_test_sets_from_path_signals()
            self.create_and_save_flag_sets(test=True)
        elif type_problem == Dictionary.string_option_web_traffic_problem:
            self.read_web_traffic_data_and_create_files(is_necessary_create_files=False)
        elif type_problem == Dictionary.string_option_retinopathy_k_problem:
            features = self.reader_features
            tf_search = Searcher(features=features, reader=self)
            tf_search.get_fullpath_and_execute_problem_operation(problem=type_problem)
            self.create_and_save_flag_sets(test=True, save_to_file=True, path_to_save=path_to_save)

    def create_and_save_flag_sets(self, validation=False, test=False, save_to_file=False, path_to_save=None):

        self.x_train = np.asarray(self.x_train)
        self.y_train = np.asarray(self.y_train)
        self.x_test = np.asarray(self.x_test)
        self.y_test = np.asarray(self.y_test)
        self.x_validation = np.asarray(self.x_validation)
        self.y_validation = np.asarray(self.y_validation)

        self.train_set.append(self.x_train)
        self.train_set.append(self.y_train)

        # Append to lists
        self.test_set.append(self.x_test)
        self.test_set.append(self.y_test)
        self.validation_set.append(self.x_validation)
        self.validation_set.append(self.y_validation)

        if save_to_file:
            np_arrays = [self.x_train, self.y_train, self.x_test, self.y_test, self.x_validation, self.y_validation]
            names = ["x_train", "y_train", "x_test", "y_test", "x_validation", "y_validation"]
            save_numpy_arrays_generic(folder_to_save=path_to_save, names=names,numpy_files=np_arrays)

    def calculate_percentages(self, percentages_sets):
        """
        
        :param percentages_sets: list of percentages
        :return: 
        """
        # TODO (@gabvaztor)
        there_is_validation = False
        train_validation_test_percentages = None
        if percentages_sets:  # If it is not None
            percentages_sets_sum = convert_to_decimal(percentages_sets)
            if type(percentages_sets) is type([]) \
                    and (len(percentages_sets) is 2 or len(percentages_sets) is 3) \
                    and all(isinstance(x, float) for x in percentages_sets) \
                    and (percentages_sets_sum == 1.0) \
                    and len([x for x in percentages_sets if
                             x > 0]):  # Must be float# list, all values must be float and all values must be positives
                if len(percentages_sets) is 3:
                    there_is_validation = True
                    if percentages_sets[1] <= percentages_sets[0]:
                        train_validation_test_percentages = percentages_sets
                    else:
                        raise RuntimeError(Errors.validation_error)
                else:
                    train_validation_test_percentages = percentages_sets
            else:
                raise RuntimeError(Errors.percentages_sets)
        return there_is_validation, train_validation_test_percentages

    @timed
    def read_web_traffic_data_and_create_files(self, is_necessary_create_files=False):
        """
        Create 9 csv files each one with "Page_Date,Visits" as header. 
        Note: The train_1.csv file must have 145063 rows with header 
        It useful one time. If you have created the files, then is_necessary_create_files need to be false.
        Attributes: 
            
            is_necessary_create_files: If True, then use this method to create files. Else it is because you have 
            created files before.
            
        """
        if is_necessary_create_files:
            pt('Reading data from ...')
            key_1 = pd.read_csv(self.paths_to_read[1], encoding="utf-8")
            train_1 = pd.read_csv(self.paths_to_read[0], encoding="utf-8")
            #ss_1 = pd.read_csv(self.paths_to_read[2])
            pt('Preprocessing...', "Changing NaN by 3")
            train_1.fillna(3, inplace=True)
            pt('Processing...')
            ids = key_1.Id.values
            pages2 = key_1.Page.values
            print('train_1...')
            pages = list(train_1.Page.values)
            columns_list = list(train_1.columns.values)
            columns_list.pop(0)
            pt("Train_1", "Getting values...")
            train_values = train_1.get_values()
            del train_1
            pages_with_date_and_label = {}
            to_save = "D:\\Machine_Learning\\Competitions\\Kaggle_Data\\Web_Traffic_Time\\Trains\\"
            part = 1
            csv = Dictionary.string_csv_extension
            pt("Train_1", "Start for...")
            for index_page in range(len(pages)):
                for index_date in range(len(columns_list)):
                    if index_page % 16118 == 0 and index_date == 0 and index_page != 0:
                        path_to_save = to_save + str(part) + csv
                        save_submission_to_csv(path_to_save, pages_with_date_and_label)
                        part += 1
                        pages_with_date_and_label = {}
                    page_with_date = pages[index_page] + Dictionary.string_char_low_stripe + str(columns_list[index_date])
                    value = train_values[index_page][index_date+1]
                    pages_with_date_and_label[page_with_date] = value
                    if index_page % 1000 == 0 and index_date == 0:
                        pt("index_page", index_page)
            path_to_save = to_save + str(part) + csv
            save_submission_to_csv(path_to_save, pages_with_date_and_label)
            pt("END Creating files ")

    def read_generic_problem(self):
        # TODO When the csv has only a type is much better use numpy. Use known_data_type
        # self.data = np.fromfile(dataFile,dtype = np.float64)
        # Time to execute Breast_Cancer_Wisconsin Data.csv with np.fromfile:  0.0s

        # TODO Parametrizable delimiter
        # TODO Do delimiter and enconding as parameter
        self.data = pd.read_csv(self.reader_features.set_data_files[0], delimiter=self.delimiter, encoding="ISO-8859-1")
        # Time to execute Breast_Cancer_Wisconsin Data.csv with pd.read_csv:  0.007000446319580078s
        pt("DataTest Shape", self.data.shape)

        # TODO Create labelData Variable from a list of strings
        # TODO For each pop we have a class
        # TODO Fix this with advanced for <--
        label_data = np.asarray([self.data.pop(self.reader_features.labels_sets[0])], dtype=np.float32)  # Data's labels
        # label_data = label_data.transpose()
        input_data = self.data  # Input data
        # self.number_classes = len(self.data.columns)
        trainSize = self.reader_features.train_validation_test_percentages[0]  # first value contains trainSize
        test_size = self.reader_features.train_validation_test_percentages[-1]  # last value contains testSize
        validationSize = None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input_data, label_data,
                                                                                test_size=test_size)
        # Divide set into train and test sets (if it has validation set, into train and validation set for the first part and test set for the second part)

        if self.reader_features.there_is_validation:  # If it has validation percentage

            validationSize = self.reader_features.train_validation_test_percentages[1]  # Get validation percentage
            totalLen = self.data.shape[0]  # All data rows
            # TODO If the data is in columns, we have to take the shape[1] value.
            trainValidationLen = self.x_train.shape[0]  # All train validation rows
            valueValidationPercentage = validationSize * totalLen  # Value of validation percentage in x_train (train and validation)
            validationSize = valueValidationPercentage / trainValidationLen  # Update validation percentage

            pt("ValidationSize: ", validationSize)
            # TODO Convert sets into Tensors
            self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(self.x_train,
                                                                                                self.y_train,
                                                                                                test_size=validationSize)  # Divide train and validation sets into two separate sets.
            # TODO If there is not train and test set with optional validation then Reader will do nothing
        self.load_sets()

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

        if percentages_sets :  # If it is not None
            percentages_sets_sum = convert_to_decimal(percentages_sets)
            if type(percentages_sets) is type([])\
                    and (len(percentages_sets) is 2 or len(percentages_sets) is 3)\
                    and all(isinstance(x, float) for x in percentages_sets)\
                    and (percentages_sets_sum == 1.0)\
                    and len([x for x in percentages_sets if x > 0]):  # Must be float# list, all values must be float and all values must be positives
                if len(percentages_sets) is 3:
                    self.there_is_validation = True
                    if percentages_sets[1] <= percentages_sets[0]:
                        self.train_validation_test_percentages = percentages_sets
                    else:
                        raise RuntimeError (Errors.validation_error)
                else:
                    self.train_validation_test_percentages = percentages_sets
            else:
                raise RuntimeError(Errors.percentages_sets)




class Searcher(Reader):

    def __init__(self, features, reader):
        super(Reader, self).__init__()
        self.path_to_read = features.set_data_files
        self.features = features
        self.reader = reader

    def get_fullpath_and_execute_problem_operation(self, problem):
        """
        Generic class to find a fullpath and do an specific operation (function) to a given problem.
        """
        pt("Creating train and test/validation data...")
        setting_object = self.reader.settings

        dataframe_labels = None

        if setting_object.labels_path:
            labels_path = setting_object.labels_path
            if problem == Dictionary.string_option_retinopathy_k_problem:
                # Read CSV Labels
                # TODO (@gabvaztor) Do generic import if more than one problem use it
                import pandas as pd
                dataframe_labels = pd.read_csv(filepath_or_buffer=labels_path)

        start_time = time.time()
        for path in self.path_to_read:
            for root, dirs, files in os.walk(path):
                for x in files:
                    lene = len("10_left.jpeg")
                    if "_" not in x and "left" not in x and "right" not in x:
                        pass
                    if len(x) == 12 or x == "1099999_left.jpeg" or len(x) == lene or x == "109979_left.jpeg":
                        pass
                for count_number, file_name in enumerate(files):

                    pt("Files Size", len(files))
                    pt("Count number", count_number)
                    progress = float(((count_number*100)/len(files)))
                    progress = "{0:.3f}".format(progress)
                    pt("Progress percent",  progress + "%")

                    if problem == Dictionary.string_option_retinopathy_k_problem:
                        if (file_name.endswith(Dictionary.string_extension_jpeg)):
                            full_path = os.path.join(root, file_name)
                            labels = np.zeros(self.features.number_of_classes, dtype=np.float32)
                            name = os.path.splitext(file_name)[0]
                            if np.where(dataframe_labels["image"] == name)[0]:
                                index = int(np.where(dataframe_labels["image"] == name)[0][0])
                                label = int(dataframe_labels.loc[[index]]["level"].iloc[0])
                                labels[label] = 1
                                # To save
                                if Dictionary.string_train in path:
                                    self.y_train.append(list(labels))
                                    self.x_train.append(full_path)
                                if Dictionary.string_test in path:
                                    self.y_test.append(list(labels))
                                    self.x_test.append(full_path)
                    elif problem == Dictionary.string_option_signals_images_problem:
                        self.find_train_and_test_sets_from_path_signals()
        pt('Time to create data_sets', str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))
        pt("Finish creating train and test/validation data...")

    def find_train_and_test_sets_from_path_signals(self):
        """
        :return: Paths list from train and test path
        """
        for path in self.path_to_read:
            for root, dirs, files in os.walk(path):
                for file_name in files:
                    if (file_name.endswith(Dictionary.string_extension_png)):
                        full_path = os.path.join(root, file_name)
                        self._get_sets_from_full_path_signals(full_path)

    def _get_sets_from_full_path_signals(self, path):
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

def build_dataset(words):
    # TODO
    words = "hola,hola,hola,hola"
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary