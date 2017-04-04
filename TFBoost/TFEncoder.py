"""
Author: @gabvaztor
StartDate: 11/03/2017

This class contains a dictionary with keys-values
"""

"""
Accessible variable
"""

dict = {} # Key-Value variable

"""
Dictionary values with comments
"""

# Data.csv from Breast_Cancer_Wisconsin project
dict['breast_Cancer_Wisconsin_Path'] = '../DataTest/Breast_Cancer_Wisconsin/data.csv'

def dict(key):
    """
    Function to return the value from a key
    :param key:
    :return: The Value
    """
    # TODO Check valid values
    return dict[key]

class Dictionary(object):
    """
    Dictionary:Encoder
    Each attribute will be a dictionary's key
    Attribute = Dictionary Key
    Value's Attribute = Dictionary Value
    Doing an abstract class like a Dictionary we have the references to all keys from another class.
    """

    # Data.csv from Breast_Cancer_Wisconsin project
    path_Breast_Cancer_Wisconsin = '../DataTest/Breast_Cancer_Wisconsin/data.csv'
    # Data.csv Label Column name
    label_column_name_Breast_Cancer_Wisconsin = 'diagnosis'
    # Signal University path train
    path_signals_university_train = 'D:\\UniversityResearching\\DITS-classification\\classification_train\\'
    # Signal University path test
    path_signals_university_test = 'D:\\UniversityResearching\\DITS-classification\\classification_test\\'
    # Hierarchy option
    labels_type_option_hierarchy = 'hierarchy'
    # Format png
    extension_png = '.png'
    # Train String
    string_train = 'train'
    # Test String
    string_test = 'test'

class Errors(object):
    """
    Error Class
    This class contains all possibles errors.
    All Raises or exceptions will call this class.
    """

    # Validation error in percentages_set
    validation_error = "'ReaderFeatures.percentages_sets.validation' Must be lower or equal than train percentage."

    # Check the correct structure in ReaderFeatures.percentages_sets
    percentages_sets = "ReaderFeatures.percentages_sets needs to be a list with 2 or 3 positives float values " \
                       "and must sum 1."

class Constant(object):
    """
    Constant Class
    This class contains all variable constants.
    All variables with a variable value will call this class.
    """
    # Weight first patch
    w_first_patch = 5
    # Weight second patch
    w_second_patch = 5
    # Weight number of inputs
    w_number_inputs = 1
    # Weight first label neurons
    first_label_neurons = 32


