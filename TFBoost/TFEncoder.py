"""
Author: @gabvaztor
StartDate: 11/03/2017

This class contains a dictionary with keys-values

Style: "Google Python Style Guide" 
https://google.github.io/styleguide/pyguide.html
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
    """
    STRINGS
    """

    # German Prices
    string_german_prices_csv_path = "D:\Machine_Learning\DataSets\German-Prices\GermanyPrices2005-2016.csv"
    # Settings path (for save models) (it has the same configuration that kaggle submit best practice)
    #string_settings_path = "../Settings/Sberbank_Russian_Houssing_Market/SETTINGS.json"
    string_settings_path = "../Settings/German_Signal/SETTINGS.json"
    # Data.csv from Breast_Cancer_Wisconsin project
    string_path_Breast_Cancer_Wisconsin = '../DataTest/Breast_Cancer_Wisconsin/data.csv'
    # Data.csv Label Column name
    string_label_column_name_Breast_Cancer_Wisconsin = 'diagnosis'
    # Signal University path train
    string_path_signals_university_signal_train = 'D:\\UniversityResearching\\DITS-classification\\classification_train\\'
    #string_path_signals_university_signal_train = 'D:\\DITS-classification\\classification_train\\'
    # Signal University path test
    string_path_signals_university_signal_test = 'D:\\UniversityResearching\\DITS-classification\\classification_test\\'
    #string_path_signals_university_signal_test = 'D:\\DITS-classification\\classification_test\\'
    # Hierarchy option
    string_labels_type_option_hierarchy = 'hierarchy'
    # Format png
    string_extension_png = '.png'
    # Format ckpt
    string_ckpt_extension = '.ckpt'
    # Train String
    string_train = 'train'
    # Test String
    string_test = 'test'
    # Separator
    string_separator = '-------------------------------------'
    # Gray scale
    string_gray_scale = 'gray_scale'
    # Want to save model
    string_want_to_save = "Do you want to save the model? Press 'Y' to Yes or 'N' to No: \n Answer:"
    # Char Y
    string_char_Y = "Y"
    # Char N
    string_char_N = "N"
    """
    STRINGS OPTIONS
    """
    # Option signals image problem
    string_option_signals_images_problem = 'signals_images_problems'
    # Option german prizes problem
    string_option_german_prizes_problem = 'german_prizes_problem'


class Errors(object):
    """
    Error Class
    This class contains all possibles errors.
    All Raises or exceptions will call this class.
    """

    # Validation error in percentages_set
    validation_error = "'ReaderFeatures.percentages_sets.validation' Must be lower or equal than train percentage."
    # Validation error in percentages_set
    can_not_restore_model_because_path_not_exists = "Can not restore model because file not exists."
    # Validation error in percentages_set
    check_dir_exists_and_create = "Error checking file and creating folders."
    # Check the correct structure in ReaderFeatures.percentages_sets
    percentages_sets = "ReaderFeatures.percentages_sets needs to be a list with 2 or 3 positives float values " \
                       "and must sum 1."
    # Not find a label for input
    not_label_from_input = "There is not label for input"
    # Not find a label for input
    write_string_to_file = "Can't write to file. Make sure the file exists and you have privileges to write"
    # Error
    error = "Error, see description"
    # Setting Object model path bad config
    model_path_bad_configuration = "Can't save model because model path is bad configured"
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
    # First label neurons
    first_label_neurons = 32
    # Second label neurons
    second_label_neurons = 16
    # Third label neurons
    third_label_neurons = 8


