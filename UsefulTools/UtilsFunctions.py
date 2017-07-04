"""
This class contains useful functions
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""
"""LOCAL IMPORTS"""
from TFBoost.TFEncoder import *

'''Time library'''
import time

'''OS'''
import os

def pt(title=None, text=None):
    """
    Use the print function to print a title and an object coverted to string
    :param title:
    :param text:
    """
    if text is None:
        text = title
        title = Dictionary.string_separator
    else:
        title += ':'
    print(str(title) + " \n " + str(text))

def timed(method):
    """
    This method print a method name and the execution time.
    Normally will be used like decorator
    :param A method
    :return: The method called
    """

    def inner(*args, **kwargs):
        start = time.time()
        try:
            return method(*args, **kwargs)
        finally:
            methodStr = str(method)
            pt("Running time method:" + str(methodStr[9:-23]), time.time() - start)

    return inner


def clear_console():
    os.system('cls')

def number_neurons(total_input_size, input_sample_size, output_size, alpha=1):
    """
    :param total_input_size: x
    :param input_sample_size: x
    :param output_size: x
    :param alpha: x
    :return: number of neurons for layer
    """
    # TODO Finish docs
    return int(total_input_size / (alpha * (input_sample_size + output_size)))


def write_string_to_pathfile(string, filepath):
    """
    Write a string to a path file
    :param string: string to write
    :param path: path where write
    """
    try:
        create_directory_from_fullpath(filepath)
        file = open(filepath, 'w+')
        file.write(str(string))
    except:
        raise ValueError(Errors.write_string_to_file)

def write_json_to_pathfile(json, filepath):
    """
    Write a string to a path file
    :param string: string to write
    :param path: path where write
    """
    try:
        create_directory_from_fullpath(filepath)
        with open(filepath, 'w+') as file:
            # file = open(filepath, 'w+')
            file.write(str(json))
    except:
        raise ValueError(Errors.write_string_to_file)


def recurrent_method_pass_true_or_false(question, method):
    response = False
    pt(Dictionary.string_get_response)
    save = str(input(question + " ")).upper()
    if save == Dictionary.string_char_Y:
        response = True
    elif save != Dictionary.string_char_N:
        method()
    return response


def recurrent_ask_to_save_model():
    """
    Wait user to get response to save a model
    :return: 
    """
    method = recurrent_ask_to_save_model
    response = recurrent_method_pass_true_or_false(question=Dictionary.string_want_to_save,
                                                   method=method)
    return response

def recurrent_ask_to_continue_without_load_model():
    """
    Wait user to get response to save a model
    :return: 
    """
    method = recurrent_ask_to_continue_without_load_model
    response = recurrent_method_pass_true_or_false(question=Dictionary.string_want_to_continue_without_load,
                                                   method=method)
    return response

def file_exists_in_path_or_create_path(filepath):
    """
    Check if filepath exists and, if not, it creates the dir
    :param filepath: the path to check
    :return True if exists filepath, False otherwise
    """
    try:
        create_directory_from_fullpath(filepath)
        if os.path.exists(filepath):
            return True
        else:
            return False
    except:
        raise ValueError(Errors.check_dir_exists_and_create)

def factorial(num):
    """
    Factorial of a number. Recursive.
    :param num: Number
    :return: Factorial
    """
    if num > 1:
        num = num * factorial(num - 1)
    return num

def create_directory_from_fullpath(fullpath):
    """
    Create directory from a fullpath if it not exists.
    """
    # TODO (@gabvaztor) Check errors
    directory = os.path.dirname(fullpath)
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_file_from_fullpath(fullpath):
    """
    Create file from a fullpath if it not exists.
    """
    # TODO (@gabvaztor) Check errors
    if not os.path.exists(fullpath):  # To create file
        file = open(fullpath, 'w+')
        file.close()

def create_historic_folder(filepath, type_file, test_accuracy=""):
    """
    Used when filepath exists to create a folder with actual_time to historicize
    :param filepath: file to save  
    :param type_file: Type of file (Information or Configuration)
    """
    actual_time = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.gmtime(time.time())))
    directory = os.path.dirname(filepath)
    filename = actual_time + "_" + os.path.basename(filepath)
    low_stripe = ""
    if test_accuracy and test_accuracy is not "":
        low_stripe = "_"
    information_folder = "\\history_information\\" + type_file + "\\" + str(test_accuracy) + low_stripe + actual_time + "\\"
    folder = directory+information_folder
    create_directory_from_fullpath(folder)
    return folder+filename

def get_directory_from_filepath(filepath):
    return os.path.dirname(filepath)

def get_filename_from_filepath(filepath):
    return os.path.basename(filepath)






