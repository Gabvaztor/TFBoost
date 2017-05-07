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

def pt(title=None,text=None):
    """
    Use the print function to print a title and an object coverted to string
    :param title:
    :param text:
    """
    if text is None:
        text = title
        title = Dictionary.string_separator
    else:
        title+=':'
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
            pt("Running time method:"+ str(methodStr[9:-23]), time.time() - start)
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
    return int(total_input_size/(alpha*(input_sample_size+output_size)))

def write_string_to_pathfile(string, filepath):
    """
    Write a string to a path file
    :param string: string to write
    :param path: path where write
    """
    try:
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file = open(filepath, 'w+')
        file.write(string)
    except:
        raise ValueError(Errors.write_string_to_file)
