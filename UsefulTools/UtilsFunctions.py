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
# --------------------------------------------------------------------------
'''
Time library
'''
import time
# --------------------------------------------------------------------------
from TFBoost.TFEncoder import *
import os
def pt(title,text):
    """
    Use the print function to print a title and and an object covert to string
    :param title:
    :param text:
    """

    print(str(title) + ": \n " + str(text))

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


"""
Signal competition
"""

def getSetsFromFullPathSignals(path):
    """
    If path contains 'train', y_label is two dir up. Else if path contains 'test', y_label is one dir up.
    :param path: the full path
    """
    pt('path',path)
    y_label_dir = ''
    if Dictionary.string_train in path:  # If 'train' in path
        y_label_dir = os.path.dirname(os.path.dirname(path))  # Directory of directory of file
    elif Dictionary.string_test in path:  # If 'test' in path
        y_label_dir = os.path.dirname(path)  # Directory of file
    else:
        raise ValueError(Errors.not_label_from_input)
    y_label_num = os.path.basename(y_label_dir)
    return y_label_num