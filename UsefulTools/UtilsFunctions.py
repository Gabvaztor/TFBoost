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