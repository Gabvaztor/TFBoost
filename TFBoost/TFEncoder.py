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
