"""

"""
import threading
from UsefulTools.UtilsFunctions import pt

import traceback
import json

def object_to_json(object, attributes_to_delete=None):
    """
    Convert class to json with properties method.
    :param attributes_to_delete: String set with all attributes' names to delete from properties method
    :return: sort json from class properties.
    """
    try:
        object_dictionary = class_properties(object=object, attributes_to_delete=attributes_to_delete)
        json_string = json.dumps(object, default=lambda m: object_dictionary, sort_keys=True, indent=4)
    except Exception as e:
        pt(e)
        pt(traceback.print_exc())
        raise ValueError("PARAR")
    return json_string

def execute_asynchronous_thread(functions, arguments=None, kwargs=None):
    Thread(functions=functions, arguments=arguments, kwargs=kwargs)

class Thread():
    """

    """
    def __init__(self, functions, arguments=None, kwargs=None):
        datatype = self.__check_type__(functions)
        if datatype == type(list()):
            pass
        else:
            self._execute_process(function_def=functions, arguments=arguments, kwargs=kwargs)

    def __check_type__(self, object):
        return type(object)

    def _execute_process(self, function_def, arguments=None, kwargs=None):
        if not arguments:
            arguments = ()
        if type(function_def) == type(str("")):
            name = function_def
        else:
            name = function_def.__name__
        process = threading.Thread(name=name, target=function_def, args=arguments, kwargs=kwargs)
        process.start()

def class_properties(object, attributes_to_delete=None):
    """
    Return a string with actual object features without not necessaries
    :param attributes_to_delete: represent witch attributes set must be deleted.
    :return: A copy of class.__dic__ without deleted attributes
    """
    pt("object", object)
    dict_copy = object.__dict__.copy()  # Need to be a copy to not get original class' attributes.
    return dict_copy

