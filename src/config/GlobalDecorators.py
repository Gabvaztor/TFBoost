"""
#3FXIAp5K
"""
import time
from config.GlobalSettings import DEBUG_MODE

import types
import functools

class DecoratorClass(object):

    __separator = "######### "
    __end_separator = "-----------"

    def __method_separator(self, method_name, step=1):
        """
        Args:
            step: If step == 1, then is START. If step == 2, then is END with method_name. Else, END without method_name
        Returns: new str line
        """
        start = "[START] "
        end = "[END] "
        method_separator = ""
        if step == 1:
            method_separator = start + self.__separator + "\"" + method_name + "()\" " + self.__separator
        elif step == 2:
            method_separator = end + self.__separator + "\"" + method_name + "()\" " + self.__separator
        else:
            method_separator = end + self.__end_separator * 4

        return method_separator

    def __decorate_all_in_module(self, modules, decorator):
        for module in modules:
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, types.FunctionType):
                    setattr(module, name, decorator(obj))

    def __global_decorator(self):
        def msg_decorator(function):
            @functools.wraps(function)
            def inner_dec(*args, **kwargs):
                method_str = str(function)[10:-23]
                method_separator_start = self.__method_separator(method_name=method_str, step=1)
                method_separator_end = self.__method_separator(method_name=method_str, step=3)
                print(method_separator_start)
                start = time.time()
                try:
                    return function(*args, DEBUG=DEBUG_MODE, **kwargs)
                except TypeError:  # Function has not kwargs
                    return function(*args, **kwargs)
                finally:
                    """
                    print("Running time method: \"" + str(method_str) + "\"",
                          str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start)))))
                    """
                    print("\"" + str(method_str) + "\"",
                          str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start)))))
                    print(method_separator_end)
            inner_dec.__name__ = function.__name__
            inner_dec.__doc__ = function.__doc__
            return inner_dec

        return msg_decorator

    def __my_decorator(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            print(f)
            return f(*args, **kwargs)
        return wrapper

    def start_wrapper_decoration(self, modules):
        self.__decorate_all_in_module(modules=modules, decorator=self.__global_decorator())