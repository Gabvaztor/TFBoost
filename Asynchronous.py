"""

"""
import multiprocessing

def execute_asynchronous_process(functions, arguments=None):
    Multiprocess(functions=functions, arguments=arguments)

class Multiprocess():
    """

    """
    def __init__(self, functions, arguments=None):
        datatype = self.__check_type__(functions)
        print("type", type(datatype))
        if datatype == type(list()):
            pass
        else:
            self._execute_process(function_def=functions, arguments=arguments)

    def __check_type__(self, object):
        return type(object)

    def _execute_process(self, function_def, arguments=None):
        if not arguments:
            arguments = ()
        if type(function_def) == type(str("")):
            name = function_def
        else:
            name = function_def.__name__
        process = multiprocessing.Process(name=name, target=function_def, args=arguments)
        process.start()
        process.join()
        while True:
            if process.is_alive()==False:
                process.terminate()
                break

