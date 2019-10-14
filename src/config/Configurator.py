"""
The function of this module is:
    - Getting all setting, create necessaries variables
    - Execute all necessaries tasks before main module (@decorators, etc)
    - Execute main module
"""
from .GlobalDecorators import DecoratorClass
from .GlobalSettings import GLOBAL_DECORATOR

class Configurator:
    def run(self, modules):
        """
        Args:
            modules: List array with modules to be executed this time
        """
        if GLOBAL_DECORATOR > 0:
            print(f"GLOBAL_DECORATOR is {GLOBAL_DECORATOR}")
            decorator_class = DecoratorClass()
            decorator_class.start_wrapper_decoration(modules=modules)
        else:
            print("GLOBAL_DECORATOR is not activated")
