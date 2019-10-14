import config.GlobalDecorators as gd

def msg_printer(**kwargs):
    print("debug ->" , kwargs["DEBUG"])

def msg_printer2():
    print("debug ->" , )

import sys
modname = globals()['__name__']

modules = sys.modules
print(modules)
print(modules[modname])

gd.DecoratorClass().start_wrapper_decoration([modules[gd.__name__], modules[modname]])


msg_printer()  # prints 'Message'
msg_printer2()

print("sads", dir())

print(modname)
print(gd.__name__)