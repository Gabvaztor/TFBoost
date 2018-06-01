# -*- coding: utf-8 -*-
"""
Author: @gabvaztor
StartDate: 04/03/2017

Style: "Google Python Style Guide" 
https://google.github.io/styleguide/pyguide.html
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

# --------------------------------------------------------------------------
''' EasyGUI
Here you can download the library: https://pypi.python.org/pypi/easygui#downloads
It had been used the version: 0.98.1
'''
import easygui as eg
# --------------------------------------------------------------------------

class EasyGui(object):
    types = set()
    def __init__(self):
        eg.msgbox("Choose an option")
        self.types.add('Lineal')
        self.types.add('CNN')

if __name__ == '__main__':
    print ("Creating GUI")
    gui = EasyGui()