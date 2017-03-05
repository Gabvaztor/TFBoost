"""
Author: @gabvaztor
StartDate: 04/03/2017

This file contains the next information:
    - Libraries to import with installation comment and reason.
    - Data Mining Algorithm.
    - Sets (train,validation and test) information.
    - ANN Arquitectures.
    - A lot of utils methods which you'll get useful advantage


The code's structure is:
    - Imports
    - Interface
    - Reading data algorithms
    - Data Mining
    - Training and test
    - Show final conclusions

Notes:
    * This file use TensorFlow code version 1.0.
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""
# --------------------------------------------------------------------------
'''LOCAL IMPORTS'''
import UsefulTools.UtilsFunctions as uf
import TensorFlowCode.TensorFlowGUI as eg
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
''' TensorFlow'''
import tensorflow as tf
print("TensorFlow: " + tf.__version__)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
''' Numpy is an extension to the Python programming language, adding support for large,
multi-dimensional arrays and matrices, along with a large library of high-level
mathematical functions to operate on these arrays.
http://www.numpy.org/
https://en.wikipedia.org/wiki/NumPy '''
import numpy as np
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
'''# Se necesita instalar la versión de 64bits de Scipy, al menos en Windows.
# Antes de instalar la librería de Scipy (o al menos para que no pueda dar
# un error asociado) hay que instalar 'NUMKY_MKL' desde la siguiente URL:
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
# We can find it in the url: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy'''
import scipy.io as sio
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
'''# Matlab URL: http://matplotlib.org/users/installing.html'''
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- USER INTERFACE ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

'''Creating user interface'''
properties = eg.EasyGui()
uf.pt("Typos GUI",properties.types)



"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- READING DATA ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""




"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---- Inicio de Sesión de TensorFlow ----
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""
init  = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# --------------------------------------------------------------------------
# UTILS FUNCTIONS
# --------------------------------------------------------------------------
def pt(title,text):
    print(str(title) + ": \n" + str(text))

