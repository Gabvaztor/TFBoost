# coding=utf-8
"""
Author: @gabvaztor
StartDate: 10/05/2016

This is the code of final work degree by Gabriel Vázquez Torres
"""


import tkinter, tkinter.constants as Tkconstants, tkinter.filedialog as tkFileDialog
from tkinter import *
import os
import numpy as np
import datetime
# Importamos Spicy, librería científica para manejar archivos mat.
import scipy.io as sio
import TensorFlowCode as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw
import random
import math
from tkinter import ttk as ttk

from tkinter.ttk import *
#ttk import Frame, Style
from scipy.stats import *


def now():
    now1 = datetime.datetime.now()
    now = now1.strftime("%H:%M:%S")

    return now


def nowDay():
    now1 = datetime.datetime.now()
    now = now1.strftime("%Y:%m:%d-%H:%M:%S")

    return now


# Archivos para creación de conjuntos

#################################################################
trainSetGlobalOriginal = ['Relajado5', 'Concentrado_SimonGameNoSound', 'Concentrado_2048-bricks-Game', 'Relajado',
                          'Relajado4', 'ConcentradoOperaciones2']
validationSetGlobalOriginal = ['ConcentradoVideoY', 'Relajado3', 'Concentrado2OperacionesPart2',
                               'Concentrado_VideoYoutubeAtencionSelectiva', 'Relajado2']

trainSetGlobal = ['Relajado5', 'Concentrado_SimonGameNoSound', 'Concentrado_2048-bricks-Game', 'Relajado', 'Relajado4',
                  'ConcentradoOperaciones2']
validationSetGlobal = ['ConcentradoVideoY', 'Relajado3', 'Concentrado2OperacionesPart2',
                       'Concentrado_VideoYoutubeAtencionSelectiva', 'Relajado2']

testSetGlobal = ['Concentrado2OperacionesPart2']
testSetGlobal2 = ['Relajado2']
###################################################################




## PARA PRUEBAS

# trainSetGlobal = ['Concentrado_SimonGameNoSound','Relajado4']
# validationSetGlobal = ['Relajado3','Concentrado_VideoYoutubeAtencionSelectiva']

#####################




folder = 'saves/weights/'
folder2 = 'saves/graphs/'

mathematicsOperations = [
    ("Covarianza", 0),
    ("Varianza", 1),
    ("Media", 2),
    ("Suma", 3),
    ("Moda", 4),
    ("Mediana", 5),
    ("Percentil", 6),
    ("Desviacion", 7),
    ("CoeficienteDesviacion", 8),
    ("Minimo", 9),
    ("Maximo", 10),
    ("ErrorEstandarMedia", 11)]
# Variables de activación

windowPercentGlobal = 0.2
windowSizeGlobal = 320
rowsGlobal = 14

showGraph = 1
showCosts = 1
numOfTraining = 1
threshold = 99.

randomFiles = 0

numbers_neurons = 70

testSetToAccuracy = ''
pathChooseGlobal = ''

isLineal = 0
isMultilayer = 0
isCNN = 0
notOperate = 0
learningRateLineal = 0.000003
learningRateMultilayer = 0.0001
learningRateCNN = 0.0001
mathematicOperationRadiobutton = "Crudo"

numColumnsCNN = len(mathematicsOperations)
firstLabelNeurons = 16
secondLabelNeurons = 32
thirdLabelNeurons = 8
weightCNNShape = 2

trainingLinealFinal = 0
trainingMultilayerFinal = 0
trainingCNNFinal = 200

######MODELS######
trainPlaceholder, trainLabels, validationPlaceholder, validationLabels, testPlaceholder, testLabels, testPlaceholder1, testLabels1 = [], [], [], [], [], [], [], []
######END MODELS######
'''INTERFAZ  GRÁFICA'''


class TkFileDialog(tkinter.Frame):
    def __init__(self, root):
        #################################
        # Checkboxs and labels
        global isLinealCheck
        global isMultilayerCheck
        global isCNNCheck
        global showGraphCheck
        global showCostsCheck
        global label
        global windowSizeLabel
        global windowPercentLabel
        global numbers_neuronsLabel
        global numOfTrainingLabel
        global radioButton
        global mathematicOperationRadiobutton
        global learningRateLabelLineal
        global learningRateLabelMultilayer
        global randomFilesCheck
        global learningRateCNNLabel
        global firstLabelNeuronsLabel
        global secondLabelNeuronsLabel
        global thirdLabelNeuronsLabel
        global weightCNNShapeLabel
        global testSetToAccuracy
        #################################



        mathematicsOpertaions2 = [
            ("Crudo", 1),
            ("Estadisticamente", 2)]

        tkinter.Frame.__init__(self, root)

        root.wm_title("Configuración de TensorFlowCodeProyect")
        # Style().configure("TFrame", background="#333")
        # root.configure(bg="grey")
        # options for buttons
        button_opt = {'fill': Tkconstants.BOTH, 'padx': 30, 'pady': 5}
        # define options for opening or saving a file
        self.file_opt = options = {}
        options['defaultextension'] = '.mat'
        options['filetypes'] = [('mat files', '.mat')]
        # options['initialdir'] = 'C:\\'
        options['initialfile'] = '.mat'
        options['parent'] = root
        options['title'] = 'Select a .mat file'

        # defining options for opening a directory
        self.dir_opt = options = {}
        # options['initialdir'] = 'C:\\'
        options['mustexist'] = True
        options['parent'] = root
        options['title'] = 'This is a title'

        # define buttons
        fmI = Frame(self)
        fmI.pack(side=TOP, fill=BOTH)
        tkinter.Button(fmI, text='Información', command=self.mhello1, height=5, width=10).pack(side=LEFT, padx=20,
                                                                                               pady=5)
        tkinter.Button(fmI, text='Funcionamiento', command=self.mhello2, height=5, width=10).pack(side=TOP, pady=5)

        fmSF = Frame(self)
        fmSF.pack(side=TOP, fill=BOTH, padx=10, pady=10)
        tkinter.Button(fmSF, text='Seleccionar archivo', command=self.askopenfilename, height=1, width=30).pack(
            **button_opt)
        label = Text(fmSF, height=1, width=50)
        label.pack()

        divider1 = tkinter.Frame(self, bg="black")
        divider1.pack(side=TOP, fill=BOTH, padx=10, pady=5)
        # CHECKBOXS 1
        fmKM = Frame(self)
        fmKM.pack(side=TOP, fill=BOTH, padx=10, pady=10)
        isLinealCheck = tkinter.IntVar()
        isLinealCheckbox = Checkbutton(fmKM, text="Lineal", command=self.checkBoxLineal, variable=isLinealCheck)
        isLinealCheckbox.pack(padx=20, side=LEFT)

        isMultilayerCheck = tkinter.IntVar()
        isMultilayerCheckbox = Checkbutton(fmKM, text="Multicapa", command=self.checkBoxMultilayer,
                                           variable=isMultilayerCheck)
        isMultilayerCheckbox.pack(padx=20, side=LEFT)

        isCNNCheck = tkinter.IntVar()
        isCNNCheckbox = Checkbutton(fmKM, text="Convolucional", command=self.checkBoxCNN, variable=isCNNCheck)
        isCNNCheckbox.pack(padx=20, side=LEFT)

        divider2 = tkinter.Frame(self, bg="black")
        divider2.pack(side=TOP, fill=BOTH, padx=10, pady=5)
        # radioButtons
        fm = Frame(self)
        fm.pack(side=TOP, fill=BOTH, padx=10, pady=5)
        fmRadioButtons = Frame(self)
        fmRadioButtons.pack(side=TOP, fill=BOTH, padx=20)
        radioButton = tkinter.StringVar()
        radioButton.set("Crudo")
        mathematicOperationRadiobutton = "Crudo"
        for txt, val in mathematicsOpertaions2:
            tkinter.Radiobutton(fmRadioButtons,
                                text=txt,
                                padx=20,
                                variable=radioButton,
                                command=self.mathematicOperationDef,
                                value=txt).pack(side=LEFT, anchor=W, fill=X)

        divider3 = tkinter.Frame(self, bg="black")
        divider3.pack(side=TOP, fill=BOTH, padx=10, pady=5)
        # Button(fm, text='Top').pack(side=TOP, anchor=W, fill=X, expand=YES)



        # CHECKBOXS2
        fmOO = Frame(self)
        fmOO.pack(side=TOP, fill=BOTH, padx=30, pady=20)
        showGraphCheck = tkinter.IntVar()
        showGraphCheck.set(1)
        showGraphCheckbox = Checkbutton(fmOO, text="Ver gráficos", command=self.checkBoxShowGraph,
                                        variable=showGraphCheck)
        showGraphCheckbox.pack(padx=10, side=LEFT)

        showCostsCheck = tkinter.IntVar()
        showCostsCheck.set(1)
        showCostsCheckbox = Checkbutton(fmOO, text="Ver gráficos de costes", command=self.checkBoxShowCosts,
                                        variable=showCostsCheck)
        showCostsCheckbox.pack(padx=10, side=LEFT)

        divider4 = tkinter.Frame(self, bg="black")
        divider4.pack(side=TOP, fill=BOTH, padx=10, pady=5)
        # Labels

        fmWL = Frame(self)
        fmWL.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        windowSizeText = Label(fmWL, text="Tamaño de ventana temporal:")
        windowSizeText.pack(side=LEFT, anchor=W, fill=X)
        windowSizeLabel = Text(fmWL, height=1, width=10)
        windowSizeLabel.insert(END, "320")
        windowSizeLabel.pack(side=TOP, anchor=W)

        fmWP = Frame(self)
        fmWP.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        windowsPercentText = Label(fmWP, text="Porcentaje de ventana temporal:")
        windowsPercentText.pack(side=LEFT, anchor=W)
        windowPercentLabel = Text(fmWP, height=1, width=10)
        windowPercentLabel.insert(END, "0.2")
        windowPercentLabel.pack(anchor=W)

        fmNN = Frame(self)
        fmNN.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        numberNeuronsMText = Label(fmNN, text="Número de neuronas multicapa:")
        numberNeuronsMText.pack(side=LEFT, anchor=W)
        numbers_neuronsLabel = Text(fmNN, height=1, width=10)
        numbers_neuronsLabel.insert(END, "70")
        numbers_neuronsLabel.pack(anchor=W)

        fmNT = Frame(self)
        fmNT.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        numOfTrainingMText = Label(fmNT, text="Número de entrenamientos:")
        numOfTrainingMText.pack(side=LEFT, anchor=W)
        numOfTrainingLabel = Text(fmNT, height=1, width=10)
        numOfTrainingLabel.insert(END, "150")
        numOfTrainingLabel.pack(anchor=W)

        fmLRL = Frame(self)
        fmLRL.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        learningRateLabelText = Label(fmLRL, text="Coeficiente de aprendizaje lineal:")
        learningRateLabelText.pack(side=LEFT, anchor=W)
        learningRateLabelLineal = Text(fmLRL, height=1, width=10)
        learningRateLabelLineal.insert(END, "0.000003")
        learningRateLabelLineal.pack(anchor=W)

        fmLRM = Frame(self)
        fmLRM.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        learningRateLabelMText = Label(fmLRM, text="Coeficiente de aprendizaje multicapa:")
        learningRateLabelMText.pack(side=LEFT, anchor=W)
        learningRateLabelMultilayer = Text(fmLRM, height=1, width=10)
        learningRateLabelMultilayer.insert(END, "1e-4")
        learningRateLabelMultilayer.pack(anchor=W)

        fmLRC = Frame(self)
        fmLRC.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        learningRateLabelCText = Label(fmLRC, text="Coeficiente de aprendizaje convolucional:")
        learningRateLabelCText.pack(side=LEFT, anchor=W)
        learningRateCNNLabel = Text(fmLRC, height=1, width=10)
        learningRateCNNLabel.insert(END, "1e-4")
        learningRateCNNLabel.pack(anchor=W)

        fmFL = Frame(self)
        fmFL.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        firstLabelNeuronsText = Label(fmFL, text="Neuronas en la primera capa CNN:")
        firstLabelNeuronsText.pack(side=LEFT, anchor=W)
        firstLabelNeuronsLabel = Text(fmFL, height=1, width=10)
        firstLabelNeuronsLabel.insert(END, "16")
        firstLabelNeuronsLabel.pack(anchor=W)

        fmSL = Frame(self)
        fmSL.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        secondLabelNeuronsText = Label(fmSL, text="Neuronas en la segunda capa CNN:")
        secondLabelNeuronsText.pack(side=LEFT, anchor=W)
        secondLabelNeuronsLabel = Text(fmSL, height=1, width=10)
        secondLabelNeuronsLabel.insert(END, "32")
        secondLabelNeuronsLabel.pack(anchor=W)

        fmTL = Frame(self)
        fmTL.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        thirdLabelNeuronsText = Label(fmTL, text="Neuronas en la tercera capa CNN:")
        thirdLabelNeuronsText.pack(side=LEFT, anchor=W)
        thirdLabelNeuronsLabel = Text(fmTL, height=1, width=10)
        thirdLabelNeuronsLabel.insert(END, "8")
        thirdLabelNeuronsLabel.pack(anchor=W)

        fmWC = Frame(self)
        fmWC.pack(side=TOP, fill=BOTH, padx=20, pady=10)
        weightCNNShapeText = Label(fmWC, text="Dimesión de los pesos CNN:")
        weightCNNShapeText.pack(side=LEFT, anchor=W)
        weightCNNShapeLabel = Text(fmWC, height=1, width=10)
        weightCNNShapeLabel.insert(END, "2")
        weightCNNShapeLabel.pack(anchor=W)

        randomFilesCheck = tkinter.IntVar()
        randomFilesCheckbox = Checkbutton(self, text="Conjuntos aleatorios", command=self.checkBoxRandomFiles,
                                          variable=randomFilesCheck)
        randomFilesCheckbox.pack(padx=10, side=LEFT)

        tkinter.Button(self, text='Ejecutar', command=self.connectTensorFlow).pack(**button_opt)
        tkinter.Button(self, text='Cancelar', command=self.exit).pack(**button_opt)

    def mathematicOperationDef(self):
        global mathematicOperationRadiobutton
        global radioButton
        print('mathematicOperationRadiobutton elegida:')
        mathematicOperationRadiobutton = radioButton.get().encode('utf-8').strip()
        print(mathematicOperationRadiobutton)

    def checkBoxRandomFiles(self):
        global randomFiles
        global trainSetGlobal
        global validationSetGlobal
        randomFiles = randomFilesCheck.get()
        print('randomFiles')
        print(randomFiles)
        # REORDENAMOS LA LISTA PARA QUE NO SE COJAN SIEMPRE EN EL MISMO ORDEN
        if (randomFiles == 1):
            random.shuffle(trainSetGlobal)
            random.shuffle(validationSetGlobal)
            print('---------Conjunto de entrenamiento: ---------\n' + str(trainSetGlobal))
            print('---------Conjunto de validación: ---------\n' + str(validationSetGlobal))
        else:
            trainSetGlobal = trainSetGlobalOriginal
            validationSetGlobal = validationSetGlobalOriginal
            print('---------Conjunto de entrenamiento original: ---------\n' + str(trainSetGlobalOriginal))
            print('---------Conjunto de validación original: ---------\n' + str(validationSetGlobalOriginal))

    def checkBoxShowGraph(self):
        global showGraph
        showGraph = showGraphCheck.get()
        print('showGraph')
        print(showGraph)

    def checkBoxShowCosts(self):
        global showCosts
        showCosts = showCostsCheck.get()
        print('showCosts')
        print(showCosts)

    def checkBoxLineal(self):
        global isLineal
        isLineal = isLinealCheck.get()
        print('isLineal')
        print(isLineal)

    def checkBoxMultilayer(self):
        global isMultilayer
        isMultilayer = isMultilayerCheck.get()
        print('isMultilayer')
        print(isMultilayer)

    def checkBoxCNN(self):
        global isCNN
        isCNN = isCNNCheck.get()
        print('isCNN')
        print(isCNN)

    def askopenfilename(self):
        global pathChooseGlobal
        global label
        """Returns an opened file in read mode.
        This time the dialog just returns a filename and the file is opened by your own code.
        """
        # get filename
        filename = tkFileDialog.askopenfilename(**self.file_opt)
        pathChooseGlobal = filename
        # print pathChooseGlobal
        # open file on your own
        if filename:
            # print filename
            label.insert(END, str(filename))
            print('label')
            print(str(label.get("1.0", END)))
            pathChooseGlobal = label.get("1.0", END)
            return open(filename, 'r')

    def connectTensorFlow(self):
        #############
        # Global variables#
        global windowSizeGlobal
        global windowPercentGlobal
        global numbers_neurons
        global numOfTraining
        global learningRateLineal
        global learningRateMultilayer
        global windowSizeLabel
        global windowPercentLabel
        global numbers_neuronsLabel
        global numOfTrainingLabel
        global learningRateLabelLineal
        global learningRateLabelMultilayer

        global learningRateCNNLabel
        global firstLabelNeuronsLabel
        global secondLabelNeuronsLabel
        global thirdLabelNeuronsLabel
        global weightCNNShapeLabel

        global learningRateCNN
        global firstLabelNeurons
        global secondLabelNeurons
        global thirdLabelNeurons
        global weightCNNShape
        #############

        if pathChooseGlobal == '':

            windowPercentGlobal = float(windowPercentLabel.get("1.0", END))
            windowSizeGlobal = float(windowSizeLabel.get("1.0", END))
            numbers_neurons = float(numbers_neuronsLabel.get("1.0", END))
            numOfTraining = float(numOfTrainingLabel.get("1.0", END))
            learningRateLineal = float(learningRateLabelLineal.get("1.0", END))
            learningRateMultilayer = float(learningRateLabelMultilayer.get("1.0", END))
            learningRateCNN = float(learningRateCNNLabel.get("1.0", END))
            firstLabelNeurons = float(firstLabelNeuronsLabel.get("1.0", END))
            secondLabelNeurons = float(secondLabelNeuronsLabel.get("1.0", END))
            thirdLabelNeurons = float(thirdLabelNeuronsLabel.get("1.0", END))
            weightCNNShape = float(weightCNNShapeLabel.get("1.0", END))

            root.destroy()
        else:
            windowPercentGlobal = float(windowPercentLabel.get("1.0", END))
            windowSizeGlobal = float(windowSizeLabel.get("1.0", END))
            numbers_neurons = float(numbers_neuronsLabel.get("1.0", END))
            numOfTraining = float(numOfTrainingLabel.get("1.0", END))
            learningRateLineal = float(learningRateLabelLineal.get("1.0", END))
            learningRateMultilayer = float(learningRateLabelMultilayer.get("1.0", END))
            learningRateCNN = float(learningRateCNNLabel.get("1.0", END))
            firstLabelNeurons = float(firstLabelNeuronsLabel.get("1.0", END))
            secondLabelNeurons = float(secondLabelNeuronsLabel.get("1.0", END))
            thirdLabelNeurons = float(thirdLabelNeuronsLabel.get("1.0", END))
            weightCNNShape = float(weightCNNShapeLabel.get("1.0", END))
            root.destroy()
            # exit(self)

    def exit(self):
        global isLinealCheck
        global isMultilayerCheck
        global isCNNCheck
        global isLineal
        global isMultilayer
        global isCNN

        isLinealCheck.set(0)
        isMultilayerCheck.set(0)
        isCNNCheck.set(0)
        isLineal = 0
        isMultilayer = 0
        isCNN = 0

        root.destroy()

    def checkIfIsAValidFile(path):
        if path == '':
            print('Debe elegir un archivo válido')
            print('Se procederá a elegir el test de ejemplo')
            return True

    def mhello1(self):
        toplevel = Toplevel(self, background="#AAAAAA")
        toplevel.title('Descripción')
        toplevel.focus_set()
        toplevel.geometry("850x400")
        # window = Frame(toplevel,background="#ffffff")
        # window.pack(side=TOP, fill=BOTH)
        # scrollbar = tk.Scrollbar(toplevel, orient="vertical", command=canvas.yview)

        # window.config(yscrollcommand=scrollbar.set)
        Text = Label(toplevel, background="#EEEEEE", text="""DESCRIPCIÓN DEL PROGRAMA:

El objetivo principal de este programa es el de poder modificar libremente una serie de parámetros de tensorflow a través de
 una interfaz gráfica para obtener una probabilidad mediante la clasificación de modelos matemáticos.Así pues este programa
 está orientado al estudio y clasificación de electroencefalogramas obtenidos a partir del lector de ondas cerebrales
 "Emotiv". Actualmente podemos obtener una probabilidad de que el conjunto dado tenga un estado general de "concentrado" o
 "relajado". Se ha creado para comparar diferentes modelos matemáticos como son el modelo lineal, el modelo multicapa y el
 modelo convolucional (deep learning) para así poder tener una comparativa de todos ellos. Es importante remarcar que se
 puede trabajar tanto con datos en crudo como con datos estadísticos.

CUESTIONES TÉCNICAS:

Los ECG dados están pasados por varios filtros para convertir los datos obtenidos a partir del Emotiv en un archivo con
 formato ".mat" y con 14 filas correspondientes a cada uno de los canales del Emotiv además de filtros ICA y eliminación de
 los primeros y últimos valores por posible ruido. Trabaja a 128Hz siendo estos los definidos en el casco Emotiv. Es
 necesario que el archivo cumpla con todas estas características y que los datos sean superiores a los obtenidos durante 5
 segundos a 128 Hz para el correcto funcionamiento.

Si la precisión para el conjunto de entrenamiento es del 100% se para el aprendizaje o si la precisión para el conjunto de
 validación supera un cierto umbral se pasa a clasificar el conjunto de test.

Todos los resultados apecerán en la consola.
""")

        # scrollbar.pack(side=RIGHT, fill=Y)
        # Text.pack(padx=10,pady=10)
        # scrollbar.config(command=window.yview)
        Text.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    def mhello2(self):
        toplevel2 = Toplevel(self, background="#AAAAAA")
        toplevel2.title('Explicación de características')
        toplevel2.focus_set()
        toplevel2.geometry("1150x900")

        Text2 = Label(toplevel2, background="#EEEEEE", text=""" EXPLICACIÓN DE CARACTERÍSTICAS:

Esta es la descripción del funcionamineto del la configuración de TensorFlowCodeProyect. A continuación se detallarán uno a uno el funcionamiento de cada widget del programa:

- SELECCIONAR ARCHIVO: Mediante este botón podremos seleccionar un archivo con formato ".mat" correspondiente al conjunto a clasificar. Debe cumplir con todas las
 condiciones previamente comentadas en el apartado "Cuestiones técnicas".

- LINEAL: Este checkbox se corresponde con el modelo lineal. Si lo seleccionamos informaremos al programa para que analice el conjunto dado a través de este modelo.
 Aprende a través del descenso por el gradiente. Se pueden añadir todos los modelos a la vez que se quiere para su comparación.

- MULTICAPA: Este se corresponde al modelo multicapa. Es algo más complejo que el modelo lineal ya que este trabaja con capas intermedias. Además, podemos modificar
 la característica "Número de neuronas multicapa" añadiendo a nuestro antojo la cantidad de neuronas deseadas. Un número alto de neuronas se correspondería con
 mayor grado de aprendizaje pero una mayor cantidad de cálculos para el procesador. Aprende a través del optimizador Adam (variante del descenso por el gradiente).

- CONVOLUCIONAL: Este el el modelo convolucional (deep learning). Es el más complejo de los modelos ya que se producen numerosos cálculos multicapa de tal forma que
 no podemos saber realmente qué ocurre en él. Para este modelo podemos modificar tanto la dimensión de sus pesos como el número de neuronas para cada una de sus
tres capas. Es importante comentar que si seleccionamos la forma de los datos "Estadísticos" es muy recomendable usar una dimensión de pesos "2" para su correcto
funcionamiento. Aprende a través del optimizador Adam (variante del descenso por el gradiente).

- CRUDO: Se corresponde con la forma con la que queremos que se trabajen los datos. Si seleccionamos en crudo estaremos trabajado con los datos sin ningún tipo de
 modificaciones.

- ESTADISTICAMENTE: Se corresponde con la forma con la que queremos que se trabajen los datos. Si seleccionamos este caso trabajaremos con datos estadísticos a
partir de varios procesos realizados en cada una de las filas del archivo ".mat". Covarianza, varianza, suma y media son alguno de ellos. Es muy importante tener
 en cuenta que si seleccionamos el tratamiento de datos estadisticamente la dimensión de los pesos cnn debe ser 2.

- VER GRÁFICOS: Si está activado veremos al final la gráfica de aciertos correspondiente a cada uno de los modelos seleccionados para los conjuntos de entrenamiento
 y validación (el conjuto dado desde la interfaz es el conjunto de testeo).

- VER GRÁFICOS DE COSTES: Si está activado veremos cómo ha ido aumentado o descendiendo el coste de los conjuntos de  entrenamiento y validación para cada uno de
los modelos seleccionados.

- TAMAÑO DE VENTANA TEMPORAL: Podemos informale a TensorFlowCodeProyect el tamaño de las columnas con las que queremos que trabaje. Le  daremos a TensorFlowCodeProyect unos datos
 correspondientes a dividir el archivo en X partes iguales (X es el número que sale si dividimos el tamaño total del conjunto dado entre el tamaño de la ventana
 temporal). Así, para un número menor, estará aprendiendo de trozos más pequeños y viceversa. Es importante comentar que 640 se corresponden a 5 segundos de la
 muestra.

- PORCENTAJE DE LA VENTANA TEMPORAL: Este número se corresponde con el porcentaje que cogemos de cada ventana temporal para  crear una nueva. Por ejemplo, si tenemos
 una ventana temporal de 100 columnas (0,100), la siguiente ventana temporal para un valor de 0.2 de porcentaje (20%) se corresponderá a una ventana temporal con los
 valores (80,180).

- NÚMERO DE NEURONAS MULTICAPA: Se corresponde con el número de neuronas que queremos que tenga nuestro modelo multicapa.

- NÚMERO DE ENTRENAMIENTOS: Se corresponde a cuántos entrenamientos queremos que procese TensorFlowCodeProyect. Cuantos más entrenamientos más procesamientos requiere pero más
aprendizaje tendrá.

- COEFICIENTE DE APRENDIZAJE LINEAL: Es el coeficiente de aprendizaje para el descenso del gradiente en el modelo lineal.

- COEFICIENTE DE APRENDIZAJE MULTICAPA: Es el coeficiente de aprendizaje para el optimizador Adam en el modelo multicapa.

- COEFICIENTE DE APRENDIZAJE CONVOLUCIONAL: Es el coeficiente de aprendizaje para el optimizador Adam en el modelo convolucional.

- NÚMERO DE NEURONAS PRIMERA CAPA CNN: Número de neuronas para la primera capa del modelo convolucional.

- NÚMERO DE NEURONAS SEGUNDA CAPA CNN: Número de neuronas para la segunda capa del modelo convolucional.

- NÚMERO DE NEURONAS TERCERA CAPA CNN: Número de neuronas para la tercera capa del modelo convolucional.

- DIMENSIÓN DE LOS PESOS CNN: Se corresponde con la dimensión de los pesos. Si el número es 2, la dimensión es 2x2. Es muy importante tener en cuenta que si
seleccionamos el tratamiento de datos estadisticamente la dimensión de los pesos debe ser 2.

- CONJUNTOS ALEATORIOS: Si seleccionamos "Conjuntos aleatorios" veremos por pantalla el orden de los conjuntos de entrenamiento y validación. Si está desactivado
el orden será uno preestablecido y comprobado para que sea el más óptimo probado en los ensayos y pruebas.  """)
        # scrollbar.pack(side=RIGHT, fill=Y)
        # Text.pack(padx=10,pady=10)
        # scrollbar.config(command=window.yview)
        Text2.pack(side="top", fill="both", expand=True, padx=10, pady=10)


if __name__ == '__main__':
    root = tkinter.Tk()
    TkFileDialog(root).pack()
    root.geometry("600x900")
    root.mainloop()

### OPERACIONES DESPUÉS DE LA IU####
print(pathChooseGlobal)
print(mathematicOperationRadiobutton)

# Se transforman a int todas las
numbers_neurons = int(numbers_neurons)
numOfTraining = int(numOfTraining)
windowSizeGlobal = int(windowSizeGlobal)
firstLabelNeurons = int(firstLabelNeurons)
secondLabelNeurons = int(secondLabelNeurons)
thirdLabelNeurons = int(thirdLabelNeurons)
weightCNNShape = int(weightCNNShape)

# Path
testSetToAccuracy = pathChooseGlobal

print('Método estadístico elegido: ' + str(mathematicOperationRadiobutton))
print('Tamaño ventana temporal : ' + str(windowSizeGlobal))
print('Porcentaje: ' + str(windowPercentGlobal * 100))
print('Neuronas multicapa: ' + str(numbers_neurons))
print('Número de entrenamientos: ' + str(numOfTraining))
print('Threshold: ' + str(threshold))
print('Ratio de aprendizaje Lineal: ' + str(learningRateLineal))
print('Ratio de aprendizaje Multicapa: ' + str(learningRateMultilayer))
print('Ratio de aprendizaje CNN: ' + str(learningRateCNN))
print('Neuronas primera capa CNN: ' + str(firstLabelNeurons))
print('Neuronas segunda capa CNN: ' + str(secondLabelNeurons))
print('Neuronas tercera capa CNN: ' + str(thirdLabelNeurons))
print('Dimension de pesos CNN: ' + str(weightCNNShape))

if isLineal == 0 and isMultilayer == 0 and isCNN == 0:
    notOperate = 1

if (mathematicOperationRadiobutton == "Crudo"):
    crudoReshape = int(rowsGlobal * windowSizeGlobal)
    rowsColumnsReal = crudoReshape
    numColumnsCNN = (windowSizeGlobal / 4)
    numColumnsCNNPool = numColumnsCNN
else:
    numColumnsCNN = len(mathematicsOperations)
    stadisticReshape = int(rowsGlobal * numColumnsCNN)
    rowsColumnsReal = stadisticReshape
    numColumnsCNNPool = numColumnsCNN / 4


### END OPERACIONES DESPUÉS DE LA IU ####



# Aprendizaje: Enseñar. (Se dan los labels)

# Conjunto de test: Comprobar si se ha aprendido correctamente. (No se dan las etiquetas pero nosotros lo sabemos)

# Conjunto de validación (Evita el problema de sobreaprendizaje) : Buscar ejemplo. Cross Validation.

######################MÉTODOS DE AYUDA############################
# ----------------getLabelType---------------------
# Devuelve 1 si la cadena pasada contiene la palabra "Concentrado" y 0 si no.
def getLabelType(mathFileName):
    labelType = 1.;
    if "Concentrado" in mathFileName:
        labelType = 2.
    return labelType


# ----------------END getLabelType------------------
# ------------------------------------
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ----------------------------------------------------------------------------
def calculateTotalIterations(matrix, windowSize, windowPercent, listIndex):
    a = math.ceil((matrix.shape[1] * 1.0 / (windowSize - (windowSize * windowPercent))))
    s = ((np.array(listIndex).shape[0] - 1.0) * 1.0)
    iterationsTraining = round(a - s, 0)
    return int(iterationsTraining)


def column(matrix, i):
    return np.asarray([[row[i] for row in matrix]])


##################################################################





# ----------------------------------------------------------------------------
'''
Modelado de entrenamiento
'''
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


# ------------------GLOBAL VARIABLES--------------------
# listIndexX_pieceStarts contiene una lista de números enteros de cada x_piece que representan los valores de su posición inicial en la matriz x_all
listIndexX_pieceStarts = []
listIndexX_pieceStartsValidation = []
listIndexX_pieceStartsTest = []
numberOfTrains = 0
x_all = []
validationSet = []
testSet = []
# ------------------Pointers----------------------------
# Deben inicializarse a 0
lastTemporalWindowValue = 0
lastTemporalWindowValueValidation = 0
lastTemporalWindowValueTest = 0
x_pieceNumber = 0
x_pieceNumberValidation = 0
x_pieceNumberTest = 0
x_timesTrain = 0
x_timesValidation = 0
x_timesTest = 0
# ------------------END Pointers------------------------




# ------------------END GLOBAL VARIABLES----------------


# ------------------Get x_all from Maths----------------

'''
Conjunto de datos:

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
De entrenamiento: 6 archivos:

Concentrado: 'Concentrado_SimonGameNoSound','ConcentradoOperaciones2', 'Concentrado_2048-bricks-Game'
Relajado: 'Relajado','Relajado5','Relajado4'
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
De validación: 4 archivos:

Concetrado: 'ConcentradoVideoY','Concentrado2OperacionesPart2','Concentrado_VideoYoutubeAtencionSelectiva'
Relajado: 'Relajado2'
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
De test: 2 archivos:

Concentrado : 'Concentrado2OperacionesPart1'
Relajado: 'Relajado3'
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''


# Duvuelve x_all: matriz con la concatenación normalizada de todos los math. Tiene un shape (16,X)
def getx_all(option, testSetToAccuracy):
    global trainSetGlobal
    global validationSetGlobal
    global testSetGlobal
    pathUbuntu = '/root/csvMat/'
    path = 'C:\\Python352\\compila\\csvMat\\'
    extension = '.mat'

    if (option == 1):
        allMathFiles = trainSetGlobal
    elif (option == 2):
        allMathFiles = validationSetGlobal
    elif (option == 3):
        if testSetToAccuracy is not '':
            allMathFiles = testSetToAccuracy.split()
            path = ''
            extension = ''
            print('pathTest')
            print(allMathFiles)
        else:
            allMathFiles = testSetGlobal
            print('pathTest')
            print(allMathFiles)
    elif (option == 4):
        allMathFiles = testSetGlobal2
    # numberOfFile lleva el número de archivos .mat diferentes
    numberOfFile = 0
    x_all = np.empty([])
    for mathFileName in allMathFiles:
        completePath = path + mathFileName + extension
        print (completePath)
        asdasj
        # Cargamos el contenido del mat
        mat_contents = sio.loadmat(completePath)['matrix2']
        # Número de columnas de la matriz
        num_columns = mat_contents.shape[1]
        # Tipo de label: concentrado = 2, relajado = 1
        labelType = getLabelType(mathFileName)
        # x_piece representa la matriz normalizada de cada uno de los archivos .mat
        x_piece = createX_piece(mat_contents, numberOfFile, num_columns, labelType)
        # Vamos uniendo cada x_piece para formar x_all
        if numberOfFile == 0:
            x_all = x_piece
            x_all = np.c_[x_all, x_piece]
            if option == 1:
                listIndexX_pieceStarts.append(0)
            elif option == 2:
                listIndexX_pieceStartsValidation.append(0)
            elif option == 3:
                listIndexX_pieceStartsTest.append(0)
        else:
            newIndex = x_all.shape[1] + 1
            if option == 1:
                listIndexX_pieceStarts.append(newIndex)
            elif option == 2:
                listIndexX_pieceStartsValidation.append(newIndex)
            elif option == 3:
                listIndexX_pieceStartsTest.append(newIndex)
            x_all = np.c_[x_all, x_piece]
        # Aumentamos la variabre numberOfFile
        numberOfFile += 1
    return x_all


# ------------------End get x_all from Math-------------

# ----------------createX_piece---------------------
# Con este método normalizaremos la matriz de entrada en un x_piece el cual contiene dos vectores más referentes a: 1- vector de 0 o 1 según sea concentrado o relajado y 2- Vector incremental según el número de archivo .mat que sea.
def createX_piece(mat_contents, numberOfFile, num_columns, labelType):
    # Adding parameters to Matrix mat_contents
    b = np.full((1, num_columns), labelType, dtype=np.float)
    c = np.full((1, num_columns), numberOfFile, dtype=np.float)
    # Creating x_piece from mat_contents
    x_piece = np.concatenate((mat_contents, b))
    x_piece = np.concatenate((x_piece, c))
    return x_piece


# -----------End createX_piece------------------------




# ----------------getX_AndY_-----------------

# Obtiene x_: matrix con 14 filas correspondiente a una ventana temporal sólo con datos específicos del EEG ( se debe obtener a partir de una matriz de columnas = 5s de datos y normalizada a 16 filas) y obtiene y_: matriz one-hot real siendo [1,0] si x_ es de label relajada o [0,1] si es de label concentrada.
def getX_AndY_(x_all, startInOtherMatrix, windowPercent, windowSize, iteration, option, mathematicOperationChoosed):
    temporalWindowValue = 0
    if option == 1:
        temporalWindowValue = lastTemporalWindowValue
    elif option == 2:
        temporalWindowValue = lastTemporalWindowValueValidation
    elif option == 3:
        temporalWindowValue = lastTemporalWindowValueTest

    x_1 = getX_1(x_all, startInOtherMatrix, windowPercent, windowSize, iteration, option, temporalWindowValue)
    y_1 = getY_(x_1)
    y_ = np.array(y_1)
    x_ = x_1[0:14, :]

    y_r = y_.reshape(1, 2)
    x_r = calculateMathOperation(x_, mathematicOperationChoosed)
    # x_r = x_r1.reshape(1,rowsColumnsReal)
    return x_r, y_r


# -------------getY_------------------
def getY_(x_1):
    y_ = []
    if x_1[14, 0] == 1.0:
        y_ = [2.0, 1.0]
    else:
        y_ = [1.0, 2.0]
    return y_


# -------------END getY_--------------

### CALCULATE MATHEMATICAL OPER
def calculateMathOperation(x_, mathematicOperationChoosed):
    matrix = np.empty([])
    mathematicOperation = 0
    count = 2
    columnXi = []

    if (mathematicOperationChoosed == "Crudo"):
        matrix = x_.reshape(1, crudoReshape)
    else:
        for stadistic, pos in mathematicsOperations:
            columnToAdd = np.empty([])
            for r in range(rowsGlobal):
                actualRow = x_[r, :]
                mathematicOperation = 0
                if (stadistic == "Covarianza"):
                    mathematicOperation = float(np.cov(actualRow))
                elif (stadistic == "Varianza"):
                    mathematicOperation = float(np.var(actualRow))
                elif (stadistic == "Media"):
                    mathematicOperation = float(np.average(actualRow))
                elif (stadistic == "Gini"):
                    mathematicOperation = float(gini(actualRow))
                elif (stadistic == "Suma"):
                    mathematicOperation = float(np.sum(actualRow))
                elif (stadistic == "Moda"):
                    mathematicOperation = float(mode(actualRow)[0][0])
                elif (stadistic == "Mediana"):
                    mathematicOperation = float(np.median(actualRow))
                elif (stadistic == "Percentil"):
                    mathematicOperation = float(np.percentile(actualRow, 50))
                elif (stadistic == "Desviacion"):
                    mathematicOperation = float(np.std(actualRow))
                elif (stadistic == "CoeficienteDesviacion"):
                    mathematicOperation = float(variation(actualRow))
                elif (stadistic == "Minimo"):
                    mathematicOperation = float(np.amin(actualRow))
                elif (stadistic == "Maximo"):
                    mathematicOperation = float(np.amax(actualRow))
                elif (stadistic == "ErrorEstandarMedia"):
                    mathematicOperation = float(tsem(actualRow))

                mathematicOperation = np.array([[mathematicOperation]])
                if (r == 0):
                    columnToAdd = mathematicOperation
                else:
                    columnToAdd = np.append(columnToAdd, mathematicOperation, axis=0)
            if (pos == 0):
                matrix = columnToAdd
            else:
                matrix = np.append(matrix, columnToAdd, axis=1)
        matrix = matrix.reshape(1, stadisticReshape)
    return np.array(matrix)


def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list[0]:
        height += value
        area += height - value / 2.
    fair_area = (height * len(list_of_values) / 2.)
    result = 0.
    if (fair_area is not 0.):
        result = float((fair_area - area) / fair_area)
    else:
        result = float((fair_area - area))
    return result


### END MATHEMATICAL OPERATION

# -------------getX_------------------
# x_all :matriz con todos los x_piece
# iteration : representa el número de iteración actual.
# windowPercent: representa el valor del porcentaje deseado para obtener la siguiente ventana temporal (x_1)

# Para este método se presupone que todos los archivos .mat tienen más de 5 segundos de datos.

# ATENCIÓN: NO COGE LOS ÚLTIMOS DATOS DE UN ESTADO MENTAL CUANDO startInOtherMatrix ES TRUE. ESTO ESTÁ HECHO ASÍ PARA QUE SE OBVIEN LOS DATOS ( O ALGUNOS DE ELLOS) DE PAUSAR LA CAPTURA.
def getX_1(x_all, startInOtherMatrix, windowPercent, windowSize, iteration, option, temporalWindowValue):
    # ------------Global variables usadas--------------------
    # x_pieceNumber: representa al x_piece que está recorriendo por su primera columna en x_all.
    global x_pieceNumber
    global x_pieceNumberValidation
    global x_pieceNumberTest
    global lastTemporalWindowValue
    global lastTemporalWindowValueValidation
    global lastTemporalWindowValueTest
    global x_timesTrain
    global x_timesValidation
    global x_timesTest
    # ------------End Global variables usadas----------------


    # indexValue representa el puntero de la primera posición de la matriz x_all que cogemos para formar x_1
    if option == 1:
        indexValue = listIndexX_pieceStarts[x_pieceNumber]
    elif option == 2:
        indexValue = listIndexX_pieceStartsValidation[x_pieceNumberValidation]
    elif option == 3:
        indexValue = listIndexX_pieceStartsTest[x_pieceNumberTest]

    # Columnas de x_all:
    totalColumns = x_all.shape[1]

    # columnSize representa el tamaño de ventana. En este caso es 128Hz por 5s = 640.
    columnSize = windowSize

    # rangePointerActual representa la tupla de valor (f,l) siendo f el primer valor del rango de la ventana temporal actual y l el último valor.
    rangePointerActual = (0, 0)

    # toDecrease contiene el valor que hay que restar a lastTemporalWindowValue para que se convierta en el primer valor siguiente del rango cuando sea necesario. El valor es columnSize - (columnSize*windowPercent)
    toDecrease = round((columnSize * windowPercent), 0)
    # Cogemos desde el primer elemento hasta el tamaño de columna en primera instancia
    # Primera iteración o se coge de otro archivo


    if (temporalWindowValue == 0):

        rangePointerActual = (temporalWindowValue, columnSize)
        temporalWindowValue = columnSize

    else:
        # Otras iteraciones
        # Si se debe entrar en otro fichero, lastTemporalWindowValue es el primer valor de ese fichero
        if (startInOtherMatrix):
            temporalWindowValue = indexValue
            # Si el tamaño del último elemento del rango es mayor al total, entonces se coge una ventana temporal de rango (listIndexX_pieceStarts[penúltimo elemento], últimoElementoDeLaMatriz)
            if (temporalWindowValue + columnSize) >= totalColumns:
                rangePointerActual = (listIndexX_pieceStarts[-1] - columnSize, listIndexX_pieceStarts[-1])
                temporalWindowValue = 0
                if option == 1:
                    x_pieceNumber = 0
                    x_timesTrain += 1
                elif option == 2:
                    x_pieceNumberValidation = 0
                    x_timesValidation += 1
                elif option == 3:
                    x_pieceNumberTest = 0
                    x_timesTest += 1
            else:
                firstTemporalWindowValue = temporalWindowValue
                temporalWindowValue = firstTemporalWindowValue + columnSize

                rangePointerActual = (firstTemporalWindowValue, temporalWindowValue)
        # Si el valor de lastTemporalWindowValue ha pasado el total de columnas de la matriz entonces se coge la última ventana temporal posible y se reinician los punteros
        elif (temporalWindowValue + columnSize >= totalColumns):

            rangePointerActual = (totalColumns - columnSize, totalColumns)
            temporalWindowValue = 0
            if option == 1:
                x_pieceNumber = 0
                x_timesTrain += 1
            elif option == 2:
                x_pieceNumberValidation = 0
                x_timesValidation += 1
            elif option == 3:
                x_pieceNumberTest = 0
                x_timesTest += 1
        # En el caso en el que no se incumpla ninguna restricción, entonces se prosigue con el algoritmo normalmente
        else:
            firstTemporalWindowValue = (temporalWindowValue) - toDecrease
            temporalWindowValue = firstTemporalWindowValue + columnSize
            rangePointerActual = (firstTemporalWindowValue, temporalWindowValue)

    # Cogemos todas las filas y desde la columna  hasta X
    x_1 = x_all[:, rangePointerActual[0]:rangePointerActual[1]]
    # Este es el vector con los valores diferentes en la última fila de x_1
    uniqueVector = np.unique(x_1[15, :])
    if option == 1:
        lastTemporalWindowValue = temporalWindowValue
    elif option == 2:
        lastTemporalWindowValueValidation = temporalWindowValue
    elif option == 3:
        lastTemporalWindowValueTest = temporalWindowValue
    if uniqueVector.shape[0] != 1:
        if option == 1:
            x_pieceNumber += 1
        elif option == 2:
            x_pieceNumberValidation += 1
        elif option == 3:
            x_pieceNumberTest += 1

        x_1 = getX_1(x_all, True, windowPercentGlobal, columnSize, iteration, option, temporalWindowValue)
    return x_1


# -------------ENDgetX_---------------

######################OBTENCIÓN DE X_PLACEHOLDER ##########

def getX_Y_Placeholder(x_all, option, listIndex, isFirstTime):
    isFirst = isFirstTime
    x_placeholder = []
    y_labels = []
    # Se debe añadir windowSizeGlobal ya que se calcula a partir de ese número
    totalIterations = calculateTotalIterations(x_all, windowSizeGlobal, windowPercentGlobal, listIndex)
    rang = 2
    if (option == 1):
        rang = x_timesTrain
    elif (option == 2):
        rang = x_timesValidation
    elif option == 3:
        rang = x_timesTest
    print(rang)
    while (rang < 1):
        # for i in range(totalIterations):
        # x_TemporalWindow tiene forma [1,8960]
        # y_labels tiene forma [1,2]
        # Se debe añadir windowSizeGlobal ya que se calcula a partir de ese número
        x_TemporalWindow, y_TemporalLabels = getX_AndY_(x_all, False, windowPercentGlobal, windowSizeGlobal, 0, option,
                                                        mathematicOperationRadiobutton)
        if isFirst:
            x_placeholder = x_TemporalWindow
            y_labels = y_TemporalLabels
            isFirst = False
            if (option == 1):
                rang = x_timesTrain
            elif (option == 2):
                rang = x_timesValidation
            elif option == 3:
                rang = x_timesTest
        else:
            x_placeholder = np.concatenate((x_placeholder, x_TemporalWindow))
            y_labels = np.concatenate((y_labels, y_TemporalLabels))
            if (option == 1):
                rang = x_timesTrain
            elif (option == 2):
                rang = x_timesValidation
            elif option == 3:
                rang = x_timesTest
    print('x_timesTrain')
    print(x_timesTrain)
    print('x_timesValidation')
    print(x_timesValidation)
    print('x_timesTest')
    print(x_timesTest)
    return np.array(x_placeholder), np.array(y_labels)


######################FIN DE GET X_PLACEHOLDER ############

# ----------------------------------------------------------------------------
'''
###################Modelado lineal del sistema###############################
'''


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_7x7(x):
    return tf.nn.max_pool(x, ksize=[1, 7, 7, 1],
                          strides=[1, 7, 7, 1], padding='SAME')


# ---- Placeholders del sistema --------
x = tf.placeholder(tf.float32, [None, rowsColumnsReal])
y_ = tf.placeholder(tf.float32, [None, 2])

# W , b , y
# Lineal
W = tf.Variable(tf.random_normal([rowsColumnsReal, 2], stddev=0.01))
B = tf.Variable(tf.random_normal([2], stddev=0.01))
y = tf.matmul(x, W) + B
yS = tf.nn.softmax(tf.matmul(x, W) + B)
# MULTIPERCEPTRON
# WD1 = tf.Variable(tf.random_normal([rowsColumnsReal,numbers_neurons],stddev=0.1))
# WD2 = tf.Variable(tf.random_normal([numbers_neurons,2],stddev=0.1))
# BD1 = tf.Variable(tf.random_normal([numbers_neurons],stddev=0.1))
# BD2 = tf.Variable(tf.random_normal([2],stddev=0.1))

WD1 = tf.get_variable("WD1", shape=[rowsColumnsReal, numbers_neurons],
                      initializer=tf.contrib.layers.xavier_initializer())
WD2 = tf.get_variable("WD2", shape=[numbers_neurons, 2], initializer=tf.contrib.layers.xavier_initializer())
BD1 = tf.Variable(tf.random_normal([numbers_neurons], stddev=0.01))
BD2 = tf.Variable(tf.random_normal([2], stddev=0.01))

# Con esto a 150 entrenamientos y todo igual da 94% de acierto
# WD1 = tf.Variable(tf.random_uniform([rowsColumnsReal,numbers_neurons],minval=-1,maxval=1))
# WD2 = tf.Variable(tf.random_uniform([numbers_neurons,2],minval=-1,maxval=1))
# BD1 = tf.Variable(tf.random_uniform([numbers_neurons],minval=-1,maxval=1))
# BD2 = tf.Variable(tf.random_uniform([2],minval=-1,maxval=1))

yD1 = (tf.matmul(x, WD1)) + BD1
yD2 = (tf.matmul(yD1, WD2)) + BD2
# yD1 = tf.nn.tanh((tf.matmul(x,WD1))+BD1)
# yD2 = tf.nn.tanh((tf.matmul(yD1,WD2))+BD2)


# CNN
x1 = tf.reshape(x, [-1, rowsGlobal, numColumnsCNN, 1])

if (mathematicOperationRadiobutton == "Crudo"):

    # First Convolutional Layer
    W_conv1 = weight_variable([weightCNNShape, weightCNNShape, 1, firstLabelNeurons])
    b_conv1 = bias_variable([firstLabelNeurons])
    h_conv1 = tf.nn.relu(conv2d(x1, W_conv1) + b_conv1)
    h_conv1NoRELU = conv2d(x1, W_conv1) + b_conv1

    h_pool1 = max_pool_2x2(h_conv1)
    # Second Convolutional Layer
    W_conv2 = weight_variable([weightCNNShape, weightCNNShape, firstLabelNeurons, secondLabelNeurons])
    b_conv2 = bias_variable([secondLabelNeurons])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer
    W_fc1 = weight_variable([4 * numColumnsCNNPool * secondLabelNeurons, thirdLabelNeurons])
    b_fc1 = bias_variable([thirdLabelNeurons])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * numColumnsCNNPool * secondLabelNeurons])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Readout Layer
    W_fc2 = weight_variable([thirdLabelNeurons, 2])
    b_fc2 = bias_variable([2])
    # y_convSoftmax=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_conv = (tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
else:
    # First Convolutional Layer
    W_conv1 = weight_variable([weightCNNShape, weightCNNShape, 1, firstLabelNeurons])
    b_conv1 = bias_variable([firstLabelNeurons])
    h_conv1 = tf.nn.relu(conv2d(x1, W_conv1) + b_conv1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([weightCNNShape, weightCNNShape, firstLabelNeurons, secondLabelNeurons])
    b_conv2 = bias_variable([secondLabelNeurons])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # Densely Connected Layer
    W_fc1 = weight_variable([14 * numColumnsCNN * secondLabelNeurons, thirdLabelNeurons])
    b_fc1 = bias_variable([thirdLabelNeurons])
    h_pool2_flat = tf.reshape(h_conv2, [-1, 14 * numColumnsCNN * secondLabelNeurons])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Readout Layer
    W_fc2 = weight_variable([thirdLabelNeurons, 2])
    b_fc2 = bias_variable([2])
    # y_convSoftmax=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_conv = (tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# ----Fin de variables------

# ---- Entrenamiento de la red neuronal ----

################LINEAL################
# cross_entropy = tf.reduce_mean(-tf.reduce_sum((y_-1) * tf.log(y), reduction_indices=[1]))
cross_entropyL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, (y_ - 1)))
# cross_entropyL = tf.nn.softmax_cross_entropy_with_logits(y, (y_-1))
# error = tf.reduce_mean(tf.square(tf.div((y_-1), tf.log(y))*100))
# error2 = tf.div((y_-1), tf.log(y))
cross_entropyL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, (y_ - 1)))
# Algoritmo de reducción del error:
train_step_Lineal = tf.train.GradientDescentOptimizer(learningRateLineal).minimize(cross_entropyL)

################################################
################MULTILAYER################
error_Multilayer = tf.reduce_mean(tf.square(yD2 - (y_ - 1)))
train_step_Multilayer = tf.train.AdamOptimizer(learningRateMultilayer).minimize(error_Multilayer)
################################################
################CNN################
# error_CNN = tf.reduce_mean(tf.square(tf.abs(y_conv) - (y_ -1)))
cross_entropyCNN = tf.nn.softmax_cross_entropy_with_logits(y_conv, (y_ - 1))
cross_entropyCNNMean = tf.reduce_mean(cross_entropyCNN)
# cross_entropyCNN2 = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step_CNN = tf.train.AdamOptimizer(learningRateCNN).minimize(cross_entropyCNN)

################################################


# ----------CÁLCULO DEL ERROR------------
################LINEAL################
correct_prediction_Lineal = tf.equal(tf.argmax(y, 1), tf.argmax((y_ - 1), 1))
accuracy_Lineal = tf.reduce_mean(tf.cast(correct_prediction_Lineal, tf.float32))
################MULTILAYER################
correct_prediction_Multilayer = tf.equal(tf.argmax(yD2, 1), tf.argmax((y_ - 1), 1))
accuracy_Multilayer = tf.reduce_mean(tf.cast(correct_prediction_Multilayer, tf.float32))
################CNN################
correct_prediction_CNN = tf.equal(tf.argmax(y_conv, 1), tf.argmax((y_ - 1), 1))
accuracy_CNN = tf.reduce_mean(tf.cast(correct_prediction_CNN, tf.float32))
# --------FIN CÁLCULO DEL ERROR -------


# --------------------------------------------------------------------------
# ---- Inicio de Sesion de tensorFlow ----
# --------------------------------------------------------------------------
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)


# --------------------------------------------------------------------------

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def checkIfZeroColumns(matrix):
    print(str(column(matrix, 71399)))
    print(str(column(matrix, 71400)))

    print(str(column(matrix, 71397)))
    for x in range(matrix.shape[1]):
        if (np.sum(column(matrix, x)) == 0):
            print('True', x)
    print('False')


def calculateGlobalModels():
    global trainPlaceholder, trainLabels, validationPlaceholder, validationLabels, testPlaceholder, testLabels, rowsColumnsReal
    # Conjunto de entrenamiento:
    x_all = getx_all(1, testSetToAccuracy)
    # Conjunto de validación:
    validationSet = getx_all(2, testSetToAccuracy)
    # Conjunto de test:
    testSet = getx_all(3, testSetToAccuracy)
    ###################DIFERENTES PLACEHOLDERS #################
    trainPlaceholder, trainLabels = getX_Y_Placeholder(x_all, 1, listIndexX_pieceStarts, True)
    validationPlaceholder, validationLabels = getX_Y_Placeholder(validationSet, 2, listIndexX_pieceStartsValidation,
                                                                 True)
    testPlaceholder, testLabels = getX_Y_Placeholder(testSet, 3, listIndexX_pieceStartsTest, True)

    print('trainPlaceholder')
    print(trainPlaceholder)
    print('trainPlaceholder')
    print(str(trainPlaceholder.shape))
    x_all = []
    validationSet = []
    testSet = []


###################FIN DE ASOCIACIÓN DE PLACEHOLDERS########






# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

accuraciesValidationLineal = []
accuraciesTrainLineal = []
accuraciesValidationLinealT = []
accuraciesTrainLinealT = []
accuraciesMeanLineal = []

accuraciesValidationMultilayer = []
accuraciesTrainMultilayer = []
accuraciesValidationMultilayerT = []
accuraciesTrainMultilayerT = []
accuraciesMeanMultilayer1 = []

accuraciesTrainCNN = []
accuraciesValidationCNN = []
accuraciesValidationCNNT = []
accuraciesTrainCNNT = []
accuraciesMeanCNN = []

validationLossLineal = []
validationLossMultilayer = []
validationLossCNN = []

trainLossLineal = []
trainLossMultilayer = []
trainLossCNN = []

test_accuracyLinealGlobal = -1
test_accuracyMULTICAPAGlobal = -1
test_accuracyCNNGlobal = -1
# DEEP LEARNING:

saver = tf.train.Saver()


def deepOrLineal():
    if isLineal == 1:
        linealModel()
    if isMultilayer == 1:
        multilayerModel()
    if isCNN == 1:
        convolutionalModel()


######################LINEAL MODEL FUNCTION#####################
def linealModel():
    global test_accuracyLinealGlobal
    stopForMean = 0
    bestAccuracy = 0.
    firstTime = 1
    global trainingLinealFinal
    print('########ENTRENAMIENTO LINEAL########')

    for i in range(numOfTraining):

        train_accuracy = accuracy_Lineal.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1}) * 100

        train_step_Lineal.run(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 0.5})
        if i == 0:
            save_path = saver.save(sess, "/tmp/modelLBest.ckpt")
        # y1 = y.eval(feed_dict={x:trainPlaceholder, y_:trainLabels, keep_prob: 1.0})
        validation_accuracy = accuracy_Lineal.eval(
            feed_dict={x: validationPlaceholder, y_: validationLabels, keep_prob: 1.0}) * 100
        crossEntropyPTrain = cross_entropyL.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0})
        crossEntropyPValidation = cross_entropyL.eval(
            feed_dict={x: validationPlaceholder, y_: validationLabels, keep_prob: 1.0})

        accuracyMeanLineal = ((train_accuracy) + (validation_accuracy)) / 2

        trainLossLineal.append(crossEntropyPTrain)
        validationLossLineal.append(crossEntropyPValidation)
        accuraciesTrainLineal.append(train_accuracy)
        accuraciesValidationLineal.append(validation_accuracy)

        if i % 30 == 0:
            print(str(now()) + " ||Step %d, training accuracy LINEAL %g " % (i, train_accuracy))
            print("validation accuracy LINEAL %g" % (validation_accuracy))

            accuraciesMeanLineal.append(accuracyMeanLineal)
            if ((accuracyMeanLineal > 60)):
                if firstTime == 1:
                    save_path = saver.save(sess, "/tmp/modelLBest.ckpt")
                    bestAccuracy = accuracyMeanLineal
                    firstTime == 0
                elif (accuracyMeanLineal > bestAccuracy):
                    bestAccuracy = accuracyMeanLineal
                    save_path = saver.save(sess, "/tmp/modelLBest.ckpt")
            if (stopForMean == 0):
                # Save the variables to disk.
                save_path = saver.save(sess, "/tmp/modelLLast.ckpt")
            if (i is not 0):
                if ((validationLossLineal[i - 5] == validationLossLineal[i]) or (np.isnan(validationLossLineal).any())):
                    print('NO SE ESTÁ APRENDIENDO EN LINEAL, SE PARA')
                    trainingLinealFinal = i
                    break

        if (validation_accuracy >= threshold or train_accuracy == 100):
            trainingLinealFinal = i
            break

        if ((accuracyMeanLineal > 85) and (i > 400)):
            print('Se ha parado el entrenamiento porque ha superado cota de acierto')
            stopForMean = 1
            bestAccuracy = accuracyMeanLineal
            trainingLinealFinal = i
            break

    print('Entrenamiento y validación LINEAL terminada')
    if trainingLinealFinal == 0:
        trainingLinealFinal = numOfTraining
    wx = W.eval()
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(folder + str(nowDay()) + '--W--LINEAL', wx)

    #########PRECISIÓN DEL TEST##############
    if stopForMean == 1:
        test_accuracyLinealMean = accuracy_Lineal.eval(feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})
        print("Test accuracy LINEAL last best accuracy mean %g " % (test_accuracyLinealMean))

        # Restore variables from disk.
        saver.restore(sess, "/tmp/modelLLast.ckpt")
        test_accuracyLinealTrain = accuracy_Lineal.eval(feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})

        print("Test accuracy LINEAL last weight %g " % (test_accuracyLinealTrain))
    ##Para el peso normal##
    else:

        test_accuracyLinealTrain = accuracy_Lineal.eval(feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})

        print("Test accuracy LINEAL last weight %g " % (test_accuracyLinealTrain))

        maxAccuracy1 = np.amax(accuraciesMeanLineal)
        maxAccuracyIndex1 = np.argmax(accuraciesMeanLineal)

        # Restore variables from disk.
        saver.restore(sess, "/tmp/modelLBest.ckpt")

        print('Max mean accuracy train and validation :')
        print(str(maxAccuracy1))
        test_accuracyLinealMean = accuracy_Lineal.eval(feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})

        print("Test accuracy LINEAL last best accuracy mean %g " % (test_accuracyLinealMean))

        if test_accuracyLinealTrain > test_accuracyLinealMean:
            test_accuracyLinealGlobal = test_accuracyLinealTrain
        else:
            test_accuracyLinealGlobal = test_accuracyLinealMean
            #########PRECISIÓN DEL TEST##############
    print('########END ENTRENAMIENTO LINEAL########')


##################END LINEAL MODEL FUNCTION #####################

######################MULTILAYER MODEL FUNCTION#####################
def multilayerModel():
    global test_accuracyMULTICAPAGlobal, trainingMultilayerFinal
    stopForMean = 0
    bestAccuracy = 0.
    firstTime = 1
    print('########ENTRENAMIENTO MULTICAPA########')
    for i in range(numOfTraining):

        train_accuracy = accuracy_Multilayer.eval(
            feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0}) * 100.

        train_step_Multilayer.run(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 0.5})
        if i == 0:
            w1 = WD1.eval()
            w2 = WD2.eval()
            print('w1')
            print(w1)
            print('w2')
            print(w2)
            save_path = saver.save(sess, "/tmp/modelMBest.ckpt")
        y1 = yD2.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0})
        validation_accuracy = (accuracy_Multilayer.eval(
            feed_dict={x: validationPlaceholder, y_: validationLabels, keep_prob: 1.0})) * 100.
        errorTrain = error_Multilayer.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0})
        errorValidation = error_Multilayer.eval(
            feed_dict={x: validationPlaceholder, y_: validationLabels, keep_prob: 1.0})
        accuracyMeanMultilayer1 = ((train_accuracy) + (validation_accuracy)) / 2

        trainLossMultilayer.append(errorTrain)
        validationLossMultilayer.append(errorValidation)
        accuraciesTrainMultilayer.append(train_accuracy)
        accuraciesValidationMultilayer.append(validation_accuracy)

        if i % 30 == 0:
            print(str(now()) + "|| Step %d, training accuracy MULTICAPA %g" % (i, train_accuracy))
            print("validation accuracy MULTICAPA %g" % (validation_accuracy))
            if i % 30 == 0:

                accuraciesMeanMultilayer1.append(accuracyMeanMultilayer1)

                print("Test accuracy MULTICAPA last weight %g " % accuracy_Multilayer.eval(
                    feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0}))

                if ((accuracyMeanMultilayer1 > 60)):
                    if firstTime == 1:
                        save_path = saver.save(sess, "/tmp/modelMBest.ckpt")
                        bestAccuracy = accuracyMeanMultilayer1
                        firstTime == 0
                    elif (accuracyMeanMultilayer1 > bestAccuracy):
                        bestAccuracy = accuracyMeanMultilayer1
                        save_path = saver.save(sess, "/tmp/modelMBest.ckpt")
                if (stopForMean == 0):
                    # Save the variables to disk.
                    save_path = saver.save(sess, "/tmp/modelMLast.ckpt")

            if (i is not 0):
                if ((validationLossMultilayer[i - 5] == validationLossMultilayer[i - 1]) or (
                np.isnan(validationLossMultilayer).any())):
                    print('NO SE ESTÁ APRENDIENDO EN MULTICAPA, SE PARA')
                    trainingMultilayerFinal = i
                    break
        if (validation_accuracy >= threshold or train_accuracy == 100):
            trainingMultilayerFinal = i
            break
        if ((accuracyMeanMultilayer1 > 85) and (i > 400)):
            print('Se ha parado el entrenamiento porque ha superado cota de acierto')
            stopForMean = 1
            bestAccuracy = accuracyMeanMultilayer1
            trainingMultilayerFinal = i
            break

    if trainingMultilayerFinal == 0:
        trainingMultilayerFinal = numOfTraining
    print('Entrenamiento y validación MULTICAPA terminada')
    print(str(now()) + "|| Step %d, training accuracy MULTICAPA %g" % (i, train_accuracy))
    print("validation accuracy MULTICAPA %g" % accuracy_Multilayer.eval(
        feed_dict={x: validationPlaceholder, y_: validationLabels, keep_prob: 1.0}))
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(folder + str(nowDay()) + '--W--MULTICAPA', WD1.eval())

    if stopForMean == 1:
        test_accuracyMULTICAPA1 = accuracy_Multilayer.eval(
            feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})
        # test_accuracyMULTICAPAGlobal2 = test_accuracyMULTICAPA2
        print("Test accuracy MULTICAPA last best accuracy mean %g " % accuracy_Multilayer.eval(
            feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0}))

        # Restore variables from disk.
        saver.restore(sess, "/tmp/modelMLast.ckpt")
        test_accuracyMULTICAPA2 = accuracy_Multilayer.eval(
            feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})

        print("Test accuracy MULTICAPA last weight %g " % accuracy_Multilayer.eval(
            feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0}))
    ##Para el peso normal##
    else:

        test_accuracyMULTICAPA1 = accuracy_Multilayer.eval(
            feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})

        print("Test accuracy MULTICAPA last weight %g " % accuracy_Multilayer.eval(
            feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0}))

        maxAccuracy1 = np.amax(accuraciesMeanMultilayer1)
        maxAccuracyIndex1 = np.argmax(accuraciesMeanMultilayer1)

        # Restore variables from disk.
        saver.restore(sess, "/tmp/modelMBest.ckpt")

        print('Max mean accuracy train and validation :')
        print(str(maxAccuracy1))
        test_accuracyMULTICAPA2 = accuracy_Multilayer.eval(
            feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})
        # test_accuracyMULTICAPAGlobal2 = test_accuracyMULTICAPA2
        print("Test accuracy MULTICAPA last best accuracy mean %g " % accuracy_Multilayer.eval(
            feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0}))

        if test_accuracyMULTICAPA1 > test_accuracyMULTICAPA2:
            test_accuracyMULTICAPAGlobal = test_accuracyMULTICAPA1
        else:
            test_accuracyMULTICAPAGlobal = test_accuracyMULTICAPA2
    print('########FIN ENTRENAMIENTO MULTICAPA########')


##################END MULTILAYER MODEL FUNCTION #####################

######################CONVOLUTIONAL MODEL FUNCTION#####################
def convolutionalModel():
    global test_accuracyCNNGlobal, trainingCNNFinal
    stopForMean = 0
    bestAccuracy = 0.
    firstTime = 1

    for i in range(trainingCNNFinal):
        if (i == 0):
            x3 = x1.eval(feed_dict={x: trainPlaceholder})
            print('trainPlaceholder')
            print(trainPlaceholder.shape)
            print('x3.shape')
            print(x3.shape)
            h_conv1X = h_conv1.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0})
            print('h_conv1X.shape')
            print(h_conv1X.shape)
            # h_pool1X = h_pool1.eval(feed_dict={x:trainPlaceholder, y_:trainLabels, keep_prob: 1.0})
            # print ('h_pool1X.shape'
            # print h_pool1X.shape
            h_conv24 = h_conv2.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0})
            print('h_conv24')
            print(h_conv24.shape)
            # h_pool22 = h_pool2.eval(feed_dict={x:trainPlaceholder, y_:trainLabels, keep_prob: 1.0})
            # print ('h_pool2'
            # print h_pool22.shape
            h_pool2_flat2 = h_pool2_flat.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0})
            print('h_pool2_flat')
            print(h_pool2_flat2.shape)
            h_fc12 = h_fc1.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0})
            print('h_fc1')
            print(h_fc12.shape)
            y_conv2 = y_conv.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0})
            print('y_conv')
            print(y_conv2)
            save_path = saver.save(sess, "/tmp/modelCBest.ckpt")
        # wx = W_conv1.eval()
        train_accuracy = accuracy_CNN.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0}) * 100
        train_step_CNN.run(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 0.5})
        validation_accuracy = accuracy_CNN.eval(
            feed_dict={x: validationPlaceholder, y_: validationLabels, keep_prob: 1.0}) * 100

        crossEntropyTrain = cross_entropyCNNMean.eval(feed_dict={x: trainPlaceholder, y_: trainLabels, keep_prob: 1.0})
        crossEntropyValidation = cross_entropyCNNMean.eval(
            feed_dict={x: validationPlaceholder, y_: validationLabels, keep_prob: 1.0})

        accuracyMeanCNN = ((train_accuracy) + (validation_accuracy)) / 2

        trainLossCNN.append(crossEntropyTrain)
        validationLossCNN.append(crossEntropyValidation)
        accuraciesTrainCNN.append(train_accuracy)
        accuraciesValidationCNN.append(validation_accuracy)

        if i % 9 == 0:
            print(str(now()) + "|| Step %d, training accuracy CNN %g" % (i, train_accuracy))
            print("validation accuracy CNN %g" % validation_accuracy)

            accuraciesMeanCNN.append(accuracyMeanCNN)

            if (accuracyMeanCNN > 50):
                if firstTime == 1:
                    save_path = saver.save(sess, "/tmp/modelCBest.ckpt")
                    bestAccuracy = accuracyMeanCNN
                    firstTime == 0
                elif (accuracyMeanCNN > bestAccuracy):
                    bestAccuracy = accuracyMeanCNN
                    save_path = saver.save(sess, "/tmp/modelCBest.ckpt")
            if (stopForMean == 0):
                # Save the variables to disk.
                save_path = saver.save(sess, "/tmp/modelCLast.ckpt")

            if (i is not 0):
                if ((validationLossCNN[-4] == validationLossCNN[-1])):
                    # or (np.isnan(validationLossCNN).any()
                    print('NO SE ESTÁ APRENDIENDO EN CNN, SE PARA')
                    trainingCNNFinal = i
                    break
        if (validation_accuracy >= threshold or train_accuracy == 100):
            trainingCNNFinal = i
            break
        if (i % 110 == 0) and not (i == 0):
            if (accuraciesTrainCNN[-99] == accuraciesTrainCNN[-1]):
                print('NO SE APRENDE EN CNN PORQUE ENTRENAMIENTO ESTÁ ESTANCADO')
                trainingCNNFinal = i
                break
        if ((accuracyMeanCNN > 85) and (i > 90)):
            print('Se ha parado el entrenamiento porque ha superado cota de acierto')
            stopForMean = 1
            bestAccuracy = accuracyMeanCNN
            trainingCNNFinal = i
            break

    if trainingCNNFinal == 0:
        trainingCNNFinal = trainingCNNFinal
    print('Entrenamiento y validación CNN terminada')
    #########PRECISIÓN DEL TEST##############
    if stopForMean == 1:
        test_accuracyCNNMean = accuracy_CNN.eval(feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})
        print("Test accuracy CNN last best accuracy mean %g " % (test_accuracyCNNMean))

        # Restore variables from disk.
        saver.restore(sess, "/tmp/modelCLast.ckpt")
        test_accuracyCNNTrain = accuracy_CNN.eval(feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})

        print("Test accuracy CNN last weight %g " % (test_accuracyCNNTrain))
    ##Para el peso normal##
    else:

        test_accuracyCNNTrain = accuracy_CNN.eval(feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})

        print("Test accuracy CNN last weight %g " % (test_accuracyCNNTrain))

        maxAccuracy1 = np.amax(accuraciesMeanCNN)
        maxAccuracyIndex1 = np.argmax(accuraciesMeanCNN)

        # Restore variables from disk.
        saver.restore(sess, "/tmp/modelCBest.ckpt")

        print('Max mean accuracy train and validation :')
        print(str(maxAccuracy1))
        test_accuracyCNNMean = accuracy_CNN.eval(feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})

        print("Test accuracy CNN last best accuracy mean %g " % (test_accuracyCNNMean))

        if test_accuracyCNNTrain > test_accuracyCNNMean:
            test_accuracyCNNGlobal = test_accuracyCNNTrain
        else:
            test_accuracyCNNGlobal = test_accuracyCNNMean
            #########PRECISIÓN DEL TEST##############

    y_conv2 = y_conv.eval(feed_dict={x: testPlaceholder, y_: testLabels, keep_prob: 1.0})
    wx = W_fc2.eval()
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(folder + str(nowDay()) + '--W--CNN', wx)
    totalElements = float(y_conv2.shape[0])
    relaxed = 0
    concentrated = 0
    for elm in y_conv2:
        maximun = np.argmax(elm)
        if (maximun == 0):
            relaxed += 1.
        else:
            concentrated += 1.
    relaxedEnd = relaxed / totalElements * 100
    concentratedEnd = float(concentrated / totalElements) * 100
    print("test accuracy CNN %g " % (test_accuracyCNNGlobal) + '%')
    print('########PROBABILIDAD TEST CNN########')
    print("Probabilidad para el conjunto de testeo:")
    print("Probabilidad de ser relajado: " + str(relaxedEnd) + '%')
    print("Probabilidad de ser concentrado: " + str(concentratedEnd) + '%')

    print('########FIN ENTRENAMIENTO CNN########')


###################END CONVOLUTIONAL MODEL FUNCTION#####################

def accuracyTest():
    print('Entrenamiento y validación terminada finalizado')

    # test_accuracy = accuracy2.eval(feed_dict={ x:testPlaceholder, y_:testLabels, keep_prob: 1.0})
    # print("test accuracy %g"%accuracy2.eval(feed_dict= {x:testPlaceholder, y_:testLabels, keep_prob: 1.0}))


def showFigure():
    if not os.path.exists(folder2):
        os.makedirs(folder2)
    global validationLossMultilayer
    global validationLossLineal
    l = plt.figure(1)
    plt.title("LINEAL MODEL RESULTS")
    plt.xlabel(
        "ITERATIONS |" + 'SO:' + str(mathematicOperationRadiobutton) + ' |W: ' + str(windowSizeGlobal) + ' |P: ' + str(
            windowPercentGlobal * 100) + '% ' + ' |ST: ' + str(numOfTraining) + ' |LR: ' + str(learningRateLineal))
    plt.ylabel("ACCURACY (BLUE --> Set Training, RED --> Set Validation)")
    plt.plot(accuraciesValidationLineal, 'r')
    plt.plot(accuraciesTrainLineal, 'b')
    plt.savefig(folder2 + str(nowDay()) + '-LINEAL-' + 'SO' + str(mathematicOperationRadiobutton) + ' |W: ' + str(
        windowSizeGlobal) + ' |P: ' + str(windowPercentGlobal) + ' |ST: ' + str(numOfTraining) + ' |LR: ' + str(
        learningRateLineal) + '.png')

    m = plt.figure(2)
    plt.title("MULTILAYER MODEL RESULTS")
    plt.xlabel("ITERATIONS | " + ' SO: ' + str(mathematicOperationRadiobutton) + ' |W: ' + str(
        windowSizeGlobal) + ' |P: ' + str(windowPercentGlobal * 100) + '% ' + ' |ST: ' + str(
        numOfTraining) + ' |LR: ' + str(learningRateMultilayer) + ' |N: ' + str(numbers_neurons))
    plt.ylabel("ACCURACY (BLUE --> Set Training, RED --> Set Validation)")
    plt.plot(accuraciesValidationMultilayer, 'r')
    plt.plot(accuraciesTrainMultilayer, 'b')
    plt.savefig(folder2 + str(nowDay()) + '-MULTICAPA-' + 'SO' + str(mathematicOperationRadiobutton) + ' |W: ' + str(
        windowSizeGlobal) + ' |P: ' + str(windowPercentGlobal) + ' |ST: ' + str(numOfTraining) + ' |LR: ' + str(
        learningRateMultilayer) + ' |N: ' + str(numbers_neurons) + '.png')

    c = plt.figure(3)
    plt.title("CNN MODEL RESULTS")
    plt.xlabel(
        "ITERATIONS | " + 'SO: ' + str(mathematicOperationRadiobutton) + '|ST: ' + str(numOfTraining) + ' |LR: ' + str(
            learningRateCNN) + ' |N1: ' + str(firstLabelNeurons) + ' |N2: ' + str(secondLabelNeurons) + ' |N3: ' + str(
            thirdLabelNeurons) + ' |W: ' + str(weightCNNShape))
    plt.ylabel("ACCURACY (BLUE --> Set Training, RED --> Set Validation)")
    plt.plot(accuraciesValidationCNN, 'r')
    plt.plot(accuraciesTrainCNN, 'b')
    plt.savefig(folder2 + str(nowDay()) + '-CNN-' + 'SO: ' + str(mathematicOperationRadiobutton) + '|ST: ' + str(
        numOfTraining) + ' |LR: ' + str(learningRateCNN) + ' |N1: ' + str(firstLabelNeurons) + ' |N2: ' + str(
        secondLabelNeurons) + ' |N3: ' + str(thirdLabelNeurons) + ' |W: ' + str(weightCNNShape) + '.png')

    if isLineal == 1:
        l.show()
        if ((not np.isnan(validationLossLineal).any()) and showCosts == 1):
            lc = plt.figure(4)
            plt.title("TEST ACCURACY: " + str(test_accuracyLinealGlobal * 100)[:6] + '% |' + "LINEAL MODEL LOSS ")
            plt.xlabel("ITERATIONS | " + ' SO: ' + str(mathematicOperationRadiobutton) + ' |W: ' + str(
                windowSizeGlobal) + ' |P: ' + str(windowPercentGlobal * 100) + '% ' + ' |T: ' + str(
                numOfTraining) + ' |LR: ' + str(learningRateLineal))
            plt.ylabel("Lineal Loss |Green --> Train Set , Orange --> Validation Set|")
            plt.plot(trainLossLineal, 'g')
            plt.plot(validationLossLineal, 'orange')

            lc.show()
            plt.savefig(
                folder2 + str(nowDay()) + '-LINEAL-' + 'SO: ' + str(mathematicOperationRadiobutton) + '|ST: ' + str(
                    numOfTraining) + ' |LR: ' + str(learningRateCNN) + ' |N1: ' + str(
                    firstLabelNeurons) + ' |N2: ' + str(secondLabelNeurons) + ' |N3: ' + str(
                    thirdLabelNeurons) + ' |W: ' + str(weightCNNShape) + '.png')

    if isMultilayer == 1:
        m.show()
        if ((not np.isnan(validationLossMultilayer).any()) and showCosts == 1):
            mc = plt.figure(5)
            plt.title("TEST ACCURACY: " + str(test_accuracyMULTICAPAGlobal * 100)[:6] + '% |' + "MULTILAYER MODEL LOSS")
            plt.xlabel(" ITERATIONS| " + 'SO: ' + str(mathematicOperationRadiobutton) + ' |W: ' + str(
                windowSizeGlobal) + ' |P: ' + str(windowPercentGlobal * 100) + '% ' + ' |ST: ' + str(
                numOfTraining) + ' |LR: ' + str(learningRateMultilayer) + ' |N: ' + str(numbers_neurons))
            plt.ylabel("Multilayer Loss |Green --> Train Set , Orange --> Validation Set|")
            # validationLossMultilayer = np.array(validationLossMultilayer)
            # validationLossMultilayer=deleteMaxs(validationLossMultilayer)
            # plt.ylim([0,2000])
            plt.plot(trainLossMultilayer, 'g')
            plt.plot(validationLossMultilayer, 'orange')

            mc.show()
            plt.savefig(
                folder2 + str(nowDay()) + '-MULTILAYER-' + 'SO: ' + str(mathematicOperationRadiobutton) + ' |W: ' + str(
                    windowSizeGlobal) + ' |P: ' + str(windowPercentGlobal * 100) + '% ' + ' |ST: ' + str(
                    numOfTraining) + ' |LR: ' + str(learningRateMultilayer) + ' |N: ' + str(numbers_neurons) + '.png')

    if isCNN == 1:
        c.show()
        if ((not np.isnan(validationLossCNN).any()) and showCosts == 1):
            mc = plt.figure(6)
            plt.title("TEST ACCURACY: " + str(test_accuracyCNNGlobal * 100)[:6] + '% |' + "CNN MODEL LOSS")
            plt.xlabel("ITERATIONS| " + 'SO: ' + str(mathematicOperationRadiobutton) + '|ST: ' + str(
                numOfTraining) + ' |LR: ' + str(learningRateCNN) + ' |N1: ' + str(firstLabelNeurons) + ' |N2: ' + str(
                secondLabelNeurons) + ' |N3: ' + str(thirdLabelNeurons) + ' |W: ' + str(weightCNNShape))
            plt.ylabel("CNN Loss|Green --> Train Set , Orange --> Validation Set|")
            # validationLossMultilayer = np.array(validationLossMultilayer)
            # validationLossMultilayer=deleteMaxs(validationLossMultilayer)
            plt.plot(trainLossCNN, 'g')
            plt.plot(validationLossCNN, 'orange')
            mc.show()
            plt.savefig(
                folder2 + str(nowDay()) + '-CNN-' + 'SO: ' + str(mathematicOperationRadiobutton) + '|ST: ' + str(
                    numOfTraining) + ' |LR: ' + str(learningRateCNN) + ' |N1: ' + str(
                    firstLabelNeurons) + ' |N2: ' + str(secondLabelNeurons) + ' |N3: ' + str(
                    thirdLabelNeurons) + ' |W: ' + str(weightCNNShape) + '.png')



if notOperate is not 1:
    calculateGlobalModels()
    deepOrLineal()
    accuracyTest()
    if showGraph:
        showFigure()
    print('#######CONFIGURACIÓN#######')
    print('Método estadístico elegido: ' + str(mathematicOperationRadiobutton))
    print('Tamaño ventana temporal : ' + str(windowSizeGlobal))
    print('Porcentaje: ' + str(windowPercentGlobal * 100) + '%')
    print('Neuronas multicapa: ' + str(numbers_neurons))
    print('Número de entrenamientos: ' + str(numOfTraining))
    print('Threshold: ' + str(threshold))
    print('Ratio de aprendizaje Lineal: ' + str(learningRateLineal))
    print('Ratio de aprendizaje Multicapa: ' + str(learningRateMultilayer))
    print('Ratio de aprendizaje CNN: ' + str(learningRateCNN))
    print('Neuronas primera capa CNN: ' + str(firstLabelNeurons))
    print('Neuronas segunda capa CNN: ' + str(secondLabelNeurons))
    print('Neuronas tercera capa CNN: ' + str(thirdLabelNeurons))
    print('Dimension de pesos CNN: ' + str(weightCNNShape))

    print('#######TESTS#######')
    print('Precisión test LINEAL: ' + str(test_accuracyLinealGlobal * 100) + '%')
    print('Precisión test MULTICAPA: ' + str(test_accuracyMULTICAPAGlobal * 100) + '%')
    print('Precisión test CNN: ' + str(test_accuracyCNNGlobal * 100) + '%')

    print('#######Entrenamientos#######')
    print('Entrenamientos modelo Lineal: ' + str(trainingLinealFinal))
    print('Entrenamientos modelo Multicapa: ' + str(trainingMultilayerFinal))
    print('Entrenamientos modelo CNN: ' + str(trainingCNNFinal))

else:
    print('No se ha elegido ningún modelo. Por favor, elija un modelo')

if notOperate is not 1:
    print('PULSA INTRO PARA FINALIZAR ')
    input()
