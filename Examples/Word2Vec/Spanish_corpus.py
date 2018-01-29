"""
From https://relopezbriega.github.io/blog/2017/09/23/procesamiento-del-lenguaje-natural-con-python/

En general, en Procesamiento del Lenguaje Natural se utilizan seis niveles de comprensión con el objetivo de descubrir el significado del discurso. Estos niveles son:

Nivel fonético: Aquí se presta atención a la fonética, la forma en que las palabras son pronunciadas. Este nivel es importante cuando procesamos la palabra hablada, no así cuando trabajamos con texto escrito.
Nivel morfológico: Aquí nos interesa realizar un análisis morfológico del discurso; estudiar la estructura de las palabras para delimitarlas y clasificarlas.
Nivel sintáctico: Aquí se realiza un análisis de sintaxis, el cual incluye la acción de dividir una oración en cada uno de sus componentes.
Nivel semántico: Este nivel es un complemente del anterior, en el análisis semántico se busca entender el significado de la oración. Las palabras pueden tener múltiples significados, la idea es identificar el significado apropiado por medio del contexto de la oración.
Nivel discursivo: El nivel discursivo examina el significado de la oración en relación a otra oración en el texto o párrafo del mismo documento.
Nivel pragmático: Este nivel se ocupa del análisis de oraciones y cómo se usan en diferentes situaciones. Además, también cómo su significado cambia dependiendo de la situación.

Todos los niveles descritos aquí son inseparables y se complementan entre sí. El objetivo de los sistemas de NLP es incluir estas definiciones en una computadora y luego usarlas para crear una oración estructurada y sin ambigüedades con un significado bien definido.
"""



# IMPORTANTE: PARA INSTALAR TEXTACY HACE FALTA Microsoft Visual C++ 14.0. Si no lo instalas, te dará el siguiente error:
#  error: Microsoft Visual C++ 14.0 is required. Get it with "Microsoft Visual C++ Build Tools":
#  http://landinghub.visualstudio.com/visual-cpp-build-tools
# Instalarlo con la configuración por defecto, sino, no funcionará
import textacy
from textacy.datasets import Wikipedia
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import warnings; warnings.simplefilter('ignore')
from UsefulTools.UtilsFunctions import *
# importando gensim y TSNE para graficar
import gensim
from sklearn.manifold import TSNE

# función leer texto
def leer_texto(texto):
    """Funcion auxiliar para leer un archivo de texto"""
    with open(texto, 'r') as text:
        return text.read()

wp = Wikipedia(data_dir='D:\Machine_Learning\Corpus\wikipedia_textacy', lang='es', version='latest')
#wp.download()  # Para cuando sea necesario descargar el corpus

pt("wiki info",wp.info)

frases = []


for text in wp.texts(min_len=1000, limit=2):
    frases.append(text)
    # Para mostrar algo del texto descargado
    #print(text[:370], "\n")
pt("frases", frases)
