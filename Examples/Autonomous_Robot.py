"""
Leemos del fichero (que es el csv que contiene la información recogida de Aquiles) las columnas:
"Tipo Evento", "Contenido", "Imagen".

Los pasos que se siguen son los siguientes:

-1º. Leemos el fichero CSV.
-2º. Obtenemos las columnas: "Tipo Evento", "Contenido" e "Imagen".
-3º. Eliminamos, de esas columnas, todas las filas cuyo "Contenido" o "Tipo Evento" sea vacío  o "Tipo Evento" no sea
     "Cursor" excepto para las que contengan en la columna "Contenido" un string o substring "{ctrl}k{ctrl}k". De esta
     forma nos quedamos solo con los "Tipo Evento" --> Cursor y con los "Tipo Evento" --> Keystrokes cuyo "Contenido"
     contenga un string o substring "{ctrl}k{ctrl}k", es decir, se haya detectado que el analista quiere que se aprenda
     a partir de ese momento hasta la siguiente vez que se encuentra "{ctrl}k{ctrl}k".
-4º. Se crea, a partir de las columnas filtradas anteriormente, un DataFrame para poder trabajar con él.
-5º. Se recogen la lista de acciones que se han realizado entre una un "{ctrl}k{ctrl}k" y otro "{ctrl}k{ctrl}k".
-6º. Se recoge, a partir del nombre de la imagen, el fingerprint de esa imagen en el archivo excel image-match.
-7º. Se transforman todas las acciones y el fingerprint de cada imagen (por acción), en un formato adecuado. En este
     caso, hay que tener en cuenta que el fingerprint tiene un tamaño variable según el tamaño de la imagen y por ello
     se debe NORMALIZAR para que todas tengan la misma. Igualmente, se tratará de que la RED asuma dicha "variabilidad"
     (en análisis). El formato para las acciones debe quedar:
     (tipo_de_click, coordenada x, coordenada y, fingerprint (normalizado o no))
     El tipo_de_click será 0 si es "{LEFT MOUSE}" y 1 si es "{RIGHT MOUSE}".
     Así, el array resultante podría quedar así: [0,450,354,[0,1,2,-1,-2,2,-1,0,0,0,1,...]] siendo el último elemento el
     fingerprint.
-8º. Creación de los datos de entrenamiento con los labels (etiquetas): a partir de las acciones normalizadas al formato
     adecuado, se debe guardar la primera firma obtenida. Esta firma servirá como entrada de todas las siguientes
     iteraciones (batch) (incluso para las que no tienen imágenes propias). Se deberá crear un array con el formato:
     ["action_1","action_2"], siendo el action_2 la siguiente acción del conjunto de acciones, siendo la etiqueta(label)
     del action_1.
     Esa action_2, para la siguiente iteración (batch), será la action_1 y esa acción tendrá su acción_2 (label) que se
     corresponde con la siguiente acción. De esta forma, formaremos nuestro conjunto de entrenamiento, que servirá de
     conjunto de testeo también.
     Para el caso en el que en la columna "imagen" exista una imagen diferente (no vacía), se deberá utilizar esa imagen
     como fingerprint para todas las siguientes y así sucesivamente hasta el final de las acciones.
     Un ejemplo de esto podría ser:
     [[[0,450,354,[fingerprint_1]], [1,451,355]], (...)] --> Conjunto de acciones divididas entre sus
     entradas y salidas(labels). Esto servirá para el entrenamiento y testeo de la red.
-9º. Una función que se llama "get_batch" que recibe por parámetro el array del anterior y un número entero y
     devuelva un número de elementos de ese array (batches) igual al número entero dado. Además, debe existir una variable
     que guarde el índice en el que está. Si tenemos un array de 1000 elementos y cogemos 50 ( la primera vez que se
     llama a la función), cada vez que se llame a esa función, debe dar los siguientes 50 elementos
     (tener en cuenta las ventanas temporales).



@(sara.moreno)
@(gabriel.vazquez)
"""
import pandas as pd
import tensorflow as tf
import numpy as np

# Keystrokes a tener en cuenta
start_event_learning_capture = "{ctrl}k{ctrl}k"
end_event_learning_capture = "{ctrl}k{ctrl}k"

"""
Funciones útiles
"""

def numpy_fillna(data):
    """
    Esta función convierte todos los elementos en un array del mismo tamaño cada uno con ceros si faltan elementos.
    :param data:
    :return:
    """

    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out

def pt(title=None, text=None):
    """
    Use the print function to print a title and an object coverted to string
    :param title:
    :param text:
    """
    if text is None:
        text = title
        title = "-----------------------------"
    else:
        title += ':'
    print(str(title) + " \n " + str(text))

def initialize_session():
    """
    Initialize interactive session and all local and global variables
    :return: Session
    """
    sess = tf.InteractiveSession()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    return sess

def split_list_in_pairs (a_list):
    """
    A partir de una lista tal que "a_list" = [1,2,3,4,5,6,7], devuelve otra lista que contenga listas de elementos 2 a 2
    excepto si es impar que devolverá el último elemento en una lista propia.
    E.g.: de "a_list" devuelve --> [[1,2],[3,4],[5,6],[7]]
    """
    if len(a_list) == 1 or not a_list:
        final_list = [a_list]
    else:
        final_list = [a_list[i:i + 2] for i in range(0, len(a_list), 2)]
    return final_list

# Paso 1º. Ruta del fichero
def read_file(path, separator):
    name_file = path
    file = pd.read_csv(name_file, sep=separator)
    return file

# Paso 2º. Obtenemos las columnas: "Tipo Evento", "Contenido" e "Imagen"
def get_columns(file):
    tipo_evento = file.pop("Tipo Evento")
    contenido = file.pop("Contenido")
    imagen = file.pop(" Imagen")
    return tipo_evento,contenido,imagen

# Paso 3. Eliminamos, de esas columnas, todas las filas cuyo "Contenido" o "Tipo Evento" sea vacío  o "Tipo Evento" no
# sea "Cursor" excepto para las que contengan en la columna "Contenido" un string o substring "{ctrl}k{ctrl}k". De esta
# forma nos quedamos solo con los "Tipo Evento" --> Cursor y con los "Tipo Evento" --> Keystrokes cuyo "Contenido"
# contenga un string o substring "{ctrl}k{ctrl}k", es decir, se haya detectado que el analista quiere que se aprenda
# a partir de ese momento hasta la siguiente vez que se encuentra "{ctrl}k{ctrl}k".
# Paso 4. Se crea, a partir de las columnas filtradas anteriormente, un DataFrame para poder trabajar con él.
def normalize_columns_and_create_dataframe(tipo_evento, contenido, imagen):
    # Paso 3
    # TODO (@IWT2) No eliminar elementos que contengan end_event_learning_capture
    for index in range(0, len(tipo_evento)):
        if (not tipo_evento[index] == "Cursor" or contenido[index] == "NaN" or contenido[index] == "") \
                and start_event_learning_capture not in contenido[index].lower():
            del tipo_evento[index]
            del contenido[index]
            del imagen[index]
    # Paso 4
    dataset = {}
    dataset[0] = tipo_evento
    dataset[1] = contenido
    dataset[2] = imagen
    final_dataframe = pd.DataFrame(data=dataset)
    final_dataframe.columns = ['Tipo Evento', 'Contenido', 'Imagen']
    print(final_dataframe)
    return final_dataframe

# Paso 5. Obtenemos las imágenes a partir de los keystrokes y array de lista de acciones (índices)
def get_images_from_keystrokes(tipo_evento, imagen):
    images_from_keystrokes = []  # Nombre de imágenes para utilizarlos como ids en el excel
    # Obtenemos los índices de las filas con "Tipo Evento" --> Keystrokes
    keystrokes_indexes = [x for x in range(len(list(tipo_evento.values))) if list(tipo_evento.values)[x].lower() ==
                          "keystrokes"]
    starts_ends_indexes_mod4 = split_list_in_pairs(
        a_list=keystrokes_indexes)  # Lista de parejas de todos los keystrokes_indexes
    print("keystrokes_indexes", keystrokes_indexes)
    print("index_in_pairs", starts_ends_indexes_mod4)
    for index in keystrokes_indexes:
        images_from_keystrokes.append(list(imagen.values)[index])
    print("images_from_keystrokes", images_from_keystrokes)
    return images_from_keystrokes, starts_ends_indexes_mod4

# Paso 6. Obtenemos las siguientes acciones a realizar a partir de los índices obtenidos previamente.
# Se debe acceder al excel image_match con los ids de las imagenes para recoger el fingerprint de cada una de ellas.
def get_next_actions(contenido, images_from_keystrokes, starts_ends_indexes):
    next_actions = []  # Lista de listas de siguientes acciones. Cada lista contenida tiene el siguiente formato:
    # [Posición 0 --> 0 si "Tipo Evento" es Cursor y 1 si es Keystrokes, Posición 1 --> (A partir de "Contenido") 0 si es
    # click izquierdo ,1 si es click derecho o la cadena string si es Keystrokes, Posición 2 --> Primera coordenada,
    # Posición 3 --> Segunda coordenada]
    if not starts_ends_indexes:
        next_actions = -1  # No hay siguientes acciones
    else:
        for index in range(len(starts_ends_indexes)):
            actual_learning = starts_ends_indexes[index]  # Contiene una lista con un número o un par de números
            if (len(actual_learning)) == 2:
                # TODO (@IWT2) Finish: Recoger la lista de elementos que hay entre los dos índices
                # TODO utilizar images_from_keystrokes para recoger las imagenes del excel image_match.
                for element_index in range(actual_learning[0],actual_learning[1]):
                    next_actions.append([contenido[element_index]])
            else: # Solo tiene un elemento
                pass  # No recogemos nada (en esta versión)
    print("next_actions",next_actions)
    return next_actions
# Paso 7
def normalize_actions(next_actions):
    # TODO
    return []
# Paso 8
def create_batches(normalized_actions):
    # TODO
    return []

# Paso 9
def get_batch(batches, quantity):
    global index_batch  # Para tener en cuenta el índice por donde va el recorrido del batch.
    # Si quantity es None, que recoja un número por defecto.
    # TODO
    return[]

index_batch = 0  # Índice para tener en cuenta el índice por donde va el recorrido del batch.
batches = []  # Que se utilizará para quedar guardada la primera vez que se cree.
file = "C:\\Users\Gabriel\Desktop\\02. Aquiles\dist Aquiles 20171113\\20180123\logfiles20180123.csv"  # Se deberá
# recoger como parámetro al igual que quantity y start_all_flag

def main_train_phase(file_path=None,quantity=None, start_all_flag=False):
    """

    :param file_path:
    :param quantity:
    :param start_all_flag:
    :return:
    """
    # TODO Docs
    global batches
    if start_all_flag:
        file = read_file(file_path,separator=";")
        tipo_evento, contenido, imagen = get_columns(file=file)
        normalize_columns_and_create_dataframe(tipo_evento=tipo_evento,contenido=contenido,imagen=imagen)
        images_from_keystrokes, starts_ends_indexes = get_images_from_keystrokes(tipo_evento=tipo_evento, imagen=imagen)
        next_actions = get_next_actions(contenido=contenido, images_from_keystrokes=images_from_keystrokes,
                                        starts_ends_indexes=starts_ends_indexes)
        normalized_actions = normalize_actions(next_actions=next_actions)
        batches = create_batches(normalized_actions)
    batch_to_train = get_batch(batches=batches,quantity=quantity)
    return batch_to_train

"""
RED NEURONAL
"""
# input con label
fingerprint_1 = np.asarray([-2,-1,1])
tipo_click = np.asarray([0])
coordenada_x = np.asarray([450])
coordenada_y = np.asarray([354])
input = np.asarray([
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1],
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1],
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1],
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1],
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1],
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1],
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1],
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1],
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1],
                  [tipo_click,coordenada_x,coordenada_y,fingerprint_1]
                  ])
label = np.asarray([1,451,355])
batches_1 = [
            [input, label]
            ]



# Parametros de la red
n_oculta_1 = 256 # 1ra capa de atributos
n_oculta_2 = 256 # 2ra capa de atributos
n_entradas = 4 # 4 datos de entrada
n_clases = 3 # 3 salidas

# input para los grafos
x = tf.placeholder("float", [None, n_entradas],  name='DatosEntrada')
y = tf.placeholder("float", [None, n_clases], name='Clases')

# Creamos el modelo
def perceptron_multicapa(x, pesos, sesgo):
    # Función de activación de la capa escondida
    capa_1 = tf.add(tf.matmul(x, pesos['h1']), sesgo['b1'])
    # activacion relu
    capa_1 = tf.nn.relu(capa_1)
    # Función de activación de la capa escondida
    capa_2 = tf.add(tf.matmul(capa_1, pesos['h2']), sesgo['b2'])
    # activación relu
    capa_2 = tf.nn.relu(capa_2)
    # Salida con activación lineal
    salida = tf.matmul(capa_2, pesos['out']) + sesgo['out']
    return salida


# Definimos los pesos y sesgo de cada capa.
pesos = {
    'h1': tf.Variable(tf.random_normal([n_entradas, n_oculta_1])),
    'h2': tf.Variable(tf.random_normal([n_oculta_1, n_oculta_2])),
    'out': tf.Variable(tf.random_normal([n_oculta_2, n_clases]))
}
sesgo = {
    'b1': tf.Variable(tf.random_normal([n_oculta_1])),
    'b2': tf.Variable(tf.random_normal([n_oculta_2])),
    'out': tf.Variable(tf.random_normal([n_clases]))
}


# Construimos el modelo
pred = perceptron_multicapa(x, pesos, sesgo)

# Definimos la funcion de costo
costo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Algoritmo de optimización
optimizar = tf.train.AdamOptimizer(learning_rate=0.001).minimize(costo)

# Evaluar el modelo
pred_correcta = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calcular la precisión
Precision = tf.reduce_mean(tf.cast(pred_correcta, "float"))

sess = initialize_session()

epochs = 100
trains = 10
# Entrenamiento
for epoca in range(epochs):
    avg_cost = 0.
    for train in range(trains):
        pt("batches_1[train][0]",batches_1[train][0])
        pt("batches_1[train][1]",batches_1[train][1])
        x_train, y_train = input[train], batches_1[train][1]
        # Optimización por backprop y funcion de costo
        _, c, summary = sess.run([optimizar, costo, Precision],
                                 feed_dict={x: x_train, y: y_train})
        y_ = y.eval()
        pt("y", y_)
    # imprimir información de entrenamiento
    if epoca % 1 == 0:
        pt("costo",costo)
        pt("Precision",Precision)
