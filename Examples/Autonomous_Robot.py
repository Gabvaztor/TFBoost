"""
Seguir estructura del código

-1º. Leemos el fichero CSV que se corresponde con el archivo salida de Aquiles.
-2º. Obtenemos las columnas: "Tipo Evento", "Contenido" e "Imagen".
-3º. Eliminamos, de esas columnas, todas las filas cuyo "Contenido" o "Tipo Evento" sea vacío  o "Tipo Evento" no sea
     "Cursor" excepto para las que contengan en la columna "Contenido" un string o substring "{ctrl}k{ctrl}k". De esta
     forma nos quedamos solo con los "Tipo Evento" --> Cursor y con los "Tipo Evento" --> Keystrokes cuyo "Contenido"
     contenga un string o substring "{ctrl}k{ctrl}k", es decir, se haya detectado que el analista quiere que se aprenda
     a partir de ese momento hasta la siguiente vez que se encuentra "{ctrl}k{ctrl}k".
     Esto se deberá modificar cuando tengamos una versión en la que necesitemos recoger keystrokes diferentes.
-4º. Se crea, a partir de las columnas filtradas anteriormente, un DataFrame para poder trabajar con él.
-5º. Se recogen la lista de acciones que se han realizado entre una un "{ctrl}k{ctrl}k" y otro "{ctrl}k{ctrl}k".
-6º. Se recoge, a partir del nombre de la imagen, el grupo de la imagen para la fase de entrenamiento (aprendizaje).
     Hay que tener en cuenta que cada acción puede tener asociado una imagen diferente (esa información está en el log
     de aquiles). En ese caso, para esa imagen hay que buscarle al grupo que pertenece (esto se hace asociando los ids
     (nombre de la imagen) de los archivos image_match (que es el log enriquecido) y del log de aquiles).
     Se debe retornar algo así por cada acción:
     [Posición 0 --> 0 si "Tipo Evento" es Cursor y 1 si es Keystrokes,
     Posición 1 --> (A partir de "Contenido") 0 si es click izquierdo ,1 si es click derecho o "la cadena string" si
     es Keystrokes,
     Posición 2 --> Primera coordenada,
     Posición 3 --> Segunda coordenada,
     Posición 4 --> Grupo de la imagen de la fila del contenido en la que está. Si esa fila no tiene imagen asociada,
     se utilizará el grupo de la imagen anterior]
-7º. Se transforman todas las acciones(por acción) en un formato adecuado. El formato para
     las acciones debe quedar:
     [tipo_de_click, coordenada x, coordenada y, grupo]
     El tipo_de_click será 0 si es "{LEFT MOUSE}" y 1 si es "{RIGHT MOUSE}".
     Así, el array resultante podría quedar así: [0,450,354,4] siendo el último elemento el grupo.
     Esta función es útil para manejar versiones del código. Así, para las versiones del código en las que necesitemos
     un tipo de información diferente, podemos modificar esta función y recoger lo que necesitemos.
-8º. Creación de los datos de entrenamiento con los labels (etiquetas):
     Crear una lista de listas (A) que contenga las entradas y salidas del conjunto de entrenamiento.
     La entrada (a) tiene la forma del paso anterior. La salida(b) es la siguiente acción del log de aquiles con esta
     forma: [1,451,355], siendo el primer elemento el tipo de click, el segundo la coordenada x y el tercer elemento la
     coordenada y.
     Para la siguiente posición del array, (b) será la entrada y la salida de esa entrada será la siguiente acción
     del log de aquiles y así sucesivamente hasta que se encuentre el siguiente fin de aprendizaje
     (end_event_learning_capture).
     Un ejemplo de esto podría ser:
     [[[0,450,354,grupo], [1,451,355]], [[1,451,355,grupo],[0,349,567]], (...)] --> Conjunto de acciones divididas entre
     sus entradas y salidas(labels). Esto servirá para el entrenamiento y testeo de la red.
-9º. Una función que se llama "get_batch" que recibe por parámetro el array del anterior y un número entero y
     devuelva un número de elementos de ese array (batches) igual al número entero dado. Además, debe existir una variable
     que guarde el índice en el que está. Si tenemos un array de 1000 elementos y cogemos 50 ( la primera vez que se
     llama a la función), cada vez que se llame a esa función, debe dar los siguientes 50 elementos
     (tener en cuenta las ventanas temporales).

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
"""
PASOS
"""
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
    # TODO (@IWT2) No eliminar elementos que contengan información relevante (para futuras versiones)
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
    # TODO images_from_keystrokes, por ahora, solo contiene los ids de las imágenes de las filas que tienen "keystrokes"
    # TODO en su tipo_evento. ¿Debe contener todas las imágenes de todas las acciones que hay en starts_ends_indexes?
    # TODO Si no es ahora, esa información la debemos tener en la siguiente fase.
    images_from_keystrokes = []  # Nombre de imágenes para utilizarlos como ids en el excel
    # Obtenemos los índices de las filas con "Tipo Evento" --> Keystrokes
    keystrokes_indexes = [x for x in range(len(list(tipo_evento.values))) if list(tipo_evento.values)[x].lower() ==
                          "keystrokes"]
    starts_ends_indexes = split_list_in_pairs(
        a_list=keystrokes_indexes)  # Lista de parejas de todos los keystrokes_indexes
    print("keystrokes_indexes", keystrokes_indexes)
    print("index_in_pairs", starts_ends_indexes)
    for index in keystrokes_indexes:
        images_from_keystrokes.append(list(imagen.values)[index])
    print("images_from_keystrokes", images_from_keystrokes)
    return images_from_keystrokes, starts_ends_indexes

# Paso 6. Obtenemos las siguientes acciones a realizar a partir de los índices obtenidos previamente.
# Se debe acceder al excel image_match con los ids de las imagenes para recoger el grupo de cada una de ellas.
def get_next_actions(contenido, images_from_keystrokes, starts_ends_indexes):
    next_actions = []  # Lista de listas de siguientes acciones. Cada lista contenida tiene el siguiente formato:
    # [Posición 0 --> 0 si "Tipo Evento" es Cursor y 1 si es Keystrokes,
    # Posición 1 --> (A partir de "Contenido") 0 si es click izquierdo ,1 si es click derecho o "la cadena string" si
    # es Keystrokes,
    # Posición 2 --> Primera coordenada,
    # Posición 3 --> Segunda coordenada,
    # Posición 4 --> Grupo de la imagen de la fila del contenido en la que está. Si esa fila no tiene imagen asociada,
    # se utilizará el grupo de la imagen anterior]
    # TODO Tener en cuenta el doc del archivo para realizar esta función
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
file_aquiles = "C:\\Users\Gabriel\Desktop\\02. Aquiles\dist Aquiles 20171113\\20180123\logfiles20180123.csv"  # Se deberá
file_mod2_3 = "log2_3"  # Se deberá
# recoger como parámetro al igual que quantity y start_all_flag

def main_train_phase(file_path_aquiles=None, file_path_mod2_3=None, quantity=None, start_all_flag=False):
    """

    :param file_path:
    :param quantity:
    :param start_all_flag:
    :return:
    """
    # TODO Docs
    global batches
    if start_all_flag:
        # TODO Tras el cambio en la forma de realizar la fase de entrenamiento (ahora es por el grupo), es posible que
        # TODO los nombres de las variables y las entradas de las funciones no sean acordes al doc del archivo.
        # TODO Cambiarlo si es necesario e intentar hacer el código para que sea granular.
        file = read_file(file_path_aquiles,separator=";")
        tipo_evento, contenido, imagen = get_columns(file=file)
        normalize_columns_and_create_dataframe(tipo_evento=tipo_evento,contenido=contenido,imagen=imagen)
        images_from_keystrokes, starts_ends_indexes = get_images_from_keystrokes(tipo_evento=tipo_evento, imagen=imagen)
        next_actions = get_next_actions(contenido=contenido, images_from_keystrokes=images_from_keystrokes,
                                        starts_ends_indexes=starts_ends_indexes)
        if not next_actions == -1:
            normalized_actions = normalize_actions(next_actions=next_actions)
            batches = create_batches(normalized_actions)
        else:
            pt("No hay acciones")
    batch_to_train = get_batch(batches=batches,quantity=quantity)
    return batch_to_train

# Get batch algorithm
#batch = main_train_phase(file_aquiles, file_mod2_3,2, True)

"""
RED NEURONAL
"""
# input con label

tipo_click = 0.
coordenada_x = 450.
coordenada_y = 354.
grupo = 4.
input = np.asarray([tipo_click,coordenada_x,coordenada_y,grupo]).reshape(1,4)
label = np.asarray([1.,451.,355.]).reshape(1,3)
batches_ = []
for i in range(10):
    batches_.append([input, label])

# Parametros de la red
n_oculta_1 = 4 # 1ra capa de atributos
n_oculta_2 = 4 # 2ra capa de atributos
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

# Definimos la funcion de coste
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
costes = []
precisiones = []
# Entrenamiento
for epoca in range(1000):
    avg_cost = 0.
    for train in range(trains):
        #pt("batches_[train][0].shape",batches_[train][0].shape)
        x_train, y_train = batches_[train][0], batches_[train][1]
        # Optimización por backprop y funcion de costo
        _, c, summary, y_ = sess.run([optimizar, costo, Precision,pred],
                                 feed_dict={x: x_train, y: y_train})
        if train == 0:
            pt("coste",c)
            pt("Precision",summary)
            costes.append(c)
            precisiones.append(summary)
            pt("y", y_)
    # imprimir información de entrenamiento
    if epoca % 1 == 0:
        pt("costes",costes)
        pt("precisiones",precisiones)

pt("FINAL",pred.eval(feed_dict={x:batches_[0][0]}))

#np.savetxt("W.csv", W_val, delimiter=",")
#np.savetxt("b.csv", b_val, delimiter=",")