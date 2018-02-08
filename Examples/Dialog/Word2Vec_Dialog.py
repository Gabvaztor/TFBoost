# coding=utf-8
"""
This is Skip Gram form.
"""

import numpy as np
import tensorflow as tf
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from UsefulTools.UtilsFunctions import *
from UsefulTools.TensorFlowUtils import *
from Examples.Dialog import Dialog

# Para eliminar tildes
import unicodedata


def delete_accents_marks(string):
    return ''.join((c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn'))

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

def process_for_senteces(sentences):
    """
    Preprocesa las frases para quitarle los singos de acentuación, los puntos, comas, (...)
    """
    processed_sentences = []
    for sentence in sentences:
        sentence_split = sentence.split()
        new_sentence_processed = []
        for word in sentence_split:
            new_sentence_processed.append(delete_accents_marks(word).lower())
        processed_sentences.append(new_sentence_processed)
    pt("processed_sentences", processed_sentences)
    return processed_sentences


def create_dictionaries(words):
    """
    Crea los diccionarios int2word y word2int a partir de las frases procesadas y las retorna en ese orden
    """
    int2word = {}
    word2int = {}
    for i, word in enumerate(words):
        word2int[word] = i
        int2word[i] = word
    pt("word2int", word2int)
    pt("int2word", int2word)
    return word2int, int2word


def get_words_set(processes_sentences):
    """
    A partir de frases preprocesadas, obtiene el conjunto de palabras (sin repetición) de las que se compone
    """
    words = []
    to_delete_marks = [",", ".", ":", ";"]
    corpus = [item for sublist in processes_sentences for item in sublist]
    for word in corpus:
        if word not in to_delete_marks:
            words.append(word)
    words = set(words)  # Removemos palabras repetidas
    pt("words",words)
    return words


def generate_training_data(processes_sentences, question_id):
    """
    Genera el conjunto de datos que se utilizará para entrenar a la red una vez estén sean one-hot-vector y a partir de
    la "question_id"
    """
    data = []
    if question_id == 99:
        pass
    else:
        WINDOW_SIZE = 5
    for sentence in processes_sentences:
        for word_index, word in enumerate(sentence):
            for nb_word in sentence[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(sentence)) + 1]:
                if nb_word != word:
                    data.append([word, nb_word])
    pt("data", data)
    pt("data", len(data))
    return data


def generate_batches(data, word2int, vocab_size):
    """
    Genera las entradas y los labels a partir de los datos generados previamente.
    """
    x_input = []
    y_label = []
    for data_word in data:
        pt("dataword", data_word)
        x_input.append(to_one_hot(word2int[data_word[0]], vocab_size))
        y_label.append(to_one_hot(word2int[data_word[1]], vocab_size))
    return np.asarray(x_input), np.asarray(y_label)


def generate_network_and_vector(x_input, y_label, vocab_size):
    """
    Crea la red neuronal con TensorFlow y utiliza las entradas y los labels para entrenarla. La red consta de 3 capas:
    - Una de entrada
    - Una intermedia
    - Una de salida
    Al hacer Skip Gram, siendo los inputs one-hot-vectors, nos quedamos con los pesos y biases de las dos primeras
    capas. Así, se hace Word Embedding y obtenemos los vectores asociados a las palabras.
    """
    # making placeholders for x_train and y_train
    x = tf.placeholder(tf.float32, shape=(None, vocab_size))
    y = tf.placeholder(tf.float32, shape=(None, vocab_size))

    EMBEDDING_DIM = 200  # you can choose your own number # Límite 282 portatil msi 820
    W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
    b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))  # bias
    hidden_representation = tf.add(tf.matmul(x, W1), b1)

    W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
    b2 = tf.Variable(tf.random_normal([vocab_size]))
    prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))

    sess = initialize_session()
    # define the loss function:
    cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

    # define the training step:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(cross_entropy_loss)
    n_iters = 10000
    # train for n_iter iterations
    for _ in range(n_iters):
        pt(cross_entropy_loss.eval(feed_dict={x: x_input, y: y_label}))
        sess.run(train_op, feed_dict={x: x_input, y: y_label})
        print('Cross Entropy es', sess.run(cross_entropy_loss, feed_dict={x: x_input, y: y_label}))

    pt("W1")
    vectors = sess.run(tf.add(W1, b1))
    pt("vectors", vectors)
    pt("vectors_shape", vectors.shape)
    return vectors

class Word2Vec():
    words_vectors = []
    word2int = {}
    int2word = {}
    vocab_size = 0

def main(sentences, question_id):
    word2vec_class = Word2Vec()
    processes_sentences = process_for_senteces(sentences)
    words = get_words_set(processes_sentences)
    word2vec_class.word2int, word2vec_class.int2word = create_dictionaries(words)
    word2vec_class.vocab_size = len(words)
    data = generate_training_data(processes_sentences, question_id)
    x_input, y_label = generate_batches(data, word2vec_class.word2int, word2vec_class.vocab_size)
    vectors = generate_network_and_vector(x_input, y_label, word2vec_class.vocab_size)
    return {question_id:vectors}
if __name__ == '__main__':
    dict = main(Dialog.Estres.palabras_destacadas_pregunta_1, Dialog.Estres.id_pregunta_1)
    pt("dict", dict)
    asd
corpus_raw = 'He is the king . The king is royal . She is the royal queen .'
#corpus_raw = 'Yo no tengo nada de estrés'
# convert to lower case
corpus_raw = corpus_raw.lower()

words = []
for word in corpus_raw.split():
    if word != '.': # because we don't want to treat . as a word
        words.append(word)
words = set(words) # so that all duplicate words are removed
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

pt("word2int", word2int)
pt("int2word", int2word)

raw_sentences = corpus_raw.split('.')
pt("raw_sentences",raw_sentences)

sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())
pt("sentences",sentences)
ASD
#training data
data = []
WINDOW_SIZE = 2
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])
pt("data",data)
pt("data",len(data))

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] # input word
y_train = [] # output word
for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))
# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

pt("x_train",x_train)
pt("y_train",y_train)
pt("x_train.shape",x_train.shape)
pt("y_train.shape",y_train.shape)
pt("vocab_size",str(vocab_size))

# making placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 200 # you can choose your own number # Límite 282 portatil msi 820
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

sess = initialize_session()
# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

# define the training step:
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cross_entropy_loss)
n_iters = 10000
# train for n_iter iterations
for _ in range(n_iters):
    pt(cross_entropy_loss.eval(feed_dict={x: x_train, y_label: y_train}))
    sess.run(train_op, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

pt("W1")
vectors = sess.run(tf.add(W1, b1))
pt("vectors",vectors)
pt("vectors_shape",vectors.shape)

#pt("Vector queen in word2int", vectors[ word2int['queen'] ])
pt("Vector estrés in word2int", vectors[ word2int['estrés'] ])

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

#pt("Más cercano a king", int2word[find_closest(word2int['king'], vectors)])
#pt("Más cercano a queen", int2word[find_closest(word2int['queen'], vectors)])
#pt("Más cercano a royal", int2word[find_closest(word2int['royal'], vectors)])
pt("Más cercano a nada", int2word[find_closest(word2int['nada'], vectors)])
pt("Más cercano a yo", int2word[find_closest(word2int['yo'], vectors)])
pt("Más cercano a estrés", int2word[find_closest(word2int['estrés'], vectors)])

from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors)

from sklearn import preprocessing
normalizer = preprocessing.Normalizer()
vectors =  normalizer.fit_transform(vectors, 'l2')

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for word in words:
    print(word, vectors[word2int[word]][1])
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
plt.show()