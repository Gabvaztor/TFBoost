import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from UsefulTools.TensorFlowUtils import *
rng = numpy.random

# Parameters
learning_rate = 0.001
training_epochs = 600
display_step = 50


# Placeholders del grafo
X = tf.placeholder(tf.float32)
x = 5.0
Y = tf.placeholder(tf.float32)
y = 10.0

# El peso es variable. Empieza en 0
W = tf.Variable(0.0, name="weight")
# Caso 1
b = tf.constant(0.0, name="bias")
# Caso 2
#b = tf.Variable(0.0, name="bias")

# y_predecida = W*x+b
y_predecida = tf.add(tf.multiply(W, X), b)
# El error es la diferencia entre la 'y' real y la 'y_predecida'
error = tf.abs(tf.subtract(Y, y_predecida))
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

# Inicializamos grafo
sess = initialize_session()

W_s = []
b_s = []

# Entrenamiento
for epoch in range(training_epochs):
    sess.run(optimizer, feed_dict={X: x, Y: y})
    w = W.eval()
    b_ = b.eval()
    W_s.append(w)
    b_s.append(b_)

    # Ver resultados
    if (epoch + 1) % display_step == 0:
        c = sess.run(error, feed_dict={X: x, Y: y})
        print("Epoch:", '%04d' % (epoch + 1), "error=", "{:.9f}".format(c), \
              "W=", sess.run(W), "b=", sess.run(b))
print("------------------------")
print("Finalizado entrenamiento")
training_cost = sess.run(error, feed_dict={X: x, Y: y})
print("Error de entrenamiento=", training_cost, " \nW =", sess.run(W), "\nb =", sess.run(b))

y_prediccion = y_predecida.eval(feed_dict={X: x, Y: y})
print("y_predicción =" , str(y_prediccion))
plt.plot(W_s, label='W')
plt.plot(b_s, label='b')
plt.legend()
plt.show()
