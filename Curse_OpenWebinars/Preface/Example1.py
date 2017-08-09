import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from UsefulTools.TensorFlowUtils import *
rng = numpy.random

# Parameters
learning_rate = 0.001
training_epochs = 600
display_step = 50

# Training Data
x = 5.0
y = 10.0

# tf Graph Input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Set model weights
W = tf.Variable(0.0, name="weight")
# Caso 1
b = tf.constant(0.0, name="bias")
# Caso 2
#b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)
# Mean squared error
error = tf.abs(tf.subtract(Y, pred))
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

# Launch the graph
sess = initialize_session()

W_s = []
b_s = []
# Fit all training data
for epoch in range(training_epochs):
    sess.run(optimizer, feed_dict={X: x, Y: y})
    w = W.eval()
    b_ = b.eval()
    W_s.append(w)
    b_s.append(b_)

    # Display logs per epoch step
    if (epoch + 1) % display_step == 0:
        c = sess.run(error, feed_dict={X: x, Y: y})
        print("Epoch:", '%04d' % (epoch + 1), "error=", "{:.9f}".format(c), \
              "W=", sess.run(W), "b=", sess.run(b))
print("------------------------")
print("Finalizado entrenamiento")
training_cost = sess.run(error, feed_dict={X: x, Y: y})
print("Error de entrenamiento=", training_cost, " \nW =", sess.run(W), "\nb =", sess.run(b))

# Graphic display
#plt.plot(train_X, train_Y, 'ro', label='Original data')
#plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')

y_prediccion = pred.eval(feed_dict={X: x, Y: y})
print("y_predicción =" , str(y_prediccion))
plt.plot(W_s, label='W')
plt.plot(b_s, label='b')
plt.legend()
plt.show()
