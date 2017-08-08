import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.003
training_epochs = 2500
display_step = 50

# Training Data
train_X = 2.0
train_Y = 10.0

# tf Graph Input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(2.0, name="bias")
#b = tf.constant(2.0, name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
#cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
cost = tf.abs(tf.subtract(Y, pred))
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

W_s = []
# Launch the graph
sess = tf.InteractiveSession()
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

# Fit all training data
for epoch in range(training_epochs):
    sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
    w = W.eval()
    W_s.append(w)
    # Display logs per epoch step
    if (epoch + 1) % display_step == 0:
        c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
              "W=", sess.run(W), "b=", sess.run(b))

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

# Graphic display
#plt.plot(train_X, train_Y, 'ro', label='Original data')
#plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
plt.plot(W_s)
#plt.legend()
plt.show()

# Testing example, as requested (Issue #2)
test_X = 10.0
test_Y = 42.0

print("Testing... (Mean square loss Comparison)")
testing_cost = sess.run(
    cost,
    feed_dict={X: test_X, Y: test_Y})  # same function as cost above
print("Testing cost=", testing_cost)
print("Absolute mean square loss difference:", abs(
    training_cost - testing_cost))
"""
plt.plot(test_X, test_Y, 'bo', label='Testing data')
plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
plt.legend()
plt.show()
"""
