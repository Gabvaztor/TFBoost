import tensorflow as tf
import time

def run(size):
  with tf.device('/gpu:0'):
      a = tf.random_uniform([size, size])
      b = tf.random_uniform([size, size])
      print('Size: ', a.shape)
      for i in range(5):
          start = time.time()
          tf.matmul(a, b)
          print('Runtime is %2.5f' % (time.time() - start))

print('One warmup run to account for GPU initialization')
run(10)
start = time.time()
for p in [500, 1000, 2000, 5000, 10000]:
    run(p)
print('Total Time is %2.5f' % (time.time() - start))