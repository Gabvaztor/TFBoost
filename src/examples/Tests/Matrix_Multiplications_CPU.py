import tensorflow as tf
import time
import os

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def run(size):
    with tf.device('/cpu:0'):
        a = tf.random_uniform([size, size])
        b = tf.random_uniform([size, size])
        print('Size: ', a.shape)
        for i in range(5):
            start = time.time()
            tf.matmul(a, b)
            print('Runtime is %2.5f' % (time.time() - start))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.list_devices()
start = time.time()
for p in [500, 1000, 2000, 5000, 10000]:
    run(p)
print('Total Time is %2.5f' % (time.time() - start))