import tensorflow as tf
from tensorflow.python.framework import ops


DETECTOR_ONES_SIZE = (1, 480, 640, 3)

object_detector1 = tf.saved_model.load('/model1')
ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)

object_detector2 = tf.saved_model.load('/model2')
ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)

print('start inference model1')
print(object_detector1(ones))

print('start inference model2')
print(object_detector2(ones))
