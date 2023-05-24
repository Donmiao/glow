import tensorflow as tf
from tensorflow.python.client import device_lib

print(tf.version)
print(tf.test.is_gpu_available())

print(device_lib.list_local_devices())