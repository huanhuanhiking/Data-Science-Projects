# Setup GPU

# Using plaidml.keras.backend backend
import os, shutil
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

# Using tensorflow backend
import tensorflow as tf
tf.config.list_physical_devices('GPU')

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
