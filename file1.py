#
import numpy as np
import tensorflow as tf
import keras_tuner
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import GridSearch, HyperModel, Objective, HyperParameters
import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

num_train_samples = 8000
num_test_samples = 2000

x_train = x_train[:num_train_samples]
y_train = y_train[:num_train_samples]

x_test = x_test[:num_test_samples]
y_test = y_test[:num_test_samples]

# Normalize the images
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure the images have the right shape
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode the labels
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

