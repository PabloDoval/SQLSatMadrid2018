import tensorflow as tf
from tensorflow import keras
import os

def get_model_definition():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

def save_model(model, folderpath, filename):
    if (not os.path.exists(folderpath)):
        os.makedirs(folderpath)
    path = os.path.join(folderpath, filename)
    model.save_weights(path)

def load_model(folderpath, filename):
    model = get_model_definition()
    path = os.path.join(folderpath, filename)
    model.load_weights(path, by_name=False)
    return model