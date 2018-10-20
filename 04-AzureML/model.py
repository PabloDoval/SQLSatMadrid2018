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
    model.save(path)

def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model